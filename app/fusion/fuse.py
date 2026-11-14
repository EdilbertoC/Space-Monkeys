# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Multi-sensor fusion of hotspot detections into fire events.

A hotspot is the centre of a satellite pixel, so its true position lies somewhere
inside that pixel. Different sensors see the same fire with very different pixels
(VIIRS ~375 m, MODIS ~1 km, GOES ABI ~2-5 km). Fusion:

1. Clusters detections that are close in space and time (``window_hours``), across
   all sensors. Two detections link when their distance is at most ``radius_km``
   plus half of each pixel's larger side, since a pixel centre can sit that far from
   the fire. Clusters are connected components, so a long fire front chains into
   one event.
2. Estimates each event's position as an inverse-variance weighted centroid. A point
   uniformly distributed over a pixel of size scan x track has variance
   (scan^2 + track^2) / 12, so a VIIRS pixel weighs ~30x more than a GOES pixel.
   The weight is also scaled by the detection's confidence.
3. Reports an uncertainty radius (~95%): twice the larger of the combined pixel
   error and the weighted spread of the detections (a big fire covers many pixels,
   so the spread dominates there).
4. Combines confidence across sensor families: 1 - prod(1 - best confidence per
   sensor). Agreement between independent sensors raises confidence.
"""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

EARTH_RADIUS_KM = 6371.0088
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON_EQUATOR = 111.320

# Fallback pixel sizes (km) when a provider gives none.
DEFAULT_PIXEL_KM = {"VIIRS": 0.375, "MODIS": 1.0, "ABI": 2.0, "OLI": 0.03}

FIRE_COLUMNS = [
    "fire_id",
    "lat",
    "lon",
    "uncertainty_km",
    "confidence",
    "sensors",
    "n_detections",
    "first_seen",
    "last_seen",
    "frp_max_mw",
]


def _unit_vectors(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))


def cluster_labels(df: pd.DataFrame, radius_km: float, window_hours: float) -> np.ndarray:
    """Connected-component label for each detection."""
    n = len(df)
    if n == 0:
        return np.empty(0, dtype=int)
    points = _unit_vectors(df["lat"].to_numpy(), df["lon"].to_numpy())
    half = 0.5 * _pixel_size_km(df).max(axis=1)
    search_km = radius_km + 2.0 * float(half.max())
    pairs = cKDTree(points).query_pairs(_chord(search_km), output_type="ndarray")
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        # Pixel centres of the same fire can sit up to half a pixel away from it.
        limit_km = radius_km + half[i] + half[j]
        distance_km = _arc_km(np.linalg.norm(points[i] - points[j], axis=1))
        t = (df["time"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy()
        keep = (distance_km <= limit_km) & (np.abs(t[i] - t[j]) <= window_hours * 3600)
        pairs = pairs[keep]
    graph = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])) if len(pairs) else ([], ([], [])),
        shape=(n, n),
    )
    _, labels = connected_components(graph, directed=False)
    return labels


def _chord(arc_km: float) -> float:
    return 2.0 * np.sin(arc_km / (2.0 * EARTH_RADIUS_KM))


def _arc_km(chord: np.ndarray) -> np.ndarray:
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))


def _pixel_size_km(df: pd.DataFrame) -> np.ndarray:
    """(n, 2) array of scan and track sizes, with per-sensor fallbacks."""
    fallback = pd.Series(df["sensor"].map(DEFAULT_PIXEL_KM).fillna(1.0), index=df.index)
    scan = df["scan_km"].fillna(fallback).to_numpy(dtype=float)
    track = df["track_km"].fillna(fallback).to_numpy(dtype=float)
    return np.column_stack((scan, track))


def _pixel_variance_km2(group: pd.DataFrame) -> np.ndarray:
    size = _pixel_size_km(group)
    return (size[:, 0] ** 2 + size[:, 1] ** 2) / 12.0


def _fuse_group(group: pd.DataFrame) -> dict:
    variance = _pixel_variance_km2(group)
    conf = group["confidence"].clip(0.05, 1.0).to_numpy()
    weight = conf / variance

    lat0 = float(group["lat"].mean())
    lon0 = float(group["lon"].mean())
    km_per_deg_lon = KM_PER_DEG_LON_EQUATOR * np.cos(np.deg2rad(lat0))
    east = (group["lon"].to_numpy() - lon0) * km_per_deg_lon
    north = (group["lat"].to_numpy() - lat0) * KM_PER_DEG_LAT

    w_sum = weight.sum()
    e_hat = float((weight * east).sum() / w_sum)
    n_hat = float((weight * north).sum() / w_sum)

    pixel_sigma = float(np.sqrt(1.0 / (1.0 / variance).sum()))
    spread = float(np.sqrt((weight * ((east - e_hat) ** 2 + (north - n_hat) ** 2)).sum() / w_sum))
    uncertainty = 2.0 * max(pixel_sigma, spread)

    best_per_sensor = group.groupby("sensor")["confidence"].max().clip(0, 0.99)
    confidence = float(1.0 - np.prod(1.0 - best_per_sensor.to_numpy()))

    earliest = group.sort_values(["time", "detection_id"]).iloc[0]
    frp = group["frp_mw"].max()
    return {
        "fire_id": earliest["detection_id"],
        "lat": lat0 + n_hat / KM_PER_DEG_LAT,
        "lon": lon0 + e_hat / km_per_deg_lon,
        "uncertainty_km": round(uncertainty, 3),
        "confidence": round(confidence, 3),
        "sensors": sorted(group["sensor"].unique().tolist()),
        "n_detections": int(len(group)),
        "first_seen": group["time"].min(),
        "last_seen": group["time"].max(),
        "frp_max_mw": None if pd.isna(frp) else float(frp),
    }


def fuse(df: pd.DataFrame, radius_km: float = 2.0, window_hours: float = 12.0) -> pd.DataFrame:
    """Fuse detections (common detections table) into one row per fire event.

    ``fire_id`` is the id of the event's earliest detection, so it stays stable
    across polls while that detection is retained.
    """
    if df.empty:
        return pd.DataFrame(columns=FIRE_COLUMNS)
    df = df.reset_index(drop=True)
    labels = cluster_labels(df, radius_km, window_hours)
    fires = [_fuse_group(group) for _, group in df.groupby(labels, sort=False)]
    return pd.DataFrame(fires, columns=FIRE_COLUMNS)
