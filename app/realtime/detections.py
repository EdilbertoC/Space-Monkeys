# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Common table format for hotspot detections from every sensor.

Every reader (FIRMS, GOES) returns a DataFrame with exactly these columns, so the
fusion and the live store never need to know where a detection came from.
"""

import hashlib

import numpy as np
import pandas as pd

COLUMNS = [
    "detection_id",  # stable hash of source + time + position
    "source",  # e.g. FIRMS:VIIRS_SNPP_NRT, GOES:G19
    "sensor",  # VIIRS | MODIS | ABI
    "satellite",  # as reported by the provider
    "time",  # acquisition time, tz-aware UTC
    "lat",
    "lon",
    "scan_km",  # pixel size along scan (east-west)
    "track_km",  # pixel size along track (north-south)
    "frp_mw",  # fire radiative power, MW (NaN when unknown)
    "confidence",  # normalized to 0..1
    "daynight",  # D | N | "" when unknown
]


def empty() -> pd.DataFrame:
    df = pd.DataFrame({col: pd.Series(dtype="object") for col in COLUMNS})
    df["time"] = pd.Series(dtype="datetime64[ns, UTC]")
    for col in ("lat", "lon", "scan_km", "track_km", "frp_mw", "confidence"):
        df[col] = pd.Series(dtype="float64")
    return df


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Add detection ids, enforce column order and drop exact duplicates."""
    if df.empty:
        return empty()
    df = df.copy()
    df["detection_id"] = make_ids(df)
    df = df.drop_duplicates("detection_id")
    return df[COLUMNS].reset_index(drop=True)


def make_ids(df: pd.DataFrame) -> pd.Series:
    keys = (
        df["source"].astype(str)
        + "|"
        + df["time"].dt.strftime("%Y%m%dT%H%M%S")
        + "|"
        + np.round(df["lat"], 4).astype(str)
        + "|"
        + np.round(df["lon"], 4).astype(str)
    )
    return keys.map(lambda k: hashlib.sha1(k.encode()).hexdigest()[:16])


def in_bbox(df: pd.DataFrame, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    w, s, e, n = bbox
    mask = df["lon"].between(w, e) & df["lat"].between(s, n)
    return df[mask]
