# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""NASA FIRMS near-real-time reader (MODIS and VIIRS active fire hotspots).

API: https://firms.modaps.eosdis.nasa.gov/api/area/
URL: /api/area/csv/{MAP_KEY}/{SOURCE}/{west,south,east,north}/{DAY_RANGE}
"""

import io
import logging
from datetime import date

import httpx
import pandas as pd

from app.realtime import detections

log = logging.getLogger(__name__)

FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# VIIRS reports a class; MODIS reports 0-100.
VIIRS_CONFIDENCE = {"l": 0.3, "low": 0.3, "n": 0.6, "nominal": 0.6, "h": 0.9, "high": 0.9}


class FirmsError(RuntimeError):
    pass


def sensor_for(source: str) -> str:
    if source.startswith("VIIRS"):
        return "VIIRS"
    if source.startswith("MODIS"):
        return "MODIS"
    if source.startswith("LANDSAT"):
        return "OLI"
    raise ValueError(f"Unknown FIRMS source: {source}")


def parse_csv(text: str, source: str) -> pd.DataFrame:
    """Turn a FIRMS area-API CSV into the common detections table."""
    if not text.strip():
        return detections.empty()
    head = text.lstrip().splitlines()[0].lower()
    if "latitude" not in head:
        # FIRMS answers errors (bad key, quota) as plain text with HTTP 200.
        raise FirmsError(f"FIRMS {source}: {text.strip()[:200]}")

    raw = pd.read_csv(io.StringIO(text), dtype={"acq_time": str, "confidence": str})
    if raw.empty:
        return detections.empty()

    sensor = sensor_for(source)
    hhmm = raw["acq_time"].str.zfill(4)
    time = pd.to_datetime(raw["acq_date"] + " " + hhmm, format="%Y-%m-%d %H%M", utc=True)

    if sensor == "MODIS":
        confidence = pd.to_numeric(raw["confidence"], errors="coerce") / 100.0
    else:
        confidence = raw["confidence"].str.strip().str.lower().map(VIIRS_CONFIDENCE)

    df = pd.DataFrame(
        {
            "source": f"FIRMS:{source}",
            "sensor": sensor,
            "satellite": raw.get("satellite", pd.Series([""] * len(raw))).astype(str),
            "time": time,
            "lat": raw["latitude"].astype(float),
            "lon": raw["longitude"].astype(float),
            "scan_km": pd.to_numeric(raw["scan"], errors="coerce"),
            "track_km": pd.to_numeric(raw["track"], errors="coerce"),
            "frp_mw": pd.to_numeric(raw.get("frp"), errors="coerce"),
            "confidence": confidence.fillna(0.5).clip(0, 1),
            "daynight": raw.get("daynight", pd.Series([""] * len(raw))).fillna("").astype(str),
        }
    )
    return detections.finalize(df)


async def fetch_source(
    client: httpx.AsyncClient,
    map_key: str,
    source: str,
    bbox: tuple[float, float, float, float],
    day_range: int = 1,
    start: date | None = None,
) -> pd.DataFrame:
    """Detections for `day_range` days: the most recent ones, or from `start` onward."""
    area = ",".join(f"{v:g}" for v in bbox)
    url = f"{FIRMS_AREA_URL}/{map_key}/{source}/{area}/{day_range}"
    if start is not None:
        url += f"/{start.isoformat()}"
    response = await client.get(url, timeout=60)
    response.raise_for_status()
    return parse_csv(response.text, source)


async def fetch_all(
    client: httpx.AsyncClient,
    map_key: str,
    sources: list[str],
    bbox: tuple[float, float, float, float],
    day_range: int = 1,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch every source; one failing source does not stop the others.

    Returns the merged detections and a per-source error map.
    """
    if not map_key:
        raise FirmsError("FIRMS_MAP_KEY is not set")
    frames: list[pd.DataFrame] = []
    errors: dict[str, str] = {}
    for source in sources:
        try:
            frames.append(await fetch_source(client, map_key, source, bbox, day_range))
        except (httpx.HTTPError, FirmsError, ValueError, KeyError) as exc:
            log.warning("FIRMS %s failed: %s", source, exc)
            errors[source] = str(exc)
    frames = [f for f in frames if not f.empty]
    merged = pd.concat(frames, ignore_index=True) if frames else detections.empty()
    return merged, errors
