# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Recent daily fire maps for the forecast, from NASA FIRMS VIIRS S-NPP NRT.

The model was trained on daily counts of VIIRS S-NPP fire pixels (nominal/high
confidence) per cell, by acquisition date (UTC). The NRT feed has no `type` column, so
static non-vegetation sources cannot be removed here; the forecast mask already limits
the model to cells that burned during training.
"""

from datetime import date, timedelta

import httpx
import numpy as np
import pandas as pd

from app.forecast.model import Grid
from app.realtime import firms

SOURCE = "VIIRS_SNPP_NRT"
MIN_CONFIDENCE = 0.6  # "nominal" and "high" in the FIRMS reader's 0..1 scale
MAX_DAYS_PER_REQUEST = 5


def grid_counts(detections: pd.DataFrame, grid: Grid, first_day: date, n_days: int) -> np.ndarray:
    """(n_days, ny, nx) uint16 counts of detections per cell and UTC acquisition day."""
    ny, nx = grid.shape
    counts = np.zeros((n_days, ny, nx), dtype=np.uint16)
    if detections.empty:
        return counts
    df = detections[detections["confidence"] >= MIN_CONFIDENCE]
    day = (df["time"].dt.tz_convert("UTC").dt.date - first_day).map(lambda d: d.days)
    day = day.to_numpy(dtype=np.int64)
    row, col, inside = grid.cell_of(df["lat"].to_numpy(), df["lon"].to_numpy())
    ok = inside & (day >= 0) & (day < n_days)
    np.add.at(counts, (day[ok], row[ok], col[ok]), 1)
    return counts


async def fetch_daily_counts(
    client: httpx.AsyncClient, map_key: str, grid: Grid, last_day: date, n_days: int
) -> tuple[np.ndarray, list[int]]:
    """Daily counts for last_day - n_days + 1 … last_day, and detections per day."""
    first_day = last_day - timedelta(days=n_days - 1)
    frames = []
    offset = 0
    while offset < n_days:
        span = min(MAX_DAYS_PER_REQUEST, n_days - offset)
        start = first_day + timedelta(days=offset)
        frames.append(
            await firms.fetch_source(client, map_key, SOURCE, grid.bbox, span, start=start)
        )
        offset += span
    frames = [f for f in frames if not f.empty]
    detections = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    counts = (
        grid_counts(detections, grid, first_day, n_days)
        if frames
        else np.zeros((n_days, *grid.shape), dtype=np.uint16)
    )
    per_day = counts.reshape(n_days, -1).sum(1).astype(int).tolist()
    return counts, per_day
