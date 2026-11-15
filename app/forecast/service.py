# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Daily fire forecast, anchored in real time by live GOES/FIRMS fire events.

1. Base forecast: once per refresh, the U-Net turns the last 30 days of VIIRS fire maps
   into probabilities for the next 7 days. The forecast is issued for the last complete
   UTC day, so lead 1 is today.
2. Real-time anchor: after every pulse, cells with a live fused fire are raised to at
   least  confidence x P(fire persists j days)  for the day the fire was seen and the
   days after it. The persistence probabilities were measured on the model's training
   data (`meta.json` → `persistence`). The anchor never lowers a forecast and is never
   fed back into the network, which never saw GOES data.
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

import httpx
import numpy as np
import pandas as pd

from app.forecast.history import DEFAULT_SOURCE, fetch_daily_counts
from app.forecast.model import ForecastModel, day_of_year0

log = logging.getLogger(__name__)

# Highest probability the anchor may set: a fire seen today does not make another VIIRS
# detection certain (clouds, overpass timing), and the fused confidence is a heuristic.
ANCHOR_MAX = 0.95


@dataclass
class Forecast:
    issued_for: date  # last day of observed data
    generated_at: datetime
    base: np.ndarray  # (leads, ny, nx) model probabilities
    detections_per_day: list[int]
    final: np.ndarray = field(init=False)  # base raised by the anchor
    anchored: np.ndarray = field(init=False)  # (ny, nx) cells touched by the anchor
    anchored_at: datetime | None = None

    def __post_init__(self) -> None:
        self.final = self.base.copy()
        self.anchored = np.zeros(self.base.shape[1:], dtype=bool)

    @property
    def valid_dates(self) -> list[date]:
        return [self.issued_for + timedelta(days=k + 1) for k in range(len(self.base))]


class ForecastService:
    def __init__(
        self, model: ForecastModel, anchor_hours: float = 12.0, source: str = DEFAULT_SOURCE
    ) -> None:
        self.model = model
        self.source = source  # FIRMS NRT source that feeds the model
        self.anchor_hours = anchor_hours
        lags = model.meta["persistence"]["p_fire_after_lag_days_given_fire"]
        # persistence[j] = P(fire j days after a fire), j = 0 … leads-1; j = 0 is 1.
        table = np.array(
            [1.0] + [float(lags[str(j)]) for j in range(1, model.leads)], dtype=np.float32
        )
        # Never let the anchor grow with time (the measured table has small wiggles).
        self.persistence = np.minimum.accumulate(table)
        self.current: Forecast | None = None
        self.last_error: str | None = None

    async def refresh(
        self, client: httpx.AsyncClient, map_key: str, now: datetime | None = None
    ) -> Forecast:
        if not map_key:
            raise RuntimeError("FIRMS_MAP_KEY is not set")
        now = now or datetime.now(UTC)
        issued_for = now.date() - timedelta(days=1)  # last complete UTC day
        counts, per_day = await fetch_daily_counts(
            client, map_key, self.model.grid, issued_for, self.model.long_window
        )
        missing = [i for i, n in enumerate(per_day) if n == 0]
        if len(missing) > self.model.long_window // 3:
            log.warning("Forecast input has %d empty days of %d", len(missing), len(per_day))
        forecast = Forecast(issued_for, now, self.model.predict(counts, issued_for), per_day)
        self.current = forecast
        self.last_error = None
        log.info("Forecast issued for %s (%d detections in 30 days)", issued_for, sum(per_day))
        return forecast

    def apply_anchor(self, fires: pd.DataFrame, now: datetime | None = None) -> int:
        """Raise the forecast where live fires burn now. Returns the cells anchored."""
        f = self.current
        if f is None:
            return 0
        now = now or datetime.now(UTC)
        final = f.base.copy()
        anchored = np.zeros_like(f.anchored)
        if not fires.empty:
            recent = fires[
                fires["last_seen"] >= pd.Timestamp(now - timedelta(hours=self.anchor_hours))
            ]
            row, col, inside = self.model.grid.cell_of(recent["lat"], recent["lon"])
            seen = recent["last_seen"].dt.tz_convert("UTC").dt.date
            first_lead = (seen - f.valid_dates[0]).map(lambda d: d.days).to_numpy(dtype=np.int64)
            # The fused confidence is a heuristic score, not a probability: cap it.
            conf = recent["confidence"].to_numpy(dtype=np.float32).clip(0, ANCHOR_MAX)
            leads = len(f.base)
            for r, c, j0, p in zip(
                row[inside], col[inside], first_lead[inside], conf[inside], strict=True
            ):
                if j0 < 0 or j0 >= leads:
                    continue
                floor = p * self.persistence[: leads - j0]
                final[j0:, r, c] = np.maximum(final[j0:, r, c], floor)
                anchored[r, c] = True
        f.final, f.anchored, f.anchored_at = final, anchored, now
        return int(anchored.sum())

    def payload(
        self, min_p: float = 0.05, bbox: tuple[float, float, float, float] | None = None
    ) -> dict:
        f = self.current
        if f is None:
            raise LookupError(self.last_error or "No forecast yet")
        lat, lon = self.model.grid.centers()
        keep = (f.final.max(0) >= min_p) & (self.model.mask | f.anchored)
        rows, cols = np.nonzero(keep)
        cells = []
        for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
            if bbox and not (bbox[0] <= lon[c] <= bbox[2] and bbox[1] <= lat[r] <= bbox[3]):
                continue
            cells.append(
                {
                    "lat": round(float(lat[r]), 3),
                    "lon": round(float(lon[c]), 3),
                    "p": [round(float(v), 3) for v in f.final[:, r, c]],
                    "anchored": bool(f.anchored[r, c]),
                }
            )
        return {
            "model": self.model.name,
            "issued_for": f.issued_for.isoformat(),
            "valid_dates": [d.isoformat() for d in f.valid_dates],
            "generated_at": f.generated_at.isoformat(),
            "anchored_at": f.anchored_at.isoformat() if f.anchored_at else None,
            "resolution_deg": self.model.grid.res,
            "min_p": min_p,
            "cells": cells,
        }

    def outlook(self, select: np.ndarray) -> dict:
        """Forecast summed over an area, next to the normal for the same dates.

        `select`: (ny, nx) boolean, the cells of the area. Both numbers are expected
        fire cells per day (sum of probabilities over forecast cells): the forecast's,
        and the climatology the model was trained with (VIIRS S-NPP, 2012-2022).
        """
        f = self.current
        if f is None:
            raise LookupError(self.last_error or "No forecast yet")
        cells = select & self.model.mask
        days = []
        for k, valid in enumerate(f.valid_dates):
            expected = float(f.final[k][cells].sum())
            normal = float(self.model.climatology[day_of_year0(valid)][cells].sum())
            days.append(
                {
                    "date": valid.isoformat(),
                    "expected_fire_cells": round(expected, 2),
                    "normal_fire_cells": round(normal, 2),
                    "ratio_to_normal": round(expected / normal, 2) if normal > 0 else None,
                }
            )
        expected = sum(d["expected_fire_cells"] for d in days)
        normal = sum(d["normal_fire_cells"] for d in days)
        return {
            "model": self.model.name,
            "issued_for": f.issued_for.isoformat(),
            "generated_at": f.generated_at.isoformat(),
            "anchored_at": f.anchored_at.isoformat() if f.anchored_at else None,
            "unit": "expected_fire_cells_per_day",
            "resolution_deg": self.model.grid.res,
            "forecast_cells": int(cells.sum()),
            "anchored_cells": int((f.anchored & select).sum()),
            "days": days,
            "week": {
                "expected_fire_cell_days": round(expected, 2),
                "normal_fire_cell_days": round(normal, 2),
                "ratio_to_normal": round(expected / normal, 2) if normal > 0 else None,
            },
        }
