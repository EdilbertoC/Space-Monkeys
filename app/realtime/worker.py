# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Real-time worker: polls GOES and FIRMS, fuses detections and publishes the pulse."""

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import httpx
import pandas as pd

from app.config import Settings
from app.forecast.service import ForecastService
from app.realtime import detections, firms, goes
from app.realtime.geojson import feature_collection
from app.realtime.live import LiveState, PulseDiff
from app.realtime.pulse import PulseBroker

log = logging.getLogger(__name__)


@dataclass
class SourceStatus:
    last_success: datetime | None = None
    last_error: str | None = None
    last_count: int = 0


@dataclass
class WorkerStatus:
    running: bool = False
    last_run: datetime | None = None
    runs: int = 0
    sources: dict[str, SourceStatus] = field(default_factory=dict)

    def source(self, name: str) -> SourceStatus:
        return self.sources.setdefault(name, SourceStatus())


class RealtimeWorker:
    def __init__(
        self,
        settings: Settings,
        state: LiveState,
        broker: PulseBroker,
        forecast: ForecastService | None = None,
    ) -> None:
        self.settings = settings
        self.state = state
        self.broker = broker
        self.forecast = forecast
        self.status = WorkerStatus()
        self._last_goes_key: str | None = None
        self._last_forecast: datetime | None = None
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="realtime-worker")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _loop(self) -> None:
        self.status.running = True
        async with httpx.AsyncClient(follow_redirects=True) as client:
            while True:
                try:
                    await self.run_once(client)
                except Exception:  # keep the pulse alive whatever one cycle does
                    log.exception("Real-time cycle failed")
                await asyncio.sleep(self.settings.poll_interval_seconds)

    async def run_once(self, client: httpx.AsyncClient) -> PulseDiff:
        goes_df, firms_df = await asyncio.gather(self._poll_goes(client), self._poll_firms(client))
        frames = [f for f in (goes_df, firms_df) if not f.empty]
        incoming = pd.concat(frames, ignore_index=True) if frames else goes_df
        diff = self.state.ingest(incoming)
        self.status.last_run = diff.at
        self.status.runs += 1
        self.broker.publish("pulse", pulse_payload(diff, len(self.state.fires)))
        log.info(
            "Pulse: %d new detections, %d new fires, %d updated, %d removed",
            diff.new_detections,
            len(diff.new),
            len(diff.updated),
            len(diff.removed),
        )
        if self.forecast:
            await self._update_forecast(client)
        return diff

    async def _update_forecast(self, client: httpx.AsyncClient) -> None:
        """Refresh the base forecast when due, then re-anchor it on the live fires."""
        assert self.forecast is not None
        status = self.status.source("FORECAST")
        now = datetime.now(UTC)
        refresh_every = timedelta(hours=self.settings.forecast_refresh_hours)
        if self._last_forecast is None or now - self._last_forecast >= refresh_every:
            try:
                await self.forecast.refresh(client, self.settings.firms_map_key, now)
                self._last_forecast = now
                status.last_success, status.last_error = now, None
            except (httpx.HTTPError, firms.FirmsError, RuntimeError, ValueError) as exc:
                log.warning("Forecast refresh failed: %s", exc)
                self.forecast.last_error = status.last_error = str(exc)
        if self.forecast.current is None:
            return
        anchored = self.forecast.apply_anchor(self.state.fires, now)
        status.last_count = anchored
        self.broker.publish(
            "forecast",
            {
                "issued_for": self.forecast.current.issued_for.isoformat(),
                "anchored_at": now.isoformat(),
                "anchored_cells": anchored,
            },
        )

    async def _poll_goes(self, client: httpx.AsyncClient) -> pd.DataFrame:
        s = self.settings
        status = self.status.source(f"GOES:{s.goes_bucket}")
        try:
            df, key = await goes.fetch_latest(
                client,
                s.goes_bucket,
                s.goes_product,
                s.aoi_bbox,
                s.goes_min_confidence,
                skip_key=self._last_goes_key,
            )
        except (httpx.HTTPError, OSError, ValueError) as exc:
            log.warning("GOES poll failed: %s", exc)
            status.last_error = str(exc)
            return detections.empty()
        self._last_goes_key = key or self._last_goes_key
        status.last_success = datetime.now(UTC)
        status.last_error = None
        status.last_count = len(df)
        return df

    async def _poll_firms(self, client: httpx.AsyncClient) -> pd.DataFrame:
        s = self.settings
        if not s.firms_map_key:
            self.status.source("FIRMS").last_error = "FIRMS_MAP_KEY is not set"
            return detections.empty()
        df, errors = await firms.fetch_all(
            client, s.firms_map_key, s.firms_sources, s.aoi_bbox, s.firms_day_range
        )
        now = datetime.now(UTC)
        for source in s.firms_sources:
            status = self.status.source(f"FIRMS:{source}")
            if source in errors:
                status.last_error = errors[source]
            else:
                status.last_success = now
                status.last_error = None
                status.last_count = int((df["source"] == f"FIRMS:{source}").sum())
        return df


def pulse_payload(diff: PulseDiff, total_fires: int) -> dict:
    return {
        "at": diff.at.isoformat(),
        "new_detections": diff.new_detections,
        "total_fires": total_fires,
        "new": feature_collection(diff.new, "fire_id"),
        "updated": feature_collection(diff.updated, "fire_id"),
        "removed": diff.removed,
    }
