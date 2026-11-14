"""Live state: recent detections from every sensor and the fire events fused from them."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pandas as pd

from app.fusion.fuse import fuse
from app.realtime import detections


@dataclass
class PulseDiff:
    at: datetime
    new: pd.DataFrame
    updated: pd.DataFrame
    removed: list[str]
    new_detections: int

    @property
    def is_empty(self) -> bool:
        return self.new.empty and self.updated.empty and not self.removed


# A fire counts as updated when any of these change.
_TRACKED = ["lat", "lon", "uncertainty_km", "confidence", "n_detections", "last_seen"]


@dataclass
class LiveState:
    retention_hours: float = 48.0
    fusion_radius_km: float = 2.0
    fusion_window_hours: float = 12.0
    detections: pd.DataFrame = field(default_factory=detections.empty)
    fires: pd.DataFrame = field(default_factory=lambda: fuse(detections.empty()))
    updated_at: datetime | None = None

    def ingest(self, incoming: pd.DataFrame, now: datetime | None = None) -> PulseDiff:
        """Add detections, drop expired ones, re-fuse and return what changed."""
        now = now or datetime.now(UTC)
        known = set(self.detections["detection_id"])
        fresh = incoming[~incoming["detection_id"].isin(known)] if not incoming.empty else incoming

        frames = [f for f in (self.detections, fresh) if not f.empty]
        merged = pd.concat(frames, ignore_index=True) if frames else detections.empty()
        cutoff = pd.Timestamp(now - timedelta(hours=self.retention_hours))
        merged = merged[merged["time"] >= cutoff].reset_index(drop=True)

        previous = self.fires.set_index("fire_id")
        current = fuse(merged, self.fusion_radius_km, self.fusion_window_hours)

        self.detections = merged
        self.fires = current
        self.updated_at = now

        current_idx = current.set_index("fire_id")
        new_ids = current_idx.index.difference(previous.index)
        common = current_idx.index.intersection(previous.index)
        changed = [
            fid
            for fid in common
            if any(current_idx.at[fid, col] != previous.at[fid, col] for col in _TRACKED)
        ]
        removed = previous.index.difference(current_idx.index).tolist()

        return PulseDiff(
            at=now,
            new=current_idx.loc[new_ids].reset_index(),
            updated=current_idx.loc[changed].reset_index(),
            removed=removed,
            new_detections=len(fresh),
        )
