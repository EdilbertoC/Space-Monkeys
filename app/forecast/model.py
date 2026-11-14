# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Runs the trained fire forecast U-Net (ONNX) on the CPU.

The input channels must be built exactly as in the training notebook
(`notebooks/fire_forecast_base.py`); `meta.json` in the model folder records the spec.
"""

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class Grid:
    """Regular lat/lon grid; row 0 is the northern edge."""

    west: float
    south: float
    east: float
    north: float
    res: float

    @property
    def shape(self) -> tuple[int, int]:
        return (
            int(round((self.north - self.south) / self.res)),
            int(round((self.east - self.west) / self.res)),
        )

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)

    def cell_of(
        self, lat: np.ndarray, lon: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(row, col, inside) for each point."""
        ny, nx = self.shape
        row = np.floor((self.north - np.asarray(lat, dtype=float)) / self.res).astype(np.int64)
        col = np.floor((np.asarray(lon, dtype=float) - self.west) / self.res).astype(np.int64)
        inside = (row >= 0) & (row < ny) & (col >= 0) & (col < nx)
        return row, col, inside

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        ny, nx = self.shape
        lat = self.north - self.res / 2 - np.arange(ny) * self.res
        lon = self.west + self.res / 2 + np.arange(nx) * self.res
        return lat, lon


def day_of_year0(d: date) -> int:
    """0-based day of year, as pandas `dayofyear - 1` in the notebook."""
    return d.timetuple().tm_yday - 1


class ForecastModel:
    def __init__(self, model_dir: Path) -> None:
        self.dir = Path(model_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self.grid = Grid(*self.meta["bbox"], self.meta["resolution_deg"])
        self.history_days = int(self.meta["history_days"])
        self.long_window = int(self.meta["long_window_days"])
        self.leads = int(self.meta["leads"])
        self.session = ort.InferenceSession(
            str(self.dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.climatology = np.load(self.dir / "climatology.npy", allow_pickle=False).astype(
            np.float32
        )
        self.mask = np.load(self.dir / "mask.npy", allow_pickle=False).astype(bool)
        ny, nx = self.grid.shape
        if self.mask.shape != (ny, nx) or self.climatology.shape != (366, ny, nx):
            raise ValueError(f"Model files do not match the grid {ny}x{nx}")
        # Position planes as in the training notebook that produced this model.
        self.lat_plane = np.repeat(np.linspace(1, -1, ny, dtype=np.float32)[:, None], nx, 1)
        self.lon_plane = np.repeat(np.linspace(-1, 1, nx, dtype=np.float32)[None, :], ny, 0)

    @property
    def name(self) -> str:
        return self.meta["model"]

    def features(self, counts: np.ndarray, issue_date: date) -> np.ndarray:
        """Input channels for a forecast issued at the end of `issue_date`.

        `counts`: (long_window, ny, nx) daily fire-pixel counts for the days
        issue_date - long_window + 1 … issue_date.
        """
        ny, nx = self.grid.shape
        if counts.shape != (self.long_window, ny, nx):
            raise ValueError(f"counts must be {(self.long_window, ny, nx)}, got {counts.shape}")
        counts = counts.astype(np.float32)
        n_ch = self.history_days + 1 + self.leads + 2 + 3
        x = np.empty((n_ch, ny, nx), dtype=np.float32)
        x[: self.history_days] = np.log1p(counts[-self.history_days :])
        x[self.history_days] = np.log1p(counts.sum(0))
        c = self.history_days + 1
        doy = day_of_year0(issue_date)
        for k in range(self.leads):
            target = date.fromordinal(issue_date.toordinal() + k + 1)
            x[c + k] = self.climatology[day_of_year0(target)]
        c += self.leads
        angle = 2 * math.pi * doy / 365.25
        x[c], x[c + 1] = math.sin(angle), math.cos(angle)
        x[c + 2] = self.mask
        x[c + 3], x[c + 4] = self.lat_plane, self.lon_plane
        return x

    def predict(self, counts: np.ndarray, issue_date: date) -> np.ndarray:
        """(leads, ny, nx) fire probabilities for issue_date + 1 … issue_date + leads."""
        x = self.features(counts, issue_date)[None]
        logits = self.session.run(None, {self.input_name: x})[0][0, : self.leads]
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
