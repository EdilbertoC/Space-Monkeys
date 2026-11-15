# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Score the fire forecast models on the whole world, 2024-2025.

Both models forecast fire (yes/no) per 0.25 deg cell for days t+1..t+7, scored on the
same days, cells (world forecast mask: cells that burned in the training years) and
truth (VIIRS S-NPP detections of every country):

* world model (`fire_forecast_world_modis`): runs on the full world grid.
* Brazil model (`fire_forecast_brazil`): never saw data outside Brazil; it runs tile by
  tile (160 x 160 cells) with each tile's own climatology and mask, which measures
  how well what it learned in Brazil transfers to other fire regimes.

References: climatology (world, training years) and persistence (fitted on 2023).
Metrics are streamed: average precision from 10,000 probability bins, exact Brier.

Usage:
  python scripts/evaluate_world.py <grids_dir> <world_model_dir> <brazil_model_dir>
grids_dir holds world_2023.npz, world_2024.npz, world_2025.npz from the notebook
(REGION "world"); world_model_dir holds fire_forecast_world_modis.onnx,
climatology_world_modis.npy and forecast_mask_world_modis.npy.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

W, S, E, N, RES = -180.0, -60.0, 180.0, 80.0, 0.25
NY, NX = 560, 1440
BR_ROW, BR_COL = 296, 424  # Brazil window inside the world grid
TILE = 160
LEADS, HIST, LONG = 7, 7, 30
BINS = 10_000
STEP = 6  # score every 6th day


class Scores:
    """Streaming average precision (binned) and Brier score per lead day."""

    def __init__(self) -> None:
        self.pos = np.zeros((LEADS, BINS), dtype=np.int64)
        self.neg = np.zeros((LEADS, BINS), dtype=np.int64)
        self.sq = np.zeros(LEADS)
        self.n = np.zeros(LEADS)

    def add(self, k: int, p: np.ndarray, y: np.ndarray) -> None:
        b = np.clip((p * BINS).astype(np.int64), 0, BINS - 1)
        self.pos[k] += np.bincount(b[y], minlength=BINS)
        self.neg[k] += np.bincount(b[~y], minlength=BINS)
        self.sq[k] += float(((p - y) ** 2).sum())
        self.n[k] += len(p)

    def table(self) -> pd.DataFrame:
        rows = []
        for k in range(LEADS):
            tp = np.cumsum(self.pos[k][::-1])
            fp = np.cumsum(self.neg[k][::-1])
            precision = tp / np.maximum(tp + fp, 1)
            recall = tp / max(tp[-1], 1)
            ap = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
            rows.append({"lead_day": k + 1, "avg_precision": ap, "brier": self.sq[k] / self.n[k]})
        return pd.DataFrame(rows).set_index("lead_day")


def main(grids: Path, world_dir: Path, brazil_dir: Path) -> None:
    days = pd.date_range("2023-01-01", "2025-12-31", freq="D")
    doy = days.dayofyear.to_numpy() - 1
    cube = np.zeros((len(days), NY, NX), dtype=np.uint8)
    i = 0
    for year in (2023, 2024, 2025):
        c = np.load(grids / f"world_{year}.npz")["counts"]
        cube[i : i + len(c)] = c
        i += len(c)
    fire = cube > 0
    print(f"World VIIRS cube {cube.shape} loaded", file=sys.stderr)

    clim = np.load(world_dir / "climatology_world_modis.npy", mmap_mode="r")  # float16
    mask = np.load(world_dir / "forecast_mask_world_modis.npy")
    world = ort.InferenceSession(str(world_dir / "fire_forecast_world_modis.onnx"))
    brazil = ort.InferenceSession(str(brazil_dir / "model.onnx"))
    in_brazil = np.zeros((NY, NX), dtype=bool)
    in_brazil[BR_ROW : BR_ROW + TILE, BR_COL : BR_COL + TILE] = True
    regions = {"world": mask, "brazil": mask & in_brazil, "outside_brazil": mask & ~in_brazil}

    lat = ((N - RES / 2 - np.arange(NY) * RES) / 90).astype(np.float32)
    lon = ((W + RES / 2 + np.arange(NX) * RES) / 180).astype(np.float32)
    lat_w = np.repeat(lat[:, None], NX, 1)
    lon_w = np.repeat(lon[None], NY, 0)
    lat_b = np.repeat(np.linspace(1, -1, TILE, dtype=np.float32)[:, None], TILE, 1)
    lon_b = np.repeat(np.linspace(-1, 1, TILE, dtype=np.float32)[None], TILE, 0)

    # Persistence fitted on 2023: P(fire at t+k | fire in the last 3 days).
    t2023 = np.arange(LONG, 365 - LEADS, 5)
    persist = np.zeros((LEADS, 2))
    for k in range(LEADS):
        r = np.stack([fire[t - 2 : t + 1].any(0)[mask] for t in t2023])
        y = np.stack([fire[t + k + 1][mask] for t in t2023])
        persist[k] = [y[~r].mean(), y[r].mean()]

    names = ["climatology", "persistence", "unet_brazil", "unet_world_modis"]
    scores = {(m, reg): Scores() for m in names for reg in regions}
    test_t = np.arange(days.get_loc(pd.Timestamp("2024-01-01")), len(days) - LEADS, STEP)
    t_start = time.time()
    for n_done, t in enumerate(test_t, 1):
        window = cube[t - LONG + 1 : t + 1].astype(np.float32)
        hist = np.log1p(window[-HIST:])
        r30 = np.log1p(window.sum(0))
        c7 = np.stack([clim[doy[t + k + 1]] for k in range(LEADS)]).astype(np.float32)
        angle = 2 * math.pi * doy[t] / 365.25
        sin = np.full((NY, NX), math.sin(angle), np.float32)
        cos = np.full((NY, NX), math.cos(angle), np.float32)
        m32 = mask.astype(np.float32)

        # World model: VIIRS channels + flag, MODIS channels and flag at 0 (after 2016).
        x = np.concatenate(
            [
                hist,
                r30[None],
                np.ones((1, NY, NX), np.float32),
                np.zeros((9, NY, NX), np.float32),
                c7,
                sin[None],
                cos[None],
                m32[None],
                lat_w[None],
                lon_w[None],
            ]
        )[None]
        p_world = 1 / (1 + np.exp(-world.run(None, {"features": x})[0][0, :LEADS]))

        # Brazil model, tile by tile. Tiles are shifted so that one of them is exactly
        # the Brazil window the model was trained on.
        x_all = np.concatenate([hist, r30[None], c7, sin[None], cos[None], m32[None]])
        top, left = (-BR_ROW) % TILE, (-BR_COL) % TILE
        bottom, right = (-(NY + top)) % TILE, (-(NX + left)) % TILE
        x_all = np.pad(x_all, ((0, 0), (top, bottom), (left, right)))
        p_pad = np.zeros((LEADS,) + x_all.shape[1:], np.float32)
        for r0 in range(0, x_all.shape[1], TILE):
            for c0 in range(0, x_all.shape[2], TILE):
                tile = x_all[:, r0 : r0 + TILE, c0 : c0 + TILE]
                if tile[-1].sum() == 0:  # no forecast cells in this tile
                    continue
                xt = np.concatenate([tile, lat_b[None], lon_b[None]])[None]
                out = brazil.run(None, {"features": xt})[0][0]
                p_pad[:, r0 : r0 + TILE, c0 : c0 + TILE] = 1 / (1 + np.exp(-out))
        p_br = p_pad[:, top : top + NY, left : left + NX]

        recent = fire[t - 2 : t + 1].any(0)
        for k in range(LEADS):
            y_full = fire[t + k + 1]
            preds = {
                "climatology": c7[k],
                "persistence": np.where(recent, persist[k, 1], persist[k, 0]),
                "unet_brazil": p_br[k],
                "unet_world_modis": p_world[k],
            }
            for reg, sel in regions.items():
                y = y_full[sel]
                for name, p in preds.items():
                    scores[(name, reg)].add(k, p[sel].astype(np.float32), y)
        if n_done % 20 == 0:
            print(f"{n_done}/{len(test_t)} days ({time.time() - t_start:.0f}s)", file=sys.stderr)

    out = {}
    for reg in regions:
        ap = pd.DataFrame({m: scores[(m, reg)].table()["avg_precision"] for m in names})
        br = pd.DataFrame({m: scores[(m, reg)].table()["brier"] for m in names})
        print(f"\n=== {reg} ({int(regions[reg].sum()):,} cells), 2024-2025, every {STEP} days")
        print("Average precision (higher is better)\n", ap.round(4))
        print("Brier (lower is better)\n", br.round(5))
        print("Mean:", ap.mean().round(4).to_dict(), br.mean().round(5).to_dict())
        out[reg] = {
            "avg_precision": ap.mean().round(4).to_dict(),
            "brier": br.mean().round(5).to_dict(),
        }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
