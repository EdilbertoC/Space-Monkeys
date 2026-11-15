# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Step 7: a small U-Net for the VIIRS -> MODIS calibration, scored against LightGBM.

Same question as pipeline.baselines: "did MODIS detect fire in this 0.1° cell on
this UTC day?" (0/1), answered only from what one VIIRS satellite saw. LightGBM
gets hand-made neighbour and lag summaries per row; the U-Net gets the region as
an image and learns its own spatial context.

Inputs for a target day t, one image per day over the region grid:

- for each day t-3..t (HISTORY), the satellite's own cell-day values from the
  cube: log1p of pixels, FRP sum, FRP max, day, night and high-confidence pixels
  and overpasses, plus the mean scan size (0 where it saw nothing). LightGBM's
  lags also reach day t-3, so both models see the same days; the U-Net sees a
  wider neighbourhood (its receptive field) instead of 3x3/5x5 summaries.
- day of year of t (sin, cos) and the cell position (row and column planes).

Never MODIS: MODIS will not exist when the model runs.

Rows, loss and split follow the training table and LightGBM exactly:

- the loss counts only candidate cell-days (the satellite saw fire in the 3x3
  neighbourhood that day or the day before, the rule of pipeline.training) in
  cells of a biome (Brazil). The rest of the image is input only. The whole grid
  is used at once rather than patches: it fits on a 6 GB GPU and the loss mask
  already restricts learning to where fire exists.
- positive class weighted by n_negative / n_positive (BCE pos_weight). The odds
  are divided back (baselines.unweight), then an isotonic calibration is fitted
  on the held-out last training year, which also picks the best epoch (early
  stopping on average precision). The test years are never seen.
- S-NPP: train 2012-2016, held-out 2017, test 2018 onward.

The test scores the U-Net and the national LightGBM (models/<SAT>/lightgbm.txt
from pipeline.crossbiome) on the same rows of the training table, with the same
metrics (baselines.evaluate_arrays), for Brazil and per biome.

Output: data/processed/<region>/models/<SAT>/unet/ (model.pt, report.json)

Usage::

    python -m pipeline.unet brazil                  # S-NPP, CUDA GPU recommended
    python -m pipeline.unet brazil --satellite N20
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, log_loss
from torch import nn

from pipeline import baselines, crossbiome
from pipeline.cube import N_COLS
from pipeline.training import FEATURES, SATELLITES, TARGET

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"

# Cube column, log1p? One input channel each, per history day.
INPUTS = [
    ("n_pixels", True),
    ("frp_sum_mw", True),
    ("frp_max_mw", True),
    ("n_day", True),
    ("n_night", True),
    ("n_high", True),
    ("n_overpasses", True),
    ("scan_mean_km", False),
]
N_IN = len(INPUTS)
HISTORY = 4  # days t-3..t, the reach of LightGBM's lag features
N_CONTEXT = 4  # doy sin, doy cos, row plane, column plane
IN_CHANNELS = HISTORY * N_IN + N_CONTEXT
MARGIN = 2  # cells around the data, so every candidate cell is on the grid
MULTIPLE = 16  # grid sides are padded to this (up to 4 poolings)
SEED = 42


# ---- data -------------------------------------------------------------------
@dataclass(frozen=True)
class Grid:
    """A window of the global 0.1° grid (cube.cell_id rows and columns)."""

    row0: int
    col0: int
    nrows: int
    ncols: int

    @classmethod
    def around(cls, cell_ids: np.ndarray) -> Grid:
        rows, cols = cell_ids // N_COLS, cell_ids % N_COLS
        r0, c0 = int(rows.min()) - MARGIN, int(cols.min()) - MARGIN
        span_r = int(rows.max()) + MARGIN + 1 - r0
        span_c = int(cols.max()) + MARGIN + 1 - c0
        up = lambda n: -(-n // MULTIPLE) * MULTIPLE  # noqa: E731
        return cls(r0, c0, up(span_r), up(span_c))

    def index(self, cell_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return cell_ids // N_COLS - self.row0, cell_ids % N_COLS - self.col0

    def cell_ids(self, r: np.ndarray, c: np.ndarray) -> np.ndarray:
        return (np.asarray(r) + self.row0) * N_COLS + np.asarray(c) + self.col0


@dataclass
class Record:
    """Sparse VIIRS inputs and MODIS targets on a grid, sorted by day index.

    Day 0 is the satellite's first day; the last day is the last date with
    standard processing for both MODIS and the satellite (as pipeline.training)."""

    satellite: str
    grid: Grid
    first: date
    n_days: int
    v_day: np.ndarray
    v_r: np.ndarray
    v_c: np.ndarray
    v_x: np.ndarray  # (rows, N_IN) float32, already transformed
    m_day: np.ndarray
    m_r: np.ndarray
    m_c: np.ndarray
    region: np.ndarray  # (nrows, ncols) bool: cells that are scored (a biome)

    def __post_init__(self) -> None:
        self.v_ptr = np.searchsorted(self.v_day, np.arange(self.n_days + 1))
        self.m_ptr = np.searchsorted(self.m_day, np.arange(self.n_days + 1))

    def day(self, d: date) -> int:
        return (d - self.first).days

    def date(self, i: int) -> date:
        return self.first + timedelta(days=int(i))

    def days_in_years(self, first_year: int, last_year: int) -> np.ndarray:
        a = max(0, self.day(date(first_year, 1, 1)))
        b = min(self.n_days, self.day(date(last_year + 1, 1, 1)))
        return np.arange(a, b) if b > a else np.arange(0)


def load_record(region: str, data_dir: Path, satellite: str) -> Record:
    cube_dir = data_dir / "processed" / region / "cube"
    if not cube_dir.exists():
        raise FileNotFoundError(f"{cube_dir} not found; run `python -m pipeline.cube {region}`")
    con = duckdb.connect()
    con.execute(f"""
        CREATE TEMP VIEW cube AS
        SELECT * FROM read_parquet('{cube_dir.as_posix()}/**/*.parquet', hive_partitioning = true)
        WHERE NOT nrt
    """)
    last = con.execute(
        "SELECT least(max(date) FILTER (sensor = 'MODIS'), max(date) FILTER (satellite = ?)) "
        "FROM cube",
        [satellite],
    ).fetchone()[0]
    if last is None:
        raise ValueError(f"no standard-processing {satellite} and MODIS data in {cube_dir}")
    first = date.fromisoformat(SATELLITES[satellite]["start"])
    period = f"date BETWEEN DATE '{first}' AND DATE '{last}'"
    day = f"date_diff('day', DATE '{first}', date)"
    cols = ", ".join(f"CAST(coalesce({c}, 0) AS FLOAT) AS {c}" for c, _ in INPUTS)
    v = con.execute(
        f"SELECT cell_id, {day} AS d, {cols} FROM cube "
        f"WHERE satellite = ? AND {period} ORDER BY d, cell_id",
        [satellite],
    ).fetchnumpy()
    m = con.execute(
        f"SELECT DISTINCT cell_id, {day} AS d FROM cube "
        f"WHERE sensor = 'MODIS' AND {period} ORDER BY d, cell_id"
    ).fetchnumpy()
    v_cells = np.asarray(v["cell_id"], dtype=np.int64)
    m_cells = np.asarray(m["cell_id"], dtype=np.int64)
    grid = Grid.around(np.concatenate([v_cells, m_cells]))

    x = np.stack([np.asarray(v[c], dtype=np.float32) for c, _ in INPUTS], axis=1)
    for j, (_, log) in enumerate(INPUTS):
        if log:
            x[:, j] = np.log1p(x[:, j])

    region_mask = np.ones((grid.nrows, grid.ncols), bool)
    biome_file = data_dir / "processed" / region / "cell_biome.parquet"
    if biome_file.exists():
        b = con.execute(
            f"SELECT cell_id FROM read_parquet('{biome_file.as_posix()}') WHERE biome IS NOT NULL"
        ).fetchnumpy()["cell_id"]
        r, c = grid.index(np.asarray(b, dtype=np.int64))
        ok = (r >= 0) & (r < grid.nrows) & (c >= 0) & (c < grid.ncols)
        region_mask[:] = False
        region_mask[r[ok], c[ok]] = True
    con.close()

    vr, vc = grid.index(v_cells)
    mr, mc = grid.index(m_cells)
    return Record(
        satellite=satellite,
        grid=grid,
        first=first,
        n_days=(last - first).days + 1,
        v_day=np.asarray(v["d"], dtype=np.int64),
        v_r=vr,
        v_c=vc,
        v_x=x,
        m_day=np.asarray(m["d"], dtype=np.int64),
        m_r=mr,
        m_c=mc,
        region=region_mask,
    )


def _segments(ptr: np.ndarray, days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row indices of the given days (sorted record) and the position of their day."""
    starts, lens = ptr[days], ptr[days + 1] - ptr[days]
    owner = np.repeat(np.arange(len(days)), lens)
    offset = np.repeat(starts - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
    return owner, offset + np.arange(int(lens.sum()))


class Batcher:
    """Builds dense input images, targets and loss masks on the device."""

    def __init__(self, rec: Record, device: torch.device) -> None:
        self.rec, self.device = rec, device
        as_t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
        self.v_r, self.v_c = as_t(rec.v_r, torch.long), as_t(rec.v_c, torch.long)
        self.v_x = as_t(rec.v_x, torch.float32)
        self.m_r, self.m_c = as_t(rec.m_r, torch.long), as_t(rec.m_c, torch.long)
        self.region = as_t(rec.region, torch.bool)
        h, w = rec.grid.nrows, rec.grid.ncols
        rows = torch.linspace(1, -1, h, device=device)[:, None].expand(h, w)
        cols = torch.linspace(-1, 1, w, device=device)[None, :].expand(h, w)
        self.position = torch.stack([rows, cols])

    def batch(self, days: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x (B, IN_CHANNELS, H, W), y (B, H, W) MODIS 0/1, mask (B, H, W) scored rows."""
        rec, dev = self.rec, self.device
        days = np.asarray(days, dtype=np.int64)
        b, h, w = len(days), rec.grid.nrows, rec.grid.ncols
        hist = (days[:, None] - np.arange(HISTORY)[None, :]).ravel()
        slots = np.flatnonzero(hist >= 0)
        owner, idx = _segments(rec.v_ptr, hist[slots])
        x5 = torch.zeros(b * HISTORY, h, w, N_IN, device=dev)
        ii = torch.as_tensor(idx, device=dev)
        x5[torch.as_tensor(slots[owner], device=dev), self.v_r[ii], self.v_c[ii]] = self.v_x[ii]
        x5 = x5.view(b, HISTORY, h, w, N_IN)

        fire = (x5[:, :2, :, :, 0] > 0).any(dim=1).float()  # own fire on t or t-1
        cand = F.max_pool2d(fire[:, None], 3, stride=1, padding=1)[:, 0] > 0
        mask = cand & self.region

        doy = np.array([rec.date(d).timetuple().tm_yday - 1 for d in days]) * 2 * math.pi / 365.25
        season = torch.tensor(np.stack([np.sin(doy), np.cos(doy)], 1), dtype=torch.float32)
        x = torch.cat(
            [
                x5.permute(0, 1, 4, 2, 3).reshape(b, HISTORY * N_IN, h, w),
                season.to(dev)[:, :, None, None].expand(b, 2, h, w),
                self.position.expand(b, 2, h, w),
            ],
            dim=1,
        )
        owner, idx = _segments(rec.m_ptr, days)
        y = torch.zeros(b, h, w, device=dev)
        ii = torch.as_tensor(idx, device=dev)
        y[torch.as_tensor(owner, device=dev), self.m_r[ii], self.m_c[ii]] = 1.0
        return x, y, mask


def chunks(days: np.ndarray, size: int) -> Iterator[np.ndarray]:
    for i in range(0, len(days), size):
        yield days[i : i + size]


# ---- model ------------------------------------------------------------------
def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(8, cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.GroupNorm(8, cout),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Small 2D U-Net: one logit per cell. base must be a multiple of 8."""

    def __init__(self, in_ch: int = IN_CHANNELS, base: int = 32, depth: int = 3) -> None:
        super().__init__()
        chs = [base * 2**i for i in range(depth + 1)]
        self.down = nn.ModuleList(
            [_block(in_ch, chs[0])] + [_block(chs[i], chs[i + 1]) for i in range(depth)]
        )
        self.up = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(depth))]
        )
        self.dec = nn.ModuleList([_block(2 * chs[i], chs[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.down):
            x = block(F.max_pool2d(x, 2) if i else x)
            skips.append(x)
        skips.pop()
        for up, dec in zip(self.up, self.dec, strict=True):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        return self.head(x)[:, 0]


def _autocast(device: torch.device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


@torch.no_grad()
def predict_grid(model: UNet, batcher: Batcher, days: np.ndarray, batch: int) -> np.ndarray:
    """Weighted-model probabilities for every cell, (len(days), H, W) float32."""
    model.eval()
    out = []
    for part in chunks(days, batch):
        x, _, _ = batcher.batch(part)
        with _autocast(batcher.device):
            out.append(torch.sigmoid(model(x).float()).cpu().numpy())
    h, w = batcher.rec.grid.nrows, batcher.rec.grid.ncols
    return np.concatenate(out) if out else np.zeros((0, h, w), np.float32)


@torch.no_grad()
def gather(
    model: UNet, batcher: Batcher, days: np.ndarray, batch: int
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted-model probabilities and MODIS labels at the scored rows of `days`."""
    model.eval()
    ps, ys = [], []
    for part in chunks(days, batch):
        x, y, m = batcher.batch(part)
        with _autocast(batcher.device):
            p = torch.sigmoid(model(x).float())
        ps.append(p[m].cpu().numpy())
        ys.append(y[m].cpu().numpy().astype(np.int8))
    return np.concatenate(ps), np.concatenate(ys)


# ---- training ---------------------------------------------------------------
def class_counts(batcher: Batcher, days: np.ndarray, batch: int) -> tuple[int, int]:
    rows = pos = 0
    for part in chunks(days, batch):
        _, y, m = batcher.batch(part)
        rows += int(m.sum())
        pos += int(y[m].sum())
    return rows, pos


def fit(
    rec: Record,
    device: torch.device,
    train_days: np.ndarray,
    valid_days: np.ndarray,
    epochs: int = 30,
    batch: int = 4,
    lr: float = 1e-3,
    patience: int = 5,
    base: int = 32,
    log: Callable[[str], None] = print,
) -> tuple[UNet, dict]:
    """Train with the positive class weighted; keep the epoch with the best average
    precision on valid_days, then calibrate on valid_days (isotonic)."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    batcher = Batcher(rec, device)
    rows, pos = class_counts(batcher, train_days, 16)
    if not pos or pos == rows:
        raise ValueError("training rows need both MODIS fire and no fire")
    w = (rows - pos) / pos
    pos_weight = torch.tensor(w, device=device)

    model = UNet(IN_CHANNELS, base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history, best_ap, best_state, best_epoch = [], -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total, steps = time.time(), 0.0, 0
        for part in chunks(rng.permutation(train_days), batch):
            x, y, m = batcher.batch(part)
            if not m.any():
                continue
            with _autocast(device):
                logit = model(x)
            loss = F.binary_cross_entropy_with_logits(logit.float()[m], y[m], pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total, steps = total + loss.item(), steps + 1
        sched.step()

        p, y = gather(model, batcher, valid_days, 2 * batch)
        pu = baselines.unweight(p, w)
        ap = float(average_precision_score(y, pu)) if y.sum() else 0.0
        entry = {
            "epoch": epoch,
            "train_loss": round(total / max(steps, 1), 5),
            "valid_ap": round(ap, 4),
            "valid_log_loss": round(
                float(log_loss(y, np.clip(pu, 1e-4, 1 - 1e-4), labels=[0, 1])), 5
            ),
            "valid_bias": round(float(pu.sum() / max(y.sum(), 1) - 1), 4),
            "seconds": round(time.time() - t0, 1),
        }
        history.append(entry)
        log(
            f"epoch {epoch:>2}: loss {entry['train_loss']:.4f}, valid AP {ap:.4f}, "
            f"log loss {entry['valid_log_loss']:.4f}, bias {entry['valid_bias']:+.1%} "
            f"({entry['seconds']:.0f} s)"
        )
        if ap > best_ap:
            best_ap, best_epoch = ap, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    p, y = gather(model, batcher, valid_days, 2 * batch)
    pu = baselines.unweight(p, w)
    info = {
        "positive_weight": w,
        "train_rows": rows,
        "train_positives": pos,
        "valid_rows": int(len(y)),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "valid_bias_before_calibration": round(float(pu.sum() / max(y.sum(), 1) - 1), 4),
        "calibrator": baselines.fit_calibration(pu, y),
        "history": history,
    }
    return model, info


# ---- evaluation -------------------------------------------------------------
def score_test(
    model: UNet,
    info: dict,
    rec: Record,
    device: torch.device,
    region: str,
    data_dir: Path,
    batch: int = 8,
    with_lightgbm: bool = True,
) -> dict:
    """U-Net and LightGBM on the test rows of the training table (Brazil cells)."""
    satellite = rec.satellite
    table = crossbiome._table(data_dir, region, satellite)
    lgb_model = None
    if with_lightgbm:
        lgb_model = baselines.load_lightgbm(data_dir / "processed" / region / "models" / satellite)
    models = ["naive", "unet_uncalibrated", "unet"] + (["lightgbm_brazil"] if lgb_model else [])
    parts: dict[str, list] = {k: [] for k in ["y", "cell_id", "year", "month", "biome", *models]}
    batcher = Batcher(rec, device)
    con = duckdb.connect()
    years = [
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT year FROM {table} WHERE split = 'test' ORDER BY 1"
        ).fetchall()
    ]
    row_check = []
    for yr in years:
        df = con.execute(f"""
            SELECT cell_id, date, year, biome, {TARGET}, {", ".join(FEATURES)}
            FROM {table} WHERE split = 'test' AND year = {yr} AND biome IS NOT NULL
        """).fetchdf()
        df["date"] = pd.to_datetime(df["date"])
        days = rec.days_in_years(yr, yr)
        grid = predict_grid(model, batcher, days, batch)
        d = (df["date"] - pd.Timestamp(rec.first)).dt.days.to_numpy(np.int64) - days[0]
        r, c = rec.grid.index(df["cell_id"].to_numpy(np.int64))
        raw = baselines.unweight(grid[d, r, c], info["positive_weight"])
        del grid
        mask_rows, _ = class_counts(batcher, days, 16)
        row_check.append({"year": int(yr), "table_rows": len(df), "unet_mask_rows": mask_rows})

        parts["y"].append(df[TARGET].to_numpy(np.int8))
        parts["cell_id"].append(df["cell_id"].to_numpy(np.int64))
        parts["year"].append(df["year"].to_numpy(np.int16))
        parts["month"].append(df["date"].dt.month.to_numpy(np.int8))
        parts["biome"].append(df["biome"].to_numpy(object))
        parts["naive"].append(baselines.naive_predict(df).astype(np.float32))
        parts["unet_uncalibrated"].append(raw.astype(np.float32))
        parts["unet"].append(
            baselines.apply_calibration(raw, info["calibrator"]).astype(np.float32)
        )
        if lgb_model:
            booster, lw, lcal = lgb_model
            parts["lightgbm_brazil"].append(
                baselines.lightgbm_predict(booster, df, lw, lcal).astype(np.float32)
            )
        del df
    con.close()
    cols = {k: np.concatenate(v) for k, v in parts.items()}
    return {
        "test_years": [years[0], years[-1]],
        "row_check": row_check,
        "results": crossbiome.score_groups(cols, models),
    }


# ---- driver -----------------------------------------------------------------
def run(
    region: str = "brazil",
    data_dir: Path = DEFAULT_DATA_DIR,
    satellite: str = "SNPP",
    epochs: int = 30,
    batch: int = 4,
    lr: float = 1e-3,
    patience: int = 5,
    base: int = 32,
    device: str | None = None,
    with_lightgbm: bool = True,
    log: Callable[[str], None] = print,
) -> dict:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.time()
    rec = load_record(region, data_dir, satellite)
    log(
        f"{satellite}: {rec.first} + {rec.n_days} days, {len(rec.v_day):,} VIIRS and "
        f"{len(rec.m_day):,} MODIS cell-days, grid {rec.grid.nrows}x{rec.grid.ncols}, "
        f"{int(rec.region.sum()):,} scored cells, device {dev}"
    )
    last_train = SATELLITES[satellite]["train_last_year"]
    first_year = rec.first.year
    train_days = rec.days_in_years(first_year, last_train - 1)
    valid_days = rec.days_in_years(last_train, last_train)
    model, info = fit(rec, dev, train_days, valid_days, epochs, batch, lr, patience, base, log)
    log("scoring the test years" + (" (U-Net and LightGBM)" if with_lightgbm else ""))
    test = score_test(model, info, rec, dev, region, data_dir, 2 * batch, with_lightgbm)

    out = data_dir / "processed" / region / "models" / satellite / "unet"
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "satellite": satellite,
        "grid": asdict(rec.grid),
        "first_day": str(rec.first),
        "inputs": [c for c, _ in INPUTS],
        "log1p": [c for c, log1p in INPUTS if log1p],
        "history_days": HISTORY,
        "context": ["doy_sin", "doy_cos", "row_plane", "col_plane"],
        "base_channels": base,
        "positive_weight": info["positive_weight"],
        "calibrator": info["calibrator"],
    }
    torch.save({"state_dict": model.state_dict(), "meta": meta}, out / "model.pt")
    report = {
        "region": region,
        "satellite": satellite,
        "train_years": [first_year, last_train - 1],
        "valid_year": last_train,
        "parameters": sum(p.numel() for p in model.parameters()),
        "settings": {"epochs": epochs, "batch": batch, "lr": lr, "patience": patience},
        "device": str(dev),
        "minutes": round((time.time() - t0) / 60, 1),
        "unet": {k: v for k, v in info.items() if k != "calibrator"},
        **test,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("region", nargs="?", default="brazil")
    parser.add_argument("--satellite", choices=list(SATELLITES), default="SNPP")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--base", type=int, default=32, help="channels of the first level")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--no-lightgbm", action="store_true", help="skip the LightGBM comparison")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)
    r = run(
        args.region,
        args.data_dir,
        args.satellite,
        args.epochs,
        args.batch,
        args.lr,
        args.patience,
        args.base,
        args.device,
        not args.no_lightgbm,
    )
    u = r["unet"]
    print(
        f"U-Net: {r['parameters']:,} parameters, best epoch {u['best_epoch']} of "
        f"{u['epochs_run']}, {r['minutes']} min; test {r['test_years'][0]}-{r['test_years'][1]}"
    )
    bad = [c for c in r["row_check"] if c["table_rows"] != c["unet_mask_rows"]]
    if bad:
        print(f"WARNING: candidate rows differ from the training table: {bad}")
    print(
        f"{'biome':<17}{'model':<19}{'bias':>9}{'month WAPE':>12}{'cell-mo WAPE':>14}"
        f"{'AUC':>7}{'AP':>7}"
    )
    for biome, models in r["results"].items():
        for i, (name, m) in enumerate(models.items()):
            t, rw = m["totals"], m["rows"]
            print(
                f"{biome if i == 0 else '':<17}{name:<19}{t['bias']:>+9.1%}"
                f"{t['monthly_wape']:>12.1%}{t['cell_month_wape']:>14.1%}"
                f"{rw['roc_auc'] or 0:>7.3f}{rw['average_precision'] or 0:>7.3f}"
            )


if __name__ == "__main__":
    main()
