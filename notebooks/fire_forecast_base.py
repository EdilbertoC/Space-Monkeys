# %% [markdown]
# # Space Monkeys — Fire Forecast, base model
#
# Predicts, for every 0.25° cell, the probability of **at least one active fire
# detection** on each of the next **7 days**, like a weather forecast for fire.
#
# Two training regions, same code: set `REGION` in section 1.
#
# | `REGION` | Learns from | Download |
# | --- | --- | --- |
# | `"brazil"` | Brazil only | ~1.3 GB |
# | `"world"` | every country (Brazil included) | ~5 GB |
#
# Both models are **scored on exactly the same test**: Brazil, 2024–2025, the same cells
# and the same truth (detections inside Brazil). Run the notebook once per region; the
# last section compares them.
#
# **Data:** NASA FIRMS VIIRS S-NPP 375 m active fire archive, 2012-01-20 to 2025-12-31
# (yearly files, no key needed). One sensor, so the series is consistent without
# cross-sensor harmonization. Only `type == 0` (presumed vegetation fire) with nominal or
# high confidence is used.
#
# **Model:** a small 2D U-Net. Input = the last 7 days of fire maps, the last 30 days,
# the historical climatology of each target day, seasonality and position, stacked as
# image channels. Output = 7 probability maps (day +1 … +7).
#
# **Honest evaluation:** split by time (train 2012–2022, validation 2023, test 2024–2025)
# and compared with two simple references: *climatology* (what usually happens on this
# day of the year) and *persistence* (recent fire means more fire).
#
# **Runtime:** Runtime → Change runtime type → **T4 GPU**. For `"world"` also enable
# **High-RAM** (Colab Pro): the global daily cube needs ~4 GB plus working memory.
#
# **Outputs** (per region): `fire_forecast_<region>.onnx`, `fire_forecast_<region>.json`,
# `climatology_<region>.npy`, `forecast_mask_<region>.npy`.
#
# *SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team.*
# *Data: NASA FIRMS, VIIRS S-NPP 375 m (doi:10.5067/FIRMS/VIIRS/VNP14IMGT_NRT.002;
# standard processing archive).*

# %%
# Colab already ships numpy, pandas, torch, scikit-learn and matplotlib.
# %pip install -q onnx onnxscript onnxruntime

# %%
import io
import json
import math
import time
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, brier_score_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# %% [markdown]
# ## 1. Configuration

# %%
REGION = "brazil"  # "brazil" or "world"

FIRST_YEAR, LAST_YEAR = 2012, 2025
TRAIN_END = "2022-12-31"  # train: 2012-01-20 … 2022-12-31
VAL_END = "2023-12-31"  # validation: 2023; test: 2024-2025

RES = 0.25  # grid resolution, degrees
# Model grid (west, south, east, north). Both grids share the same 0.25° lattice.
GRIDS = {
    "brazil": (-74.0, -34.0, -34.0, 6.0),
    "world": (-180.0, -60.0, 180.0, 80.0),  # no vegetation fires south of 60°S
}
EVAL_BBOX = GRIDS["brazil"]  # where both models are scored

HISTORY_DAYS = 7  # daily fire maps fed to the model
LONG_WINDOW = 30  # one extra channel: total fires in the last 30 days
LEADS = 7  # forecast days +1 … +7
CLIM_HALF_WINDOW = 7  # climatology is smoothed over ±7 days of the year

# Keep downloads, gridded data and outputs in Google Drive between sessions.
USE_DRIVE = True

# %%
WORK = Path("work")
if USE_DRIVE:
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
        WORK = Path("/content/drive/MyDrive/space-monkeys")
    except Exception as exc:  # e.g. the permission pop-up was closed or blocked
        print(f"Google Drive not mounted ({exc}). Saving to the local session instead;")
        print("download work/outputs from the Files panel before the session ends.")
print("Working directory:", WORK.resolve())
DATA_DIR = WORK / "firms"
CACHE_DIR = WORK / "grids"
OUT_DIR = WORK / "outputs"
for d in (DATA_DIR, CACHE_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

assert REGION in GRIDS, REGION
W, S, E, N = GRIDS[REGION]
NX = int(round((E - W) / RES))
NY = int(round((N - S) / RES))
# The Brazil evaluation window inside the model grid.
EW, ES, EE, EN = EVAL_BBOX
ER0, ER1 = int(round((N - EN) / RES)), int(round((N - ES) / RES))
EC0, EC1 = int(round((EW - W) / RES)), int(round((EE - W) / RES))
EVAL_WIN = (slice(ER0, ER1), slice(EC0, EC1))
EH, EWD = ER1 - ER0, EC1 - EC0
print(f"Model grid: {NY} x {NX} cells. Brazil window: rows {ER0}:{ER1}, cols {EC0}:{EC1}")

# %% [markdown]
# ## 2. Download and grid the FIRMS archive
#
# Each year becomes `counts[day, row, col]` (fire pixels per cell per day, capped at 255)
# and is cached as a compressed file, so later runs skip the download. Brazil-only
# detections are also gridded separately: they are the shared test truth.

# %%
COUNTRY_URL = (
    "https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/{y}/viirs-snpp_{y}_Brazil.csv"
)
WORLD_URL = (
    "https://firms.modaps.eosdis.nasa.gov/data/country/zips/viirs-snpp_{y}_all_countries.zip"
)
USECOLS = ["latitude", "longitude", "acq_date", "confidence", "type"]

DATES = pd.date_range(f"{FIRST_YEAR}-01-01", f"{LAST_YEAR}-12-31", freq="D")
DAY0 = DATES[0]


def download(url: str, path: Path) -> Path:
    if path.exists() and path.stat().st_size > 1000:
        return path
    tmp = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.rename(path)
    return path


def read_detections(source) -> pd.DataFrame:
    df = pd.read_csv(source, usecols=USECOLS)
    return df[(df["type"] == 0) & df["confidence"].isin(["n", "h"])]


def grid_into(target: np.ndarray, df: pd.DataFrame, bbox, year_start: pd.Timestamp) -> int:
    w, _, _, n = bbox
    ny, nx = target.shape[1:]
    col = np.floor((df["longitude"].to_numpy() - w) / RES).astype(np.int64)
    row = np.floor((n - df["latitude"].to_numpy()) / RES).astype(np.int64)
    codes, uniq = pd.factorize(df["acq_date"])
    day = (pd.to_datetime(uniq) - year_start).days.to_numpy()[codes]
    ok = (col >= 0) & (col < nx) & (row >= 0) & (row < ny) & (day >= 0) & (day < len(target))
    np.add.at(target, (day[ok], row[ok], col[ok]), 1)
    return int(ok.sum())


def grid_year(year: int) -> tuple[np.ndarray, np.ndarray]:
    """(model-grid counts, Brazil-only counts) for one year, both uint8."""
    cache = CACHE_DIR / f"{REGION}_{year}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["counts"], z["brazil"]
    start = pd.Timestamp(f"{year}-01-01")
    days = 366 if start.is_leap_year else 365
    counts = np.zeros((days, NY, NX), dtype=np.uint16)
    brazil = np.zeros((days, EH, EWD), dtype=np.uint16)
    if REGION == "brazil":
        path = download(COUNTRY_URL.format(y=year), DATA_DIR / f"viirs-snpp_{year}_Brazil.csv")
        df = read_detections(path)
        n = grid_into(counts, df, GRIDS[REGION], start)
        grid_into(brazil, df, EVAL_BBOX, start)
    else:
        path = download(WORLD_URL.format(y=year), DATA_DIR / f"viirs-snpp_{year}_all.zip")
        n = 0
        with zipfile.ZipFile(path) as zf:
            for member in zf.namelist():
                if not member.endswith(".csv"):
                    continue
                with zf.open(member) as fh:
                    df = read_detections(io.TextIOWrapper(fh, encoding="utf-8"))
                n += grid_into(counts, df, GRIDS[REGION], start)
                if member.endswith("_Brazil.csv"):
                    grid_into(brazil, df, EVAL_BBOX, start)
        path.unlink()  # the gridded cache below replaces the ~370 MB zip
    counts = np.minimum(counts, 255).astype(np.uint8)
    brazil = np.minimum(brazil, 255).astype(np.uint8)
    np.savez_compressed(cache, counts=counts, brazil=brazil)
    print(f"  {year}: {n:,} detections gridded")
    return counts, brazil


counts = np.zeros((len(DATES), NY, NX), dtype=np.uint8)
brazil_counts = np.zeros((len(DATES), EH, EWD), dtype=np.uint8)
for year in range(FIRST_YEAR, LAST_YEAR + 1):
    t0 = time.time()
    c, b = grid_year(year)
    i = (pd.Timestamp(f"{year}-01-01") - DAY0).days
    counts[i : i + len(c)] = c
    brazil_counts[i : i + len(b)] = b
    print(f"{year} ready ({time.time() - t0:.0f}s)")
del c, b

first_day = int(np.argmax(brazil_counts.reshape(len(DATES), -1).any(1)))
train_last = int((pd.Timestamp(TRAIN_END) - DAY0).days)
print("First day with data:", DATES[first_day].date())
print(f"Cube: {counts.nbytes / 1e9:.1f} GB")

# %%
# Forecast mask: cells that burned at least once in the training period.
mask = np.zeros((NY, NX), dtype=bool)
eval_mask = np.zeros((EH, EWD), dtype=bool)  # Brazil cells scored for both models
for d0 in range(first_day, train_last + 1, 365):
    d1 = min(d0 + 365, train_last + 1)
    mask |= (counts[d0:d1] > 0).any(0)
    eval_mask |= (brazil_counts[d0:d1] > 0).any(0)
print(f"Forecast cells: {mask.sum():,}; Brazil test cells: {eval_mask.sum():,}")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
fire_days = np.zeros((NY, NX), dtype=np.int32)
for d0 in range(first_day, train_last + 1, 365):
    fire_days += (counts[d0 : min(d0 + 365, train_last + 1)] > 0).sum(0)
im = ax[0].imshow(np.log1p(fire_days), extent=(W, E, S, N), cmap="inferno")
ax[0].set_title(f"Fire days per cell, 2012–2022 (log) — {REGION}")
plt.colorbar(im, ax=ax[0])
daily = pd.Series(brazil_counts.reshape(len(DATES), -1).sum(1), index=DATES)
daily.groupby(daily.index.dayofyear).mean().plot(ax=ax[1])
ax[1].set_title("Brazil: mean detections per day of year")
ax[1].set_xlabel("Day of year")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Climatology (training years only)
#
# For each cell and day of year: the share of training years with a fire that day,
# smoothed over ±7 days. It is an input channel and the first reference model.

# %%
doy = DATES.dayofyear.to_numpy() - 1  # 0..365
kernel = np.ones(2 * CLIM_HALF_WINDOW + 1, dtype=np.float32) / (2 * CLIM_HALF_WINDOW + 1)


def build_climatology(cube: np.ndarray) -> np.ndarray:
    total = np.zeros((366,) + cube.shape[1:], dtype=np.float32)
    n = np.zeros(366, dtype=np.float32)
    for d in range(first_day, train_last + 1):
        total[doy[d]] += cube[d] > 0
        n[doy[d]] += 1
    raw = total / np.maximum(n, 1)[:, None, None]
    padded = np.concatenate([raw[-CLIM_HALF_WINDOW:], raw, raw[:CLIM_HALF_WINDOW]])
    out = np.zeros_like(raw)
    for i, w in enumerate(kernel):
        out += w * padded[i : i + 366]
    return out


climatology = build_climatology(counts)  # model input
eval_climatology = (
    climatology[:, ER0:ER1, EC0:EC1] if REGION == "brazil" else (build_climatology(brazil_counts))
)  # reference model, identical in both runs
print("Climatology ready:", climatology.shape)

# %% [markdown]
# ## 4. Features and targets
#
# Channels for a forecast issued at the end of day `t`:
#
# | Channels | Content |
# | --- | --- |
# | 7 | log(1 + fire count) for days t-6 … t |
# | 1 | log(1 + total fires in the last 30 days) |
# | 7 | climatology for target days t+1 … t+7 |
# | 2 | sin/cos of day of year |
# | 3 | forecast mask, latitude/90, longitude/180 |
#
# Targets: fire (yes/no) on days t+1 … t+7. Scoring always uses Brazil-only detections.

# %%
N_CHANNELS = HISTORY_DAYS + 1 + LEADS + 2 + 3
lat_plane = np.repeat(((N - RES / 2 - np.arange(NY) * RES) / 90).astype(np.float32)[:, None], NX, 1)
lon_plane = np.repeat(((W + RES / 2 + np.arange(NX) * RES) / 180).astype(np.float32)[None], NY, 0)
mask_plane = mask.astype(np.float32)
FULL = (slice(None), slice(None))


def features(t: int, win: tuple[slice, slice] = FULL) -> np.ndarray:
    """Input channels for a forecast issued at the end of day t (optionally a window)."""
    ys, xs = win
    h, w = mask_plane[ys, xs].shape
    x = np.empty((N_CHANNELS, h, w), dtype=np.float32)
    hist = counts[t - HISTORY_DAYS + 1 : t + 1, ys, xs].astype(np.float32)
    x[:HISTORY_DAYS] = np.log1p(hist)  # float32: log1p on uint8 would give float16
    recent30 = counts[t - LONG_WINDOW + 1 : t + 1, ys, xs].sum(0, dtype=np.float32)
    x[HISTORY_DAYS] = np.log1p(recent30)
    c = HISTORY_DAYS + 1
    for k in range(LEADS):
        x[c + k] = climatology[doy[t + k + 1], ys, xs]
    c += LEADS
    angle = 2 * math.pi * doy[t] / 365.25
    x[c], x[c + 1] = math.sin(angle), math.cos(angle)
    x[c + 2], x[c + 3], x[c + 4] = mask_plane[ys, xs], lat_plane[ys, xs], lon_plane[ys, xs]
    return x


def eval_targets(t: int) -> np.ndarray:
    """Brazil-only truth for days t+1 … t+7 (the shared test truth)."""
    return (brazil_counts[t + 1 : t + 1 + LEADS] > 0).astype(np.float32)


start = first_day + LONG_WINDOW
val_last = int((pd.Timestamp(VAL_END) - DAY0).days)
end = len(DATES) - LEADS - 1
TRAIN_T = np.arange(start, train_last - LEADS + 1)
VAL_T = np.arange(train_last + 1, val_last - LEADS + 1)
TEST_T = np.arange(val_last + 1, end + 1)
print(f"Forecast days — train {len(TRAIN_T)}, val {len(VAL_T)}, test {len(TEST_T)}")

# %% [markdown]
# ## 5. Reference models (Brazil, identical in both runs)
#
# * **Climatology:** probability = climatology of the target day.
# * **Persistence:** probability learned on training data from whether the cell burned
#   in the last 3 days, per lead day.


# %%
def eval_recent(t: int) -> np.ndarray:
    return (brazil_counts[t - 2 : t + 1] > 0).any(0)


persist_p = np.zeros((LEADS, 2), dtype=np.float32)  # [lead, recent fire?]
for k in range(LEADS):
    rs, ys_ = [], []
    for t in TRAIN_T[::5]:
        rs.append(eval_recent(t)[eval_mask])
        ys_.append(brazil_counts[t + k + 1][eval_mask] > 0)
    r, y = np.concatenate(rs), np.concatenate(ys_)
    persist_p[k] = [y[~r].mean(), y[r].mean()]
print("Per lead day: [P(fire | no recent fire), P(fire | recent fire)]\n", persist_p.round(3))


def predict_climatology(t: int) -> np.ndarray:
    return np.stack([eval_climatology[doy[t + k + 1]] for k in range(LEADS)])


def predict_persistence(t: int) -> np.ndarray:
    r = eval_recent(t)
    return np.stack([np.where(r, persist_p[k, 1], persist_p[k, 0]) for k in range(LEADS)])


def evaluate(predict, days: np.ndarray, step: int = 2) -> pd.DataFrame:
    """Average precision and Brier score per lead day, over the Brazil test cells.

    `predict(t)` returns 7 maps over the Brazil window.
    """
    preds = [[] for _ in range(LEADS)]
    truth = [[] for _ in range(LEADS)]
    for t in days[::step]:
        p, y = predict(t), eval_targets(t)
        for k in range(LEADS):
            preds[k].append(p[k][eval_mask])
            truth[k].append(y[k][eval_mask])
    rows = []
    for k in range(LEADS):
        p, y = np.concatenate(preds[k]), np.concatenate(truth[k])
        rows.append(
            {
                "lead_day": k + 1,
                "avg_precision": average_precision_score(y, p),
                "brier": brier_score_loss(y, np.clip(p, 0, 1)),
                "base_rate": y.mean(),
            }
        )
    return pd.DataFrame(rows).set_index("lead_day")


for name, fn in [("climatology", predict_climatology), ("persistence", predict_persistence)]:
    print(f"\n{name} (validation 2023)\n", evaluate(fn, VAL_T).round(4))

# %% [markdown]
# ## 6. U-Net
#
# Plain binary cross-entropy keeps the probabilities calibrated (a 30% forecast should
# come true about 30% of the time). Cells outside the forecast mask are ignored.


# %%
class Block(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, c_in: int, c_out: int, width: int = 32) -> None:
        super().__init__()
        w = [width, width * 2, width * 4, width * 8]
        self.down = nn.ModuleList([Block(c_in, w[0]), Block(w[0], w[1]), Block(w[1], w[2])])
        self.mid = Block(w[2], w[3])
        self.up = nn.ModuleList(
            [nn.ConvTranspose2d(w[i + 1], w[i], 2, stride=2) for i in reversed(range(3))]
        )
        self.dec = nn.ModuleList([Block(w[i] * 2, w[i]) for i in reversed(range(3))])
        self.head = nn.Conv2d(w[0], c_out, 1)

    def forward(self, x):
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.mid(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips), strict=True):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)  # logits


model = UNet(N_CHANNELS, LEADS).to(DEVICE)
# Start from the training base rate so the first predictions are already sensible.
base_rate = float(np.mean([(counts[t + 1][mask] > 0).mean() for t in TRAIN_T[::20]]))
with torch.no_grad():
    model.head.bias.fill_(math.log(base_rate / (1 - base_rate)))
print(f"Training base rate: {base_rate:.4f}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

# %% [markdown]
# ## 7. Training
#
# The daily cube is copied to the GPU once and every batch is built there, so the GPU
# never waits for the CPU. Random 64×64 crops with random flips and rotations; early
# stopping on Brazil validation average precision (the same rule for both regions).

# %%
CROP = 64
BATCH = 128
STEPS_PER_EPOCH = 300
EPOCHS = 20
PATIENCE = 4
torch.backends.cudnn.benchmark = True
gen = torch.Generator(device=DEVICE).manual_seed(42)

counts_g = torch.from_numpy(counts).to(DEVICE)  # uint8
clim_g = torch.from_numpy(climatology).to(DEVICE)
doy_g = torch.from_numpy(doy).to(DEVICE)
static_g = torch.from_numpy(np.stack([mask_plane, lat_plane, lon_plane])).to(DEVICE)
train_t_g = torch.from_numpy(TRAIN_T).to(DEVICE)

# Top-left corners of crops that are at least 20% forecast cells.
_cum = np.pad(mask_plane.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
_cover = (
    _cum[CROP:, CROP:] - _cum[:-CROP, CROP:] - _cum[CROP:, :-CROP] + _cum[:-CROP, :-CROP]
) / CROP**2
corners_g = torch.from_numpy(np.argwhere(_cover >= 0.2)).to(DEVICE)
print(f"Valid crop positions: {len(corners_g):,}")


def gpu_features(t: int, ys: slice, xs: slice) -> torch.Tensor:
    """Same channels as `features()`, for one day and window, built on the GPU."""
    window = counts_g[t - LONG_WINDOW + 1 : t + 1, ys, xs].float()
    hist = torch.log1p(window[-HISTORY_DAYS:])
    recent30 = torch.log1p(window.sum(0, keepdim=True))
    clim = clim_g[doy_g[t + 1 : t + 1 + LEADS], ys, xs]
    h, w = hist.shape[-2:]
    angle = 2 * math.pi * doy[t] / 365.25
    season = torch.tensor([math.sin(angle), math.cos(angle)], device=DEVICE)
    season = season[:, None, None].expand(-1, h, w)
    return torch.cat([hist, recent30, clim, season, static_g[:, ys, xs]], dim=0)


def random_batch():
    ts = train_t_g[torch.randint(len(train_t_g), (BATCH,), device=DEVICE, generator=gen)]
    corners = corners_g[torch.randint(len(corners_g), (BATCH,), device=DEVICE, generator=gen)]
    xs, ys = [], []
    for t, (r, c) in zip(ts.tolist(), corners.tolist(), strict=True):
        win = (slice(r, r + CROP), slice(c, c + CROP))
        xs.append(gpu_features(t, *win))
        ys.append(counts_g[t + 1 : t + 1 + LEADS, win[0], win[1]] > 0)
    x, y = torch.stack(xs), torch.stack(ys).float()
    m = x[:, N_CHANNELS - 3]  # the mask channel
    k = int(torch.randint(4, (1,), generator=gen, device=DEVICE))
    x, y, m = (torch.rot90(a, k, dims=(-2, -1)) for a in (x, y, m))
    if torch.rand(1, generator=gen, device=DEVICE).item() < 0.5:
        x, y, m = (torch.flip(a, dims=(-1,)) for a in (x, y, m))
    return x.contiguous(), y.contiguous(), m.contiguous()


# The model sees a margin around Brazil when predicting it (context across borders).
MARGIN = 16
PR0, PR1 = max(ER0 - MARGIN, 0), min(ER1 + MARGIN, NY)
PC0, PC1 = max(EC0 - MARGIN, 0), min(EC1 + MARGIN, NX)
PR1 = PR0 + (PR1 - PR0) // 8 * 8  # the U-Net needs sizes divisible by 8
PC1 = PC0 + (PC1 - PC0) // 8 * 8
assert PR1 >= ER1 and PC1 >= EC1


@torch.no_grad()
def predict_unet(t: int) -> np.ndarray:
    """7 probability maps over the Brazil window."""
    model.eval()
    x = gpu_features(t, slice(PR0, PR1), slice(PC0, PC1))[None]
    with torch.autocast(device_type="cuda", enabled=DEVICE == "cuda"):
        logits = model(x)
    p = torch.sigmoid(logits.float())[0]
    return p[:, ER0 - PR0 : ER1 - PR0, EC0 - PC0 : EC1 - PC0].cpu().numpy()


# The GPU features must match the NumPy ones used later by the backend.
_t = int(VAL_T[0])
_gpu = gpu_features(_t, *EVAL_WIN).cpu().numpy()
assert np.abs(_gpu - features(_t, EVAL_WIN)).max() < 1e-4
print("GPU and NumPy features match")

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=2e-3, total_steps=EPOCHS * STEPS_PER_EPOCH
)
use_amp = DEVICE == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
best_ap, best_state, bad_epochs, history = -1.0, None, 0, []

for epoch in range(1, EPOCHS + 1):
    model.train()
    t0, total = time.time(), 0.0
    for _ in range(STEPS_PER_EPOCH):
        x, y, m = random_batch()
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none")
        loss = (loss * m[:, None]).sum() / (m.sum() * LEADS)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += loss.item()
    val = evaluate(predict_unet, VAL_T, step=4)
    ap = val["avg_precision"].mean()
    history.append({"epoch": epoch, "loss": total / STEPS_PER_EPOCH, "val_ap": ap})
    print(
        f"epoch {epoch:2d}  loss {total / STEPS_PER_EPOCH:.4f}  val AP {ap:.4f}  "
        f"({time.time() - t0:.0f}s)"
    )
    if ap > best_ap:
        best_ap, bad_epochs = ap, 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print("Early stopping")
            break

model.load_state_dict(best_state)
torch.save(best_state, OUT_DIR / f"fire_forecast_{REGION}.pt")

# %% [markdown]
# ## 8. Test results — Brazil, 2024–2025 (never seen in training)

# %%
results = {
    "climatology": evaluate(predict_climatology, TEST_T),
    "persistence": evaluate(predict_persistence, TEST_T),
    f"unet_{REGION}": evaluate(predict_unet, TEST_T),
}
summary = pd.concat({name: t["avg_precision"] for name, t in results.items()}, axis=1)
brier = pd.concat({name: t["brier"] for name, t in results.items()}, axis=1)
print("Average precision (higher is better)\n", summary.round(4))
print("\nBrier score (lower is better)\n", brier.round(4))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
summary.plot(marker="o", ax=ax[0], title="Average precision by lead day")
brier.plot(marker="o", ax=ax[1], title="Brier score by lead day")
for a in ax:
    a.set_xlabel("Lead day")
plt.tight_layout()
plt.show()

(OUT_DIR / f"test_results_{REGION}.json").write_text(
    json.dumps(
        {
            "region": REGION,
            "epochs_trained": len(history),
            "history": history,
            "avg_precision": summary.round(5).to_dict(),
            "brier": brier.round(6).to_dict(),
        },
        indent=2,
    )
)

# %%
# Reliability: do 30% forecasts come true ~30% of the time? (lead day 1, test set)
p_all, y_all = [], []
for t in TEST_T[::3]:
    p_all.append(predict_unet(t)[0][eval_mask])
    y_all.append(eval_targets(t)[0][eval_mask])
p_all, y_all = np.concatenate(p_all), np.concatenate(y_all)
bins = np.linspace(0, 1, 11)
idx = np.clip(np.digitize(p_all, bins) - 1, 0, 9)
observed = [y_all[idx == i].mean() if (idx == i).any() else np.nan for i in range(10)]
plt.plot([0, 1], [0, 1], "k--", label="perfect")
plt.plot((bins[:-1] + bins[1:]) / 2, observed, "o-", label=f"U-Net ({REGION}), day +1")
plt.xlabel("Forecast probability")
plt.ylabel("Observed frequency")
plt.legend()
plt.title("Reliability, Brazil test 2024–2025")
plt.show()

# %%
# One example forecast during the 2024 dry season.
t = int((pd.Timestamp("2024-08-20") - DAY0).days)
pred = predict_unet(t)
extent = (EW, EE, ES, EN)
fig, ax = plt.subplots(2, 4, figsize=(16, 7))
ax[0, 0].imshow(np.log1p(brazil_counts[t - 6 : t + 1].sum(0)), cmap="inferno", extent=extent)
ax[0, 0].set_title(f"Last 7 days of fires\n(up to {DATES[t].date()})")
for i, k in enumerate([0, 2, 6]):
    shown = np.where(eval_mask, pred[k], np.nan)
    ax[0, i + 1].imshow(shown, cmap="YlOrRd", vmin=0, vmax=1, extent=extent)
    ax[0, i + 1].set_title(f"Forecast day +{k + 1} ({REGION} model)")
    ax[1, i + 1].imshow(brazil_counts[t + k + 1] > 0, cmap="gray_r", extent=extent)
    ax[1, i + 1].set_title(f"Observed {DATES[t + k + 1].date()}")
ax[1, 0].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Export for the backend
#
# The network is fully convolutional, so the ONNX file accepts any window whose height
# and width are multiples of 8.

# %%
model.eval().cpu()
dummy = torch.from_numpy(features(int(TEST_T[0]), EVAL_WIN))[None]
onnx_path = OUT_DIR / f"fire_forecast_{REGION}.onnx"
torch.onnx.export(
    model,
    dummy,
    onnx_path,
    input_names=["features"],
    output_names=["logits"],
    dynamic_axes={"features": {2: "height", 3: "width"}, "logits": {2: "height", 3: "width"}},
    opset_version=17,
    dynamo=False,
)
np.save(OUT_DIR / f"climatology_{REGION}.npy", climatology.astype(np.float16))
np.save(OUT_DIR / f"forecast_mask_{REGION}.npy", mask)

meta = {
    "model": f"fire_forecast_{REGION}",
    "data": "NASA FIRMS VIIRS S-NPP 375 m, type 0, confidence n/h, cell counts capped at 255",
    "region": REGION,
    "train_period": [str(DATES[TRAIN_T[0]].date()), TRAIN_END],
    "bbox": GRIDS[REGION],
    "resolution_deg": RES,
    "grid": [NY, NX],
    "history_days": HISTORY_DAYS,
    "long_window_days": LONG_WINDOW,
    "leads": LEADS,
    "channels": [f"log1p_count_t-{HISTORY_DAYS - 1 - i}" for i in range(HISTORY_DAYS)]
    + ["log1p_sum_30d"]
    + [f"climatology_t+{k + 1}" for k in range(LEADS)]
    + ["doy_sin", "doy_cos", "mask", "lat_over_90", "lon_over_180"],
    "test_region": "Brazil",
    "test_avg_precision": summary.mean().round(4).to_dict(),
    "test_brier": brier.mean().round(5).to_dict(),
}
(OUT_DIR / f"fire_forecast_{REGION}.json").write_text(json.dumps(meta, indent=2))

import onnxruntime as ort  # noqa: E402

session = ort.InferenceSession(str(onnx_path))
onnx_out = session.run(None, {"features": dummy.numpy()})[0]
torch_out = model(dummy).detach().numpy()
print("ONNX matches PyTorch:", np.allclose(onnx_out, torch_out, atol=1e-4))
print("Saved to", OUT_DIR.resolve())
model.to(DEVICE)

# %% [markdown]
# ## 10. Compare the Brazil and world models
#
# Reads the saved test results of both runs (same Brazil test cells, days and truth).

# %%
saved = {}
for region in ("brazil", "world"):
    path = OUT_DIR / f"test_results_{region}.json"
    if path.exists():
        saved[region] = json.loads(path.read_text())

if len(saved) < 2:
    missing = {"brazil", "world"} - set(saved)
    print(f"Run the notebook with REGION = {missing.pop()!r} to compare both models.")
else:
    ap = pd.DataFrame(
        {
            "climatology": saved["brazil"]["avg_precision"]["climatology"],
            "persistence": saved["brazil"]["avg_precision"]["persistence"],
            "unet_brazil": saved["brazil"]["avg_precision"]["unet_brazil"],
            "unet_world": saved["world"]["avg_precision"]["unet_world"],
        }
    )
    br = pd.DataFrame(
        {
            "climatology": saved["brazil"]["brier"]["climatology"],
            "persistence": saved["brazil"]["brier"]["persistence"],
            "unet_brazil": saved["brazil"]["brier"]["unet_brazil"],
            "unet_world": saved["world"]["brier"]["unet_world"],
        }
    )
    ap.index.name = br.index.name = "lead_day"
    print("Average precision, Brazil test 2024–2025 (higher is better)\n", ap.round(4))
    print("\nBrier score (lower is better)\n", br.round(5))
    print("\nMean over lead days:")
    print(pd.DataFrame({"avg_precision": ap.mean(), "brier": br.mean()}).round(5))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ap.plot(marker="o", ax=axes[0], title="Average precision by lead day")
    br.plot(marker="o", ax=axes[1], title="Brier score by lead day")
    plt.tight_layout()
    plt.show()
