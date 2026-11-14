# %% [markdown]
# # Space Monkeys — Fire Forecast, base model
#
# Predicts, for every 0.25° cell, the probability of **at least one active fire
# detection** on each of the next **7 days**, like a weather forecast for fire.
#
# Choose the training data in section 1, same code for every run:
#
# | Setting | Options |
# | --- | --- |
# | `REGION` | `"brazil"` (Brazil only) or `"world"` (every country, Brazil included) |
# | `USE_MODIS` | `False`: VIIRS 2012–2025. `True`: adds MODIS 2000–2016 as extra history |
#
# Every model is **scored on exactly the same test**: Brazil, 2024–2025, the same cells
# and the same truth (VIIRS detections inside Brazil). Run the notebook once per setting;
# the last section compares every run found in the outputs folder.
#
# **Data:** NASA FIRMS active fire archives (yearly files, no key needed):
# VIIRS S-NPP 375 m (2012-01-20 → 2025-12-31) and, optionally, MODIS Terra/Aqua 1 km
# Collection 6.1 (2000-11-01 → 2016-12-31). Presumed vegetation fires only (`type == 0`),
# nominal/high confidence.
#
# **Sensor harmonization with MODIS:** VIIRS sees far more fires than MODIS, so their
# maps are never mixed. Each sensor has its own input channels plus an "available"
# flag, and the network has two output heads (fires as VIIRS would see them, and as
# MODIS would). 2000–2011 trains the MODIS head, 2017–2025 the VIIRS head, and the
# 2012–2016 overlap teaches the network how the two relate. During the overlap MODIS
# inputs are randomly hidden, so the model also works with VIIRS alone, which is how it
# runs from 2017 on and in production (Terra and Aqua are near the end of their life).
#
# **Model:** a small 2D U-Net. Inputs per sensor: the last 7 days of fire maps and the
# last 30 days; plus the climatology of each target day, seasonality and position.
#
# **Honest evaluation:** split by time (train ≤ 2022, validation 2023, test 2024–2025),
# compared with *climatology* (what usually happens on this day of the year) and
# *persistence* (recent fire means more fire).
#
# **Runtime:** Runtime → Change runtime type → **T4 GPU** (Brazil). For `"world"` use
# **High-RAM**; for `"world"` with MODIS prefer an **L4** GPU (~10 GB of data on the GPU).
#
# **Outputs** (per run tag, e.g. `brazil_modis`): `fire_forecast_<tag>.onnx`,
# `fire_forecast_<tag>.json`, `climatology_<tag>.npy`, `forecast_mask_<tag>.npy`.
#
# *SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team.*
# *Data: NASA FIRMS — VIIRS S-NPP 375 m (doi:10.5067/FIRMS/VIIRS/VNP14IMGT_NRT.002) and
# MODIS C6.1 (doi:10.5067/FIRMS/MODIS/MCD14DL.NRT.0061); standard processing archives.*

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
USE_MODIS = False  # True: add MODIS 2000–2016 as extra history

TRAIN_END = "2022-12-31"
VAL_END = "2023-12-31"  # validation: 2023; test: 2024–2025
LAST_DAY = "2025-12-31"

# Yearly archives per sensor: first year, last year, minimum confidence kept.
SENSORS = {
    "viirs": {"first": 2012, "last": 2025},
    "modis": {"first": 2000, "last": 2016},
}
ACTIVE = ["viirs", "modis"] if USE_MODIS else ["viirs"]  # "viirs" is always head 0
MODIS_DROPOUT = 0.5  # share of overlap samples trained with MODIS hidden

RES = 0.25  # grid resolution, degrees
GRIDS = {  # model grid (west, south, east, north), same 0.25° lattice for both
    "brazil": (-74.0, -34.0, -34.0, 6.0),
    "world": (-180.0, -60.0, 180.0, 80.0),  # no vegetation fires south of 60°S
}
EVAL_BBOX = GRIDS["brazil"]  # where every model is scored

HISTORY_DAYS = 7  # daily fire maps per sensor
LONG_WINDOW = 30  # one extra channel per sensor: total fires in the last 30 days
LEADS = 7  # forecast days +1 … +7
CLIM_HALF_WINDOW = 7  # climatology is smoothed over ±7 days of the year

USE_DRIVE = True  # keep downloads, grids and outputs in Google Drive

TAG = REGION + ("_modis" if USE_MODIS else "")
print("Run tag:", TAG)

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
EW, ES, EE, EN = EVAL_BBOX
ER0, ER1 = int(round((N - EN) / RES)), int(round((N - ES) / RES))
EC0, EC1 = int(round((EW - W) / RES)), int(round((EE - W) / RES))
EVAL_WIN = (slice(ER0, ER1), slice(EC0, EC1))
EH, EWD = ER1 - ER0, EC1 - EC0
print(f"Model grid: {NY} x {NX} cells. Brazil window: rows {ER0}:{ER1}, cols {EC0}:{EC1}")

# %% [markdown]
# ## 2. Download and grid the FIRMS archives
#
# Each sensor-year becomes `counts[day, row, col]` (fire pixels per cell per day, capped
# at 255), cached as a compressed file so later runs skip the download. Brazil-only VIIRS
# detections are gridded separately: they are the shared test truth.

# %%
BASE = "https://firms.modaps.eosdis.nasa.gov/data/country"
URLS = {
    "brazil": BASE + "/{s}/{y}/{s}_{y}_Brazil.csv",
    "world": BASE + "/zips/{s}_{y}_all_countries.zip",
}
FIRMS_NAME = {"viirs": "viirs-snpp", "modis": "modis"}
USECOLS = ["latitude", "longitude", "acq_date", "confidence", "type"]

YEAR0 = min(SENSORS[s]["first"] for s in ACTIVE)
DATES = pd.date_range(f"{YEAR0}-01-01", LAST_DAY, freq="D")
DAY0 = DATES[0]


def day_index(date: str | pd.Timestamp) -> int:
    return int((pd.Timestamp(date) - DAY0).days)


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


def read_detections(source, sensor: str) -> pd.DataFrame:
    df = pd.read_csv(source, usecols=USECOLS, dtype={"confidence": str})
    df = df[df["type"] == 0]
    if sensor == "viirs":
        return df[df["confidence"].isin(["n", "h"])]
    return df[pd.to_numeric(df["confidence"], errors="coerce") >= 30]  # MODIS nominal+


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


def grid_year(sensor: str, year: int) -> tuple[np.ndarray, np.ndarray]:
    """(model-grid counts, Brazil-only counts) for one sensor-year, both uint8."""
    name = f"{REGION}_{year}.npz" if sensor == "viirs" else f"{REGION}_{sensor}_{year}.npz"
    cache = CACHE_DIR / name
    if cache.exists():
        z = np.load(cache)
        return z["counts"], z["brazil"]
    start = pd.Timestamp(f"{year}-01-01")
    days = 366 if start.is_leap_year else 365
    counts = np.zeros((days, NY, NX), dtype=np.uint16)
    brazil = np.zeros((days, EH, EWD), dtype=np.uint16)
    url = URLS[REGION].format(s=FIRMS_NAME[sensor], y=year)
    path = download(url, DATA_DIR / url.rsplit("/", 1)[1])
    n = 0
    if REGION == "brazil":
        df = read_detections(path, sensor)
        n = grid_into(counts, df, GRIDS[REGION], start)
        grid_into(brazil, df, EVAL_BBOX, start)
    else:
        with zipfile.ZipFile(path) as zf:
            for member in zf.namelist():
                if not member.endswith(".csv"):
                    continue
                with zf.open(member) as fh:
                    df = read_detections(io.TextIOWrapper(fh, encoding="utf-8"), sensor)
                n += grid_into(counts, df, GRIDS[REGION], start)
                if member.endswith("_Brazil.csv"):
                    grid_into(brazil, df, EVAL_BBOX, start)
        path.unlink()  # the gridded cache below replaces the zip
    counts = np.minimum(counts, 255).astype(np.uint8)
    brazil = np.minimum(brazil, 255).astype(np.uint8)
    np.savez_compressed(cache, counts=counts, brazil=brazil)
    print(f"  {sensor} {year}: {n:,} detections gridded")
    return counts, brazil


# One cube per sensor, each starting on Jan 1 of its first year.
cubes, offsets, avail = {}, {}, {}
for s in ACTIVE:
    first, last = SENSORS[s]["first"], SENSORS[s]["last"]
    offsets[s] = day_index(f"{first}-01-01")
    n_days = day_index(f"{last}-12-31") - offsets[s] + 1
    cubes[s] = np.zeros((n_days, NY, NX), dtype=np.uint8)
    if s == "viirs":
        brazil_counts = np.zeros((n_days, EH, EWD), dtype=np.uint8)
    for year in range(first, last + 1):
        t0 = time.time()
        c, b = grid_year(s, year)
        i = day_index(f"{year}-01-01") - offsets[s]
        cubes[s][i : i + len(c)] = c
        if s == "viirs":
            brazil_counts[i : i + len(b)] = b
        print(f"{s} {year} ready ({time.time() - t0:.0f}s)")
    has_data = cubes[s].reshape(n_days, -1).any(1)
    first_data = offsets[s] + int(np.argmax(has_data))
    last_data = offsets[s] + n_days - 1
    avail[s] = (first_data, last_data)
    print(
        f"{s}: {DATES[first_data].date()} to {DATES[last_data].date()}, "
        f"{cubes[s].nbytes / 1e9:.1f} GB"
    )

V_OFF = offsets["viirs"]
train_last = day_index(TRAIN_END)
val_last = day_index(VAL_END)


# %%
def sensor_days(s: str, t0: int, t1: int, win=(slice(None), slice(None))) -> np.ndarray:
    """Days t0..t1-1 of a sensor cube over a window, zeros outside its archive."""
    ys, xs = win
    h, w = cubes[s][0, ys, xs].shape
    out = np.zeros((t1 - t0, h, w), dtype=np.uint8)
    a, b = t0 - offsets[s], t1 - offsets[s]
    lo, hi = max(a, 0), min(b, len(cubes[s]))
    if hi > lo:
        out[lo - a : hi - a] = cubes[s][lo:hi, ys, xs]
    return out


def input_available(s: str, t: int) -> bool:
    """A sensor feeds the model on day t only with a full 30-day history."""
    first, last = avail[s]
    return first + LONG_WINDOW - 1 <= t <= last


def target_available(s: str, day: int) -> bool:
    first, last = avail[s]
    return first <= day <= last


# Forecast mask: cells that burned at least once in the training period (any sensor).
mask = np.zeros((NY, NX), dtype=bool)
eval_mask = np.zeros((EH, EWD), dtype=bool)  # Brazil cells scored for every model
fire_days = np.zeros((NY, NX), dtype=np.int32)
for s in ACTIVE:
    lo, hi = avail[s][0], min(avail[s][1], train_last)
    for d0 in range(lo, hi + 1, 365):
        chunk = sensor_days(s, d0, min(d0 + 365, hi + 1)) > 0
        mask |= chunk.any(0)
        if s == "viirs":
            fire_days += chunk.sum(0)
for d0 in range(avail["viirs"][0], train_last + 1, 365):
    b = brazil_counts[d0 - V_OFF : min(d0 + 365, train_last + 1) - V_OFF]
    eval_mask |= (b > 0).any(0)
print(f"Forecast cells: {mask.sum():,}; Brazil test cells: {eval_mask.sum():,}")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
im = ax[0].imshow(np.log1p(fire_days), extent=(W, E, S, N), cmap="inferno")
ax[0].set_title(f"VIIRS fire days per cell, 2012–2022 (log) — {REGION}")
plt.colorbar(im, ax=ax[0])
for s in ACTIVE:
    lo, hi = avail[s]
    daily = pd.Series(
        cubes[s][lo - offsets[s] : hi - offsets[s] + 1].reshape(hi - lo + 1, -1).sum(1),
        index=DATES[lo : hi + 1],
    )
    daily.resample("YS").sum().plot(ax=ax[1], marker="o", label=s.upper())
ax[1].set_title(f"Detections per year — {REGION} (VIIRS sees far more fires)")
ax[1].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Climatology (VIIRS training years only)
#
# For each cell and day of year: the share of training years with a VIIRS fire that day,
# smoothed over ±7 days. It is an input channel and the first reference model.

# %%
doy = DATES.dayofyear.to_numpy() - 1  # 0..365
kernel = np.ones(2 * CLIM_HALF_WINDOW + 1, dtype=np.float32) / (2 * CLIM_HALF_WINDOW + 1)


def build_climatology(cube: np.ndarray, offset: int) -> np.ndarray:
    total = np.zeros((366,) + cube.shape[1:], dtype=np.float32)
    n = np.zeros(366, dtype=np.float32)
    for d in range(avail["viirs"][0], train_last + 1):
        total[doy[d]] += cube[d - offset] > 0
        n[doy[d]] += 1
    raw = total / np.maximum(n, 1)[:, None, None]
    padded = np.concatenate([raw[-CLIM_HALF_WINDOW:], raw, raw[:CLIM_HALF_WINDOW]])
    out = np.zeros_like(raw)
    for i, w in enumerate(kernel):
        out += w * padded[i : i + 366]
    return out


climatology = build_climatology(cubes["viirs"], V_OFF)  # model input
if REGION == "brazil":
    eval_climatology = climatology[:, ER0:ER1, EC0:EC1]
else:
    eval_climatology = build_climatology(brazil_counts, V_OFF)  # same in every run
print("Climatology ready:", climatology.shape)

# %% [markdown]
# ## 4. Features and targets
#
# Channels for a forecast issued at the end of day `t`:
#
# | Channels | Content |
# | --- | --- |
# | 9 per sensor | log(1 + count) for days t-6 … t, log(1 + total of last 30 days), available flag |
# | 7 | VIIRS climatology for target days t+1 … t+7 |
# | 2 | sin/cos of day of year |
# | 3 | forecast mask, latitude/90, longitude/180 |
#
# Outputs: 7 maps per sensor (VIIRS first). Scoring always uses the VIIRS head and
# Brazil-only VIIRS detections.

# %%
PER_SENSOR = HISTORY_DAYS + 2
N_CHANNELS = PER_SENSOR * len(ACTIVE) + LEADS + 2 + 3
N_OUT = LEADS * len(ACTIVE)
MASK_CH = N_CHANNELS - 3
lat_plane = np.repeat(((N - RES / 2 - np.arange(NY) * RES) / 90).astype(np.float32)[:, None], NX, 1)
lon_plane = np.repeat(((W + RES / 2 + np.arange(NX) * RES) / 180).astype(np.float32)[None], NY, 0)
mask_plane = mask.astype(np.float32)
FULL = (slice(None), slice(None))


def features(t: int, win: tuple[slice, slice] = FULL, hide: tuple[str, ...] = ()) -> np.ndarray:
    """Input channels for a forecast issued at the end of day t (optionally a window)."""
    ys, xs = win
    h, w = mask_plane[ys, xs].shape
    x = np.zeros((N_CHANNELS, h, w), dtype=np.float32)
    for i, s in enumerate(ACTIVE):
        c = i * PER_SENSOR
        if s in hide or not input_available(s, t):
            continue  # channels and flag stay 0
        window = sensor_days(s, t - LONG_WINDOW + 1, t + 1, win).astype(np.float32)
        x[c : c + HISTORY_DAYS] = np.log1p(window[-HISTORY_DAYS:])
        x[c + HISTORY_DAYS] = np.log1p(window.sum(0))
        x[c + HISTORY_DAYS + 1] = 1.0
    c = PER_SENSOR * len(ACTIVE)
    for k in range(LEADS):
        x[c + k] = climatology[doy[t + k + 1], ys, xs]
    c += LEADS
    angle = 2 * math.pi * doy[t] / 365.25
    x[c], x[c + 1] = math.sin(angle), math.cos(angle)
    x[c + 2], x[c + 3], x[c + 4] = mask_plane[ys, xs], lat_plane[ys, xs], lon_plane[ys, xs]
    return x


def eval_targets(t: int) -> np.ndarray:
    """Brazil-only VIIRS truth for days t+1 … t+7 (the shared test truth)."""
    return (brazil_counts[t + 1 - V_OFF : t + 1 + LEADS - V_OFF] > 0).astype(np.float32)


first_t = min(avail[s][0] for s in ACTIVE) + LONG_WINDOW - 1
TRAIN_T = np.arange(first_t, train_last - LEADS + 1)
VAL_T = np.arange(train_last + 1, val_last - LEADS + 1)
TEST_T = np.arange(val_last + 1, len(DATES) - LEADS)
print(
    f"Forecast days — train {len(TRAIN_T)} (from {DATES[first_t].date()}), "
    f"val {len(VAL_T)}, test {len(TEST_T)}"
)

# %% [markdown]
# ## 5. Reference models (Brazil, identical in every run)
#
# * **Climatology:** probability = climatology of the target day.
# * **Persistence:** probability learned on VIIRS training data from whether the cell
#   burned in the last 3 days, per lead day.


# %%
def eval_recent(t: int) -> np.ndarray:
    return (brazil_counts[t - 2 - V_OFF : t + 1 - V_OFF] > 0).any(0)


viirs_start = avail["viirs"][0] + LONG_WINDOW
viirs_train_t = np.arange(max(first_t, viirs_start), train_last - LEADS + 1)
persist_p = np.zeros((LEADS, 2), dtype=np.float32)  # [lead, recent fire?]
for k in range(LEADS):
    rs, ys_ = [], []
    for t in viirs_train_t[::5]:
        rs.append(eval_recent(t)[eval_mask])
        ys_.append(brazil_counts[t + k + 1 - V_OFF][eval_mask] > 0)
    r, y = np.concatenate(rs), np.concatenate(ys_)
    persist_p[k] = [y[~r].mean(), y[r].mean()]
print("Per lead day: [P(fire | no recent fire), P(fire | recent fire)]\n", persist_p.round(3))


def predict_climatology(t: int) -> np.ndarray:
    return np.stack([eval_climatology[doy[t + k + 1]] for k in range(LEADS)])


def predict_persistence(t: int) -> np.ndarray:
    r = eval_recent(t)
    return np.stack([np.where(r, persist_p[k, 1], persist_p[k, 0]) for k in range(LEADS)])


def evaluate(predict, days: np.ndarray, step: int = 2) -> pd.DataFrame:
    """Average precision and Brier score per lead day, over the Brazil test cells."""
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
# come true about 30% of the time). Cells outside the forecast mask, and target days a
# sensor did not observe, are ignored.


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
        return self.head(x)  # logits: LEADS maps per sensor, VIIRS first


model = UNet(N_CHANNELS, N_OUT).to(DEVICE)
# Start each head from its sensor's training base rate.
with torch.no_grad():
    for i, s in enumerate(ACTIVE):
        lo, hi = avail[s][0], min(avail[s][1], train_last)
        days = np.arange(lo, hi + 1, 20)
        rate = float(np.mean([(cubes[s][d - offsets[s]][mask] > 0).mean() for d in days]))
        model.head.bias[i * LEADS : (i + 1) * LEADS] = math.log(rate / (1 - rate))
        print(f"{s} training base rate: {rate:.4f}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

# %% [markdown]
# ## 7. Training
#
# The cubes are copied to the GPU once and every batch is built there, so the GPU never
# waits for the CPU. Random 64×64 crops with random flips and rotations; early stopping
# on Brazil validation average precision of the VIIRS head (same rule for every run).

# %%
CROP = 64
BATCH = 128
STEPS_PER_EPOCH = 300
EPOCHS = 20
PATIENCE = 4
torch.backends.cudnn.benchmark = True
rng = np.random.default_rng(42)

cubes_g = {s: torch.from_numpy(cubes[s]).to(DEVICE) for s in ACTIVE}  # uint8
clim_g = torch.from_numpy(climatology).to(DEVICE)
doy_g = torch.from_numpy(doy).to(DEVICE)
static_g = torch.from_numpy(np.stack([mask_plane, lat_plane, lon_plane])).to(DEVICE)

# Top-left corners of crops that are at least 20% forecast cells.
_cum = np.pad(mask_plane.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
_cover = (
    _cum[CROP:, CROP:] - _cum[:-CROP, CROP:] - _cum[CROP:, :-CROP] + _cum[:-CROP, :-CROP]
) / CROP**2
corners = np.argwhere(_cover >= 0.2)
print(f"Valid crop positions: {len(corners):,}")


def gpu_days(s: str, t0: int, t1: int, ys: slice, xs: slice) -> torch.Tensor:
    """GPU twin of `sensor_days`."""
    h = len(range(*ys.indices(NY)))
    w = len(range(*xs.indices(NX)))
    out = torch.zeros((t1 - t0, h, w), dtype=torch.uint8, device=DEVICE)
    a, b = t0 - offsets[s], t1 - offsets[s]
    lo, hi = max(a, 0), min(b, len(cubes[s]))
    if hi > lo:
        out[lo - a : hi - a] = cubes_g[s][lo:hi, ys, xs]
    return out


def gpu_features(t: int, ys: slice, xs: slice, hide: tuple[str, ...] = ()) -> torch.Tensor:
    """Same channels as `features()`, for one day and window, built on the GPU."""
    h = len(range(*ys.indices(NY)))
    w = len(range(*xs.indices(NX)))
    parts = []
    for s in ACTIVE:
        if s in hide or not input_available(s, t):
            parts.append(torch.zeros((PER_SENSOR, h, w), device=DEVICE))
            continue
        window = gpu_days(s, t - LONG_WINDOW + 1, t + 1, ys, xs).float()
        parts += [
            torch.log1p(window[-HISTORY_DAYS:]),
            torch.log1p(window.sum(0, keepdim=True)),
            torch.ones((1, h, w), device=DEVICE),
        ]
    parts.append(clim_g[doy_g[t + 1 : t + 1 + LEADS], ys, xs])
    angle = 2 * math.pi * doy[t] / 365.25
    season = torch.tensor([math.sin(angle), math.cos(angle)], device=DEVICE)
    parts.append(season[:, None, None].expand(-1, h, w))
    parts.append(static_g[:, ys, xs])
    return torch.cat(parts, dim=0)


def gpu_targets(t: int, ys: slice, xs: slice) -> tuple[torch.Tensor, torch.Tensor]:
    """Targets for every head and a per-channel weight (0 where a sensor saw nothing)."""
    ys_, ws = [], []
    for s in ACTIVE:
        ys_.append(gpu_days(s, t + 1, t + 1 + LEADS, ys, xs) > 0)
        ok = [target_available(s, t + 1 + k) for k in range(LEADS)]
        ws.append(torch.tensor(ok, dtype=torch.float32, device=DEVICE))
    return torch.cat(ys_).float(), torch.cat(ws)


def random_batch():
    ts = rng.choice(TRAIN_T, BATCH)
    picks = corners[rng.integers(len(corners), size=BATCH)]
    xs, ys, ws = [], [], []
    for t, (r, c) in zip(ts.tolist(), picks.tolist(), strict=True):
        win = (slice(r, r + CROP), slice(c, c + CROP))
        hide = ()
        if USE_MODIS and input_available("viirs", t) and rng.random() < MODIS_DROPOUT:
            hide = ("modis",)  # learn to work without MODIS during the overlap
        xs.append(gpu_features(t, *win, hide=hide))
        y, w = gpu_targets(t, *win)
        ys.append(y)
        ws.append(w)
    x, y, w = torch.stack(xs), torch.stack(ys), torch.stack(ws)
    m = x[:, MASK_CH]
    k = int(rng.integers(4))
    x, y, m = (torch.rot90(a, k, dims=(-2, -1)) for a in (x, y, m))
    if rng.random() < 0.5:
        x, y, m = (torch.flip(a, dims=(-1,)) for a in (x, y, m))
    return x.contiguous(), y.contiguous(), m.contiguous(), w


# The model sees a margin around Brazil when predicting it (context across borders).
MARGIN = 16
PR0, PR1 = max(ER0 - MARGIN, 0), min(ER1 + MARGIN, NY)
PC0, PC1 = max(EC0 - MARGIN, 0), min(EC1 + MARGIN, NX)
PR1 = PR0 + (PR1 - PR0) // 8 * 8  # the U-Net needs sizes divisible by 8
PC1 = PC0 + (PC1 - PC0) // 8 * 8
assert PR1 >= ER1 and PC1 >= EC1


@torch.no_grad()
def predict_unet(t: int) -> np.ndarray:
    """VIIRS-head probability maps (7) over the Brazil window."""
    model.eval()
    x = gpu_features(t, slice(PR0, PR1), slice(PC0, PC1))[None]
    with torch.autocast(device_type="cuda", enabled=DEVICE == "cuda"):
        logits = model(x)
    p = torch.sigmoid(logits[:, :LEADS].float())[0]
    return p[:, ER0 - PR0 : ER1 - PR0, EC0 - PC0 : EC1 - PC0].cpu().numpy()


# The GPU features must match the NumPy ones used later by the backend.
for _t in [int(VAL_T[0])] + ([int(TRAIN_T[0])] if USE_MODIS else []):
    _gpu = gpu_features(_t, *EVAL_WIN).cpu().numpy()
    assert np.abs(_gpu - features(_t, EVAL_WIN)).max() < 1e-4, _t
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
        x, y, m, w = random_batch()
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none")
        weight = m[:, None] * w[:, :, None, None]
        loss = (loss * weight).sum() / weight.sum().clamp(min=1.0)
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
torch.save(best_state, OUT_DIR / f"fire_forecast_{TAG}.pt")

# %% [markdown]
# ## 8. Test results — Brazil, 2024–2025 (never seen in training)

# %%
results = {
    "climatology": evaluate(predict_climatology, TEST_T),
    "persistence": evaluate(predict_persistence, TEST_T),
    f"unet_{TAG}": evaluate(predict_unet, TEST_T),
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

(OUT_DIR / f"test_results_{TAG}.json").write_text(
    json.dumps(
        {
            "tag": TAG,
            "region": REGION,
            "sensors": ACTIVE,
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
plt.plot((bins[:-1] + bins[1:]) / 2, observed, "o-", label=f"U-Net ({TAG}), day +1")
plt.xlabel("Forecast probability")
plt.ylabel("Observed frequency")
plt.legend()
plt.title("Reliability, Brazil test 2024–2025")
plt.show()

# %%
# One example forecast during the 2024 dry season.
t = day_index("2024-08-20")
pred = predict_unet(t)
extent = (EW, EE, ES, EN)
fig, ax = plt.subplots(2, 4, figsize=(16, 7))
last7 = brazil_counts[t - 6 - V_OFF : t + 1 - V_OFF].sum(0)
ax[0, 0].imshow(np.log1p(last7), cmap="inferno", extent=extent)
ax[0, 0].set_title(f"Last 7 days of fires\n(up to {DATES[t].date()})")
for i, k in enumerate([0, 2, 6]):
    shown = np.where(eval_mask, pred[k], np.nan)
    ax[0, i + 1].imshow(shown, cmap="YlOrRd", vmin=0, vmax=1, extent=extent)
    ax[0, i + 1].set_title(f"Forecast day +{k + 1} ({TAG})")
    ax[1, i + 1].imshow(brazil_counts[t + k + 1 - V_OFF] > 0, cmap="gray_r", extent=extent)
    ax[1, i + 1].set_title(f"Observed {DATES[t + k + 1].date()}")
ax[1, 0].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Export for the backend
#
# The network is fully convolutional, so the ONNX file accepts any window whose height
# and width are multiples of 8. Use the first 7 output channels (VIIRS head); in
# production the MODIS input channels and flag stay 0.

# %%
model.eval().cpu()
dummy = torch.from_numpy(features(int(TEST_T[0]), EVAL_WIN))[None]
onnx_path = OUT_DIR / f"fire_forecast_{TAG}.onnx"
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
np.save(OUT_DIR / f"climatology_{TAG}.npy", climatology.astype(np.float16))
np.save(OUT_DIR / f"forecast_mask_{TAG}.npy", mask)

channels = []
for s in ACTIVE:
    channels += [f"{s}_log1p_count_t-{HISTORY_DAYS - 1 - i}" for i in range(HISTORY_DAYS)]
    channels += [f"{s}_log1p_sum_30d", f"{s}_available"]
channels += [f"climatology_t+{k + 1}" for k in range(LEADS)]
channels += ["doy_sin", "doy_cos", "mask", "lat_over_90", "lon_over_180"]
meta = {
    "model": f"fire_forecast_{TAG}",
    "data": "NASA FIRMS active fires, type 0, nominal/high confidence, counts capped at 255",
    "region": REGION,
    "sensors": {
        s: [str(DATES[avail[s][0]].date()), str(DATES[avail[s][1]].date())] for s in ACTIVE
    },
    "train_until": TRAIN_END,
    "bbox": GRIDS[REGION],
    "resolution_deg": RES,
    "grid": [NY, NX],
    "history_days": HISTORY_DAYS,
    "long_window_days": LONG_WINDOW,
    "leads": LEADS,
    "channels": channels,
    "outputs": [f"{s}_fire_t+{k + 1}" for s in ACTIVE for k in range(LEADS)],
    "use_outputs": f"0..{LEADS - 1} (VIIRS head)",
    "test_region": "Brazil",
    "test_avg_precision": summary.mean().round(4).to_dict(),
    "test_brier": brier.mean().round(5).to_dict(),
}
(OUT_DIR / f"fire_forecast_{TAG}.json").write_text(json.dumps(meta, indent=2))

import onnxruntime as ort  # noqa: E402

session = ort.InferenceSession(str(onnx_path))
onnx_out = session.run(None, {"features": dummy.numpy()})[0]
torch_out = model(dummy).detach().numpy()
print("ONNX matches PyTorch:", np.allclose(onnx_out, torch_out, atol=1e-4))
print("Saved to", OUT_DIR.resolve())
model.to(DEVICE)

# %% [markdown]
# ## 10. Compare every run
#
# Reads the saved test results of every run in the outputs folder (same Brazil test
# cells, days and truth).

# %%
runs = {}
for path in sorted(OUT_DIR.glob("test_results_*.json")):
    r = json.loads(path.read_text())
    runs[r.get("tag", r.get("region"))] = r

if len(runs) < 2:
    print("Only one run so far. Change REGION / USE_MODIS and run again to compare.")
else:
    ref = next(iter(runs.values()))
    ap = {
        "climatology": ref["avg_precision"]["climatology"],
        "persistence": ref["avg_precision"]["persistence"],
    }
    br = {"climatology": ref["brier"]["climatology"], "persistence": ref["brier"]["persistence"]}
    for tag, r in runs.items():
        ap[f"unet_{tag}"] = r["avg_precision"][f"unet_{tag}"]
        br[f"unet_{tag}"] = r["brier"][f"unet_{tag}"]
    ap, br = pd.DataFrame(ap), pd.DataFrame(br)
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
