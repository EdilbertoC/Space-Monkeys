# %% [markdown]
# # Space Monkeys — Fire Forecast, base model
#
# Predicts, for every grid cell, the probability of **at least one active fire
# detection** on each of the next **7 days**, like a weather forecast for fire.
#
# Choose the training setup in section 1, same code for every run:
#
# | Setting | Options |
# | --- | --- |
# | `REGION` | `"brazil"` (Brazil only) or `"world"` (every country, Brazil included) |
# | `RES` | `0.25` (~27 km cells) or `0.1` (~11 km, Brazil only) |
# | `USE_MODIS` | `False`: VIIRS 2012–2025. `True`: adds MODIS 2000–2016 as extra history |
# | `RICH_INPUTS` | `True`: each cell-day also carries total FRP and night-time detections |
# | `USE_WEATHER` | `True`: daily weather up to the forecast day (Brazil only) |
# | `N_MODELS`, `TTA` | ensemble size, and averaging over mirrored maps |
#
# Every run is **scored on the same test**: Brazil, 2024–2025, truth = VIIRS detections
# inside Brazil. Scores are reported on the run's own grid and on a **common 0.5° grid**
# (cells merged), so runs at different resolutions can be compared fairly. The last
# section compares every run found in the outputs folder.
#
# **Data:** NASA FIRMS active fire archives (yearly files, no key needed):
# VIIRS S-NPP 375 m (2012-01-20 → 2025-12-31) and, optionally, MODIS Terra/Aqua 1 km
# Collection 6.1 (2000-11-01 → 2016-12-31). Presumed vegetation fires only (`type == 0`),
# nominal/high confidence.
#
# **Why rich inputs:** the satellites see each fire at 375 m–1 km, far finer than a grid
# cell. Instead of predicting at pixel level (pixel positions jitter between passes and
# almost every pixel-day is empty), each cell summarizes *how* it is burning: how many
# pixels, how much radiative power, and whether it keeps burning at night.
#
# **Sensor harmonization with MODIS:** VIIRS sees about twice as many fire cell-days as
# MODIS, so their maps are never mixed. Each sensor has its own input channels plus an
# "available" flag, and the network has one output head per sensor. 2000–2011 trains
# the MODIS head, 2017–2025 the VIIRS head, and the 2012–2016 overlap teaches how they
# relate. MODIS inputs are randomly hidden during the overlap, so the model also works
# with VIIRS alone, which is how it runs from 2017 on and in production.
#
# **Model:** a small 2D U-Net. **Honest evaluation:** split by time (train ≤ 2022,
# validation 2023, test 2024–2025), compared with *climatology* and *persistence*.
#
# **Runtime:** **T4 GPU**. The daily cubes live on the GPU, not in system RAM; section 2
# prints how much GPU memory the chosen setup needs before loading anything.
#
# *SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team.*
# *Data: NASA FIRMS — VIIRS S-NPP 375 m (doi:10.5067/FIRMS/VIIRS/VNP14IMGT_NRT.002) and
# MODIS C6.1 (doi:10.5067/FIRMS/MODIS/MCD14DL.NRT.0061); standard processing archives.*

# %%
# Colab already ships numpy, pandas, torch, scikit-learn and matplotlib.
# %pip install -q onnx onnxscript onnxruntime zarr s3fs

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
RES = 0.25  # grid resolution in degrees: 0.25 or 0.1 (0.1 only for "brazil")
USE_MODIS = False  # True: add MODIS 2000–2016 as extra history
RICH_INPUTS = True  # True: add FRP and night-time detections per cell-day
USE_WEATHER = True  # True: add NASA POWER weather (IMERG rain, MERRA-2), Brazil only
N_MODELS = 3  # ensemble: models trained with different seeds, probabilities averaged
TTA = True  # also average each model over mirrored maps (test-time augmentation)

TRAIN_END = "2022-12-31"
VAL_END = "2023-12-31"  # validation: 2023; test: 2024–2025
LAST_DAY = "2025-12-31"

SENSORS = {
    "viirs": {"first": 2012, "last": 2025},
    "modis": {"first": 2000, "last": 2016},
}
ACTIVE = ["viirs", "modis"] if USE_MODIS else ["viirs"]  # "viirs" is always head 0
FEATS = ["count", "frp", "night"] if RICH_INPUTS else ["count"]
MODIS_DROPOUT = 0.5  # share of overlap samples trained with MODIS hidden

GRIDS = {  # model grid (west, south, east, north)
    "brazil": (-74.0, -34.0, -34.0, 6.0),
    "world": (-180.0, -60.0, 180.0, 80.0),  # no vegetation fires south of 60°S
}
EVAL_BBOX = GRIDS["brazil"]  # where every model is scored
COMMON_RES = 0.5  # common grid for comparing runs at different resolutions

HISTORY_DAYS = 7  # daily maps per sensor and feature
LONG_WINDOW = 30  # one extra channel per sensor: total fires in the last 30 days
LEADS = 7  # forecast days +1 … +7
CLIM_HALF_WINDOW = 7  # climatology is smoothed over ±7 days of the year
FRP_SCALE = 20  # FRP is stored as round(log1p(sum of FRP in MW) * 20), capped at 255

USE_DRIVE = True  # keep downloads, grids and outputs in Google Drive

assert REGION in GRIDS, REGION
assert RES in (0.25, 0.1), "RES must be 0.25 or 0.1"
assert not (REGION == "world" and RES < 0.25), "the world at 0.1° does not fit on a Colab GPU"
assert not (REGION == "world" and USE_WEATHER), "weather inputs are prepared for Brazil only"
TAG = (
    REGION
    + ("_modis" if USE_MODIS else "")
    + ("_rich" if RICH_INPUTS else "")
    + ("_wx" if USE_WEATHER else "")
    + ("" if RES == 0.25 else f"_res{RES}")
)
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

W, S, E, N = GRIDS[REGION]
NX = int(round((E - W) / RES))
NY = int(round((N - S) / RES))
EW, ES, EE, EN = EVAL_BBOX
ER0, ER1 = int(round((N - EN) / RES)), int(round((N - ES) / RES))
EC0, EC1 = int(round((EW - W) / RES)), int(round((EE - W) / RES))
EVAL_WIN = (slice(ER0, ER1), slice(EC0, EC1))
EH, EWD = ER1 - ER0, EC1 - EC0
BLOCK = int(round(COMMON_RES / RES))  # cells per common-grid cell, per side
assert EH % BLOCK == 0 and EWD % BLOCK == 0
print(f"Model grid: {NY} x {NX} cells of {RES}°. Brazil window: rows {ER0}:{ER1}, cols {EC0}:{EC1}")

# %% [markdown]
# ## 2. Download and grid the FIRMS archives
#
# Each sensor-year becomes daily maps per cell: fire-pixel `count` (capped at 255) and,
# with rich inputs, quantized log FRP and `night` detections. Grids are cached as
# compressed files, so later runs skip the download. Brazil-only VIIRS counts are
# gridded separately: they are the shared test truth.

# %%
BASE = "https://firms.modaps.eosdis.nasa.gov/data/country"
URLS = {
    "brazil": BASE + "/{s}/{y}/{s}_{y}_Brazil.csv",
    "world": BASE + "/zips/{s}_{y}_all_countries.zip",
}
FIRMS_NAME = {"viirs": "viirs-snpp", "modis": "modis"}
USECOLS = ["latitude", "longitude", "acq_date", "confidence", "type", "frp", "daynight"]

YEAR0 = min(SENSORS[s]["first"] for s in ACTIVE)
DATES = pd.date_range(f"{YEAR0}-01-01", LAST_DAY, freq="D")
DAY0 = DATES[0]


def day_index(date: str | pd.Timestamp) -> int:
    return int((pd.Timestamp(date) - DAY0).days)


# GPU memory check before loading anything.
_bytes = 0
for _s in ACTIVE:
    _days = day_index(f"{SENSORS[_s]['last']}-12-31") - day_index(f"{SENSORS[_s]['first']}-01-01")
    _bytes += (_days + 1) * NY * NX * len(FEATS)  # one byte per feature
_bytes += 366 * NY * NX * 2  # climatology, float16
print(f"Daily cubes + climatology need ~{_bytes / 1e9:.1f} GB on the {DEVICE}")
if DEVICE == "cuda":
    _free = torch.cuda.mem_get_info()[0]
    if _bytes > 0.8 * _free:
        raise MemoryError(
            f"Needs ~{_bytes / 1e9:.1f} GB but the GPU has {_free / 1e9:.1f} GB free. "
            "Use a bigger GPU (L4/A100), RICH_INPUTS = False, USE_MODIS = False or "
            "REGION = 'brazil'."
        )


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


def grid_into(targets: dict, df: pd.DataFrame, bbox, year_start: pd.Timestamp) -> int:
    """Accumulate detections into per-feature (days, ny, nx) arrays."""
    w, _, _, n = bbox
    days, ny, nx = next(iter(targets.values())).shape
    col = np.floor((df["longitude"].to_numpy() - w) / RES).astype(np.int64)
    row = np.floor((n - df["latitude"].to_numpy()) / RES).astype(np.int64)
    codes, uniq = pd.factorize(df["acq_date"])
    day = (pd.to_datetime(uniq) - year_start).days.to_numpy()[codes]
    ok = (col >= 0) & (col < nx) & (row >= 0) & (row < ny) & (day >= 0) & (day < days)
    idx = (day[ok], row[ok], col[ok])
    np.add.at(targets["count"], idx, 1)
    if "frp" in targets:
        np.add.at(targets["frp"], idx, np.nan_to_num(df["frp"].to_numpy(float)[ok]))
    if "night" in targets:
        np.add.at(targets["night"], idx, (df["daynight"].to_numpy() == "N")[ok])
    return int(ok.sum())


def finish(arrays: dict) -> dict:
    """Accumulators → compact uint8 grids."""
    out = {"count": np.minimum(arrays["count"], 255).astype(np.uint8)}
    if "frp" in arrays:
        q = np.round(np.log1p(arrays["frp"]) * FRP_SCALE)
        out["frp"] = np.minimum(q, 255).astype(np.uint8)
    if "night" in arrays:
        out["night"] = np.minimum(arrays["night"], 255).astype(np.uint8)
    return out


def grid_year(sensor: str, year: int) -> tuple[dict, np.ndarray]:
    """({feature: uint8 grid}, Brazil-only uint8 counts) for one sensor-year."""
    stem = f"{REGION}_{year}" if sensor == "viirs" else f"{REGION}_{sensor}_{year}"
    if RES != 0.25:
        stem += f"_res{RES}"
    cache = CACHE_DIR / f"{stem}.npz"
    if cache.exists():
        z = dict(np.load(cache))
        if "counts" in z:  # caches from earlier notebook versions
            z["count"] = z.pop("counts")
        if all(f in z for f in FEATS):  # earlier caches lack the rich features
            return {f: z[f] for f in FEATS}, z["brazil"]
    start = pd.Timestamp(f"{year}-01-01")
    days = 366 if start.is_leap_year else 365
    acc = {
        "count": np.zeros((days, NY, NX), dtype=np.uint16),
        "frp": np.zeros((days, NY, NX), dtype=np.float32),
        "night": np.zeros((days, NY, NX), dtype=np.uint16),
    }
    brazil = {"count": np.zeros((days, EH, EWD), dtype=np.uint16)}
    url = URLS[REGION].format(s=FIRMS_NAME[sensor], y=year)
    path = download(url, DATA_DIR / url.rsplit("/", 1)[1])
    n = 0
    if REGION == "brazil":
        df = read_detections(path, sensor)
        n = grid_into(acc, df, GRIDS[REGION], start)
        grid_into(brazil, df, EVAL_BBOX, start)
    else:
        with zipfile.ZipFile(path) as zf:
            for member in zf.namelist():
                if not member.endswith(".csv"):
                    continue
                with zf.open(member) as fh:
                    df = read_detections(io.TextIOWrapper(fh, encoding="utf-8"), sensor)
                n += grid_into(acc, df, GRIDS[REGION], start)
                if member.endswith("_Brazil.csv"):
                    grid_into(brazil, df, EVAL_BBOX, start)
        path.unlink()  # the gridded cache below replaces the zip
    grids = finish(acc)
    del acc
    brazil_counts_year = np.minimum(brazil["count"], 255).astype(np.uint8)
    np.savez_compressed(cache, brazil=brazil_counts_year, **grids)
    print(f"  {sensor} {year}: {n:,} detections gridded")
    return {f: grids[f] for f in FEATS}, brazil_counts_year


# Cubes: cubes[sensor][feature] = (days, NY, NX) uint8 tensor on the GPU, starting on
# Jan 1 of the sensor's first year, filled year by year (never whole in system RAM).
cubes, offsets, avail, yearly = {}, {}, {}, {}
for s in ACTIVE:
    first, last = SENSORS[s]["first"], SENSORS[s]["last"]
    offsets[s] = day_index(f"{first}-01-01")
    n_days = day_index(f"{last}-12-31") - offsets[s] + 1
    cubes[s] = {f: torch.zeros((n_days, NY, NX), dtype=torch.uint8, device=DEVICE) for f in FEATS}
    if s == "viirs":
        brazil_counts = np.zeros((n_days, EH, EWD), dtype=np.uint8)
    first_data, yearly[s] = None, {}
    for year in range(first, last + 1):
        t0 = time.time()
        grids, b = grid_year(s, year)
        i = day_index(f"{year}-01-01") - offsets[s]
        for f in FEATS:
            cubes[s][f][i : i + len(b)] = torch.from_numpy(grids[f]).to(DEVICE)
        if s == "viirs":
            brazil_counts[i : i + len(b)] = b
        c = grids["count"]
        if first_data is None and c.any():
            first_data = offsets[s] + i + int(np.argmax(c.reshape(len(c), -1).any(1)))
        yearly[s][year] = int(c.sum(dtype=np.int64))
        del grids, b, c
        print(f"{s} {year} ready ({time.time() - t0:.0f}s)")
    avail[s] = (first_data, offsets[s] + n_days - 1)
    print(f"{s}: {DATES[avail[s][0]].date()} to {DATES[avail[s][1]].date()}")


# %% [markdown]
# ## 2b. Weather (NASA POWER, analysis-ready data on AWS, no login)
#
# Daily values up to the forecast day only (no information from the future):
#
# * **Rain:** GPM IMERG total precipitation, 0.1° (same series from 1998 to today).
# * **Heat, humidity, wind, soil moisture:** MERRA-2 (0.5° × 0.625°) for training.
#   In production the backend uses GEOS-IT, the near-real-time GEOS product in POWER,
#   which agrees with MERRA-2 over Brazil (correlation 0.85–0.90) after a bias fix.
#
# Weather channels: log1p(rain today, last 7 days, last 30 days), consecutive dry days
# (< 1 mm, capped at 60) / 60, (Tmax − 25) / 10, humidity / 100 today and 7-day mean,
# wind / 5, surface soil wetness (0–1).

# %%
WX_VARS = ["T2M_MAX", "RH2M", "WS2M", "GWETTOP"]  # MERRA-2
WX_CHANNELS = [
    "rain_log1p_t",
    "rain_log1p_7d",
    "rain_log1p_30d",
    "dry_days_over_60",
    "tmax_minus25_over10",
    "rh_over100_t",
    "rh_over100_7d",
    "wind_over5",
    "soil_wetness",
]
DRY_WINDOW = 60
POWER_S3 = "s3://nasa-power"


def load_weather():
    """Rain on the model grid, MERRA-2 at its own grid, and model-cell → MERRA-2 index maps.

    Cached in Drive after the first run.
    """
    import xarray as xr

    cache = CACHE_DIR / f"weather_{REGION}_res{RES}_{DATES[0].year}_{DATES[-1].year}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["rain"], z["merra"], z["mrow"], z["mcol"]
    span = slice(str(DATES[0].date()), str(DATES[-1].date()))
    opts = {"anon": True}
    t0 = time.time()
    imerg = xr.open_zarr(
        f"{POWER_S3}/imerg/temporal/power_imerg_daily_temporal_utc.zarr", storage_options=opts
    )["IMERG_PRECTOT"]
    imerg = imerg.sel(time=span, lat=slice(S, N), lon=slice(W, E)).load()
    imerg = imerg.reindex(time=DATES, method="nearest", tolerance="1D")
    p = np.nan_to_num(imerg.values[:, ::-1, :].astype(np.float32))  # north first
    # Average the 0.1° IMERG cells that fall inside each model cell.
    r_idx = np.floor((N - imerg.lat.values[::-1]) / RES).astype(int).clip(0, NY - 1)
    c_idx = np.floor((imerg.lon.values - W) / RES).astype(int).clip(0, NX - 1)
    a_r = np.zeros((NY, len(r_idx)), np.float32)
    a_r[r_idx, np.arange(len(r_idx))] = 1
    a_c = np.zeros((len(c_idx), NX), np.float32)
    a_c[np.arange(len(c_idx)), c_idx] = 1
    a_r /= np.maximum(a_r.sum(1, keepdims=True), 1)
    a_c /= np.maximum(a_c.sum(0, keepdims=True), 1)
    rain = np.stack([a_r @ day @ a_c for day in p]).astype(np.float16)
    del p
    print(f"IMERG rain {rain.shape} ({time.time() - t0:.0f}s)")

    t0 = time.time()
    merra = xr.open_zarr(
        f"{POWER_S3}/merra2/temporal/power_merra2_daily_temporal_utc.zarr", storage_options=opts
    )[WX_VARS]
    merra = merra.sel(time=span, lat=slice(S - 1, N + 1), lon=slice(W - 1, E + 1)).load()
    merra = merra.reindex(time=DATES, method="nearest", tolerance="1D")
    arr = np.stack([merra[v].values for v in WX_VARS]).astype(np.float32)  # var,t,lat,lon
    for i in range(len(WX_VARS)):  # fill gaps with the variable's mean
        arr[i] = np.where(np.isfinite(arr[i]), arr[i], np.nanmean(arr[i]))
    lat_c = N - RES / 2 - np.arange(NY) * RES
    lon_c = W + RES / 2 + np.arange(NX) * RES
    mrow = np.abs(merra.lat.values[None, :] - lat_c[:, None]).argmin(1)
    mcol = np.abs(merra.lon.values[None, :] - lon_c[:, None]).argmin(1)
    merra_arr = arr.astype(np.float16)
    print(f"MERRA-2 {merra_arr.shape} ({time.time() - t0:.0f}s)")
    np.savez_compressed(cache, rain=rain, merra=merra_arr, mrow=mrow, mcol=mcol)
    return rain, merra_arr, mrow, mcol


if USE_WEATHER:
    rain_np, merra_np, mrow, mcol = load_weather()
    rain_g = torch.from_numpy(rain_np).to(DEVICE)
    merra_g = torch.from_numpy(merra_np).to(DEVICE)
    mrow_g = torch.from_numpy(mrow).to(DEVICE)
    mcol_g = torch.from_numpy(mcol).to(DEVICE)
    WX_N = len(WX_CHANNELS)
else:
    WX_N = 0

V_OFF = offsets["viirs"]
train_last = day_index(TRAIN_END)
val_last = day_index(VAL_END)


# %%
def sensor_days(s: str, f: str, t0: int, t1: int, win=(slice(None), slice(None))) -> np.ndarray:
    """Days t0..t1-1 of one sensor feature over a window, zeros outside its archive."""
    ys, xs = win
    cube = cubes[s][f]
    h, w = cube[0, ys, xs].shape
    out = np.zeros((t1 - t0, h, w), dtype=np.uint8)
    a, b = t0 - offsets[s], t1 - offsets[s]
    lo, hi = max(a, 0), min(b, len(cube))
    if hi > lo:
        out[lo - a : hi - a] = cube[lo:hi, ys, xs].cpu().numpy()
    return out


def input_available(s: str, t: int) -> bool:
    """A sensor feeds the model on day t only with a full 30-day history."""
    first, last = avail[s]
    return first + LONG_WINDOW - 1 <= t <= last


def target_available(s: str, day: int) -> bool:
    first, last = avail[s]
    return first <= day <= last


# Forecast mask: cells that burned at least once in the training period (any sensor).
mask_t = torch.zeros((NY, NX), dtype=torch.bool, device=DEVICE)
fire_days_t = torch.zeros((NY, NX), dtype=torch.int32, device=DEVICE)
for s in ACTIVE:
    lo, hi = avail[s][0], min(avail[s][1], train_last)
    for d0 in range(lo, hi + 1, 365):
        chunk = cubes[s]["count"][d0 - offsets[s] : min(d0 + 365, hi + 1) - offsets[s]] > 0
        mask_t |= chunk.any(0)
        if s == "viirs":
            fire_days_t += chunk.sum(0, dtype=torch.int32)
        del chunk
mask = mask_t.cpu().numpy()
fire_days = fire_days_t.cpu().numpy()
eval_mask = np.zeros((EH, EWD), dtype=bool)  # Brazil cells scored for every model
for d0 in range(avail["viirs"][0], train_last + 1, 365):
    b = brazil_counts[d0 - V_OFF : min(d0 + 365, train_last + 1) - V_OFF]
    eval_mask |= (b > 0).any(0)
print(f"Forecast cells: {mask.sum():,}; Brazil test cells: {eval_mask.sum():,}")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
im = ax[0].imshow(np.log1p(fire_days), extent=(W, E, S, N), cmap="inferno")
ax[0].set_title(f"VIIRS fire days per cell, 2012–2022 (log) — {REGION}, {RES}°")
plt.colorbar(im, ax=ax[0])
for s in ACTIVE:
    pd.Series(yearly[s]).plot(ax=ax[1], marker="o", label=s.upper())
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


def build_climatology(cube: torch.Tensor, offset: int) -> torch.Tensor:
    """(366, H, W) float16, computed where the cube lives (GPU), with little memory."""
    total = torch.zeros((366,) + tuple(cube.shape[1:]), dtype=torch.int16, device=cube.device)
    n = np.zeros(366, dtype=np.float32)
    for d in range(avail["viirs"][0], train_last + 1):
        total[doy[d]] += (cube[d - offset] > 0).to(torch.int16)
        n[doy[d]] += 1
    n = np.maximum(n, 1)
    out = torch.empty(total.shape, dtype=torch.float16, device=cube.device)
    width = 2 * CLIM_HALF_WINDOW + 1
    for i in range(366):
        acc = torch.zeros(total.shape[1:], dtype=torch.float32, device=cube.device)
        for j in range(-CLIM_HALF_WINDOW, CLIM_HALF_WINDOW + 1):
            k = (i + j) % 366
            acc += total[k].float() / float(n[k])
        out[i] = (acc / width).half()
    return out


clim_t = build_climatology(cubes["viirs"]["count"], V_OFF)  # model input, on the GPU
if REGION == "brazil":
    eval_climatology = clim_t[:, ER0:ER1, EC0:EC1].float().cpu().numpy()
else:  # the same Brazil-only climatology in every run
    eval_climatology = build_climatology(torch.from_numpy(brazil_counts), V_OFF).float().numpy()
print("Climatology ready:", tuple(clim_t.shape))

# %% [markdown]
# ## 4. Features and targets
#
# Channels for a forecast issued at the end of day `t`, per sensor:
#
# | Channels | Content |
# | --- | --- |
# | 7 | log(1 + fire-pixel count), days t-6 … t |
# | 7 (rich) | log(1 + total FRP in MW), days t-6 … t (stored quantized, /20) |
# | 7 (rich) | log(1 + night-time detections), days t-6 … t |
# | 1 | log(1 + total count of the last 30 days) |
# | 1 | available flag |
#
# then 7 climatology maps (t+1 … t+7), sin/cos of day of year, forecast mask,
# latitude/90 and longitude/180. Outputs: 7 maps per sensor (VIIRS first). Scoring uses
# the VIIRS head and Brazil-only VIIRS detections.

# %%
PER_SENSOR = HISTORY_DAYS * len(FEATS) + 2
N_CHANNELS = PER_SENSOR * len(ACTIVE) + WX_N + LEADS + 2 + 3
N_OUT = LEADS * len(ACTIVE)
MASK_CH = N_CHANNELS - 3
lat_plane = np.repeat(((N - RES / 2 - np.arange(NY) * RES) / 90).astype(np.float32)[:, None], NX, 1)
lon_plane = np.repeat(((W + RES / 2 + np.arange(NX) * RES) / 180).astype(np.float32)[None], NY, 0)
mask_plane = mask.astype(np.float32)
FULL = (slice(None), slice(None))


def transform(f: str, raw: np.ndarray) -> np.ndarray:
    """Stored uint8 values → model input (float32)."""
    raw = raw.astype(np.float32)
    return raw / FRP_SCALE if f == "frp" else np.log1p(raw)


def features(t: int, win: tuple[slice, slice] = FULL, hide: tuple[str, ...] = ()) -> np.ndarray:
    """Input channels for a forecast issued at the end of day t (optionally a window)."""
    ys, xs = win
    h, w = mask_plane[ys, xs].shape
    x = np.zeros((N_CHANNELS, h, w), dtype=np.float32)
    for i, s in enumerate(ACTIVE):
        c = i * PER_SENSOR
        if s in hide or not input_available(s, t):
            continue  # channels and flag stay 0
        for j, f in enumerate(FEATS):
            raw = sensor_days(s, f, t - HISTORY_DAYS + 1, t + 1, win)
            x[c + j * HISTORY_DAYS : c + (j + 1) * HISTORY_DAYS] = transform(f, raw)
        c += HISTORY_DAYS * len(FEATS)
        month = sensor_days(s, "count", t - LONG_WINDOW + 1, t + 1, win)
        x[c] = np.log1p(month.sum(0, dtype=np.float32))
        x[c + 1] = 1.0
    c = PER_SENSOR * len(ACTIVE)
    if USE_WEATHER:
        x[c : c + WX_N] = gpu_weather(t, ys, xs).cpu().numpy()
        c += WX_N
    lead_doy = torch.as_tensor(doy[t + 1 : t + 1 + LEADS], device=clim_t.device)
    x[c : c + LEADS] = clim_t[lead_doy][:, ys, xs].float().cpu().numpy()
    c += LEADS
    angle = 2 * math.pi * doy[t] / 365.25
    x[c], x[c + 1] = math.sin(angle), math.cos(angle)
    x[c + 2], x[c + 3], x[c + 4] = mask_plane[ys, xs], lat_plane[ys, xs], lon_plane[ys, xs]
    return x


def gpu_weather(t: int, ys: slice, xs: slice) -> torch.Tensor:
    """Weather channels for day t over a window (uses only days <= t)."""
    lo = max(t - DRY_WINDOW + 1, 0)
    rain = rain_g[lo : t + 1, ys, xs].float()  # (<= 60, h, w)
    dry = torch.cumprod((rain < 1.0).flip(0).float(), dim=0).sum(0)  # dry days ending at t
    rows, cols = mrow_g[ys], mcol_g[xs]
    m = merra_g[:, max(t - 6, 0) : t + 1][:, :, rows][:, :, :, cols].float()  # var, 7, h, w
    return torch.stack(
        [
            torch.log1p(rain[-1]),
            torch.log1p(rain[-7:].sum(0)),
            torch.log1p(rain[-30:].sum(0)),
            dry.clamp(max=DRY_WINDOW) / DRY_WINDOW,
            (m[0, -1] - 25) / 10,
            m[1, -1] / 100,
            m[1].mean(0) / 100,
            m[2, -1] / 5,
            m[3, -1],
        ]
    )


def eval_targets(t: int) -> np.ndarray:
    """Brazil-only VIIRS truth for days t+1 … t+7 (the shared test truth)."""
    return (brazil_counts[t + 1 - V_OFF : t + 1 + LEADS - V_OFF] > 0).astype(np.float32)


first_t = min(avail[s][0] for s in ACTIVE) + LONG_WINDOW - 1
TRAIN_T = np.arange(first_t, train_last - LEADS + 1)
VAL_T = np.arange(train_last + 1, val_last - LEADS + 1)
TEST_T = np.arange(val_last + 1, len(DATES) - LEADS)
print(
    f"Channels: {N_CHANNELS}. Forecast days — train {len(TRAIN_T)} "
    f"(from {DATES[first_t].date()}), val {len(VAL_T)}, test {len(TEST_T)}"
)

# %% [markdown]
# ## 5. Reference models and scoring
#
# * **Climatology:** probability = climatology of the target day.
# * **Persistence:** probability learned on VIIRS training data from whether the cell
#   burned in the last 3 days, per lead day.
#
# Scores: average precision and Brier score per lead day over the Brazil test cells, on
# the run's own grid and on the common 0.5° grid. Merging cells uses
# P(any fire) = 1 − ∏(1 − p) and "any fire" as truth.


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


def to_common(a: np.ndarray, kind: str) -> np.ndarray:
    """Merge BLOCK x BLOCK cells: probabilities → P(any), truth/mask → any."""
    lead = a.shape[:-2]
    blocks = a.reshape(*lead, EH // BLOCK, BLOCK, EWD // BLOCK, BLOCK)
    if kind == "prob":
        return 1 - np.prod(1 - blocks, axis=(-3, -1))
    return blocks.max(axis=(-3, -1))


common_mask = to_common(eval_mask, "any").astype(bool)


def evaluate(predict, days: np.ndarray, step: int = 2, grids=("native", "common")) -> dict:
    """{grid: DataFrame of average precision, Brier, base rate per lead day}."""
    store = {g: ([[] for _ in range(LEADS)], [[] for _ in range(LEADS)]) for g in grids}
    for t in days[::step]:
        p, y = predict(t), eval_targets(t)
        for g in grids:
            pp, yy, mm = (
                (p, y, eval_mask)
                if g == "native"
                else (to_common(p, "prob"), to_common(y, "any"), common_mask)
            )
            for k in range(LEADS):
                store[g][0][k].append(pp[k][mm])
                store[g][1][k].append(yy[k][mm])
    out = {}
    for g in grids:
        rows = []
        for k in range(LEADS):
            p, y = np.concatenate(store[g][0][k]), np.concatenate(store[g][1][k])
            rows.append(
                {
                    "lead_day": k + 1,
                    "avg_precision": average_precision_score(y, p),
                    "brier": brier_score_loss(y, np.clip(p, 0, 1)),
                    "base_rate": y.mean(),
                }
            )
        out[g] = pd.DataFrame(rows).set_index("lead_day")
    return out


for name, fn in [("climatology", predict_climatology), ("persistence", predict_persistence)]:
    print(f"\n{name} (validation 2023, native grid)\n", evaluate(fn, VAL_T)["native"].round(4))


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


class Forecaster(nn.Module):
    """Ensemble of U-Nets, optionally averaged over mirrored inputs.

    Returns logits of the mean probability, so sigmoid(output) is the forecast and the
    exported ONNX file is used exactly like a single model.
    """

    def __init__(self, members: list[nn.Module], tta: bool) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)
        self.flips = [(), (-1,), (-2,), (-2, -1)] if tta else [()]

    def forward(self, x):
        total = 0.0
        for member in self.members:
            for dims in self.flips:
                xi = torch.flip(x, dims) if dims else x
                pi = torch.sigmoid(member(xi).float())
                total = total + (torch.flip(pi, dims) if dims else pi)
        p = (total / (len(self.members) * len(self.flips))).clamp(1e-6, 1 - 1e-6)
        return torch.log(p) - torch.log1p(-p)


# Each head starts from its sensor's training base rate.
BASE_RATES = {}
for s in ACTIVE:
    lo, hi = avail[s][0], min(avail[s][1], train_last)
    days = np.arange(lo, hi + 1, 20)
    BASE_RATES[s] = float(
        np.mean(
            [(cubes[s]["count"][d - offsets[s]][mask_t] > 0).float().mean().item() for d in days]
        )
    )
    print(f"{s} training base rate: {BASE_RATES[s]:.4f}")


def new_model(seed: int) -> UNet:
    torch.manual_seed(seed)
    net = UNet(N_CHANNELS, N_OUT).to(DEVICE)
    with torch.no_grad():
        for i, s in enumerate(ACTIVE):
            rate = BASE_RATES[s]
            net.head.bias[i * LEADS : (i + 1) * LEADS] = math.log(rate / (1 - rate))
    return net


model = new_model(0)  # `model` is whatever `predict_unet` should run
print(f"Parameters per model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

# %% [markdown]
# ## 7. Training
#
# Every batch is built on the GPU. Random crops (≈ 16° × 16°) with random flips and
# rotations; early stopping on the Brazil validation average precision of the VIIRS head.
# `N_MODELS` networks are trained from different seeds and averaged.

# %%
CROP = 64 if RES == 0.25 else 160
BATCH = 128 if RES == 0.25 else 32
STEPS_PER_EPOCH = 300
EPOCHS = 20
PATIENCE = 4
torch.backends.cudnn.benchmark = True
rng = np.random.default_rng(42)
clim_g = clim_t  # float16
doy_g = torch.from_numpy(doy).to(DEVICE)
static_g = torch.from_numpy(np.stack([mask_plane, lat_plane, lon_plane])).to(DEVICE)

# Top-left corners of crops that are at least 20% forecast cells.
_cum = np.pad(mask_plane.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
_cover = (
    _cum[CROP:, CROP:] - _cum[:-CROP, CROP:] - _cum[CROP:, :-CROP] + _cum[:-CROP, :-CROP]
) / CROP**2
corners = np.argwhere(_cover >= 0.2)
print(f"Crop {CROP}x{CROP} cells; valid crop positions: {len(corners):,}")


def gpu_days(s: str, f: str, t0: int, t1: int, ys: slice, xs: slice) -> torch.Tensor:
    """GPU twin of `sensor_days`."""
    cube = cubes[s][f]
    h = len(range(*ys.indices(NY)))
    w = len(range(*xs.indices(NX)))
    out = torch.zeros((t1 - t0, h, w), dtype=torch.uint8, device=DEVICE)
    a, b = t0 - offsets[s], t1 - offsets[s]
    lo, hi = max(a, 0), min(b, len(cube))
    if hi > lo:
        out[lo - a : hi - a] = cube[lo:hi, ys, xs]
    return out


def gpu_transform(f: str, raw: torch.Tensor) -> torch.Tensor:
    raw = raw.float()
    return raw / FRP_SCALE if f == "frp" else torch.log1p(raw)


def gpu_features(t: int, ys: slice, xs: slice, hide: tuple[str, ...] = ()) -> torch.Tensor:
    """Same channels as `features()`, for one day and window, built on the GPU."""
    h = len(range(*ys.indices(NY)))
    w = len(range(*xs.indices(NX)))
    parts = []
    for s in ACTIVE:
        if s in hide or not input_available(s, t):
            parts.append(torch.zeros((PER_SENSOR, h, w), device=DEVICE))
            continue
        for f in FEATS:
            parts.append(gpu_transform(f, gpu_days(s, f, t - HISTORY_DAYS + 1, t + 1, ys, xs)))
        month = gpu_days(s, "count", t - LONG_WINDOW + 1, t + 1, ys, xs).float()
        parts += [torch.log1p(month.sum(0, keepdim=True)), torch.ones((1, h, w), device=DEVICE)]
    if USE_WEATHER:
        parts.append(gpu_weather(t, ys, xs))
    parts.append(clim_g[doy_g[t + 1 : t + 1 + LEADS]][:, ys, xs].float())
    angle = 2 * math.pi * doy[t] / 365.25
    season = torch.tensor([math.sin(angle), math.cos(angle)], device=DEVICE)
    parts.append(season[:, None, None].expand(-1, h, w))
    parts.append(static_g[:, ys, xs])
    return torch.cat(parts, dim=0)


def gpu_targets(t: int, ys: slice, xs: slice) -> tuple[torch.Tensor, torch.Tensor]:
    """Targets for every head and a per-channel weight (0 where a sensor saw nothing)."""
    ys_, ws = [], []
    for s in ACTIVE:
        ys_.append(gpu_days(s, "count", t + 1, t + 1 + LEADS, ys, xs) > 0)
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


# The model sees a ~4° margin around Brazil when predicting it (context across borders).
MARGIN = int(round(4 / RES))
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

use_amp = DEVICE == "cuda"


def train_member(seed: int) -> tuple[UNet, list[dict]]:
    """Train one network; returns it at its best validation epoch, and its history."""
    global model, rng
    rng = np.random.default_rng(42 + seed)
    net = model = new_model(seed)
    optimizer = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3, total_steps=EPOCHS * STEPS_PER_EPOCH
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_ap, best_state, bad_epochs, history = -1.0, None, 0, []
    for epoch in range(1, EPOCHS + 1):
        net.train()
        t0, total = time.time(), 0.0
        for _ in range(STEPS_PER_EPOCH):
            x, y, m, w = random_batch()
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = net(x)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none")
            weight = m[:, None] * w[:, :, None, None]
            loss = (loss * weight).sum() / weight.sum().clamp(min=1.0)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item()
        val = evaluate(predict_unet, VAL_T, step=4, grids=("native",))["native"]
        ap = val["avg_precision"].mean()
        history.append({"epoch": epoch, "loss": total / STEPS_PER_EPOCH, "val_ap": ap})
        print(
            f"model {seed + 1}/{N_MODELS}  epoch {epoch:2d}  loss {total / STEPS_PER_EPOCH:.4f}  "
            f"val AP {ap:.4f}  ({time.time() - t0:.0f}s)"
        )
        if ap > best_ap:
            best_ap, bad_epochs = ap, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("Early stopping")
                break
    net.load_state_dict(best_state)
    return net, history


members, histories = [], []
for seed in range(N_MODELS):
    net, history = train_member(seed)
    members.append(net)
    histories.append(history)
torch.save([m.state_dict() for m in members], OUT_DIR / f"fire_forecast_{TAG}.pt")

# %% [markdown]
# ## 8. Test results — Brazil, 2024–2025 (never seen in training)

# %%
model = members[0]
single = evaluate(predict_unet, TEST_T, grids=("native",))["native"]
model = Forecaster(members, tta=TTA).to(DEVICE)  # the forecast model from here on
results = {
    "climatology": evaluate(predict_climatology, TEST_T),
    "persistence": evaluate(predict_persistence, TEST_T),
    f"unet_{TAG}": evaluate(predict_unet, TEST_T),
}
ensemble_ap = results[f"unet_{TAG}"]["native"]["avg_precision"].mean()
print(
    f"One model: AP {single['avg_precision'].mean():.4f}  ->  "
    f"{N_MODELS} models{' + mirrored maps' if TTA else ''}: AP {ensemble_ap:.4f}"
)
tables = {}
for g in ("native", "common"):
    ap = pd.concat({n: r[g]["avg_precision"] for n, r in results.items()}, axis=1)
    br = pd.concat({n: r[g]["brier"] for n, r in results.items()}, axis=1)
    tables[g] = (ap, br)
    label = f"{RES}° grid" if g == "native" else f"common {COMMON_RES}° grid"
    print(f"\n=== {label} — base rate {results['climatology'][g]['base_rate'].mean():.3f}")
    print("Average precision (higher is better)\n", ap.round(4))
    print("Brier score (lower is better)\n", br.round(5))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
tables["native"][0].plot(marker="o", ax=ax[0], title=f"Average precision by lead day ({RES}°)")
tables["native"][1].plot(marker="o", ax=ax[1], title=f"Brier score by lead day ({RES}°)")
for a in ax:
    a.set_xlabel("Lead day")
plt.tight_layout()
plt.show()

(OUT_DIR / f"test_results_{TAG}.json").write_text(
    json.dumps(
        {
            "tag": TAG,
            "region": REGION,
            "resolution_deg": RES,
            "sensors": ACTIVE,
            "features": FEATS + (["weather"] if USE_WEATHER else []),
            "n_models": N_MODELS,
            "tta": TTA,
            "single_model_avg_precision": round(float(single["avg_precision"].mean()), 5),
            "epochs_trained": [len(h) for h in histories],
            "history": histories,
            "avg_precision": tables["native"][0].round(5).to_dict(),
            "brier": tables["native"][1].round(6).to_dict(),
            "common_grid": {
                "resolution_deg": COMMON_RES,
                "avg_precision": tables["common"][0].round(5).to_dict(),
                "brier": tables["common"][1].round(6).to_dict(),
            },
        },
        indent=2,
    )
)

# %%
# New fires vs continuing fires, day +1: cells with no fire in the last 3 days are the
# hard, early-warning case; cells already burning are the easy one.
groups = {"new fires": ([], [], []), "continuing fires": ([], [], [])}
for t in TEST_T[::3]:
    recent = eval_recent(t)
    p, c, y = predict_unet(t)[0], predict_climatology(t)[0], eval_targets(t)[0]
    for name, sel in (("new fires", eval_mask & ~recent), ("continuing fires", eval_mask & recent)):
        groups[name][0].append(p[sel])
        groups[name][1].append(c[sel])
        groups[name][2].append(y[sel])
rows = []
for name, (ps, cs, ys_) in groups.items():
    p, c, y = np.concatenate(ps), np.concatenate(cs), np.concatenate(ys_)
    rows.append(
        {
            "cells": name,
            "share of cell-days": len(y) / sum(len(np.concatenate(g[2])) for g in groups.values()),
            "fire rate": y.mean(),
            "AP climatology": average_precision_score(y, c),
            "AP U-Net": average_precision_score(y, p),
        }
    )
new_vs_continuing = pd.DataFrame(rows).set_index("cells")
print("Day +1, Brazil test 2024–2025\n", new_vs_continuing.round(4))

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
# The exported file contains the whole ensemble (and the mirrored-map averaging), and
# is used like a single model: probability = sigmoid(output). It is fully
# convolutional, so it accepts any window whose height and width are multiples of 8.
# Use the first 7 output channels (VIIRS head); in
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
np.save(OUT_DIR / f"climatology_{TAG}.npy", clim_t.cpu().numpy())  # float16
np.save(OUT_DIR / f"forecast_mask_{TAG}.npy", mask)

channels = []
for s in ACTIVE:
    for f in FEATS:
        channels += [f"{s}_{f}_t-{HISTORY_DAYS - 1 - i}" for i in range(HISTORY_DAYS)]
    channels += [f"{s}_log1p_count_sum_30d", f"{s}_available"]
channels += WX_CHANNELS if USE_WEATHER else []
channels += [f"climatology_t+{k + 1}" for k in range(LEADS)]
channels += ["doy_sin", "doy_cos", "mask", "lat_over_90", "lon_over_180"]
meta = {
    "model": f"fire_forecast_{TAG}",
    "data": "NASA FIRMS active fires, type 0, nominal/high confidence",
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
    "features": {
        "count": "fire pixels per cell-day, capped at 255; channel = log1p(count)",
        "frp": (
            f"q = min(round(log1p(sum of FRP in MW) * {FRP_SCALE}), 255); channel = q / {FRP_SCALE}"
        ),
        "night": "night-time (daynight == N) detections, capped at 255; channel = log1p(n)",
    },
    "weather": {
        "rain": "NASA POWER GPM IMERG daily precipitation (mm/day), mean over each cell",
        "merra2": f"NASA POWER MERRA-2 daily {WX_VARS}, nearest cell; production: GEOS-IT",
        "dry_day_threshold_mm": 1.0,
    }
    if USE_WEATHER
    else None,
    "channels": channels,
    "outputs": [f"{s}_fire_t+{k + 1}" for s in ACTIVE for k in range(LEADS)],
    "use_outputs": f"0..{LEADS - 1} (VIIRS head)",
    "ensemble": {"n_models": N_MODELS, "mirrored_map_averaging": TTA},
    "test_region": "Brazil",
    "test_new_vs_continuing_day1": new_vs_continuing.round(4).to_dict(orient="index"),
    "test_avg_precision": tables["native"][0].mean().round(4).to_dict(),
    "test_brier": tables["native"][1].mean().round(5).to_dict(),
    "test_common_grid_avg_precision": tables["common"][0].mean().round(4).to_dict(),
}
(OUT_DIR / f"fire_forecast_{TAG}.json").write_text(json.dumps(meta, indent=2))

import onnxruntime as ort  # noqa: E402

session = ort.InferenceSession(str(onnx_path))
onnx_out = session.run(None, {"features": dummy.numpy()})[0]
torch_out = model(dummy).detach().numpy()
print("ONNX matches PyTorch:", np.allclose(onnx_out, torch_out, atol=1e-3))
print("Saved to", OUT_DIR.resolve())
model.to(DEVICE)

# %% [markdown]
# ## 10. Compare every run
#
# Same Brazil test days and truth for every run. Compare runs on the **common 0.5° grid**
# when their resolutions differ; the native grid is comparable only between runs at the
# same resolution.

# %%
runs = {}
for path in sorted(OUT_DIR.glob("test_results_*.json")):
    r = json.loads(path.read_text())
    runs[r.get("tag", r.get("region"))] = r

if len(runs) < 2:
    print("Only one run so far. Change the settings in section 1 and run again to compare.")
else:
    rows = []
    for tag, r in runs.items():
        res = r.get("resolution_deg", 0.25)
        row = {"run": tag, "res": res, "features": "+".join(r.get("features", ["count"]))}
        row["native_AP"] = float(np.mean(list(r["avg_precision"][f"unet_{tag}"].values())))
        row["native_brier"] = float(np.mean(list(r["brier"][f"unet_{tag}"].values())))
        if "common_grid" in r:
            row["common_AP"] = float(
                np.mean(list(r["common_grid"]["avg_precision"][f"unet_{tag}"].values()))
            )
            row["common_brier"] = float(
                np.mean(list(r["common_grid"]["brier"][f"unet_{tag}"].values()))
            )
            row["common_climatology_AP"] = float(
                np.mean(list(r["common_grid"]["avg_precision"]["climatology"].values()))
            )
        row["native_climatology_AP"] = float(
            np.mean(list(r["avg_precision"]["climatology"].values()))
        )
        rows.append(row)
    table = (
        pd.DataFrame(rows)
        .set_index("run")
        .sort_values(
            "common_AP" if "common_AP" in pd.DataFrame(rows) else "native_AP", ascending=False
        )
    )
    print("Brazil test 2024–2025, mean over the 7 lead days")
    print(table.round(4).to_string())
