# %% [markdown]
# # Space Monkeys — Fire Forecast, base model
#
# Predicts, for every 0.25° cell of Brazil, the probability of **at least one active fire
# detection** on each of the next **7 days**, like a weather forecast for fire.
#
# **Data:** NASA FIRMS VIIRS S-NPP 375 m active fire archive for Brazil, 2012-01-20 to
# 2025-12-31 (yearly country files, no key needed). One sensor only, so the series is
# consistent without cross-sensor harmonization. Only `type == 0` (presumed vegetation
# fire) with nominal or high confidence is used.
#
# **Model:** a small 2D U-Net. Input = the last 7 days of fire maps, the last 30 days,
# the historical climatology for each target day, seasonality and position, stacked as
# image channels. Output = 7 probability maps (day +1 … +7).
#
# **Honest evaluation:** split by time (train 2012–2022, validation 2023, test 2024–2025)
# and compared against two simple references: *climatology* (what usually happens on
# this day of the year) and *persistence* (recent fire means more fire).
#
# **Runtime:** Runtime → Change runtime type → **T4 GPU**. The full run takes roughly
# 30–60 minutes, most of it downloading (~1.3 GB) and training.
#
# **Outputs:** `fire_forecast_unet.onnx` and `fire_forecast_meta.json`, consumed by the
# backend's `/forecast` endpoint.
#
# *SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team.*
# *Data: NASA FIRMS, VIIRS S-NPP 375 m (doi:10.5067/FIRMS/VIIRS/VNP14IMGT_NRT.002;
# standard processing archive).*

# %%
# Colab already ships numpy, pandas, torch, scikit-learn and matplotlib.
# %pip install -q onnx onnxscript onnxruntime

# %%
import json
import math
import time
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
FIRST_YEAR, LAST_YEAR = 2012, 2025
TRAIN_END = "2022-12-31"  # train: 2012-01-20 … 2022-12-31
VAL_END = "2023-12-31"  # validation: 2023; test: 2024-2025

# Brazil bounding box (west, south, east, north) and grid resolution in degrees.
BBOX = (-74.0, -34.0, -34.0, 6.0)
RES = 0.25
HISTORY_DAYS = 7  # daily fire maps fed to the model
LONG_WINDOW = 30  # one extra channel: total fires in the last 30 days
LEADS = 7  # forecast days +1 … +7
CLIM_HALF_WINDOW = 7  # climatology is smoothed over ±7 days of the year

# Set to True to keep downloads and outputs in Google Drive between sessions.
USE_DRIVE = False

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
OUT_DIR = WORK / "outputs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, S, E, N = BBOX
NX = int(round((E - W) / RES))
NY = int(round((N - S) / RES))
print(f"Grid: {NY} rows x {NX} cols at {RES}°")

# %% [markdown]
# ## 2. Download the FIRMS archive (VIIRS S-NPP, Brazil)

# %%
FIRMS_URL = (
    "https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/{y}/viirs-snpp_{y}_Brazil.csv"
)


def download_year(year: int) -> Path:
    path = DATA_DIR / f"viirs-snpp_{year}_Brazil.csv"
    if path.exists() and path.stat().st_size > 1000:
        return path
    tmp = path.with_suffix(".part")
    with requests.get(FIRMS_URL.format(y=year), stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.rename(path)
    return path


for year in range(FIRST_YEAR, LAST_YEAR + 1):
    t0 = time.time()
    p = download_year(year)
    print(f"{year}: {p.stat().st_size / 1e6:.0f} MB ({time.time() - t0:.0f}s)")

# %% [markdown]
# ## 3. Grid the detections into a daily cube
#
# `counts[day, row, col]` = number of fire pixels in that cell on that day.

# %%
DATES = pd.date_range(f"{FIRST_YEAR}-01-01", f"{LAST_YEAR}-12-31", freq="D")
DAY0 = DATES[0]
counts = np.zeros((len(DATES), NY, NX), dtype=np.uint16)

USECOLS = ["latitude", "longitude", "acq_date", "confidence", "type"]
for year in range(FIRST_YEAR, LAST_YEAR + 1):
    df = pd.read_csv(DATA_DIR / f"viirs-snpp_{year}_Brazil.csv", usecols=USECOLS)
    df = df[(df["type"] == 0) & df["confidence"].isin(["n", "h"])]
    col = np.floor((df["longitude"].to_numpy() - W) / RES).astype(int)
    row = np.floor((N - df["latitude"].to_numpy()) / RES).astype(int)
    day = (pd.to_datetime(df["acq_date"]) - DAY0).dt.days.to_numpy()
    ok = (col >= 0) & (col < NX) & (row >= 0) & (row < NY)
    np.add.at(counts, (day[ok], row[ok], col[ok]), 1)
    print(f"{year}: {ok.sum():,} detections")

first_day = int(np.argmax(counts.reshape(len(DATES), -1).sum(1) > 0))
print("First day with data:", DATES[first_day].date())

# %%
# Cells that ever burned in the training period = the area we forecast and score.
train_last = int((pd.Timestamp(TRAIN_END) - DAY0).days)
mask = (counts[first_day : train_last + 1] > 0).any(axis=0)
print(f"Forecast cells: {mask.sum():,} of {mask.size:,}")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
fire_days = (counts[first_day : train_last + 1] > 0).sum(0)
im = ax[0].imshow(np.log1p(fire_days), extent=(W, E, S, N), cmap="inferno")
ax[0].set_title("Fire days per cell, 2012–2022 (log)")
plt.colorbar(im, ax=ax[0])
daily = pd.Series(counts.reshape(len(DATES), -1).sum(1), index=DATES)
daily.groupby(daily.index.dayofyear).mean().plot(ax=ax[1])
ax[1].set_title("Mean detections per day of year (Brazil)")
ax[1].set_xlabel("Day of year")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Climatology (training years only)
#
# For each cell and day of year: the share of training years with a fire that day,
# smoothed over ±7 days. It is both an input channel and the first reference model.

# %%
fire = counts > 0
doy = DATES.dayofyear.to_numpy() - 1  # 0..365
train_days = np.arange(first_day, train_last + 1)
clim_sum = np.zeros((366, NY, NX), dtype=np.float32)
clim_n = np.zeros(366, dtype=np.float32)
for d in train_days:
    clim_sum[doy[d]] += fire[d]
    clim_n[doy[d]] += 1
clim_n[365] = max(clim_n[365], 1)  # leap day
clim_raw = clim_sum / clim_n[:, None, None]
# Circular smoothing over the year.
kernel = np.ones(2 * CLIM_HALF_WINDOW + 1, dtype=np.float32)
kernel /= kernel.sum()
padded = np.concatenate([clim_raw[-CLIM_HALF_WINDOW:], clim_raw, clim_raw[:CLIM_HALF_WINDOW]])
climatology = np.zeros_like(clim_raw)
for i, w in enumerate(kernel):
    climatology += w * padded[i : i + 366]
print("Climatology ready:", climatology.shape)

# %% [markdown]
# ## 5. Features and targets
#
# Channels for a forecast issued at the end of day `t`:
#
# | Channels | Content |
# | --- | --- |
# | 7 | log(1 + fire count) for days t-6 … t |
# | 1 | log(1 + total fires in the last 30 days) |
# | 7 | climatology for target days t+1 … t+7 |
# | 2 | sin/cos of day of year |
# | 3 | forecast mask, normalized latitude and longitude |
#
# Targets: fire (yes/no) on days t+1 … t+7.

# %%
N_CHANNELS = HISTORY_DAYS + 1 + LEADS + 2 + 3
lat_plane = np.repeat(np.linspace(1, -1, NY, dtype=np.float32)[:, None], NX, axis=1)
lon_plane = np.repeat(np.linspace(-1, 1, NX, dtype=np.float32)[None, :], NY, axis=0)
mask_plane = mask.astype(np.float32)


FULL = (slice(None), slice(None))


def features(t: int, win: tuple[slice, slice] = FULL) -> np.ndarray:
    """Input channels for a forecast issued at the end of day t (optionally a window)."""
    ys, xs = win
    h, w = mask_plane[ys, xs].shape
    x = np.empty((N_CHANNELS, h, w), dtype=np.float32)
    x[:HISTORY_DAYS] = np.log1p(counts[t - HISTORY_DAYS + 1 : t + 1, ys, xs])
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


def targets(t: int, win: tuple[slice, slice] = FULL) -> np.ndarray:
    ys, xs = win
    return fire[t + 1 : t + 1 + LEADS, ys, xs].astype(np.float32)


start = first_day + LONG_WINDOW
val_last = int((pd.Timestamp(VAL_END) - DAY0).days)
end = len(DATES) - LEADS - 1
TRAIN_T = np.arange(start, train_last - LEADS + 1)
VAL_T = np.arange(train_last + 1, val_last - LEADS + 1)
TEST_T = np.arange(val_last + 1, end + 1)
print(f"Forecast days — train {len(TRAIN_T)}, val {len(VAL_T)}, test {len(TEST_T)}")

# %% [markdown]
# ## 6. Reference models
#
# * **Climatology:** probability = climatology of the target day.
# * **Persistence:** probability learned on training data from whether the cell burned
#   in the last 3 days, per lead day.

# %%
recent = np.zeros((len(DATES), NY, NX), dtype=bool)
for lag in range(3):
    recent[lag:] |= fire[: len(DATES) - lag]

persist_p = np.zeros((LEADS, 2), dtype=np.float32)  # [lead, recent fire?]
for k in range(LEADS):
    sample = TRAIN_T[::5]
    r = recent[sample][:, mask]
    y = fire[sample + k + 1][:, mask]
    persist_p[k, 1] = y[r].mean()
    persist_p[k, 0] = y[~r].mean()
print("Per lead day: [P(fire | no recent fire), P(fire | recent fire)]\n", persist_p.round(3))


def predict_climatology(t: int) -> np.ndarray:
    return np.stack([climatology[doy[t + k + 1]] for k in range(LEADS)])


def predict_persistence(t: int) -> np.ndarray:
    r = recent[t]
    return np.stack([np.where(r, persist_p[k, 1], persist_p[k, 0]) for k in range(LEADS)])


def evaluate(predict, days: np.ndarray, step: int = 2) -> pd.DataFrame:
    """Average precision and Brier score per lead day, over forecast cells."""
    preds = [[] for _ in range(LEADS)]
    truth = [[] for _ in range(LEADS)]
    for t in days[::step]:
        p, y = predict(t), targets(t)
        for k in range(LEADS):
            preds[k].append(p[k][mask])
            truth[k].append(y[k][mask])
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


ref_val = {
    "climatology": evaluate(predict_climatology, VAL_T),
    "persistence": evaluate(predict_persistence, VAL_T),
}
for name, table in ref_val.items():
    print(f"\n{name} (validation 2023)\n", table.round(4))

# %% [markdown]
# ## 7. U-Net
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
base_rate = float(fire[TRAIN_T[::5] + 1][:, mask].mean())
with torch.no_grad():
    model.head.bias.fill_(math.log(base_rate / (1 - base_rate)))
print(f"Training base rate: {base_rate:.4f}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

# %% [markdown]
# ## 8. Training
#
# The whole dataset (~1.5 GB) is copied to the GPU once, and every batch is built there,
# so the GPU never waits for the CPU. Random 64×64 crops with random flips and
# rotations; early stopping on validation average precision.

# %%
CROP = 64
BATCH = 128
STEPS_PER_EPOCH = 300
EPOCHS = 20
PATIENCE = 4
torch.backends.cudnn.benchmark = True
gen = torch.Generator(device=DEVICE).manual_seed(42)

# Dataset on the device.
_counts = torch.from_numpy(counts.astype(np.int32)).to(DEVICE)
logc_g = torch.log1p(_counts.float())
csum_g = torch.cumsum(_counts, dim=0, dtype=torch.int32)
fire_g = _counts > 0
del _counts
clim_g = torch.from_numpy(climatology).to(DEVICE)
doy_g = torch.from_numpy(doy).to(DEVICE)
static_g = torch.from_numpy(np.stack([mask_plane, lat_plane, lon_plane])).to(DEVICE)
train_t_g = torch.from_numpy(TRAIN_T).to(DEVICE)
hist_offsets = torch.arange(-HISTORY_DAYS + 1, 1, device=DEVICE)
lead_offsets = torch.arange(1, LEADS + 1, device=DEVICE)

# Top-left corners of crops that are at least 20% forecast cells.
_cum = np.pad(mask_plane.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
_cover = (
    _cum[CROP:, CROP:] - _cum[:-CROP, CROP:] - _cum[CROP:, :-CROP] + _cum[:-CROP, :-CROP]
) / CROP**2
corners_g = torch.from_numpy(np.argwhere(_cover >= 0.2)).to(DEVICE)
print(f"Valid crop positions: {len(corners_g):,}")


def gpu_features(ts: torch.Tensor) -> torch.Tensor:
    """Same channels as `features()`, for a batch of days, built on the GPU."""
    b = ts.shape[0]
    hist = logc_g[ts[:, None] + hist_offsets]  # B×7×NY×NX
    recent30 = torch.log1p((csum_g[ts] - csum_g[ts - LONG_WINDOW]).float())[:, None]
    clim = clim_g[doy_g[ts[:, None] + lead_offsets]]  # B×7×NY×NX
    angle = 2 * math.pi * doy_g[ts].float() / 365.25
    season = torch.stack([angle.sin(), angle.cos()], 1)[:, :, None, None].expand(-1, -1, NY, NX)
    static = static_g.expand(b, -1, -1, -1)
    return torch.cat([hist, recent30, clim, season, static], dim=1)


def random_batch():
    ts = train_t_g[torch.randint(len(train_t_g), (BATCH,), device=DEVICE, generator=gen)]
    corners = corners_g[torch.randint(len(corners_g), (BATCH,), device=DEVICE, generator=gen)]
    x_full = gpu_features(ts)
    y_full = fire_g[ts[:, None] + lead_offsets].float()
    rows, cols = corners[:, 0].tolist(), corners[:, 1].tolist()
    x = torch.stack(
        [
            x_full[i, :, r : r + CROP, c : c + CROP]
            for i, (r, c) in enumerate(zip(rows, cols, strict=True))
        ]
    )
    y = torch.stack(
        [
            y_full[i, :, r : r + CROP, c : c + CROP]
            for i, (r, c) in enumerate(zip(rows, cols, strict=True))
        ]
    )
    m = x[:, N_CHANNELS - 3]  # the mask channel
    k = int(torch.randint(4, (1,), generator=gen, device=DEVICE))
    x, y, m = (torch.rot90(a, k, dims=(-2, -1)) for a in (x, y, m))
    if torch.rand(1, generator=gen, device=DEVICE).item() < 0.5:
        x, y, m = (torch.flip(a, dims=(-1,)) for a in (x, y, m))
    return x.contiguous(), y.contiguous(), m.contiguous()


@torch.no_grad()
def predict_unet(t: int) -> np.ndarray:
    model.eval()
    x = gpu_features(torch.tensor([t], device=DEVICE))
    with torch.autocast(device_type="cuda", enabled=DEVICE == "cuda"):
        logits = model(x)
    return torch.sigmoid(logits.float())[0].cpu().numpy()


# The GPU features must match the NumPy ones used later by the backend.
_t = int(VAL_T[0])
_diff = np.abs(gpu_features(torch.tensor([_t], device=DEVICE))[0].cpu().numpy() - features(_t))
assert _diff.max() < 1e-4, _diff.max()
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
torch.save(best_state, OUT_DIR / "fire_forecast_unet.pt")

# %% [markdown]
# ## 9. Test results (2024–2025, never seen in training)

# %%
results = {
    "climatology": evaluate(predict_climatology, TEST_T),
    "persistence": evaluate(predict_persistence, TEST_T),
    "unet": evaluate(predict_unet, TEST_T),
}
summary = pd.concat({name: t["avg_precision"] for name, t in results.items()}, axis=1)
brier = pd.concat({name: t["brier"] for name, t in results.items()}, axis=1)
print("Average precision (higher is better)\n", summary.round(4))
print("\nBrier score (lower is better)\n", brier.round(4))
print("\nBase rate per lead:", results["unet"]["base_rate"].round(4).tolist())

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
summary.plot(marker="o", ax=ax[0], title="Average precision by lead day")
brier.plot(marker="o", ax=ax[1], title="Brier score by lead day")
for a in ax:
    a.set_xlabel("Lead day")
plt.tight_layout()
plt.show()

# %%
# Reliability: do 30% forecasts come true ~30% of the time? (lead day 1, test set)
p_all, y_all = [], []
for t in TEST_T[::3]:
    p_all.append(predict_unet(t)[0][mask])
    y_all.append(targets(t)[0][mask])
p_all, y_all = np.concatenate(p_all), np.concatenate(y_all)
bins = np.linspace(0, 1, 11)
idx = np.clip(np.digitize(p_all, bins) - 1, 0, 9)
observed = [y_all[idx == i].mean() if (idx == i).any() else np.nan for i in range(10)]
plt.plot([0, 1], [0, 1], "k--", label="perfect")
plt.plot((bins[:-1] + bins[1:]) / 2, observed, "o-", label="U-Net, day +1")
plt.xlabel("Forecast probability")
plt.ylabel("Observed frequency")
plt.legend()
plt.title("Reliability, test 2024–2025")
plt.show()

# %%
# One example forecast during the 2024 dry season.
t = int((pd.Timestamp("2024-08-20") - DAY0).days)
pred = predict_unet(t)
fig, ax = plt.subplots(2, 4, figsize=(16, 7))
ax[0, 0].imshow(np.log1p(counts[t - 6 : t + 1].sum(0)), cmap="inferno", extent=(W, E, S, N))
ax[0, 0].set_title(f"Last 7 days of fires\n(up to {DATES[t].date()})")
for i, k in enumerate([0, 2, 6]):
    shown = np.where(mask, pred[k], np.nan)
    ax[0, i + 1].imshow(shown, cmap="YlOrRd", vmin=0, vmax=1, extent=(W, E, S, N))
    ax[0, i + 1].set_title(f"Forecast day +{k + 1}")
    ax[1, i + 1].imshow(fire[t + k + 1], cmap="gray_r", extent=(W, E, S, N))
    ax[1, i + 1].set_title(f"Observed {DATES[t + k + 1].date()}")
ax[1, 0].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Export for the backend

# %%
model.eval().cpu()
dummy = torch.from_numpy(features(int(TEST_T[0])))[None]
onnx_path = OUT_DIR / "fire_forecast_unet.onnx"
torch.onnx.export(
    model,
    dummy,
    onnx_path,
    input_names=["features"],
    output_names=["logits"],
    opset_version=17,
    dynamo=False,
)
np.save(OUT_DIR / "climatology.npy", climatology.astype(np.float16))
np.save(OUT_DIR / "forecast_mask.npy", mask)

meta = {
    "model": "fire_forecast_unet",
    "data": "NASA FIRMS VIIRS S-NPP 375 m, Brazil, type 0, confidence n/h",
    "train_period": [str(DATES[TRAIN_T[0]].date()), TRAIN_END],
    "bbox": BBOX,
    "resolution_deg": RES,
    "grid": [NY, NX],
    "history_days": HISTORY_DAYS,
    "long_window_days": LONG_WINDOW,
    "leads": LEADS,
    "channels": ["log1p_count_t-6..t"] * HISTORY_DAYS
    + ["log1p_sum_30d"]
    + [f"climatology_t+{k + 1}" for k in range(LEADS)]
    + ["doy_sin", "doy_cos", "mask", "lat", "lon"],
    "test_avg_precision": summary.mean().round(4).to_dict(),
    "test_brier": brier.mean().round(5).to_dict(),
}
(OUT_DIR / "fire_forecast_meta.json").write_text(json.dumps(meta, indent=2))

import onnxruntime as ort  # noqa: E402

session = ort.InferenceSession(str(onnx_path))
onnx_out = session.run(None, {"features": dummy.numpy()})[0]
torch_out = model(dummy).detach().numpy()
print("ONNX matches PyTorch:", np.allclose(onnx_out, torch_out, atol=1e-4))
print("Saved to", OUT_DIR.resolve())
