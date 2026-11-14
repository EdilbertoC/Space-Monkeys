# Space Monkeys — Backend

NASA Space Apps Challenge: a **burning activity calendar** built from satellite-detected active fire hotspots.

The goal is to harmonize records from different sensors (MODIS, VIIRS, GOES ABI) into a consistent time series and provide a web application that lets early warning and emergency responders, scientists, and land managers examine historical fire patterns, unusual conditions, and critical periods within a selected area of interest.

This repository holds the API, the real-time worker and the multi-sensor fusion. The frontend lives in [Space-Monkeys-Front](https://github.com/ryangustav/Space-Monkeys-Front).

## What runs today

| Part | What it does |
| --- | --- |
| Real-time worker | Every 10 minutes, pulls the newest GOES-East fire scan and the latest NASA FIRMS near-real-time hotspots (VIIRS S-NPP, NOAA-20, NOAA-21, MODIS) for the monitored area |
| Multi-sensor fusion | Merges detections of the same fire from different sensors into one event with a better position, an uncertainty radius and a combined confidence |
| Pulse stream | Pushes new, updated and expired fires to the frontend over Server-Sent Events after each poll |
| API | FastAPI, GeoJSON responses, interactive docs at `/docs` |

Historical harmonization (MODIS/VIIRS calendar, climatology, anomalies) is in progress.

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # then set FIRMS_MAP_KEY
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs. Without a FIRMS key the worker still runs on GOES data alone.

Get a free FIRMS key at https://firms.modaps.eosdis.nasa.gov/api/map_key/ and never commit it.

```bash
pytest          # tests
ruff check .    # lint
ruff format .   # format
```

With Docker:

```bash
docker build -t space-monkeys-back .
docker run -p 8000:8000 --env-file .env space-monkeys-back
```

## API

| Method | Path | Returns |
| --- | --- | --- |
| GET | `/health` | `{"status": "ok"}` |
| GET | `/status` | Worker runs, per-source last success/error/count, live totals |
| GET | `/fires/live?bbox=w,s,e,n&min_confidence=0.5` | Fused fire events (GeoJSON) |
| GET | `/detections/recent?bbox=w,s,e,n&sensor=VIIRS` | Raw detections from every sensor (GeoJSON) |
| GET | `/fires/stream` | SSE: one `snapshot` event on connect, then a `pulse` event after every poll and a `forecast` event when the forecast is re-anchored |
| GET | `/forecast?min_p=0.05&bbox=w,s,e,n` | 7-day fire probability per 0.25° cell (lead 1 = today), with GOES anchoring |
| GET | `/forecast/model` | Forecast model card: data, inputs, test scores, status |

Fire event properties: `fire_id`, `uncertainty_km` (~95% radius), `confidence` (0–1), `sensors`, `n_detections`, `first_seen`, `last_seen`, `frp_max_mw`.

A `pulse` event carries `at`, `new_detections`, `total_fires`, `new` and `updated` (GeoJSON) and `removed` (fire ids). Frontend usage:

```ts
const stream = new EventSource(`${API_URL}/fires/stream`);
stream.addEventListener("snapshot", (e) => setFires(JSON.parse(e.data)));
stream.addEventListener("pulse", (e) => applyPulse(JSON.parse(e.data)));
```

## How the fusion works

A hotspot is the centre of a satellite pixel, so the fire lies somewhere inside that pixel. The sensors differ a lot in pixel size: VIIRS ~375 m, MODIS ~1 km, GOES ABI ~2 km at nadir and ~3–5 km over Brazil. GOES sees the fire within minutes; VIIRS sees it a few times a day but far more precisely.

1. **Cluster.** Two detections belong to the same fire when they are within `FUSION_RADIUS_KM` plus half of each pixel's larger side, and within `FUSION_WINDOW_HOURS` of each other.
2. **Position.** Inverse-variance weighted centroid: a point spread uniformly over a pixel of size scan × track has variance (scan² + track²) / 12, so one VIIRS pixel outweighs a GOES pixel about 30 to 1. Each weight is also scaled by the detection's confidence.
3. **Uncertainty.** Twice the larger of the combined pixel error and the weighted spread of the detections.
4. **Confidence.** 1 − ∏(1 − best confidence per sensor): independent sensors that agree raise confidence.

The gain in position comes mainly from VIIRS. The main gains from fusion are earlier detection (GOES) and higher confidence (agreement).

GOES pixel positions are computed from the GOES-R fixed-grid projection (PUG Vol. 5, section 4.2.8). The real ground size of each pixel is measured from its corners.

## Fire forecast model

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ryangustav/Space-Monkeys/blob/main/notebooks/fire_forecast_base.ipynb)

[`notebooks/fire_forecast_base.ipynb`](notebooks/fire_forecast_base.ipynb) trains the base fire forecast: for every 0.25° cell of Brazil, the probability of at least one fire detection on each of the next 7 days.

- **Data:** NASA FIRMS VIIRS S-NPP yearly archive, 2012–2025 (presumed vegetation fires, nominal/high confidence), downloaded inside the notebook.
- **Two training regions:** `REGION = "brazil"` learns from Brazil only; `REGION = "world"` learns from every country. Every run is scored on the same Brazil test (same cells, days and Brazil-only VIIRS truth), and the last section of the notebook compares all runs.
- **Optional MODIS history (`USE_MODIS = True`):** adds MODIS C6.1 2000–2016. VIIRS sees about twice as many fire cell-days as MODIS, so the two are never mixed: each sensor has its own input channels and an "available" flag, and the network has one output head per sensor. The 2012–2016 overlap teaches how they relate, and MODIS inputs are randomly hidden there so the model runs on VIIRS alone from 2017 on and in production.
- **Model:** small 2D U-Net (~1.9 M parameters). Inputs are image channels: the last 7 days of fire maps, the last 30 days, the climatology of each target day, seasonality and position.
- **Evaluation:** time-based split (train 2012–2022, validation 2023, test 2024–2025), scored with average precision and Brier score against two references: climatology and persistence.
- **Output:** `fire_forecast_<region>.onnx` + `fire_forecast_<region>.json`, run by the backend with ONNX Runtime.

### Forecast in the API

The backend serves the Brazil model from [`models/fire_forecast_brazil/`](models/fire_forecast_brazil/) (ONNX + climatology + mask + [`meta.json`](models/fire_forecast_brazil/meta.json), the model card). Test scores on Brazil 2024–2025, never seen in training:

| Model | Average precision ↑ | Brier score ↓ |
| --- | --- | --- |
| Climatology | 0.303 | 0.0468 |
| Persistence | 0.154 | 0.0497 |
| **U-Net (Brazil, VIIRS)** | **0.396** | **0.0434** |
| U-Net (world + MODIS) | 0.379 | 0.0438 |

How a forecast is made:

1. **Base forecast** (every `FORECAST_REFRESH_HOURS`): the last 30 days of VIIRS S-NPP NRT detections from FIRMS are gridded exactly like the training data and fed to the U-Net. It is issued for the last complete UTC day, so lead 1 is today.
2. **GOES real-time anchor** (every pulse): a cell with a live fused fire seen today is raised to at least *confidence × P(fire persists j days)* for today and the following days. The persistence probabilities were measured on the training years ([`scripts/persistence_table.py`](scripts/persistence_table.py)). The anchor never lowers the model's forecast and is never fed into the network, which never saw GOES data. Anchored cells are flagged `anchored: true`.

The NRT feed has no `type` column, so non-vegetation static heat sources cannot be removed at forecast time; the forecast mask limits the model to cells that burned during training.

Run the training notebook in Colab with a T4 GPU (Runtime → Change runtime type). Edit `notebooks/fire_forecast_base.py`, the reviewable source, then rebuild the notebook with `python notebooks/build_notebooks.py`.

## Configuration

All settings come from environment variables or `.env`; see [.env.example](.env.example).

| Variable | Default | Meaning |
| --- | --- | --- |
| `FIRMS_MAP_KEY` | empty | NASA FIRMS API key |
| `FIRMS_SOURCES` | all VIIRS NRT + `MODIS_NRT` | Comma-separated FIRMS sources |
| `FIRMS_DAY_RANGE` | `1` | Days per FIRMS request (1–5) |
| `GOES_BUCKET` | `noaa-goes19` | GOES-East bucket on AWS |
| `GOES_PRODUCT` | `ABI-L2-FDCF` | Full Disk fire product |
| `GOES_MIN_CONFIDENCE` | `0.5` | Minimum normalized confidence for GOES pixels |
| `AOI_BBOX` | `-74.5,-34.5,-28.5,6` | Monitored area (Brazil): west,south,east,north |
| `ENABLE_WORKER` | `true` | Run the real-time worker inside the API process |
| `POLL_INTERVAL_SECONDS` | `600` | Time between polls |
| `LIVE_RETENTION_HOURS` | `48` | How long live detections are kept |
| `FUSION_RADIUS_KM` | `2` | Base linking distance |
| `FUSION_WINDOW_HOURS` | `12` | Linking time window |
| `ENABLE_FORECAST` | `true` | Load the forecast model and serve `/forecast` |
| `FORECAST_MODEL_DIR` | `models/fire_forecast_brazil` | Folder with `model.onnx`, `climatology.npy`, `mask.npy`, `meta.json` |
| `FORECAST_REFRESH_HOURS` | `6` | How often the base forecast is recomputed (needs `FIRMS_MAP_KEY`) |
| `FORECAST_ANCHOR_HOURS` | `12` | How recent a live fire must be to anchor the forecast |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated frontend origins |

## Project layout

```
app/
  main.py            FastAPI app, CORS, worker lifecycle
  config.py          settings
  api/               HTTP routes (system, live, forecast)
  realtime/
    firms.py         NASA FIRMS reader
    goes.py          GOES ABI fire product reader + fixed-grid projection
    detections.py    common table format for all sensors
    live.py          live state and change detection
    pulse.py         publish/subscribe for SSE clients
    worker.py        polling loop
  fusion/fuse.py     multi-sensor fusion
  forecast/          fire forecast: ONNX model, FIRMS history, GOES anchor
models/             trained forecast model (ONNX, climatology, mask, model card)
notebooks/          Colab notebooks (fire forecast model)
scripts/            one-off data scripts (persistence table)
tests/
```

## Data sources

Every dataset this project reads, with its citation.

### NASA FIRMS (Fire Information for Resource Management System)

- FIRMS: https://firms.modaps.eosdis.nasa.gov/ — area API: https://firms.modaps.eosdis.nasa.gov/api/area/
- **VIIRS S-NPP 375 m active fire, NRT** (`VNP14IMGTDL_NRT`, FIRMS source `VIIRS_SNPP_NRT`). NASA VIIRS Land Science Team, NASA LANCE MODIS at the MODAPS. https://doi.org/10.5067/FIRMS/VIIRS/VNP14IMGT_NRT.002
- **VIIRS NOAA-20 (JPSS-1) 375 m active fire, NRT** (`VJ114IMGTDL_NRT`, source `VIIRS_NOAA20_NRT`). NASA VIIRS Land Science Team, NASA LANCE MODIS at the MODAPS. https://doi.org/10.5067/FIRMS/VIIRS/VJ114IMGT_NRT.002
- **VIIRS NOAA-21 (JPSS-2) 375 m active fire, NRT** (`VJ214IMGTDL_NRT`, source `VIIRS_NOAA21_NRT`). NASA VIIRS Land Science Team (2024), NASA LANCE MODIS at the MODAPS. https://doi.org/10.5067/VIIRS/VJ214IMG_NRT.002
- **MODIS Collection 6.1 active fire, NRT** (`MCD14DL`, source `MODIS_NRT`), Terra and Aqua, 1 km. https://doi.org/10.5067/FIRMS/MODIS/MCD14DL.NRT.0061
- **FIRMS yearly country archive**, used to train the forecast model: VIIRS S-NPP 2012–2025 and MODIS C6.1 2000–2016. Brazil `https://firms.modaps.eosdis.nasa.gov/data/country/{viirs-snpp|modis}/{year}/{sensor}_{year}_Brazil.csv` and all countries `https://firms.modaps.eosdis.nasa.gov/data/country/zips/{sensor}_{year}_all_countries.zip`
- FIRMS FAQ (attribute meanings, e.g. `scan`/`track` pixel sizes): https://www.earthdata.nasa.gov/data/tools/firms/faq

We acknowledge the use of data and imagery from LANCE FIRMS operated by NASA's Earth Science Data and Information System (ESDIS), with funding provided by NASA Headquarters.

License: NASA Earth science data are openly shared without restriction under the [EOSDIS Data Use Policy](https://www.earthdata.nasa.gov/engage/open-data-services-software-policies/data-use-guidance).

### NOAA GOES-R Series

- **GOES-19 (GOES-East) ABI L2+ Fire/Hot Spot Characterization, Full Disk** (`ABI-L2-FDCF`), every 10 min, ~2 km at nadir. NOAA / GOES-R Series Program.
- Access: NOAA Open Data Dissemination (NODD) on AWS, bucket `noaa-goes19`. https://registry.opendata.aws/noaa-goes/
- GOES-R Product Definition and User's Guide (PUG), Volume 5: Level 2+ Products (fire mask codes, fixed-grid projection), and the ABI Fire/Hot Spot Characterization ATBD: https://www.goes-r.gov/resources/docs.html
- GOES-19 ABI L2 Fire/Hot Spot Provisional maturity ReadMe (known issues): https://noaasis.noaa.gov/pdf/ps-pvr/goes19/ABI/Fires%20and%20Hot%20Spots%20Characterization/Provisional/GOES-19_ABI_L2_FireHotSpot_Provisional_ReadMe.pdf
- Band and mask-code summary (Google Earth Engine catalog): https://developers.google.com/earth-engine/datasets/catalog/NOAA_GOES_19_FDCF

License: NOAA data distributed through NODD are open and free to use without restriction (see the AWS registry entry above).

### Algorithm references

- Schroeder, W., Oliva, P., Giglio, L., Csiszar, I. A. (2014). The New VIIRS 375 m active fire detection data product: Algorithm description and initial assessment. *Remote Sensing of Environment*, 143, 85–96. https://doi.org/10.1016/j.rse.2013.12.008
- Giglio, L., Schroeder, W., Justice, C. O. (2016). The collection 6 MODIS active fire detection algorithm and fire products. *Remote Sensing of Environment*, 178, 31–41. https://doi.org/10.1016/j.rse.2016.02.054
- Giglio, L., Descloitres, J., Justice, C. O., Kaufman, Y. J. (2003). An enhanced contextual fire detection algorithm for MODIS. *Remote Sensing of Environment*, 87, 273–282. https://doi.org/10.1016/S0034-4257(03)00184-6

### Software

All open source; versions in [pyproject.toml](pyproject.toml).

| Library | License |
| --- | --- |
| FastAPI, Pydantic, pydantic-settings, pytest, Ruff, ONNX Runtime | MIT |
| Uvicorn, httpx, NumPy, pandas, SciPy, h5netcdf, h5py, PyTorch, scikit-learn | BSD-3-Clause |
| xarray, ONNX, onnxscript, Requests | Apache-2.0 |
| Matplotlib | Matplotlib License (PSF-based) |
| Python | PSF License |

The forecast notebook runs on Google Colab (hosted Jupyter runtime).

## NASA Space Apps Challenge 2026 compliance

- **NASA data:** the project's core uses NASA FIRMS (VIIRS and MODIS active fire data). NOAA GOES data complement it.
- **References:** every dataset, document and library used is listed above, with its license.
- **License:** our original code is released under the [Apache License 2.0](LICENSE).
- **Language:** all code, documentation and project materials are in English.
- **No personal data** and no images, voices or likenesses of anyone under 18.
- **No NASA logos or branding** in generated content.

### Use of Super Intelligence (SI)

The team designed the solution: a burning activity calendar from harmonized hotspots, a live map refreshed on a pulse of real-time data, and refinement of hotspot positions after harmonization.

SI was used as a coding assistant:

- **Tool:** Claude (Anthropic), a large language model, used through Claude Code.
- **What it did:** wrote most of the backend code, the forecast notebook, tests and documentation in this repository from the team's design and instructions, looked up data access details (FIRMS API, GOES file layout, product DOIs), and checked the code against real GOES-19 data.
- **Review:** every SI-assisted file is reviewed and run by the team. Each one starts with the line `# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team.`
- **Not used for:** data values. All detections come straight from the public NASA and NOAA sources above; nothing is generated or simulated.
