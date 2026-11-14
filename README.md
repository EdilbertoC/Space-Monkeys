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
| GET | `/fires/stream` | SSE: one `snapshot` event on connect, then a `pulse` event after every poll |

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
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated frontend origins |

## Project layout

```
app/
  main.py            FastAPI app, CORS, worker lifecycle
  config.py          settings
  api/               HTTP routes (system, live)
  realtime/
    firms.py         NASA FIRMS reader
    goes.py          GOES ABI fire product reader + fixed-grid projection
    detections.py    common table format for all sensors
    live.py          live state and change detection
    pulse.py         publish/subscribe for SSE clients
    worker.py        polling loop
  fusion/fuse.py     multi-sensor fusion
pipeline/            historical batch pipeline (needs the [science] extra)
  ingest.py          FIRMS archive zips -> Parquet detections table
  cube.py            detections -> fire-day cube (0.1° cell x UTC day x satellite)
  viewer.py          cube -> self-contained HTML viewer (dev tool), template in viewer_template.html
  training.py        cube -> training table for the VIIRS -> MODIS calibration, one per VIIRS satellite
  baselines.py       naive, ratio and LightGBM calibration, evaluated on the held-out years
tests/
```

## Historical pipeline

The batch side reads the FIRMS archive and never runs inside the API.

```bash
pip install -e ".[dev,science]"
# put the FIRMS archive zips in data/raw/firms/<region>/, e.g. data/raw/firms/pantanal/
python -m pipeline.ingest pantanal
```

`ingest` writes `data/processed/<region>/detections/year=YYYY/*.parquet` and `ingest_report.json` (rows read, dropped and kept per source). By default it:

- keeps confidence classes nominal and high for both sensors (MODIS 30–100, VIIRS `n`/`h`); `--min-confidence-class l` keeps everything;
- drops static land sources using one mask for every sensor: points within ~1 km of a location VIIRS labels as volcano or static source (FIRMS `type` 1 or 2). MODIS type 2 labels are not used for the mask because in the Pantanal they are scattered and peak in the fire season, like vegetation fires;
- keeps FIRMS `type` 3 ("offshore"): in the Pantanal these are fires over the seasonal wetland water mask.

```bash
python -m pipeline.cube pantanal
```

`cube` writes `data/processed/<region>/cube/year=YYYY/*.parquet` and `cube_report.json`. One row per 0.1° cell, UTC day and satellite (Terra, Aqua, SNPP, N20, N21) with at least one detection, so every row is a fire-day. Columns: `cell_id`, cell centre, `date`, `sensor`, `satellite`, `fire_day`, `n_pixels`, `n_overpasses`, `n_day`, `n_night`, `n_high`, `n_type2`, `frp_sum_mw`, `frp_max_mw`, `scan_mean_km`, `nrt`. Sensors are combined with OR on the cell-day, never summed. The report lists fire cell-days per year for MODIS, VIIRS, both and either, and P(MODIS | VIIRS).

The cube keeps each VIIRS satellite apart on purpose: in the Pantanal, P(MODIS | any VIIRS) drops from about 0.37 to 0.26 when NOAA-20 joins in 2018, while P(MODIS | S-NPP) shows no break. The drop comes from the growing VIIRS constellation, not from MODIS.

The cube is not yet clipped to land or to a country: the region box also covers sea and neighbouring countries.

To look at a cube (a development tool, not the product frontend):

```bash
python -m pipeline.viewer pantanal              # data/processed/pantanal/viewer.html, ~2 MB
python -m pipeline.viewer brazil --factor 10    # national cube needs coarser 1° cells, or --bbox
python -m http.server 8765 --directory data/processed   # then open http://localhost:8765/pantanal/viewer.html
```

The page shows MODIS and VIIRS fire-days side by side on a map for any year and month, a burning activity calendar (years × weeks) and the selected year against the 10th–90th percentile of the other years, for the whole region, one clicked cell or a drawn box. All counts are computed in the browser from the cube's fire cell-days.

```bash
python -m pipeline.training pantanal --satellite SNPP   # train 2012–2017, test 2018+
python -m pipeline.training pantanal --satellite N20    # train 2018–2022, test 2023+
```

`training` writes `data/processed/<region>/training/<SAT>/year=YYYY/*.parquet` and `<SAT>_report.json`. The target is `modis`: did MODIS (Terra or Aqua) detect fire in the 0.1° cell that UTC day. Decisions:

- **One table per VIIRS satellite.** Summed VIIRS is not stationary: NOAA-20 and NOAA-21 add pixels, so the same fire looks bigger after 2018, and the MODIS rate for a given VIIRS pixel count roughly halves from train to test. For S-NPP alone it is stable (Pantanal, 4–9 S-NPP pixels: P(MODIS) 0.60 in 2012–2017, 0.59 in 2018–2022, 0.57 in 2023–2026). S-NPP ends on 2026-11-02, so the record continues on NOAA-20.
- **Rows are picked by VIIRS only:** a cell-day is a row when the satellite saw fire in its 3×3 neighbourhood that day or the day before, the same rule that inference will use once MODIS is gone. Features never use MODIS.
- **Coverage gap:** 74–90% of MODIS fire-days per year fall inside the rows (Pantanal, single satellite). The rest are mostly small, short daytime fires (81% single MODIS pixel, median FRP 15 MW vs 37 MW overall, mostly Terra at ~10:30 local, before the VIIRS ~13:30 pass). No row-based model can recover them; the harmonized totals must correct for this gap (step 8).
- Only dates where MODIS and the satellite both have standard processing are used; NRT rows are left out.

```bash
python -m pipeline.baselines pantanal --satellite SNPP
python -m pipeline.baselines pantanal --satellite N20
```

`baselines` predicts, for every row, the probability that MODIS saw fire, and writes `report.json`, `lightgbm.txt`, `ratio.json` and `test_predictions.parquet` to `data/processed/<region>/models/<SAT>/`. Three approaches on the same test years:

- **naive:** "VIIRS saw fire, so MODIS would have", i.e. no harmonization;
- **ratio:** share of MODIS fire in the training years per month, for cells with and without their own VIIRS detection;
- **LightGBM:** all features, positive class weighted by n_negative / n_positive; the weight multiplies the predicted odds, so they are divided back to keep probabilities (and totals) honest. The last training year is used for early stopping.

The calendar shows totals, so the main metrics are on totals: bias of the summed probabilities against MODIS fire-days, WAPE of monthly regional totals and of cell-month totals. Pantanal, test years only (MODIS fire-days inside the rows):

| Satellite, test years | Model | Bias | Monthly WAPE | Cell-month WAPE | ROC AUC | Avg. precision |
|---|---|---|---|---|---|---|
| S-NPP, 2018–2026 | naive | +103.8% | 103.8% | 114.3% | 0.890 | 0.361 |
| | ratio | −13.0% | 20.3% | 75.1% | 0.905 | 0.432 |
| | **LightGBM** | **+0.1%** | **4.0%** | **48.4%** | **0.951** | **0.730** |
| NOAA-20, 2023–2026 | naive | +98.6% | 98.6% | 107.6% | 0.896 | 0.379 |
| | ratio | −7.7% | 22.6% | 72.6% | 0.910 | 0.457 |
| | **LightGBM** | **−2.8%** | **4.8%** | **45.5%** | **0.955** | **0.750** |

Yearly LightGBM totals stay within about ±5% of MODIS in every test year, including 2025–2026, so no strong MODIS drift shows up yet. Most of the LightGBM gain comes from the fire radiative power seen by VIIRS (`v_frp_sum`, ~70% of the gain): MODIS misses weak fires, and FRP measures how strong a fire is. Next: the U-Net (step 7) must beat these numbers on the same test years to be used.

`data/` is not committed.

## Data sources

Every dataset this project reads, with its citation.

### NASA FIRMS (Fire Information for Resource Management System)

- FIRMS: https://firms.modaps.eosdis.nasa.gov/ — area API: https://firms.modaps.eosdis.nasa.gov/api/area/
- **VIIRS S-NPP 375 m active fire, NRT** (`VNP14IMGTDL_NRT`, FIRMS source `VIIRS_SNPP_NRT`). NASA VIIRS Land Science Team, NASA LANCE MODIS at the MODAPS. https://doi.org/10.5067/FIRMS/VIIRS/VNP14IMGT_NRT.002
- **VIIRS NOAA-20 (JPSS-1) 375 m active fire, NRT** (`VJ114IMGTDL_NRT`, source `VIIRS_NOAA20_NRT`). NASA VIIRS Land Science Team, NASA LANCE MODIS at the MODAPS. https://doi.org/10.5067/FIRMS/VIIRS/VJ114IMGT_NRT.002
- **VIIRS NOAA-21 (JPSS-2) 375 m active fire, NRT** (`VJ214IMGTDL_NRT`, source `VIIRS_NOAA21_NRT`). NASA VIIRS Land Science Team (2024), NASA LANCE MODIS at the MODAPS. https://doi.org/10.5067/VIIRS/VJ214IMG_NRT.002
- **MODIS Collection 6.1 active fire, NRT** (`MCD14DL`, source `MODIS_NRT`), Terra and Aqua, 1 km. https://doi.org/10.5067/FIRMS/MODIS/MCD14DL.NRT.0061
- FIRMS FAQ (attribute meanings, e.g. `scan`/`track` pixel sizes): https://www.earthdata.nasa.gov/data/tools/firms/faq

Historical record (pipeline), from the FIRMS archive download (https://firms.modaps.eosdis.nasa.gov/download/), CSV, one request per sensor:

- **MODIS Collection 6.1 standard active fire locations** (`MCD14ML`, file `fire_archive_M-C61_*`), Terra (from 2000-11-01) and Aqua (from 2002-07-04), processed by the University of Maryland with a ~3-month lag and distributed by FIRMS.
- **VIIRS 375 m standard active fire product, Collection 2**: S-NPP (`VNP14IMGTML`, file `fire_archive_SV-C2_*`, from 2012-01-20), NOAA-20 (file `fire_archive_J1V-C2_*`, from 2018-04-01) and NOAA-21 (file `fire_archive_J2V-C2_*`, from 2023-03-01), processed by the University of Maryland and distributed by FIRMS.
- The months the standard archive has not reached yet come in the same download as NRT files (`fire_nrt_*`, the NRT products listed above) and are flagged `processing = NRT`.
- Product names and the known MODIS archive gaps (late June to early July 2001, parts of 2002, mid-August 2007, 21–22 April 2009) are from the FIRMS download Readme: https://firms.modaps.eosdis.nasa.gov/download/Readme.txt
- Attribute fields (including `type`: 0 vegetation fire, 1 volcano, 2 other static land source, 3 offshore): https://www.earthdata.nasa.gov/data/tools/firms/active-fire-data-attributes-modis-viirs
- Pilot area: box west −60, south −22.5, east −54.5, north −15 (Pantanal), 2000-11-01 to 2026-11-14.
- National area: box west −74.5, south −34.5, east −28.5, north 6 (Brazil and surroundings), same dates (requests 848644–848647).

### Basemap (cube viewer only)

- OpenStreetMap standard tiles, © OpenStreetMap contributors, data under the Open Database License (ODbL): https://www.openstreetmap.org/copyright. Used lightly for a development viewer, following the tile usage policy: https://operations.osmfoundation.org/policies/tiles/

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
| FastAPI, Pydantic, pydantic-settings, pytest, Ruff | MIT |
| Uvicorn, httpx, NumPy, pandas, SciPy, h5netcdf, h5py | BSD-3-Clause |
| xarray | Apache-2.0 |
| DuckDB, LightGBM (pipeline, optional `[science]` extra) | MIT |
| PyArrow (pipeline) | Apache-2.0 |
| scikit-learn (pipeline metrics) | BSD-3-Clause |
| PyTorch (pipeline) | BSD-3-Clause |
| Leaflet 1.9.4 (cube viewer, loaded from cdnjs) | BSD-2-Clause |
| Python | PSF License |

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
- **What it did:** wrote most of the backend code, tests and documentation in this repository from the team's design and instructions, looked up data access details (FIRMS API, GOES file layout, product DOIs), and checked the code against real GOES-19 data.
- **Review:** every SI-assisted file is reviewed and run by the team. Each one starts with the line `# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team.`
- **Not used for:** data values. All detections come straight from the public NASA and NOAA sources above; nothing is generated or simulated.
