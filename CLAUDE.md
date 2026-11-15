# CLAUDE.md — Space Monkeys

Context for Claude Code. Read this before working on the repository.

## The challenge

NASA Space Apps 2026: "Harmonization of MODIS and VIIRS Hot Spots". Build a web app that harmonizes active fire hotspot records into a burning activity calendar, so responders, scientists and land managers can examine historical fire patterns, unusual conditions and critical periods within a selected area of interest.

## The core problem (what we decided)

- A hotspot is a pixel flagged as anomalously hot during a satellite overpass. It is not a fire count and not burned area.
- MODIS (Terra/Aqua, ~1 km pixel) covers 2000 to now. VIIRS (375 m pixel) starts in 2012 (S-NPP), then NOAA-20 (2018) and NOAA-21. Both sensors overlap from 2012 to now.
- VIIRS sees small fires MODIS misses and splits one fire into several pixels. Both effects inflate counts, so the naive record shows a step in 2012 that looks like a trend.
- MODIS is ending (NASA projects end of science data in 2027) and its orbit has drifted in recent years. The harmonization exists to carry the record forward on VIIRS, expressed in MODIS-equivalent units. The 2012-to-end-of-MODIS overlap is the only window to learn that bridge.
- Questions the product answers for an area: seasonal pattern (start, peak, end), anomaly (is this period above normal?), long-term trend. Trend is the most sensitive to harmonization errors.
- The differentiator is the science, not the UI.

## Method

- Grid: 0.1° cells, daily (UTC). Pilot region: Pantanal (2020 is the validation season).
- Base unit: fire-day per cell (1 if at least one detection), not raw pixel counts. Combining sensors after 2012 is a logical OR, never a sum.
- Calibration target: "did MODIS detect fire in this cell on this day?" (0/1), predicted from what VIIRS saw (pixel count, FRP, day/night) plus context (month, region, neighbouring cells, previous days).
- Train only on cell-days where at least one sensor detected fire. Weight the positive class in the loss.
- Temporal split only: train 2012–2017, test 2018 onward. Never a random split. If test error is much worse than train, suspect MODIS orbital drift first.
- Models, compared on the same test set and metric:
  1. Ratio per region and month (simplest baseline)
  2. LightGBM with neighbourhood and lag features (must get the same information the U-Net sees, so the comparison is architecture vs architecture)
  3. Small 2D U-Net, 5–7 days stacked as channels, patches sampled where fire exists, weighted BCE or focal loss. This is our bet; if it does not beat LightGBM, we ship LightGBM and report the result honestly.
- Inference runs once in batch and writes a harmonized cube to Parquet. The API never runs a model.

## Plan

Critical path: 1 → 2 → 3 → 5 → 6 → 8 → 9

1. Request the FIRMS archive (standard processing, not NRT) for the pilot region: MODIS C6.1, VIIRS S-NPP, NOAA-20, NOAA-21. Cover the recent gap with NRT, flagged.
2. Ingest CSV → Parquet using the common schema in `app/realtime/detections.py`. Drop static sources (`type` column) and apply per-sensor confidence thresholds.
3. Build the cube in DuckDB: cell × day × sensor → fire_day, n_pixels, frp_sum, day/night counts.
4. Explore: plot MODIS-only, VIIRS-only and naive series; measure P(MODIS | VIIRS) by month.
5. Build the training table (features above) with the temporal split.
6. Baselines: ratio, then LightGBM.
7. U-Net (can run in parallel after step 5).
8. Run the best model over the full VIIRS record; validate on Pantanal 2020; before/after plot.
9. Analytics as functions of (area, date range): climatology, anomalies (percentile or z-score), season start/peak/end.
10. New FastAPI endpoints reading the Parquet cube via DuckDB; calendar in the frontend.
11. Finish `docs/methodology.md` (written along the way, in plain English).

## Repository state

- `app/` is the real-time side (GOES + FIRMS NRT worker, multi-sensor fusion, SSE). It works and is frozen: no new effort goes there. The historical pipeline is the priority.
- The fusion's combined confidence `1 − ∏(1 − c)` treats sensor confidences as calibrated independent probabilities. They are not; document it as a heuristic.
- Put the batch pipeline in a new `pipeline/` package, independent from `app/`.
- Add `duckdb`, `pyarrow`, `lightgbm`, `torch` as an optional extra (e.g. `[science]`) so the API Docker image stays small.

## Rules

- All code, docs and materials in English (hackathon compliance).
- Every SI-assisted file starts with: `# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.`
- Never commit `FIRMS_MAP_KEY`; it lives in `.env`.
- Never generate or simulate data values; everything comes from NASA/NOAA sources.
- Cite every new dataset and library in the README's Data sources / Software sections.
