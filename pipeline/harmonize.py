# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Step 8: the harmonized record, VIIRS expressed in MODIS-equivalent fire cell-days.

The reference is MODIS while its orbits were stable. Terra and Aqua have drifted
since 2022: in the FIRMS detections over Brazil the daytime overpass moved from
about 10:25 to 9:14 local solar time (Terra) and from 13:40 to 14:56 (Aqua) by
2025, while VIIRS stayed at 13:37. MODIS of those years is a different
instrument, so everything that defines "MODIS-equivalent" (training, calibration,
gap, interval) uses only the years up to REFERENCE_LAST_YEAR, without the days a
MODIS satellite was not observing (pipeline.cube.observation_gaps).

For one VIIRS satellite:

1. Cross-fitting by year blocks over the reference years. Each block is
   predicted by a LightGBM trained on the other blocks only (pipeline.baselines,
   weighted, early stopping on its last full training year), so every reference
   cell-day gets an out-of-sample estimate. The years after the reference are
   predicted by one final model trained on every reference year: a forward
   prediction, as it will be after MODIS ends.
2. Calibration per biome: an isotonic map from the out-of-fold probability to the
   observed MODIS rate, one per IBGE biome (one national map left the Pantanal
   6% high and Caatinga/Pampa 11-13% low). It is fitted on the reference
   out-of-fold rows, the same rows the reference scores use: their bias is close
   to zero by construction and says nothing; the monthly error is the score.
3. Coverage gap: MODIS fire-days with no VIIRS fire nearby can never be rows.
   Per biome and calendar month, g = all MODIS fire-days / MODIS fire-days inside
   the rows (reference years); each calibrated probability is multiplied by g,
   so area totals include the gap. The gap cannot be placed on the map: per cell
   it is spread over the cells where VIIRS saw activity.
4. Uncertainty: per biome, the 5th and 95th percentiles of the relative error of
   reference monthly totals (actual / estimate - 1, months with at least 20
   estimated fire-days) give a 90% interval: [est * (1 + q05), est * (1 + q95)].

After the reference years the report compares the estimate with observed MODIS
(`after_reference`). That difference is not a model error: it mixes the forward
error with the change in MODIS itself.

5. Beyond the training table: the training table stops on the last day with
   standard processing for both MODIS and the satellite. The days after it (the
   satellite's NRT days now, every day once MODIS ends) come from the application
   table (pipeline.training --application), picked and described from VIIRS alone,
   and get the final model with the same calibration and gap (`applied` in the
   report; `--apply-only` redoes just this step when new VIIRS days arrive).

Outputs (data/processed/<region>/harmonized/):
- <SAT>/year=YYYY/*.parquet: cell_id, date, biome, modis (observed 0/1), equiv
  (MODIS-equivalent expected fire-days, after calibration and gap); applied.parquet
  holds the days after the training table (modis NULL)
- <SAT>_monthly.parquet: per biome and month: MODIS observed, VIIRS naive
  (fire cell-days seen by the satellite), estimate, interval
- <SAT>_model.json: folds, calibrators, gap factors, intervals, scores on the
  reference years and the comparison with MODIS after them
- models in data/processed/<region>/models/<SAT>/folds/

The harmonized series itself is MODIS until the switch date chosen in
pipeline.serving (by default while MODIS exists) and this estimate after it.

Usage::

    python -m pipeline.harmonize brazil --satellite SNPP
    python -m pipeline.harmonize brazil --satellite N20
    python -m pipeline.harmonize brazil --satellite SNPP --apply-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from pipeline import baselines, crossbiome
from pipeline.cube import observation_gaps, sql_dates
from pipeline.training import FEATURES, SATELLITES, TARGET

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"

# Last year MODIS is used as the reference: Terra and Aqua orbits drift from 2022.
REFERENCE_LAST_YEAR = 2021
# Year blocks of the reference years. NOAA-21 (2023) never overlapped the reference
# MODIS, so it cannot be calibrated this way.
FOLDS = {
    "SNPP": [(2012, 2014), (2015, 2017), (2018, 2019), (2020, 2021)],
    "N20": [(2018, 2019), (2020, 2021)],
}
FINAL = "final"  # the model trained on every reference year, for the years after
MIN_FULL_YEAR_DAYS = 330
MIN_MONTH_ESTIMATE = 20.0  # months below this are too noisy for the interval
EQUIV_FLOOR = 1e-3  # rows below this expected value are not written
CALIBRATION_MAX_ROWS = 8_000_000  # per biome, hash-sampled above this
DEFAULT_THREADS = max(1, (os.cpu_count() or 4) - 4)  # leave cores for the rest of the PC


def _table(data_dir: Path, region: str, satellite: str) -> str:
    return crossbiome._table(data_dir, region, satellite)


def full_years(data_dir: Path, region: str, satellite: str) -> list[int]:
    rows = (
        duckdb.connect()
        .execute(
            f"SELECT year, count(DISTINCT date) FROM {_table(data_dir, region, satellite)} "
            "GROUP BY year"
        )
        .fetchall()
    )
    return sorted(y for y, days in rows if days >= MIN_FULL_YEAR_DAYS)


def fold_of(year: int, folds: list[tuple[int, int]]) -> str:
    """Name of the model that predicts a year: its fold, or FINAL after the reference."""
    for k, (a, b) in enumerate(folds):
        if a <= year <= b:
            return str(k)
    if year > REFERENCE_LAST_YEAR:
        return FINAL
    raise ValueError(f"year {year} is in no fold")


def modis_gap_days(data_dir: Path, region: str) -> list[date]:
    """Days Terra or Aqua was not observing: MODIS counts are low, not a reference."""
    gaps = observation_gaps(region, data_dir)
    return sorted(set(gaps.get("Terra", [])) | set(gaps.get("Aqua", [])))


def _fold_dir(data_dir: Path, region: str, satellite: str) -> Path:
    return data_dir / "processed" / region / "models" / satellite / "folds"


def fit_folds(
    data_dir: Path,
    region: str,
    satellite: str,
    folds: list[tuple[int, int]],
    retrain: bool = False,
    exclude_dates: list[date] | None = None,
) -> list[dict]:
    """Train one model per fold on every other block, and the final model on every
    reference year; save and describe them.

    Models already saved for the same held-out years and reference are reused unless
    `retrain`, so an interrupted run resumes where it stopped.
    """
    out = _fold_dir(data_dir, region, satellite)
    full = [y for y in full_years(data_dir, region, satellite) if y <= REFERENCE_LAST_YEAR]
    plan: list[tuple[str, tuple[int, int] | None]] = [(str(k), f) for k, f in enumerate(folds)]
    plan.append((FINAL, None))

    def saved(name: str, held: tuple[int, int] | None) -> dict | None:
        path = out / name / "meta.json"
        if retrain or not path.exists():
            return None
        meta = json.loads(path.read_text(encoding="utf-8"))
        same = meta.get("held_out_years") == (list(held) if held else None)
        return meta if same and meta.get("reference_last_year") == REFERENCE_LAST_YEAR else None

    if out.exists():  # models of another fold layout
        for d in out.iterdir():
            if d.is_dir() and d.name not in {name for name, _ in plan}:
                shutil.rmtree(d)
    sample = None
    described = []
    for name, held in plan:
        meta = saved(name, held)
        if meta is None:
            if sample is None:
                sample = crossbiome.load_training_sample(
                    data_dir,
                    region,
                    satellite,
                    splits=("train", "test"),
                    last_year=REFERENCE_LAST_YEAR,
                    exclude_dates=exclude_dates,
                )
            lo, hi = held if held else (None, None)
            years = [y for y in full if held is None or y < lo or y > hi]
            train = (
                sample if held is None else sample[(sample["year"] < lo) | (sample["year"] > hi)]
            )
            valid_year = max(years)
            booster, info = baselines.fit_lightgbm(train, valid_year=valid_year, weight="w")
            baselines.save_lightgbm(out / name, booster, info)
            meta = {
                "fold": name,
                "held_out_years": list(held) if held else None,
                "reference_last_year": REFERENCE_LAST_YEAR,
                "valid_year": valid_year,
                "rounds": info["rounds"],
                "positive_weight": info["positive_weight"],
                "train_rows_sampled": len(train),
            }
            (out / name / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            del booster, train
        described.append(meta)
    return described


def predict_out_of_fold(
    data_dir: Path,
    region: str,
    satellite: str,
    folds: list[tuple[int, int]],
    raw_dir: Path,
    threads: int = 0,
) -> list[int]:
    """Predict every row (Brazil cells) with the model of the fold that held its year
    out (the final model after the reference years), one year at a time, written to
    raw_dir/year=YYYY/raw.parquet."""
    models = {
        name: baselines.load_lightgbm(_fold_dir(data_dir, region, satellite) / name)
        for name in [*(str(k) for k in range(len(folds))), FINAL]
    }
    con = duckdb.connect()
    years = [
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT year FROM {_table(data_dir, region, satellite)} ORDER BY 1"
        ).fetchall()
    ]
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    for yr in years:
        df = con.execute(f"""
            SELECT cell_id, date, year, biome, {TARGET}, {", ".join(FEATURES)}
            FROM {_table(data_dir, region, satellite)}
            WHERE year = {yr} AND biome IS NOT NULL
        """).fetchdf()
        booster, w, _ = models[fold_of(yr, folds)]
        raw = baselines.unweight(booster.predict(df[FEATURES], num_threads=threads), w)
        part = raw_dir / f"year={yr}"
        part.mkdir(parents=True)
        pd.DataFrame(
            {
                "cell_id": df["cell_id"].to_numpy(np.int64),
                "date": pd.to_datetime(df["date"]).to_numpy(),
                "biome": df["biome"].astype(str).to_numpy(),
                "y": df[TARGET].to_numpy(np.int8),
                "raw": raw.astype(np.float32),
            }
        ).to_parquet(part / "raw.parquet", index=False)
        del df, raw
    return years


def calibrate_by_biome(oof: pd.DataFrame) -> tuple[np.ndarray, dict]:
    p = np.empty(len(oof), dtype=np.float32)
    calibrators = {}
    for biome, idx in oof.groupby("biome", observed=True).indices.items():
        cal = baselines.fit_calibration(oof["raw"].to_numpy()[idx], oof["y"].to_numpy()[idx])
        calibrators[str(biome)] = cal
        p[idx] = baselines.apply_calibration(oof["raw"].to_numpy()[idx], cal)
    return p, calibrators


def fit_biome_calibrators(
    raw_dir: Path,
    max_rows: int = CALIBRATION_MAX_ROWS,
    last_year: int | None = None,
    exclude_dates: list[date] | None = None,
) -> dict:
    """One isotonic calibrator per biome from the out-of-fold rows on disk, optionally
    only years up to `last_year` and without the days in `exclude_dates`. Biomes with
    more rows than max_rows are fitted on a fixed hash sample of them."""
    con = duckdb.connect()
    src = f"read_parquet('{raw_dir.as_posix()}/**/*.parquet', hive_partitioning = true)"
    where = f"CAST(date AS DATE) NOT IN ({sql_dates(exclude_dates or [])})"
    if last_year is not None:
        where += f" AND year <= {int(last_year)}"
    calibrators = {}
    counts = con.execute(f"SELECT biome, count(*) FROM {src} WHERE {where} GROUP BY 1").fetchall()
    for biome, n in counts:
        keep = min(1000, max(1, int(1000 * max_rows / n)))
        arr = con.execute(
            f"SELECT raw, y FROM {src} WHERE biome = ? AND {where} "
            f"AND hash(cell_id, date) % 1000 < {keep}",
            [biome],
        ).fetchnumpy()
        calibrators[biome] = baselines.fit_calibration(arr["raw"], arr["y"])
        del arr
    return calibrators


def gap_factors(
    data_dir: Path,
    region: str,
    satellite: str,
    first: str,
    last: str,
    exclude_dates: list[date] | None = None,
) -> dict[str, dict[int, float]]:
    """All MODIS fire-days / MODIS fire-days inside the rows, per biome and month,
    from `first` to `last` without the days in `exclude_dates`."""
    cube = data_dir / "processed" / region / "cube"
    biome = data_dir / "processed" / region / "cell_biome.parquet"
    rows = (
        duckdb.connect()
        .execute(f"""
            WITH m AS (
                SELECT DISTINCT c.cell_id, c.date, b.biome
                FROM read_parquet('{cube.as_posix()}/**/*.parquet', hive_partitioning = true) c
                JOIN read_parquet('{biome.as_posix()}') b USING (cell_id)
                WHERE c.sensor = 'MODIS' AND NOT c.nrt AND b.biome IS NOT NULL
                  AND c.date BETWEEN DATE '{first}' AND DATE '{last}'
                  AND c.date NOT IN ({sql_dates(exclude_dates or [])})
            ),
            t AS (
                SELECT cell_id, date FROM {_table(data_dir, region, satellite)} WHERE modis = 1
            )
            SELECT m.biome, month(m.date), count(*), count(t.cell_id)
            FROM m LEFT JOIN t USING (cell_id, date)
            GROUP BY ALL
        """)
        .fetchall()
    )
    factors: dict[str, dict[int, float]] = {}
    for b, month, total, inside in rows:
        factors.setdefault(b, {})[int(month)] = round(total / inside, 4) if inside else 1.0
    return factors


def monthly_series(
    data_dir: Path, region: str, satellite: str, first: str, last: str, est: pd.DataFrame
) -> pd.DataFrame:
    """Per biome and month: all MODIS fire-days, naive VIIRS (satellite fire cell-days),
    and the estimate (est has biome, year, month, estimate)."""
    cube = data_dir / "processed" / region / "cube"
    biome = data_dir / "processed" / region / "cell_biome.parquet"
    obs = (
        duckdb.connect()
        .execute(f"""
            WITH cd AS (
                SELECT c.cell_id, c.date, b.biome,
                       bool_or(c.sensor = 'MODIS') AS m,
                       bool_or(c.satellite = '{satellite}') AS v
                FROM read_parquet('{cube.as_posix()}/**/*.parquet', hive_partitioning = true) c
                JOIN read_parquet('{biome.as_posix()}') b USING (cell_id)
                WHERE NOT c.nrt AND b.biome IS NOT NULL
                  AND c.date BETWEEN DATE '{first}' AND DATE '{last}'
                GROUP BY ALL
            )
            SELECT biome, year(date) AS year, month(date) AS month,
                   count(*) FILTER (m) AS modis, count(*) FILTER (v) AS viirs_naive
            FROM cd GROUP BY ALL
        """)
        .fetchdf()
    )
    out = obs.merge(est, on=["biome", "year", "month"], how="outer").fillna(0)
    return out.sort_values(["biome", "year", "month"]).reset_index(drop=True)


def intervals(monthly: pd.DataFrame) -> dict[str, list[float]]:
    out = {}
    for b, g in monthly.groupby("biome"):
        g = g[g["estimate"] >= MIN_MONTH_ESTIMATE]
        if len(g) < 10:
            continue
        r = g["modis"] / g["estimate"] - 1
        out[str(b)] = [round(float(r.quantile(0.05)), 4), round(float(r.quantile(0.95)), 4)]
    return out


def score(monthly: pd.DataFrame) -> dict:
    def one(g: pd.DataFrame) -> dict:
        m, e, v = g["modis"], g["estimate"], g["viirs_naive"]
        return {
            "modis_fire_days": int(m.sum()),
            "estimate": round(float(e.sum()), 1),
            "bias": round(float(e.sum() / m.sum() - 1), 4) if m.sum() else None,
            "monthly_wape": round(float((e - m).abs().sum() / m.sum()), 4) if m.sum() else None,
            "naive_bias": round(float(v.sum() / m.sum() - 1), 4) if m.sum() else None,
        }

    whole = monthly.groupby(["year", "month"])[["modis", "estimate", "viirs_naive"]].sum()
    out = {"Brazil": one(whole)}
    for b, g in monthly.groupby("biome"):
        out[str(b)] = one(g)
    yearly = monthly.groupby("year")[["modis", "estimate", "viirs_naive"]].sum()
    out["Brazil"]["by_year"] = [
        {
            "year": int(y),
            "modis": int(r["modis"]),
            "estimate": round(float(r["estimate"]), 1),
            "viirs_naive": int(r["viirs_naive"]),
        }
        for y, r in yearly.iterrows()
    ]
    return out


def equivalent(
    raw: np.ndarray,
    biome: np.ndarray,
    month: np.ndarray,
    calibrators: dict,
    gaps: dict[str, dict[int, float]],
) -> np.ndarray:
    """MODIS-equivalent expected fire-days per row: calibrated per biome, times the
    coverage gap of its biome and calendar month."""
    out = np.zeros(len(raw), dtype=np.float32)
    for b, idx in pd.Series(biome).groupby(biome).indices.items():
        g = np.array([gaps.get(b, {}).get(m, 1.0) for m in range(1, 13)])
        p = baselines.apply_calibration(raw[idx], calibrators.get(b))
        out[idx] = p * g[month[idx] - 1]
    return out


def apply_after_training(
    data_dir: Path,
    region: str,
    satellite: str,
    model: dict,
    threads: int = 0,
) -> dict | None:
    """The estimate where the training table cannot reach (pipeline.training
    --application): the final model with the reference calibration and gap, on rows
    picked from VIIRS alone. Written next to the harmonized rows as applied.parquet
    (modis is NULL: not used). Returns what was applied, or None without a table.

    `model` is the harmonization report (<SAT>_model.json): its period end is the
    last day of the training table, and only later days are applied."""
    app_dir = data_dir / "processed" / region / "application" / satellite
    sat_dir = data_dir / "processed" / region / "harmonized" / satellite
    for old in sat_dir.glob("year=*/applied.parquet"):
        old.unlink()
    files = list(app_dir.glob("**/*.parquet"))
    if not files:
        return None
    after = model["period"][1]
    next_day = date.fromisoformat(after) + timedelta(days=1)
    src = f"read_parquet('{app_dir.as_posix()}/**/*.parquet', hive_partitioning = true)"
    con = duckdb.connect()
    first = con.execute(f"SELECT CAST(min(date) AS DATE) FROM {src}").fetchone()[0]
    if first > next_day:
        raise ValueError(
            f"the application table starts on {first}, but the harmonized rows end on "
            f"{after}; rebuild it with `python -m pipeline.training {region} "
            f"--satellite {satellite} --application`"
        )
    booster, w, _ = baselines.load_lightgbm(_fold_dir(data_dir, region, satellite) / FINAL)
    calibrators = model["calibrators"]
    gaps = {b: {int(m): f for m, f in g.items()} for b, g in model["gap_factors"].items()}
    years = [
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT year FROM {src} WHERE date > DATE '{after}' ORDER BY 1"
        ).fetchall()
    ]
    rows, total, last = 0, 0.0, None
    for yr in years:
        df = con.execute(f"""
            SELECT cell_id, date, biome, {", ".join(FEATURES)}
            FROM {src}
            WHERE year = {yr} AND date > DATE '{after}' AND biome IS NOT NULL
        """).fetchdf()
        if df.empty:
            continue
        raw = baselines.unweight(booster.predict(df[FEATURES], num_threads=threads), w)
        dates = pd.to_datetime(df["date"])
        equiv = equivalent(
            raw, df["biome"].astype(str).to_numpy(), dates.dt.month.to_numpy(), calibrators, gaps
        )
        keep = equiv >= EQUIV_FLOOR
        part = sat_dir / f"year={yr}"
        part.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "cell_id": df["cell_id"].to_numpy(np.int64)[keep],
                "date": dates.to_numpy()[keep],
                "biome": df["biome"].astype(str).to_numpy()[keep],
                "modis": pd.array([pd.NA] * int(keep.sum()), dtype="Int8"),
                "equiv": equiv[keep],
            }
        ).to_parquet(part / "applied.parquet", index=False)
        rows += int(keep.sum())
        total += float(equiv.sum())
        last = max(last, dates.max().date()) if last else dates.max().date()
        del df, raw, equiv
    if last is None:
        return None
    report = json.loads(
        (data_dir / "processed" / region / "application" / f"{satellite}_report.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "period": [str(next_day), str(last)],
        "standard_until": report.get("standard_until"),
        "rows_written": rows,
        "estimate": round(total, 1),
        "model": FINAL,
    }


def run(
    region: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    satellite: str = "SNPP",
    threads: int = DEFAULT_THREADS,
    retrain: bool = False,
) -> dict:
    baselines.LGB_PARAMS["num_threads"] = threads
    folds = FOLDS[satellite]
    first = SATELLITES[satellite]["start"]
    out = data_dir / "processed" / region / "harmonized"
    raw_dir = out / f"{satellite}_raw"

    modis_gaps = modis_gap_days(data_dir, region)
    fold_info = fit_folds(data_dir, region, satellite, folds, retrain, modis_gaps)
    years = predict_out_of_fold(data_dir, region, satellite, folds, raw_dir, threads)
    calibrators = fit_biome_calibrators(
        raw_dir, last_year=REFERENCE_LAST_YEAR, exclude_dates=modis_gaps
    )
    last = str(
        duckdb.connect()
        .execute(f"SELECT max(date) FROM read_parquet('{raw_dir.as_posix()}/**/*.parquet')")
        .fetchone()[0]
    )[:10]
    reference_end = min(last, f"{REFERENCE_LAST_YEAR}-12-31")
    gaps = gap_factors(data_dir, region, satellite, first, reference_end, modis_gaps)

    sat_dir = out / satellite
    if sat_dir.exists():
        shutil.rmtree(sat_dir)
    sums = []
    for yr in years:
        df = pd.read_parquet(raw_dir / f"year={yr}" / "raw.parquet")
        month = pd.to_datetime(df["date"]).dt.month.to_numpy()
        equiv = equivalent(
            df["raw"].to_numpy(), df["biome"].astype(str).to_numpy(), month, calibrators, gaps
        )
        df["equiv"], df["month"], df["year"] = equiv, month, yr
        sums.append(df.groupby(["biome", "year", "month"])["equiv"].sum().rename("estimate"))
        keep = df[(df["equiv"] >= EQUIV_FLOOR) | (df["y"] == 1)].rename(columns={"y": "modis"})
        part = sat_dir / f"year={yr}"
        part.mkdir(parents=True)
        keep[["cell_id", "date", "biome", "modis", "equiv"]].to_parquet(
            part / "harmonized.parquet", index=False
        )
        del df, keep
    shutil.rmtree(raw_dir)

    est = pd.concat(sums).reset_index()
    monthly = monthly_series(data_dir, region, satellite, first, last, est)
    reference = monthly[monthly["year"] <= REFERENCE_LAST_YEAR]
    later = monthly[monthly["year"] > REFERENCE_LAST_YEAR]
    q = intervals(reference)
    lo = monthly["biome"].map(lambda b: q.get(b, [0.0, 0.0])[0])
    hi = monthly["biome"].map(lambda b: q.get(b, [0.0, 0.0])[1])
    monthly["low"] = monthly["estimate"] * (1 + lo)
    monthly["high"] = monthly["estimate"] * (1 + hi)
    monthly.to_parquet(out / f"{satellite}_monthly.parquet", index=False)

    report = {
        "region": region,
        "satellite": satellite,
        "period": [first, last],
        "reference_last_year": REFERENCE_LAST_YEAR,
        "modis_gap_days_left_out": [str(d) for d in modis_gaps if str(d) >= first],
        "folds": fold_info,
        "gap_factors": gaps,
        "interval_90": q,
        "scores_out_of_fold": score(reference),
        "after_reference": score(later) if len(later) else None,
        "calibrators": calibrators,
    }
    report["applied"] = apply_after_training(data_dir, region, satellite, report, threads)
    (out / f"{satellite}_model.json").write_text(json.dumps(report), encoding="utf-8")
    return report


def run_apply(
    region: str, data_dir: Path = DEFAULT_DATA_DIR, satellite: str = "SNPP", threads: int = 0
) -> dict | None:
    """Apply the saved harmonization to a new application table, without refitting."""
    path = data_dir / "processed" / region / "harmonized" / f"{satellite}_model.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m pipeline.harmonize {region}`")
    report = json.loads(path.read_text(encoding="utf-8"))
    report["applied"] = apply_after_training(data_dir, region, satellite, report, threads)
    path.write_text(json.dumps(report), encoding="utf-8")
    return report["applied"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("region", nargs="?", default="brazil")
    parser.add_argument("--satellite", choices=list(FOLDS), default="SNPP")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--retrain", action="store_true", help="retrain folds already saved")
    parser.add_argument(
        "--apply-only",
        action="store_true",
        help="only apply the saved model to the application table (pipeline.training "
        "--application), e.g. after new VIIRS days arrive",
    )
    args = parser.parse_args(argv)
    if args.apply_only:
        a = run_apply(args.region, args.data_dir, args.satellite, args.threads)
        if a is None:
            print("no application rows after the harmonized period")
        else:
            print(
                f"{args.satellite}: applied {a['period'][0]} to {a['period'][1]} "
                f"(standard processing until {a['standard_until']}), "
                f"{a['rows_written']:,} rows, {a['estimate']:,.0f} expected fire cell-days"
            )
        return
    r = run(args.region, args.data_dir, args.satellite, args.threads, args.retrain)

    print(
        f"{r['satellite']}: {r['period'][0]} to {r['period'][1]}, {len(r['folds']) - 1} folds "
        f"and a final model; MODIS reference up to {r['reference_last_year']}"
    )
    print(
        f"{'reference years':<17}{'MODIS':>11}{'estimate':>12}{'bias':>9}{'month WAPE':>12}"
        f"{'naive':>9}{'90% band':>18}"
    )
    for area, s in r["scores_out_of_fold"].items():
        band = r["interval_90"].get(area)
        band_txt = f"{band[0]:+.0%} / {band[1]:+.0%}" if band else "-"
        print(
            f"{area:<17}{s['modis_fire_days']:>11,}{s['estimate']:>12,.0f}{s['bias']:>+9.1%}"
            f"{s['monthly_wape']:>12.1%}{s['naive_bias']:>+9.0%}{band_txt:>18}"
        )
    by_year = list(r["scores_out_of_fold"]["Brazil"]["by_year"])
    if r["after_reference"]:
        by_year += r["after_reference"]["Brazil"]["by_year"]
    print("Brazil by year: MODIS / estimate / estimate vs MODIS / naive VIIRS")
    for y in by_year:
        mark = "" if y["year"] <= r["reference_last_year"] else "  (MODIS orbits drifting)"
        diff = y["estimate"] / y["modis"] - 1 if y["modis"] else float("nan")
        print(
            f"  {y['year']}: {y['modis']:>9,} / {y['estimate']:>11,.0f} / {diff:>+6.1%} / "
            f"{y['viirs_naive']:>9,}{mark}"
        )


if __name__ == "__main__":
    main()
