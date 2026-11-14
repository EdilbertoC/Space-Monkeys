# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Step 8: the harmonized record, VIIRS expressed in MODIS-equivalent fire cell-days.

For one VIIRS satellite over the whole period it overlaps MODIS:

1. Cross-fitting by year blocks. The years are split into blocks; each block is
   predicted by a LightGBM trained on the other blocks only (pipeline.baselines,
   weighted, early stopping on its last full training year). Every cell-day of
   the overlap therefore gets an out-of-sample estimate, which is what lets us
   check the record for a step where MODIS hands over to VIIRS.
2. Calibration per biome: an isotonic map from the out-of-fold probability to the
   observed MODIS rate, one per IBGE biome (one national map left the Pantanal
   6% high and Caatinga/Pampa 11-13% low). It is fitted on the same out-of-fold
   rows it is scored on, so the totals it reports are slightly optimistic.
3. Coverage gap: MODIS fire-days with no VIIRS fire nearby can never be rows.
   Per biome and calendar month, g = all MODIS fire-days / MODIS fire-days inside
   the rows; each calibrated probability is multiplied by g, so area totals
   include the gap. The gap cannot be placed on the map: per cell it is spread
   over the cells where VIIRS saw activity.
4. Uncertainty: per biome, the 5th and 95th percentiles of the relative error of
   monthly totals (actual / estimate - 1, months with at least 20 estimated
   fire-days) give a 90% interval: [est * (1 + q05), est * (1 + q95)].

Outputs (data/processed/<region>/harmonized/):
- <SAT>/year=YYYY/*.parquet: cell_id, date, biome, modis (observed 0/1), equiv
  (MODIS-equivalent expected fire-days, after calibration and gap)
- <SAT>_monthly.parquet: per biome and month: MODIS observed, VIIRS naive
  (fire cell-days seen by the satellite), estimate, interval
- <SAT>_model.json: folds, calibrators, gap factors, intervals, metrics
- models in data/processed/<region>/models/<SAT>/folds/

The harmonized series itself is MODIS where MODIS exists and this estimate where
it does not (after MODIS ends); see the README.

Usage::

    python -m pipeline.harmonize brazil --satellite SNPP
    python -m pipeline.harmonize brazil --satellite N20
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from pipeline import baselines, crossbiome
from pipeline.training import FEATURES, SATELLITES, TARGET

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"

FOLDS = {
    "SNPP": [(2012, 2014), (2015, 2017), (2018, 2020), (2021, 2023), (2024, 2026)],
    "N20": [(2018, 2019), (2020, 2021), (2022, 2023), (2024, 2026)],
    "N21": [(2023, 2024), (2025, 2026)],
}
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


def fold_of(year: int, folds: list[tuple[int, int]]) -> int:
    for k, (a, b) in enumerate(folds):
        if a <= year <= b:
            return k
    raise ValueError(f"year {year} is in no fold")


def _fold_dir(data_dir: Path, region: str, satellite: str) -> Path:
    return data_dir / "processed" / region / "models" / satellite / "folds"


def fit_folds(
    data_dir: Path,
    region: str,
    satellite: str,
    folds: list[tuple[int, int]],
    retrain: bool = False,
) -> list[dict]:
    """Train one model per fold on every other block; save and describe them.

    Folds whose model is already saved (with meta.json) are reused unless
    `retrain`, so an interrupted run resumes where it stopped.
    """
    out = _fold_dir(data_dir, region, satellite)
    full = full_years(data_dir, region, satellite)
    todo = [k for k in range(len(folds)) if retrain or not (out / str(k) / "meta.json").exists()]
    sample = None
    if todo:
        sample = crossbiome.load_training_sample(
            data_dir, region, satellite, splits=("train", "test")
        )
    described = []
    for k, (a, b) in enumerate(folds):
        meta_path = out / str(k) / "meta.json"
        if k in todo:
            train = sample[(sample["year"] < a) | (sample["year"] > b)]
            valid_year = max(y for y in full if y < a or y > b)
            booster, info = baselines.fit_lightgbm(train, valid_year=valid_year, weight="w")
            baselines.save_lightgbm(out / str(k), booster, info)
            meta = {
                "fold": k,
                "held_out_years": [a, b],
                "valid_year": valid_year,
                "rounds": info["rounds"],
                "positive_weight": info["positive_weight"],
                "train_rows_sampled": len(train),
            }
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            del booster, train
        described.append(json.loads(meta_path.read_text(encoding="utf-8")))
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
    out, one year at a time, written to raw_dir/year=YYYY/raw.parquet."""
    models = [
        baselines.load_lightgbm(_fold_dir(data_dir, region, satellite) / str(k))
        for k in range(len(folds))
    ]
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


def fit_biome_calibrators(raw_dir: Path, max_rows: int = CALIBRATION_MAX_ROWS) -> dict:
    """One isotonic calibrator per biome from the out-of-fold rows on disk. Biomes with
    more rows than max_rows are fitted on a fixed hash sample of them."""
    con = duckdb.connect()
    src = f"read_parquet('{raw_dir.as_posix()}/**/*.parquet', hive_partitioning = true)"
    calibrators = {}
    for biome, n in con.execute(f"SELECT biome, count(*) FROM {src} GROUP BY 1").fetchall():
        keep = min(1000, max(1, int(1000 * max_rows / n)))
        arr = con.execute(
            f"SELECT raw, y FROM {src} WHERE biome = ? AND hash(cell_id, date) % 1000 < {keep}",
            [biome],
        ).fetchnumpy()
        calibrators[biome] = baselines.fit_calibration(arr["raw"], arr["y"])
        del arr
    return calibrators


def gap_factors(
    data_dir: Path, region: str, satellite: str, first: str, last: str
) -> dict[str, dict[int, float]]:
    """All MODIS fire-days / MODIS fire-days inside the rows, per biome and month."""
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

    fold_info = fit_folds(data_dir, region, satellite, folds, retrain)
    years = predict_out_of_fold(data_dir, region, satellite, folds, raw_dir, threads)
    calibrators = fit_biome_calibrators(raw_dir)
    last = str(
        duckdb.connect()
        .execute(f"SELECT max(date) FROM read_parquet('{raw_dir.as_posix()}/**/*.parquet')")
        .fetchone()[0]
    )[:10]
    gaps = gap_factors(data_dir, region, satellite, first, last)

    sat_dir = out / satellite
    if sat_dir.exists():
        shutil.rmtree(sat_dir)
    sums = []
    for yr in years:
        df = pd.read_parquet(raw_dir / f"year={yr}" / "raw.parquet")
        month = pd.to_datetime(df["date"]).dt.month.to_numpy()
        equiv = np.zeros(len(df), dtype=np.float32)
        for b, idx in df.groupby("biome").indices.items():
            g = np.array([gaps.get(b, {}).get(m, 1.0) for m in range(1, 13)])
            p = baselines.apply_calibration(df["raw"].to_numpy()[idx], calibrators.get(b))
            equiv[idx] = p * g[month[idx] - 1]
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
    q = intervals(monthly)
    lo = monthly["biome"].map(lambda b: q.get(b, [0.0, 0.0])[0])
    hi = monthly["biome"].map(lambda b: q.get(b, [0.0, 0.0])[1])
    monthly["low"] = monthly["estimate"] * (1 + lo)
    monthly["high"] = monthly["estimate"] * (1 + hi)
    monthly.to_parquet(out / f"{satellite}_monthly.parquet", index=False)

    report = {
        "region": region,
        "satellite": satellite,
        "period": [first, last],
        "folds": fold_info,
        "gap_factors": gaps,
        "interval_90": q,
        "scores_out_of_fold": score(monthly),
        "calibrators": calibrators,
    }
    (out / f"{satellite}_model.json").write_text(json.dumps(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("region", nargs="?", default="brazil")
    parser.add_argument("--satellite", choices=list(FOLDS), default="SNPP")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--retrain", action="store_true", help="retrain folds already saved")
    args = parser.parse_args(argv)
    r = run(args.region, args.data_dir, args.satellite, args.threads, args.retrain)

    print(f"{r['satellite']}: {r['period'][0]} to {r['period'][1]}, {len(r['folds'])} folds")
    print(
        f"{'area':<17}{'MODIS':>11}{'estimate':>12}{'bias':>9}{'month WAPE':>12}"
        f"{'naive':>9}{'90% band':>18}"
    )
    for area, s in r["scores_out_of_fold"].items():
        band = r["interval_90"].get(area)
        band_txt = f"{band[0]:+.0%} / {band[1]:+.0%}" if band else "-"
        print(
            f"{area:<17}{s['modis_fire_days']:>11,}{s['estimate']:>12,.0f}{s['bias']:>+9.1%}"
            f"{s['monthly_wape']:>12.1%}{s['naive_bias']:>+9.0%}{band_txt:>18}"
        )
    print("Brazil by year: MODIS / estimate / naive VIIRS")
    for y in r["scores_out_of_fold"]["Brazil"]["by_year"]:
        print(f"  {y['year']}: {y['modis']:>9,} / {y['estimate']:>11,.0f} / {y['viirs_naive']:>9,}")


if __name__ == "__main__":
    main()
