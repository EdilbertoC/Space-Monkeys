# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Does the VIIRS -> MODIS calibration hold outside the area it learned from?

Scores, per Brazilian biome (IBGE 2019) on the national cube's test years
(Brazil cells only, so sea and neighbouring countries are left out):

- naive: VIIRS as MODIS, no harmonization (reference)
- ratio_brazil, lightgbm_brazil: trained on every biome in the training years
- lightgbm_<source>: the model trained on another region's box (default the
  Pantanal pilot, pipeline.baselines), applied to biomes it never saw

The national training table is too big for memory, so training keeps every row
where the satellite itself saw fire and a fixed 1-in-5 sample of the
neighbour/lag-only rows, chosen by a hash of (cell, date), never by the target,
each weighted 5. Test years are read one year at a time, with every row.

Also reports, per biome, the share of MODIS fire-days that fall inside the rows
(coverage): no row-based model can recover the rest.

Inputs: data/processed/<region>/training/<SAT>/ (with biome, pipeline.biomes
        before pipeline.training), data/processed/<source>/models/<SAT>/
Output: data/processed/<region>/models/<SAT>/crossbiome.json, lightgbm.txt

Usage::

    python -m pipeline.crossbiome brazil
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from pipeline import baselines
from pipeline.cube import sql_dates
from pipeline.training import FEATURES, SATELLITES, TARGET

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
SAMPLE_RATE = 0.2  # share of neighbour/lag-only training rows kept


def _table(data_dir: Path, region: str, satellite: str) -> str:
    path = data_dir / "processed" / region / "training" / satellite
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run pipeline.biomes then pipeline.training {region}"
        )
    return f"read_parquet('{path.as_posix()}/**/*.parquet', hive_partitioning = true)"


def load_training_sample(
    data_dir: Path,
    region: str,
    satellite: str,
    rate: float = SAMPLE_RATE,
    splits: tuple[str, ...] = ("train",),
    last_year: int | None = None,
    exclude_dates: list[date] | None = None,
) -> pd.DataFrame:
    """Every own-detection row plus a hash-chosen `rate` of the neighbour/lag-only
    rows (weight 1/rate), Brazil cells only. Features are read as 32-bit floats.
    Optionally only years up to `last_year` and without the days in `exclude_dates`."""
    cols = ", ".join(f"CAST({f} AS FLOAT) AS {f}" for f in FEATURES)
    keep = int(rate * 1000)
    split_list = ", ".join(f"'{s}'" for s in splits)
    extra = f"AND year <= {int(last_year)} " if last_year is not None else ""
    extra += f"AND date NOT IN ({sql_dates(exclude_dates or [])})"
    return (
        duckdb.connect()
        .execute(f"""
            SELECT {cols}, {TARGET}, year, biome,
                   CASE WHEN v_pixels > 0 THEN 1.0 ELSE {1 / rate} END AS w
            FROM {_table(data_dir, region, satellite)}
            WHERE split IN ({split_list}) AND biome IS NOT NULL
              AND (v_pixels > 0 OR hash(cell_id, date) % 1000 < {keep})
              {extra}
        """)
        .fetchdf()
    )


def _source_model(
    data_dir: Path, source: str, satellite: str
) -> tuple[lgb.Booster, float, dict | None]:
    return baselines.load_lightgbm(data_dir / "processed" / source / "models" / satellite)


def coverage_by_biome(data_dir: Path, region: str, satellite: str, test_from: int) -> dict:
    """Share of MODIS fire-days in Brazil cells (test years) that are rows of the table."""
    cube = data_dir / "processed" / region / "cube"
    biome = data_dir / "processed" / region / "cell_biome.parquet"
    rows = (
        duckdb.connect()
        .execute(f"""
            WITH t AS (
                SELECT cell_id, date, biome FROM {_table(data_dir, region, satellite)}
                WHERE split = 'test' AND biome IS NOT NULL
            ),
            last AS (SELECT max(date) AS d FROM t),
            m AS (
                SELECT DISTINCT c.cell_id, c.date, b.biome
                FROM read_parquet('{cube.as_posix()}/**/*.parquet', hive_partitioning = true) c
                JOIN read_parquet('{biome.as_posix()}') b USING (cell_id)
                WHERE c.sensor = 'MODIS' AND NOT c.nrt AND b.biome IS NOT NULL
                  AND year(c.date) >= {test_from} AND c.date <= (SELECT d FROM last)
            )
            SELECT m.biome, count(*), count(t.cell_id)
            FROM m LEFT JOIN t USING (cell_id, date)
            GROUP BY m.biome
        """)
        .fetchall()
    )
    out = {b: round(inside / total, 4) for b, total, inside in rows}
    total, inside = sum(r[1] for r in rows), sum(r[2] for r in rows)
    out["Brazil"] = round(inside / total, 4) if total else None
    return out


def score_groups(cols: dict[str, np.ndarray], models: list[str]) -> dict:
    """baselines.evaluate_arrays for Brazil and each biome, every model on the same rows.

    cols holds y, cell_id, year, month, biome and one probability array per model;
    by-year totals are kept for Brazil only."""
    results = {}
    for g in ["Brazil", *sorted(set(cols["biome"]))]:
        mask = np.ones(len(cols["y"]), bool) if g == "Brazil" else cols["biome"] == g
        if not cols["y"][mask].sum():
            continue
        results[g] = {
            m: {
                k: v
                for k, v in baselines.evaluate_arrays(
                    cols["y"][mask],
                    cols[m][mask].astype(float),
                    cols["cell_id"][mask],
                    cols["year"][mask],
                    cols["month"][mask],
                ).items()
                if k != "by_year" or g == "Brazil"
            }
            for m in models
        }
    return results


def run(
    region: str = "brazil",
    data_dir: Path = DEFAULT_DATA_DIR,
    satellite: str = "SNPP",
    source: str = "pantanal",
) -> dict:
    last_train = SATELLITES[satellite]["train_last_year"]
    train = load_training_sample(data_dir, region, satellite)
    if train.empty or train[TARGET].sum() == 0:
        raise ValueError("no training rows with MODIS fire")
    ratio = baselines.fit_ratio(train, weight="w")
    booster, info = baselines.fit_lightgbm(train, valid_year=last_train, weight="w")
    train_rows, train_biomes = len(train), train["biome"].value_counts().to_dict()
    del train
    w, cal = info["positive_weight"], info["calibrator"]
    src_booster, src_w, src_cal = _source_model(data_dir, source, satellite)

    models = [
        "naive",
        "ratio_brazil",
        "lightgbm_brazil_uncalibrated",
        "lightgbm_brazil",
        f"lightgbm_{source}",
    ]
    parts: dict[str, list] = {k: [] for k in ["y", "cell_id", "year", "month", "biome", *models]}
    con = duckdb.connect()
    years = [
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT year FROM {_table(data_dir, region, satellite)} "
            "WHERE split = 'test' ORDER BY 1"
        ).fetchall()
    ]
    for yr in years:
        df = con.execute(f"""
            SELECT cell_id, date, year, biome, {TARGET}, {", ".join(FEATURES)}
            FROM {_table(data_dir, region, satellite)}
            WHERE split = 'test' AND year = {yr} AND biome IS NOT NULL
        """).fetchdf()
        df["date"] = pd.to_datetime(df["date"])
        parts["y"].append(df[TARGET].to_numpy(np.int8))
        parts["cell_id"].append(df["cell_id"].to_numpy(np.int64))
        parts["year"].append(df["year"].to_numpy(np.int16))
        parts["month"].append(df["date"].dt.month.to_numpy(np.int8))
        parts["biome"].append(df["biome"].to_numpy(object))
        parts["naive"].append(baselines.naive_predict(df).astype(np.float32))
        parts["ratio_brazil"].append(baselines.ratio_predict(df, ratio).astype(np.float32))
        raw = baselines.lightgbm_predict(booster, df, w)
        parts["lightgbm_brazil_uncalibrated"].append(raw.astype(np.float32))
        parts["lightgbm_brazil"].append(baselines.apply_calibration(raw, cal).astype(np.float32))
        parts[f"lightgbm_{source}"].append(
            baselines.lightgbm_predict(src_booster, df, src_w, src_cal).astype(np.float32)
        )
        del df
    cols = {k: np.concatenate(v) for k, v in parts.items()}
    results = score_groups(cols, models)

    out = data_dir / "processed" / region / "models" / satellite
    baselines.save_lightgbm(out, booster, info)
    report = {
        "region": region,
        "satellite": satellite,
        "source_model": source,
        "train_years": [SATELLITES[satellite]["start"][:4], last_train],
        "test_years": [years[0], years[-1]],
        "train_rows_sampled": train_rows,
        "train_rows_by_biome": train_biomes,
        "sample_rate_neighbour_only": SAMPLE_RATE,
        "lightgbm_brazil": {k: v for k, v in info.items() if k != "calibrator"},
        "coverage": coverage_by_biome(data_dir, region, satellite, years[0]),
        "results": results,
    }
    (out / "crossbiome.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("region", nargs="?", default="brazil")
    parser.add_argument("--satellite", choices=list(SATELLITES), default="SNPP")
    parser.add_argument("--source", default="pantanal", help="region whose model is transferred")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)
    r = run(args.region, args.data_dir, args.satellite, args.source)

    print(
        f"{r['satellite']}: train {r['train_years'][0]}-{r['train_years'][1]} "
        f"({r['train_rows_sampled']:,} sampled rows), test {r['test_years'][0]}-"
        f"{r['test_years'][1]}; Brazil LightGBM {r['lightgbm_brazil']['rounds']} rounds"
    )
    print(f"{'biome':<17}{'model':<20}{'bias':>9}{'month WAPE':>12}{'cell-mo WAPE':>14}{'AUC':>7}")
    for biome, models in r["results"].items():
        cov = r["coverage"].get(biome)
        for i, (name, m) in enumerate(models.items()):
            t = m["totals"]
            label = f"{biome} ({cov:.0%} cov.)" if i == 0 and cov is not None else ""
            print(
                f"{label:<17}{name:<20}{t['bias']:>+9.1%}{t['monthly_wape']:>12.1%}"
                f"{t['cell_month_wape']:>14.1%}{m['rows']['roc_auc'] or 0:>7.3f}"
            )


if __name__ == "__main__":
    main()
