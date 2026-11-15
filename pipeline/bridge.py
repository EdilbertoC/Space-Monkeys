# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Bridge a new satellite onto the harmonized record through the satellite before it.

The record is in MODIS-equivalent fire cell-days. S-NPP and NOAA-20 were calibrated
against MODIS directly (pipeline.harmonize), over years when MODIS orbits were
stable. A newer satellite, such as NOAA-21 (2023), never overlapped that MODIS. It
overlapped NOAA-20 instead, so it learns NOAA-20's record value:

    for each cell-day the new satellite can see, E[reference record value | new satellite]

Over the years both observed, the target of a row is the reference's MODIS-equivalent
expected fire-days on that cell-day (0 where the reference record has none). Because
the target is already in the record's unit, the chain needs one model per link,
not one per link and unit, and it can go on (a later satellite bridges through this
one). The assumption is that the new satellite tells nothing about MODIS that the
reference does not; the chain adds each link's error, so its scores are reported.

Same recipe as the direct harmonization:

1. Rows and features: pipeline.training rules, from the new satellite alone.
2. Cross-fitting by single overlap years: each year predicted by a LightGBM
   regression trained on the other years; days after the overlap (the new
   satellite's NRT days now, every day once the reference ends) by a final model
   trained on every overlap year.
3. Calibration per biome: isotonic map from the out-of-fold prediction to the
   target (keeps each biome's total).
4. Coverage gap: per biome and calendar month, all reference record mass / the
   mass on the new satellite's rows; each value is multiplied by it.

Days either satellite was not observing (pipeline.cube.observation_gaps) are left out
of training, calibration, gap and scores.

Input:  the cube and the reference's record (harmonized/<REF>/ or bridge/<REF>_via_*/)
Output: data/processed/<region>/bridge/<NEW>_via_<REF>/year=YYYY/bridged.parquet
        (cell_id, date, biome, modis NULL, equiv), <NEW>_via_<REF>_model.json and
        <NEW>_via_<REF>_monthly.parquet

Usage::

    python -m pipeline.bridge brazil --new N21 --reference N20
    python -m pipeline.bridge brazil --new N20 --reference SNPP   # check a known link
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from pipeline import baselines, training
from pipeline.crossbiome import SAMPLE_RATE
from pipeline.cube import observation_gaps, sql_dates
from pipeline.harmonize import DEFAULT_THREADS, EQUIV_FLOOR
from pipeline.training import FEATURES, SATELLITES

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
PARAMS = {
    **baselines.LGB_PARAMS,
    "objective": "regression",
    "metric": "l2",
}
CALIBRATION_MAX_ROWS = 8_000_000  # per biome, hash-sampled above this


def name(new: str, reference: str) -> str:
    return f"{new}_via_{reference}"


def record_dir(data_dir: Path, region: str, satellite: str) -> Path:
    """Where a satellite's MODIS-equivalent record is: its direct harmonization, or
    the bridge that brought it in (so links can be chained)."""
    processed = data_dir / "processed" / region
    direct = processed / "harmonized" / satellite
    if direct.exists():
        return direct
    bridged = [
        d
        for d in sorted((processed / "bridge").glob(f"{satellite}_via_*"))
        if d.is_dir() and not d.name.endswith(("_table", "_raw"))
    ]
    if not bridged:
        raise FileNotFoundError(
            f"no record for {satellite}: run pipeline.harmonize or pipeline.bridge first"
        )
    return bridged[0]


def _src(path: Path) -> str:
    return f"read_parquet('{path.as_posix()}/**/*.parquet', hive_partitioning = true)"


def overlap(data_dir: Path, region: str, new: str, reference: str) -> tuple[date, date, list[date]]:
    """First and last day both satellites have standard processing, and the days
    either was not observing."""
    cube = data_dir / "processed" / region / "cube"
    row = (
        duckdb.connect()
        .execute(
            f"""SELECT max(date) FILTER (satellite = ? AND NOT nrt),
                       max(date) FILTER (satellite = ? AND NOT nrt)
                FROM {_src(cube)}""",
            [new, reference],
        )
        .fetchone()
    )
    if row[0] is None or row[1] is None:
        raise ValueError(f"no standard-processing {new} and {reference} data")
    first = max(date.fromisoformat(SATELLITES[new]["start"]), _first_day(reference))
    gaps = observation_gaps(region, data_dir)
    left_out = sorted(set(gaps.get(new, [])) | set(gaps.get(reference, [])))
    return first, min(row[0], row[1]), left_out


def _first_day(satellite: str) -> date:
    return date.fromisoformat(SATELLITES[satellite]["start"])


def build_table(data_dir: Path, region: str, new: str, reference: str, out: Path) -> dict:
    """Rows and features of the new satellite from its first day to its last (NRT
    included), with the reference record value as `target` up to the overlap end
    (NULL after it). Brazil cells only."""
    first, last_overlap, _ = overlap(data_dir, region, new, reference)
    cube = data_dir / "processed" / region / "cube"
    ref = record_dir(data_dir, region, reference)
    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{(data_dir / 'tmp').as_posix()}'")
    training._cube_view(con, cube, nrt=True)
    last = con.execute("SELECT max(date) FROM cube WHERE satellite = ?", [new]).fetchone()[0]
    training._viirs_tables(con, new, str(first), str(last))
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    con.execute(f"""
        COPY (
            SELECT c.cell_id, c.date, year(c.date) AS year, {training.FEATURE_SQL}, b.biome,
                   CASE WHEN c.date <= DATE '{last_overlap}' THEN coalesce(r.equiv, 0) END
                       AS target
            FROM ctx c
            LEFT JOIN v USING (cell_id, date)
            JOIN {training._biome_source(data_dir, region)} b USING (cell_id)
            LEFT JOIN (
                SELECT cell_id, CAST(date AS DATE) AS date, equiv FROM {_src(ref)}
            ) r USING (cell_id, date)
            WHERE c.candidate AND b.biome IS NOT NULL
            ORDER BY c.date, c.cell_id
        ) TO '{out.as_posix()}' (FORMAT parquet, PARTITION_BY (year), COMPRESSION zstd)
    """)
    return {"first": str(first), "overlap_end": str(last_overlap), "last": str(last)}


def gap_factors(
    data_dir: Path,
    region: str,
    reference: str,
    table: Path,
    first: str,
    last: str,
    left_out: list[date],
) -> dict[str, dict[int, float]]:
    """Per biome and calendar month: all reference record mass / the mass on the
    new satellite's rows (first..last, without the left-out days)."""
    ref = record_dir(data_dir, region, reference)
    days = sql_dates(left_out)
    rows = (
        duckdb.connect()
        .execute(f"""
            WITH r AS (
                SELECT biome, month(date) AS m, sum(equiv) AS total
                FROM {_src(ref)}
                WHERE CAST(date AS DATE) BETWEEN DATE '{first}' AND DATE '{last}'
                  AND CAST(date AS DATE) NOT IN ({days})
                GROUP BY ALL
            ),
            t AS (
                SELECT biome, month(date) AS m, sum(target) AS inside
                FROM {_src(table)}
                WHERE date BETWEEN DATE '{first}' AND DATE '{last}'
                  AND date NOT IN ({days})
                GROUP BY ALL
            )
            SELECT r.biome, r.m, r.total, t.inside FROM r LEFT JOIN t USING (biome, m)
        """)
        .fetchall()
    )
    out: dict[str, dict[int, float]] = {}
    for b, m, total, inside in rows:
        out.setdefault(b, {})[int(m)] = round(total / inside, 4) if inside else 1.0
    return out


def load_sample(table: Path, years: list[int], left_out: list[date]) -> pd.DataFrame:
    """Every own-detection row plus a hash-chosen share of the neighbour/lag-only
    rows (weighted back), for the given overlap years."""
    keep = int(SAMPLE_RATE * 1000)
    cols = ", ".join(f"CAST({f} AS FLOAT) AS {f}" for f in FEATURES)
    return (
        duckdb.connect()
        .execute(f"""
            SELECT {cols}, CAST(target AS FLOAT) AS target, year, biome,
                   CASE WHEN v_pixels > 0 THEN 1.0 ELSE {1 / SAMPLE_RATE} END AS w
            FROM {_src(table)}
            WHERE target IS NOT NULL AND year IN ({", ".join(map(str, years))})
              AND date NOT IN ({sql_dates(left_out)})
              AND (v_pixels > 0 OR hash(cell_id, date) % 1000 < {keep})
        """)
        .fetchdf()
    )


def fit(sample: pd.DataFrame, valid_year: int | None, rounds: int | None = None):
    """LightGBM regression of the target; early stopping on valid_year, or a fixed
    number of rounds."""

    def ds(part: pd.DataFrame) -> lgb.Dataset:
        return lgb.Dataset(part[FEATURES], part["target"], weight=part["w"])

    if rounds is not None:
        return lgb.train(PARAMS, ds(sample), rounds), rounds
    fit_part = sample[sample["year"] != valid_year]
    valid = sample[sample["year"] == valid_year]
    booster = lgb.train(
        PARAMS,
        ds(fit_part),
        num_boost_round=baselines.MAX_ROUNDS,
        valid_sets=[ds(valid)],
        callbacks=[lgb.early_stopping(baselines.EARLY_STOP, verbose=False)],
    )
    return booster, booster.best_iteration or baselines.MAX_ROUNDS


def fit_isotonic(x: np.ndarray, y: np.ndarray) -> dict:
    """Increasing map from prediction to target; keeps the total, no upper bound
    (a record value can exceed 1 once the reference's own gap is in it)."""
    iso = IsotonicRegression(y_min=0.0, out_of_bounds="clip")
    iso.fit(x, y)
    return {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}


def monthly(est: pd.DataFrame, ref_monthly: pd.DataFrame) -> pd.DataFrame:
    out = ref_monthly.merge(est, on=["biome", "year", "month"], how="outer").fillna(0)
    return out.sort_values(["biome", "year", "month"]).reset_index(drop=True)


def score(m: pd.DataFrame, against: str) -> dict:
    def one(g: pd.DataFrame) -> dict:
        r, e = g[against], g["estimate"]
        return {
            against: round(float(r.sum()), 1),
            "estimate": round(float(e.sum()), 1),
            "bias": round(float(e.sum() / r.sum() - 1), 4) if r.sum() else None,
            "monthly_wape": round(float((e - r).abs().sum() / r.sum()), 4) if r.sum() else None,
        }

    out = {"Brazil": one(m.groupby(["year", "month"])[[against, "estimate"]].sum())}
    for b, g in m.groupby("biome"):
        out[str(b)] = one(g)
    return out


def run(
    region: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    new: str = "N21",
    reference: str = "N20",
    threads: int = DEFAULT_THREADS,
) -> dict:
    PARAMS["num_threads"] = threads
    root = data_dir / "processed" / region / "bridge"
    out = root / name(new, reference)
    table = root / f"{name(new, reference)}_table"
    raw_dir = root / f"{name(new, reference)}_raw"
    span = build_table(data_dir, region, new, reference, table)
    first, overlap_end, left_out = overlap(data_dir, region, new, reference)
    con = duckdb.connect()
    years = [
        r[0] for r in con.execute(f"SELECT DISTINCT year FROM {_src(table)} ORDER BY 1").fetchall()
    ]
    overlap_years = [y for y in years if y <= overlap_end.year]

    # 1-2. Cross-fitting by single overlap years, then the final model.
    sample = load_sample(table, overlap_years, left_out)
    folds, models = [], {}
    for y in overlap_years:
        rest = [x for x in overlap_years if x != y]
        booster, rounds = fit(sample[sample["year"] != y], valid_year=max(rest))
        models[y] = booster
        folds.append({"held_out_year": y, "valid_year": max(rest), "rounds": rounds})
    final_rounds = int(round(np.mean([f["rounds"] for f in folds])))
    models["final"], _ = fit(sample, None, rounds=final_rounds)
    del sample

    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    for y in years:
        df = con.execute(f"""
            SELECT cell_id, date, biome, target, {", ".join(FEATURES)}
            FROM {_src(table)} WHERE year = {y}
        """).fetchdf()
        after = pd.to_datetime(df["date"]).dt.date > overlap_end
        raw = np.empty(len(df), dtype=np.float32)
        if (~after).any():
            raw[~after.to_numpy()] = models[y].predict(
                df.loc[~after, FEATURES], num_threads=threads
            )
        if after.any():
            raw[after.to_numpy()] = models["final"].predict(
                df.loc[after, FEATURES], num_threads=threads
            )
        part = raw_dir / f"year={y}"
        part.mkdir(parents=True)
        pd.DataFrame(
            {
                "cell_id": df["cell_id"].to_numpy(np.int64),
                "date": pd.to_datetime(df["date"]).to_numpy(),
                "biome": df["biome"].astype(str).to_numpy(),
                "target": df["target"].to_numpy(np.float32),
                "raw": np.clip(raw, 0, None),
            }
        ).to_parquet(part / "raw.parquet", index=False)
        del df, raw

    # 3. Calibration per biome on the out-of-fold overlap rows.
    days = sql_dates(left_out)
    calibrators = {}
    for b, n in con.execute(
        f"SELECT biome, count(*) FROM {_src(raw_dir)} WHERE target IS NOT NULL "
        f"AND CAST(date AS DATE) NOT IN ({days}) GROUP BY 1"
    ).fetchall():
        keep = min(1000, max(1, int(1000 * CALIBRATION_MAX_ROWS / n)))
        arr = con.execute(
            f"SELECT raw, target FROM {_src(raw_dir)} WHERE biome = ? AND target IS NOT NULL "
            f"AND CAST(date AS DATE) NOT IN ({days}) AND hash(cell_id, date) % 1000 < {keep}",
            [b],
        ).fetchnumpy()
        calibrators[b] = fit_isotonic(arr["raw"], arr["target"])

    # 4. Coverage gap, then the record values.
    gaps = gap_factors(data_dir, region, reference, table, str(first), str(overlap_end), left_out)
    if out.exists():
        shutil.rmtree(out)
    sums = []
    for y in years:
        df = pd.read_parquet(raw_dir / f"year={y}" / "raw.parquet")
        month = pd.to_datetime(df["date"]).dt.month.to_numpy()
        equiv = np.zeros(len(df), dtype=np.float32)
        for b, idx in df.groupby("biome").indices.items():
            g = np.array([gaps.get(b, {}).get(m, 1.0) for m in range(1, 13)])
            p = baselines.apply_calibration(df["raw"].to_numpy()[idx], calibrators.get(b))
            equiv[idx] = p * g[month[idx] - 1]
        df["equiv"], df["month"], df["year"] = equiv, month, y
        scored = ~pd.to_datetime(df["date"]).dt.date.isin(set(left_out))
        sums.append(
            df[scored].groupby(["biome", "year", "month"])["equiv"].sum().rename("estimate")
        )
        keep = df["equiv"] >= EQUIV_FLOOR
        part = out / f"year={y}"
        part.mkdir(parents=True)
        pd.DataFrame(
            {
                "cell_id": df.loc[keep, "cell_id"].to_numpy(),
                "date": df.loc[keep, "date"].to_numpy(),
                "biome": df.loc[keep, "biome"].to_numpy(),
                "modis": pd.array([pd.NA] * int(keep.sum()), dtype="Int8"),
                "equiv": df.loc[keep, "equiv"].to_numpy(),
            }
        ).to_parquet(part / "bridged.parquet", index=False)
        del df
    shutil.rmtree(raw_dir)

    # Scores: against the reference record (the link) and against observed MODIS.
    ref = record_dir(data_dir, region, reference)
    cube = data_dir / "processed" / region / "cube"
    biome_file = data_dir / "processed" / region / "cell_biome.parquet"
    ref_monthly = con.execute(f"""
        SELECT biome, year(date) AS year, month(date) AS month, sum(equiv) AS reference
        FROM {_src(ref)}
        WHERE CAST(date AS DATE) BETWEEN DATE '{first}' AND DATE '{overlap_end}'
          AND CAST(date AS DATE) NOT IN ({days})
        GROUP BY ALL
    """).fetchdf()
    modis_monthly = con.execute(f"""
        SELECT b.biome, year(c.date) AS year, month(c.date) AS month,
               count(DISTINCT (c.cell_id, c.date)) AS modis
        FROM {_src(cube)} c JOIN read_parquet('{biome_file.as_posix()}') b USING (cell_id)
        WHERE c.sensor = 'MODIS' AND NOT c.nrt AND b.biome IS NOT NULL
          AND c.date BETWEEN DATE '{first}' AND DATE '{overlap_end}'
          AND c.date NOT IN ({days})
        GROUP BY ALL
    """).fetchdf()
    est = pd.concat(sums).reset_index()
    in_overlap = est[
        (est["year"] * 100 + est["month"]) <= overlap_end.year * 100 + overlap_end.month
    ]
    m = monthly(in_overlap, ref_monthly).merge(
        modis_monthly, on=["biome", "year", "month"], how="left"
    )
    m["modis"] = m["modis"].fillna(0)
    m.to_parquet(root / f"{name(new, reference)}_monthly.parquet", index=False)
    by_year = m.groupby("year")[["reference", "estimate", "modis"]].sum()

    report = {
        "region": region,
        "new": new,
        "reference": reference,
        "reference_record": str(ref.relative_to(data_dir / "processed" / region)),
        "overlap": [str(first), str(overlap_end)],
        "period": [span["first"], span["last"]],
        "left_out_days": [str(d) for d in left_out if first <= d <= overlap_end],
        "folds": folds,
        "final_rounds": final_rounds,
        "gap_factors": gaps,
        "calibrators": calibrators,
        "scores_out_of_fold": score(m, "reference"),
        "against_modis": score(m, "modis"),
        "by_year": [
            {
                "year": int(y),
                "reference": round(float(r["reference"]), 1),
                "estimate": round(float(r["estimate"]), 1),
                "modis": int(r["modis"]),
            }
            for y, r in by_year.iterrows()
        ],
    }
    (root / f"{name(new, reference)}_model.json").write_text(json.dumps(report), encoding="utf-8")
    shutil.rmtree(table)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("region", nargs="?", default="brazil")
    parser.add_argument("--new", choices=list(SATELLITES), default="N21")
    parser.add_argument("--reference", choices=list(SATELLITES), default="N20")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    args = parser.parse_args(argv)
    if args.new == args.reference:
        parser.error("--new and --reference must differ")
    r = run(args.region, args.data_dir, args.new, args.reference, args.threads)

    print(
        f"{r['new']} via {r['reference']}: overlap {r['overlap'][0]} to {r['overlap'][1]}, "
        f"estimate {r['period'][0]} to {r['period'][1]}"
    )
    print(f"{'out of fold':<17}{'reference':>12}{'estimate':>12}{'bias':>9}{'month WAPE':>12}")
    for area, s in r["scores_out_of_fold"].items():
        print(
            f"{area:<17}{s['reference']:>12,.0f}{s['estimate']:>12,.0f}"
            f"{s['bias']:>+9.1%}{s['monthly_wape']:>12.1%}"
        )
    print("Brazil by year: reference record / estimate / observed MODIS")
    for y in r["by_year"]:
        print(
            f"  {y['year']}: {y['reference']:>10,.0f} / {y['estimate']:>10,.0f} / {y['modis']:>9,}"
        )


if __name__ == "__main__":
    main()
