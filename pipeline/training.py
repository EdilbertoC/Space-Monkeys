# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Step 5: fire-day cube -> training table for the VIIRS -> MODIS calibration.

Question the model answers: "did MODIS detect fire in this 0.1° cell on this UTC
day?" (0/1), predicted only from what one VIIRS satellite saw plus context.

One table per VIIRS satellite. Summing the VIIRS satellites is not stationary:
NOAA-20 (2018) and NOAA-21 (2023) add pixels, so the same fire looks bigger
later, and a model trained on 2012-2017 would never see them. For S-NPP alone,
P(MODIS | S-NPP pixels) is stable from 2012 to 2026 in the Pantanal. Each table
uses rows, features and a time split from its own satellite:

- SNPP: train 2012-2017, test 2018 onward
- N20:  train 2018-2022, test 2023 onward (S-NPP ends on 2026-11-02, so the
  record continues on NOAA-20)
- N21:  train 2023-2024, test 2025 onward (short overlap; use with care)

Rows (candidates). A cell-day is a row when the satellite saw fire in its 3x3
neighbourhood that day or the day before. The rule uses VIIRS only, because after
MODIS ends the model can only run where VIIRS shows activity: training and
inference must pick rows the same way. MODIS fire-days outside the candidates
cannot be recovered by any model; their share is reported as coverage.

Features never use MODIS (no MODIS lags or neighbours): MODIS will not exist at
inference time.

- own cell, same day: pixels, FRP sum/max, day/night and high-confidence pixels,
  overpasses, mean scan size
- neighbours, same day: pixels and cells with fire in the 8 neighbours (3x3) and
  in the 16-cell ring around them (5x5)
- previous days: own pixels on day-1 and days-2..3, neighbour pixels on day-1
- context: month, day of year (sin/cos), cell centre

Only dates where both sensors have standard processing are used (no NRT tail).

Input:  data/processed/<region>/cube/
Output: data/processed/<region>/training/<SAT>/year=YYYY/*.parquet
        data/processed/<region>/training/<SAT>_report.json

Usage::

    python -m pipeline.training pantanal              # S-NPP
    python -m pipeline.training pantanal --satellite N20
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import duckdb

from pipeline.cube import N_COLS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"

# VIIRS data start per satellite (FIRMS archive download form) and time split.
SATELLITES = {
    "SNPP": {"start": "2012-01-20", "train_last_year": 2017},
    "N20": {"start": "2018-04-01", "train_last_year": 2022},
    "N21": {"start": "2023-03-01", "train_last_year": 2024},
}

FEATURES = [
    "v_pixels",
    "v_frp_sum",
    "v_frp_max",
    "v_day",
    "v_night",
    "v_high",
    "v_overpasses",
    "v_scan_mean",
    "nb3_pixels",
    "nb3_cells",
    "nb5_pixels",
    "nb5_cells",
    "lag1_pixels",
    "lag23_pixels",
    "lag1_nb3_pixels",
    "month",
    "doy_sin",
    "doy_cos",
    "cell_lat",
    "cell_lon",
]
TARGET = "modis"
KEYS = ["cell_id", "date", "year", "split"]
DIAGNOSTICS = ["modis_pixels", "terra", "aqua"]  # never use as features


def _offsets(radius: int) -> str:
    rows = [
        f"({dr}, {dc}, {max(abs(dr), abs(dc))})"
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
    ]
    return "(VALUES " + ", ".join(rows) + ") o(dr, dc, ring)"


def build_training(region: str, data_dir: Path = DEFAULT_DATA_DIR, satellite: str = "SNPP") -> dict:
    if satellite not in SATELLITES:
        raise ValueError(f"satellite must be one of {', '.join(SATELLITES)}")
    cfg = SATELLITES[satellite]
    cube_dir = data_dir / "processed" / region / "cube"
    out_dir = data_dir / "processed" / region / "training" / satellite
    if not cube_dir.exists():
        raise FileNotFoundError(f"{cube_dir} not found; run `python -m pipeline.cube {region}`")

    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{(data_dir / 'tmp').as_posix()}'")
    con.execute(f"""
        CREATE TEMP VIEW cube AS
        SELECT * FROM read_parquet('{cube_dir.as_posix()}/**/*.parquet', hive_partitioning = true)
        WHERE NOT nrt
    """)
    # Last date with standard processing for both MODIS and this satellite.
    last_sp = con.execute(
        "SELECT least(max(date) FILTER (sensor = 'MODIS'), max(date) FILTER (satellite = ?)) "
        "FROM cube",
        [satellite],
    ).fetchone()[0]
    if last_sp is None:
        raise ValueError(f"no standard-processing {satellite} and MODIS data in {cube_dir}")
    first = cfg["start"]
    period = f"date BETWEEN DATE '{first}' AND DATE '{last_sp}'"

    con.execute(f"""
        CREATE TEMP TABLE v AS
        SELECT cell_id, date,
               n_pixels AS v_pixels, frp_sum_mw AS v_frp_sum, frp_max_mw AS v_frp_max,
               n_day AS v_day, n_night AS v_night, n_high AS v_high,
               n_overpasses AS v_overpasses, scan_mean_km AS v_scan_mean
        FROM cube WHERE satellite = '{satellite}' AND {period}
    """)
    con.execute(f"""
        CREATE TEMP TABLE m AS
        SELECT cell_id, date,
               sum(n_pixels) AS modis_pixels,
               bool_or(satellite = 'Terra') AS terra,
               bool_or(satellite = 'Aqua') AS aqua
        FROM cube WHERE sensor = 'MODIS' AND {period}
        GROUP BY cell_id, date
    """)

    # Spread every VIIRS cell-day to the cells and days it informs, then sum.
    con.execute(f"""
        CREATE TEMP TABLE ctx AS
        WITH spread AS (
            SELECT v.cell_id + o.dr * {N_COLS} + o.dc AS cell_id, v.date + lag.l AS date,
                   o.ring, lag.l, v.v_pixels
            FROM v, {_offsets(2)}, (VALUES (0), (1), (2), (3)) lag(l)
            WHERE (lag.l = 0) OR o.ring <= 1
        )
        SELECT cell_id, date,
               coalesce(sum(v_pixels) FILTER (l = 0 AND ring = 1), 0) AS nb3_pixels,
               count(*) FILTER (l = 0 AND ring = 1) AS nb3_cells,
               coalesce(sum(v_pixels) FILTER (l = 0 AND ring = 2), 0) AS nb5_pixels,
               count(*) FILTER (l = 0 AND ring = 2) AS nb5_cells,
               coalesce(sum(v_pixels) FILTER (l = 1 AND ring = 0), 0) AS lag1_pixels,
               coalesce(sum(v_pixels) FILTER (l IN (2, 3) AND ring = 0), 0) AS lag23_pixels,
               coalesce(sum(v_pixels) FILTER (l = 1 AND ring = 1), 0) AS lag1_nb3_pixels,
               bool_or(ring <= 1 AND l <= 1) AS candidate
        FROM spread
        WHERE date <= DATE '{last_sp}'
        GROUP BY cell_id, date
    """)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    con.execute(f"""
        COPY (
            SELECT
                c.cell_id, c.date, year(c.date) AS year,
                CASE WHEN year(c.date) <= {cfg["train_last_year"]} THEN 'train' ELSE 'test' END
                    AS split,
                coalesce(v.v_pixels, 0) AS v_pixels,
                coalesce(v.v_frp_sum, 0) AS v_frp_sum,
                coalesce(v.v_frp_max, 0) AS v_frp_max,
                coalesce(v.v_day, 0) AS v_day,
                coalesce(v.v_night, 0) AS v_night,
                coalesce(v.v_high, 0) AS v_high,
                coalesce(v.v_overpasses, 0) AS v_overpasses,
                v.v_scan_mean,
                c.nb3_pixels, c.nb3_cells, c.nb5_pixels, c.nb5_cells,
                c.lag1_pixels, c.lag23_pixels, c.lag1_nb3_pixels,
                month(c.date) AS month,
                sin(2 * pi() * (dayofyear(c.date) - 1) / 365.25) AS doy_sin,
                cos(2 * pi() * (dayofyear(c.date) - 1) / 365.25) AS doy_cos,
                round((c.cell_id // {N_COLS}) / 10 - 90 + 0.05, 2) AS cell_lat,
                round((c.cell_id % {N_COLS}) / 10 - 180 + 0.05, 2) AS cell_lon,
                CAST(m.cell_id IS NOT NULL AS TINYINT) AS modis,
                coalesce(m.modis_pixels, 0) AS modis_pixels,
                coalesce(m.terra, false) AS terra,
                coalesce(m.aqua, false) AS aqua
            FROM ctx c
            LEFT JOIN v USING (cell_id, date)
            LEFT JOIN m USING (cell_id, date)
            WHERE c.candidate AND c.date >= DATE '{first}'
            ORDER BY c.date, c.cell_id
        ) TO '{out_dir.as_posix()}' (FORMAT parquet, PARTITION_BY (year), COMPRESSION zstd)
    """)

    con.execute(f"""
        CREATE TEMP VIEW t AS
        SELECT * FROM read_parquet('{out_dir.as_posix()}/**/*.parquet', hive_partitioning = true)
    """)
    splits = {
        r[0]: {
            "rows": r[1],
            "positives": int(r[2]),
            "positive_rate": round(r[3], 4),
            "rows_with_own_detection": r[4],
            "first_date": str(r[5]),
            "last_date": str(r[6]),
        }
        for r in con.execute("""
            SELECT split, count(*), sum(modis), avg(modis), count(*) FILTER (v_pixels > 0),
                   min(date), max(date)
            FROM t GROUP BY split
        """).fetchall()
    }
    coverage = [
        dict(zip(["year", "modis_fire_days", "in_candidates", "coverage"], r, strict=True))
        for r in con.execute("""
            SELECT year(m.date) AS y, count(*), count(t.cell_id),
                   round(count(t.cell_id) / count(*), 4)
            FROM m LEFT JOIN t USING (cell_id, date)
            GROUP BY y ORDER BY y
        """).fetchall()
    ]
    con.close()

    report = {
        "region": region,
        "satellite": satellite,
        "period": [first, str(last_sp)],
        "train_last_year": cfg["train_last_year"],
        "features": FEATURES,
        "target": TARGET,
        "splits": splits,
        "modis_coverage_by_year": coverage,
    }
    (out_dir.parent / f"{satellite}_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("region", help="folder name under data/processed/, e.g. pantanal")
    parser.add_argument("--satellite", choices=list(SATELLITES), default="SNPP")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)
    report = build_training(args.region, args.data_dir, args.satellite)

    print(
        f"{report['satellite']}: {report['period'][0]} to {report['period'][1]}, "
        f"{len(FEATURES)} features, train up to {report['train_last_year']}"
    )
    for name, s in sorted(report["splits"].items(), reverse=True):
        print(
            f"{name:>5}: {s['rows']:>10,} rows, {s['positives']:>9,} MODIS fire-days "
            f"({s['positive_rate']:.1%}), {s['first_date']} to {s['last_date']}"
        )
    print("MODIS fire-days inside the candidates (coverage):")
    for c in report["modis_coverage_by_year"]:
        print(
            f"  {c['year']}: {c['in_candidates']:>8,} / {c['modis_fire_days']:>8,}"
            f" = {c['coverage']:.1%}"
        )


if __name__ == "__main__":
    main()
