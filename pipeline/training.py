# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Step 5: fire-day cube -> training table for the VIIRS -> MODIS calibration.

Question the model answers: "did MODIS detect fire in this 0.1° cell on this UTC
day?" (0/1), predicted only from what one VIIRS satellite saw plus context.

One table per VIIRS satellite. Summing the VIIRS satellites is not stationary:
NOAA-20 (2018) and NOAA-21 (2023) add pixels, so the same fire looks bigger
later, and a model trained on 2012-2017 would never see them. For S-NPP alone,
P(MODIS | S-NPP pixels) is stable while the MODIS orbits are (to 2021; see
pipeline.harmonize for the drift after that). Each table
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

The application table (`--application`) has the same rows and features where the
training table cannot reach: after its last date, including the satellite's NRT
days and the days after MODIS ends. It has no MODIS column, so it never needs MODIS.

Input:  data/processed/<region>/cube/
Output: data/processed/<region>/training/<SAT>/year=YYYY/*.parquet
        data/processed/<region>/training/<SAT>_report.json
        data/processed/<region>/application/<SAT>/year=YYYY/*.parquet (--application)

Usage::

    python -m pipeline.training pantanal              # S-NPP
    python -m pipeline.training pantanal --satellite N20
    python -m pipeline.training brazil --satellite SNPP --application
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date, timedelta
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
KEYS = ["cell_id", "date", "year", "split", "biome"]
DIAGNOSTICS = ["modis_pixels", "terra", "aqua"]  # never use as features


def _offsets(radius: int) -> str:
    rows = [
        f"({dr}, {dc}, {max(abs(dr), abs(dc))})"
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
    ]
    return "(VALUES " + ", ".join(rows) + ") o(dr, dc, ring)"


def _cube_view(con: duckdb.DuckDBPyConnection, cube_dir: Path, nrt: bool) -> None:
    """Temp view `cube`, with or without the NRT rows."""
    where = "" if nrt else "WHERE NOT nrt"
    con.execute(f"""
        CREATE TEMP VIEW cube AS
        SELECT * FROM read_parquet('{cube_dir.as_posix()}/**/*.parquet', hive_partitioning = true)
        {where}
    """)


def _viirs_tables(con: duckdb.DuckDBPyConnection, satellite: str, first: str, last: str) -> None:
    """Temp tables from the satellite alone, for the days first..last:

    - v: its own cell-days (read from 3 days before `first`, so the lags and the
      candidate rule of the first days see the days before them);
    - ctx: neighbourhood and lag sums, and the candidate flag, per cell-day."""
    con.execute(f"""
        CREATE TEMP TABLE v AS
        SELECT cell_id, date,
               n_pixels AS v_pixels, frp_sum_mw AS v_frp_sum, frp_max_mw AS v_frp_max,
               n_day AS v_day, n_night AS v_night, n_high AS v_high,
               n_overpasses AS v_overpasses, scan_mean_km AS v_scan_mean
        FROM cube
        WHERE satellite = '{satellite}'
          AND date BETWEEN DATE '{first}' - INTERVAL 3 DAY AND DATE '{last}'
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
        SELECT cell_id, CAST(date AS DATE) AS date,
               coalesce(sum(v_pixels) FILTER (l = 0 AND ring = 1), 0) AS nb3_pixels,
               count(*) FILTER (l = 0 AND ring = 1) AS nb3_cells,
               coalesce(sum(v_pixels) FILTER (l = 0 AND ring = 2), 0) AS nb5_pixels,
               count(*) FILTER (l = 0 AND ring = 2) AS nb5_cells,
               coalesce(sum(v_pixels) FILTER (l = 1 AND ring = 0), 0) AS lag1_pixels,
               coalesce(sum(v_pixels) FILTER (l IN (2, 3) AND ring = 0), 0) AS lag23_pixels,
               coalesce(sum(v_pixels) FILTER (l = 1 AND ring = 1), 0) AS lag1_nb3_pixels,
               bool_or(ring <= 1 AND l <= 1) AS candidate
        FROM spread
        WHERE date BETWEEN DATE '{first}' AND DATE '{last}'
        GROUP BY cell_id, CAST(date AS DATE)
    """)


# The feature columns of a candidate row, from ctx c and v (same names as FEATURES).
FEATURE_SQL = f"""
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
    round((c.cell_id % {N_COLS}) / 10 - 180 + 0.05, 2) AS cell_lon
"""


def _biome_source(data_dir: Path, region: str) -> str:
    """Biome per cell (pipeline.biomes), a label for evaluation, never a feature."""
    biome_file = data_dir / "processed" / region / "cell_biome.parquet"
    if biome_file.exists():
        return f"(SELECT cell_id, biome FROM read_parquet('{biome_file.as_posix()}'))"
    return "(SELECT CAST(NULL AS BIGINT) AS cell_id, CAST(NULL AS VARCHAR) AS biome)"


def _check(satellite: str, cube_dir: Path, region: str) -> None:
    if satellite not in SATELLITES:
        raise ValueError(f"satellite must be one of {', '.join(SATELLITES)}")
    if not cube_dir.exists():
        raise FileNotFoundError(f"{cube_dir} not found; run `python -m pipeline.cube {region}`")


def build_training(region: str, data_dir: Path = DEFAULT_DATA_DIR, satellite: str = "SNPP") -> dict:
    cfg = SATELLITES.get(satellite, {})
    cube_dir = data_dir / "processed" / region / "cube"
    out_dir = data_dir / "processed" / region / "training" / satellite
    _check(satellite, cube_dir, region)

    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{(data_dir / 'tmp').as_posix()}'")
    _cube_view(con, cube_dir, nrt=False)
    # Last date with standard processing for both MODIS and this satellite (least()
    # skips NULL, so a missing sensor is checked first: no MODIS would read as no fire).
    last_modis, last_sat = con.execute(
        "SELECT max(date) FILTER (sensor = 'MODIS'), max(date) FILTER (satellite = ?) FROM cube",
        [satellite],
    ).fetchone()
    if last_modis is None or last_sat is None:
        raise ValueError(f"no standard-processing {satellite} and MODIS data in {cube_dir}")
    last_sp = min(last_modis, last_sat)
    first = cfg["start"]
    period = f"date BETWEEN DATE '{first}' AND DATE '{last_sp}'"

    _viirs_tables(con, satellite, first, str(last_sp))
    con.execute(f"""
        CREATE TEMP TABLE m AS
        SELECT cell_id, date,
               sum(n_pixels) AS modis_pixels,
               bool_or(satellite = 'Terra') AS terra,
               bool_or(satellite = 'Aqua') AS aqua
        FROM cube WHERE sensor = 'MODIS' AND {period}
        GROUP BY cell_id, date
    """)
    biome_source = _biome_source(data_dir, region)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    con.execute(f"""
        COPY (
            SELECT
                c.cell_id, c.date, year(c.date) AS year,
                CASE WHEN year(c.date) <= {cfg["train_last_year"]} THEN 'train' ELSE 'test' END
                    AS split,
                {FEATURE_SQL},
                CAST(m.cell_id IS NOT NULL AS TINYINT) AS modis,
                coalesce(m.modis_pixels, 0) AS modis_pixels,
                coalesce(m.terra, false) AS terra,
                coalesce(m.aqua, false) AS aqua,
                b.biome
            FROM ctx c
            LEFT JOIN v USING (cell_id, date)
            LEFT JOIN m USING (cell_id, date)
            LEFT JOIN {biome_source} b USING (cell_id)
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


def build_application(
    region: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    satellite: str = "SNPP",
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """The rows and features of the training table where it cannot reach, from VIIRS
    alone: no MODIS column, so it can be built after MODIS ends.

    By default it runs from the day after the training table's last date to the
    satellite's last day in the cube. It includes the satellite's NRT days: their
    estimate is provisional, as the API already marks every period after standard
    processing. The model, calibration and gap stay those of the reference years
    (pipeline.harmonize applies them).

    Output: data/processed/<region>/application/<SAT>/year=YYYY/*.parquet and
    <SAT>_report.json.
    """
    cube_dir = data_dir / "processed" / region / "cube"
    out_dir = data_dir / "processed" / region / "application" / satellite
    _check(satellite, cube_dir, region)

    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{(data_dir / 'tmp').as_posix()}'")
    _cube_view(con, cube_dir, nrt=True)
    sat_last, sat_standard_until = con.execute(
        "SELECT max(date), max(date) FILTER (NOT nrt) FROM cube WHERE satellite = ?",
        [satellite],
    ).fetchone()
    if sat_last is None:
        raise ValueError(f"no {satellite} data in {cube_dir}")
    if start is None:
        report_file = data_dir / "processed" / region / "training" / f"{satellite}_report.json"
        if report_file.exists():
            train_last = json.loads(report_file.read_text(encoding="utf-8"))["period"][1]
            start = str(date.fromisoformat(train_last) + timedelta(days=1))
        else:
            start = SATELLITES[satellite]["start"]
    end = end or str(sat_last)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    rows = 0
    if start <= end:
        _viirs_tables(con, satellite, start, end)
        con.execute(f"""
            COPY (
                SELECT c.cell_id, c.date, year(c.date) AS year, {FEATURE_SQL}, b.biome
                FROM ctx c
                LEFT JOIN v USING (cell_id, date)
                LEFT JOIN {_biome_source(data_dir, region)} b USING (cell_id)
                WHERE c.candidate
                ORDER BY c.date, c.cell_id
            ) TO '{out_dir.as_posix()}' (FORMAT parquet, PARTITION_BY (year), COMPRESSION zstd)
        """)
        rows = con.execute(
            f"SELECT count(*) FROM read_parquet('{out_dir.as_posix()}/**/*.parquet')"
        ).fetchone()[0]
    con.close()

    report = {
        "region": region,
        "satellite": satellite,
        "period": [start, end],
        "standard_until": str(sat_standard_until) if sat_standard_until else None,
        "rows": rows,
        "features": FEATURES,
    }
    (out_dir.parent / f"{satellite}_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("region", help="folder name under data/processed/, e.g. pantanal")
    parser.add_argument("--satellite", choices=list(SATELLITES), default="SNPP")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--application",
        action="store_true",
        help="build the VIIRS-only application table after the training table's last date",
    )
    args = parser.parse_args(argv)
    if args.application:
        r = build_application(args.region, args.data_dir, args.satellite)
        print(
            f"{r['satellite']} application table: {r['period'][0]} to {r['period'][1]}, "
            f"{r['rows']:,} rows (standard processing until {r['standard_until']}, NRT after)"
        )
        return
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
