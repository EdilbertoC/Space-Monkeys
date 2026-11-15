# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Step 3: detections -> fire-day cube (0.1° cell x UTC day x satellite).

The base unit of the harmonization is the fire-day: a cell has fire on a day for
a satellite if that satellite made at least one detection there. Raw pixel
counts are kept as features, never summed across sensors.

The cube is sparse: one row per (cell, day, satellite) with at least one
detection, so every row has ``fire_day = 1``. A missing row means that satellite
detected nothing there that day (it may also not have seen the ground: FIRMS
carries no clear-sky observation record, a known limitation).

Days are UTC. Over Brazil (UTC-3/-4) a UTC day holds the Terra night overpass of
the previous local evening, then the Aqua/VIIRS night, Terra day and Aqua/VIIRS
day overpasses of the same local day, for every sensor alike.

Input:  data/processed/<region>/detections/  (from pipeline.ingest)
Output: data/processed/<region>/cube/year=YYYY/*.parquet and cube_report.json

Usage::

    python -m pipeline.cube pantanal
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import date
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"

CELLS_PER_DEGREE = 10  # 0.1° grid
N_COLS = 360 * CELLS_PER_DEGREE
# Coordinates come with at most 5 decimals; the epsilon keeps a point that sits
# exactly on a cell edge (e.g. -15.1 * 10 = -150.99999...) in the cell it names.
EPS = 1e-9

COLUMNS = [
    "cell_id",  # row * 3600 + col, row/col counted from (-90, -180)
    "cell_lat",  # cell centre
    "cell_lon",
    "date",  # UTC
    "sensor",  # MODIS | VIIRS
    "satellite",  # Terra | Aqua | SNPP | N20 | N21
    "fire_day",  # always 1 (sparse cube)
    "n_pixels",  # detections in the cell that day
    "n_overpasses",  # distinct acquisition times
    "n_day",
    "n_night",
    "n_high",  # high-confidence detections
    "n_type2",  # detections FIRMS labels "other static land source" (kept by ingest)
    "frp_sum_mw",
    "frp_max_mw",
    "scan_mean_km",  # mean along-scan pixel size: grows with view angle
    "nrt",  # true if any detection is near real time, not standard processing
    "year",  # partition key
]


def cell_index(lat: float, lon: float) -> tuple[int, int]:
    """Row and column of the 0.1° cell holding a point (same rule as the SQL)."""
    return (
        math.floor(lat * CELLS_PER_DEGREE + 90 * CELLS_PER_DEGREE + EPS),
        math.floor(lon * CELLS_PER_DEGREE + 180 * CELLS_PER_DEGREE + EPS),
    )


def cell_id(lat: float, lon: float) -> int:
    row, col = cell_index(lat, lon)
    return row * N_COLS + col


def _row_sql(lat: str) -> str:
    return f"CAST(floor({lat} * {CELLS_PER_DEGREE} + {90 * CELLS_PER_DEGREE} + {EPS}) AS INTEGER)"


def _col_sql(lon: str) -> str:
    return f"CAST(floor({lon} * {CELLS_PER_DEGREE} + {180 * CELLS_PER_DEGREE} + {EPS}) AS INTEGER)"


def build_cube(region: str, data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    det_dir = data_dir / "processed" / region / "detections"
    out_dir = data_dir / "processed" / region / "cube"
    if not det_dir.exists():
        raise FileNotFoundError(f"{det_dir} not found; run `python -m pipeline.ingest {region}`")

    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    con.execute(f"SET temp_directory = '{(data_dir / 'tmp').as_posix()}'")
    con.execute(f"""
        CREATE TEMP VIEW det AS
        SELECT *, {_row_sql("lat")} AS row_ix, {_col_sql("lon")} AS col_ix
        FROM read_parquet('{det_dir.as_posix()}/**/*.parquet', hive_partitioning = true)
    """)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    half = 0.5 / CELLS_PER_DEGREE
    con.execute(f"""
        COPY (
            SELECT
                row_ix * {N_COLS} + col_ix AS cell_id,
                round(row_ix / {CELLS_PER_DEGREE} - 90 + {half}, 2) AS cell_lat,
                round(col_ix / {CELLS_PER_DEGREE} - 180 + {half}, 2) AS cell_lon,
                date,
                any_value(sensor) AS sensor,
                satellite,
                CAST(1 AS TINYINT) AS fire_day,
                CAST(count(*) AS INTEGER) AS n_pixels,
                CAST(count(DISTINCT time) AS SMALLINT) AS n_overpasses,
                CAST(count(*) FILTER (daynight = 'D') AS INTEGER) AS n_day,
                CAST(count(*) FILTER (daynight = 'N') AS INTEGER) AS n_night,
                CAST(count(*) FILTER (confidence_class = 'h') AS INTEGER) AS n_high,
                CAST(count(*) FILTER (fire_type = 2) AS INTEGER) AS n_type2,
                sum(frp_mw) AS frp_sum_mw,
                max(frp_mw) AS frp_max_mw,
                avg(scan_km) AS scan_mean_km,
                bool_or(processing = 'NRT') AS nrt,
                year(date) AS year
            FROM det
            GROUP BY row_ix, col_ix, date, satellite
            ORDER BY date, cell_id, satellite
        ) TO '{out_dir.as_posix()}' (FORMAT parquet, PARTITION_BY (year), COMPRESSION zstd)
    """)

    con.execute(f"""
        CREATE TEMP VIEW cube AS
        SELECT * FROM read_parquet('{out_dir.as_posix()}/**/*.parquet', hive_partitioning = true)
    """)
    # Per cell-day, which sensor families saw fire. Combining sensors is an OR.
    con.execute("""
        CREATE TEMP VIEW cell_days AS
        SELECT cell_id, date, year,
               bool_or(sensor = 'MODIS') AS modis,
               bool_or(sensor = 'VIIRS') AS viirs
        FROM cube GROUP BY cell_id, date, year
    """)
    by_year = [
        dict(
            zip(["year", "modis", "viirs", "both", "either", "p_modis_given_viirs"], r, strict=True)
        )
        for r in con.execute("""
            SELECT year,
                   count(*) FILTER (modis),
                   count(*) FILTER (viirs),
                   count(*) FILTER (modis AND viirs),
                   count(*),
                   round(count(*) FILTER (modis AND viirs)
                         / nullif(count(*) FILTER (viirs), 0), 4)
            FROM cell_days GROUP BY year ORDER BY year
        """).fetchall()
    ]
    rows, cells, first, last = con.execute(
        "SELECT count(*), count(DISTINCT cell_id), min(date), max(date) FROM cube"
    ).fetchone()
    pixels = con.execute("SELECT count(*) FROM det").fetchone()[0]
    pixels_in_cube = con.execute("SELECT sum(n_pixels) FROM cube").fetchone()[0]
    con.close()

    report = {
        "region": region,
        "grid_degrees": 1 / CELLS_PER_DEGREE,
        "first_date": str(first),
        "last_date": str(last),
        "rows": rows,
        "cells_with_fire": cells,
        "detections_in": pixels,
        "detections_in_cube": int(pixels_in_cube),
        "fire_cell_days_by_year": by_year,
    }
    (out_dir.parent / "cube_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    return report


MIN_OTHER_CELL_DAYS = 100  # fire cell-days the other satellites need to call a day a gap


def observation_gaps(
    region: str, data_dir: Path = DEFAULT_DATA_DIR, min_other: int = MIN_OTHER_CELL_DAYS
) -> dict[str, list[date]]:
    """Days a satellite most likely did not observe the region, per satellite.

    FIRMS has no record of what a satellite looked at, so a day without detections
    reads as "no fire". It is called a gap when the satellite has no detection in the
    whole region while the other satellites together have at least `min_other` fire
    cell-days, between the satellite's first and last day in the cube. In a small or
    quiet region a short outage stays unseen (and matters little)."""
    cube_dir = data_dir / "processed" / region / "cube"
    rows = (
        duckdb.connect()
        .execute(
            f"""
            WITH per AS (
                SELECT satellite, date, count(*) AS n
                FROM read_parquet('{cube_dir.as_posix()}/**/*.parquet', hive_partitioning = true)
                GROUP BY ALL
            ),
            life AS (SELECT satellite, min(date) AS a, max(date) AS b FROM per GROUP BY 1),
            total AS (SELECT date, sum(n) AS n FROM per GROUP BY 1)
            SELECT l.satellite, t.date
            FROM life l
            JOIN total t ON t.date BETWEEN l.a AND l.b
            LEFT JOIN per p ON p.satellite = l.satellite AND p.date = t.date
            WHERE p.n IS NULL AND t.n >= ?
            ORDER BY 1, 2
            """,
            [min_other],
        )
        .fetchall()
    )
    gaps: dict[str, list[date]] = {}
    for satellite, day in rows:
        gaps.setdefault(satellite, []).append(day)
    return gaps


def date_runs(days: list[date]) -> list[tuple[date, date]]:
    """Sorted days as (first, last) runs of consecutive days."""
    runs: list[tuple[date, date]] = []
    for d in sorted(days):
        if runs and (d - runs[-1][1]).days == 1:
            runs[-1] = (runs[-1][0], d)
        else:
            runs.append((d, d))
    return runs


def sql_dates(days: list[date]) -> str:
    """A SQL list of dates for `date NOT IN (...)`; never empty."""
    return ", ".join(f"DATE '{d}'" for d in days) or "DATE '0001-01-01'"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("region", help="folder name under data/processed/, e.g. pantanal")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)
    report = build_cube(args.region, args.data_dir)

    print(
        f"{report['rows']:,} rows, {report['cells_with_fire']:,} cells with fire, "
        f"{report['first_date']} to {report['last_date']}"
    )
    print("Fire cell-days per year (MODIS = Terra or Aqua, VIIRS = any VIIRS satellite)")
    print(f"{'year':>6}{'MODIS':>11}{'VIIRS':>11}{'both':>11}{'either':>11}{'P(M|V)':>9}")
    for y in report["fire_cell_days_by_year"]:
        p = "" if y["p_modis_given_viirs"] is None else f"{y['p_modis_given_viirs']:.3f}"
        print(
            f"{y['year']:>6}{y['modis']:>11,}{y['viirs']:>11,}{y['both']:>11,}"
            f"{y['either']:>11,}{p:>9}"
        )


if __name__ == "__main__":
    main()
