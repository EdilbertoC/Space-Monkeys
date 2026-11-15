# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Step 10 (data side): the table the /history API reads. The API never runs a model.

One row per 0.1° cell and UTC day with any fire signal:

- modis: 1 if MODIS (Terra or Aqua) detected fire there that day (observed)
- viirs: 1 if any VIIRS satellite did (observed)
- est:   MODIS-equivalent expected fire-days from the VIIRS model (pipeline.harmonize):
         S-NPP where it has an estimate, NOAA-20 after the last S-NPP estimate and
         on the days S-NPP was not observing (pipeline.cube.observation_gaps);
         NULL where there is none. It runs to the last VIIRS day: the days after the
         training table come from the VIIRS-only application table, and those after
         VIIRS standard processing are provisional (meta coverage standard_until)

plus meta.json: coverage per sensor, biomes, harmonization method and switch date,
model scores and monthly intervals per biome, known data gaps (from the FIRMS Readme
and the days a satellite was not observing, found in the data) and limits.

Output: data/processed/<region>/serving/cell_day/year=YYYY/*.parquet, meta.json and
cell_biome.parquet (copied, so the API needs only this folder).

Usage::

    python -m pipeline.serving brazil
    python -m pipeline.serving brazil --switch-date 2027-01-01   # serve VIIRS estimate from then
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

import duckdb

from pipeline.cube import date_runs, observation_gaps, sql_dates

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"

# From the FIRMS archive download form (data availability per satellite).
SATELLITES = [
    {"name": "Terra", "sensor": "MODIS", "start": "2000-11-01", "end": None},
    {"name": "Aqua", "sensor": "MODIS", "start": "2002-07-04", "end": None},
    {"name": "SNPP", "sensor": "VIIRS", "start": "2012-01-20", "end": "2026-11-02"},
    {"name": "N20", "sensor": "VIIRS", "start": "2018-04-01", "end": None},
    {"name": "N21", "sensor": "VIIRS", "start": "2023-03-01", "end": None},
]
# From the FIRMS download Readme.
KNOWN_GAPS = [
    {
        "sensor": "MODIS",
        "period": "2001-06/2001-07",
        "note": "Missing data, end of June to beginning of July 2001 (FIRMS download Readme).",
    },
    {
        "sensor": "MODIS",
        "period": "2002",
        "note": "Some data missing throughout the year (FIRMS download Readme).",
    },
    {
        "sensor": "MODIS",
        "period": "2007-08",
        "note": "Some data missing from mid-August 2007 (FIRMS download Readme).",
    },
    {
        "sensor": "MODIS",
        "period": "2009-04-21/2009-04-22",
        "note": "Missing for part of 21 April and all of 22 April 2009 (FIRMS download Readme).",
    },
]
LIMITS = {"max_area_cells": 50000, "max_map_cells": 20000, "max_day_buckets": 1000}
METHOD = (
    "MODIS observed while MODIS delivers data; afterwards MODIS-equivalent expected fire "
    "cell-days estimated from VIIRS (LightGBM cross-fitted by year blocks on the years of "
    "stable MODIS orbits, calibrated per biome, coverage gap added back)."
)


SENSOR = {s["name"]: s["sensor"] for s in SATELLITES}
FULL_NAME = {"SNPP": "S-NPP", "N20": "NOAA-20", "N21": "NOAA-21"}


def detected_gaps(gaps: dict[str, list[date]], estimate_from: set[str]) -> list[dict]:
    """known_gaps entries for the days a satellite was not observing (from the data).

    estimate_from: VIIRS satellites with an estimate, to say what covers an S-NPP gap."""
    out = []
    for sat in sorted(gaps, key=lambda s: (SENSOR.get(s, ""), s)):
        name = FULL_NAME.get(sat, sat)
        if SENSOR.get(sat) == "MODIS":
            effect = "MODIS counts are low on these days; they are left out of the model."
        elif sat == "SNPP" and "N20" in estimate_from:
            effect = "the VIIRS estimate uses NOAA-20 on these days (none before April 2018)."
        else:
            effect = "VIIRS counts are low on these days."
        for a, b in date_runs(gaps[sat]):
            out.append(
                {
                    "sensor": SENSOR.get(sat, sat),
                    "period": str(a) if a == b else f"{a}/{b}",
                    "note": f"No {name} detection in the whole region while other satellites "
                    f"saw fire (satellite not observing); {effect}",
                }
            )
    return out


def build_serving(
    region: str, data_dir: Path = DEFAULT_DATA_DIR, switch_date: str | None = None
) -> dict:
    processed = data_dir / "processed" / region
    cube = processed / "cube"
    harmonized = processed / "harmonized"
    biome_file = processed / "cell_biome.parquet"
    out = processed / "serving"
    for need in (cube, biome_file):
        if not need.exists():
            raise FileNotFoundError(f"{need} not found")

    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{(data_dir / 'tmp').as_posix()}'")
    con.execute(f"""
        CREATE TEMP VIEW obs AS
        SELECT cell_id, date,
               CAST(bool_or(sensor = 'MODIS') AS TINYINT) AS modis,
               CAST(bool_or(sensor = 'VIIRS') AS TINYINT) AS viirs
        FROM read_parquet('{cube.as_posix()}/**/*.parquet', hive_partitioning = true)
        GROUP BY cell_id, date
    """)

    # VIIRS estimate: S-NPP first; NOAA-20 after the last S-NPP estimate and on the
    # days S-NPP was not observing (an outage would read as "no fire").
    gaps = observation_gaps(region, data_dir)
    est_parts, est_sats, viirs_sources, est_last = [], [], [], None
    covered: list[str] = []  # SQL: the day already has an estimate from an earlier satellite
    for sat in ("SNPP", "N20"):
        d = harmonized / sat
        if not d.exists():
            continue
        src = f"read_parquet('{d.as_posix()}/**/*.parquet', hive_partitioning = true)"
        first, last = con.execute(
            f"SELECT CAST(min(date) AS DATE), CAST(max(date) AS DATE) FROM {src}"
        ).fetchone()
        est_sats.append(sat)
        observing = f"CAST(date AS DATE) NOT IN ({sql_dates(gaps.get(sat, []))})"
        free = " AND ".join(f"NOT ({c})" for c in covered) or "true"
        est_parts.append(
            f"SELECT cell_id, CAST(date AS DATE) AS date, equiv FROM {src} "
            f"WHERE equiv > 0 AND {observing} AND {free}"
        )
        covered.append(
            f"CAST(date AS DATE) BETWEEN DATE '{first}' AND DATE '{last}' AND {observing}"
        )
        if est_last is None or last > est_last:
            start = first if est_last is None else date.fromordinal(est_last.toordinal() + 1)
            viirs_sources.append({"name": sat, "start": str(start), "end": str(last)})
            est_last = last
    est_sql = (
        " UNION ALL ".join(est_parts)
        if est_parts
        else "SELECT CAST(NULL AS BIGINT) AS cell_id, CAST(NULL AS DATE) AS date, "
        "CAST(NULL AS FLOAT) AS equiv WHERE false"
    )

    if out.exists():
        shutil.rmtree(out)
    (out / "cell_day").mkdir(parents=True)
    con.execute(f"""
        COPY (
            SELECT coalesce(o.cell_id, e.cell_id) AS cell_id,
                   coalesce(o.date, e.date) AS date,
                   coalesce(o.modis, 0) AS modis,
                   coalesce(o.viirs, 0) AS viirs,
                   CAST(e.equiv AS FLOAT) AS est,
                   year(coalesce(o.date, e.date)) AS year
            FROM obs o FULL OUTER JOIN ({est_sql}) e USING (cell_id, date)
            ORDER BY year, cell_id, date
        ) TO '{(out / "cell_day").as_posix()}'
          (FORMAT parquet, PARTITION_BY (year), COMPRESSION zstd, ROW_GROUP_SIZE 100000)
    """)
    shutil.copy(biome_file, out / "cell_biome.parquet")

    cov = con.execute(f"""
        SELECT min(date) FILTER (sensor = 'MODIS'),
               max(date) FILTER (sensor = 'MODIS' AND NOT nrt),
               max(date) FILTER (sensor = 'MODIS'),
               min(date) FILTER (sensor = 'VIIRS'),
               max(date) FILTER (sensor = 'VIIRS' AND NOT nrt),
               max(date) FILTER (sensor = 'VIIRS')
        FROM read_parquet('{cube.as_posix()}/**/*.parquet', hive_partitioning = true)
    """).fetchone()
    biomes = con.execute(f"""
        SELECT biome, count(*) FROM read_parquet('{biome_file.as_posix()}')
        WHERE biome IS NOT NULL GROUP BY 1 ORDER BY 1
    """).fetchall()

    scores, intervals = {}, {}
    model_json = harmonized / "SNPP_model.json"
    if model_json.exists():
        model = json.loads(model_json.read_text(encoding="utf-8"))
        scores = {
            area: {"monthly_wape": s["monthly_wape"], "bias": s["bias"]}
            for area, s in model["scores_out_of_fold"].items()
        }
        intervals = model["interval_90"]

    meta = {
        "region": region,
        "unit": "fire_cell_days",
        "grid_degrees": 0.1,
        "time_basis": "UTC",
        "coverage": {
            "modis": {"start": str(cov[0]), "standard_until": str(cov[1]), "end": str(cov[2])},
            "viirs": {"start": str(cov[3]), "standard_until": str(cov[4]), "end": str(cov[5])},
            "satellites": SATELLITES,
        },
        "biomes": [{"name": b, "cells": n} for b, n in biomes],
        "harmonization": {
            "status": "available",
            "method": METHOD,
            "switch_date": switch_date,
            "viirs_satellites": viirs_sources,
            "estimate_until": str(est_last) if est_last else None,
            "model": "lightgbm-crossfit-isotonic-per-biome" if est_parts else None,
            "version": str(date.today()),
            "test_metrics": scores,
            "interval_90_monthly_by_biome": intervals,
        },
        "known_gaps": KNOWN_GAPS + detected_gaps(gaps, estimate_from=set(est_sats)),
        "limits": LIMITS,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta["rows"] = con.execute(
        f"SELECT count(*) FROM read_parquet('{(out / 'cell_day').as_posix()}/**/*.parquet')"
    ).fetchone()[0]
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("region", nargs="?", default="brazil")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--switch-date", help="first day served from the VIIRS estimate instead of MODIS"
    )
    args = parser.parse_args(argv)
    meta = build_serving(args.region, args.data_dir, args.switch_date)
    cov = meta["coverage"]
    print(
        f"{meta['rows']:,} cell-days; MODIS {cov['modis']['start']}..{cov['modis']['end']}, "
        f"VIIRS {cov['viirs']['start']}..{cov['viirs']['end']}; estimate from "
        f"{meta['harmonization']['viirs_satellites']} until "
        f"{meta['harmonization']['estimate_until']}"
    )


if __name__ == "__main__":
    main()
