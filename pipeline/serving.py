# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Step 10 (data side): the table the /history API reads. The API never runs a model.

One row per 0.1° cell and UTC day with any fire signal:

- modis: 1 if MODIS (Terra or Aqua) detected fire there that day (observed)
- viirs: 1 if any VIIRS satellite did (observed)
- est:   MODIS-equivalent expected fire-days from the VIIRS model (pipeline.harmonize):
         S-NPP where it has an estimate, NOAA-20 after the last S-NPP estimate and
         on the days S-NPP was not observing (pipeline.cube.observation_gaps), and
         NOAA-21 (bridged through NOAA-20, pipeline.bridge) on days neither covers;
         NULL where there is none. It runs to the last VIIRS day: the days after the
         training table come from the VIIRS-only application table, and those after
         VIIRS standard processing are provisional (meta coverage standard_until)
- fine:  the VIIRS-only layer, expected fire-days the estimating satellite saw that
         MODIS would not have: 1 - p on each cell-day the satellite itself saw fire,
         where p = est / coverage-gap factor is the calibrated P(MODIS also sees it).
         Only for the directly harmonized satellites (S-NPP, NOAA-20); NULL elsewhere.

The layers split VIIRS instead of rescaling it: est is what both sensors can see
(continuous with MODIS since 2000) and fine is what only VIIRS sees (from 2012).
Their sum is the expected number of cell-days with fire seen by MODIS or by the
satellite; meta harmonization.layers.check compares it with that observed union on
the reference years, which no model input ever saw as a target.

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
import re
import shutil
from datetime import date
from pathlib import Path

import duckdb

from pipeline.bridge import record_dir
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


LAYERS_METHOD = (
    "VIIRS split in two layers that add up, not rescaled: the common layer is the "
    "MODIS-equivalent estimate (fire both sensors can see, continuous since 2000); the "
    "VIIRS-only layer is 1 - P(MODIS also sees it) on each cell-day the estimating "
    "satellite saw fire (small fires only VIIRS detects, from 2012). Common + VIIRS-only "
    "is the expected number of cell-days with fire seen by MODIS or by that satellite."
)
# Satellites with a direct VIIRS -> MODIS model (a calibrated probability and gap
# factors); NOAA-21 is bridged through NOAA-20 as a regression, so it has no p.
LAYER_SATELLITES = ("SNPP", "N20")
REFERENCE_LAST_YEAR = 2021  # as pipeline.harmonize: MODIS orbits drift from 2022
MIN_MONTH_EXPECTED = 20.0  # as pipeline.harmonize: smaller months are too noisy for a band
MIN_BAND_MONTHS = 10

SENSOR = {s["name"]: s["sensor"] for s in SATELLITES}
FULL_NAME = {"SNPP": "S-NPP", "N20": "NOAA-20", "N21": "NOAA-21"}


def _on(condition: str, column: str) -> str:
    """A date condition written on `date`, applied to another column expression."""
    return re.sub(r"\bdate\b", column, condition)


def gap_table_sql(gap_factors: dict) -> str:
    """The coverage-gap factors of a harmonization report as an SQL table."""
    rows = [
        f"('{str(b).replace(chr(39), chr(39) * 2)}', {int(m)}, {float(f)})"
        for b, months in gap_factors.items()
        for m, f in months.items()
    ]
    if not rows:
        return "SELECT CAST(NULL AS VARCHAR) AS biome, 0 AS month, 1.0 AS g WHERE false"
    return f"SELECT * FROM (VALUES {', '.join(rows)}) AS t(biome, month, g)"


def fine_sql(sat: str, record_src: str, gap_factors: dict, days: str) -> str:
    """VIIRS-only expected fire-days per cell-day for one satellite: 1 - p on the
    cell-days it saw fire in a biome cell, p = equiv / g (0 where the record has no
    row: its estimate was below the write floor). `days` filters the dates."""
    return f"""
        SELECT c.cell_id, c.date,
               greatest(0.0, 1.0 - least(1.0, coalesce(e.equiv, 0) / coalesce(g.g, 1.0))) AS fine
        FROM (SELECT DISTINCT cell_id, CAST(date AS DATE) AS date
              FROM cube_rows WHERE satellite = '{sat}') c
        JOIN biome_t b ON b.cell_id = c.cell_id
        LEFT JOIN (SELECT cell_id, CAST(date AS DATE) AS date, equiv FROM {record_src}) e
               ON e.cell_id = c.cell_id AND e.date = c.date
        LEFT JOIN ({gap_table_sql(gap_factors)}) g
               ON g.biome = b.biome AND g.month = month(c.date)
        WHERE b.biome IS NOT NULL AND {_on(days, "c.date")}
    """


def layers_check(
    con: duckdb.DuckDBPyConnection,
    sat: str,
    record_src: str,
    gap_factors: dict,
    start: date,
    excluded: list[date],
) -> dict | None:
    """Common + VIIRS-only layer against the observed union (MODIS or the satellite
    saw fire), biome cells, standard processing, from `start` to the end of the
    reference years, without the days a satellite was not observing. Monthly totals:
    bias and WAPE for the whole region and per biome."""
    end = date(REFERENCE_LAST_YEAR, 12, 31)
    if start > end:
        return None
    days = f"date BETWEEN DATE '{start}' AND DATE '{end}' AND date NOT IN ({sql_dates(excluded)})"
    rows = con.execute(f"""
        WITH obs AS (
            SELECT c.cell_id, CAST(c.date AS DATE) AS date, b.biome
            FROM cube_rows c JOIN biome_t b ON b.cell_id = c.cell_id
            WHERE b.biome IS NOT NULL AND NOT c.nrt
              AND (c.sensor = 'MODIS' OR c.satellite = '{sat}')
              AND {_on(days, "CAST(c.date AS DATE)")}
            GROUP BY ALL
        ),
        est AS (
            SELECT CAST(e.date AS DATE) AS date, b.biome, e.equiv AS v
            FROM {record_src} e JOIN biome_t b ON b.cell_id = e.cell_id
            WHERE b.biome IS NOT NULL AND {_on(days, "CAST(e.date AS DATE)")}
        ),
        fine AS (
            SELECT f.date, b.biome, f.fine AS v
            FROM ({fine_sql(sat, record_src, gap_factors, days)}) f
            JOIN biome_t b ON b.cell_id = f.cell_id
        ),
        expected AS (SELECT * FROM est UNION ALL SELECT * FROM fine)
        SELECT coalesce(o.biome, x.biome) AS biome, coalesce(o.y, x.y) AS year,
               coalesce(o.m, x.m) AS month, coalesce(o.n, 0) AS observed,
               coalesce(x.v, 0) AS expected
        FROM (SELECT biome, year(date) AS y, month(date) AS m, count(*) AS n
              FROM obs GROUP BY ALL) o
        FULL OUTER JOIN (SELECT biome, year(date) AS y, month(date) AS m, sum(v) AS v
                         FROM expected GROUP BY ALL) x
          ON o.biome = x.biome AND o.y = x.y AND o.m = x.m
    """).fetchall()
    if not rows:
        return None

    def score(items: list[tuple]) -> dict:
        months: dict[tuple, list[float]] = {}
        for _, y, m, o, x in items:
            acc = months.setdefault((y, m), [0.0, 0.0])
            acc[0] += o
            acc[1] += x
        obs = sum(a[0] for a in months.values())
        exp = sum(a[1] for a in months.values())
        err = sum(abs(a[1] - a[0]) for a in months.values())
        return {
            "observed_union": int(obs),
            "expected": round(exp, 2),
            "bias": round(exp / obs - 1, 4) if obs else None,
            "monthly_wape": round(err / obs, 4) if obs else None,
        }

    def band(items: list[tuple]) -> list[float] | None:
        """5th/95th percentiles of observed / expected - 1 over the monthly totals."""
        r = sorted(o / x - 1 for _, _, _, o, x in items if x >= MIN_MONTH_EXPECTED)
        if len(r) < MIN_BAND_MONTHS:
            return None
        q = lambda f: r[min(len(r) - 1, max(0, round(f * (len(r) - 1))))]  # noqa: E731
        return [round(q(0.05), 4), round(q(0.95), 4)]

    by_biome: dict[str, list[tuple]] = {}
    for r in rows:
        by_biome.setdefault(r[0], []).append(r)
    bands = {b: band(items) for b, items in sorted(by_biome.items())}
    return {
        "satellite": sat,
        "period": [str(start), str(end)],
        "target": f"cell-days with fire seen by MODIS or {FULL_NAME.get(sat, sat)}",
        "total": score(rows),
        "biomes": {b: score(items) for b, items in sorted(by_biome.items())},
        "interval_90_monthly_by_biome": {b: v for b, v in bands.items() if v},
    }


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
        elif sat == "N20" and "N21" in estimate_from:
            effect = (
                "S-NPP covers these days; where it was not observing either or had ended, "
                "the VIIRS estimate uses NOAA-21, bridged through NOAA-20 (none before "
                "March 2023)."
            )
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
    con.execute(
        "CREATE TEMP VIEW cube_rows AS SELECT * FROM "
        f"read_parquet('{cube.as_posix()}/**/*.parquet', hive_partitioning = true)"
    )
    con.execute(
        "CREATE TEMP VIEW biome_t AS SELECT cell_id, biome FROM "
        f"read_parquet('{biome_file.as_posix()}')"
    )

    # VIIRS estimate: S-NPP first; NOAA-20 after the last S-NPP estimate and on the
    # days S-NPP was not observing (an outage would read as "no fire"); NOAA-21 on the
    # days neither covers.
    gaps = observation_gaps(region, data_dir)
    est_parts, est_sats, viirs_sources, est_last = [], [], [], None
    fine_parts, layer_sats, check = [], set(), None
    modis_gaps = sorted(set(gaps.get("Terra", [])) | set(gaps.get("Aqua", [])))
    covered: list[str] = []  # SQL: the day already has an estimate from an earlier satellite
    for sat in ("SNPP", "N20", "N21"):
        try:
            d = record_dir(data_dir, region, sat)  # N21: bridged through NOAA-20
        except FileNotFoundError:
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
        report = harmonized / f"{sat}_model.json"
        if sat in LAYER_SATELLITES and d == harmonized / sat and report.exists():
            gap_factors = json.loads(report.read_text(encoding="utf-8")).get("gap_factors", {})
            days = f"date BETWEEN DATE '{first}' AND DATE '{last}' AND {observing} AND {free}"
            fine_parts.append(fine_sql(sat, src, gap_factors, days))
            layer_sats.add(sat)
            if check is None:
                check = layers_check(
                    con,
                    sat,
                    src,
                    gap_factors,
                    first,
                    sorted(set(gaps.get(sat, [])) | set(modis_gaps)),
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
    fine_all = (
        " UNION ALL ".join(fine_parts)
        if fine_parts
        else "SELECT CAST(NULL AS BIGINT) AS cell_id, CAST(NULL AS DATE) AS date, "
        "CAST(NULL AS FLOAT) AS fine WHERE false"
    )
    # The layer exists while the estimate comes from a satellite that has one.
    layer_span = None
    for s in viirs_sources:
        if s["name"] not in layer_sats:
            break
        layer_span = (layer_span[0] if layer_span else s["start"], s["end"])

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
                   CAST(f.fine AS FLOAT) AS fine,
                   year(coalesce(o.date, e.date)) AS year
            FROM obs o FULL OUTER JOIN ({est_sql}) e USING (cell_id, date)
            LEFT JOIN ({fine_all}) f
              ON f.cell_id = coalesce(o.cell_id, e.cell_id) AND f.date = coalesce(o.date, e.date)
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
            "layers": {
                "status": "available" if layer_span else "unavailable",
                "method": LAYERS_METHOD,
                "satellites": sorted(layer_sats),
                "start": layer_span[0] if layer_span else None,
                "end": layer_span[1] if layer_span else None,
                "check": check,
            },
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
