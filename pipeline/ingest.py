# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Step 2: FIRMS archive downloads (zip of CSVs) -> one clean Parquet detections table.

Input: the zips from https://firms.modaps.eosdis.nasa.gov/download/ saved in
``data/raw/firms/<region>/``. Each zip holds a ``fire_archive_*.csv`` (standard
processing) and a ``fire_nrt_*.csv`` (near real time, covering the months the
archive has not reached yet).

Output: ``data/processed/<region>/detections/year=YYYY/*.parquet`` plus an
``ingest_report.json`` with every count the methodology needs to cite.

The first columns match ``app/realtime/detections.py`` (checked by a test); the
extra columns keep what the harmonization needs (processing level, confidence
class, FIRMS fire type, brightness).

Usage::

    python -m pipeline.ingest pantanal
    python -m pipeline.ingest brazil --min-confidence-class l   # keep low confidence
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"

# FIRMS product code in the file name -> (sensor family, FIRMS source prefix).
PRODUCTS = {
    "M-C61": ("MODIS", "MODIS"),
    "SV-C2": ("VIIRS", "VIIRS_SNPP"),
    "J1V-C2": ("VIIRS", "VIIRS_NOAA20"),
    "J2V-C2": ("VIIRS", "VIIRS_NOAA21"),
}
_PRODUCT_RE = "|".join(map(re.escape, PRODUCTS))
CSV_NAME = re.compile(rf"^fire_(archive|nrt)_({_PRODUCT_RE})_\d+\.csv$")

# Same scale as app/realtime/firms.py so live and historical confidences agree.
VIIRS_CONFIDENCE = {"l": 0.3, "n": 0.6, "h": 0.9}
CLASS_RANK = {"l": 0, "n": 1, "h": 2}

# FIRMS `type`: 0 vegetation fire, 1 active volcano, 2 other static land source,
# 3 offshore. Static sites are taken from VIIRS type 1/2 labels only and applied
# to every sensor and processing level (see ingest() for why). Type 3 is kept on
# purpose: in the Pantanal it marks fires over the seasonal wetland water mask
# (same season as type 0). Real offshore sources (oil platforms) are removed
# later by clipping to the land area of the region.
STATIC_TYPES = (1, 2)

COLUMNS = [
    # common schema (app/realtime/detections.py)
    "detection_id",
    "source",
    "sensor",
    "satellite",
    "time",
    "lat",
    "lon",
    "scan_km",
    "track_km",
    "frp_mw",
    "confidence",
    "daynight",
    # historical extras
    "processing",  # SP (standard processing archive) | NRT
    "version",  # as reported, e.g. 6.03, 2, 2.0NRT
    "confidence_class",  # l | n | h for both sensors
    "fire_type",  # FIRMS type; NULL for NRT rows (not provided)
    "brightness_k",  # MODIS band 21/22, VIIRS I4 brightness temperature
    "bright_t31_k",  # MODIS band 31, VIIRS I5 brightness temperature
    "date",  # UTC acquisition date
    "year",  # partition key
]


@dataclass
class IngestReport:
    region: str
    files: list[str]
    min_confidence_class: str
    by_source: list[dict]
    totals: dict

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, default=str)


def extract_csvs(raw_dir: Path, staging: Path) -> list[Path]:
    """Copy the FIRMS CSVs out of every zip in raw_dir into staging.

    Only members whose name matches the FIRMS pattern are written, so a zip
    cannot place files anywhere else. Already-extracted files are reused.
    """
    staging.mkdir(parents=True, exist_ok=True)
    found: list[Path] = []
    for zpath in sorted(raw_dir.glob("*.zip")):
        with zipfile.ZipFile(zpath) as zf:
            for info in zf.infolist():
                if not CSV_NAME.match(info.filename):
                    continue
                dest = staging / info.filename
                if not dest.exists() or dest.stat().st_size != info.file_size:
                    with zf.open(info) as src, open(dest, "wb") as out:
                        shutil.copyfileobj(src, out)
                found.append(dest)
    found += [p for p in sorted(staging.glob("*.csv")) if CSV_NAME.match(p.name) and p not in found]
    if not found:
        raise FileNotFoundError(f"No FIRMS archive CSVs found in {raw_dir}")
    return sorted(found)


def _sql_list(paths: list[Path]) -> str:
    return "[" + ", ".join("'" + p.as_posix().replace("'", "''") + "'" for p in paths) + "]"


def _brightness_columns(con: duckdb.DuckDBPyConnection) -> tuple[str, str]:
    cols = {row[0] for row in con.execute("DESCRIBE raw").fetchall()}
    b4 = [c for c in ("brightness", "bright_ti4") if c in cols]
    b5 = [c for c in ("bright_t31", "bright_ti5") if c in cols]
    to_expr = lambda names: f"coalesce({', '.join(names)})" if names else "NULL"  # noqa: E731
    return to_expr(b4), to_expr(b5)


def ingest(
    region: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    min_confidence_class: str = "n",
) -> IngestReport:
    if min_confidence_class not in CLASS_RANK:
        raise ValueError("min_confidence_class must be one of l, n, h")
    raw_dir = data_dir / "raw" / "firms" / region
    out_dir = data_dir / "processed" / region / "detections"
    csvs = extract_csvs(raw_dir, raw_dir / "csv")

    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    con.execute(f"SET temp_directory = '{(data_dir / 'tmp').as_posix()}'")

    con.execute(f"""
        CREATE TEMP VIEW raw AS
        SELECT * FROM read_csv({_sql_list(csvs)},
            header = true, union_by_name = true, filename = true,
            types = {{'acq_date': 'DATE', 'acq_time': 'VARCHAR', 'confidence': 'VARCHAR',
                      'version': 'VARCHAR', 'satellite': 'VARCHAR', 'daynight': 'VARCHAR'}})
    """)
    has_type = "type" in {r[0] for r in con.execute("DESCRIBE raw").fetchall()}
    type_expr = "TRY_CAST(type AS TINYINT)" if has_type else "CAST(NULL AS TINYINT)"
    bright4, bright5 = _brightness_columns(con)

    products = ", ".join(f"('{k}', '{v[0]}', '{v[1]}')" for k, v in PRODUCTS.items())
    viirs_conf = ", ".join(f"'{k}': {v}" for k, v in VIIRS_CONFIDENCE.items())
    con.execute(f"""
        CREATE TEMP VIEW parsed AS
        WITH base AS (
            SELECT
                r.*,
                regexp_extract(r.filename, 'fire_(archive|nrt)_', 1) AS kind,
                regexp_extract(r.filename, 'fire_(?:archive|nrt)_([A-Z0-9-]+)_', 1) AS product,
                lpad(trim(r.acq_time), 4, '0') AS hhmm,
                lower(trim(r.confidence)) AS conf_txt,
                {type_expr} AS fire_type
            FROM raw r
        )
        SELECT
            p.sensor,
            'FIRMS:' || p.prefix || '_' || CASE b.kind WHEN 'archive' THEN 'SP' ELSE 'NRT' END
                AS source,
            CASE b.kind WHEN 'archive' THEN 'SP' ELSE 'NRT' END AS processing,
            trim(b.satellite) AS satellite,
            b.acq_date + make_time(CAST(substr(b.hhmm, 1, 2) AS BIGINT),
                                   CAST(substr(b.hhmm, 3, 2) AS BIGINT), 0) AS ts,
            CAST(b.latitude AS DOUBLE) AS lat,
            CAST(b.longitude AS DOUBLE) AS lon,
            CAST(b.scan AS DOUBLE) AS scan_km,
            CAST(b.track AS DOUBLE) AS track_km,
            TRY_CAST(b.frp AS DOUBLE) AS frp_mw,
            CASE WHEN p.sensor = 'MODIS' THEN
                CASE WHEN TRY_CAST(b.conf_txt AS INTEGER) < 30 THEN 'l'
                     WHEN TRY_CAST(b.conf_txt AS INTEGER) < 80 THEN 'n'
                     WHEN TRY_CAST(b.conf_txt AS INTEGER) IS NOT NULL THEN 'h' END
            ELSE substr(b.conf_txt, 1, 1) END AS confidence_class,
            CASE WHEN p.sensor = 'MODIS' THEN TRY_CAST(b.conf_txt AS DOUBLE) / 100.0
                 ELSE MAP {{{viirs_conf}}}[substr(b.conf_txt, 1, 1)] END AS confidence,
            coalesce(trim(b.daynight), '') AS daynight,
            trim(b.version) AS version,
            b.fire_type,
            TRY_CAST({bright4} AS DOUBLE) AS brightness_k,
            TRY_CAST({bright5} AS DOUBLE) AS bright_t31_k,
            b.acq_date AS date
        FROM base b
        JOIN (VALUES {products}) p(product, sensor, prefix) USING (product)
    """)

    # Static land sources (volcanoes, industry). One mask for every row:
    # - NRT rows carry no `type`, so the archive labels must be reused anyway.
    # - VIIRS (375 m) labels a few compact, year-round sites. MODIS C6.1 type 2
    #   labels in the Pantanal are scattered and peak in the fire season, i.e.
    #   they behave like vegetation fires, so they are not used to build the mask.
    # A row is static when it falls within ~1 km (one 0.01° key in any direction)
    # of a VIIRS type 1/2 detection. A heuristic; documented in the methodology.
    con.execute(f"""
        CREATE TEMP TABLE static_keys AS
        SELECT DISTINCT CAST(round(lat * 100) AS INTEGER) + dy AS ky,
                        CAST(round(lon * 100) AS INTEGER) + dx AS kx
        FROM parsed, (VALUES (-1), (0), (1)) a(dy), (VALUES (-1), (0), (1)) b(dx)
        WHERE sensor = 'VIIRS' AND fire_type IN {STATIC_TYPES}
    """)

    min_rank = CLASS_RANK[min_confidence_class]
    ranks = ", ".join(f"'{k}': {v}" for k, v in CLASS_RANK.items())
    con.execute(f"""
        CREATE TEMP VIEW flagged AS
        SELECT p.*,
            s.ky IS NOT NULL AS drop_static,
            coalesce(MAP {{{ranks}}}[p.confidence_class], -1) < {min_rank} AS drop_low_conf
        FROM parsed p
        LEFT JOIN static_keys s
          ON s.ky = CAST(round(p.lat * 100) AS INTEGER)
         AND s.kx = CAST(round(p.lon * 100) AS INTEGER)
    """)

    by_source = [
        dict(
            zip(
                [
                    "source",
                    "first_date",
                    "last_date",
                    "rows",
                    "static",
                    "low_confidence",
                    "kept",
                    "kept_firms_type_2",
                ],
                row,
                strict=True,
            )
        )
        for row in con.execute(f"""
            SELECT source, min(date), max(date), count(*),
                   count(*) FILTER (drop_static),
                   count(*) FILTER (drop_low_conf AND NOT drop_static),
                   count(*) FILTER (NOT drop_static AND NOT drop_low_conf),
                   count(*) FILTER (NOT drop_static AND NOT drop_low_conf
                                    AND fire_type IN {STATIC_TYPES})
            FROM flagged GROUP BY source ORDER BY source
        """).fetchall()
    ]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    con.execute(f"""
        COPY (
            SELECT
                substr(sha1(source || '|' || strftime(ts, '%Y%m%dT%H%M%S') || '|'
                            || CAST(round(lat, 4) AS VARCHAR) || '|'
                            || CAST(round(lon, 4) AS VARCHAR)), 1, 16) AS detection_id,
                source, sensor, satellite,
                timezone('UTC', ts) AS time,
                lat, lon, scan_km, track_km, frp_mw, confidence, daynight,
                processing, version, confidence_class, fire_type,
                brightness_k, bright_t31_k, date, year(date) AS year
            FROM flagged
            WHERE NOT drop_static AND NOT drop_low_conf
            QUALIFY row_number() OVER (PARTITION BY detection_id ORDER BY version) = 1
            ORDER BY time
        ) TO '{out_dir.as_posix()}' (FORMAT parquet, PARTITION_BY (year), COMPRESSION zstd)
    """)

    written = con.execute(
        f"SELECT count(*) FROM read_parquet('{out_dir.as_posix()}/**/*.parquet')"
    ).fetchone()[0]
    kept = sum(s["kept"] for s in by_source)
    totals = {
        "rows_read": sum(s["rows"] for s in by_source),
        "dropped_static": sum(s["static"] for s in by_source),
        "dropped_low_confidence": sum(s["low_confidence"] for s in by_source),
        "dropped_duplicates": kept - written,
        "rows_written": written,
        "kept_firms_type_2": sum(s["kept_firms_type_2"] for s in by_source),
        "static_keys": con.execute("SELECT count(*) FROM static_keys").fetchone()[0],
    }
    report = IngestReport(
        region=region,
        files=[p.name for p in csvs],
        min_confidence_class=min_confidence_class,
        by_source=by_source,
        totals=totals,
    )
    (out_dir.parent / "ingest_report.json").write_text(report.to_json(), encoding="utf-8")
    con.close()
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("region", help="folder name under data/raw/firms/, e.g. pantanal")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--min-confidence-class",
        choices=list(CLASS_RANK),
        default="n",
        help="lowest confidence class kept (MODIS l<30, n 30-79, h>=80; VIIRS l/n/h)",
    )
    args = parser.parse_args(argv)
    report = ingest(args.region, args.data_dir, args.min_confidence_class)

    print(
        f"{'source':<22}{'first':>12}{'last':>12}{'rows':>13}{'static':>11}{'low':>11}{'kept':>13}"
    )
    for s in report.by_source:
        print(
            f"{s['source']:<22}{s['first_date']!s:>12}{s['last_date']!s:>12}"
            f"{s['rows']:>13,}{s['static']:>11,}{s['low_confidence']:>11,}{s['kept']:>13,}"
        )
    print(json.dumps(report.totals, indent=2))


if __name__ == "__main__":
    main()
