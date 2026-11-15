# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Write a self-contained HTML viewer for a region's fire-day cube.

A development tool to look at the cube, not the product frontend. The page shows
MODIS and VIIRS fire-days side by side on a map for any year/month, and a burning
activity calendar (years x weeks) for the whole region or a selected area.

Every (cell, UTC day) with fire is shipped to the browser with two flags (MODIS,
VIIRS), so all counts are computed client-side from the cube itself. Sensors are
combined with OR, as in the cube report.

Usage::

    python -m pipeline.viewer pantanal
    python -m pipeline.viewer brazil --factor 5                 # 0.5° cells
    python -m pipeline.viewer brazil --bbox -62,-24,-50,-14     # a sub-area

Output: data/processed/<region>/viewer.html (open it in a browser).
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import duckdb

from pipeline.cube import CELLS_PER_DEGREE, N_COLS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
TEMPLATE = Path(__file__).with_name("viewer_template.html")
PLACEHOLDER = "/*__CUBE_DATA__*/null"
MAX_RECORDS = 6_000_000  # ~3.4 bytes per cell-day: keeps the page around 20 MB


def export(
    region: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    factor: int = 1,
    bbox: tuple[float, float, float, float] | None = None,
) -> dict:
    """Pack the cube as {cells, first_date, n_days, records} for the page.

    records is a delta-encoded sorted list of (day * n_cells + cell) * 4 + flags,
    flags = 1 MODIS, 2 VIIRS, 3 both.
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    cube_dir = data_dir / "processed" / region / "cube"
    if not cube_dir.exists():
        raise FileNotFoundError(f"{cube_dir} not found; run `python -m pipeline.cube {region}`")

    where = ""
    if bbox:
        w, s, e, n = bbox
        where = f"WHERE cell_lon BETWEEN {w} AND {e} AND cell_lat BETWEEN {s} AND {n}"

    con = duckdb.connect()
    con.execute(f"""
        CREATE TEMP TABLE cd AS
        SELECT (cell_id // {N_COLS}) // {factor} AS r,
               (cell_id % {N_COLS}) // {factor} AS c,
               date,
               bool_or(sensor = 'MODIS') AS modis,
               bool_or(sensor = 'VIIRS') AS viirs
        FROM read_parquet('{cube_dir.as_posix()}/**/*.parquet', hive_partitioning = true)
        {where}
        GROUP BY ALL
    """)
    cells = con.execute("SELECT DISTINCT r, c FROM cd ORDER BY r, c").fetchall()
    if not cells:
        raise ValueError("no fire cell-days in the selection")
    first_date, last_date = con.execute("SELECT min(date), max(date) FROM cd").fetchone()
    n_cells = len(cells)
    con.execute(
        "CREATE TEMP TABLE idx AS SELECT r, c, row_number() OVER (ORDER BY r, c) - 1 AS i "
        "FROM (SELECT DISTINCT r, c FROM cd)"
    )
    packed = [
        row[0]
        for row in con.execute(f"""
            SELECT ((date - DATE '{first_date}') * {n_cells} + idx.i) * 4
                   + CAST(modis AS INTEGER) + 2 * CAST(viirs AS INTEGER) AS v
            FROM cd JOIN idx USING (r, c)
            ORDER BY v
        """).fetchall()
    ]
    con.close()
    if len(packed) > MAX_RECORDS:
        raise ValueError(
            f"{len(packed):,} cell-days is too many for one page; use --factor or --bbox"
        )

    deltas = [packed[0]] + [b - a for a, b in zip(packed, packed[1:], strict=False)]
    size = factor / CELLS_PER_DEGREE
    return {
        "region": region,
        "grid_degrees": round(size, 4),
        "first_date": str(first_date),
        "n_days": (last_date - first_date).days + 1,
        "cells": [
            # [south, west] corner; the page adds grid_degrees
            [round(r * size - 90, 4), round(c * size - 180, 4)]
            for r, c in cells
        ],
        "records": deltas,
        "generated": str(date.today()),
    }


def write_viewer(payload: dict, out: Path) -> Path:
    html = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in html:
        raise RuntimeError(f"{TEMPLATE.name} has no data placeholder")
    data = json.dumps(payload, separators=(",", ":"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace(PLACEHOLDER, data), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("region", help="folder name under data/processed/, e.g. pantanal")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--factor", type=int, default=1, help="merge NxN cells (5 -> 0.5°)")
    parser.add_argument("--bbox", help="west,south,east,north to keep")
    parser.add_argument("--out", type=Path, help="output HTML path")
    args = parser.parse_args(argv)

    bbox = tuple(float(v) for v in args.bbox.split(",")) if args.bbox else None
    if bbox is not None and len(bbox) != 4:
        parser.error("--bbox needs four numbers: west,south,east,north")
    payload = export(args.region, args.data_dir, args.factor, bbox)
    out = args.out or args.data_dir / "processed" / args.region / "viewer.html"
    write_viewer(payload, out)
    print(
        f"{len(payload['records']):,} fire cell-days, {len(payload['cells']):,} cells "
        f"({payload['grid_degrees']}°), {out} ({out.stat().st_size / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
