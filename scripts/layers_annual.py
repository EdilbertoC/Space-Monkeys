# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Print the VIIRS-only layer check and the yearly totals of every series, Brazil cells.

Reads the serving table written by `python -m pipeline.serving brazil`:

- LAYERS / CHECK / BANDS: harmonization.layers from meta.json, the check of
  `detected` (common + VIIRS-only) against the cell-days where MODIS or S-NPP saw
  fire on the reference years, and its 90% band per biome;
- a CSV table per year: modis (observed), viirs_raw (observed, any VIIRS
  satellite), est (common layer, MODIS-equivalent) and viirs_only (VIIRS-only layer).

Usage::

    python scripts/layers_annual.py
    python scripts/layers_annual.py --serving data/processed/brazil/serving
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

DEFAULT = Path(__file__).resolve().parents[1] / "data" / "processed" / "brazil" / "serving"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--serving", type=Path, default=DEFAULT)
    args = parser.parse_args(argv)
    serving = args.serving

    meta = json.loads((serving / "meta.json").read_text(encoding="utf-8"))
    layers = meta["harmonization"].get("layers")
    if not layers:
        raise SystemExit(
            "NO LAYERS in meta.json: update the code (git pull) and run "
            "`python -m pipeline.serving brazil` again"
        )
    print("LAYERS", layers["status"], layers["start"], layers["end"], layers["satellites"])
    check = layers.get("check") or {}
    print("CHECK TOTAL", json.dumps(check.get("total")))
    for name, s in (check.get("biomes") or {}).items():
        print("CHECK", name, json.dumps(s))
    print("BANDS", json.dumps(check.get("interval_90_monthly_by_biome")))

    cell_day = (serving / "cell_day").as_posix()
    biome = (serving / "cell_biome.parquet").as_posix()
    table = duckdb.sql(f"""
        SELECT year, sum(modis) AS modis, sum(viirs) AS viirs_raw,
               round(sum(est)) AS est, round(sum(fine)) AS viirs_only
        FROM read_parquet('{cell_day}/**/*.parquet', hive_partitioning = true) c
        JOIN read_parquet('{biome}') b USING (cell_id)
        WHERE b.biome IS NOT NULL
        GROUP BY year ORDER BY year
    """).df()
    print(table.to_csv(index=False))


if __name__ == "__main__":
    main()
