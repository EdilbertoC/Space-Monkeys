# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Label every 0.1° cell of a region with its Brazilian biome (IBGE 2019).

Source: IBGE, Biomas do Brasil 2019, 1:250,000 (Biomas_250mil.zip), downloaded
to data/raw/ibge/. Coordinates are SIRGAS 2000 geographic, which matches WGS84
to within about a metre, far below the 11 km cell size.

A cell takes the biome that contains its centre. Cells whose centre is outside
Brazil (sea, neighbouring countries) get no biome: this is also how the national
cube is clipped to Brazil.

Output: data/processed/<region>/cell_biome.parquet (cell_id, cell_lat, cell_lon,
biome) for every cell in the box covered by the region's cube.

Usage::

    python -m pipeline.biomes brazil
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import shapefile
import shapely

from pipeline.cube import CELLS_PER_DEGREE, N_COLS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
IBGE_ZIP = Path("raw") / "ibge" / "Biomas_250mil.zip"

# IBGE CD_Bioma -> English name (the codebase is in English).
BIOMES = {
    1: "Amazon",
    2: "Caatinga",
    3: "Cerrado",
    4: "Atlantic Forest",
    5: "Pampa",
    6: "Pantanal",
}


def load_biomes(zip_path: Path) -> dict[str, shapely.Geometry]:
    reader = shapefile.Reader(str(zip_path))
    codes = [f[0] for f in reader.fields[1:]].index("CD_Bioma")
    out = {}
    for shape_rec in reader.iterShapeRecords():
        name = BIOMES[int(shape_rec.record[codes])]
        geom = shapely.geometry.shape(shape_rec.shape.__geo_interface__)
        shapely.prepare(geom)
        out[name] = geom
    return out


def label_cells(cell_ids: np.ndarray, biomes: dict[str, shapely.Geometry]) -> pd.DataFrame:
    rows, cols = cell_ids // N_COLS, cell_ids % N_COLS
    lat = rows / CELLS_PER_DEGREE - 90 + 0.5 / CELLS_PER_DEGREE
    lon = cols / CELLS_PER_DEGREE - 180 + 0.5 / CELLS_PER_DEGREE
    biome = np.full(len(cell_ids), None, dtype=object)
    for name, geom in biomes.items():
        inside = shapely.contains_xy(geom, lon, lat) & (biome == None)  # noqa: E711
        biome[inside] = name
    return pd.DataFrame(
        {
            "cell_id": cell_ids.astype(np.int64),
            "cell_lat": np.round(lat, 2),
            "cell_lon": np.round(lon, 2),
            "biome": biome,
        }
    )


def build_cell_biome(region: str, data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    zip_path = data_dir / IBGE_ZIP
    cube_dir = data_dir / "processed" / region / "cube"
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} not found (IBGE Biomas_250mil.zip)")
    if not cube_dir.exists():
        raise FileNotFoundError(f"{cube_dir} not found; run `python -m pipeline.cube {region}`")

    # Every cell of the box the cube covers, plus a 2-cell margin for neighbours.
    r0, r1, c0, c1 = (
        duckdb.connect()
        .execute(f"""
        SELECT min(cell_id // {N_COLS}) - 2, max(cell_id // {N_COLS}) + 2,
               min(cell_id % {N_COLS}) - 2, max(cell_id % {N_COLS}) + 2
        FROM read_parquet('{cube_dir.as_posix()}/**/*.parquet', hive_partitioning = true)
    """)
        .fetchone()
    )
    rr, cc = np.meshgrid(np.arange(r0, r1 + 1), np.arange(c0, c1 + 1), indexing="ij")
    cells = (rr * N_COLS + cc).ravel()
    df = label_cells(cells, load_biomes(zip_path))
    df.to_parquet(data_dir / "processed" / region / "cell_biome.parquet", index=False)
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("region", help="folder name under data/processed/, e.g. brazil")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)
    df = build_cell_biome(args.region, args.data_dir)
    counts = df["biome"].fillna("(outside Brazil)").value_counts()
    print(f"{len(df):,} cells")
    for name, n in counts.items():
        print(f"  {name:<18}{n:>9,}")


if __name__ == "__main__":
    main()
