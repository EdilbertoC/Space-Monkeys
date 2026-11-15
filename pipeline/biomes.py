# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Label every 0.1° cell with its calibration region: IBGE biome, or a state abroad.

In Brazil the regions are the biomes (IBGE 2019).

Source: IBGE, Biomas do Brasil 2019, 1:250,000 (Biomas_250mil.zip), downloaded
to data/raw/ibge/. Coordinates are SIRGAS 2000 geographic, which matches WGS84
to within about a metre, far below the 11 km cell size.

A cell takes the biome that contains its centre. Cells whose centre is outside
Brazil (sea, neighbouring countries) get no biome: this is also how the national
cube is clipped to Brazil.

Outside Brazil (`--country`), the calibration regions are the country's states or
provinces from Natural Earth admin-1 at 1:50m (public domain), downloaded to
data/raw/naturalearth/. They stand in for biomes: the column keeps its name so the
rest of the pipeline is unchanged.

Output: data/processed/<region>/cell_biome.parquet (cell_id, cell_lat, cell_lon,
biome) for every cell in the box covered by the region's cube.

Usage::

    python -m pipeline.biomes brazil
    python -m pipeline.biomes australia --country Australia
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
NATURAL_EARTH_ZIP = Path("raw") / "naturalearth" / "ne_50m_admin_1_states_provinces.zip"
# Admin-1 units merged into the state around them (too few fire cells on their own).
MERGE = {
    "Australian Capital Territory": "New South Wales",
    "Jervis Bay Territory": "New South Wales",
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


def load_admin1(zip_path: Path, country: str) -> dict[str, shapely.Geometry]:
    """States or provinces of one country from Natural Earth admin-1 (1:50m), used
    as calibration regions outside Brazil. Units too small to calibrate on their
    own are merged into the state around them (MERGE)."""
    reader = shapefile.Reader(str(zip_path))
    fields = [f[0] for f in reader.fields[1:]]
    admin, name_ix = fields.index("admin"), fields.index("name")
    parts: dict[str, list] = {}
    for shape_rec in reader.iterShapeRecords():
        if shape_rec.record[admin] != country:
            continue
        name = MERGE.get(shape_rec.record[name_ix], shape_rec.record[name_ix])
        parts.setdefault(name, []).append(shapely.geometry.shape(shape_rec.shape.__geo_interface__))
    if not parts:
        raise ValueError(f"no admin-1 units for {country} in {zip_path}")
    out = {}
    for name, geoms in parts.items():
        geom = shapely.union_all(geoms)
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


def build_cell_biome(
    region: str, data_dir: Path = DEFAULT_DATA_DIR, country: str | None = None
) -> pd.DataFrame:
    """Brazil (default): IBGE biomes. Another country: its Natural Earth states."""
    zip_path = data_dir / (NATURAL_EARTH_ZIP if country else IBGE_ZIP)
    cube_dir = data_dir / "processed" / region / "cube"
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} not found")
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
    regions = load_admin1(zip_path, country) if country else load_biomes(zip_path)
    df = label_cells(cells, regions)
    df.to_parquet(data_dir / "processed" / region / "cell_biome.parquet", index=False)
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("region", help="folder name under data/processed/, e.g. brazil")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--country",
        help="label with this country's Natural Earth states instead of IBGE biomes, "
        "e.g. Australia",
    )
    args = parser.parse_args(argv)
    df = build_cell_biome(args.region, args.data_dir, args.country)
    counts = df["biome"].fillna(f"(outside {args.country or 'Brazil'})").value_counts()
    print(f"{len(df):,} cells")
    for name, n in counts.items():
        print(f"  {name:<18}{n:>9,}")


if __name__ == "__main__":
    main()
