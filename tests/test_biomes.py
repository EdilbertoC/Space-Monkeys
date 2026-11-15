# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Biome labelling and cross-biome sampling tests. The IBGE file is not needed:
label_cells is checked with a plain box polygon; detections are the real FIRMS
rows of the cube tests."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shapely")
pytest.importorskip("duckdb")

import shapely  # noqa: E402

from pipeline import biomes, crossbiome, cube, training  # noqa: E402
from tests.test_cube import report_and_cube  # noqa: E402, F401  (fixture)

FIRE = (-15.21718, -55.5285)  # 2020-09-18
LONE = (-15.46617, -59.75149)  # 2012-01-20


def test_ibge_codes_have_english_names():
    assert sorted(biomes.BIOMES) == [1, 2, 3, 4, 5, 6]
    assert biomes.BIOMES[6] == "Pantanal"


def test_admin1_keeps_one_country_and_merges_small_territories(tmp_path):
    shapefile = pytest.importorskip("shapefile")
    path = tmp_path / "admin1"
    with shapefile.Writer(str(path), shapeType=shapefile.POLYGON) as w:
        w.field("admin", "C")
        w.field("name", "C")
        for admin, name, (x0, y0) in [
            ("Australia", "New South Wales", (0, 0)),
            ("Australia", "Australian Capital Territory", (1, 0)),
            ("Australia", "Tasmania", (0, 5)),
            ("New Zealand", "Otago", (9, 9)),
        ]:
            w.poly([[[x0, y0], [x0, y0 + 1], [x0 + 1, y0 + 1], [x0 + 1, y0], [x0, y0]]])
            w.record(admin, name)
    regions = biomes.load_admin1(path.with_suffix(".shp"), "Australia")
    assert sorted(regions) == ["New South Wales", "Tasmania"]
    assert regions["New South Wales"].area == pytest.approx(2.0)  # the territory merged in
    with pytest.raises(ValueError):
        biomes.load_admin1(path.with_suffix(".shp"), "Chile")


def test_label_cells_uses_the_cell_centre():
    box = shapely.box(-56.0, -15.5, -55.0, -15.0)  # lon/lat
    inside, outside = cube.cell_id(*FIRE), cube.cell_id(*LONE)
    df = biomes.label_cells(np.array([inside, outside]), {"Box": box})
    assert df.loc[0, "biome"] == "Box"
    assert pd.isna(df.loc[1, "biome"])  # outside every biome: no label
    assert df.loc[0, "cell_lat"] == pytest.approx(-15.25)
    assert df.loc[0, "cell_lon"] == pytest.approx(-55.55)


def test_training_sample_keeps_every_own_detection(report_and_cube, tmp_path):  # noqa: F811
    # Every cell of the test box is "Brazil" here, so all rows have a biome.
    cells = np.array(
        [
            cube.cell_id(*p) + dr * cube.N_COLS + dc
            for p in (FIRE, LONE)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
        ]
    )
    labels = biomes.label_cells(cells, {"Pantanal": shapely.box(-61, -16, -54, -14)})
    labels.to_parquet(tmp_path / "processed" / "test" / "cell_biome.parquet", index=False)
    training.build_training("test", tmp_path, "SNPP")

    sample = crossbiome.load_training_sample(tmp_path, "test", "SNPP")
    own = sample[sample["v_pixels"] > 0]
    assert len(own) == 1  # the 2012 detection (2020 is a test year)
    assert (own["w"] == 1.0).all()
    assert (sample.loc[sample["v_pixels"] == 0, "w"] == 1 / crossbiome.SAMPLE_RATE).all()
    assert set(sample["biome"]) == {"Pantanal"}
    full = pd.read_parquet(tmp_path / "processed" / "test" / "training" / "SNPP")
    assert len(sample) <= (full["split"] == "train").sum()
