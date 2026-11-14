# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Harmonization tests. Detections are the real FIRMS rows of the cube tests; the
arithmetic checks (calibration, intervals) use plain numbers."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("shapely")

import shapely  # noqa: E402

from pipeline import biomes, cube, harmonize, training  # noqa: E402
from tests.test_cube import report_and_cube  # noqa: E402, F401  (fixture)

FIRE = (-15.21718, -55.5285)  # 2020-09-18, seen by S-NPP, NOAA-20 and Aqua


def test_every_overlap_year_has_exactly_one_fold():
    for sat, folds in harmonize.FOLDS.items():
        start = int(training.SATELLITES[sat]["start"][:4])
        for year in range(start, 2027):
            assert sum(a <= year <= b for a, b in folds) == 1, (sat, year)
    with pytest.raises(ValueError):
        harmonize.fold_of(2011, harmonize.FOLDS["SNPP"])


def test_calibration_per_biome_keeps_each_biome_total():
    oof = pd.DataFrame(
        {
            "biome": pd.Categorical(["A", "A", "A", "B", "B", "B"]),
            "raw": np.array([0.1, 0.5, 0.9, 0.1, 0.5, 0.9], dtype=np.float32),
            "y": np.array([0, 1, 1, 0, 0, 1], dtype=np.int8),
        }
    )
    p, cals = harmonize.calibrate_by_biome(oof)
    assert set(cals) == {"A", "B"}
    assert p[:3].sum() == pytest.approx(2.0)  # biome A had 2 MODIS fire-days
    assert p[3:].sum() == pytest.approx(1.0)


def test_interval_from_monthly_relative_errors():
    monthly = pd.DataFrame(
        {
            "biome": ["A"] * 20,
            "estimate": [100.0] * 20,
            "modis": [90.0 + i for i in range(20)],  # errors from -10% to +9%
        }
    )
    lo, hi = harmonize.intervals(monthly)["A"]
    assert -0.10 < lo < -0.08 and 0.07 < hi < 0.09


def test_gap_factor_counts_modis_fire_days_outside_the_rows(report_and_cube, tmp_path):  # noqa: F811
    cells = np.array(
        [cube.cell_id(*FIRE) + dr * cube.N_COLS + dc for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    )
    labels = biomes.label_cells(cells, {"Pantanal": shapely.box(-56, -16, -55, -15)})
    labels.to_parquet(tmp_path / "processed" / "test" / "cell_biome.parquet", index=False)
    training.build_training("test", tmp_path, "SNPP")

    gaps = harmonize.gap_factors(tmp_path, "test", "SNPP", "2012-01-20", "2020-09-18")
    assert gaps == {"Pantanal": {9: 1.0}}  # the one Aqua fire-day is inside the rows
