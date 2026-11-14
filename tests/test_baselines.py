# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Baseline tests. Model checks run on the training tables built from the real
FIRMS rows of the cube tests (fire on 2020-09-18 near -15.2, -55.5)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("duckdb")

from pipeline import baselines, training  # noqa: E402
from tests.test_cube import report_and_cube  # noqa: E402, F401  (fixture)


def _table(tmp_path, satellite):
    training.build_training("test", tmp_path, satellite)
    return baselines.load_table("test", tmp_path, satellite)


def test_unweight_undoes_the_class_weight():
    assert baselines.unweight(np.array([0.5]), 3.0)[0] == pytest.approx(0.25)
    assert baselines.unweight(np.array([0.2]), 1.0)[0] == pytest.approx(0.2)
    assert baselines.positive_weight(np.array([0, 0, 0, 1])) == 3.0


def test_wape():
    assert baselines.wape(pd.Series([2.0, 0.0]), pd.Series([1.0, 1.0])) == pytest.approx(1.0)
    assert baselines.wape(pd.Series([1.0]), pd.Series([0.0])) is None


def test_ratio_by_month_and_own_detection(report_and_cube, tmp_path):  # noqa: F811
    df = _table(tmp_path, "SNPP")
    model = baselines.fit_ratio(df)
    table = model["by_month_and_own"]
    assert table["9|1"] == 1.0  # the fire cell: S-NPP saw it, so did Aqua
    assert table["9|0"] == 0.0  # its 8 neighbours
    assert table["1|0"] == 0.0  # neighbours and next day of the 2012 detection


def test_naive_totals_on_the_test_year(report_and_cube, tmp_path):  # noqa: F811
    df = _table(tmp_path, "SNPP")
    test = df[df["split"] == "test"]
    m = baselines.evaluate(test, baselines.naive_predict(test))
    assert m["totals"]["modis_fire_days"] == 1
    assert m["totals"]["bias"] == 0.0
    assert m["totals"]["monthly_wape"] == 0.0
    assert m["by_year"] == [{"year": 2020, "modis": 1, "predicted": 1.0}]


def test_lightgbm_fits_with_the_class_weight(report_and_cube, tmp_path):  # noqa: F811
    # NOAA-20 table: 9 rows in 2020, one with MODIS fire. Too few rows to grow trees,
    # so this is a smoke test; the calibration itself is checked on the real test
    # years (total bias in report.json).
    df = _table(tmp_path, "N20")
    booster, info = baselines.fit_lightgbm(df, valid_year=2022)
    assert info["positive_weight"] == pytest.approx(8.0)
    raw = booster.predict(df[training.FEATURES])
    p = baselines.lightgbm_predict(booster, df, info["positive_weight"])
    assert np.allclose(p, baselines.unweight(raw, 8.0))
    assert ((p > 0) & (p < 1)).all()


def test_run_needs_modis_fire_in_training(report_and_cube, tmp_path):  # noqa: F811
    _table(tmp_path, "SNPP")  # the 2012 training rows have no MODIS fire
    with pytest.raises(ValueError):
        baselines.run("test", tmp_path, "SNPP")
