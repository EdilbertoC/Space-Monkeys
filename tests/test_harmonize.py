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
from tests import test_training  # noqa: E402
from tests.test_cube import report_and_cube  # noqa: E402, F401  (fixture)

FIRE = (-15.21718, -55.5285)  # 2020-09-18, seen by S-NPP, NOAA-20 and Aqua


def test_reference_years_have_one_fold_and_later_years_the_final_model():
    last = harmonize.REFERENCE_LAST_YEAR
    for sat, folds in harmonize.FOLDS.items():
        start = int(training.SATELLITES[sat]["start"][:4])
        assert all(b <= last for _, b in folds)  # no fold uses MODIS of the drift years
        for year in range(start, last + 1):
            assert sum(a <= year <= b for a, b in folds) == 1, (sat, year)
            assert harmonize.fold_of(year, folds) != harmonize.FINAL
        assert harmonize.fold_of(last + 1, folds) == harmonize.FINAL
    with pytest.raises(ValueError):
        harmonize.fold_of(2011, harmonize.FOLDS["SNPP"])


def test_calibrators_use_only_reference_rows(tmp_path):
    def part(year, day, raw, y):
        d = tmp_path / f"year={year}"
        d.mkdir(exist_ok=True)
        pd.DataFrame(
            {
                "cell_id": np.arange(len(raw)),
                "date": pd.Timestamp(day),
                "biome": "A",
                "y": np.array(y, dtype=np.int8),
                "raw": np.array(raw, dtype=np.float32),
            }
        ).to_parquet(d / f"{day}.parquet", index=False)

    part(2021, "2021-09-01", [0.2, 0.2, 0.8, 0.8], [0, 0, 1, 1])
    part(2021, "2021-09-02", [0.2, 0.8], [1, 0])  # a day MODIS was not observing
    part(2022, "2022-09-01", [0.2, 0.2, 0.8, 0.8], [1, 1, 0, 0])  # after the reference
    gap = [pd.Timestamp("2021-09-02").date()]
    cal = harmonize.fit_biome_calibrators(tmp_path, last_year=2021, exclude_dates=gap)["A"]
    assert harmonize.baselines.apply_calibration(np.array([0.2, 0.8]), cal) == pytest.approx([0, 1])
    everything = harmonize.fit_biome_calibrators(tmp_path)["A"]
    assert harmonize.baselines.apply_calibration(np.array([0.2]), everything)[0] > 0.4


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


def test_equivalent_is_calibrated_probability_times_the_gap_of_its_month():
    raw = np.array([0.2, 0.8, 0.8], dtype=np.float32)
    biome = np.array(["A", "A", "B"])
    month = np.array([3, 9, 9])
    cal = {"A": {"x": [0.0, 1.0], "y": [0.0, 0.5]}}  # halves A; B has no calibrator
    gaps = {"A": {3: 4.0, 9: 1.25}}
    out = harmonize.equivalent(raw, biome, month, cal, gaps)
    assert out == pytest.approx([0.1 * 4.0, 0.4 * 1.25, 0.8])


class _Booster:
    """Stands in for the saved final LightGBM: one fixed raw probability per row."""

    def predict(self, x, num_threads=0):
        return np.full(len(x), 0.5)


def test_apply_after_training_writes_only_the_days_after_the_harmonized_rows(tmp_path, monkeypatch):
    test_training._cube_from(tmp_path)
    cells = [
        cube.cell_id(*p) + dr * cube.N_COLS + dc
        for p in (FIRE, test_training.NRT_A, test_training.NRT_B)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
    ]
    labels = biomes.label_cells(np.array(cells), {"Pantanal": shapely.box(-61, -19, -55, -15)})
    labels.to_parquet(tmp_path / "processed" / "test" / "cell_biome.parquet", index=False)
    training.build_training("test", tmp_path, "SNPP")
    training.build_application("test", tmp_path, "SNPP")
    monkeypatch.setattr(harmonize.baselines, "load_lightgbm", lambda _: (_Booster(), 1.0, None))

    model = {
        "period": ["2012-01-20", "2020-09-18"],
        "calibrators": {},
        "gap_factors": {"Pantanal": {"8": 2.0}},  # keys are strings after JSON
    }
    applied = harmonize.apply_after_training(tmp_path, "test", "SNPP", model)
    assert applied["period"] == ["2020-09-19", "2026-08-01"]
    assert applied["standard_until"] == "2020-09-18"
    out = pd.read_parquet(
        tmp_path / "processed" / "test" / "harmonized" / "SNPP",
        columns=["cell_id", "date", "modis", "equiv"],
    )
    assert len(out) == applied["rows_written"] == 27
    assert out["modis"].isna().all()
    aug = pd.to_datetime(out["date"]).dt.month == 8
    assert out.loc[aug, "equiv"].tolist() == pytest.approx([1.0] * 18)  # 0.5 x gap 2
    assert out.loc[~aug, "equiv"].tolist() == pytest.approx([0.5] * 9)  # September, no gap

    model["period"][1] = "2026-07-31"  # harmonized rows now reach further: no overlap
    assert harmonize.apply_after_training(tmp_path, "test", "SNPP", model)["rows_written"] == 18
    model["period"][1] = "2019-12-31"  # the table starts after the next day: stale
    with pytest.raises(ValueError):
        harmonize.apply_after_training(tmp_path, "test", "SNPP", model)


def test_gap_factor_counts_modis_fire_days_outside_the_rows(report_and_cube, tmp_path):  # noqa: F811
    cells = np.array(
        [cube.cell_id(*FIRE) + dr * cube.N_COLS + dc for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    )
    labels = biomes.label_cells(cells, {"Pantanal": shapely.box(-56, -16, -55, -15)})
    labels.to_parquet(tmp_path / "processed" / "test" / "cell_biome.parquet", index=False)
    training.build_training("test", tmp_path, "SNPP")

    gaps = harmonize.gap_factors(tmp_path, "test", "SNPP", "2012-01-20", "2020-09-18")
    assert gaps == {"Pantanal": {9: 1.0}}  # the one Aqua fire-day is inside the rows
    left_out = [pd.Timestamp("2020-09-18").date()]
    assert (
        harmonize.gap_factors(tmp_path, "test", "SNPP", "2012-01-20", "2020-09-18", left_out) == {}
    )
