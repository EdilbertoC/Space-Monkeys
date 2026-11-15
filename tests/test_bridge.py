# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Bridge tests. Detections are the real FIRMS rows of the cube tests (one fire on
2020-09-18 seen by S-NPP, NOAA-20 and Aqua); the reference record values are plain
numbers, as in the harmonization tests."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from pipeline import bridge, cube  # noqa: E402
from tests.test_cube import report_and_cube  # noqa: E402, F401  (fixture)

FIRE = cube.cell_id(-15.21718, -55.5285)
FAR = FIRE + 20 * cube.N_COLS  # 2° north: never a NOAA-20 row
DAY = pd.Timestamp("2020-09-18")


def _setup(tmp_path):
    processed = tmp_path / "processed" / "test"
    cells = [FIRE + dr * cube.N_COLS + dc for dr in (-1, 0, 1) for dc in (-1, 0, 1)] + [FAR]
    pd.DataFrame({"cell_id": cells, "biome": "Pantanal"}).to_parquet(
        processed / "cell_biome.parquet", index=False
    )
    part = processed / "harmonized" / "SNPP" / "year=2020"
    part.mkdir(parents=True)
    pd.DataFrame(
        {
            "cell_id": [FIRE, FIRE + cube.N_COLS, FAR],
            "date": [DAY, DAY, DAY],
            "biome": "Pantanal",
            "modis": np.array([1, 0, 0], dtype=np.int8),
            "equiv": np.array([0.8, 0.3, 0.5], dtype=np.float32),
        }
    ).to_parquet(part / "harmonized.parquet", index=False)
    return processed


def test_table_takes_the_reference_record_as_target(report_and_cube, tmp_path):  # noqa: F811
    processed = _setup(tmp_path)
    first, end, left_out = bridge.overlap(tmp_path, "test", "N20", "SNPP")
    assert (first, end, left_out) == (date(2018, 4, 1), date(2020, 9, 18), [])

    table = processed / "bridge" / "t"
    span = bridge.build_table(tmp_path, "test", "N20", "SNPP", table)
    assert span == {"first": "2018-04-01", "overlap_end": "2020-09-18", "last": "2020-09-18"}
    df = pd.read_parquet(table).set_index("cell_id")
    assert len(df) == 9  # the 3x3 of the NOAA-20 fire; the day after is past its last day
    assert df.loc[FIRE, "target"] == pytest.approx(0.8)
    assert df.loc[FIRE + cube.N_COLS, "target"] == pytest.approx(0.3)
    assert df["target"].sum() == pytest.approx(1.1)  # FAR is in the record, not a row
    assert df.loc[FIRE, "v_pixels"] == 3  # NOAA-20's own detections

    gaps = bridge.gap_factors(tmp_path, "test", "SNPP", table, str(first), str(end), [])
    assert gaps == {"Pantanal": {9: round(1.6 / 1.1, 4)}}
    assert (
        bridge.gap_factors(tmp_path, "test", "SNPP", table, str(first), str(end), [DAY.date()])
        == {}
    )


def test_record_dir_prefers_the_direct_harmonization_then_a_bridge(tmp_path):
    processed = tmp_path / "processed" / "r"
    with pytest.raises(FileNotFoundError):
        bridge.record_dir(tmp_path, "r", "N21")
    (processed / "bridge" / "N21_via_N20_table").mkdir(parents=True)  # work dir, not a record
    with pytest.raises(FileNotFoundError):
        bridge.record_dir(tmp_path, "r", "N21")
    (processed / "bridge" / "N21_via_N20").mkdir()
    assert bridge.record_dir(tmp_path, "r", "N21").name == "N21_via_N20"
    (processed / "harmonized" / "N21").mkdir(parents=True)
    assert bridge.record_dir(tmp_path, "r", "N21") == processed / "harmonized" / "N21"


def test_isotonic_keeps_the_total_and_is_not_capped_at_one():
    x = np.array([0.1, 0.2, 0.5, 0.9])
    y = np.array([0.0, 0.4, 1.0, 2.6])  # record values include the reference's own gap
    cal = bridge.fit_isotonic(x, y)
    fitted = np.interp(x, cal["x"], cal["y"])
    assert fitted.sum() == pytest.approx(y.sum())
    assert fitted.max() == pytest.approx(2.6)
