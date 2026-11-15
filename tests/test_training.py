# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Training table tests, on the real FIRMS rows used by the cube tests: one fire on
2020-09-18 seen by S-NPP (4 pixels kept), NOAA-20 (3) and Aqua (2), and one S-NPP
detection on 2012-01-20."""

from datetime import date

import pandas as pd
import pytest

pytest.importorskip("duckdb")

from pipeline import cube, ingest, training  # noqa: E402
from tests.firms_samples import ARCHIVE_HEADER, NRT_HEADER, csv_text, write_zip  # noqa: E402
from tests.test_cube import AQUA, N20, SNPP, report_and_cube  # noqa: E402, F401  (fixture)
from tests.test_ingest import SNPP_NRT  # noqa: E402


def _build(tmp_path, satellite):
    report = training.build_training("test", tmp_path, satellite)
    df = pd.read_parquet(tmp_path / "processed" / "test" / "training" / satellite)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return report, df.set_index(["cell_id", "date"])


def _cell(lat, lon, drow=0, dcol=0):
    return cube.cell_id(lat, lon) + drow * cube.N_COLS + dcol


FIRE = (-15.21718, -55.5285)  # 2020-09-18
LONE = (-15.46617, -59.75149)  # 2012-01-20


def test_features_never_use_modis():
    for name in training.FEATURES:
        assert not any(s in name for s in ("modis", "terra", "aqua"))
    assert training.TARGET not in training.FEATURES


def test_snpp_candidates_are_its_own_neighbourhood(report_and_cube, tmp_path):  # noqa: F811
    report, df = _build(tmp_path, "SNPP")
    # 2012: 3x3 cells on the day and the day after. 2020: 3x3 on the day only, because
    # the day after is past the last date with standard processing.
    assert len(df) == 9 * 2 + 9
    assert report["splits"]["train"]["rows"] == 18
    assert report["splits"]["test"]["rows"] == 9
    assert report["modis_coverage_by_year"] == [
        {"year": 2020, "modis_fire_days": 1, "in_candidates": 1, "coverage": 1.0}
    ]


def test_snpp_own_cell_neighbour_and_lag(report_and_cube, tmp_path):  # noqa: F811
    _, df = _build(tmp_path, "SNPP")
    own = df.loc[(_cell(*FIRE), date(2020, 9, 18))]
    assert (own["modis"], own["aqua"], own["terra"]) == (1, True, False)
    assert (own["v_pixels"], own["v_overpasses"], own["nb3_pixels"]) == (4, 2, 0)
    assert own["v_frp_sum"] == pytest.approx(3.1 + 15.5 + 4.37 + 26.41)  # S-NPP only
    assert own["split"] == "test"

    nb = df.loc[(_cell(*FIRE, drow=1), date(2020, 9, 18))]
    assert (nb["v_pixels"], nb["nb3_pixels"], nb["nb3_cells"], nb["modis"]) == (0, 4, 1, 0)

    lag = df.loc[(_cell(*LONE), date(2012, 1, 21))]
    assert (lag["v_pixels"], lag["lag1_pixels"], lag["split"]) == (0, 1, "train")


def test_n20_table_uses_only_noaa20(report_and_cube, tmp_path):  # noqa: F811
    report, df = _build(tmp_path, "N20")
    assert report["period"][0] == "2018-04-01"
    assert set(report["splits"]) == {"train"}  # 2020 is inside N20's training years
    assert len(df) == 9
    own = df.loc[(_cell(*FIRE), date(2020, 9, 18))]
    assert (own["v_pixels"], own["v_overpasses"], own["modis"]) == (3, 2, 1)


def test_unknown_satellite_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        training.build_training("test", tmp_path, "Terra")
    with pytest.raises(ValueError):
        training.build_application("test", tmp_path, "Terra")


# ---- application table ---------------------------------------------------------------
NRT_A = (-17.40159, -59.57053)  # 2026-08-01, S-NPP NRT
NRT_B = (-18.03111, -59.62153)


def _cube_from(tmp_path, with_modis=True, with_nrt=True):
    """Cube from the real rows of the cube tests, plus two real S-NPP NRT detections
    of 2026-08-01 (from the ingest tests)."""
    raw = tmp_path / "raw" / "firms" / "test"
    raw.mkdir(parents=True)
    snpp = {"fire_archive_SV-C2_1.csv": csv_text(ARCHIVE_HEADER, SNPP)}
    if with_nrt:
        snpp["fire_nrt_SV-C2_1.csv"] = csv_text(NRT_HEADER, SNPP_NRT)
    write_zip(raw / "DL_FIRE_SV-C2_1.zip", snpp)
    write_zip(
        raw / "DL_FIRE_J1V-C2_1.zip", {"fire_archive_J1V-C2_1.csv": csv_text(ARCHIVE_HEADER, N20)}
    )
    if with_modis:
        write_zip(
            raw / "DL_FIRE_M-C61_1.zip",
            {"fire_archive_M-C61_1.csv": csv_text(ARCHIVE_HEADER, AQUA)},
        )
    ingest.ingest("test", tmp_path)
    cube.build_cube("test", tmp_path)


def _read_app(tmp_path, satellite="SNPP"):
    df = pd.read_parquet(tmp_path / "processed" / "test" / "application" / satellite)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.set_index(["cell_id", "date"])


def test_application_continues_after_the_training_table_with_nrt(tmp_path):
    _cube_from(tmp_path)
    training.build_training("test", tmp_path, "SNPP")  # ends 2020-09-18 (last MODIS day)
    report = training.build_application("test", tmp_path, "SNPP")
    assert report["period"] == ["2020-09-19", "2026-08-01"]
    assert report["standard_until"] == "2020-09-18"
    df = _read_app(tmp_path)
    # The day after the 2020 fire (3x3, lag), and the 3x3 of each NRT detection.
    assert report["rows"] == len(df) == 9 + 9 + 9
    assert not any(c in df.columns for c in ("modis", "terra", "aqua", "split"))
    assert set(training.FEATURES) <= set(df.columns)
    after = df.loc[(_cell(*FIRE), date(2020, 9, 19))]
    assert (after["v_pixels"], after["lag1_pixels"]) == (0, 4)  # sees the day before
    own = df.loc[(_cell(*NRT_A), date(2026, 8, 1))]
    assert (own["v_pixels"], own["v_night"]) == (1, 1)


def test_application_rows_match_the_training_table(tmp_path):
    _cube_from(tmp_path, with_nrt=False)
    _, train = _build(tmp_path, "SNPP")
    training.build_application("test", tmp_path, "SNPP", start="2012-01-20", end="2020-09-18")
    app = _read_app(tmp_path)
    assert sorted(app.index) == sorted(train.index)
    pd.testing.assert_frame_equal(
        app.loc[train.index, training.FEATURES], train[training.FEATURES], check_dtype=False
    )


def test_application_needs_no_modis(tmp_path):
    _cube_from(tmp_path, with_modis=False)
    with pytest.raises(ValueError):
        training.build_training("test", tmp_path, "SNPP")
    report = training.build_application("test", tmp_path, "SNPP")
    assert report["period"] == ["2012-01-20", "2026-08-01"]
    assert report["rows"] == len(_read_app(tmp_path)) == 9 * 2 + 9 * 2 + 9 * 2
