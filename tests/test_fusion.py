# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
import pytest

from app.fusion.fuse import fuse
from tests.helpers import make_detections

LAT, LON = -15.0, -55.0
KM = 1 / 110.574  # one km in degrees of latitude


def test_empty_input_gives_empty_output():
    assert fuse(make_detections([])).empty


def test_same_fire_seen_by_three_sensors_becomes_one_event():
    df = make_detections(
        [
            {"sensor": "ABI", "lat": LAT + 1.5 * KM, "lon": LON, "time": "2026-08-01T15:00"},
            {"sensor": "MODIS", "lat": LAT + 0.4 * KM, "lon": LON, "time": "2026-08-01T16:10"},
            {"sensor": "VIIRS", "lat": LAT, "lon": LON, "time": "2026-08-01T17:30"},
        ]
    )
    fires = fuse(df)
    assert len(fires) == 1
    fire = fires.iloc[0]
    assert fire["sensors"] == ["ABI", "MODIS", "VIIRS"]
    assert fire["n_detections"] == 3
    # The fused point sits next to the VIIRS pixel, not the GOES one.
    assert abs(fire["lat"] - LAT) < 0.1 * KM
    # GOES saw it first.
    assert fire["first_seen"].hour == 15


def test_agreement_raises_confidence():
    single = fuse(
        make_detections(
            [
                {
                    "sensor": "VIIRS",
                    "lat": LAT,
                    "lon": LON,
                    "time": "2026-08-01T17:30",
                    "confidence": 0.6,
                }
            ]
        )
    )
    pair = fuse(
        make_detections(
            [
                {
                    "sensor": "VIIRS",
                    "lat": LAT,
                    "lon": LON,
                    "time": "2026-08-01T17:30",
                    "confidence": 0.6,
                },
                {
                    "sensor": "ABI",
                    "lat": LAT,
                    "lon": LON,
                    "time": "2026-08-01T17:20",
                    "confidence": 0.6,
                },
            ]
        )
    )
    assert pair.iloc[0]["confidence"] == pytest.approx(1 - 0.4 * 0.4)
    assert pair.iloc[0]["confidence"] > single.iloc[0]["confidence"]


def test_distant_fires_stay_separate():
    df = make_detections(
        [
            {"sensor": "VIIRS", "lat": LAT, "lon": LON, "time": "2026-08-01T17:30"},
            {"sensor": "VIIRS", "lat": LAT + 10 * KM, "lon": LON, "time": "2026-08-01T17:30"},
        ]
    )
    assert len(fuse(df)) == 2


def test_time_window_splits_events():
    df = make_detections(
        [
            {"sensor": "VIIRS", "lat": LAT, "lon": LON, "time": "2026-08-01T04:00"},
            {"sensor": "VIIRS", "lat": LAT, "lon": LON, "time": "2026-08-02T04:00"},
        ]
    )
    assert len(fuse(df, window_hours=12)) == 2
    assert len(fuse(df, window_hours=30)) == 1


def test_large_pixel_links_from_further_away():
    # 3.5 km apart: too far for two VIIRS pixels, close enough for a 3 km GOES pixel.
    base = {"lon": LON, "time": "2026-08-01T17:30"}
    viirs_pair = make_detections(
        [
            {"sensor": "VIIRS", "lat": LAT, **base},
            {"sensor": "VIIRS", "lat": LAT + 3.5 * KM, **base},
        ]
    )
    with_goes = make_detections(
        [
            {"sensor": "VIIRS", "lat": LAT, **base},
            {"sensor": "ABI", "lat": LAT + 3.5 * KM, **base},
        ]
    )
    assert len(fuse(viirs_pair)) == 2
    assert len(fuse(with_goes)) == 1


def test_fire_id_is_earliest_detection():
    df = make_detections(
        [
            {"sensor": "VIIRS", "lat": LAT, "lon": LON, "time": "2026-08-01T17:30"},
            {"sensor": "ABI", "lat": LAT, "lon": LON, "time": "2026-08-01T15:00"},
        ]
    )
    earliest_id = df.sort_values("time").iloc[0]["detection_id"]
    assert fuse(df).iloc[0]["fire_id"] == earliest_id


def test_uncertainty_is_smaller_with_viirs():
    goes_only = fuse(
        make_detections([{"sensor": "ABI", "lat": LAT, "lon": LON, "time": "2026-08-01T15:00"}])
    )
    with_viirs = fuse(
        make_detections(
            [
                {"sensor": "ABI", "lat": LAT, "lon": LON, "time": "2026-08-01T15:00"},
                {"sensor": "VIIRS", "lat": LAT, "lon": LON, "time": "2026-08-01T17:30"},
            ]
        )
    )
    assert with_viirs.iloc[0]["uncertainty_km"] < goes_only.iloc[0]["uncertainty_km"] / 4
