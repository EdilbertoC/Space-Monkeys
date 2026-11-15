# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Analytics rules on plain numbers (arithmetic checks, not fire data)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.history import analytics as an

COV = an.Coverage(
    modis=(date(2000, 11, 1), date(2026, 12, 31)),
    viirs=(date(2012, 1, 20), date(2026, 12, 31)),
    estimate=(date(2012, 1, 20), date(2026, 12, 31)),
    switch_date=None,
    standard_until=date(2026, 8, 31),
)


def daily(start, end, modis=0.0, viirs=0.0, est=0.0):
    idx = pd.date_range(start, end, freq="D")
    return pd.DataFrame({"modis": modis, "viirs": viirs, "est": est}, index=idx)


def test_weeks():
    assert an.week_index(date(2021, 1, 7)) == 0
    assert an.week_index(date(2021, 1, 8)) == 1
    assert an.week_index(date(2020, 12, 31)) == 51  # leap year, day 366
    assert an.week_label(0) == "Jan 1–7"
    assert an.week_label(51) == "Dec 24–31"


def test_month_buckets_clip_to_the_period():
    assert an.buckets(date(2020, 1, 15), date(2020, 3, 3), "month") == [
        (date(2020, 1, 15), date(2020, 1, 31)),
        (date(2020, 2, 1), date(2020, 2, 29)),
        (date(2020, 3, 1), date(2020, 3, 3)),
    ]


def test_harmonized_switches_to_the_estimate():
    cov = an.Coverage(**{**COV.__dict__, "switch_date": date(2025, 1, 2)})
    d = daily("2025-01-01", "2025-01-03", modis=1.0, est=0.5)
    (b,) = an.calendar(d, date(2025, 1, 1), date(2025, 1, 3), "week", cov, (-0.2, 0.4))
    assert b["harmonized_source"] == "mixed"
    assert b["harmonized"]["value"] == pytest.approx(1.0 + 0.5 + 0.5)
    assert b["harmonized"]["low"] == pytest.approx(1.0 + 1.0 * 0.8)  # band on the estimate only
    assert b["harmonized"]["high"] == pytest.approx(1.0 + 1.0 * 1.4)


def test_anomaly_classes_and_percentile():
    years = range(2003, 2013)
    idx = pd.date_range("2003-01-01", "2012-12-31", freq="D")
    v = pd.Series(0.0, index=idx)
    for i, y in enumerate(years):  # base years: 1..10 fire cell-days on 5 September
        v[pd.Timestamp(date(y, 9, 5))] = i + 1
    v[pd.Timestamp(date(2012, 9, 5))] = 20  # the year asked about
    r = an.anomaly(v, date(2012, 9, 5), date(2012, 9, 5), years)
    assert r["class"] == "record"
    assert r["percentile"] == 100.0
    assert r["excluded_years"] == [2012]
    assert r["normal"]["median"] == 5.0


def test_season_weeks():
    weekly = np.zeros(52)
    weekly[[10, 30, 31, 40]] = [10, 40, 40, 10]
    # 10% is reached in week 10; 90% (10 + 40 + 40 of 100) in week 31; peak week 30.
    assert an.season_weeks(weekly) == (10, 30, 31)
    assert an.season_weeks(np.zeros(52)) is None


def test_trend_detects_a_steady_rise():
    idx = pd.date_range("2003-01-01", "2012-12-31", freq="D")
    v = pd.Series(0.0, index=idx)
    for i, y in enumerate(range(2003, 2013)):
        v[pd.Timestamp(date(y, 6, 1))] = 100 + 10 * i
    r = an.trend(v, 2003, 2012)
    assert r["slope_per_year"] == 10.0
    assert r["significant"] is True


def test_last_complete_year():
    assert an.last_complete_year(date(2026, 11, 14)) == 2025
    assert an.last_complete_year(date(2025, 12, 31)) == 2025


def test_harmonized_interval_is_exact_on_modis_days_and_banded_on_the_estimate():
    d = daily("2026-12-01", "2026-12-31", modis=2.0, est=4.0)
    cov = an.Coverage(
        modis=(date(2000, 11, 1), date(2026, 12, 15)),
        viirs=(date(2012, 1, 20), date(2026, 12, 31)),
        estimate=(date(2012, 1, 20), date(2026, 12, 31)),
        switch_date=None,
        standard_until=date(2026, 8, 31),
    )
    exact = an.harmonized_interval(d, date(2026, 12, 1), date(2026, 12, 10), cov, (-0.2, 0.3))
    assert exact == {"value": 20.0, "low": 20.0, "high": 20.0}
    mixed = an.harmonized_interval(d, date(2026, 12, 14), date(2026, 12, 17), cov, (-0.2, 0.3))
    # 2 MODIS days (2 each) + 2 estimate days (4 each): 4 exact + 8 estimated.
    assert mixed == {"value": 12.0, "low": 4 + 8 * 0.8, "high": 4 + 8 * 1.3}


def test_anomaly_scan_flags_the_month_in_progress():
    idx = pd.date_range("2003-01-01", "2012-06-10", freq="D")
    v = pd.Series(1.0, index=idx)
    months = an.anomaly_scan(v, 2012, range(2003, 2012), date(2012, 6, 10))
    assert [m["start"] for m in months][-1] == "2012-06-01"
    assert months[-1]["partial"] and months[-1]["end"] == "2012-06-10"
    assert not months[0]["partial"]
    assert all(m["class"] == "normal" for m in months)
    assert "excluded_years" not in months[0]


def test_anomaly_scan_gives_a_reason_instead_of_a_class():
    v = pd.Series(1.0, index=pd.date_range("2011-01-01", "2012-12-31", freq="D"))
    months = an.anomaly_scan(v, 2012, range(2011, 2012), date(2012, 12, 31))
    assert len(months) == 12
    assert months[0]["class"] is None and "base years" in months[0]["reason"]


def test_aggregate_cells_sums_into_the_coarse_cell_of_each_centre():
    def cid(lat, lon):  # 0.1° cell holding (lat, lon)
        return int((lat + 90) * 10) * 3600 + int((lon + 180) * 10)

    cells = pd.DataFrame(
        {
            "cell_id": [cid(-17.55, -57.05), cid(-17.65, -57.15), cid(-15.05, -55.05)],
            "modis": [2, 3, 1],
            "viirs": [4, 5, 1],
            "harmonized": [2.0, 3.0, 1.0],
        }
    )
    same = an.aggregate_cells(cells, 0.1)
    assert list(same["lat"]) == [-17.55, -17.65, -15.05]
    coarse = an.aggregate_cells(cells, 1.0).sort_values("lat").reset_index(drop=True)
    assert len(coarse) == 2  # the two Pantanal cells share the 1° cell (-18..-17, -58..-57)
    assert coarse.loc[0, ["modis", "viirs", "harmonized"]].tolist() == [5, 9, 5.0]
    assert (coarse.loc[0, "lat"], coarse.loc[0, "lon"]) == (-17.5, -57.5)
    assert coarse["cell_id"].tolist() == [72 * 360 + 122, 74 * 360 + 124]


def test_layers_add_up_and_carry_their_band():
    cov = an.Coverage(**{**COV.__dict__, "layers": (date(2012, 1, 20), date(2026, 12, 31))})
    d = daily("2025-01-01", "2025-01-03", modis=1.0, viirs=1.0, est=0.5)
    d["fine"] = 0.25
    (b,) = an.calendar(d, date(2025, 1, 1), date(2025, 1, 3), "month", cov, (0, 0), (-0.1, 0.2))
    assert b["viirs_only"] == pytest.approx(0.75)
    assert b["detected"]["value"] == pytest.approx(3 * (0.5 + 0.25))
    assert b["detected"]["low"] == pytest.approx(2.25 * 0.9, abs=0.01)
    assert b["detected"]["high"] == pytest.approx(2.25 * 1.2, abs=0.01)
    early = daily("2011-12-01", "2011-12-03", modis=1.0)
    early["fine"] = 0.0
    (b,) = an.calendar(early, date(2011, 12, 1), date(2011, 12, 3), "month", cov, (0, 0))
    assert b["viirs_only"] is None and b["detected"] is None  # before the layer: missing
