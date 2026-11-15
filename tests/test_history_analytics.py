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
