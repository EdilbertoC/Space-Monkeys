# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Calendar analytics on the daily fire cell-days of one area (pure pandas/numpy).

Input is a daily frame indexed by date with the area's sums: `modis` and `viirs`
(observed fire cell-days) and `est` (MODIS-equivalent expected fire cell-days from
the VIIRS model). `Coverage` says which days each series has data for: outside it a
value is missing (None), not zero.

Definitions (also in docs/api/openapi.yaml):
- weeks are days 1-7, 8-14, ... of the year; week 51 runs to 31 December;
- harmonized = MODIS observed while MODIS has data (and before the switch date),
  otherwise the VIIRS estimate;
- viirs_only = the VIIRS-only layer: expected fire cell-days the estimating VIIRS
  satellite saw that MODIS would not have (small fires), where the layer exists;
- detected = the VIIRS estimate + the VIIRS-only layer: expected cell-days with fire
  seen by MODIS or that satellite, one satellite at a time (comparable across years);
- normal = median and 10th/90th percentiles over the base years;
- season start/end = first week by which 10% / 90% of the year's fire cell-days had
  happened; peak = the week with the most;
- trend = Theil-Sen slope over complete years, with its 95% confidence interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
REF_YEAR = 2021  # non-leap year used for week labels and "normal" month-days
SERIES = ("harmonized", "modis", "viirs", "viirs_only", "detected")


class AnalyticsError(ValueError):
    """The request is valid but cannot be answered for this area or period."""


@dataclass(frozen=True)
class Coverage:
    modis: tuple[date, date]
    viirs: tuple[date, date]
    estimate: tuple[date, date] | None
    switch_date: date | None
    standard_until: date  # earliest day after which data is provisional (NRT)
    layers: tuple[date, date] | None = None  # days with the VIIRS-only layer

    def harmonized_modis_end(self) -> date:
        end = self.modis[1]
        if self.switch_date is not None:
            end = min(end, self.switch_date - timedelta(days=1))
        return end


# ---- weeks -----------------------------------------------------------------------
def week_index(d: date) -> int:
    return min(51, (d.timetuple().tm_yday - 1) // 7)


def week_start(year: int, w: int) -> date:
    return date(year, 1, 1) + timedelta(days=7 * w)


def week_end(year: int, w: int) -> date:
    return date(year, 12, 31) if w == 51 else week_start(year, w) + timedelta(days=6)


def week_label(w: int) -> str:
    s, e = week_start(REF_YEAR, w), week_end(REF_YEAR, w)
    if s.month == e.month:
        return f"{MONTHS[s.month - 1]} {s.day}–{e.day}"
    return f"{MONTHS[s.month - 1]} {s.day}–{MONTHS[e.month - 1]} {e.day}"


# ---- series ----------------------------------------------------------------------
def _within(index: pd.DatetimeIndex, span: tuple[date, date] | None) -> np.ndarray:
    if span is None:
        return np.zeros(len(index), dtype=bool)
    return (index >= pd.Timestamp(span[0])) & (index <= pd.Timestamp(span[1]))


def series_values(daily: pd.DataFrame, series: str, cov: Coverage) -> pd.DataFrame:
    """Daily `value` (NaN where the series has no data), `source` and, for the
    harmonized series, the part of the value that is an estimate (`est_part`)."""
    if series not in SERIES:
        raise ValueError(f"unknown series {series}")
    idx = daily.index
    out = pd.DataFrame(index=idx)
    out["est_part"] = 0.0
    if series in ("modis", "viirs"):
        span = cov.modis if series == "modis" else cov.viirs
        out["value"] = daily[series].where(_within(idx, span)).astype(float)
        out["source"] = np.where(_within(idx, span), series, None)
        return out
    if series in ("viirs_only", "detected"):
        inside = _within(idx, cov.layers)
        fine = daily["fine"] if "fine" in daily else pd.Series(0.0, index=idx)
        value = fine if series == "viirs_only" else daily["est"] + fine
        out["value"] = value.where(inside).astype(float)
        out["est_part"] = np.where(inside, daily["est"], 0.0) if series == "detected" else 0.0
        out["source"] = np.where(inside, "viirs_estimate", None)
        return out
    use_modis = _within(idx, (cov.modis[0], cov.harmonized_modis_end()))
    use_est = ~use_modis & _within(idx, cov.estimate)
    out["value"] = np.where(use_modis, daily["modis"], np.where(use_est, daily["est"], np.nan))
    out["est_part"] = np.where(use_est, daily["est"], 0.0)
    out["source"] = np.where(use_modis, "modis", np.where(use_est, "viirs_estimate", None))
    return out


def _sum(values: pd.Series) -> float | None:
    return None if values.isna().all() else float(values.sum(min_count=1))


# ---- calendar --------------------------------------------------------------------
def buckets(start: date, end: date, granularity: str) -> list[tuple[date, date]]:
    out, d = [], start
    while d <= end:
        if granularity == "day":
            b_end = d
        elif granularity == "week":
            b_end = week_end(d.year, week_index(d))
        elif granularity == "month":
            nxt = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
            b_end = nxt - timedelta(days=1)
        else:
            raise ValueError(f"unknown granularity {granularity}")
        b_end = min(b_end, end)
        out.append((d, b_end))
        d = b_end + timedelta(days=1)
    return out


def calendar(
    daily: pd.DataFrame,
    start: date,
    end: date,
    granularity: str,
    cov: Coverage,
    band: tuple[float, float],
) -> list[dict]:
    harm = series_values(daily, "harmonized", cov)
    modis = series_values(daily, "modis", cov)["value"]
    viirs = series_values(daily, "viirs", cov)["value"]
    only = series_values(daily, "viirs_only", cov)["value"]
    detected = series_values(daily, "detected", cov)["value"]
    out = []
    for b_start, b_end in buckets(start, end, granularity):
        sl = slice(pd.Timestamp(b_start), pd.Timestamp(b_end))
        h = harm.loc[sl]
        value = _sum(h["value"])
        harmonized, source = None, None
        if value is not None:
            est = float(h["est_part"].sum())
            exact = value - est
            harmonized = {
                "value": round(value, 2),
                "low": round(exact + est * (1 + band[0]), 2),
                "high": round(exact + est * (1 + band[1]), 2),
            }
            sources = set(h["source"].dropna())
            source = sources.pop() if len(sources) == 1 else "mixed"
        m, v = _sum(modis.loc[sl]), _sum(viirs.loc[sl])
        f, det = _sum(only.loc[sl]), _sum(detected.loc[sl])
        out.append(
            {
                "start": b_start.isoformat(),
                "end": b_end.isoformat(),
                "harmonized": harmonized,
                "harmonized_source": source,
                "modis": None if m is None else int(m),
                "viirs": None if v is None else int(v),
                "viirs_only": None if f is None else round(f, 2),
                "detected": None if det is None else round(det, 2),
                "provisional": b_end > cov.standard_until,
            }
        )
    return out


# ---- climatology -----------------------------------------------------------------
def _complete_years(values: pd.Series, years: range) -> list[int]:
    """Years whose every day has data."""
    out = []
    for y in years:
        sl = values.loc[pd.Timestamp(date(y, 1, 1)) : pd.Timestamp(date(y, 12, 31))]
        if len(sl) >= 365 and not sl.isna().any():
            out.append(y)
    return out


def period_totals(values: pd.Series, granularity: str) -> pd.DataFrame:
    """Totals per (year, week 0-51) or (year, month 1-12)."""
    idx = values.index
    key = (
        np.array([week_index(d) for d in idx.date])
        if granularity == "week"
        else idx.month.to_numpy()
    )
    frame = pd.DataFrame({"year": idx.year, "key": key, "v": values.to_numpy()})
    return frame.groupby(["year", "key"])["v"].sum().unstack("key")


def climatology(values: pd.Series, granularity: str, base: range) -> tuple[list[dict], list[int]]:
    years = _complete_years(values, base)
    if len(years) < 3:
        raise AnalyticsError("fewer than 3 complete base years with data for this series")
    table = period_totals(values, granularity).loc[years]
    rows = []
    for key in table.columns:
        xs = table[key].to_numpy(dtype=float)
        rows.append(
            {
                "index": int(key),
                "label": week_label(int(key)) if granularity == "week" else MONTHS[int(key) - 1],
                "median": round(float(np.percentile(xs, 50)), 1),
                "p10": round(float(np.percentile(xs, 10)), 1),
                "p90": round(float(np.percentile(xs, 90)), 1),
                "mean": round(float(xs.mean()), 1),
            }
        )
    return rows, years


# ---- anomaly ---------------------------------------------------------------------
def _shift(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # 29 February
        return d.replace(year=d.year + years, day=28)


def anomaly(values: pd.Series, start: date, end: date, base: range) -> dict:
    period = values.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    if period.isna().any() or period.empty:
        raise AnalyticsError("the period is outside the data of this series")
    value = float(period.sum())
    excluded = sorted({start.year, end.year})
    totals, used = [], []
    for y in base:
        if y in excluded:
            continue
        s, e = _shift(start, y - start.year), _shift(end, y - start.year)
        sl = values.loc[pd.Timestamp(s) : pd.Timestamp(e)]
        if sl.empty or sl.isna().any() or len(sl) < (e - s).days + 1:
            continue
        totals.append(float(sl.sum()))
        used.append(y)
    if len(totals) < 3:
        raise AnalyticsError("fewer than 3 base years cover this calendar period")
    xs = np.array(totals)
    p10, med, p90 = (float(np.percentile(xs, q)) for q in (10, 50, 90))
    if value > xs.max():
        cls = "record"
    elif value > p90:
        cls = "above"
    elif value < p10:
        cls = "below"
    else:
        cls = "normal"
    return {
        "value": round(value, 2),
        "normal": {"median": round(med, 1), "p10": round(p10, 1), "p90": round(p90, 1)},
        "percentile": round(100.0 * float((xs < value).mean()), 1),
        "ratio_to_median": round(value / med, 2) if med > 0 else None,
        "class": cls,
        "base_years": [min(used), max(used)],
        "excluded_years": excluded,
    }


# ---- season ----------------------------------------------------------------------
def season_weeks(weekly: np.ndarray) -> tuple[int, int, int] | None:
    total = weekly.sum()
    if total <= 0:
        return None
    share = np.cumsum(weekly) / total
    return int(np.argmax(share >= 0.1)), int(np.argmax(weekly)), int(np.argmax(share >= 0.9))


def season(values: pd.Series, year: int, base: range) -> dict:
    sl = values.loc[pd.Timestamp(date(year, 1, 1)) : pd.Timestamp(date(year, 12, 31))]
    if sl.empty or sl.isna().all():
        raise AnalyticsError("no data for this series in that year")
    weekly = period_totals(sl.fillna(0), "week")
    weeks = season_weeks(weekly.loc[year].reindex(range(52), fill_value=0).to_numpy(float))
    if weeks is None:
        raise AnalyticsError("no fire cell-days in that year for this area")
    years = [y for y in _complete_years(values, base) if y != year]
    table = period_totals(values, "week")
    normal = [
        w
        for y in years
        if (w := season_weeks(table.loc[y].reindex(range(52), fill_value=0).to_numpy(float)))
    ]
    if len(normal) < 3:
        raise AnalyticsError("fewer than 3 base years with fire for the normal season")
    med = [int(np.median([n[i] for n in normal])) for i in range(3)]
    md = [week_start(REF_YEAR, w).strftime("%m-%d") for w in med]
    last_day = sl.dropna().index.max().date()
    return {
        "year": year,
        "total": round(float(sl.sum()), 2),
        "start_week": week_start(year, weeks[0]).isoformat(),
        "peak_week": week_start(year, weeks[1]).isoformat(),
        "end_week": week_start(year, weeks[2]).isoformat(),
        "normal": {"start": md[0], "peak": md[1], "end": md[2]},
        "base_years": [min(years), max(years)],
        "excluded_years": [year],
        "complete_year": last_day >= date(year, 12, 31),
    }


# ---- trend -----------------------------------------------------------------------
def trend(values: pd.Series, start_year: int, end_year: int) -> dict:
    complete = set(_complete_years(values, range(start_year, end_year + 1)))
    annual = []
    for y in range(start_year, end_year + 1):
        sl = values.loc[pd.Timestamp(date(y, 1, 1)) : pd.Timestamp(date(y, 12, 31))]
        total = _sum(sl)
        if total is not None:
            annual.append({"year": y, "value": round(total, 2), "complete": y in complete})
    pts = [(a["year"], a["value"]) for a in annual if a["complete"]]
    if len(pts) < 3:
        raise AnalyticsError("fewer than 3 complete years for a trend")
    res = theilslopes([v for _, v in pts], [y for y, _ in pts])
    lo, hi = float(res.low_slope), float(res.high_slope)
    return {
        "annual": annual,
        "slope_per_year": round(float(res.slope), 1),
        "ci95": [round(lo, 1), round(hi, 1)],
        "significant": not (lo <= 0 <= hi),
        "method": "theil-sen",
    }


def last_complete_year(span_end: date) -> int:
    return span_end.year if (span_end.month, span_end.day) == (12, 31) else span_end.year - 1
