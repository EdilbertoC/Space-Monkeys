# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""/history routes: the burning activity calendar. Contract: docs/api/openapi.yaml."""

from __future__ import annotations

from datetime import date
from typing import Literal

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request

from app.api.deps import parse_bbox
from app.history import analytics as an
from app.history.store import Area, AreaError, AreaTooLarge, HistoryStore

router = APIRouter(prefix="/history", tags=["history"])

Biome = Literal["Amazon", "Caatinga", "Cerrado", "Atlantic Forest", "Pampa", "Pantanal"]
Series = Literal["harmonized", "modis", "viirs", "viirs_only", "detected"]


def get_store(request: Request) -> HistoryStore:
    store = getattr(request.app.state, "history", None)
    if store is None:
        raise HTTPException(
            503, "historical data not available; build it with `python -m pipeline.serving`"
        )
    return store


def get_area(
    bbox: tuple[float, float, float, float] | None = Depends(parse_bbox),
    biome: Biome | None = Query(None, description="A Brazilian biome (IBGE 2019)."),
    area_id: str | None = Query(None, description="Id returned by POST /history/areas."),
    country: Literal["BR"] | None = Query(None, description="A whole country: BR (Brazil)."),
    store: HistoryStore = Depends(get_store),
) -> Area:
    if sum(v is not None for v in (bbox, biome, area_id, country)) != 1:
        raise HTTPException(400, "give exactly one of bbox, biome, area_id, country")
    try:
        if bbox is not None:
            return store.area_from_bbox(bbox)
        if biome is not None:
            return store.area_from_biome(biome)
        if country is not None:
            return store.area_from_country(country)
        return store.area_from_id(area_id)
    except AreaTooLarge as exc:
        raise HTTPException(413, str(exc)) from exc
    except AreaError as exc:
        raise HTTPException(422, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, f"unknown or expired area_id {area_id}") from exc


VIIRS_NOT_COMPARABLE = (
    "series=viirs cannot be compared across years: it counts fire seen by any VIIRS "
    "satellite, and NOAA-20 (2018) and NOAA-21 (2023) add detections. Use "
    "series=harmonized or series=modis; the calendar and the map still show viirs."
)


def _check_series(store: HistoryStore, series: str) -> None:
    """For the routes that compare years (climatology, anomaly, season, trend)."""
    if series == "viirs":
        raise HTTPException(422, VIIRS_NOT_COMPARABLE)
    if series == "harmonized" and store.meta["harmonization"]["status"] != "available":
        raise HTTPException(409, "harmonized series not available yet; use series=modis")
    if series in ("viirs_only", "detected") and store.coverage.layers is None:
        raise HTTPException(
            409,
            "the VIIRS-only layer is not in this serving table; rebuild it with "
            "`python -m pipeline.serving` after pipeline.harmonize",
        )


def _span(store: HistoryStore, series: str) -> tuple[date, date]:
    cov = store.coverage
    if series == "modis":
        return cov.modis
    if series == "viirs":
        return cov.viirs
    if series in ("viirs_only", "detected"):
        return cov.layers
    end = cov.harmonized_modis_end()
    if cov.estimate is not None:
        end = max(end, cov.estimate[1])
    return cov.modis[0], end


def _base(store: HistoryStore, series: str, start: int | None, end: int | None) -> range:
    first = start if start is not None else 2003
    if start is None and series in ("viirs_only", "detected"):
        s = _span(store, series)[0]  # the first complete year of the layer
        first = s.year if (s.month, s.day) == (1, 1) else s.year + 1
    last = end if end is not None else an.last_complete_year(_span(store, series)[1])
    if last < first:
        raise HTTPException(422, "base_end_year must not be before base_start_year")
    return range(first, last + 1)


def _band(store: HistoryStore, area: Area) -> tuple[float, float]:
    """Relative 90% band of the VIIRS estimate for this area (0, 0 without an estimate)."""
    return store.estimate_band(area) if store.coverage.estimate else (0.0, 0.0)


def _values(store: HistoryStore, area: Area, series: str, start: date, end: date):
    daily = store.daily(area, start, end)
    return an.series_values(daily, series, store.coverage)["value"]


def _analytics(fn, *args):
    try:
        return fn(*args)
    except an.AnalyticsError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/meta", summary="What the historical record covers")
def meta(store: HistoryStore = Depends(get_store)) -> dict:
    keys = ("unit", "grid_degrees", "time_basis", "coverage", "biomes", "harmonization")
    out = {k: store.meta[k] for k in keys}
    out["known_gaps"], out["limits"] = store.meta["known_gaps"], store.meta["limits"]
    return out


@router.post("/areas", summary="Register a drawn polygon and get an area id")
def create_area(
    body: dict = Body(..., examples=[{"geometry": {"type": "Polygon", "coordinates": []}}]),
    store: HistoryStore = Depends(get_store),
) -> dict:
    geometry = body.get("geometry") if isinstance(body, dict) else None
    if not isinstance(geometry, dict):
        raise HTTPException(422, 'body must be {"geometry": GeoJSON Polygon or MultiPolygon}')
    try:
        area, extra = store.create_area(geometry)
    except AreaTooLarge as exc:
        raise HTTPException(413, str(exc)) from exc
    except AreaError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"area_id": area.summary["area_id"], "cells": area.cells, **extra}


@router.get("/calendar", summary="Fire cell-days over time for an area")
def calendar(
    start: date,
    end: date,
    granularity: Literal["day", "week", "month"] = "week",
    area: Area = Depends(get_area),
    store: HistoryStore = Depends(get_store),
) -> dict:
    if end < start:
        raise HTTPException(422, "end must not be before start")
    if granularity == "day" and (end - start).days + 1 > store.limits["max_day_buckets"]:
        raise HTTPException(413, f"more than {store.limits['max_day_buckets']} daily buckets")
    daily = store.daily(area, start, end)
    band = _band(store, area)
    det_band = store.detected_band(area) if store.coverage.layers else (0.0, 0.0)
    return {
        "area": area.as_summary(),
        "unit": "fire_cell_days",
        "granularity": granularity,
        "buckets": an.calendar(daily, start, end, granularity, store.coverage, band, det_band),
    }


@router.get("/climatology", summary="The normal year for an area")
def climatology(
    series: Series = "harmonized",
    granularity: Literal["week", "month"] = "week",
    base_start_year: int | None = None,
    base_end_year: int | None = None,
    area: Area = Depends(get_area),
    store: HistoryStore = Depends(get_store),
) -> dict:
    _check_series(store, series)
    base = _base(store, series, base_start_year, base_end_year)
    values = _values(store, area, series, date(base.start, 1, 1), date(base.stop - 1, 12, 31))
    rows, years = _analytics(an.climatology, values, granularity, base)
    return {
        "area": area.as_summary(),
        "unit": "fire_cell_days",
        "series": series,
        "granularity": granularity,
        "base_years": [years[0], years[-1]],
        "buckets": rows,
    }


@router.get("/anomaly", summary="Is this period above normal?")
def anomaly(
    start: date,
    end: date,
    series: Series = "harmonized",
    base_start_year: int | None = None,
    base_end_year: int | None = None,
    area: Area = Depends(get_area),
    store: HistoryStore = Depends(get_store),
) -> dict:
    _check_series(store, series)
    if end < start or (end - start).days > 365:
        raise HTTPException(422, "the period must run forward and last at most one year")
    base = _base(store, series, base_start_year, base_end_year)
    first = date(min(base.start, start.year), 1, 1)
    last = max(date(base.stop - 1, 12, 31), end)
    values = _values(store, area, series, first, last)
    result = _analytics(an.anomaly, values, start, end, base)
    interval = None
    if series == "harmonized":
        interval = an.harmonized_interval(
            store.daily(area, start, end), start, end, store.coverage, _band(store, area)
        )
    return {
        "area": area.as_summary(),
        "unit": "fire_cell_days",
        "series": series,
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        **result,
        "value_interval": interval,
    }


@router.get("/anomalies", summary="Every month of a year against normal, for many areas")
def anomalies(
    year: int = Query(..., ge=2001),
    series: Series = "harmonized",
    areas: Literal["biomes", "biomes+country"] = "biomes",
    base_start_year: int | None = None,
    base_end_year: int | None = None,
    store: HistoryStore = Depends(get_store),
) -> dict:
    """One call instead of one per area and month: each month of `year` in every biome
    (and Brazil) compared with the same dates in the base years."""
    _check_series(store, series)
    base = _base(store, series, base_start_year, base_end_year)
    last_day = _span(store, series)[1]
    if date(year, 1, 1) > last_day or year < _span(store, series)[0].year:
        raise HTTPException(422, f"no {series} data in {year}")
    targets = [store.area_from_biome(b) for b in store.biome_cells]
    if areas == "biomes+country":
        targets.append(store.area_from_country("BR"))
    first = date(min(base.start, year), 1, 1)
    last = date(max(base.stop - 1, year), 12, 31)
    rows = []
    for area in targets:
        values = _values(store, area, series, first, last)
        rows.append(
            {"area": area.as_summary(), "months": an.anomaly_scan(values, year, base, last_day)}
        )
    return {
        "unit": "fire_cell_days",
        "series": series,
        "year": year,
        "base_years": [base.start, base.stop - 1],
        "areas": rows,
    }


@router.get("/season", summary="When the fire season started, peaked and ended")
def season(
    year: int = Query(..., ge=2001),
    series: Series = "harmonized",
    base_start_year: int | None = None,
    base_end_year: int | None = None,
    area: Area = Depends(get_area),
    store: HistoryStore = Depends(get_store),
) -> dict:
    _check_series(store, series)
    base = _base(store, series, base_start_year, base_end_year)
    first = date(min(base.start, year), 1, 1)
    last = date(max(base.stop - 1, year), 12, 31)
    values = _values(store, area, series, first, last)
    result = _analytics(an.season, values, year, base)
    return {"area": area.as_summary(), "series": series, **result}


@router.get("/trend", summary="Yearly totals and the long-term trend")
def trend(
    series: Series = "harmonized",
    start_year: int | None = Query(None, ge=2001),
    end_year: int | None = None,
    area: Area = Depends(get_area),
    store: HistoryStore = Depends(get_store),
) -> dict:
    _check_series(store, series)
    base = _base(store, series, start_year, end_year)
    values = _values(store, area, series, date(base.start, 1, 1), date(base.stop - 1, 12, 31))
    result = _analytics(an.trend, values, base.start, base.stop - 1)
    return {"area": area.as_summary(), "unit": "fire_cell_days", "series": series, **result}


MAP_DEGREES = (0.1, 0.25, 0.5, 1.0)


@router.get("/map", summary="Fire cell-days per cell in a period")
def fire_map(
    start: date,
    end: date,
    cell_degrees: Literal["0.1", "0.25", "0.5", "1", "auto"] = Query(
        "0.1",
        description="Grid of the answer. Coarser cells sum the 0.1° cells inside them; "
        "`auto` picks the finest grid within the cell limit.",
    ),
    area: Area = Depends(get_area),
    store: HistoryStore = Depends(get_store),
) -> dict:
    if end < start:
        raise HTTPException(422, "end must not be before start")
    limit = store.limits["max_map_cells"]
    fine = store.per_cell(area, start, end)
    options = MAP_DEGREES if cell_degrees == "auto" else (float(cell_degrees),)
    for degrees in options:
        cells = an.aggregate_cells(fine, degrees)
        if len(cells) <= limit:
            break
    else:
        hint = "" if cell_degrees == "auto" else "; try a coarser cell_degrees or auto"
        raise HTTPException(413, f"{len(cells)} cells with fire; the limit is {limit}{hint}")
    cov = store.coverage
    has_modis = start <= cov.modis[1] and end >= cov.modis[0]
    has_viirs = start <= cov.viirs[1] and end >= cov.viirs[0]
    has_layer = cov.layers is not None and start <= cov.layers[1] and end >= cov.layers[0]
    out = [
        {
            "cell_id": int(r.cell_id),
            "lat": float(r.lat),
            "lon": float(r.lon),
            "harmonized": round(float(r.harmonized), 2),
            "modis": int(r.modis) if has_modis else None,
            "viirs": int(r.viirs) if has_viirs else None,
            "viirs_only": round(float(r.viirs_only), 2) if has_layer else None,
        }
        for r in cells.itertuples(index=False)
    ]
    return {
        "area": area.as_summary(),
        "unit": "fire_cell_days",
        "grid_degrees": degrees,
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "cells": out,
    }
