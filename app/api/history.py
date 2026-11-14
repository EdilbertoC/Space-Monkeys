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
Series = Literal["harmonized", "modis", "viirs"]


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
    store: HistoryStore = Depends(get_store),
) -> Area:
    if sum(v is not None for v in (bbox, biome, area_id)) != 1:
        raise HTTPException(400, "give exactly one of bbox, biome, area_id")
    try:
        if bbox is not None:
            return store.area_from_bbox(bbox)
        if biome is not None:
            return store.area_from_biome(biome)
        return store.area_from_id(area_id)
    except AreaTooLarge as exc:
        raise HTTPException(413, str(exc)) from exc
    except AreaError as exc:
        raise HTTPException(422, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, f"unknown or expired area_id {area_id}") from exc


def _check_series(store: HistoryStore, series: str) -> None:
    if series == "harmonized" and store.meta["harmonization"]["status"] != "available":
        raise HTTPException(409, "harmonized series not available yet; use series=modis")


def _span(store: HistoryStore, series: str) -> tuple[date, date]:
    cov = store.coverage
    if series == "modis":
        return cov.modis
    if series == "viirs":
        return cov.viirs
    end = cov.harmonized_modis_end()
    if cov.estimate is not None:
        end = max(end, cov.estimate[1])
    return cov.modis[0], end


def _base(store: HistoryStore, series: str, start: int | None, end: int | None) -> range:
    first = start if start is not None else (2013 if series == "viirs" else 2003)
    last = end if end is not None else an.last_complete_year(_span(store, series)[1])
    if last < first:
        raise HTTPException(422, "base_end_year must not be before base_start_year")
    return range(first, last + 1)


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
    band = store.estimate_band(area) if store.coverage.estimate else (0.0, 0.0)
    return {
        "area": area.as_summary(),
        "unit": "fire_cell_days",
        "granularity": granularity,
        "buckets": an.calendar(daily, start, end, granularity, store.coverage, band),
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
    return {
        "area": area.as_summary(),
        "unit": "fire_cell_days",
        "series": series,
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        **result,
        "value_interval": None,
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


@router.get("/map", summary="Fire cell-days per cell in a period")
def fire_map(
    start: date,
    end: date,
    area: Area = Depends(get_area),
    store: HistoryStore = Depends(get_store),
) -> dict:
    if end < start:
        raise HTTPException(422, "end must not be before start")
    cells = store.per_cell(area, start, end)
    if len(cells) > store.limits["max_map_cells"]:
        raise HTTPException(
            413, f"{len(cells)} cells with fire; the limit is {store.limits['max_map_cells']}"
        )
    cov = store.coverage
    has_modis = start <= cov.modis[1] and end >= cov.modis[0]
    has_viirs = start <= cov.viirs[1] and end >= cov.viirs[0]
    out = []
    for cell_id, modis, viirs, harm in cells.itertuples(index=False):
        r, c = int(cell_id) // 3600, int(cell_id) % 3600
        out.append(
            {
                "cell_id": int(cell_id),
                "lat": round(r / 10 - 90 + 0.05, 2),
                "lon": round(c / 10 - 180 + 0.05, 2),
                "harmonized": round(float(harm), 2),
                "modis": int(modis) if has_modis else None,
                "viirs": int(viirs) if has_viirs else None,
            }
        )
    return {
        "area": area.as_summary(),
        "unit": "fire_cell_days",
        "grid_degrees": 0.1,
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "cells": out,
    }
