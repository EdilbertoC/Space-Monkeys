# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Fire forecast endpoints."""

import threading
from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.api.deps import parse_bbox
from app.forecast.regions import grid_labels
from app.forecast.service import ForecastService

router = APIRouter(tags=["forecast"])


def get_forecast(request: Request) -> ForecastService:
    service = request.app.state.forecast
    if service is None:
        raise HTTPException(503, "The forecast is disabled")
    return service


@router.get(
    "/forecast",
    summary="Probability of fire per 0.25° cell for the next 7 days (lead 1 = today)",
)
def forecast(
    min_p: float = Query(0.05, ge=0, le=1, description="Only cells reaching this on any day"),
    bbox: tuple[float, float, float, float] | None = Depends(parse_bbox),
    service: ForecastService = Depends(get_forecast),
) -> dict:
    try:
        return service.payload(min_p=min_p, bbox=bbox)
    except LookupError as exc:
        raise HTTPException(503, f"Forecast not available yet: {exc}") from exc


@router.get("/forecast/model", summary="Forecast model card: data, inputs and test scores")
def forecast_model(service: ForecastService = Depends(get_forecast)) -> dict:
    meta = dict(service.model.meta)
    meta["status"] = {
        "issued_for": service.current.issued_for.isoformat() if service.current else None,
        "last_error": service.last_error,
    }
    return meta


_labels_lock = threading.Lock()


@router.get(
    "/forecast/summary",
    summary="Expected fire cells per biome (and Brazil) for each of the next 7 days",
)
def forecast_summary(request: Request, service: ForecastService = Depends(get_forecast)) -> dict:
    store = getattr(request.app.state, "history", None)
    if store is None:
        raise HTTPException(503, "Biome labels need the historical data (HISTORY_DATA_DIR)")
    with _labels_lock:
        labels = getattr(request.app.state, "forecast_labels", None)
        if labels is None:
            labels = grid_labels(store.cell_biome_frame(), service.model.grid)
            request.app.state.forecast_labels = labels
    try:
        return service.region_summary(*labels)
    except LookupError as exc:
        raise HTTPException(503, f"Forecast not available yet: {exc}") from exc


Biome = Literal["Amazon", "Caatinga", "Cerrado", "Atlantic Forest", "Pampa", "Pantanal"]


@router.get(
    "/forecast/outlook",
    summary="The 7-day forecast summed over an area, next to the normal for those dates",
)
def forecast_outlook(
    request: Request,
    bbox: tuple[float, float, float, float] | None = Depends(parse_bbox),
    biome: Biome | None = Query(None, description="A Brazilian biome (needs the history data)."),
    service: ForecastService = Depends(get_forecast),
) -> dict:
    if (bbox is None) == (biome is None):
        raise HTTPException(400, "give exactly one of bbox, biome")
    lat, lon = service.model.grid.centers()
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    if bbox is not None:
        w, s, e, n = bbox
        select = (lon2 >= w) & (lon2 <= e) & (lat2 >= s) & (lat2 <= n)
        area = {"kind": "bbox", "bbox": list(bbox)}
    else:
        store = getattr(request.app.state, "history", None)
        if store is None:
            raise HTTPException(503, "biome areas need the historical data; use bbox")
        # The 0.1° history cell that holds each forecast cell centre.
        rows = np.floor((lat2 + 90) * 10).astype(np.int64)
        cols = np.floor((lon2 + 180) * 10).astype(np.int64)
        select = np.isin(rows * 3600 + cols, store.biome_cell_ids(biome))
        area = {"kind": "biome", "biome": biome}
    try:
        out = service.outlook(select)
    except LookupError as exc:
        raise HTTPException(503, f"Forecast not available yet: {exc}") from exc
    if out["forecast_cells"] == 0:
        raise HTTPException(422, "the area holds no forecast cell")
    return {"area": area, **out}
