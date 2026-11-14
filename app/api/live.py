"""Live endpoints: fused fires, raw recent detections and the SSE pulse stream."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from app.api.deps import get_broker, get_state, parse_bbox
from app.config import get_settings
from app.realtime.geojson import feature_collection, filter_bbox
from app.realtime.live import LiveState
from app.realtime.pulse import PulseBroker

router = APIRouter(tags=["live"])

BBox = tuple[float, float, float, float] | None


@router.get("/fires/live", summary="Fused fire events from the last retention window (GeoJSON)")
def live_fires(
    bbox: BBox = Depends(parse_bbox),
    min_confidence: float = Query(0.0, ge=0, le=1),
    state: LiveState = Depends(get_state),
) -> dict:
    fires = filter_bbox(state.fires, bbox)
    fires = fires[fires["confidence"] >= min_confidence]
    collection = feature_collection(fires, "fire_id")
    collection["updated_at"] = state.updated_at.isoformat() if state.updated_at else None
    return collection


@router.get("/detections/recent", summary="Raw detections from every sensor (GeoJSON)")
def recent_detections(
    bbox: BBox = Depends(parse_bbox),
    sensor: Literal["VIIRS", "MODIS", "ABI"] | None = None,
    limit: int = Query(5000, ge=1, le=50000),
    state: LiveState = Depends(get_state),
) -> dict:
    df = filter_bbox(state.detections, bbox)
    if sensor:
        df = df[df["sensor"] == sensor]
    df = df.sort_values("time", ascending=False).head(limit)
    return feature_collection(df, "detection_id")


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


@router.get(
    "/fires/stream",
    summary="Server-Sent Events: a `snapshot` on connect, then a `pulse` after every poll",
    response_class=StreamingResponse,
)
async def fire_stream(
    request: Request,
    state: LiveState = Depends(get_state),
    broker: PulseBroker = Depends(get_broker),
) -> StreamingResponse:
    heartbeat = get_settings().sse_heartbeat_seconds

    async def events() -> AsyncIterator[str]:
        queue = broker.subscribe()
        try:
            snapshot = feature_collection(state.fires, "fire_id")
            snapshot["updated_at"] = state.updated_at.isoformat() if state.updated_at else None
            yield _sse("snapshot", snapshot)
            while not await request.is_disconnected():
                try:
                    event, data = await asyncio.wait_for(queue.get(), timeout=heartbeat)
                except TimeoutError:
                    yield ": ping\n\n"  # keeps proxies from closing an idle stream
                    continue
                yield _sse(event, data)
        finally:
            broker.unsubscribe(queue)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
