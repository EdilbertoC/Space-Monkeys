from fastapi import HTTPException, Query, Request

from app.realtime.live import LiveState
from app.realtime.pulse import PulseBroker
from app.realtime.worker import RealtimeWorker


def get_state(request: Request) -> LiveState:
    return request.app.state.live


def get_broker(request: Request) -> PulseBroker:
    return request.app.state.broker


def get_worker(request: Request) -> RealtimeWorker | None:
    return request.app.state.worker


def parse_bbox(
    bbox: str | None = Query(
        default=None,
        description="Bounding box as west,south,east,north in degrees",
        examples=["-60,-20,-50,-10"],
    ),
) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    try:
        w, s, e, n = (float(v) for v in bbox.split(","))
    except ValueError as exc:
        raise HTTPException(422, "bbox must be four numbers: west,south,east,north") from exc
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise HTTPException(422, "bbox must satisfy west<east and south<north within lon/lat range")
    return w, s, e, n
