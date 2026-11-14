# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
from fastapi import APIRouter, Depends

from app.api.deps import get_broker, get_state, get_worker
from app.realtime.live import LiveState
from app.realtime.pulse import PulseBroker
from app.realtime.worker import RealtimeWorker

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/status", summary="Real-time worker and data source status")
def status(
    state: LiveState = Depends(get_state),
    broker: PulseBroker = Depends(get_broker),
    worker: RealtimeWorker | None = Depends(get_worker),
) -> dict:
    sources = {}
    if worker:
        for name, s in worker.status.sources.items():
            sources[name] = {
                "last_success": s.last_success.isoformat() if s.last_success else None,
                "last_error": s.last_error,
                "last_count": s.last_count,
            }
    return {
        "worker": {
            "enabled": worker is not None,
            "running": bool(worker and worker.status.running),
            "runs": worker.status.runs if worker else 0,
            "last_run": worker.status.last_run.isoformat()
            if worker and worker.status.last_run
            else None,
        },
        "sources": sources,
        "live": {
            "detections": len(state.detections),
            "fires": len(state.fires),
            "updated_at": state.updated_at.isoformat() if state.updated_at else None,
        },
        "stream_subscribers": broker.subscriber_count,
    }
