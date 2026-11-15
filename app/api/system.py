# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse

from app.api.deps import get_broker, get_state, get_worker
from app.realtime.live import LiveState
from app.realtime.pulse import PulseBroker
from app.realtime.worker import RealtimeWorker

router = APIRouter(tags=["system"])

LLMS_TXT = Path(__file__).resolve().parents[2] / "llms.txt"


@router.get("/llms.txt", include_in_schema=False, response_class=PlainTextResponse)
def llms_txt() -> str:
    """Plain-language guide to this API for language models (llmstxt.org format)."""
    if not LLMS_TXT.exists():
        raise HTTPException(404, "llms.txt is not deployed with this server")
    return LLMS_TXT.read_text(encoding="utf-8")


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
