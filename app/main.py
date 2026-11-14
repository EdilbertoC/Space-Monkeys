# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import live, system
from app.config import get_settings
from app.realtime.live import LiveState
from app.realtime.pulse import PulseBroker
from app.realtime.worker import RealtimeWorker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.live = LiveState(
        retention_hours=settings.live_retention_hours,
        fusion_radius_km=settings.fusion_radius_km,
        fusion_window_hours=settings.fusion_window_hours,
    )
    app.state.broker = PulseBroker()
    app.state.worker = None
    if settings.enable_worker:
        app.state.worker = RealtimeWorker(settings, app.state.live, app.state.broker)
        app.state.worker.start()
    yield
    if app.state.worker:
        await app.state.worker.stop()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Space Monkeys API",
        version="0.1.0",
        description=(
            "Burning activity calendar backend. Live fire events fused from NOAA GOES-East "
            "and NASA FIRMS (VIIRS, MODIS) detections, streamed as a pulse over SSE."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    app.include_router(system.router)
    app.include_router(live.router)
    return app


app = create_app()
