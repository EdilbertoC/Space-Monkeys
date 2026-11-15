# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import docs, forecast, history, live, system
from app.config import get_settings
from app.forecast.model import ForecastModel
from app.forecast.service import ForecastService
from app.history.store import HistoryStore
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
    app.state.history = None
    if (settings.history_data_dir / "meta.json").exists():
        app.state.history = HistoryStore(settings.history_data_dir)
    else:
        logging.getLogger(__name__).warning(
            "No historical data in %s: /history answers 503", settings.history_data_dir
        )
    app.state.forecast = None
    if settings.enable_forecast:
        model = ForecastModel(Path(settings.forecast_model_dir))
        app.state.forecast = ForecastService(
            model,
            anchor_hours=settings.forecast_anchor_hours,
            source=settings.forecast_source,
        )
    app.state.worker = None
    if settings.enable_worker:
        app.state.worker = RealtimeWorker(
            settings, app.state.live, app.state.broker, forecast=app.state.forecast
        )
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
            "and NASA FIRMS (VIIRS, MODIS) detections, streamed as a pulse over SSE, and a "
            "7-day fire forecast from a U-Net trained on the NASA FIRMS archive."
        ),
        lifespan=lifespan,
        docs_url=None,  # /docs and /redoc show the hand-written contract (app/api/docs.py)
        redoc_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(system.router)
    app.include_router(live.router)
    app.include_router(history.router)
    app.include_router(forecast.router)
    app.include_router(docs.router)
    return app


app = create_app()
