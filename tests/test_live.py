# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
import asyncio
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from app.config import get_settings
from app.realtime.live import LiveState
from app.realtime.pulse import PulseBroker
from tests.helpers import make_detections

NOW = datetime(2026, 8, 1, 18, 0, tzinfo=UTC)


def test_ingest_reports_new_then_updated_then_removed():
    state = LiveState(retention_hours=24)
    goes_first = make_detections(
        [{"sensor": "ABI", "lat": -15.0, "lon": -55.0, "time": "2026-08-01T15:00"}]
    )
    diff = state.ingest(goes_first, now=NOW)
    assert len(diff.new) == 1 and diff.updated.empty

    viirs_later = make_detections(
        [{"sensor": "VIIRS", "lat": -15.0, "lon": -55.0, "time": "2026-08-01T17:30"}]
    )
    diff = state.ingest(viirs_later, now=NOW)
    assert diff.new.empty and len(diff.updated) == 1
    assert diff.updated.iloc[0]["sensors"] == ["ABI", "VIIRS"]

    # The same detections again change nothing.
    assert state.ingest(viirs_later, now=NOW).is_empty

    # Two days later everything has expired.
    diff = state.ingest(make_detections([]), now=datetime(2026, 8, 3, 18, 0, tzinfo=UTC))
    assert diff.removed and state.fires.empty


async def test_broker_fan_out_and_slow_subscriber():
    broker = PulseBroker(queue_size=2)
    fast, slow = broker.subscribe(), broker.subscribe()
    for i in range(3):
        broker.publish("pulse", i)
    assert [slow.get_nowait()[1] for _ in range(2)] == [1, 2]  # oldest dropped
    assert fast.qsize() == 2
    broker.unsubscribe(fast)
    assert broker.subscriber_count == 1
    await asyncio.sleep(0)


def _client(monkeypatch) -> TestClient:
    monkeypatch.setenv("ENABLE_WORKER", "false")
    get_settings.cache_clear()
    from app.main import create_app

    return TestClient(create_app())


def test_api_live_endpoints(monkeypatch):
    with _client(monkeypatch) as client:
        assert client.get("/health").json() == {"status": "ok"}

        client.app.state.live.ingest(
            make_detections(
                [
                    {"sensor": "VIIRS", "lat": -15.0, "lon": -55.0, "time": "2026-08-01T17:30"},
                    {"sensor": "VIIRS", "lat": -5.0, "lon": -45.0, "time": "2026-08-01T17:30"},
                ]
            ),
            now=NOW,
        )
        all_fires = client.get("/fires/live").json()
        assert all_fires["type"] == "FeatureCollection"
        assert len(all_fires["features"]) == 2

        boxed = client.get("/fires/live", params={"bbox": "-60,-20,-50,-10"}).json()
        assert len(boxed["features"]) == 1
        assert boxed["features"][0]["geometry"]["coordinates"] == [-55.0, -15.0]

        assert client.get("/fires/live", params={"bbox": "1,2,3"}).status_code == 422
        assert len(client.get("/detections/recent").json()["features"]) == 2

        status = client.get("/status").json()
        assert status["worker"]["enabled"] is False
        assert status["live"]["fires"] == 2
    get_settings.cache_clear()
