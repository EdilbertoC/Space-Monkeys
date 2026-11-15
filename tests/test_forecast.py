# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
import math
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.forecast.history import grid_counts
from app.forecast.model import ForecastModel, Grid, day_of_year0
from app.forecast.service import Forecast, ForecastService
from tests.helpers import make_detections

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "fire_forecast_brazil"


@pytest.fixture(scope="module")
def model() -> ForecastModel:
    return ForecastModel(MODEL_DIR)


def test_grid_cells():
    grid = Grid(-74.0, -34.0, -34.0, 6.0, 0.25)
    assert grid.shape == (160, 160)
    row, col, inside = grid.cell_of(np.array([5.9, -33.9, 10.0]), np.array([-73.9, -34.1, -50]))
    assert (row[:2] == [0, 159]).all() and (col[:2] == [0, 159]).all()
    assert inside.tolist() == [True, True, False]
    lat, lon = grid.centers()
    assert lat[0] == pytest.approx(5.875) and lon[0] == pytest.approx(-73.875)


def test_features_follow_the_training_layout(model):
    counts = np.zeros((30, 160, 160), dtype=np.uint16)
    counts[-1, 50, 60] = 3  # issue day
    counts[0, 50, 60] = 2  # 29 days earlier: only in the 30-day sum
    issue = date(2024, 8, 20)
    x = model.features(counts, issue)
    assert x.shape == (20, 160, 160)
    assert x[6, 50, 60] == pytest.approx(math.log1p(3))
    assert x[:6, 50, 60].sum() == 0
    assert x[7, 50, 60] == pytest.approx(math.log1p(5))
    assert np.array_equal(x[8], model.climatology[day_of_year0(date(2024, 8, 21))])
    angle = 2 * math.pi * day_of_year0(issue) / 365.25
    assert x[15, 0, 0] == pytest.approx(math.sin(angle))
    assert np.array_equal(x[17], model.mask.astype(np.float32))
    assert x[18, 0, 0] == 1 and x[18, -1, 0] == -1 and x[19, 0, 0] == -1


def test_predict_returns_probabilities(model):
    counts = np.zeros((30, 160, 160), dtype=np.uint16)
    counts[-7:, 80, 80] = 20  # a week of intense fire in one cell
    p = model.predict(counts, date(2024, 8, 20))
    assert p.shape == (7, 160, 160)
    assert ((p >= 0) & (p <= 1)).all()
    quiet = model.predict(np.zeros_like(counts), date(2024, 8, 20))
    assert p[0, 80, 80] > quiet[0, 80, 80]  # recent fire raises tomorrow's risk


def test_grid_counts_by_day_and_confidence():
    grid = Grid(-74.0, -34.0, -34.0, 6.0, 0.25)
    df = make_detections(
        [
            {"sensor": "VIIRS", "lat": -10.1, "lon": -50.1, "time": "2024-08-19T04:00"},
            {"sensor": "VIIRS", "lat": -10.1, "lon": -50.1, "time": "2024-08-20T17:00"},
            {
                "sensor": "VIIRS",
                "lat": -10.1,
                "lon": -50.1,
                "time": "2024-08-20T17:00",
                "confidence": 0.3,
            },  # low confidence: ignored
        ]
    )
    counts = grid_counts(df, grid, date(2024, 8, 19), 2)
    row, col, _ = grid.cell_of(np.array([-10.1]), np.array([-50.1]))
    assert counts[:, row[0], col[0]].tolist() == [1, 1]
    assert counts.sum() == 2


def _service_with_flat_forecast(model, p: float = 0.01) -> ForecastService:
    service = ForecastService(model, anchor_hours=12)
    base = np.full((7, 160, 160), p, dtype=np.float32)
    service.current = Forecast(
        date(2024, 8, 20), datetime(2024, 8, 21, 1, tzinfo=UTC), base, [1] * 30
    )
    return service


def test_goes_anchor_raises_today_and_decays(model):
    service = _service_with_flat_forecast(model)
    fires = pd.DataFrame(
        {
            "lat": [-10.1],
            "lon": [-50.1],
            "confidence": [0.8],
            "last_seen": [pd.Timestamp("2024-08-21T15:00", tz="UTC")],  # today = lead 1
        }
    )
    n = service.apply_anchor(fires, now=datetime(2024, 8, 21, 16, tzinfo=UTC))
    assert n == 1
    row, col, _ = model.grid.cell_of(np.array([-10.1]), np.array([-50.1]))
    p = service.current.final[:, row[0], col[0]]
    assert p[0] == pytest.approx(0.8)
    assert p[1] == pytest.approx(max(0.01, 0.8 * service.persistence[1]))
    assert (np.diff(p) <= 1e-6).all()  # decays over the week
    assert service.current.base[0, row[0], col[0]] == pytest.approx(0.01)  # base untouched


def test_anchor_ignores_old_fires_and_never_lowers(model):
    service = _service_with_flat_forecast(model, p=0.9)
    fires = pd.DataFrame(
        {
            "lat": [-10.1],
            "lon": [-50.1],
            "confidence": [0.5],
            "last_seen": [pd.Timestamp("2024-08-21T15:00", tz="UTC")],
        }
    )
    service.apply_anchor(fires, now=datetime(2024, 8, 21, 16, tzinfo=UTC))
    assert np.allclose(service.current.final, 0.9)
    assert service.apply_anchor(fires, now=datetime(2024, 8, 22, 12, tzinfo=UTC)) == 0


def test_forecast_endpoint(monkeypatch, model):
    monkeypatch.setenv("ENABLE_WORKER", "false")
    get_settings.cache_clear()
    from app.main import create_app

    with TestClient(create_app()) as client:
        assert client.get("/forecast").status_code == 503  # nothing issued yet
        service = client.app.state.forecast
        flat = _service_with_flat_forecast(model, p=0.2)
        service.current = flat.current
        body = client.get("/forecast", params={"min_p": 0.1}).json()
        assert body["issued_for"] == "2024-08-20"
        assert body["valid_dates"][0] == "2024-08-21"
        assert len(body["cells"]) == int(model.mask.sum())
        assert body["cells"][0]["p"] == [0.2] * 7
        assert client.get("/forecast", params={"min_p": 0.5}).json()["cells"] == []
        card = client.get("/forecast/model").json()
        assert card["test_avg_precision"]["unet"] > card["test_avg_precision"]["climatology"]
    get_settings.cache_clear()


def test_region_summary_counts_expected_fire_cells(model):
    from app.forecast.regions import grid_labels

    service = _service_with_flat_forecast(model, p=0.6)
    grid = model.grid
    # Label one forecast cell inside the mask with a biome, through two 0.1° cells.
    rows, cols = np.nonzero(model.mask)
    lat_c = grid.north - grid.res * (rows[0] + 0.5)
    lon_c = grid.west + grid.res * (cols[0] + 0.5)

    def cid(lat, lon):
        return int(np.floor((lat + 90) * 10)) * 3600 + int(np.floor((lon + 180) * 10))

    cells = pd.DataFrame(
        {
            "cell_id": [cid(lat_c, lon_c), cid(lat_c, lon_c - 0.1), cid(0.0, 0.0)],
            "biome": ["Pantanal", "Pantanal", None],
        }
    )
    labels, names = grid_labels(cells, grid)
    assert names == ["Pantanal"] and int((labels == 0).sum()) == 1
    summary = service.region_summary(labels, names)
    pantanal, brazil = summary["regions"]
    assert (pantanal["kind"], pantanal["name"], pantanal["cells"]) == ("biome", "Pantanal", 1)
    assert pantanal["expected_fire_cells"] == [0.6] * 7
    assert pantanal["cells_at_least_50"] == [1] * 7
    assert brazil == {**pantanal, "kind": "country", "name": "Brazil"}
    assert summary["valid_dates"][0] == "2024-08-21"
