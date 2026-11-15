# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""/history API tests on a serving table built from the real FIRMS rows of the cube
tests (one S-NPP detection on 2012-01-20; one fire on 2020-09-18 seen by S-NPP,
NOAA-20 and Aqua). Every 200 response is checked against the contract schema."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("shapely")
yaml = pytest.importorskip("yaml")
jsonschema = pytest.importorskip("jsonschema")

import shapely  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from referencing import Registry, Resource  # noqa: E402
from referencing.jsonschema import DRAFT202012  # noqa: E402

from app.config import get_settings  # noqa: E402
from pipeline import biomes, cube, serving  # noqa: E402
from tests.test_cube import report_and_cube  # noqa: E402, F401  (fixture)
from tests.test_openapi import SPEC  # noqa: E402

FIRE = (-15.21718, -55.5285)
LONE = (-15.46617, -59.75149)
REGISTRY = Registry().with_resource(
    "urn:spec", Resource.from_contents(SPEC, default_specification=DRAFT202012)
)


def conforms(body, schema_name):
    jsonschema.Draft202012Validator(
        {"$ref": f"urn:spec#/components/schemas/{schema_name}"},
        registry=REGISTRY,
        format_checker=jsonschema.FormatChecker(),
    ).validate(body)


@pytest.fixture
def client(report_and_cube, tmp_path, monkeypatch):  # noqa: F811
    cells = np.array(
        [
            cube.cell_id(*p) + dr * cube.N_COLS + dc
            for p in (FIRE, LONE)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
        ]
    )
    labels = biomes.label_cells(cells, {"Pantanal": shapely.box(-61, -16, -54, -14)})
    labels.to_parquet(tmp_path / "processed" / "test" / "cell_biome.parquet", index=False)
    serving.build_serving("test", tmp_path)
    monkeypatch.setenv("ENABLE_WORKER", "false")
    monkeypatch.setenv("HISTORY_DATA_DIR", str(tmp_path / "processed" / "test" / "serving"))
    get_settings.cache_clear()
    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()


def test_meta(client):
    body = client.get("/history/meta").json()
    conforms(body, "Meta")
    assert body["harmonization"]["status"] == "available"
    assert body["coverage"]["modis"]["start"] == "2020-09-18"  # the only MODIS day here
    assert body["biomes"] == [{"name": "Pantanal", "cells": 18}]


def test_calendar_counts_cell_days_and_follows_modis(client):
    r = client.get(
        "/history/calendar",
        params={
            "biome": "Pantanal",
            "start": "2020-09-01",
            "end": "2020-09-30",
            "granularity": "month",
        },
    )
    assert r.status_code == 200
    body = r.json()
    conforms(body, "Calendar")
    (bucket,) = body["buckets"]
    # S-NPP and NOAA-20 on the same cell-day count once (OR, never summed).
    assert (bucket["modis"], bucket["viirs"]) == (1, 1)
    assert bucket["harmonized"] == {"value": 1.0, "low": 1.0, "high": 1.0}
    assert bucket["harmonized_source"] == "modis"


def test_calendar_is_null_where_a_sensor_has_no_data(client):
    body = client.get(
        "/history/calendar",
        params={
            "biome": "Pantanal",
            "start": "2012-01-01",
            "end": "2012-01-31",
            "granularity": "month",
        },
    ).json()
    (bucket,) = body["buckets"]
    assert bucket["viirs"] == 1  # the S-NPP detection of 2012-01-20
    assert bucket["modis"] is None and bucket["harmonized"] is None  # before MODIS data here


def test_weekly_buckets_follow_the_contract_weeks(client):
    body = client.get(
        "/history/calendar",
        params={"bbox": "-56,-16,-55,-15", "start": "2020-12-20", "end": "2020-12-31"},
    ).json()
    assert [(b["start"], b["end"]) for b in body["buckets"]] == [
        ("2020-12-20", "2020-12-22"),
        ("2020-12-23", "2020-12-31"),  # week 51 runs to 31 December
    ]


@pytest.mark.parametrize(
    ("params", "status"),
    [
        ({"start": "2020-09-01", "end": "2020-09-30"}, 400),  # no area
        (
            {
                "biome": "Pantanal",
                "bbox": "-56,-16,-55,-15",
                "start": "2020-09-01",
                "end": "2020-09-30",
            },
            400,
        ),  # two areas
        ({"biome": "Mars", "start": "2020-09-01", "end": "2020-09-30"}, 422),
        ({"area_id": "nope", "start": "2020-09-01", "end": "2020-09-30"}, 404),
        ({"bbox": "-74,-34,-28,6", "start": "2020-09-01", "end": "2020-09-30"}, 413),
        ({"biome": "Pantanal", "start": "2020-09-30", "end": "2020-09-01"}, 422),
    ],
)
def test_calendar_errors(client, params, status):
    assert client.get("/history/calendar", params=params).status_code == status


def test_drawn_polygon_becomes_an_area(client):
    lat, lon = FIRE
    ring = [
        [lon - 0.2, lat - 0.2],
        [lon + 0.2, lat - 0.2],
        [lon + 0.2, lat + 0.2],
        [lon - 0.2, lat + 0.2],
        [lon - 0.2, lat - 0.2],
    ]
    r = client.post("/history/areas", json={"geometry": {"type": "Polygon", "coordinates": [ring]}})
    assert r.status_code == 200
    area = r.json()
    conforms(area, "Area")
    assert area["cells"] == 16  # 4 x 4 cell centres inside a 0.4° square
    again = client.post(
        "/history/areas", json={"geometry": {"type": "Polygon", "coordinates": [ring]}}
    )
    assert again.json()["area_id"] == area["area_id"]  # same cells, same id

    body = client.get(
        "/history/calendar",
        params={
            "area_id": area["area_id"],
            "start": "2020-09-18",
            "end": "2020-09-18",
            "granularity": "day",
        },
    ).json()
    assert body["area"]["kind"] == "area_id"
    assert body["buckets"][0]["modis"] == 1


def test_bad_polygon_is_rejected(client):
    r = client.post("/history/areas", json={"geometry": {"type": "Point", "coordinates": [0, 0]}})
    assert r.status_code == 422


def test_map_lists_cells_with_fire(client):
    body = client.get(
        "/history/map", params={"biome": "Pantanal", "start": "2020-09-18", "end": "2020-09-18"}
    ).json()
    conforms(body, "Map")
    (cell,) = body["cells"]
    assert cell["cell_id"] == cube.cell_id(*FIRE)
    assert (cell["modis"], cell["viirs"], cell["harmonized"]) == (1, 1, 1.0)
    assert (cell["lat"], cell["lon"]) == (-15.25, -55.55)


def test_analytics_need_enough_years(client):
    # Two days of data are not a climate: the routes answer 422, not 500.
    for path, params in [
        ("/history/climatology", {"biome": "Pantanal", "series": "modis"}),
        ("/history/anomaly", {"biome": "Pantanal", "start": "2020-09-18", "end": "2020-09-18"}),
        ("/history/season", {"biome": "Pantanal", "year": 2020}),
        ("/history/trend", {"biome": "Pantanal"}),
    ]:
        assert client.get(path, params=params).status_code == 422, path


def test_history_unavailable_without_data(tmp_path, monkeypatch):
    monkeypatch.setenv("ENABLE_WORKER", "false")
    monkeypatch.setenv("HISTORY_DATA_DIR", str(tmp_path / "missing"))
    get_settings.cache_clear()
    from app.main import create_app

    with TestClient(create_app()) as c:
        assert c.get("/history/meta").status_code == 503
    get_settings.cache_clear()


def test_serving_uses_noaa20_when_snpp_was_not_observing(report_and_cube, tmp_path, monkeypatch):  # noqa: F811
    """Which satellite's estimate is served, per day (plain numbers as estimates)."""
    cell = cube.cell_id(*FIRE)
    labels = biomes.label_cells(np.array([cell]), {"Pantanal": shapely.box(-61, -16, -54, -14)})
    labels.to_parquet(tmp_path / "processed" / "test" / "cell_biome.parquet", index=False)
    for sat, rows in (
        ("SNPP", {"2020-09-18": 0.5, "2020-09-19": 0.1}),
        ("N20", {"2020-09-18": 0.7, "2020-09-19": 0.6, "2020-09-20": 0.9}),
    ):
        d = tmp_path / "processed" / "test" / "harmonized" / sat / "year=2020"
        d.mkdir(parents=True)
        pd.DataFrame(
            {"cell_id": cell, "date": pd.to_datetime(list(rows)), "equiv": list(rows.values())}
        ).to_parquet(d / "harmonized.parquet", index=False)
    outage = date(2020, 9, 19)  # S-NPP not observing: its 0.1 is only a lag row
    monkeypatch.setattr(serving, "observation_gaps", lambda region, data_dir: {"SNPP": [outage]})

    meta = serving.build_serving("test", tmp_path)
    served = pd.read_parquet(tmp_path / "processed" / "test" / "serving" / "cell_day")
    est = {str(pd.Timestamp(r.date).date()): r.est for r in served.itertuples() if r.est == r.est}
    assert est == pytest.approx({"2020-09-18": 0.5, "2020-09-19": 0.6, "2020-09-20": 0.9})
    assert meta["harmonization"]["viirs_satellites"] == [
        {"name": "SNPP", "start": "2020-09-18", "end": "2020-09-19"},
        {"name": "N20", "start": "2020-09-20", "end": "2020-09-20"},
    ]
    gap = meta["known_gaps"][-1]
    assert (gap["sensor"], gap["period"]) == ("VIIRS", "2020-09-19")
    assert "NOAA-20" in gap["note"]
    conforms(meta, "Meta")
