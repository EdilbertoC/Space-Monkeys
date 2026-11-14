# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Viewer export tests, on the real FIRMS rows used by the cube tests."""

import json
from datetime import date

import pytest

pytest.importorskip("duckdb")

from pipeline import cube, viewer  # noqa: E402
from tests.test_cube import report_and_cube  # noqa: E402, F401  (fixture)


def _decode(payload):
    n_cells, v, out = len(payload["cells"]), 0, []
    first = date.fromisoformat(payload["first_date"])
    for d in payload["records"]:
        v += d
        flags, q = v % 4, v // 4
        cell, day = q % n_cells, q // n_cells
        out.append((date.fromordinal(first.toordinal() + day), payload["cells"][cell], flags))
    return out


def test_export_round_trips_cell_days(report_and_cube, tmp_path):  # noqa: F811
    payload = viewer.export("test", tmp_path)
    rows = _decode(payload)
    assert len(rows) == 2  # one cell-day per cell: 2012 (S-NPP) and 2020 (all three)
    by_date = {d: (c, f) for d, c, f in rows}

    corner, flags = by_date[date(2020, 9, 18)]
    assert flags == 3  # MODIS and VIIRS
    row, col = cube.cell_index(-15.21718, -55.5285)
    assert corner == [round(row / 10 - 90, 4), round(col / 10 - 180, 4)]
    assert by_date[date(2012, 1, 20)][1] == 2  # VIIRS only
    assert payload["n_days"] == (date(2020, 9, 18) - date(2012, 1, 20)).days + 1


def test_factor_merges_cells(report_and_cube, tmp_path):  # noqa: F811
    payload = viewer.export("test", tmp_path, factor=100)  # 10° cells
    assert payload["grid_degrees"] == 10
    assert len(payload["cells"]) == 1  # both fires fall in the same 10° cell


def test_write_viewer_embeds_the_data(report_and_cube, tmp_path):  # noqa: F811
    payload = viewer.export("test", tmp_path)
    out = viewer.write_viewer(payload, tmp_path / "v.html")
    html = out.read_text(encoding="utf-8")
    assert viewer.PLACEHOLDER not in html
    assert json.dumps(payload["records"], separators=(",", ":")) in html
