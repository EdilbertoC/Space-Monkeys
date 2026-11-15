# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""U-Net tests, on the real FIRMS rows used by the cube tests: one fire on 2020-09-18
seen by S-NPP (4 pixels kept), NOAA-20 and Aqua, and one S-NPP detection on 2012-01-20.

The key check: the cell-days the U-Net is trained and scored on are exactly the rows
of the training table, so it is compared with LightGBM on the same rows."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("lightgbm")
torch = pytest.importorskip("torch")

from pipeline import cube, training, unet  # noqa: E402
from tests.test_cube import report_and_cube  # noqa: E402, F401  (fixture)

FIRE = (-15.21718, -55.5285)  # 2020-09-18
LONE = (-15.46617, -59.75149)  # 2012-01-20
CPU = torch.device("cpu")


@pytest.fixture
def rec_and_table(report_and_cube, tmp_path):  # noqa: F811
    training.build_training("test", tmp_path, "SNPP")
    table = pd.read_parquet(tmp_path / "processed" / "test" / "training" / "SNPP")
    table["date"] = pd.to_datetime(table["date"]).dt.date
    return unet.load_record("test", tmp_path, "SNPP"), table


def _scored(rec, batch=256):
    """(cell_id, date) -> MODIS label for every cell-day in the U-Net loss mask."""
    b = unet.Batcher(rec, CPU)
    out = {}
    for part in unet.chunks(np.arange(rec.n_days), batch):
        _, y, m = b.batch(part)
        for k, r, c in torch.nonzero(m).tolist():
            out[(int(rec.grid.cell_ids(r, c)), rec.date(part[k]))] = int(y[k, r, c])
    return out


def test_grid_is_padded_around_the_data():
    cells = np.array([cube.cell_id(*FIRE), cube.cell_id(*LONE)])
    g = unet.Grid.around(cells)
    assert g.nrows % unet.MULTIPLE == 0 and g.ncols % unet.MULTIPLE == 0
    r, c = g.index(cells)
    assert (r >= unet.MARGIN).all() and (c >= unet.MARGIN).all()
    assert (r < g.nrows - unet.MARGIN).all() and (c < g.ncols - unet.MARGIN).all()
    assert (g.cell_ids(r, c) == cells).all()


def test_scored_cells_are_the_training_table_rows(rec_and_table):
    rec, table = rec_and_table
    scored = _scored(rec)
    expected = {(int(r.cell_id), r.date): int(r.modis) for r in table.itertuples()}
    assert scored == expected  # same rows, same MODIS labels
    assert rec.date(rec.n_days - 1) == date(2020, 9, 18)  # last standard-processing day


def test_inputs_hold_the_satellite_history(rec_and_table):
    rec, _ = rec_and_table
    b = unet.Batcher(rec, CPU)
    fire_day, lone_next = rec.day(date(2020, 9, 18)), rec.day(date(2012, 1, 21))
    x, _, _ = b.batch(np.array([fire_day, lone_next, 0]))
    assert x.shape == (3, unet.IN_CHANNELS, rec.grid.nrows, rec.grid.ncols)
    pix = unet.INPUTS.index(("n_pixels", True))
    r, c = rec.grid.index(np.array([cube.cell_id(*FIRE)]))
    assert float(x[0, pix, r[0], c[0]]) == pytest.approx(np.log1p(4))  # day t, 4 pixels
    assert float(x[0, unet.N_IN + pix, r[0], c[0]]) == 0  # day t-1: nothing
    r, c = rec.grid.index(np.array([cube.cell_id(*LONE)]))
    assert float(x[1, pix, r[0], c[0]]) == 0  # 2012-01-21: nothing that day
    assert float(x[1, unet.N_IN + pix, r[0], c[0]]) == pytest.approx(np.log1p(1))  # t-1
    assert float(x[2, unet.N_IN + pix].abs().sum()) == 0  # before day 0: empty, no error


def test_unet_keeps_the_grid_size():
    model = unet.UNet(unet.IN_CHANNELS, base=8)
    out = model(torch.zeros(2, unet.IN_CHANNELS, 32, 48))
    assert out.shape == (2, 32, 48)


def test_fit_runs_and_calibrates(rec_and_table):
    rec, _ = rec_and_table
    fire_day = rec.day(date(2020, 9, 18))
    days = np.array([0, 1, fire_day])
    model, info = unet.fit(
        rec, CPU, days, np.array([fire_day]), epochs=2, batch=2, base=8, log=lambda s: None
    )
    assert info["train_rows"] == 27 and info["train_positives"] == 1
    assert info["positive_weight"] == pytest.approx(26.0)
    assert info["epochs_run"] == 2 and info["calibrator"]["x"]
    p = unet.predict_grid(model, unet.Batcher(rec, CPU), days, 2)
    assert p.shape == (3, rec.grid.nrows, rec.grid.ncols) and np.isfinite(p).all()
