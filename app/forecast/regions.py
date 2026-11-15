# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Label every cell of the forecast grid with a region (an IBGE biome), so the
per-cell forecast can be summarised per region without the client guessing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.forecast.model import Grid

N_COLS = 3600  # 0.1° history grid


def grid_labels(cell_biome: pd.DataFrame, grid: Grid) -> tuple[np.ndarray, list[str]]:
    """(labels, names): labels[row, col] is the index in `names` of the biome holding
    most of the 0.1° cells whose centre falls in that forecast cell, or -1 for none.

    `cell_biome` has `cell_id` (0.1° grid) and `biome` (None outside Brazil)."""
    labelled = cell_biome.dropna(subset=["biome"])
    lat = (labelled["cell_id"] // N_COLS) / 10 - 90 + 0.05
    lon = (labelled["cell_id"] % N_COLS) / 10 - 180 + 0.05
    row, col, inside = grid.cell_of(lat.to_numpy(), lon.to_numpy())
    frame = pd.DataFrame(
        {"row": row[inside], "col": col[inside], "biome": labelled["biome"].to_numpy()[inside]}
    )
    names = sorted(frame["biome"].unique())
    majority = (
        frame.groupby(["row", "col"])["biome"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
    )
    labels = np.full(grid.shape, -1, dtype=np.int64)
    index = {name: i for i, name in enumerate(names)}
    labels[majority["row"].to_numpy(), majority["col"].to_numpy()] = (
        majority["biome"].map(index).to_numpy()
    )
    return labels, names
