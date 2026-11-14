# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Empirical fire persistence used by the GOES real-time anchor.

For the forecast cells of Brazil, computes P(VIIRS fire on day d+j | VIIRS fire on day d)
for j = 1..6 over the model's training years (2012-2022), from the NASA FIRMS VIIRS
S-NPP yearly archive (same filters as the forecast model).

Usage: python scripts/persistence_table.py <download_dir> <forecast_mask.npy>
Prints a JSON object to paste into the model's meta.json under "persistence".
"""

import json
import sys
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

URL = "https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/{y}/viirs-snpp_{y}_Brazil.csv"
W, N, RES, NY, NX = -74.0, 6.0, 0.25, 160, 160
YEARS = range(2012, 2023)
MAX_LAG = 6


def main(download_dir: Path, mask_path: Path) -> None:
    mask = np.load(mask_path, allow_pickle=False)
    days = pd.date_range("2012-01-01", "2022-12-31", freq="D")
    fire = np.zeros((len(days), NY, NX), dtype=bool)
    for year in YEARS:
        path = download_dir / f"viirs-snpp_{year}_Brazil.csv"
        if not path.exists():
            with httpx.stream("GET", URL.format(y=year), timeout=300) as r:
                r.raise_for_status()
                path.write_bytes(r.read())
        df = pd.read_csv(path, usecols=["latitude", "longitude", "acq_date", "confidence", "type"])
        df = df[(df["type"] == 0) & df["confidence"].isin(["n", "h"])]
        col = np.floor((df["longitude"].to_numpy() - W) / RES).astype(int)
        row = np.floor((N - df["latitude"].to_numpy()) / RES).astype(int)
        day = (pd.to_datetime(df["acq_date"]) - days[0]).dt.days.to_numpy()
        ok = (col >= 0) & (col < NX) & (row >= 0) & (row < NY)
        fire[day[ok], row[ok], col[ok]] = True
        print(f"{year}: {ok.sum():,} detections", file=sys.stderr)

    fire = fire[:, mask]  # days x forecast cells
    table = {}
    for lag in range(1, MAX_LAG + 1):
        today, later = fire[:-lag], fire[lag:]
        table[str(lag)] = round(float(later[today].mean()), 4)
    base = round(float(fire.mean()), 4)
    print(json.dumps({"p_fire_after_lag_days_given_fire": table, "base_rate": base}, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
