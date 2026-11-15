# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Validation figures for the harmonized record (step 8).

1. pantanal_2020.png: Pantanal biome, 2020 by month. MODIS observed, what VIIRS
   saw (naive), and the MODIS-equivalent estimates from S-NPP and NOAA-20 (each
   predicted by a model that never saw 2020) with the 90% interval.
2. brazil_annual.png: Brazil, fire cell-days per year. MODIS observed (2003 on),
   the naive record that switches to VIIRS in 2012 (the step), and the
   MODIS-equivalent estimate from S-NPP with its interval (no step).

Input:  data/processed/<region>/harmonized/<SAT>_monthly.parquet, the cube and
        cell_biome.parquet. Output: docs/figures/.

Usage::

    python -m pipeline.figures brazil
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "docs" / "figures"

# Validated categorical slots (light mode) and chart chrome.
MODIS = "#2a78d6"
ESTIMATE = "#eb6834"
NAIVE = "#1baf7a"
ESTIMATE_2 = "#eda100"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_axisbelow(True)


def _label_end(ax, x, y, text, color):
    ax.annotate(
        text,
        (x, y),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=INK,
        fontweight="bold" if color == ESTIMATE else "normal",
    )


def _monthly(data_dir: Path, region: str, sat: str) -> pd.DataFrame:
    return pd.read_parquet(
        data_dir / "processed" / region / "harmonized" / f"{sat}_monthly.parquet"
    )


def pantanal_2020(data_dir: Path, region: str, out: Path) -> Path:
    s = _monthly(data_dir, region, "SNPP")
    s = s[(s["biome"] == "Pantanal") & (s["year"] == 2020)].set_index("month").reindex(range(1, 13))
    n = _monthly(data_dir, region, "N20")
    n = n[(n["biome"] == "Pantanal") & (n["year"] == 2020)].set_index("month").reindex(range(1, 13))
    x = list(range(1, 13))

    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    _style(ax)
    ax.fill_between(
        x,
        s["low"],
        s["high"],
        color=ESTIMATE,
        alpha=0.15,
        linewidth=0,
        label="S-NPP estimate, 90% interval",
    )
    ax.plot(
        x, s["viirs_naive"], color=NAIVE, lw=2, ls="--", label="S-NPP as seen (no harmonization)"
    )
    ax.plot(x, s["modis"], color=MODIS, lw=2, marker="o", ms=4, label="MODIS observed")
    ax.plot(x, s["estimate"], color=ESTIMATE, lw=2, label="MODIS-equivalent from S-NPP")
    ax.plot(x, n["estimate"], color=ESTIMATE_2, lw=2, ls=":", label="MODIS-equivalent from NOAA-20")
    _label_end(ax, 12, s["viirs_naive"].iloc[-1], "S-NPP as seen", NAIVE)
    ax.set_xticks(x, MONTHS)
    ax.set_xlim(0.6, 13.3)
    ax.set_ylabel("Fire cell-days per month", color=INK_2, fontsize=9)
    ax.set_title(
        "Pantanal biome, 2020: VIIRS translated to MODIS units",
        loc="left",
        fontsize=12,
        color=INK,
        fontweight="bold",
        pad=20,
    )
    ax.text(
        0,
        1.025,
        "Estimates come from models that never saw 2020 (cross-fitted by year blocks).",
        transform=ax.transAxes,
        fontsize=8.5,
        color=INK_2,
    )
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    path = out / "pantanal_2020.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def brazil_annual(data_dir: Path, region: str, out: Path) -> Path:
    cube = data_dir / "processed" / region / "cube"
    biome = data_dir / "processed" / region / "cell_biome.parquet"
    modis = (
        duckdb.connect()
        .execute(f"""
            SELECT year(c.date) AS year, count(DISTINCT (c.cell_id, c.date)) AS modis
            FROM read_parquet('{cube.as_posix()}/**/*.parquet', hive_partitioning = true) c
            JOIN read_parquet('{biome.as_posix()}') b USING (cell_id)
            WHERE c.sensor = 'MODIS' AND NOT c.nrt AND b.biome IS NOT NULL
              AND year(c.date) BETWEEN 2003 AND 2025
            GROUP BY 1 ORDER BY 1
        """)
        .fetchdf()
        .set_index("year")["modis"]
    )
    s = _monthly(data_dir, region, "SNPP")
    s = (
        s[s["year"].between(2012, 2025)]
        .groupby("year")[["estimate", "low", "high", "viirs_naive"]]
        .sum()
    )
    naive = pd.concat([modis.loc[:2011], s["viirs_naive"]])

    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    _style(ax)
    ax.plot(
        naive.index,
        naive / 1e3,
        color=NAIVE,
        lw=2,
        ls="--",
        label="Naive record: MODIS, then S-NPP from 2012",
    )
    ax.plot(modis.index, modis / 1e3, color=MODIS, lw=2, marker="o", ms=3.5, label="MODIS observed")
    ax.plot(s.index, s["estimate"] / 1e3, color=ESTIMATE, lw=2, label="MODIS-equivalent from S-NPP")
    # No interval band here: the 90% interval is calibrated on monthly biome totals,
    # and summing monthly limits would overstate the yearly uncertainty.
    ax.axvline(2012, color=MUTED, lw=1, ls=":")
    ax.text(2012.15, ax.get_ylim()[1] * 0.97, "VIIRS starts", fontsize=8.5, color=INK_2, va="top")
    ax.set_ylabel("Fire cell-days per year (thousands)", color=INK_2, fontsize=9)
    ax.set_title(
        "Brazil: the naive record jumps in 2012; the harmonized one does not",
        loc="left",
        fontsize=12,
        color=INK,
        fontweight="bold",
        pad=20,
    )
    ax.text(
        0,
        1.025,
        "Estimates for each year come from a model that never saw it.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=INK_2,
    )
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    path = out / "brazil_annual.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("region", nargs="?", default="brazil")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in (pantanal_2020, brazil_annual):
        print(f(args.data_dir, args.region, OUT_DIR))


if __name__ == "__main__":
    main()
