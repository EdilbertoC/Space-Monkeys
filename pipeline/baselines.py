# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Step 6: baselines for the VIIRS -> MODIS calibration, on one test set.

Every model predicts p = P(MODIS saw fire in the cell that day) for each row of a
training table (pipeline.training). Compared on the same held-out years:

- naive: "VIIRS saw fire, so MODIS would have" (p = 1 if the cell has its own
  VIIRS detection, else 0). No harmonization; shown as the reference to beat.
- ratio: share of rows with MODIS fire in the training years, per month and per
  "own VIIRS detection or neighbour/lag only". The simplest calibration.
- lightgbm: gradient-boosted trees on all features (pipeline.training.FEATURES).
  The positive class is weighted by n_negative / n_positive, as planned; that
  multiplies the predicted odds by the weight, so the odds are divided back
  (p / (p + (1 - p) * w)) to keep probabilities, and therefore totals, honest.
  The last training year is held out for early stopping, then the model is
  refit on all training years with the best number of rounds.

Metrics (test years only):

- rows: log loss, Brier score, ROC AUC, average precision
- totals, what the calendar shows: sum of p against the MODIS fire-days, per
  month for the region (WAPE, bias, correlation) and per cell and month (WAPE)
- by year: predicted against actual MODIS fire-days (watch for drift)

Totals cover MODIS fire-days inside the rows only; the coverage gap reported by
pipeline.training is corrected later (step 8).

Output: data/processed/<region>/models/<SAT>/ (report.json, lightgbm.txt,
ratio.json, test_predictions.parquet)

Usage::

    python -m pipeline.baselines pantanal
    python -m pipeline.baselines pantanal --satellite N20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from pipeline.training import FEATURES, SATELLITES, TARGET

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
SEED = 42
LGB_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": SEED,
    "deterministic": True,
    "verbose": -1,
}
MAX_ROUNDS = 3000
EARLY_STOP = 100


def load_table(region: str, data_dir: Path, satellite: str) -> pd.DataFrame:
    path = data_dir / "processed" / region / "training" / satellite
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `python -m pipeline.training {region} --satellite {satellite}`"
        )
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["year"].astype(int)
    return df


# ---- models -----------------------------------------------------------------
def naive_predict(df: pd.DataFrame) -> np.ndarray:
    return (df["v_pixels"] > 0).astype(float).to_numpy()


def _ratio_key(df: pd.DataFrame) -> pd.Series:
    return df["month"].astype(str) + "|" + (df["v_pixels"] > 0).astype(int).astype(str)


def fit_ratio(train: pd.DataFrame) -> dict:
    table = train.groupby(_ratio_key(train))[TARGET].mean().to_dict()
    return {"by_month_and_own": table, "overall": float(train[TARGET].mean())}


def ratio_predict(df: pd.DataFrame, model: dict) -> np.ndarray:
    table = model["by_month_and_own"]
    return _ratio_key(df).map(table).fillna(model["overall"]).to_numpy(dtype=float)


def positive_weight(y: pd.Series | np.ndarray) -> float:
    y = np.asarray(y)
    pos = y.sum()
    return float((len(y) - pos) / pos) if pos else 1.0


def unweight(p: np.ndarray, w: float) -> np.ndarray:
    """Undo a positive-class weight w: the weighted model's odds are w times too high."""
    p = np.asarray(p, dtype=float)
    return p / (p + (1.0 - p) * w)


def fit_lightgbm(train: pd.DataFrame, valid_year: int) -> tuple[lgb.Booster, dict]:
    w = positive_weight(train[TARGET])
    params = {**LGB_PARAMS, "scale_pos_weight": w}
    fit_part, valid = train[train["year"] < valid_year], train[train["year"] == valid_year]
    best_rounds = MAX_ROUNDS
    if len(fit_part) and len(valid) and valid[TARGET].nunique() == 2:
        booster = lgb.train(
            params,
            lgb.Dataset(fit_part[FEATURES], fit_part[TARGET]),
            num_boost_round=MAX_ROUNDS,
            valid_sets=[lgb.Dataset(valid[FEATURES], valid[TARGET])],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
        )
        best_rounds = booster.best_iteration or MAX_ROUNDS
    final = lgb.train(params, lgb.Dataset(train[FEATURES], train[TARGET]), best_rounds)
    return final, {"positive_weight": w, "rounds": best_rounds, "valid_year": valid_year}


def lightgbm_predict(booster: lgb.Booster, df: pd.DataFrame, w: float) -> np.ndarray:
    return unweight(booster.predict(df[FEATURES]), w)


# ---- metrics ----------------------------------------------------------------
def wape(pred: pd.Series, actual: pd.Series) -> float | None:
    total = float(actual.sum())
    return float((pred - actual).abs().sum() / total) if total else None


def evaluate(df: pd.DataFrame, p: np.ndarray) -> dict:
    y = df[TARGET].to_numpy()
    pc = np.clip(p, 1e-4, 1 - 1e-4)
    rows = {
        "log_loss": float(log_loss(y, pc, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, p)) if y.sum() else None,
    }
    frame = pd.DataFrame(
        {
            "cell_id": df["cell_id"].to_numpy(),
            "year": df["year"].to_numpy(),
            "month": df["date"].dt.month.to_numpy(),
            "y": y,
            "p": p,
        }
    )
    monthly = frame.groupby(["year", "month"])[["y", "p"]].sum()
    cell_month = frame.groupby(["cell_id", "year", "month"])[["y", "p"]].sum()
    yearly = frame.groupby("year")[["y", "p"]].sum()
    total_y = float(frame["y"].sum())
    totals = {
        "modis_fire_days": int(total_y),
        "predicted": round(float(frame["p"].sum()), 1),
        "bias": round(float(frame["p"].sum() - total_y) / total_y, 4) if total_y else None,
        "monthly_wape": wape(monthly["p"], monthly["y"]),
        "monthly_corr": float(monthly["p"].corr(monthly["y"])) if len(monthly) > 2 else None,
        "cell_month_wape": wape(cell_month["p"], cell_month["y"]),
    }
    by_year = [
        {"year": int(yr), "modis": int(r["y"]), "predicted": round(float(r["p"]), 1)}
        for yr, r in yearly.iterrows()
    ]
    return {"rows": rows, "totals": totals, "by_year": by_year}


# ---- driver -----------------------------------------------------------------
def run(region: str, data_dir: Path = DEFAULT_DATA_DIR, satellite: str = "SNPP") -> dict:
    df = load_table(region, data_dir, satellite)
    last_train = SATELLITES[satellite]["train_last_year"]
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    if train.empty or test.empty or train[TARGET].sum() == 0:
        raise ValueError("need training rows with MODIS fire and test rows")

    ratio = fit_ratio(train)
    booster, lgb_info = fit_lightgbm(train, valid_year=last_train)
    preds = {
        "naive": naive_predict(test),
        "ratio": ratio_predict(test, ratio),
        "lightgbm": lightgbm_predict(booster, test, lgb_info["positive_weight"]),
    }
    importance = sorted(
        zip(FEATURES, booster.feature_importance("gain"), strict=True), key=lambda t: -t[1]
    )
    total_gain = sum(g for _, g in importance) or 1.0

    out = data_dir / "processed" / region / "models" / satellite
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / "lightgbm.txt"))
    (out / "ratio.json").write_text(json.dumps(ratio, indent=2), encoding="utf-8")
    pd.DataFrame(
        {
            "cell_id": test["cell_id"].to_numpy(),
            "date": test["date"].dt.date.to_numpy(),
            "modis": test[TARGET].to_numpy(),
            **{f"p_{k}": v for k, v in preds.items()},
        }
    ).to_parquet(out / "test_predictions.parquet", index=False)

    report = {
        "region": region,
        "satellite": satellite,
        "train_years": [int(train["year"].min()), last_train],
        "test_years": [int(test["year"].min()), int(test["year"].max())],
        "train_rows": len(train),
        "test_rows": len(test),
        "lightgbm": lgb_info,
        "feature_importance_gain_share": {
            f: round(float(g) / total_gain, 4) for f, g in importance
        },
        "train_fit": {
            "ratio": evaluate(train, ratio_predict(train, ratio))["totals"],
            "lightgbm": evaluate(
                train, lightgbm_predict(booster, train, lgb_info["positive_weight"])
            )["totals"],
        },
        "test": {name: evaluate(test, p) for name, p in preds.items()},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{v:+.1%}" if pct == "signed" else (f"{v:.1%}" if pct else f"{v:.3f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("region", help="folder name under data/processed/, e.g. pantanal")
    parser.add_argument("--satellite", choices=list(SATELLITES), default="SNPP")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)
    r = run(args.region, args.data_dir, args.satellite)

    print(
        f"{r['satellite']}: train {r['train_years'][0]}-{r['train_years'][1]} "
        f"({r['train_rows']:,} rows), test {r['test_years'][0]}-{r['test_years'][1]} "
        f"({r['test_rows']:,} rows); LightGBM {r['lightgbm']['rounds']} rounds"
    )
    head = f"{'model':<10}{'bias':>9}{'month WAPE':>12}{'cell-mo WAPE':>14}{'corr':>7}"
    head += f"{'logloss':>9}{'AUC':>7}{'AP':>7}"
    print(head)
    for name, m in r["test"].items():
        t, rw = m["totals"], m["rows"]
        print(
            f"{name:<10}{_fmt(t['bias'], 'signed'):>9}{_fmt(t['monthly_wape'], True):>12}"
            f"{_fmt(t['cell_month_wape'], True):>14}{_fmt(t['monthly_corr']):>7}"
            f"{_fmt(rw['log_loss']):>9}{_fmt(rw['roc_auc']):>7}{_fmt(rw['average_precision']):>7}"
        )
    print("MODIS fire-days by test year: actual / naive / ratio / lightgbm")
    years = zip(*(r["test"][k]["by_year"] for k in ("naive", "ratio", "lightgbm")), strict=True)
    for n, ra, lg_ in years:
        print(
            f"  {n['year']}: {n['modis']:>7,} / {n['predicted']:>9,.0f} / "
            f"{ra['predicted']:>9,.0f} / {lg_['predicted']:>9,.0f}"
        )
    top = list(r["feature_importance_gain_share"].items())[:6]
    print("Top LightGBM features (gain share): " + ", ".join(f"{f} {g:.0%}" for f, g in top))


if __name__ == "__main__":
    main()
