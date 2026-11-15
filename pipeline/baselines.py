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
  (p / (p + (1 - p) * w)). That alone left the national model 5% low even on its
  training years, so the probabilities are then calibrated: the last training
  year is held out, used for early stopping and to fit an isotonic regression
  from the model's probability to the observed MODIS rate (which makes the
  held-out year's total match exactly). The model is not refit afterwards, so the
  calibration never sees data the trees were trained on.

Metrics (test years only):

- rows: log loss, Brier score, ROC AUC, average precision
- totals, what the calendar shows: sum of p against the MODIS fire-days, per
  month for the region (WAPE, bias, correlation) and per cell and month (WAPE)
- by year: predicted against actual MODIS fire-days (watch for drift)

Totals cover MODIS fire-days inside the rows only; the coverage gap reported by
pipeline.training is corrected later (step 8).

Output: data/processed/<region>/models/<SAT>/ (report.json, lightgbm.txt,
calibration.json, ratio.json, test_predictions.parquet)

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
from sklearn.isotonic import IsotonicRegression
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


def fit_ratio(train: pd.DataFrame, weight: str | None = None) -> dict:
    """Weighted share of MODIS fire per (month, own detection); weight = row weights."""
    w = train[weight] if weight else pd.Series(1.0, index=train.index)
    key = _ratio_key(train)
    num = (train[TARGET] * w).groupby(key).sum()
    den = w.groupby(key).sum()
    table = (num / den).to_dict()
    return {"by_month_and_own": table, "overall": float((train[TARGET] * w).sum() / w.sum())}


def ratio_predict(df: pd.DataFrame, model: dict) -> np.ndarray:
    table = model["by_month_and_own"]
    return _ratio_key(df).map(table).fillna(model["overall"]).to_numpy(dtype=float)


def positive_weight(y: pd.Series | np.ndarray, w: np.ndarray | None = None) -> float:
    """Weighted negatives over weighted positives (row weights w default to 1)."""
    y = np.asarray(y)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    pos = float(w[y == 1].sum())
    return float(w[y == 0].sum() / pos) if pos else 1.0


def unweight(p: np.ndarray, w: float) -> np.ndarray:
    """Undo a positive-class weight w: the weighted model's odds are w times too high."""
    p = np.asarray(p, dtype=float)
    return p / (p + (1.0 - p) * w)


def fit_calibration(p: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> dict:
    """Isotonic map from predicted probability to observed rate, as interpolation
    points. Isotonic regression keeps the weighted total: sum(w * p_cal) = sum(w * y)."""
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p, y, sample_weight=w)
    return {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}


def apply_calibration(p: np.ndarray, calibrator: dict | None) -> np.ndarray:
    if not calibrator:
        return np.asarray(p, dtype=float)
    return np.interp(p, calibrator["x"], calibrator["y"])


def fit_lightgbm(
    train: pd.DataFrame, valid_year: int, weight: str | None = None
) -> tuple[lgb.Booster, dict]:
    """Fit with the positive class weighted, then calibrate on the held-out year.

    `weight` names a row-weight column (e.g. for down-sampled rows), used in the
    class weight, in training and in the calibration. info["calibrator"] is None
    when the held-out year cannot be used (too few rows or no fire), and the model
    is then trained on every year without calibration.
    """
    rw = train[weight].to_numpy() if weight else None
    w = positive_weight(train[TARGET], rw)
    params = {**LGB_PARAMS, "scale_pos_weight": w}

    def ds(part: pd.DataFrame) -> lgb.Dataset:
        return lgb.Dataset(part[FEATURES], part[TARGET], weight=part[weight] if weight else None)

    fit_part, valid = train[train["year"] < valid_year], train[train["year"] == valid_year]
    info = {"positive_weight": w, "valid_year": valid_year, "calibrator": None}
    if not (len(fit_part) and len(valid) and valid[TARGET].nunique() == 2):
        booster = lgb.train(params, ds(train), MAX_ROUNDS)
        return booster, {**info, "rounds": MAX_ROUNDS}

    booster = lgb.train(
        params,
        ds(fit_part),
        num_boost_round=MAX_ROUNDS,
        valid_sets=[ds(valid)],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )
    p_valid = unweight(booster.predict(valid[FEATURES]), w)
    vw = valid[weight].to_numpy() if weight else np.ones(len(valid))
    y_valid = valid[TARGET].to_numpy()
    total = float((vw * y_valid).sum())
    info.update(
        rounds=booster.best_iteration or MAX_ROUNDS,
        calibrator=fit_calibration(p_valid, y_valid, vw),
        valid_bias_before_calibration=round(float((vw * p_valid).sum()) / total - 1, 4),
    )
    return booster, info


def lightgbm_predict(
    booster: lgb.Booster, df: pd.DataFrame, w: float, calibrator: dict | None = None
) -> np.ndarray:
    return apply_calibration(unweight(booster.predict(df[FEATURES]), w), calibrator)


def save_lightgbm(out: Path, booster: lgb.Booster, info: dict) -> None:
    """lightgbm.txt (best iteration) and calibration.json (class weight + calibrator)."""
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / "lightgbm.txt"))
    calibration = {"positive_weight": info["positive_weight"], "calibrator": info["calibrator"]}
    (out / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")


def load_lightgbm(model_dir: Path) -> tuple[lgb.Booster, float, dict | None]:
    if not (model_dir / "lightgbm.txt").exists():
        raise FileNotFoundError(f"{model_dir} has no model; run pipeline.baselines first")
    cal = json.loads((model_dir / "calibration.json").read_text(encoding="utf-8"))
    booster = lgb.Booster(model_file=str(model_dir / "lightgbm.txt"))
    return booster, cal["positive_weight"], cal["calibrator"]


# ---- metrics ----------------------------------------------------------------
def wape(pred: pd.Series, actual: pd.Series) -> float | None:
    total = float(actual.sum())
    return float((pred - actual).abs().sum() / total) if total else None


def evaluate(df: pd.DataFrame, p: np.ndarray) -> dict:
    return evaluate_arrays(
        df[TARGET].to_numpy(),
        np.asarray(p),
        df["cell_id"].to_numpy(),
        df["year"].to_numpy(),
        df["date"].dt.month.to_numpy(),
    )


def evaluate_arrays(
    y: np.ndarray, p: np.ndarray, cell_id: np.ndarray, year: np.ndarray, month: np.ndarray
) -> dict:
    pc = np.clip(p, 1e-4, 1 - 1e-4)
    rows = {
        "log_loss": float(log_loss(y, pc, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, p)) if y.sum() else None,
    }
    frame = pd.DataFrame({"cell_id": cell_id, "year": year, "month": month, "y": y, "p": p})
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
    w, cal = lgb_info["positive_weight"], lgb_info["calibrator"]
    preds = {
        "naive": naive_predict(test),
        "ratio": ratio_predict(test, ratio),
        "lightgbm": lightgbm_predict(booster, test, w, cal),
    }
    importance = sorted(
        zip(FEATURES, booster.feature_importance("gain"), strict=True), key=lambda t: -t[1]
    )
    total_gain = sum(g for _, g in importance) or 1.0

    out = data_dir / "processed" / region / "models" / satellite
    save_lightgbm(out, booster, lgb_info)
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
        "lightgbm": {k: v for k, v in lgb_info.items() if k != "calibrator"},
        "feature_importance_gain_share": {
            f: round(float(g) / total_gain, 4) for f, g in importance
        },
        "train_fit": {
            "ratio": evaluate(train, ratio_predict(train, ratio))["totals"],
            "lightgbm": evaluate(train, lightgbm_predict(booster, train, w, cal))["totals"],
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
