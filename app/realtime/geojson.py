"""GeoJSON serialization for fused fires and raw detections."""

import math
from typing import Any

import pandas as pd


def _clean(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    return value


def feature_collection(df: pd.DataFrame, id_column: str) -> dict:
    features = []
    for row in df.to_dict(orient="records"):
        props = {k: _clean(v) for k, v in row.items() if k not in ("lat", "lon")}
        features.append(
            {
                "type": "Feature",
                "id": row[id_column],
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(float(row["lon"]), 5), round(float(row["lat"]), 5)],
                },
                "properties": props,
            }
        )
    return {"type": "FeatureCollection", "features": features}


def filter_bbox(df: pd.DataFrame, bbox: tuple[float, float, float, float] | None) -> pd.DataFrame:
    if bbox is None or df.empty:
        return df
    w, s, e, n = bbox
    return df[df["lon"].between(w, e) & df["lat"].between(s, n)]
