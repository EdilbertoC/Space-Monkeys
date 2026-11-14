import pandas as pd

from app.realtime import detections

PIXEL_KM = {"VIIRS": 0.375, "MODIS": 1.0, "ABI": 3.0}


def make_detections(rows: list[dict]) -> pd.DataFrame:
    """rows: dicts with sensor, lat, lon, time (str) and optional confidence/frp_mw."""
    records = []
    for r in rows:
        size = PIXEL_KM[r["sensor"]]
        records.append(
            {
                "source": f"TEST:{r['sensor']}",
                "sensor": r["sensor"],
                "satellite": "",
                "time": pd.Timestamp(r["time"], tz="UTC"),
                "lat": r["lat"],
                "lon": r["lon"],
                "scan_km": r.get("scan_km", size),
                "track_km": r.get("track_km", size),
                "frp_mw": r.get("frp_mw", float("nan")),
                "confidence": r.get("confidence", 0.9),
                "daynight": "D",
            }
        )
    return detections.finalize(pd.DataFrame(records))
