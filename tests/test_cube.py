# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Cube tests. The detections are rows copied verbatim from the FIRMS archive
download for the Pantanal box: one fire on 2020-09-18 near (-15.2, -55.5) seen by
S-NPP (7 pixels), NOAA-20 and Aqua, plus one S-NPP detection from 2012."""

import pandas as pd
import pytest

pytest.importorskip("duckdb")

from pipeline import cube, ingest  # noqa: E402
from tests.firms_samples import ARCHIVE_HEADER, csv_text, write_zip  # noqa: E402

SNPP = [
    "-15.21718,-55.5285,312.43,0.52,0.5,2020-09-18,0429,SNPP,SNPP,n,2,286.35,3.1,N,0",
    "-15.2286,-55.51354,347.52,0.5,0.41,2020-09-18,1701,SNPP,SNPP,n,2,307.39,15.5,D,0",
    "-15.21792,-55.51962,338.95,0.5,0.41,2020-09-18,1701,SNPP,SNPP,l,2,312.01,6.43,D,0",
    "-15.20291,-55.52195,342.92,0.5,0.41,2020-09-18,1701,SNPP,SNPP,n,2,309.96,4.37,D,0",
    "-15.20353,-55.52657,339.42,0.5,0.41,2020-09-18,1701,SNPP,SNPP,l,2,311.41,4.37,D,0",
    "-15.20101,-55.5364,348.67,0.5,0.41,2020-09-18,1701,SNPP,SNPP,n,2,311.54,26.41,D,0",
    "-15.22798,-55.50897,334.43,0.5,0.41,2020-09-18,1701,SNPP,SNPP,l,2,305.75,15.5,D,0",
    "-15.46617,-59.75149,296.6,0.39,0.44,2012-01-20,0453,SNPP,SNPP,n,2,270.56,0.52,N,0",
]
N20 = [
    "-15.21759,-55.52882,305.05,0.4,0.44,2020-09-18,0518,N20,VIIRS,n,2,287.28,1.39,N,0",
    "-15.2171,-55.53258,310.06,0.4,0.44,2020-09-18,0518,N20,VIIRS,n,2,289.16,1.39,N,0",
    "-15.20786,-55.52306,334.79,0.33,0.55,2020-09-18,1750,N20,VIIRS,n,2,303.73,3.33,D,0",
]
AQUA = [
    "-15.2044,-55.5286,325.8,1.5,1.2,2020-09-18,1709,Aqua,MODIS,49,6.03,307.9,14.3,D,0",
    "-15.2061,-55.5424,326.9,1.5,1.2,2020-09-18,1709,Aqua,MODIS,67,6.03,306.8,15.9,D,0",
]


@pytest.fixture
def report_and_cube(tmp_path):
    raw = tmp_path / "raw" / "firms" / "test"
    raw.mkdir(parents=True)
    for product, rows in (("SV-C2", SNPP), ("J1V-C2", N20), ("M-C61", AQUA)):
        write_zip(
            raw / f"DL_FIRE_{product}_1.zip",
            {f"fire_archive_{product}_1.csv": csv_text(ARCHIVE_HEADER, rows)},
        )
    ingest.ingest("test", tmp_path)
    report = cube.build_cube("test", tmp_path)
    df = pd.read_parquet(tmp_path / "processed" / "test" / "cube")
    return report, df


def test_cell_index_edges():
    assert cube.cell_index(-15.21718, -55.5285) == (747, 1244)
    assert cube.cell_index(-15.1, -55.5) == (749, 1245)  # on an edge: the cell it names
    assert cube.cell_index(-15.10001, -55.50001) == (748, 1244)
    assert cube.cell_id(-90, -180) == 0


def test_one_row_per_cell_day_satellite(report_and_cube):
    _, df = report_and_cube
    assert list(df.columns) == cube.COLUMNS[:-1] + ["year"]
    fire = df[df["date"] == pd.Timestamp("2020-09-18").date()]
    assert sorted(fire["satellite"]) == ["Aqua", "N20", "SNPP"]
    assert (fire["cell_id"] == cube.cell_id(-15.21718, -55.5285)).all()
    assert (fire["fire_day"] == 1).all()
    assert fire["cell_lat"].iloc[0] == pytest.approx(-15.25)
    assert fire["cell_lon"].iloc[0] == pytest.approx(-55.55)


def test_features_aggregate_pixels(report_and_cube):
    _, df = report_and_cube
    by_sat = df[df["date"] == pd.Timestamp("2020-09-18").date()].set_index("satellite")

    snpp = by_sat.loc["SNPP"]  # 3 low-confidence pixels dropped by ingest
    assert snpp["n_pixels"] == 4
    assert snpp["n_overpasses"] == 2
    assert (snpp["n_day"], snpp["n_night"]) == (3, 1)
    assert snpp["frp_sum_mw"] == pytest.approx(3.1 + 15.5 + 4.37 + 26.41)
    assert snpp["frp_max_mw"] == pytest.approx(26.41)
    assert snpp["scan_mean_km"] == pytest.approx((0.52 + 0.5 * 3) / 4)

    assert by_sat.loc["N20"]["n_pixels"] == 3
    assert by_sat.loc["N20"]["n_overpasses"] == 2
    aqua = by_sat.loc["Aqua"]
    assert (aqua["sensor"], aqua["n_pixels"], aqua["n_overpasses"]) == ("MODIS", 2, 1)
    assert not by_sat["nrt"].any()


def test_report_combines_sensors_with_or(report_and_cube):
    report, _ = report_and_cube
    years = {y["year"]: y for y in report["fire_cell_days_by_year"]}
    assert years[2020] == {
        "year": 2020,
        "modis": 1,
        "viirs": 1,  # S-NPP and NOAA-20 on the same cell-day count once
        "both": 1,
        "either": 1,
        "p_modis_given_viirs": 1.0,
    }
    assert years[2012]["modis"] == 0
    assert years[2012]["p_modis_given_viirs"] == 0.0
    assert report["detections_in"] == report["detections_in_cube"] == 10


def test_observation_gaps_find_the_snpp_outage_of_july_2022(tmp_path):
    """Real rows around the S-NPP outage of 2022-07-27..08-10: NOAA-20 saw this fire
    on 2022-07-28, the S-NPP archive has no detection anywhere that day."""
    n20_outage = [
        "-15.34047,-55.04046,342.1,0.44,0.46,2022-07-28,1738,N20,VIIRS,n,2,302.6,31.64,D,0",
        "-15.337,-55.04549,367,0.44,0.46,2022-07-28,1738,N20,VIIRS,h,2,326.48,97.15,D,0",
    ]
    snpp_back = ["-15.4444,-55.7379,331.45,0.42,0.38,2022-08-11,1727,SNPP,SNPP,n,2,300.58,1.07,D,0"]
    raw = tmp_path / "raw" / "firms" / "test"
    raw.mkdir(parents=True)
    for product, rows in (("SV-C2", SNPP + snpp_back), ("J1V-C2", N20 + n20_outage)):
        write_zip(
            raw / f"DL_FIRE_{product}_1.zip",
            {f"fire_archive_{product}_1.csv": csv_text(ARCHIVE_HEADER, rows)},
        )
    ingest.ingest("test", tmp_path)
    cube.build_cube("test", tmp_path)

    outage = pd.Timestamp("2022-07-28").date()
    assert cube.observation_gaps("test", tmp_path, min_other=1) == {"SNPP": [outage]}
    # 2012-01-20 is before NOAA-20's first day, so NOAA-20 has no gap there; and one
    # fire cell-day is too little to call a gap with the default threshold.
    assert cube.observation_gaps("test", tmp_path) == {}


def test_date_runs_and_sql_dates():
    d = [pd.Timestamp(x).date() for x in ("2022-07-28", "2022-07-27", "2022-08-11")]
    assert cube.date_runs(d) == [(d[1], d[0]), (d[2], d[2])]
    assert cube.sql_dates(d[:1]) == "DATE '2022-07-28'"
    assert "DATE" in cube.sql_dates([])  # a valid, never-matching list
