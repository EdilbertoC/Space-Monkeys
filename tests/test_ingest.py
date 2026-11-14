# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Ingestion tests. Every row below is copied verbatim from the FIRMS archive
download for the Pantanal box (requests 848583 and 848586); nothing is made up."""

import zipfile

import pandas as pd
import pytest

pytest.importorskip("duckdb")

from app.realtime import detections  # noqa: E402
from pipeline import ingest  # noqa: E402

ARCHIVE_HEADER = (
    "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,"
    "confidence,version,bright_t31,frp,daynight,type"
)
NRT_HEADER = ARCHIVE_HEADER.rsplit(",", 1)[0]

MODIS_ARCHIVE = [
    "-21.0319,-55.577,304.4,1,1,2000-11-01,0234,Terra,MODIS,60,6.03,293.7,5.1,N,0",
    "-21.536,-58.403,315.8,4.3,1.9,2000-11-01,1506,Terra,MODIS,29,6.03,294,59,D,0",  # low
    "-19.6246,-57.4627,339.5,1,1,2001-04-18,1414,Terra,MODIS,78,6.03,305,35.1,D,2",  # MODIS type 2
]
SNPP_ARCHIVE = [
    "-15.46617,-59.75149,296.6,0.39,0.44,2012-01-20,0453,SNPP,SNPP,n,2,270.56,0.52,N,0",
    "-19.4999,-56.71805,367,0.38,0.36,2012-01-20,1724,SNPP,SNPP,h,2,304.66,16.11,D,0",
    "-22.29536,-56.1136,344.73,0.38,0.36,2012-01-20,1723,SNPP,SNPP,l,2,315.98,25.25,D,0",  # low
    "-20.26743,-58.14076,367,0.35,0.56,2012-02-12,1652,SNPP,SNPP,h,2,297.79,15.66,D,3",  # wetland
    "-19.02484,-57.61978,306.63,0.44,0.38,2016-12-15,0514,SNPP,SNPP,n,2,287.72,1.56,N,2",  # static
    "-19.02375,-57.62327,303.95,0.47,0.4,2014-04-26,0454,SNPP,SNPP,n,2,292.22,0.62,N,0",  # near it
]
SNPP_NRT = [
    "-17.40159,-59.57053,303.42,0.42,0.45,2026-08-01,0459,SNPP,SNPP,n,2.0NRT,292.53,0.96,N",
    "-18.03111,-59.62153,309.13,0.41,0.45,2026-08-01,0459,SNPP,SNPP,n,2.0NRT,292.47,0.93,N",
]


def _write_zip(path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in members.items():
            zf.writestr(name, text)


@pytest.fixture
def data_dir(tmp_path):
    raw = tmp_path / "raw" / "firms" / "test"
    raw.mkdir(parents=True)
    _write_zip(
        raw / "DL_FIRE_M-C61_848583.zip",
        {
            "fire_archive_M-C61_848583.csv": "\n".join([ARCHIVE_HEADER, *MODIS_ARCHIVE]) + "\n",
            "Readme.txt": "not a csv",
        },
    )
    _write_zip(
        raw / "DL_FIRE_SV-C2_848586.zip",
        {
            "fire_archive_SV-C2_848586.csv": "\n".join([ARCHIVE_HEADER, *SNPP_ARCHIVE]) + "\n",
            "fire_nrt_SV-C2_848586.csv": "\n".join([NRT_HEADER, *SNPP_NRT]) + "\n",
            "../outside.csv": ARCHIVE_HEADER + "\n",
        },
    )
    return tmp_path


def _read(data_dir) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "processed" / "test" / "detections")


def test_columns_extend_the_common_schema():
    assert ingest.COLUMNS[: len(detections.COLUMNS)] == detections.COLUMNS


def test_ingest_keeps_and_drops_the_right_rows(data_dir):
    report = ingest.ingest("test", data_dir)
    df = _read(data_dir)

    kept = set(df["lat"].round(5))
    assert kept == {-21.0319, -19.6246, -15.46617, -19.4999, -20.26743, -17.40159, -18.03111}
    assert report.totals["rows_read"] == 11
    assert report.totals["dropped_low_confidence"] == 2
    assert report.totals["dropped_static"] == 2  # the VIIRS site and a fire 400 m from it
    assert report.totals["kept_firms_type_2"] == 1  # scattered MODIS label, kept
    assert report.totals["rows_written"] == 7


def test_ingest_normalizes_fields(data_dir):
    ingest.ingest("test", data_dir)
    df = _read(data_dir).set_index(_read(data_dir)["lat"].round(5))

    snpp = df.loc[-15.46617]
    assert snpp["sensor"] == "VIIRS"  # FIRMS writes "SNPP" in the instrument column
    assert snpp["source"] == "FIRMS:VIIRS_SNPP_SP"
    assert snpp["confidence_class"] == "n"
    assert snpp["confidence"] == pytest.approx(0.6)

    modis = df.loc[-21.0319]
    assert modis["sensor"] == "MODIS"
    assert modis["confidence"] == pytest.approx(0.6)
    assert modis["time"] == pd.Timestamp("2000-11-01 02:34", tz="UTC")

    nrt = df.loc[-17.40159]
    assert nrt["processing"] == "NRT"
    assert nrt["source"] == "FIRMS:VIIRS_SNPP_NRT"
    assert pd.isna(nrt["fire_type"])
    assert nrt["time"] == pd.Timestamp("2026-08-01 04:59", tz="UTC")  # "0459" keeps its zero
    assert df.loc[-20.26743]["fire_type"] == 3


def test_min_confidence_class_l_keeps_low_rows(data_dir):
    report = ingest.ingest("test", data_dir, min_confidence_class="l")
    assert report.totals["dropped_low_confidence"] == 0
    assert report.totals["rows_written"] == 9
    assert {-21.536, -22.29536} <= set(_read(data_dir)["lat"].round(5))


def test_only_firms_csvs_are_extracted(data_dir):
    ingest.ingest("test", data_dir)
    staged = {p.name for p in (data_dir / "raw" / "firms" / "test" / "csv").iterdir()}
    assert staged == {
        "fire_archive_M-C61_848583.csv",
        "fire_archive_SV-C2_848586.csv",
        "fire_nrt_SV-C2_848586.csv",
    }
    assert not (data_dir / "raw" / "firms" / "outside.csv").exists()


def test_missing_region_raises(tmp_path):
    (tmp_path / "raw" / "firms" / "empty").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        ingest.ingest("empty", tmp_path)
