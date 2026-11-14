# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
from datetime import UTC, datetime

import numpy as np
import pytest

from app.realtime import firms, goes

GOES_EAST = {
    "semi_major_axis": 6378137.0,
    "semi_minor_axis": 6356752.31414,
    "perspective_point_height": 35786023.0,
    "longitude_of_projection_origin": -75.0,
}

VIIRS_CSV = """latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight
-15.12345,-55.54321,330.1,0.41,0.37,2026-08-01,1730,N,VIIRS,h,2.0NRT,295.2,12.5,D
-15.20000,-55.60000,310.4,0.39,0.36,2026-08-01,0412,N20,VIIRS,n,2.0NRT,290.0,3.1,N
"""

MODIS_CSV = """latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight
-9.5,-63.1,320.0,1.2,1.1,2026-08-01,1405,Aqua,MODIS,77,6.1NRT,300.0,25.0,D
"""


def test_parse_viirs_csv():
    df = firms.parse_csv(VIIRS_CSV, "VIIRS_SNPP_NRT")
    assert len(df) == 2
    first = df.iloc[0]
    assert first["sensor"] == "VIIRS"
    assert first["time"] == datetime(2026, 8, 1, 17, 30, tzinfo=UTC)
    assert first["confidence"] == 0.9
    assert first["scan_km"] == pytest.approx(0.41)
    assert df.iloc[1]["time"].hour == 4  # "0412" keeps its leading zero


def test_parse_modis_csv():
    df = firms.parse_csv(MODIS_CSV, "MODIS_NRT")
    assert df.iloc[0]["sensor"] == "MODIS"
    assert df.iloc[0]["confidence"] == pytest.approx(0.77)


def test_firms_text_error_is_raised():
    with pytest.raises(firms.FirmsError):
        firms.parse_csv("Invalid MAP_KEY.", "VIIRS_SNPP_NRT")


def test_firms_empty_csv():
    header = VIIRS_CSV.splitlines()[0] + "\n"
    assert firms.parse_csv(header, "VIIRS_SNPP_NRT").empty


def test_goes_sub_satellite_point():
    lat, lon = goes.fixed_grid_to_latlon(np.array([0.0]), np.array([0.0]), GOES_EAST)
    assert lat[0] == pytest.approx(0.0, abs=1e-9)
    assert lon[0] == pytest.approx(-75.0, abs=1e-9)


def test_goes_off_disk_is_nan():
    lat, _ = goes.fixed_grid_to_latlon(np.array([0.2]), np.array([0.0]), GOES_EAST)
    assert np.isnan(lat[0])


def test_goes_pixel_is_about_2km_at_nadir_and_larger_over_brazil():
    dx = 5.6e-5
    scan, track = goes.pixel_footprint_km(np.array([0.0]), np.array([0.0]), dx, dx, GOES_EAST)
    assert scan[0] == pytest.approx(2.0, rel=0.03)
    assert track[0] == pytest.approx(2.0, rel=0.03)
    # Roughly over Sao Paulo (x ~ 0.04 rad, y ~ -0.065 rad).
    scan, track = goes.pixel_footprint_km(np.array([0.04]), np.array([-0.065]), dx, dx, GOES_EAST)
    assert scan[0] > 2.0 and track[0] > 2.0


def test_goes_scan_start_from_file_name():
    key = "ABI-L2-FDCF/2026/318/12/OR_ABI-L2-FDCF-M6_G19_s20263181200210_e2026_c2026.nc"
    assert goes.scan_start(key) == datetime(2026, 11, 14, 12, 0, 21, tzinfo=UTC)
    assert goes.satellite_id(key) == "G19"
