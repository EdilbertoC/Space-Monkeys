# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Download the NASA FIRMS yearly country archive for one country, ready for
pipeline.ingest.

FIRMS publishes one standard-processing CSV per sensor, year and country at
https://firms.modaps.eosdis.nasa.gov/data/country/<sensor>/<year>/<sensor>_<year>_<Country>.csv
(no request queue). Each file is saved as data/raw/firms/<region>/csv/
fire_archive_<PRODUCT>_<year>.csv, the name pipeline.ingest reads; its content is
not changed. Files already there with the same size are skipped. Years the archive
does not have (404) are reported and skipped.

Usage::

    python scripts/download_firms_country.py Australia australia
"""

from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE = "https://firms.modaps.eosdis.nasa.gov/data/country"
# Country-archive folder -> (product code read by pipeline.ingest, first year).
SENSORS = {
    "modis": ("M-C61", 2000),
    "viirs-snpp": ("SV-C2", 2012),
    "viirs-jpss1": ("J1V-C2", 2018),
}
HEADERS = {"User-Agent": "space-monkeys-pipeline"}


def download(country: str, region: str, data_dir: Path, last_year: int) -> list[str]:
    out_dir = data_dir / "raw" / "firms" / region / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for folder, (product, first) in SENSORS.items():
        for year in range(first, last_year + 1):
            url = f"{BASE}/{folder}/{year}/{folder}_{year}_{country}.csv"
            dest = out_dir / f"fire_archive_{product}_{year}.csv"
            try:
                head = urllib.request.urlopen(
                    urllib.request.Request(url, headers=HEADERS, method="HEAD"), timeout=60
                )
                size = int(head.headers.get("Content-Length", -1))
                if dest.exists() and dest.stat().st_size == size:
                    continue
                with (
                    urllib.request.urlopen(
                        urllib.request.Request(url, headers=HEADERS), timeout=600
                    ) as src,
                    open(dest.with_suffix(".part"), "wb") as out,
                ):
                    while chunk := src.read(1 << 20):
                        out.write(chunk)
                dest.with_suffix(".part").replace(dest)
                print(f"{folder} {year}: {dest.stat().st_size / 1e6:,.0f} MB")
            except urllib.error.HTTPError as err:
                if err.code != 404:
                    raise
                missing.append(f"{folder} {year}")
    return missing


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("country", help="country name as FIRMS writes it, e.g. Australia")
    parser.add_argument("region", help="folder name under data/, e.g. australia")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--last-year", type=int, default=date.today().year)
    args = parser.parse_args(argv)
    missing = download(args.country, args.region, args.data_dir, args.last_year)
    if missing:
        print("not in the archive:", ", ".join(missing))


if __name__ == "__main__":
    main()
