# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Helpers to rebuild small FIRMS archive zips from rows copied verbatim from the
real FIRMS downloads. Tests never invent detection values."""

import zipfile
from pathlib import Path

ARCHIVE_HEADER = (
    "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,"
    "confidence,version,bright_t31,frp,daynight,type"
)
NRT_HEADER = ARCHIVE_HEADER.rsplit(",", 1)[0]


def write_zip(path: Path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in members.items():
            zf.writestr(name, text)


def csv_text(header: str, rows: list[str]) -> str:
    return "\n".join([header, *rows]) + "\n"
