"""
Fallback path for in-situ downloads when Copernicus Marine's ARCO
(subsettable) service is unavailable for a dataset_part: fetch the
lightweight in-situ TAC index file for that part, select only the platform
files that intersect the requested bbox/time/variables, download those
original NetCDF files, and parse them into the same long-format
(variable, platform_id, platform_type, time, longitude, latitude, depth,
value, institution) schema that copernicusmarine.subset() itself produces,
so downstream code does not need to know which path produced a CSV.
"""

from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .base import copernicus_marine_download_kwargs


@dataclass
class IndexRow:
    file_name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    time_start: datetime
    time_end: datetime
    institution: str
    parameters: set[str]


def fetch_index_file(
    dataset_id: str, dataset_part: str, work_dir: Path, force_download: bool = False,
) -> Path:
    """Download (or reuse, via skip_existing) one dataset/part's in-situ TAC
    index file and return its local path."""
    import copernicusmarine

    work_dir.mkdir(parents=True, exist_ok=True)
    result = copernicusmarine.get(
        dataset_id=dataset_id,
        dataset_part=dataset_part,
        index_parts=True,
        output_directory=str(work_dir),
        disable_progress_bar=True,
        **copernicus_marine_download_kwargs(force_download),
    )
    for f in result.files:
        if f.filename == f"index_{dataset_part}.txt":
            return Path(f.file_path)
    raise RuntimeError(f"index_{dataset_part}.txt not found among fetched files for {dataset_id}")


def _parse_index_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().rstrip("Z"))


def iter_index_rows(index_path: Path) -> Iterator[IndexRow]:
    """Yield one IndexRow per data line of an in-situ TAC index file,
    skipping the leading comment block that carries the column header."""
    with index_path.open(newline="", encoding="utf-8", errors="replace") as f:
        header_line = None
        first_data_line = None
        for line in f:
            if line.startswith("#"):
                header_line = line
            else:
                first_data_line = line
                break
        if header_line is None or first_data_line is None:
            return

        fieldnames = [c.strip() for c in header_line.lstrip("#").split(",")]
        reader = csv.reader(itertools.chain([first_data_line], f))
        for row in reader:
            if len(row) != len(fieldnames):
                continue
            fields = dict(zip(fieldnames, row))
            try:
                yield IndexRow(
                    file_name=fields["file_name"].strip(),
                    lat_min=float(fields["geospatial_lat_min"]),
                    lat_max=float(fields["geospatial_lat_max"]),
                    lon_min=float(fields["geospatial_lon_min"]),
                    lon_max=float(fields["geospatial_lon_max"]),
                    time_start=_parse_index_timestamp(fields["time_coverage_start"]),
                    time_end=_parse_index_timestamp(fields["time_coverage_end"]),
                    institution=fields.get("institution", "").strip(),
                    parameters=set(fields.get("parameters", "").split()),
                )
            except (KeyError, ValueError):
                continue
