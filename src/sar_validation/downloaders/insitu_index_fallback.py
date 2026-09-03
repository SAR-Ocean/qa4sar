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

from .base import copernicus_marine_download_kwargs, split_antimeridian_bbox


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


def _row_matches_bbox(
    row: IndexRow, min_lon: float, max_lon: float, min_lat: float, max_lat: float,
) -> bool:
    """Check if a row's bbox overlaps the requested region, handling the
    antimeridian correctly."""
    if row.lat_max < min_lat or row.lat_min > max_lat:
        return False
    if row.lon_max - row.lon_min > 180:
        # The row's own reported bbox wraps the antimeridian (a drifting
        # platform crossed the dateline mid-deployment) rather than the
        # platform having genuinely visited both far sides of the globe --
        # treat it as a longitude match anywhere rather than mis-testing a
        # wrapped interval against a non-wrapped query window.
        return True
    for win_min_lon, win_max_lon in split_antimeridian_bbox(min_lon, max_lon):
        if row.lon_max >= win_min_lon and row.lon_min <= win_max_lon:
            return True
    return False


def rows_matching_query(
    index_path: Path,
    min_lon: float, max_lon: float, min_lat: float, max_lat: float,
    start: datetime, end: datetime,
    wanted_variables: set[str],
) -> list[IndexRow]:
    """Index rows whose parameters intersect wanted_variables, whose time
    span overlaps [start, end], and whose bbox overlaps the requested
    region."""
    matched = []
    for row in iter_index_rows(index_path):
        if not (row.parameters & wanted_variables):
            continue
        if row.time_end < start or row.time_start > end:
            continue
        if not _row_matches_bbox(row, min_lon, max_lon, min_lat, max_lat):
            continue
        matched.append(row)
    return matched
