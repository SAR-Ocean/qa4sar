"""
Fallback path for in-situ downloads when Copernicus Marine's ARCO
(subsettable) service is unavailable for a dataset_part: fetch the
lightweight in-situ TAC index file for that part, select only the platform
files that intersect the requested bbox/time/variables, download those
original NetCDF files, and parse them into the same long-format
(variable, platform_id, platform_type, time, longitude, latitude, depth,
value, value_qc, institution) schema that copernicusmarine.subset() itself
produces, so downstream code does not need to know which path produced a
CSV -- including from_insitu_csv's QC-code filtering, which only applies
when a value_qc column is present.
"""

from __future__ import annotations

import csv
import itertools
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import xarray as xr

from .base import copernicus_marine_download_kwargs, normalize_datetime, split_antimeridian_bbox

logger = logging.getLogger(__name__)


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


#: How long a locally cached index file is trusted before being re-fetched.
#: Once work_dir is a persistent, cross-run shared cache (rather than a
#: fresh per-run scratch dir), copernicusmarine's own skip_existing check
#: is a bare "does the file exist" test with no staleness awareness -- a
#: cached index would otherwise never be refreshed again, so a platform
#: newly added or newly extended into a query's time window would
#: silently stop matching with no error. The index's own "Date of update"
#: header changes roughly daily in practice, so a day-old copy is stale
#: enough to matter.
_INDEX_MAX_AGE = timedelta(days=1)


def fetch_index_file(
    dataset_id: str, dataset_part: str, work_dir: Path, force_download: bool = False,
) -> Path:
    """Download (or reuse, while younger than _INDEX_MAX_AGE) one
    dataset/part's in-situ TAC index file and return its local path.

    copernicusmarine.get(index_parts=True) always fetches all of a
    dataset's index files together (confirmed live: its own filter/regex
    parameters have no effect in this mode) -- a real bandwidth cost
    (several hundred MB combined) but one this function limits to roughly
    once per day per dataset, not once per run, via the staleness check
    above. no_directories=True keeps the files directly under work_dir
    (confirmed live to still work under index_parts=True) rather than
    copernicusmarine's default nested product-id/version subdirectories,
    which the staleness check below relies on to find the cached copy at
    all.
    """
    import copernicusmarine

    cached_path = work_dir / f"index_{dataset_part}.txt"
    if not force_download and cached_path.exists():
        age = datetime.now() - datetime.fromtimestamp(cached_path.stat().st_mtime)
        if age < _INDEX_MAX_AGE:
            return cached_path

    work_dir.mkdir(parents=True, exist_ok=True)
    result = copernicusmarine.get(
        dataset_id=dataset_id,
        dataset_part=dataset_part,
        index_parts=True,
        no_directories=True,
        output_directory=str(work_dir),
        disable_progress_bar=True,
        overwrite=True,
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
                logger.debug(
                    "Skipping index row in %s: expected %d fields, got %d.",
                    index_path, len(fieldnames), len(row),
                )
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
            except (KeyError, ValueError) as exc:
                logger.debug(
                    "Skipping unparseable index row for %r in %s: %s.",
                    fields.get("file_name"), index_path, exc,
                )
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


def _observations_overlap_window(
    times: "np.ndarray", lons: "np.ndarray", lats: "np.ndarray",
    min_lon: float, max_lon: float, min_lat: float, max_lat: float,
    start: pd.Timestamp, end: pd.Timestamp,
) -> bool:
    """True if at least one (time, lon, lat) observation falls inside the
    requested window. NaN coordinates/timestamps never match. Antimeridian
    wrap (min_lon > max_lon) uses the same split_antimeridian_bbox
    convention as the rest of this module."""
    times = pd.to_datetime(times)
    valid = ~(pd.isna(times) | np.isnan(lons) | np.isnan(lats))
    if not valid.any():
        return False

    time_ok = (times >= start) & (times <= end) & valid

    lon_ok = np.zeros_like(lons, dtype=bool)
    for win_min_lon, win_max_lon in split_antimeridian_bbox(min_lon, max_lon):
        lon_ok |= (lons >= win_min_lon) & (lons <= win_max_lon)
    lat_ok = (lats >= min_lat) & (lats <= max_lat)

    return bool(np.any(time_ok & lon_ok & lat_ok))


#: A row whose own reported bbox spans less than this in both dimensions is
#: treated as effectively stationary (an anchored mooring's index bbox is
#: GPS-jitter-sized, well under this) and skips the overlap pre-check
#: entirely -- rows_matching_query's own bbox/time match is already precise
#: enough for a platform that does not move. A row spanning more than this
#: (a genuinely moving platform, or a wide-coverage fixed network such as
#: HF-radar) is exactly the case the pre-check exists for.
_STATIONARY_BBOX_DEGREES = 0.1


def _row_is_effectively_stationary(row: IndexRow, threshold_deg: float = _STATIONARY_BBOX_DEGREES) -> bool:
    """True if *row*'s own reported bbox is tight enough that its
    coordinates can be trusted as-is, without checking the platform's
    actual per-observation track."""
    return (
        row.lat_max - row.lat_min < threshold_deg
        and row.lon_max - row.lon_min < threshold_deg
    )


_LON_NAMES = ("PRECISE_LONGITUDE", "LONGITUDE")
_LAT_NAMES = ("PRECISE_LATITUDE", "LATITUDE")
_DEPTH_NAMES = ("DEPH", "PRES")


def _first_present(ds: xr.Dataset, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in ds.variables:
            return name
    return None


def _get_remote_url(
    dataset_id: str, dataset_part: str, file_name: str, work_dir: Path,
) -> str | None:
    """Return the direct HTTPS URL of one index-listed file without
    downloading it (copernicusmarine.get(..., dry_run=True)), or None if
    the file cannot be located."""
    import copernicusmarine

    work_dir.mkdir(parents=True, exist_ok=True)
    file_list_path = work_dir / f"_file_list_{uuid.uuid4().hex}.txt"
    file_list_path.write_text(file_name + "\n")
    try:
        result = copernicusmarine.get(
            dataset_id=dataset_id,
            dataset_part=dataset_part,
            file_list=str(file_list_path),
            dry_run=True,
            disable_progress_bar=True,
        )
    finally:
        file_list_path.unlink(missing_ok=True)
    return result.files[0].https_url if result.files else None


def row_overlaps_window(
    row: IndexRow,
    dataset_id: str, dataset_part: str, work_dir: Path,
    min_lon: float, max_lon: float, min_lat: float, max_lat: float,
    start: pd.Timestamp, end: pd.Timestamp,
) -> bool:
    """True if *row*'s platform has at least one real observation inside
    the requested window. A row whose own reported bbox is effectively
    stationary (see _row_is_effectively_stationary) skips the check
    entirely -- rows_matching_query's own bbox/time match already is the
    answer for a platform that does not move. Otherwise checks the local
    cache (*work_dir*) first, falling back to a lazy remote open of the
    file's own coordinate variables via its direct HTTPS URL -- never
    downloading the file just to answer this question. Any failure to
    complete the check (network, missing coordinates, unreadable file)
    fails open: returns True, since this is an optimization that must
    never cause data loss."""
    if _row_is_effectively_stationary(row):
        return True

    local_path = work_dir / Path(row.file_name).name
    try:
        if local_path.exists():
            source: "Path | str" = local_path
        else:
            url = _get_remote_url(dataset_id, dataset_part, row.file_name, work_dir)
            if url is None:
                return True
            source = url

        with xr.open_dataset(source, engine="h5netcdf") as ds:
            lon_name = _first_present(ds, _LON_NAMES)
            lat_name = _first_present(ds, _LAT_NAMES)
            if lon_name is None or lat_name is None or "TIME" not in ds.variables:
                return True
            return _observations_overlap_window(
                ds["TIME"].values, ds[lon_name].values, ds[lat_name].values,
                min_lon, max_lon, min_lat, max_lat, start, end,
            )
    except Exception:
        return True


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


def download_index_files(
    dataset_id: str, dataset_part: str, rows: list[IndexRow], work_dir: Path,
    force_download: bool = False,
) -> list[Path]:
    """Download exactly the matched platform files (not the whole dataset)
    via copernicusmarine's file_list option, and return their local
    paths."""
    if not rows:
        return []

    import copernicusmarine

    print(f"  Downloading {len(rows)} matched platform file(s) via in-situ index …")

    work_dir.mkdir(parents=True, exist_ok=True)
    file_list_path = work_dir / f"_file_list_{uuid.uuid4().hex}.txt"
    file_list_path.write_text("\n".join(row.file_name for row in rows) + "\n")

    try:
        result = copernicusmarine.get(
            dataset_id=dataset_id,
            dataset_part=dataset_part,
            file_list=str(file_list_path),
            output_directory=str(work_dir),
            no_directories=True,
            # Unlike the index/dry-run calls elsewhere in this module,
            # these are whole-archive platform files that can individually
            # run to hundreds of MB or more -- a real download bar here
            # (rather than silence for however long that takes) is the
            # difference between "working" and "looks hung".
            disable_progress_bar=False,
            **copernicus_marine_download_kwargs(force_download),
        )
    finally:
        file_list_path.unlink(missing_ok=True)

    if result.files_not_found:
        # A non-empty files_not_found makes copernicusmarine fall back to
        # listing every file on the remote server to find a match -- a real
        # cost, so this is worth surfacing even though a missing platform
        # file is not itself an error here.
        logger.debug(
            "%d requested file(s) not found on the remote server: %s",
            len(result.files_not_found), result.files_not_found,
        )
    if result.total_size is not None:
        print(f"  Downloaded {len(result.files)} file(s), {result.total_size:.1f} MB total.")
    return [Path(f.file_path) for f in result.files]


def parse_platform_file(
    nc_path: Path,
    wanted_variables: set[str],
    min_lon: float, max_lon: float, min_lat: float, max_lat: float,
    start: pd.Timestamp, end: pd.Timestamp,
    min_depth: float, max_depth: float,
    platform_type_code: str,
) -> pd.DataFrame:
    """Read one in-situ TAC NetCDF file and return a long-format dataframe
    (variable, platform_id, platform_type, time, longitude, latitude,
    depth, value, value_qc, institution), trimmed to the requested
    bbox/time/depth window - matching the schema copernicusmarine.subset()
    itself produces, so downstream code does not need to know which path
    produced a given CSV. A variable whose own "<code>_QC" companion is
    absent from the file gets an all-NaN value_qc, which from_insitu_csv's
    QC filter already treats as unusable -- the same "no QC code, no
    value" policy applied to every other in-situ source, not a fallback
    QC-blindspot.

    min_lon/max_lon are compared directly (min_lon <= longitude <=
    max_lon), unlike every other bbox check in this module: the caller is
    expected to have already split an antimeridian-crossing query into
    non-wrapping windows (as InSituDownloader.download() does via
    split_antimeridian_bbox) before reaching a single platform file."""
    empty_columns = [
        "variable", "platform_id", "platform_type", "time",
        "longitude", "latitude", "depth", "value", "value_qc", "institution",
    ]
    with xr.open_dataset(nc_path) as ds:
        lon_name = _first_present(ds, _LON_NAMES)
        lat_name = _first_present(ds, _LAT_NAMES)
        depth_name = _first_present(ds, _DEPTH_NAMES)
        present_vars = [v for v in wanted_variables if v in ds.data_vars]
        if lon_name is None or lat_name is None or "TIME" not in ds.variables or not present_vars:
            return pd.DataFrame(columns=empty_columns)

        platform_id = str(ds.attrs.get("platform_code", nc_path.stem))
        institution = str(ds.attrs.get("institution", ""))

        qc_names = {v: f"{v}_QC" for v in present_vars if f"{v}_QC" in ds.data_vars}
        keep = {"TIME", lon_name, lat_name, *present_vars, *qc_names.values()}
        if depth_name is not None:
            keep.add(depth_name)
        df = ds[sorted(keep)].to_dataframe().reset_index()

    rename = {"TIME": "time", lon_name: "longitude", lat_name: "latitude"}
    if depth_name is not None:
        rename[depth_name] = "depth"
    df = df.rename(columns=rename)
    if "depth" not in df.columns:
        df["depth"] = 0.0

    df = df[
        (df["time"] >= start) & (df["time"] <= end)
        & (df["longitude"] >= min_lon) & (df["longitude"] <= max_lon)
        & (df["latitude"] >= min_lat) & (df["latitude"] <= max_lat)
        & (df["depth"] >= min_depth) & (df["depth"] <= max_depth)
    ]
    if df.empty:
        return pd.DataFrame(columns=empty_columns)

    df["platform_id"] = platform_id
    df["platform_type"] = platform_type_code
    df["institution"] = institution

    id_cols = ["platform_id", "platform_type", "time", "longitude", "latitude", "depth", "institution"]
    # pd.melt only carries a single value column, so each variable's QC
    # companion (when present) is stacked alongside it per-variable rather
    # than via one combined melt.
    parts = []
    for var in present_vars:
        part = df[id_cols].copy()
        part["variable"] = var
        part["value"] = df[var]
        part["value_qc"] = df[qc_names[var]] if var in qc_names else np.nan
        parts.append(part)
    long_df = pd.concat(parts, ignore_index=True)
    return long_df.dropna(subset=["value"])[empty_columns].reset_index(drop=True)


def download_via_index(
    dataset_id: str,
    dataset_part: str,
    min_lon: float, max_lon: float, min_lat: float, max_lat: float,
    start_dt: str, end_dt: str,
    min_depth: float, max_depth: float,
    wanted_variables: set[str],
    dest_path: Path,
    work_dir: Path,
    force_download: bool = False,
    platform_codes: "set[str] | None" = None,
) -> Path | None:
    """Fetch the in-situ TAC index for one dataset/part, select the
    platform files intersecting the requested bbox/time/variables,
    download and parse just those, and write the combined long-format CSV
    to *dest_path*. Returns None (no CSV written) when no platform file
    matches the query, matching the "no data" convention already used by
    the ARCO/subset() download path.

    *platform_codes*, when given, narrows the matched rows to just those
    platform-type directories (e.g. {"MO"} for moorings only) before any
    file is downloaded -- unlike a post-hoc filter on the finished CSV,
    this actually avoids fetching an irrelevant platform's whole-archive
    file in the first place.
    """
    start = pd.Timestamp(normalize_datetime(start_dt))
    end = pd.Timestamp(normalize_datetime(end_dt))

    index_path = fetch_index_file(dataset_id, dataset_part, work_dir, force_download)
    rows = rows_matching_query(
        index_path, min_lon, max_lon, min_lat, max_lat, start.to_pydatetime(),
        end.to_pydatetime(), wanted_variables,
    )
    if platform_codes:
        rows = [row for row in rows if Path(row.file_name).parent.name in platform_codes]
    if not rows:
        return None

    matched_count = len(rows)
    rows = [
        row for row in rows
        if row_overlaps_window(
            row, dataset_id, dataset_part, work_dir,
            min_lon, max_lon, min_lat, max_lat, start, end,
        )
    ]
    skipped_count = matched_count - len(rows)
    if skipped_count:
        print(
            f"  In-situ index: {matched_count} platform(s) matched by bbox/time/"
            f"variable, {skipped_count} skipped (track does not overlap the "
            f"query window) -- {len(rows)} remaining to download."
        )
    if not rows:
        return None

    nc_paths = download_index_files(dataset_id, dataset_part, rows, work_dir, force_download)
    row_by_stem: "dict[str, IndexRow]" = {}
    for matched_row in rows:
        stem = Path(matched_row.file_name).stem
        if stem in row_by_stem:
            logger.warning(
                "Duplicate platform file stem %r in index rows; keeping %s over %s.",
                stem, matched_row.file_name, row_by_stem[stem].file_name,
            )
        row_by_stem[stem] = matched_row

    frames = []
    for nc_path in nc_paths:
        row = row_by_stem.get(nc_path.stem)
        platform_type_code = Path(row.file_name).parent.name if row is not None else "unknown"
        df = parse_platform_file(
            nc_path, wanted_variables,
            min_lon, max_lon, min_lat, max_lat, start, end,
            min_depth, max_depth, platform_type_code,
        )
        if not df.empty:
            frames.append(df)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(dest_path, index=False)
    return dest_path
