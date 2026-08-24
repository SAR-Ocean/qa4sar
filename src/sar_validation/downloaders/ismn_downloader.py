"""
Select ISMN (International Soil Moisture Network) in-situ soil-moisture
stations from a manually-downloaded local archive.

ISMN has no download API — data must be obtained via the registration-gated
web portal (https://ismn.earth/en/dataviewer/), which itself
supports filtering by geographic rectangle, date range, sensor depth, and
variable before download. This module is a **local-archive selector**, not
a network downloader: given a path to that manually-downloaded archive (zip
or extracted folder), it filters to stations/sensors inside the recipe's
bbox/depth window and writes one CSV per surviving sensor, in the same
long-format schema the Copernicus Marine in-situ CSVs use for compatibility
with ``DataTreeConverter.from_insitu_csv``.

No ``archive_path`` needs to be configured for the common case: if it is
absent (or stale), ``download()`` auto-detects the most-recently-modified
``*.zip`` sitting directly in ``output_dir`` — just drop the downloaded
zip into this run's own ISMN folder and re-run. An explicit
``archive_path`` still takes priority when given, for reusing one archive
across multiple recipes.

Library usage::

    from sar_validation.downloaders.ismn_downloader import ISMNDownloader
    dl = ISMNDownloader(output_dir=Path("data/run1/ismn"))
    dl.download(
        min_lon=-10, max_lon=20, min_lat=40, max_lat=55,
        start="2026-01-01", end="2026-01-02",
        min_depth=0.0, max_depth=0.05,
        archive_path="/path/to/ismn_export.zip",  # optional -- see above
    )
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["ISMNDownloader"]

#: Shared, manually-refreshed complete ISMN export used as a fallback when
#: a recipe has not downloaded its own region/date-specific archive (yet).
#: It needs periodic manual refreshing from https://ismn.earth/en/dataviewer/
#: (90-day staleness warning in download()).
_SHARED_ARCHIVE_CACHE_DIR = Path("data") / "_archive_cache" / "ismn"

#: Age threshold (days) past which download() logs a refresh reminder
#: when it falls back to the shared archive cache.
_SHARED_ARCHIVE_STALE_DAYS = 90

#: Cache directory for the per-archive station-coordinate index (see
#: _build_station_index).
_STATION_INDEX_CACHE_DIR = Path("data") / "_archive_cache" / "ismn"

#: Zip member names are always copied into an extracted subset alongside
#: whatever station directories match a bbox filter. There are small
#: reference files (ISMN_sensor_list.csv, ISMN_network_flags_descriptions.csv),
#: which are not required by every ismn package version, but copied 
#: unconditionally to avoid the reader expecting them to exist.
_GLOBAL_REFERENCE_FILENAMES = ("ISMN_sensor_list.csv", "ISMN_network_flags_descriptions.csv")

#: Marker file written inside an extracted-subset directory once
#: extraction genuinely finishes -- see _extract_matching_stations and
#: the reuse check in ISMNDownloader.download().
_EXTRACTION_COMPLETE_MARKER = ".extraction_complete"


def _archive_age_days(path: Path) -> float:
    """
    Age of *path* in days, preferring a trailing ``_YYYYMMDD`` date
    embedded in the filename (e.g. ``ISMN_archive_20260724.zip``, the
    convention this project's shared-cache archives have settled on) over
    the file's own mtime.

    mtime resets on any copy/checkout/sync of the shared cache directory,
    so it is an unreliable proxy for how stale the underlying ISMN data
    actually is whenever a name-embedded date is available. Falls back to
    mtime for any filename without a recognizable trailing date.
    """
    match = re.search(r"_(\d{8})$", path.stem)
    if match:
        try:
            archive_date = datetime.strptime(match.group(1), "%Y%m%d")
            return (datetime.now() - archive_date).total_seconds() / 86400
        except ValueError:
            pass
    return (time.time() - path.stat().st_mtime) / 86400


def _auto_detect_archive(output_dir: Path) -> Optional[Path]:
    """
    Return the most-recently-modified ``*.zip`` sitting directly in
    *output_dir*, or ``None`` if it does not exist or contains none.

    Lets a user just drop the manually-downloaded ISMN zip into this run's
    own ISMN output folder instead of having to edit the recipe's
    ``download_kwargs`` — the common case of "one archive for one run".
    """
    if not output_dir.exists():
        return None
    zips = sorted(output_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0] if zips else None


def _print_portal_instructions(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float,
    start: str, end: str, min_depth: float, max_depth: float,
    output_dir: Path,
) -> None:
    """
    Print copy-pasteable filter values for the ISMN web portal.
    """
    print(
        "ISMN has no download API — data must be requested manually.\n"
        "1. Go to https://ismn.earth/en/dataviewer/ and log in\n"
        "   (registration is free).\n"
        "2. Use these filter values:\n"
        f"     Bounding box:  lon [{min_lon}, {max_lon}], lat [{min_lat}, {max_lat}]\n"
        f"     Date range:    {start} to {end}\n"
        "     Variable:      soil moisture\n"
        f"     Sensor depth:  {min_depth} to {max_depth} m (positive, below ground)\n"
        "3. Use these download options:\n"
        "     Format:        'Variables stored in separate files (CEOP\n"
        "                    formatted) (zipped)' — see\n"
        "                    https://ismn.earth/en/data/ceop-standard/.\n"
        "                    (ismn_downloader auto-detects this format;\n"
        "                    Header+Values also works if CEOP is\n"
        "                    unavailable for a given network.)\n"
        "     Quality flags: 'Good' only — see\n"
        "                    https://ismn.earth/en/data/ismn-quality-flags/.\n"
        "                    Flagged-bad observations would otherwise\n"
        "                    corrupt the validation statistics.\n"
        "     Gap filling:   leave disabled (do not fill gaps with NaN) —\n"
        "                    this toolbox matches on actual observation\n"
        "                    timestamps, so synthetic NaN-filled slots\n"
        "                    add nothing but archive size.\n"
        "4. Download the resulting zip and drop it directly into:\n"
        f"     {output_dir}\n"
        "   No recipe edits needed.\n"
        "5. Re-run this recipe — the archive is picked up automatically\n"
        "   once it is there.\n"
        "   To reuse one archive across multiple recipes instead, set its\n"
        "   path explicitly via 'ismn_archive_path' in this recipe's\n"
        "   download_kwargs for the 'ismn' validation source."
    )


def _build_station_index(archive_path: Path) -> pd.DataFrame:
    """
    Scan *archive_path* (a zip) for the first ``.stm`` file under each
    station directory and parse its CEOP-format first line for
    network/station/lat/lon, without reading the rest of that file or
    any other ``.stm`` file under the same station directory.

    CEOP first-line format (confirmed consistent across a random
    15-file sample spanning 8 networks in a real ISMN export):
    ``NETWORK SUBNETWORK STATION LAT LON ELEV DEPTH_FROM DEPTH_TO 'SENSOR'``.

    Returns a DataFrame with one row per station directory: columns
    ``network``, ``station``, ``lat``, ``lon``, ``dir_prefix`` (the zip
    path prefix, e.g. ``"REMEDHUS/Canizal/"``, used later to select
    every file belonging to that station for extraction). A row whose
    first line could not be parsed as expected gets ``lat=NaN,
    lon=NaN`` rather than being dropped -- callers must treat a NaN
    row as "always include", never silently excluding a real station
    over an unexpected format variant.
    """
    rows: list[dict] = []
    seen_dirs: set[str] = set()

    with zipfile.ZipFile(archive_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".stm"):
                continue
            dir_prefix = name.rsplit("/", 1)[0] + "/"
            if dir_prefix in seen_dirs:
                continue
            seen_dirs.add(dir_prefix)

            parts = dir_prefix.strip("/").split("/")
            network = parts[0] if parts else "unknown"
            station = parts[-1] if len(parts) > 1 else "unknown"
            lat = lon = float("nan")
            try:
                with zf.open(name) as f:
                    line = f.readline().decode("utf-8", errors="replace")
                tokens = line.split()
                lat = float(tokens[3])
                lon = float(tokens[4])
            except Exception:
                logger.warning(
                    "ISMNDownloader: could not parse coordinates from %s's "
                    "first line -- including it unconditionally in every "
                    "bbox filter rather than risking dropping a real station.",
                    name,
                )
            rows.append({
                "network": network, "station": station,
                "lat": lat, "lon": lon, "dir_prefix": dir_prefix,
            })

    return pd.DataFrame(rows, columns=["network", "station", "lat", "lon", "dir_prefix"])


def _station_index_path(archive_path: Path) -> Path:
    # Include the archive file's size and mtime in the cache key so a
    # replaced archive (same filename, different content -- the module
    # docstring already notes the shared archive "needs periodic manual
    # refreshing") invalidates the cache automatically instead of silently
    # serving a stale station list. Old cache files are simply orphaned,
    # not actively cleaned up -- they are small CSVs, not worth the added
    # cleanup logic.
    stat = archive_path.stat()
    fingerprint = f"{archive_path.stem}_{stat.st_size}_{int(stat.st_mtime)}"
    return _STATION_INDEX_CACHE_DIR / f"station_index_{fingerprint}.csv"


def _load_or_build_station_index(archive_path: Path) -> pd.DataFrame:
    """
    Load the cached station-coordinate index for *archive_path* if one
    exists, or build it and cache it if not.
    """
    index_path = _station_index_path(archive_path)
    if index_path.exists():
        return pd.read_csv(index_path)

    print(
        f"  Building ISMN station index for {archive_path.name} "
        f"(one-time cost, cached at {index_path})\u2026"
    )
    index_df = _build_station_index(archive_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(index_path, index=False)
    return index_df


def _extract_matching_stations(
    archive_path: Path, dir_prefixes: set[str], out_dir: Path,
) -> None:
    """
    Extract every zip member under one of *dir_prefixes*, plus the
    small global reference files (see _GLOBAL_REFERENCE_FILENAMES) if
    present, into *out_dir*.

    Writes ``_EXTRACTION_COMPLETE_MARKER`` inside *out_dir* once
    ``extractall`` returns -- callers reusing a persistent extraction
    directory (see ``_extracted_subset_dir``) must check for this
    marker, not just the directory's existence, since ``extractall``
    populates the directory incrementally and a killed/interrupted
    process (real risk here: the source archive is multi-GB) would
    otherwise leave a directory that *exists* but is silently missing
    an unknown subset of its files, with no error and no warning.
    """
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        members = [
            n for n in names
            if n in _GLOBAL_REFERENCE_FILENAMES or any(n.startswith(p) for p in dir_prefixes)
        ]
        zf.extractall(path=out_dir, members=members)
    (out_dir / _EXTRACTION_COMPLETE_MARKER).touch()


def _extracted_subset_dir(
    archive_path: Path, min_lon: float, max_lon: float, min_lat: float, max_lat: float,
) -> Path:
    """
    Deterministic, persistent extraction target for a given (archive,
    bbox) pair.

    Keyed on the archive's own fingerprint (same size+mtime scheme as 
    ``_station_index_path``) plus the bbox, so a second call with the
    same archive and bbox reuses the exact same directory -- including 
    ``ismn``'s own cache written inside it (e.g.
    ``<subset_dir>/python_metadata/<name>.csv``). Only the first-ever
    call for that (archive, bbox) pair pays the extraction and metadata
    build cost.

    Never cleaned up automatically: it is a persistent cache, behaving
    the same way the station-coordinate index does.
    """
    stat = archive_path.stat()
    fingerprint = f"{archive_path.stem}_{stat.st_size}_{int(stat.st_mtime)}"
    bbox_key = f"{min_lon:.2f}_{max_lon:.2f}_{min_lat:.2f}_{max_lat:.2f}"
    return _STATION_INDEX_CACHE_DIR / "extracted" / f"{fingerprint}_{bbox_key}"


def _sensor_depth_from(meta: pd.Series, default: float) -> float:
    """
    Extract a sensor's ``depth_from`` from its real per-sensor metadata
    ``pandas.Series`` (see ``ismn.meta.MetaData.to_pd``), preferring the
    "instrument" entry (matching ``ismn.filehandlers.IsmnFile.check_metadata``'s
    own precedence) and falling back to "variable", since ``ismn`` assigns the
    same ``Depth`` to both. Either key may be entirely absent (dropped by
    ``to_pd(dropna=True)``) if the sensor has no depth, e.g. for
    non-depth-resolved variables.
    """
    for key in (("instrument", "depth_from"), ("variable", "depth_from")):
        if key in meta.index:
            return float(meta.loc[key])
    return default


class ISMNDownloader:
    """
    Local-archive selector for ISMN soil-moisture stations.

    Parameters
    ----------
    output_dir : Path
        Directory to write one CSV per surviving sensor.
    dry_run : bool
        If True, print what would be selected without writing any files.
    """

    def __init__(self, output_dir: Path, dry_run: bool = False) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run

    def download(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start: str,
        end: str,
        min_depth: float = 0.0,
        max_depth: float = 0.05,
        archive_path: Optional[str] = None,
    ) -> list[Path]:
        """
        Filter a local ISMN archive to the recipe's window and write one CSV
        per surviving sensor.

        Returns
        -------
        list[Path]
            Paths to the written CSV files (empty if no archive is
            configured, no stations/sensors survive filtering, or
            ``dry_run`` is True).
        """
        resolved_archive_path = (
            Path(archive_path) if archive_path and Path(archive_path).exists()
            else _auto_detect_archive(self.output_dir)
            or _auto_detect_archive(_SHARED_ARCHIVE_CACHE_DIR)
        )
        if resolved_archive_path is None:
            _print_portal_instructions(
                min_lon, max_lon, min_lat, max_lat, start, end, min_depth, max_depth,
                self.output_dir,
            )
            return []

        if resolved_archive_path.parent == _SHARED_ARCHIVE_CACHE_DIR:
            age_days = _archive_age_days(resolved_archive_path)
            print(f"  Using shared ISMN archive cache: {resolved_archive_path} ({age_days:.0f} days old)")
            if age_days > _SHARED_ARCHIVE_STALE_DAYS:
                logger.warning(
                    "ISMNDownloader: shared archive cache is %.0f days old — "
                    "consider refreshing %s from https://ismn.earth/en/dataviewer/.",
                    age_days, resolved_archive_path,
                )

        index_df = _load_or_build_station_index(resolved_archive_path)

        station_count: Optional[int] = None
        if index_df.empty:
            # No .stm files found in the archive at all -- nothing to
            # filter. Fall back to the original, unfiltered archive
            # path, exactly the pre-existing behavior. A real ISMN
            # export always has stations; this only fires for a
            # degenerate/malformed archive.
            reader_source: Path = resolved_archive_path
        else:
            in_bbox_idx = (
                index_df["lat"].isna() | index_df["lon"].isna()
                | (
                    (index_df["lon"] >= min_lon) & (index_df["lon"] <= max_lon)
                    & (index_df["lat"] >= min_lat) & (index_df["lat"] <= max_lat)
                )
            )
            matched_prefixes = set(index_df.loc[in_bbox_idx, "dir_prefix"])
            if not matched_prefixes:
                logger.warning("ISMNDownloader: no stations in archive fall inside the requested bbox.")
                return []
            station_count = len(matched_prefixes)
            subset_dir = _extracted_subset_dir(resolved_archive_path, min_lon, max_lon, min_lat, max_lat)
            if (subset_dir / _EXTRACTION_COMPLETE_MARKER).exists():
                print(f"  Reusing previously-extracted ISMN station subset: {subset_dir}")
            else:
                # Either never extracted, or a prior extraction into
                # this exact directory was interrupted before finishing
                # (no completion marker) -- (re-)extract. extractall
                # overwrites/completes an existing partial tree.
                _extract_matching_stations(resolved_archive_path, matched_prefixes, subset_dir)
            reader_source = subset_dir

        if self.dry_run:
            # Bail out before constructing ISMN_Interface below: doing so
            # triggers ismn's own full sensor-level metadata scan
            # (IsmnFileCollection.build_from_scratch), which is slow on a large
            # archive and floods the terminal with a per-station progress bar --
            # exactly what --dry-run should avoid. Report the cheaper
            # station-level count from the index built above instead of the
            # exact sensor-level count a real run would report.
            if station_count is None:
                print(
                    "[DRY RUN] archive has no parseable station index -- "
                    "skipping the full metadata scan; would filter by "
                    "bbox/depth/variable and write CSVs."
                )
            else:
                print(
                    f"[DRY RUN] {station_count} station(s) in bbox "
                    "(sensor-level counts require the full metadata scan, "
                    "skipped here for speed); would filter by depth/"
                    "variable and write CSVs."
                )
            return []

        from ismn.interface import ISMN_Interface

        try:
            # Redirect stderr (where tqdm writes its progress bar by
            # default) to devnull for the duration of this call. ismn's
            # build_from_scratch hardcodes show_progress_bars=True with no
            # way for a caller to disable it; its \r-based updates only
            # render in place on an interactive tty -- piped/redirected/
            # logged output turns every update into its own line, flooding
            # the terminal with one line per station on the first scan of
            # a given archive+bbox. Only affects the one-time scan; a cached 
            # run reads the metadata CSV directly and never constructs a 
            # progress bar.
            with open(os.devnull, "w") as _devnull, contextlib.redirect_stderr(_devnull):
                reader = ISMN_Interface(reader_source, parallel=True)
        except ValueError as exc:
            if "No objects to concatenate" in str(exc):
                # ismn's own file reader needs at least two lines per .stm
                # file to bootstrap metadata (it reads the first, second,
                # and last line) -- a portal request for a single calendar
                # day produces exactly one reading per station file, so
                # every file fails to parse and this is what ismn raises
                # once none of them survive.
                raise ValueError(
                    "ismn could not parse any station file in this archive "
                    "-- every file appears to contain only a single "
                    "reading. This usually means the portal request's date "
                    "range covered a single calendar day; ismn's file "
                    "reader needs at least two readings per station file. "
                    "Re-download from https://ismn.earth/en/dataviewer/ "
                    "with a wider date range (e.g. a full week) and "
                    "replace this archive."
                ) from exc
            raise

        # ``ISMN_Interface.metadata`` is already a plain pandas.DataFrame (one
        # row per sensor/filehandler, indexed by the same integer id used by
        # ``get_dataset_ids``/``read``), with ("<name>", "val")-style
        # MultiIndex columns — there is no ``.to_pd()`` method on it. There is
        # no dedicated bbox-filter method in the ismn package (as of the
        # version pinned in pyproject.toml), so bbox filtering is done with
        # plain pandas indexing here; variable/depth filtering is delegated
        # to ``get_dataset_ids``, and the two id sets are intersected.
        meta_df = reader.metadata

        in_bbox = (
            (meta_df[("longitude", "val")] >= min_lon)
            & (meta_df[("longitude", "val")] <= max_lon)
            & (meta_df[("latitude", "val")] >= min_lat)
            & (meta_df[("latitude", "val")] <= max_lat)
        )
        bbox_ids = set(meta_df.index[in_bbox].tolist())

        self.output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        sensor_ids = reader.get_dataset_ids(
            variable="soil_moisture",
            min_depth=min_depth,
            max_depth=max_depth,
        )
        candidate_ids = [sensor_id for sensor_id in sensor_ids if sensor_id in bbox_ids]

        for sensor_id in candidate_ids:
            ts, meta = reader.read(sensor_id, return_meta=True)
            if ts is None or ts.empty:
                continue
            ts = ts.loc[start:end]
            if ts.empty:
                continue

            # ``meta`` is a pandas.Series with a (variable, key)-style
            # MultiIndex (built by ``MetaData.to_pd()`` in ismn/meta.py), not
            # a flat dict — index it with the real (name, "val") tuples.
            station_name = str(meta.loc[("station", "val")])
            lon = float(meta.loc[("longitude", "val")])
            lat = float(meta.loc[("latitude", "val")])
            depth = _sensor_depth_from(meta, default=min_depth)

            df_out = pd.DataFrame({
                "platform_id":   station_name,
                "platform_type": "ismn",
                "time":          ts.index,
                "lon":           lon,
                "lat":           lat,
                "depth":         depth,
                "variable":      "SOIL_MOISTURE",
                "value":         ts["soil_moisture"].values,
            })
            out_path = self.output_dir / f"ismn_{station_name}_{sensor_id}.csv"
            df_out.to_csv(out_path, index=False)
            written.append(out_path)

        if not written:
            logger.warning("ISMNDownloader: no stations/sensors survived bbox+depth+window filtering.")
        else:
            # One line total, not one per file -- a bbox covering a large
            # region of the shared/"total" archive can survive filtering
            # to thousands of sensors, and a line per file just floods the
            # terminal for no benefit (the files themselves are on disk).
            print(f"  Wrote {len(written)} ISMN sensor CSV(s) to {self.output_dir}")
        return written
