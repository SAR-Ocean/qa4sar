"""
Download Copernicus Marine in-situ ocean observation data.

Downloads from the CMEMS in-situ dataset:
    cmems_obs-ins_glo_phybgcwav_mynrt_na_irr

Covers moorings (MO), drifting buoys (DB), ferryboxes (FB),
tide gauges (TG), and autonomous drifters (AD).

Library usage::

    from sar_validation.downloaders.insitu_downloader import InSituDownloader
    dl = InSituDownloader(output_dir=Path("data/run1/copernicus_insitu"))
    paths = dl.download(
        min_lon=-20, max_lon=0, min_lat=35, max_lat=60,
        start="2026-01-01", end="2026-01-02",
        source_types=["mooring", "buoy"],
    )

CLI usage::

    python -m sar_validation.downloaders.insitu_downloader \\
        --min-lon -20 --max-lon 0 --min-lat 35 --max-lat 60 \\
        --start 2026-01-01 --end 2026-01-02 \\
        --source-types mooring,buoy,tidal_gauge
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .base import build_output_dir, is_date_recent, normalize_datetime, split_antimeridian_bbox

__all__ = [
    "InSituDownloader",
    "SOURCE_TYPE_TO_PLATFORM",
    "PLATFORM_CODE_TO_SOURCE_TYPE",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

DATASET_ID = "cmems_obs-ins_glo_phybgcwav_mynrt_na_irr"
ALL_VARIABLES = ["WSPD", "WDIR", "VAVH", "VGHS", "VHM0", "HCDT", "HCSP", "EWCT", "NSCT"]

#: Recipe cfg.variable -> the ALL_VARIABLES subset actually comparable to
#: that physical quantity. "waves" mirrors core/_variable_map.py's own
#: WAVE_HEIGHT_VAL_VARS precedence set (VHM0/VAVH/VGHS -- see
#: DataTreeConverter.from_insitu_csv, the single source of truth for
#: which raw column wins when a station reports more than one). A
#: platform reporting only variables outside its recipe's own set (e.g.
#: a pure-currents drifter, EWCT/NSCT only, on a "waves" recipe) has
#: nothing to actually validate against and must not be treated as a
#: real collocation candidate at all -- see dry_collocation.py's
#: _predict_insitu, which uses this to filter both the query itself
#: (fewer irrelevant rows fetched/parsed, so faster too) and, in turn,
#: which stations even reach its own point-vs-footprint refinement.
RECIPE_VARIABLE_TO_INSITU_VARIABLES: "dict[str, tuple[str, ...]]" = {
    "waves": ("VHM0", "VAVH", "VGHS"),
    "wind": ("WSPD", "WDIR"),
    "currents": ("EWCT", "NSCT", "HCDT", "HCSP"),
}


def variables_for_recipe(variable: str) -> "tuple[str, ...]":
    """The ALL_VARIABLES subset relevant to *variable* (a recipe's own
    cfg.variable, e.g. "waves") -- falls back to the full ALL_VARIABLES
    set for an unrecognized value (e.g. "soil_moisture", which this
    dataset has no comparable variable for at all), fail-toward-inclusion
    rather than silently returning nothing comparable."""
    return RECIPE_VARIABLE_TO_INSITU_VARIABLES.get(variable, tuple(ALL_VARIABLES))

# Mapping from recipe source types to Copernicus platform codes.
# "drifter" resolves to both "DB" (drifting buoy) and "AD" (autonomous
# drifter).
SOURCE_TYPE_TO_PLATFORM = {
    "mooring":     ["MO"],
    "buoy":        ["DB"],
    "ferrybox":    ["FB"],
    "drifter":     ["DB", "AD"],
    "tidal_gauge": ["TG"],
}

# Reverse mapping, Copernicus platform code -> canonical source-type label,
# used to label individual observations by platform type. Not auto-derived
# from SOURCE_TYPE_TO_PLATFORM because "DB" is shared: it always labels as
# "buoy" (physical instrument identity) even though "drifter" also requests
# it; only "AD" labels as "drifter".
PLATFORM_CODE_TO_SOURCE_TYPE = {
    "MO": "mooring",
    "DB": "buoy",
    "FB": "ferrybox",
    "TG": "tidal_gauge",
    "AD": "drifter",
}


#: _fetch_stations_dry's own shared, per-process, network-fetch cache
#: (unfiltered by source_types -- see _fetch_stations_dry's own docstring
#: for why). Keyed by every query parameter that actually affects
#: copernicusmarine's own server-side result (bbox, window, dataset_part,
#: variables) -- deliberately NOT source_types, which is applied locally
#: afterward.
_fetch_stations_cache: "dict[tuple, pd.DataFrame]" = {}

#: One lock per cache key (created lazily, guarded by
#: _fetch_stations_locks_guard), not one lock for the whole cache: a
#: predict_collocation run's own ThreadPoolExecutor checks --dry-collocation's
#: five real in-situ source types (mooring/buoy/ferrybox/drifter/
#: tidal_gauge) concurrently, and every one of them shares the exact same
#: bbox/window/dataset_part/variables for a single recipe run (cfg.variable
#: is recipe-wide, not per-source) -- without this cache, that concurrency
#: turns into five simultaneous, otherwise-identical network requests
#: (and five simultaneous auth.marine.copernicus.eu token requests)
#: instead of one -- enough concurrent load to trip that server's own
#: read timeout under real-world load. The first caller for a given key
#: holds that key's own lock for the
#: duration of the real fetch; concurrent callers for the SAME key block
#: only on each other, never on a caller with a different key.
_fetch_stations_locks: "dict[tuple, threading.Lock]" = {}
_fetch_stations_locks_guard = threading.Lock()


def _get_fetch_stations_lock(cache_key: tuple) -> "threading.Lock":
    with _fetch_stations_locks_guard:
        lock = _fetch_stations_locks.get(cache_key)
        if lock is None:
            lock = threading.Lock()
            _fetch_stations_locks[cache_key] = lock
        return lock


def _resolve_platform_codes(source_types: list[str]) -> list[str]:
    """Map source-type names to Copernicus platform codes."""
    codes: list[str] = []
    for st in source_types:
        source_codes = SOURCE_TYPE_TO_PLATFORM.get(st.lower())
        if source_codes is None:
            raise ValueError(
                f"Unknown source type '{st}'. "
                f"Valid types: {', '.join(sorted(SOURCE_TYPE_TO_PLATFORM))}"
            )
        for code in source_codes:
            if code not in codes:
                codes.append(code)
    return codes


def _build_csv_filename(
    min_lon: float, max_lon: float,
    min_lat: float, max_lat: float,
    start_dt: str, end_dt: str,
    min_depth: float, max_depth: float,
) -> str:
    """Return the filename that copernicusmarine creates for the subset call."""
    def lon_sfx(v: float) -> str:
        return "W" if v < 0 else "E"

    def lat_sfx(v: float) -> str:
        return "S" if v < 0 else "N"
    vars_str = "-".join(ALL_VARIABLES)
    start_d = start_dt.split("T")[0]
    end_d   = end_dt.split("T")[0]
    date_str = start_d if start_d == end_d else f"{start_d}-{end_d}"
    depth_str = f"{abs(min_depth):.2f}-{abs(max_depth):.2f}m"
    return (
        f"{DATASET_ID}_{vars_str}_"
        f"{abs(min_lon):.2f}{lon_sfx(min_lon)}-{abs(max_lon):.2f}{lon_sfx(max_lon)}_"
        f"{abs(min_lat):.2f}{lat_sfx(min_lat)}-{abs(max_lat):.2f}{lat_sfx(max_lat)}_"
        f"{depth_str}_{date_str}.csv"
    )


class InSituDownloader:
    """
    Download in-situ observations from Copernicus Marine.

    Parameters
    ----------
    output_dir : Path
        Directory to save downloaded CSV files.
    dry_run : bool
        If True, print what would be downloaded without actually downloading.
    min_depth, max_depth : float
        Depth range for the query (metres; negative = below sea surface).
    """

    def __init__(
        self,
        output_dir: Path,
        dry_run: bool = False,
        min_depth: float = -20.0,
        max_depth: float = 20.0,
        force_download: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.force_download = force_download

    def _get_copernicusmarine(self):
        """Lazy import of copernicusmarine, isolated into its own method
        (rather than an inline ``import`` in each caller) so tests can
        monkeypatch this one method instead of faking ``sys.modules``."""
        try:
            import copernicusmarine
        except ImportError as exc:
            raise ImportError(
                "copernicusmarine is required for in-situ downloads.\n"
                "Install it with:  pip install copernicusmarine"
            ) from exc
        return copernicusmarine

    def download(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start: str,
        end: str,
        source_types: Optional[list[str]] = None,
        dataset_part: Optional[str] = None,
    ) -> list[Path]:
        """
        Download in-situ observations and save one CSV per non-crossing
        longitude window (see ``split_antimeridian_bbox``).

        Parameters
        ----------
        source_types : list[str], optional
            Filter by platform type(s): mooring, buoy, ferrybox, drifter, tidal_gauge.
            None or empty list means keep all platform types.
        dataset_part : str, optional
            Which dataset part to use: "monthly" (historical) or "latest" (recent).
            If None, auto-detects based on whether end_date is within 30 days.

        Returns
        -------
        list[Path]
            Paths to the downloaded CSVs (one per window that produced data).
        """
        copernicusmarine = self._get_copernicusmarine()

        start_dt = normalize_datetime(start)
        end_dt   = normalize_datetime(end)

        downloaded: list[Path] = []
        for win_min_lon, win_max_lon in split_antimeridian_bbox(min_lon, max_lon):
            path = self._download_window(
                copernicusmarine, win_min_lon, win_max_lon, min_lat, max_lat,
                start_dt, end_dt, source_types, dataset_part,
            )
            if path is not None:
                downloaded.append(path)
        return downloaded

    def _fetch_stations_dry(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start: str,
        end: str,
        source_types: Optional[list[str]] = None,
        dataset_part: Optional[str] = None,
        variables: Optional["Iterable[str]"] = None,
    ) -> "pd.DataFrame":
        """
        Shared fetch behind check_availability_dry and station_ranges_dry:
        the real, platform_type-filtered in-situ dataframe for this
        bbox/time window, without writing anything to disk.

        Uses ``copernicusmarine.read_dataframe()`` rather than ``subset()``
        (the real download path's call) or ``open_dataset()`` (used by
        this codebase's other Copernicus Marine sources): this dataset's
        storage format doesn't support lazy ``xarray`` loading, so
        ``read_dataframe()`` -- an in-memory, bbox/time-filtered fetch
        with no local file written -- is the lightest real existence
        check available for it. ``source_types`` filters the result the
        same way ``download()``'s own post-hoc ``platform_type`` filter
        does. ``variables`` narrows the query to just the physical
        quantities relevant to the caller (see ``variables_for_recipe``)
        rather than always querying the full ``ALL_VARIABLES`` set --
        both a genuine availability question (a station reporting only
        wind speed has nothing comparable to a "waves" recipe) and a
        real speed win (fewer irrelevant rows fetched/parsed). Defaults
        to ``ALL_VARIABLES`` for a caller with no recipe-specific
        variable in scope (e.g. the real ``download()`` path, which
        historically wants every variable a platform might report).

        An antimeridian-crossing bbox (min_lon > max_lon, this codebase's
        own wrap convention) is split into one query per non-wrapping
        range via ``split_antimeridian_bbox`` first, mirroring
        ``download()``'s own identical handling -- passed straight
        through unsplit, ``copernicusmarine.read_dataframe`` has no
        concept of a wrapping bbox, so min_lon > max_lon there is simply
        an empty/invalid range, not "wrap through 180". Concatenated
        with duplicates dropped, since a station whose position matches
        both split ranges' rounding could otherwise appear twice.

        The real network fetch (everything except the final
        ``source_types`` filter) is shared via ``_fetch_stations_cache``
        across every caller requesting the same bbox/window/dataset_part/
        variables/depth -- see that cache's own module-level comment for
        why this matters for --dry-collocation specifically.
        """
        start_dt = normalize_datetime(start)
        end_dt = normalize_datetime(end)
        resolved_part = dataset_part or ("latest" if is_date_recent(end_dt) else "monthly")
        resolved_variables = list(variables) if variables is not None else ALL_VARIABLES

        cache_key = (
            min_lon, max_lon, min_lat, max_lat, start_dt, end_dt,
            resolved_part, tuple(resolved_variables), self.min_depth, self.max_depth,
        )
        lock = _get_fetch_stations_lock(cache_key)
        with lock:
            df = _fetch_stations_cache.get(cache_key)
            if df is None:
                df = self._fetch_stations_uncached(
                    min_lon, max_lon, min_lat, max_lat, start_dt, end_dt, resolved_part, resolved_variables,
                )
                _fetch_stations_cache[cache_key] = df

        if df.empty:
            return df

        if source_types:
            platform_codes = _resolve_platform_codes(source_types)
            if platform_codes and "platform_type" in df.columns:
                df = df[df["platform_type"].isin(platform_codes)]

        return df

    def _fetch_stations_uncached(
        self,
        min_lon: float, max_lon: float, min_lat: float, max_lat: float,
        start_dt: str, end_dt: str, resolved_part: str, resolved_variables: "list[str]",
    ) -> "pd.DataFrame":
        """The actual copernicusmarine.read_dataframe() call(s)
        _fetch_stations_dry shares via its own cache -- split out so the
        cache-hit path never re-imports copernicusmarine or repeats this
        method's own logging for a request another caller already
        satisfied."""
        copernicusmarine = self._get_copernicusmarine()

        frames = []
        for win_min_lon, win_max_lon in split_antimeridian_bbox(min_lon, max_lon):
            print(
                f"  Checking in-situ availability: bbox=[{win_min_lon:.2f}, {win_max_lon:.2f}, "
                f"{min_lat:.2f}, {max_lat:.2f}]  window={start_dt} → {end_dt}  "
                f"dataset={DATASET_ID} ({resolved_part})  variables={resolved_variables}"
            )
            frames.append(copernicusmarine.read_dataframe(
                dataset_id=DATASET_ID,
                dataset_part=resolved_part,
                variables=resolved_variables,
                minimum_longitude=win_min_lon,
                maximum_longitude=win_max_lon,
                minimum_latitude=min_lat,
                maximum_latitude=max_lat,
                start_datetime=start_dt,
                end_datetime=end_dt,
                minimum_depth=self.min_depth,
                maximum_depth=self.max_depth,
                disable_progress_bar=True,
            ))
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if df.empty:
            return df
        return df.drop_duplicates().reset_index(drop=True)

    def check_availability_dry(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start: str,
        end: str,
        source_types: Optional[list[str]] = None,
        dataset_part: Optional[str] = None,
        variables: Optional["Iterable[str]"] = None,
    ) -> bool:
        """
        Whether any in-situ observation exists in this bbox/time window,
        without writing anything to disk. See _fetch_stations_dry's own
        docstring for the underlying query this collapses to a boolean
        (including what ``variables`` narrows); station_ranges_dry is the
        same query kept as real per-station coordinates, for a caller
        wanting to refine against a real SAR footprint shape rather than
        just its bbox (see dry_collocation.py's _predict_insitu, which
        does exactly this).
        """
        df = self._fetch_stations_dry(
            min_lon, max_lon, min_lat, max_lat, start, end, source_types, dataset_part, variables,
        )
        return not df.empty

    def station_ranges_dry(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start: str,
        end: str,
        source_types: Optional[list[str]] = None,
        dataset_part: Optional[str] = None,
        variables: Optional["Iterable[str]"] = None,
    ) -> "dict[str, tuple[float, float, datetime, datetime]]":
        """
        platform_id -> (lat, lon, earliest, latest) for every real in-situ
        station reporting data in this bbox/time window.

        Mirrors ISMNDownloader.station_date_ranges_dry's exact return
        shape (see dry_collocation.py's _predict_ismn), so a caller can
        apply the same real point-vs-footprint-shape refinement
        (dry_collocation._point_in_footprint) this dataset's boolean-only
        check_availability_dry can't offer: a WV-mode SAR footprint's own
        bbox is the bounding envelope of dozens of small vignette points
        scattered across up to ~30 minutes of orbit track, sometimes
        thousands of km wide -- checking a station against that whole
        bbox, rather than against the real vignette locations, badly
        over-matches. Uses the same _fetch_stations_dry() call
        check_availability_dry makes, just keeping the real
        per-observation coordinates instead of collapsing them to a
        boolean. See _fetch_stations_dry's own docstring for what
        ``variables`` narrows and why.
        """
        df = self._fetch_stations_dry(
            min_lon, max_lon, min_lat, max_lat, start, end, source_types, dataset_part, variables,
        )
        if df.empty:
            return {}

        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])
        ranges: "dict[str, tuple[float, float, datetime, datetime]]" = {}
        for platform_id, group in df.groupby("platform_id"):
            ranges[str(platform_id)] = (
                float(group["latitude"].iloc[0]), float(group["longitude"].iloc[0]),
                group["time"].min().to_pydatetime(), group["time"].max().to_pydatetime(),
            )
        return ranges

    def _download_window(
        self,
        copernicusmarine,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start_dt: str,
        end_dt: str,
        source_types: Optional[list[str]],
        dataset_part: Optional[str],
    ) -> Optional[Path]:
        """
        Download and save one CSV for a single (non-crossing) window.
        """
        expected_filename = _build_csv_filename(
            min_lon, max_lon, min_lat, max_lat,
            start_dt, end_dt, self.min_depth, self.max_depth,
        )
        dest_path = self.output_dir / expected_filename

        if self.dry_run:
            print(
                f"[DRY RUN] Would download in-situ data to:\n  {dest_path}"
            )
            return None

        if not self.force_download and dest_path.exists():
            print(f"  Already downloaded: {dest_path}")
            return dest_path

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect dataset_part if not provided
        resolved_part = dataset_part
        if resolved_part is None:
            resolved_part = "latest" if is_date_recent(end_dt) else "monthly"

        # Run the copernicusmarine subset call (downloads to CWD by default)
        print("Downloading in-situ data …")
        print(f"  Region: lon [{min_lon}, {max_lon}] lat [{min_lat}, {max_lat}]")
        print(f"  Time:   {start_dt} → {end_dt}")
        print(f"  Depth:  {self.min_depth} to {self.max_depth} m")
        print(f"  Dataset: {resolved_part}")

        # Try initial dataset_part, with fallback if data not available
        try:
            self._download_with_part(
                copernicusmarine,
                resolved_part,
                min_lon, max_lon, min_lat, max_lat,
                start_dt, end_dt,
            )
        except Exception as e:
            error_msg = str(e)
            # Check if error is about data exceeding coordinates (date outside available range)
            if "exceed the dataset coordinates" in error_msg or "out of bounds" in error_msg.lower():
                # Try the opposite dataset_part
                alt_dataset_part = "monthly" if resolved_part == "latest" else "latest"
                print(f"  Retrying with dataset_part='{alt_dataset_part}' due to: {error_msg[:100]}…")
                try:
                    self._download_with_part(
                        copernicusmarine,
                        alt_dataset_part,
                        min_lon, max_lon, min_lat, max_lat,
                        start_dt, end_dt,
                    )
                    resolved_part = alt_dataset_part
                except Exception as e2:
                    raise RuntimeError(
                        f"In-situ data unavailable for [{start_dt}, {end_dt}] in either "
                        f"CMEMS dataset_part of {DATASET_ID}:\n"
                        f"  '{resolved_part}': {error_msg}\n"
                        f"  '{alt_dataset_part}': {e2}"
                    ) from e2
            else:
                # Not a data availability error, re-raise original
                raise

        # Move the file (copernicusmarine writes it to CWD) to our output_dir
        if Path(expected_filename).exists():
            shutil.move(str(expected_filename), str(dest_path))
            print(f"  Saved to {dest_path}")
        elif dest_path.exists():
            print(f"  Already at {dest_path}")
        else:
            # Try to find a recently-created CSV in CWD
            candidates = sorted(Path(".").glob(f"{DATASET_ID}*.csv"), key=os.path.getmtime, reverse=True)
            if candidates:
                shutil.move(str(candidates[0]), str(dest_path))
                print(f"  Saved to {dest_path}")
            else:
                logger.debug(
                    "No in-situ observations in [%s, %s]; copernicusmarine "
                    "wrote no output file.", start_dt, end_dt,
                )
                return None

        # Apply platform-type filter
        if source_types:
            platform_codes = _resolve_platform_codes(source_types)
            if platform_codes:
                df = pd.read_csv(dest_path)
                if "platform_type" in df.columns:
                    df = df[df["platform_type"].isin(platform_codes)]
                    df.to_csv(dest_path, index=False)
                    print(f"  Filtered to {len(df)} rows ({', '.join(source_types)})")

        return dest_path

    def _download_with_part(
        self,
        copernicusmarine,
        dataset_part: str,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start_dt: str,
        end_dt: str,
    ) -> None:
        """Internal helper to run copernicusmarine.subset() with a specific dataset_part."""
        copernicusmarine.subset(
            dataset_id=DATASET_ID,
            dataset_part=dataset_part,
            variables=ALL_VARIABLES,
            minimum_longitude=min_lon,
            maximum_longitude=max_lon,
            minimum_latitude=min_lat,
            maximum_latitude=max_lat,
            start_datetime=start_dt,
            end_datetime=end_dt,
            minimum_depth=self.min_depth,
            maximum_depth=self.max_depth,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download Copernicus Marine in-situ observations.",
    )
    p.add_argument("--params-file", metavar="FILE")
    p.add_argument("--min-lon", type=float)
    p.add_argument("--max-lon", type=float)
    p.add_argument("--min-lat", type=float)
    p.add_argument("--max-lat", type=float)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--min-depth", type=float, default=-20.0)
    p.add_argument("--max-depth", type=float, default=20.0)
    p.add_argument(
        "--source-types",
        help="Comma-separated: mooring,buoy,ferrybox,drifter,tidal_gauge",
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if args.params_file:
        with open(args.params_file) as f:
            params = json.load(f)
        min_lon   = params["minimum_longitude"]
        max_lon   = params["maximum_longitude"]
        min_lat   = params["minimum_latitude"]
        max_lat   = params["maximum_latitude"]
        start     = params["start_datetime"]
        end       = params["end_datetime"]
        min_depth = params.get("minimum_depth", -20.0)
        max_depth = params.get("maximum_depth", 20.0)
    else:
        for attr in ("min_lon", "max_lon", "min_lat", "max_lat", "start", "end"):
            if getattr(args, attr) is None:
                print(f"Error: --{attr.replace('_','-')} is required (or use --params-file)")
                sys.exit(1)
        min_lon, max_lon = args.min_lon, args.max_lon
        min_lat, max_lat = args.min_lat, args.max_lat
        start, end = args.start, args.end
        min_depth, max_depth = args.min_depth, args.max_depth

    output_dir = Path(args.output_dir) if args.output_dir else (
        build_output_dir(start, end, min_lon, max_lon, min_lat, max_lat)
        / "copernicus_insitu"
    )

    source_types = (
        [s.strip() for s in args.source_types.split(",")]
        if args.source_types else None
    )

    dl = InSituDownloader(
        output_dir=output_dir,
        dry_run=args.dry_run,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    dl.download(
        min_lon=min_lon, max_lon=max_lon,
        min_lat=min_lat, max_lat=max_lat,
        start=start, end=end,
        source_types=source_types,
    )


if __name__ == "__main__":
    main()
