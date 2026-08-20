"""
Download delayed-mode (historical) HF-radar surface-current grids from
Copernicus Marine.

Data source: INSITU_GLO_PHY_UV_DISCRETE_MY_013_044
    Dataset ID: cmems_obs-ins_glo_phy-cur_my_radar-total_irr

Unlike the NRT product (see ``hf_radar_downloader.py``), this dataset is not
subsettable server-side: it exposes one bulk NetCDF file per named region
under its "original-files" service (``history/HF/GL_TV_HF_HFR-<Region>_Total[_<YYYY>].nc``),
1979-2026-ish depending on region. This downloader fetches the one matching
region file (cached via ``skip_existing``, since a region's file covers many
years and multiple runs will reuse it), then subsets it locally in xarray to
the requested time window and bbox, normalizing the on-disk shape (uppercase
OceanSITES dims + a singleton DEPTH axis) to match the NRT grid downloader's
output so both share the same converter path.

Library usage::

    from sar_validation.downloaders.hf_radar_historical_downloader import (
        HFRadarHistoricalDownloader,
    )
    dl = HFRadarHistoricalDownloader(output_dir=Path("data/run1/hf_radar_historical"))
    dl.download(min_lon=-90, max_lon=-60, min_lat=30, max_lat=40,
                start="2021-06-05", end="2021-06-06")

CLI usage::

    python -m sar_validation.downloaders.hf_radar_historical_downloader \\
        --min-lon -90 --max-lon -60 --min-lat 30 --max-lat 40 \\
        --start 2021-06-05 --end 2021-06-06
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from ._hf_radar_regions import resolve_hfr_region
from .base import build_output_dir, normalize_datetime, split_antimeridian_bbox

logger = logging.getLogger(__name__)

__all__ = ["HFRadarHistoricalDownloader"]

DATASET_ID = "cmems_obs-ins_glo_phy-cur_my_radar-total_irr"

# Fixed, run-independent location for the raw multi-year archive files
# (100s of MB each, one per region, covering that region's entire record).
# Every run reuses whatever's already here via copernicusmarine.get's
# skip_existing=True, instead of re-fetching into a fresh per-run folder.
_ARCHIVE_CACHE_DIR = Path("data") / "_archive_cache" / "hf_radar_historical"

# Regions present in the delayed-mode archive, verified via
# copernicusmarine.get(dataset_id=DATASET_ID, dry_run=True) on 2026-07-15.
# Regions in HFR_REGIONS but absent here (GoS, Granitola, WHub) have no
# historical archive. US-EastGulfCoast is split into one file per year
# (2019-2024); every other region is one file covering its whole record.
_SPLIT_BY_YEAR_REGION = "US-EastGulfCoast"
_SPLIT_BY_YEAR_YEARS = (2019, 2020, 2021, 2022, 2023, 2024)

_REGION_FILENAMES = {
    "ARPAS": "GL_TV_HF_HFR-ARPAS_Total.nc",
    "CALYPSO": "GL_TV_HF_HFR-CALYPSO_Total.nc",
    "COSYNA": "GL_TV_HF_HFR-COSYNA_Total.nc",
    "DeltaEbro": "GL_TV_HF_HFR-DeltaEbro_Total.nc",
    "EUSKOOS": "GL_TV_HF_HFR-EUSKOOS_Total.nc",
    "Finnmark": "GL_TV_HF_HFR-Finnmark_Total.nc",
    "Galicia": "GL_TV_HF_HFR-Galicia_Total.nc",
    "Gibraltar": "GL_TV_HF_HFR-Gibraltar_Total.nc",
    "ICATMAR": "GL_TV_HF_HFR-ICATMAR_Total.nc",
    "Ibiza": "GL_TV_HF_HFR-Ibiza_Total.nc",
    "Lisboa": "GL_TV_HF_HFR-Lisboa_Total.nc",
    "NAdr": "GL_TV_HF_HFR-NAdr_Total.nc",
    "PLOCAN": "GL_TV_HF_HFR-PLOCAN_Total.nc",
    "Skagerrak": "GL_TV_HF_HFR-Skagerrak_Total.nc",
    "South": "GL_TV_HF_HFR-South_Total.nc",
    "TirLig": "GL_TV_HF_HFR-TirLig_Total.nc",
    "US-Alaska": "GL_TV_HF_HFR-US-Alaska_Total.nc",
    "US-Hawaii": "GL_TV_HF_HFR-US-Hawaii_Total.nc",
    "US-PuertoRicoVirginIslands": "GL_TV_HF_HFR-US-PuertoRicoVirginIslands_Total.nc",
    "US-WestCoast": "GL_TV_HF_HFR-US-WestCoast_Total.nc",
    "Vestlandet": "GL_TV_HF_HFR-Vestlandet_Total.nc",
}

_MIN_AGE_DAYS: int = 182


def _parse_iso_dt(s: str) -> datetime:
    """Convert ISO date string from normalize_datetime to timezone-aware UTC datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=timezone.utc)


def _region_filename(region: str, start_dt: str, end_dt: str) -> str:
    if region == _SPLIT_BY_YEAR_REGION:
        start_year = int(start_dt[:4])
        end_year = int(end_dt[:4])
        if start_year != end_year:
            raise NotImplementedError(
                f"{_SPLIT_BY_YEAR_REGION}'s historical archive is split into one "
                "file per year; requests spanning more than a single calendar "
                f"year (got {start_dt} .. {end_dt}) are not yet supported."
            )
        if start_year not in _SPLIT_BY_YEAR_YEARS:
            raise ValueError(
                f"No {_SPLIT_BY_YEAR_REGION} historical archive for year {start_year}; "
                f"available years: {_SPLIT_BY_YEAR_YEARS}"
            )
        return f"GL_TV_HF_HFR-US-EastGulfCoast_Total_{start_year}.nc"
    if region not in _REGION_FILENAMES:
        raise ValueError(
            f"no delayed-mode HF-radar archive for region '{region}'. "
            f"Available: {sorted(_REGION_FILENAMES) + [_SPLIT_BY_YEAR_REGION]}"
        )
    return _REGION_FILENAMES[region]


class HFRadarHistoricalDownloader:
    """
    Download and locally subset a Copernicus Marine delayed-mode HF-radar
    current grid (dataset 013_044) for the region overlapping the request bbox.

    Parameters
    ----------
    output_dir : Path
        Directory to save the subsetted, normalized NetCDF.
    dry_run : bool
        If True, print what would be downloaded without fetching anything.
    """

    def __init__(self, output_dir: Path, dry_run: bool = False, force_download: bool = False) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.force_download = force_download

    def download(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start: str,
        end: str,
    ) -> list[Path]:
        start_dt = normalize_datetime(start)
        end_dt = normalize_datetime(end)
        cutoff = datetime.now(timezone.utc) - timedelta(days=_MIN_AGE_DAYS)
        if _parse_iso_dt(end_dt) > cutoff:
            logger.warning(
                "Skipping delayed-mode HF-radar download: end date %s is less than "
                "%d days old. The historical archive lags real-time by ~6 months. "
                "Re-run after %s.",
                end_dt, _MIN_AGE_DAYS, cutoff.strftime("%Y-%m-%d"),
            )
            return []
        windows = split_antimeridian_bbox(min_lon, max_lon)

        downloaded: list[Path] = []
        last_error: Optional[ValueError] = None
        resolved_any = False
        for i, (win_min_lon, win_max_lon) in enumerate(windows):
            suffix = f"_w{i}" if len(windows) > 1 else ""
            try:
                region = resolve_hfr_region(win_min_lon, win_max_lon, min_lat, max_lat)
            except ValueError as exc:
                if len(windows) == 1:
                    raise
                last_error = exc
                continue
            resolved_any = True
            path = self._download_region_window(
                region, win_min_lon, win_max_lon, min_lat, max_lat,
                start_dt, end_dt, suffix,
            )
            if path is not None:
                downloaded.append(path)

        if not resolved_any and last_error is not None:
            raise last_error
        return downloaded

    def check_availability_dry(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start: str,
        end: str,
    ) -> bool:
        """
        Whether a delayed-mode HF-radar archive plausibly covers this
        bbox/window, without fetching the archive file itself (100s of MB,
        one per region, covering that region's entire multi-year record --
        there is no cheaper remote metadata endpoint for this dataset that
        would report a specific archive file's own real temporal extent
        short of opening it).

        Purely a local lookup, reusing ``download()``'s own pre-flight
        checks exactly: the ``_MIN_AGE_DAYS`` recency guard, region
        resolution via ``resolve_hfr_region``, and archive-filename
        resolution via ``_region_filename`` (which already encodes,
        offline, which regions/years have an archive at all -- see its own
        ``ValueError`` cases, mirrored here as a False return). A
        region/year combination that resolves successfully here may still
        turn out to have no data for the exact requested days once the
        file is actually opened -- a plausible, not certain, "yes", the
        same fail-toward-inclusion convention used throughout this
        codebase's dry-collocation prediction path.

        A multi-year-spanning request against the split-by-year region
        (``NotImplementedError``) is a genuine, different limitation and is
        intentionally not caught here, mirroring ``_download_region_window``'s
        own handling of the same case.
        """
        start_dt = normalize_datetime(start)
        end_dt = normalize_datetime(end)
        cutoff = datetime.now(timezone.utc) - timedelta(days=_MIN_AGE_DAYS)
        if _parse_iso_dt(end_dt) > cutoff:
            return False

        try:
            region = resolve_hfr_region(min_lon, max_lon, min_lat, max_lat)
        except ValueError:
            return False

        try:
            _region_filename(region, start_dt, end_dt)
        except ValueError:
            return False
        return True

    def _download_region_window(
        self,
        region: str,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start_dt: str,
        end_dt: str,
        filename_suffix: str,
    ) -> Optional[Path]:
        try:
            remote_filename = _region_filename(region, start_dt, end_dt)
        except ValueError as exc:
            # No delayed-mode archive exists for this region, or (for the
            # split-by-year US-EastGulfCoast case) not for this year. This
            # is a "no data available" outcome, not a real failure — the
            # caller falls back to the NRT downloader. NotImplementedError
            # (multi-year-spanning request) is a different, genuine
            # limitation and is intentionally not caught here.
            logger.warning(
                "Skipping delayed-mode HF-radar download for region '%s': %s",
                region, exc,
            )
            return None

        start_d = start_dt.split("T")[0]
        end_d = end_dt.split("T")[0]
        date_str = start_d if start_d == end_d else f"{start_d}-{end_d}"
        dest_path = self.output_dir / f"{DATASET_ID}_{region}_{date_str}{filename_suffix}.nc"

        if self.dry_run:
            print(
                f"[DRY RUN] Would fetch Copernicus HF-radar historical archive "
                f"'{remote_filename}' for region '{region}' and subset to:\n  {dest_path}"
            )
            return None

        if not self.force_download and dest_path.exists():
            print(f"  Already downloaded: {dest_path}")
            return dest_path

        try:
            import copernicusmarine
        except ImportError as exc:
            raise ImportError(
                "copernicusmarine is required for HF radar downloads.\n"
                "Install it with:  pip install copernicusmarine"
            ) from exc
        import xarray as xr

        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw_cache_dir = _ARCHIVE_CACHE_DIR
        raw_cache_dir.mkdir(parents=True, exist_ok=True)

        print("Fetching Copernicus HF-radar delayed-mode archive …")
        print(f"  Region: {region}")
        print(f"  Archive file: {remote_filename}")
        resp = copernicusmarine.get(
            dataset_id=DATASET_ID,
            filter=f"*{remote_filename}",
            output_directory=str(raw_cache_dir),
            no_directories=True,
            skip_existing=True,
            disable_progress_bar=True,
        )
        if not resp.files:
            raise FileNotFoundError(
                f"No archive file matched '{remote_filename}' for region '{region}'."
            )
        raw_path = Path(resp.files[0].file_path)

        raw = xr.open_dataset(raw_path)
        try:
            # Keep EWCT/NSCT plus every ancillary uncertainty/QC field the
            # converter (Task 3) knows how to retain — standard deviations
            # (EWCS/NSCS), the geometric-dilution field (GDOP), the overall
            # QCflag, and each per-parameter QC flag — whichever of these
            # this archive file actually has.
            _ancillary_vars = (
                "GDOP", "EWCS", "NSCS", "QCflag",
                "CSPD_QC", "DDNS_QC", "GDOP_QC", "VART_QC", "POSITION_QC",
            )
            normalized = (
                raw[["EWCT", "NSCT"] + [v for v in _ancillary_vars if v in raw]]
                .squeeze("DEPTH", drop=True)
                .rename({"TIME": "time", "LATITUDE": "latitude", "LONGITUDE": "longitude"})
                .sortby(["time", "latitude", "longitude"])
                .sel(
                    time=slice(start_dt, end_dt),
                    latitude=slice(min_lat, max_lat),
                    longitude=slice(min_lon, max_lon),
                )
            )
            if normalized.sizes.get("time", 0) == 0:
                logger.warning(
                    "No delayed-mode HF-radar data for region '%s' in [%s, %s] "
                    "(the archive's real coverage may not extend this far "
                    "yet, even though the request is old enough per the "
                    "recency guard).",
                    region, start_dt, end_dt,
                )
                return None
            normalized.to_netcdf(dest_path)
        finally:
            raw.close()

        print(f"  Saved to {dest_path}")
        return dest_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download a Copernicus Marine delayed-mode HF-radar current grid.",
    )
    p.add_argument("--params-file", metavar="FILE")
    p.add_argument("--min-lon", type=float)
    p.add_argument("--max-lon", type=float)
    p.add_argument("--min-lat", type=float)
    p.add_argument("--max-lat", type=float)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if args.params_file:
        with open(args.params_file) as f:
            params = json.load(f)
        min_lon = params["minimum_longitude"]
        max_lon = params["maximum_longitude"]
        min_lat = params["minimum_latitude"]
        max_lat = params["maximum_latitude"]
        start = params["start_datetime"]
        end = params["end_datetime"]
    else:
        for attr in ("min_lon", "max_lon", "min_lat", "max_lat", "start", "end"):
            if getattr(args, attr) is None:
                print(f"Error: --{attr.replace('_','-')} is required (or use --params-file)")
                sys.exit(1)
        min_lon, max_lon = args.min_lon, args.max_lon
        min_lat, max_lat = args.min_lat, args.max_lat
        start, end = args.start, args.end

    output_dir = Path(args.output_dir) if args.output_dir else (
        build_output_dir(start, end, min_lon, max_lon, min_lat, max_lat) / "hf_radar_historical"
    )

    dl = HFRadarHistoricalDownloader(output_dir=output_dir, dry_run=args.dry_run)
    dl.download(
        min_lon=min_lon, max_lon=max_lon,
        min_lat=min_lat, max_lat=max_lat,
        start=start, end=end,
    )


if __name__ == "__main__":
    main()
