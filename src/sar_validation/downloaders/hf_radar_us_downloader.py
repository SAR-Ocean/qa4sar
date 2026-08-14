"""
Pick NOAA (ERDDAP, then THREDDS archive) or Copernicus Marine for a US
HF-radar current-grid request.

Tries, in order, the first to return at least one file wins:
  1. ERDDAP griddap (rolling ~90-day window) -- noaa_hfradar_downloader.py.
  2. THREDDS archive (2006-present, published a few weeks behind
     real-time) -- noaa_hfradar_thredds_downloader.py.
  3. Copernicus Marine (historical then NRT) -- hf_radar_downloader.py /
     hf_radar_historical_downloader.py. Only ever reached when NOAA (both
     of its own backends) produced nothing for the exact window -- this
     structurally avoids the double-counting risk of using NOAA and
     Copernicus for the same stations in the same run.

Applies uniformly across all 6 regions in NOAA_HFR_REGIONS. A bbox that
doesn't match any of the 6 (non-US networks, or Copernicus-only regions)
skips straight to step 3.

All backends write to the same folder names the rest of the toolbox
already discovers (``hfr_noaa/`` for ERDDAP+THREDDS, ``hf_radar/`` +
``hf_radar_historical/`` for Copernicus), so no converter or collocation
changes are needed to consume any path's output.

CLI usage::

    python -m sar_validation.downloaders.hf_radar_us_downloader \\
        --min-lon -130 --max-lon -115 --min-lat 33 --max-lat 48 \\
        --start 2026-06-01 --end 2026-06-01 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Union

from ._noaa_hfr_regions import finest_resolution_km, match_noaa_hfr_region
from .hf_radar_downloader import HFRadarDownloader
from .hf_radar_historical_downloader import HFRadarHistoricalDownloader
from .noaa_hfradar_downloader import DEFAULT_RESOLUTION_KM, NOAAHFRadarDownloader
from .noaa_hfradar_thredds_downloader import NOAATHREDDSHFRadarDownloader

logger = logging.getLogger(__name__)

__all__ = ["HFRadarUSDownloader"]


class HFRadarUSDownloader:
    """Download a US HF-radar current grid via the ERDDAP->THREDDS->Copernicus waterfall.

    Parameters
    ----------
    output_dir : Path
        The recipe run's base download directory (NOT a per-source
        subfolder) — ERDDAP and THREDDS both write into
        ``<output_dir>/hfr_noaa``, Copernicus writes into
        ``<output_dir>/hf_radar`` and ``<output_dir>/hf_radar_historical``.
    dry_run : bool
        Forwarded to whichever backend is tried.
    resolution_km : float, "finest", or None
        None (default) auto-picks the matched region's
        ``default_resolution_km``. ``"finest"`` picks
        ``finest_resolution_km(region)``. A float is used as-is. Ignored on
        the Copernicus fallback path (Copernicus publishes one
        fixed-resolution grid per region). Not applied at all for a bbox
        that matches no NOAA region.
    force_download : bool
        Forwarded to whichever backend is tried.
    """

    def __init__(
        self,
        output_dir: Path,
        dry_run: bool = False,
        resolution_km: Optional[Union[float, str]] = None,
        force_download: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.resolution_km = resolution_km
        self.force_download = force_download
        #: Set by download() to "erddap"/"thredds"/"copernicus" once
        #: resolved, so callers (the orchestrator) can record which backend
        #: actually produced the run's data.
        self.resolved_backend: Optional[str] = None
        #: Full ordered list of backends actually tried this call, even if
        #: every one came back empty -- used to build the combined
        #: "no data found" notice's wording.
        self.attempted_backends: list[str] = []

    def download(
        self, min_lon: float, max_lon: float, min_lat: float, max_lat: float,
        start: str, end: str,
    ) -> list[Path]:
        self.attempted_backends = []
        self.resolved_backend = None
        try:
            region_name, region = match_noaa_hfr_region(min_lon, max_lon, min_lat, max_lat)
        except ValueError:
            region_name, region = None, None

        resolution_km: float
        if self.resolution_km == "finest":
            resolution_km = finest_resolution_km(region) if region is not None else DEFAULT_RESOLUTION_KM
        elif isinstance(self.resolution_km, (int, float)):
            resolution_km = self.resolution_km
        else:
            resolution_km = region["default_resolution_km"] if region is not None else DEFAULT_RESOLUTION_KM

        if region is not None:
            if region["erddap_datasets"] is not None:
                self.attempted_backends.append("erddap")
                self._warn_if_stale_output("hf_radar", "Copernicus NRT")
                self._warn_if_stale_output("hf_radar_historical", "Copernicus delayed-mode")
                try:
                    erddap_dl = NOAAHFRadarDownloader(
                        output_dir=self.output_dir / "hfr_noaa",
                        dry_run=self.dry_run,
                        resolution_km=resolution_km,
                        force_download=self.force_download,
                    )
                    files = erddap_dl.download(min_lon, max_lon, min_lat, max_lat, start, end)
                    if files:
                        self.resolved_backend = "erddap"
                        return files
                    elif self.dry_run:
                        self.resolved_backend = "erddap"
                except (ValueError, NotImplementedError) as exc:
                    logger.info(
                        "hf_radar_us: ERDDAP not applicable for %s (%s), trying THREDDS",
                        region_name, exc,
                    )

            self.attempted_backends.append("thredds")
            self._warn_if_stale_output("hf_radar", "Copernicus NRT")
            self._warn_if_stale_output("hf_radar_historical", "Copernicus delayed-mode")
            try:
                thredds_dl = NOAATHREDDSHFRadarDownloader(
                    output_dir=self.output_dir / "hfr_noaa",
                    dry_run=self.dry_run,
                    resolution_km=resolution_km,
                    force_download=self.force_download,
                )
                files = thredds_dl.download(min_lon, max_lon, min_lat, max_lat, start, end)
                if files:
                    self.resolved_backend = "thredds"
                    return files
                elif self.dry_run and self.resolved_backend is None:
                    self.resolved_backend = "thredds"
            except ValueError as exc:
                logger.info(
                    "hf_radar_us: THREDDS not applicable for %s (%s), trying Copernicus",
                    region_name, exc,
                )

        self.attempted_backends.append("copernicus")
        self._warn_if_stale_output("hfr_noaa", "NOAA")

        # Historical-first, mirroring the orchestrator's _HISTORICAL_FIRST_PAIRS
        # gate for plain hf_radar/hf_radar_historical sources.
        historical = HFRadarHistoricalDownloader(
            output_dir=self.output_dir / "hf_radar_historical",
            dry_run=self.dry_run,
            force_download=self.force_download,
        )
        downloaded: list[Path] = list(
            historical.download(min_lon, max_lon, min_lat, max_lat, start, end)
        )
        if self.resolved_backend is None:
            self.resolved_backend = "copernicus"
        if downloaded:
            return downloaded

        nrt = HFRadarDownloader(
            output_dir=self.output_dir / "hf_radar",
            dry_run=self.dry_run,
            force_download=self.force_download,
        )
        downloaded.extend(nrt.download(min_lon, max_lon, min_lat, max_lat, start, end))
        return downloaded

    def _warn_if_stale_output(self, subdir: str, backend_label: str) -> None:
        """Warn if <output_dir>/subdir already holds cached .nc files from a
        run where a DIFFERENT backend was selected than this run's. Folder
        discovery in convert_downloaded_data is unconditional (checks only
        whether the folder exists), so leftover data from a prior run using
        the other backend would get discovered and collocated alongside
        this run's, double-counting the same stations."""
        d = self.output_dir / subdir
        if d.exists() and any(d.glob("*.nc")):
            logger.warning(
                "hf_radar_us: %s already contains cached %s data from a prior "
                "run, but this run may resolve to a different backend -- both "
                "will be discovered and collocated, double-counting the same "
                "stations. Delete %s if you only want this run's backend used.",
                d, backend_label, d,
            )


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download a US HF-radar current grid (ERDDAP -> THREDDS -> Copernicus).",
    )
    p.add_argument("--min-lon", type=float, required=True)
    p.add_argument("--max-lon", type=float, required=True)
    p.add_argument("--min-lat", type=float, required=True)
    p.add_argument("--max-lat", type=float, required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--resolution", type=float, default=None, choices=[0.5, 1, 2, 6])
    p.add_argument("--output-dir", default="data/hf_radar_us")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    dl = HFRadarUSDownloader(
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
        resolution_km=args.resolution,
    )
    out = dl.download(
        args.min_lon, args.max_lon, args.min_lat, args.max_lat, args.start, args.end,
    )
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
