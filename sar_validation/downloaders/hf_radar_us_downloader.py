"""
Pick NOAA ERDDAP or Copernicus Marine for a US HF-radar current-grid request.

NOAA's ERDDAP griddap distribution of the U.S. IOOS/HFRNet national HF-radar
network has substantially denser real-world coverage than Copernicus's
re-ingestion of the same network for US-WestCoast/US-EastGulfCoast (confirmed
2026-07-30: ~5.8x-17x more valid grid cells for an identical bbox/date/
resolution — see
docs/superpowers/specs/2026-07-30-hf-radar-us-source-noaa-primary-copernicus-fallback-design.md).
This downloader therefore prefers NOAA whenever the request's region and
date are within NOAA's ERDDAP coverage (~90 rolling days), and falls back to
Copernicus (NRT + delayed-mode) otherwise — for regions NOAA doesn't cover,
or dates older than its window (NOAA's THREDDS/OPeNDAP archive backend,
which would cover those dates directly from NOAA, is not implemented).

Both backends write to the same folder names the rest of the toolbox
already discovers (``hfr_noaa/``, ``hf_radar/``, ``hf_radar_historical/``),
so no converter or collocation changes are needed to consume either path's
output.

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
from typing import Optional

from .hf_radar_downloader import HFRadarDownloader
from .hf_radar_historical_downloader import HFRadarHistoricalDownloader
from .noaa_hfradar_downloader import (
    DEFAULT_RESOLUTION_KM,
    NOAAHFRadarDownloader,
    match_region,
    select_backend,
)

logger = logging.getLogger(__name__)

__all__ = ["HFRadarUSDownloader", "resolve_hf_radar_us_backend"]


def resolve_hf_radar_us_backend(min_lon, max_lon, min_lat, max_lat, end: str) -> str:
    """Return ``"noaa"`` if NOAA's ERDDAP covers this bbox/date, else ``"copernicus"``.

    NOAA covers a bbox if it resolves to one of its configured US regions
    (``match_region``) AND the request's end date is within its rolling
    ERDDAP window (``select_backend``). Either condition failing (a region
    mismatch, or a date outside the window) means Copernicus is used
    instead. A malformed ``end`` date string still propagates its
    ``ValueError`` from ``select_backend``'s date parsing — this function
    only absorbs the two conditions above, not input validation.
    """
    try:
        match_region(min_lon, max_lon, min_lat, max_lat)
    except ValueError:
        return "copernicus"
    try:
        select_backend(end)
    except NotImplementedError:
        return "copernicus"
    return "noaa"


class HFRadarUSDownloader:
    """Download a US HF-radar current grid, preferring NOAA over Copernicus.

    Parameters
    ----------
    output_dir : Path
        The recipe run's base download directory (NOT a per-source
        subfolder) — the NOAA path writes into ``<output_dir>/hfr_noaa``,
        the Copernicus path writes into ``<output_dir>/hf_radar`` and
        ``<output_dir>/hf_radar_historical``, matching the folder names the
        plain ``hf_radar_noaa``/``hf_radar``/``hf_radar_historical`` sources
        already use.
    dry_run : bool
        Forwarded to whichever backend is selected.
    resolution_km : int
        Forwarded to ``NOAAHFRadarDownloader`` only; ignored on the
        Copernicus fallback path (Copernicus publishes one fixed-resolution
        grid per region, no resolution parameter to pass).
    force_download : bool
        Forwarded to whichever backend is selected.
    """

    def __init__(
        self,
        output_dir: Path,
        dry_run: bool = False,
        resolution_km: int = DEFAULT_RESOLUTION_KM,
        force_download: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.resolution_km = resolution_km
        self.force_download = force_download
        #: Set by download() to "noaa" or "copernicus" once resolved, so
        #: callers (the orchestrator) can record which backend actually ran
        #: without re-deriving it or inspecting the output folder layout.
        self.resolved_backend: Optional[str] = None

    def download(
        self, min_lon: float, max_lon: float, min_lat: float, max_lat: float,
        start: str, end: str,
    ) -> list[Path]:
        backend = resolve_hf_radar_us_backend(min_lon, max_lon, min_lat, max_lat, end)
        self.resolved_backend = backend
        logger.info(
            "hf_radar_us: resolved to %s backend for bbox=[%s, %s, %s, %s] window=[%s, %s]",
            backend, min_lon, max_lon, min_lat, max_lat, start, end,
        )

        if backend == "noaa":
            self._warn_if_stale_output("hf_radar", "Copernicus NRT")
            self._warn_if_stale_output("hf_radar_historical", "Copernicus delayed-mode")
            dl = NOAAHFRadarDownloader(
                output_dir=self.output_dir / "hfr_noaa",
                dry_run=self.dry_run,
                resolution_km=self.resolution_km,
                force_download=self.force_download,
            )
            return dl.download(min_lon, max_lon, min_lat, max_lat, start, end)

        self._warn_if_stale_output("hfr_noaa", "NOAA")

        # Historical-first, mirroring the orchestrator's _HISTORICAL_FIRST_PAIRS
        # gate for plain hf_radar/hf_radar_historical sources: the delayed-mode
        # archive is checked first, and the NRT grid is only fetched if the
        # archive produced nothing for this window. Calling both
        # unconditionally would double-count the same HFRNet stations via two
        # differently-processed grids of the same network -- exactly the
        # failure mode this downloader exists to eliminate.
        historical = HFRadarHistoricalDownloader(
            output_dir=self.output_dir / "hf_radar_historical",
            dry_run=self.dry_run,
            force_download=self.force_download,
        )
        downloaded: list[Path] = list(
            historical.download(min_lon, max_lon, min_lat, max_lat, start, end)
        )
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
        the other backend would get collocated alongside this run's
        selected backend, double-counting the same stations."""
        d = self.output_dir / subdir
        if d.exists() and any(d.glob("*.nc")):
            logger.warning(
                "hf_radar_us: %s already contains cached %s data from a prior "
                "run, but this run resolved to a different backend -- both "
                "will be discovered and collocated, double-counting the same "
                "stations. Delete %s if you only want this run's backend used.",
                d, backend_label, d,
            )


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download a US HF-radar current grid (NOAA primary, Copernicus fallback).",
    )
    p.add_argument("--min-lon", type=float, required=True)
    p.add_argument("--max-lon", type=float, required=True)
    p.add_argument("--min-lat", type=float, required=True)
    p.add_argument("--max-lat", type=float, required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION_KM, choices=[1, 2, 6])
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
