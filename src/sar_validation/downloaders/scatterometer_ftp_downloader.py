"""
Download HY-2B/HY-2C/Oceansat-3 25 km scatterometer wind data via the
OSI-SAF wind FTP server (ftppro.knmi.nl).

These satellites are processed by OSI-SAF but not distributed through the
EUMDAC Data Store that ``scatterometer_downloader.py`` uses for ASCAT-B/C —
the only access path is this FTP server, which retains a rolling 3-day
window of data. ASCAT-B/C stays exclusively on the EUMDAC path; this
downloader never touches it.

Filename matching is by substring (25 km resolution token + wind-vector
product suffix), not the exact per-satellite numeric processing code, since
that code isn't guaranteed stable across satellites/versions and can't be
verified for HY-2B/HY-2C while their FTP directories are empty.

Library usage::

    from sar_validation.downloaders.scatterometer_ftp_downloader import ScatterometerFTPDownloader
    dl = ScatterometerFTPDownloader(satellite="oceansat3", output_dir=Path("data/run1/scatterometer_oceansat3"))
    dl.download(min_lon=-20, max_lon=0, min_lat=35, max_lat=60,
                start="2026-07-18", end="2026-07-19")

CLI usage::

    python -m sar_validation.downloaders.scatterometer_ftp_downloader \\
        --satellite oceansat3 \\
        --min-lon -20 --max-lon 0 --min-lat 35 --max-lat 60 \\
        --start 2026-07-18 --end 2026-07-19
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .base import authenticate_osi_saf_ftp, build_output_dir, normalize_datetime

logger = logging.getLogger(__name__)

__all__ = ["ScatterometerFTPDownloader"]

FTP_HOST = "ftppro.knmi.nl"

# FTP directory per satellite. Verified live 2026-07-21: hy2b/hy2c are
# currently empty (no active files), oceansat3 has ~350 recent files.
_SATELLITE_PATHS = {
    "hy2b": "/scat/netcdf/hy2b",
    "hy2c": "/scat/netcdf/hy2c",
    "oceansat3": "/scat/netcdf/oceansat3",
}

# The FTP server only retains this many days of data.
_MAX_AGE_DAYS = 3

_TIMESTAMP_RE = re.compile(r"(\d{8})_(\d{6})")

#: One file per orbit revolution (confirmed by the incrementing
#: revolution-counter token in real filenames, e.g. "_19234_") -- pad
#: each file's single embedded timestamp by one orbital period so the
#: sampled arc passed to orbit_overlaps_bbox covers the file's real
#: acquisition span, not a degenerate single instant (which would fail
#: open and silently disable filtering -- see orbit_coverage.py's
#: orbit_overlaps_bbox docstring). ~100 minutes approximates one LEO
#: orbital period at these satellites' sun-synchronous altitudes;
#: conservatively long, not short, per the fail-toward-inclusion
#: principle -- verify against real filename timestamp deltas or
#: OSI-SAF's Product User Manual and widen if real data disagrees.
_ASSUMED_PASS_DURATION = timedelta(minutes=100)


def _matches_25km(filename: str) -> bool:
    """True if filename is a 25 km wind-vector product file (not its .md5 sidecar)."""
    if filename.endswith(".md5"):
        return False
    return "_250_" in filename and "_ovw" in filename


def _parse_filename_timestamp(filename: str) -> Optional[datetime]:
    """Extract the embedded _YYYYMMDD_HHMMSS_ timestamp from an OSI-SAF FTP filename."""
    m = _TIMESTAMP_RE.search(filename)
    if not m:
        return None
    try:
        return datetime.strptime(
            m.group(1) + m.group(2), "%Y%m%d%H%M%S"
        ).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_iso_dt(s: str) -> datetime:
    """Convert an ISO datetime string (from normalize_datetime) to a UTC-aware datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=timezone.utc)


class ScatterometerFTPDownloader:
    """
    Download 25 km scatterometer wind files for one satellite from the
    OSI-SAF wind FTP server.

    Parameters
    ----------
    satellite : str
        One of "hy2b", "hy2c", "oceansat3".
    output_dir : Path
        Directory to save downloaded (and gunzipped) files.
    dry_run : bool
        If True and credentials are already configured, connect and report
        real matching file availability without downloading anything; if
        credentials are not configured, print a message to set them up
        (never raises/prompts).
    username, password : str, optional
        OSI-SAF FTP credentials. If omitted, resolved via authenticate_osi_saf_ftp().
    """

    def __init__(
        self,
        satellite: str,
        output_dir: Path,
        dry_run: bool = False,
        force_download: bool = False,
        username: Optional[str] = None,
        password: Optional[str] = None,
        orbit_prefilter: bool = True,
    ) -> None:
        if satellite not in _SATELLITE_PATHS:
            raise ValueError(
                f"Unknown satellite '{satellite}'. Valid: {sorted(_SATELLITE_PATHS)}"
            )
        self.satellite = satellite
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.force_download = force_download
        self._username = username
        self._password = password
        self.orbit_prefilter = orbit_prefilter

    def download(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start: str,
        end: str,
    ) -> list[Path]:
        """
        Download every 25 km wind file for this satellite whose filename
        timestamp falls in [start, end].

        No server-side bbox filtering exists on this FTP (full-orbit swaths
        only) — bbox is accepted for interface consistency with every other
        downloader but not used to filter here; domain cropping happens
        downstream in DataTreeConverter.convert_downloaded_data's
        recipe-domain filter, same as the EUMDAC scatterometer path.
        """
        end_dt = _parse_iso_dt(normalize_datetime(end))

        cutoff = datetime.now(timezone.utc) - timedelta(days=_MAX_AGE_DAYS)
        if end_dt < cutoff:
            logger.warning(
                "Skipping %s FTP scatterometer download: end date %s is more than "
                "%d days old. The OSI-SAF wind FTP server only retains a rolling "
                "%d-day window.",
                self.satellite, end, _MAX_AGE_DAYS, _MAX_AGE_DAYS,
            )
            return []

        # If end was a date-only string, expand the matching-window bound to
        # include the full day (files can be timestamped later than
        # midnight on the end date). This expanded value is only used for
        # the file-matching comparison below, never for the recency guard
        # above, which must use the plain, un-expanded end_dt.
        is_end_date_only = len(end.strip().rstrip("Z")) == 10  # YYYY-MM-DD
        end_dt_for_matching = end_dt + timedelta(days=1) if is_end_date_only else end_dt

        start_dt = _parse_iso_dt(normalize_datetime(start))
        path = _SATELLITE_PATHS[self.satellite]

        if self.dry_run:
            # Report real file availability when credentials are already
            # configured; authenticate_osi_saf_ftp never prompts
            # interactively (unlike G-Portal's SFTP path) -- it either
            # resolves credentials or raises RuntimeError immediately, so
            # there's no interactive-prompt risk to guard against here.
            try:
                username, password = authenticate_osi_saf_ftp(self._username, self._password)
            except RuntimeError:
                print(
                    "[DRY RUN] OSI-SAF FTP credentials are not configured -- run "
                    "`sar-validate --set-credential osi_saf` (or set "
                    "OSI_SAF_FTP_USERNAME / OSI_SAF_FTP_PASSWORD) to see which "
                    f"{self.satellite} files are available for "
                    f"[{start_dt}, {end_dt_for_matching}]."
                )
                return []
        else:
            username, password = authenticate_osi_saf_ftp(self._username, self._password)

        import ftplib

        ftp = ftplib.FTP(FTP_HOST, timeout=60)
        try:
            ftp.login(username, password)
            ftp.cwd(path)
            names = ftp.nlst()

            matches = []
            for name in names:
                if not _matches_25km(name):
                    continue
                ts = _parse_filename_timestamp(name)
                if ts is None:
                    continue
                # If end was date-only, end_dt_for_matching was expanded to
                # the next day, so use a strict upper bound. If end was a
                # full datetime, use an inclusive upper bound.
                end_check = ts < end_dt_for_matching if is_end_date_only else ts <= end_dt_for_matching
                if not (start_dt <= ts and end_check):
                    continue
                matches.append(name)

            if self.orbit_prefilter:
                matches = self._filter_by_orbit_overlap(matches, min_lon, max_lon, min_lat, max_lat)

            print(f"Found {len(matches)} {self.satellite} 25 km file(s) in window.")
            if not matches:
                return []

            if self.dry_run:
                print("[DRY RUN] Would download:")
                for name in sorted(matches):
                    print(f"  {name}")
                return []

            self.output_dir.mkdir(parents=True, exist_ok=True)
            downloaded: list[Path] = []
            for name in sorted(matches):
                fetched = self._fetch_one(ftp, name)
                if fetched is not None:
                    downloaded.append(fetched)
        finally:
            try:
                ftp.quit()
            except Exception:
                ftp.close()

        print(f"Downloaded {len(downloaded)} {self.satellite} file(s).")
        return downloaded

    def _filter_by_orbit_overlap(
        self, matches: list[str], min_lon: float, max_lon: float, min_lat: float, max_lat: float,
    ) -> list[str]:
        """Drop filenames whose padded sensing window shows no predicted
        orbit overlap with the requested bbox -- see
        orbit_coverage.orbit_overlaps_bbox. Fails open per-file: any
        prediction failure inside orbit_overlaps_bbox itself already
        returns True (never raises), so this method never needs its own
        try/except."""
        from ..core.orbit_coverage import orbit_overlaps_bbox

        kept = []
        dropped = 0
        for name in matches:
            start = _parse_filename_timestamp(name)
            assert start is not None  # matches was already filtered to only parseable timestamps
            end = start + _ASSUMED_PASS_DURATION
            if orbit_overlaps_bbox(self.satellite, start, end, min_lon, max_lon, min_lat, max_lat):
                kept.append(name)
            else:
                dropped += 1
        if dropped:
            print(f"Orbit pre-filter: skipped {dropped} file(s) with no predicted overlap.")
        return kept

    def _fetch_one(self, ftp, name: str) -> Optional[Path]:
        is_gz = name.endswith(".gz")
        final_name = name[:-3] if is_gz else name
        final_path = self.output_dir / final_name

        if not self.force_download and final_path.exists():
            print(f"  Already downloaded: {final_path}")
            return final_path

        raw_path = self.output_dir / name
        print(f"  Downloading {name} …")
        with open(raw_path, "wb") as fdst:
            ftp.retrbinary("RETR " + name, fdst.write)

        if is_gz:
            with open(final_path, "wb") as f_out, gzip.open(raw_path, "rb") as f_in:
                shutil.copyfileobj(f_in, f_out)
            raw_path.unlink()

        print(f"  Saved to {final_path}")
        return final_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download 25 km scatterometer wind data from the OSI-SAF wind FTP server.",
    )
    p.add_argument("--satellite", required=True, choices=sorted(_SATELLITE_PATHS))
    p.add_argument("--params-file", metavar="FILE")
    p.add_argument("--min-lon", type=float)
    p.add_argument("--max-lon", type=float)
    p.add_argument("--min-lat", type=float)
    p.add_argument("--max-lat", type=float)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--username", default=None)
    p.add_argument("--password", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--no-orbit-prefilter", dest="orbit_prefilter", action="store_false", default=True,
        help="Disable the orbit-based geographic pre-filter (default: enabled).",
    )
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
        start   = params["start_datetime"]
        end     = params["end_datetime"]
    else:
        for attr in ("min_lon", "max_lon", "min_lat", "max_lat", "start", "end"):
            if getattr(args, attr) is None:
                print(f"Error: --{attr.replace('_','-')} is required (or use --params-file)")
                sys.exit(1)
        min_lon, max_lon = args.min_lon, args.max_lon
        min_lat, max_lat = args.min_lat, args.max_lat
        start, end = args.start, args.end

    output_dir = Path(args.output_dir) if args.output_dir else (
        build_output_dir(start, end, min_lon, max_lon, min_lat, max_lat)
        / f"scatterometer_{args.satellite}"
    )

    dl = ScatterometerFTPDownloader(
        satellite=args.satellite,
        output_dir=output_dir,
        dry_run=args.dry_run,
        username=args.username,
        password=args.password,
        orbit_prefilter=args.orbit_prefilter,
    )
    dl.download(
        min_lon=min_lon, max_lon=max_lon,
        min_lat=min_lat, max_lat=max_lat,
        start=start, end=end,
    )


if __name__ == "__main__":
    main()
