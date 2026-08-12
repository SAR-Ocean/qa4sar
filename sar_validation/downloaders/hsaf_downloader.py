"""
Download H-SAF ASCAT Surface Soil Moisture NRT (H29, 12.5km) products from
the H-SAF FTP server (ftphsaf.meteoam.it).

This is the replacement for dates after the EUMDAC/SOMO12 downloader's
coverage cutoff (2025-07-15, see ascat_soil_moisture_downloader.py) --
that downloader is kept unchanged for historical dates; this one only
covers H-SAF's on-line NRT archive.

Live-confirmed (real downloaded file, 2026-08-12): the FTP directory
/h29/h29_cur_mon_nc/ holds a rolling *last-60-days* on-line NRT archive
(confirmed by the user directly; the directory's own name is misleading
here) -- one file per ~3-minute orbit segment, not a daily composite.
Real filename:
  W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOPB-12.5km-H29_C_LIIB_20260609001514_20260608231200_20260608231459____.nc
Pattern: W_IT-HSAF-ROME,SAT,SSM-ASCAT-<SAT>-12.5km-H29_C_LIIB_<created>_<sensing_start>_<sensing_end>____.nc,
all timestamps YYYYMMDDHHMMSS. Plain FTP (port 21, ftplib) -- confirmed
live via a raw socket connect, server banner "220 Welcome to Italian Air
Force Meteorological Service H-SAF FTP service."

No server-side bbox filtering exists (full-orbit swaths only) -- bbox is
accepted for interface consistency with every other downloader but not
used to filter here; domain cropping happens downstream in
DataTreeConverter.convert_downloaded_data's recipe-domain filter, same
as scatterometer_ftp_downloader.py.

Library usage::

    from sar_validation.downloaders.hsaf_downloader import HSAFDownloader
    dl = HSAFDownloader(output_dir=Path("data/run1/hsaf_ssm"))
    dl.download(min_lon=-70, max_lon=-30, min_lat=50, max_lat=67,
                start="2026-06-08", end="2026-06-09")

CLI usage::

    python -m sar_validation.downloaders.hsaf_downloader \\
        --min-lon -70 --max-lon -30 --min-lat 50 --max-lat 67 \\
        --start 2026-06-08 --end 2026-06-09
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .base import authenticate_hsaf_ftp, build_output_dir, normalize_datetime

__all__ = ["HSAFDownloader"]

FTP_HOST = "ftphsaf.meteoam.it"
FTP_PATH = "/h29/h29_cur_mon_nc"

_SENSING_START_RE = re.compile(r"H29_C_LIIB_\d{14}_(\d{14})_\d{14}____\.nc$")


def _matches_h29_nc(filename: str) -> bool:
    """True if filename is an H29 netCDF product file (not its .md5 sidecar)."""
    return bool(_SENSING_START_RE.search(filename))


def _parse_sensing_start(filename: str) -> Optional[datetime]:
    """Extract the embedded sensing-start timestamp from an H29 filename."""
    m = _SENSING_START_RE.search(filename)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_iso_dt(s: str) -> datetime:
    """Convert an ISO datetime string (from normalize_datetime) to a UTC-aware datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=timezone.utc)


class HSAFDownloader:
    """
    Download H-SAF ASCAT Surface Soil Moisture NRT (H29) files from the
    H-SAF on-line FTP archive.

    Only the rolling last-60-days on-line directory is queried -- H-SAF's
    off-line/CDR archive (order-based access) is out of scope. A
    request for dates older than 60 days simply returns no files;
    callers wanting older data use ascat_soil_moisture_downloader.py
    (up to its 2025-07-15 cutoff) or accept the gap.

    Parameters
    ----------
    output_dir : Path
        Directory to save downloaded files.
    dry_run : bool
        If True, print what would be downloaded without actually downloading.
    force_download : bool
        Re-download even if a same-named file already exists locally.
    username, password : str, optional
        H-SAF FTP credentials. If omitted, resolved via authenticate_hsaf_ftp().
    """

    def __init__(
        self,
        output_dir: Path,
        dry_run: bool = False,
        force_download: bool = False,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.force_download = force_download
        self._username = username
        self._password = password

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
        Download every H29 file whose sensing-start timestamp falls in
        [start, end]. bbox is accepted for interface consistency but not
        used to filter (see module docstring).
        """
        start_dt = _parse_iso_dt(normalize_datetime(start))
        end_dt = _parse_iso_dt(normalize_datetime(end))
        # A date-only `end` (e.g. "2026-06-09") normalizes to midnight;
        # expand it to the full day so files timestamped later than
        # midnight on the end date still match, mirroring
        # scatterometer_ftp_downloader.py's identical handling.
        is_end_date_only = len(end.strip().rstrip("Z")) == 10
        end_dt_for_matching = end_dt + timedelta(days=1) if is_end_date_only else end_dt

        username, password = authenticate_hsaf_ftp(self._username, self._password)

        import ftplib

        ftp = ftplib.FTP(FTP_HOST, timeout=60)
        try:
            ftp.login(username, password)
            ftp.cwd(FTP_PATH)
            names = ftp.nlst()

            matches = []
            for name in names:
                if not _matches_h29_nc(name):
                    continue
                ts = _parse_sensing_start(name)
                if ts is None:
                    continue
                end_check = ts < end_dt_for_matching if is_end_date_only else ts <= end_dt_for_matching
                if not (start_dt <= ts and end_check):
                    continue
                matches.append(name)

            print(f"Found {len(matches)} H-SAF H29 file(s) in window.")
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

        print(f"Downloaded {len(downloaded)} H-SAF H29 file(s).")
        return downloaded

    def _fetch_one(self, ftp, name: str) -> Optional[Path]:
        final_path = self.output_dir / name

        if not self.force_download and final_path.exists():
            print(f"  Already downloaded: {final_path}")
            return final_path

        print(f"  Downloading {name} …")
        with open(final_path, "wb") as fdst:
            ftp.retrbinary("RETR " + name, fdst.write)

        print(f"  Saved to {final_path}")
        return final_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download H-SAF ASCAT SSM NRT (H29) data from the H-SAF FTP server.",
    )
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
        build_output_dir(start, end, min_lon, max_lon, min_lat, max_lat) / "hsaf_ssm"
    )

    dl = HSAFDownloader(
        output_dir=output_dir,
        dry_run=args.dry_run,
        username=args.username,
        password=args.password,
    )
    dl.download(
        min_lon=min_lon, max_lon=max_lon,
        min_lat=min_lat, max_lat=max_lat,
        start=start, end=end,
    )


if __name__ == "__main__":
    main()
