"""
Download H-SAF ASCAT Surface Soil Moisture NRT products from the H-SAF
FTP server (ftphsaf.meteoam.it). Two products are available: H122
(6.25km sampling, the default) and H29 (12.5km, opt-in via product="h29"
within the recipe). One file per ~3-minute orbit segment, not a daily composite. 

Filename examples:
  W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOPB-12.5km-H29_C_LIIB_20260609001514_20260608231200_20260608231459____.nc
  W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOPB-6.25km-H122_C_LIIB_20260609001515_20260608231200_20260608231459____.nc
Pattern: W_IT-HSAF-ROME,SAT,SSM-ASCAT-<SAT>-<res>km-<product>_C_LIIB_<created>_<sensing_start>_<sensing_end>____.nc,
all timestamps YYYYMMDDHHMMSS. Plain FTP (port 21, ftplib).

No server-side bbox filtering exists (full-orbit swaths only) -- bbox is
accepted for interface consistency with every other downloader but not
used to filter here; domain cropping happens downstream in
DataTreeConverter.convert_downloaded_data's recipe-domain filter, same
as scatterometer_ftp_downloader.py.

Library usage::

    from sar_validation.downloaders.hsaf_downloader import HSAFDownloader
    dl = HSAFDownloader(output_dir=Path("data/run1/hsaf_ascat_ssm"))  # H122 by default
    dl.download(min_lon=-70, max_lon=-30, min_lat=50, max_lat=67,
                start="2026-06-08", end="2026-06-09")
    dl29 = HSAFDownloader(output_dir=Path("data/run1/hsaf_ascat_ssm"), product="h29")

CLI usage::

    python -m sar_validation.downloaders.hsaf_downloader \\
        --min-lon -70 --max-lon -30 --min-lat 50 --max-lat 67 \\
        --start 2026-06-08 --end 2026-06-09 --product h122
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

#: FTP directory per product:
_PRODUCT_PATHS = {
    "h29": "/h29/h29_cur_mon_nc",
    "h122": "/h122/h122_cur_mon_nc",
}

_SENSING_START_RE = re.compile(r"H\d+_C_LIIB_\d{14}_(\d{14})_\d{14}____\.nc$")


def _matches_ascat_nc(filename: str) -> bool:
    """
    True if filename is an ASCAT SSM netCDF product file -- H29 or
    H122, not a .md5 sidecar. Resolution itself is not derived here (see
    datatree_converter.py's _parse_ascat_resolution_km). This only
    matches the filename shape and extracts the sensing-start timestamp.
    """
    return bool(_SENSING_START_RE.search(filename))


def _parse_sensing_start(filename: str) -> Optional[datetime]:
    """
    Extract the embedded sensing-start timestamp from an H29 or H122 filename.
    """
    m = _SENSING_START_RE.search(filename)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


_SATELLITE_RE = re.compile(r"SSM-ASCAT-(METOP[ABC])-")


def _parse_satellite(filename: str) -> Optional[str]:
    """
    Extract the embedded satellite name from an H29/H122 filename,
    normalized to orbit_coverage.py's SATELLITE_ORBIT_SPECS key format
    (e.g. "METOPB" -> "metop-b"), or None if unparseable.
    """
    m = _SATELLITE_RE.search(filename)
    if not m:
        return None
    raw = m.group(1)
    return f"metop-{raw[-1].lower()}"


_SENSING_END_RE = re.compile(r"H\d+_C_LIIB_\d{14}_\d{14}_(\d{14})____\.nc$")


def _parse_sensing_end(filename: str) -> Optional[datetime]:
    """
    Extract the embedded sensing-end timestamp from an H29/H122 filename.
    """
    m = _SENSING_END_RE.search(filename)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_iso_dt(s: str) -> datetime:
    """
    Convert an ISO datetime string (from normalize_datetime) to a UTC-aware datetime.
    """
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=timezone.utc)


class HSAFDownloader:
    """
    Download H-SAF ASCAT Surface Soil Moisture NRT files (H122 6.25km by
    default, or H29 12.5km) from the H-SAF on-line FTP archive.

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
    product : str
        "h122" (default, 6.25km) or "h29" (12.5km, legacy resolution).
    orbit_prefilter : bool
        When True (default), drop files whose embedded satellite/sensing
        window shows no predicted orbit overlap with the requested bbox
        (see orbit_coverage.orbit_overlaps_bbox) before downloading them.
        Fails open per-file -- an unparseable filename, unregistered
        satellite, or any prediction failure keeps the file, never drops
        it.
    """

    def __init__(
        self,
        output_dir: Path,
        dry_run: bool = False,
        force_download: bool = False,
        username: Optional[str] = None,
        password: Optional[str] = None,
        product: str = "h122",
        orbit_prefilter: bool = True,
    ) -> None:
        if product not in _PRODUCT_PATHS:
            raise ValueError(
                f"Unknown H-SAF product {product!r}. Valid: {sorted(_PRODUCT_PATHS)}"
            )
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.force_download = force_download
        self._username = username
        self._password = password
        self.product = product
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
        Download every file of this instance's product whose sensing-start
        timestamp falls in [start, end]. bbox is accepted for interface
        consistency but not used to filter (see module docstring).
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
            ftp.cwd(_PRODUCT_PATHS[self.product])
            names = ftp.nlst()

            matches = []
            for name in names:
                if not _matches_ascat_nc(name):
                    continue
                ts = _parse_sensing_start(name)
                if ts is None:
                    continue
                end_check = ts < end_dt_for_matching if is_end_date_only else ts <= end_dt_for_matching
                if not (start_dt <= ts and end_check):
                    continue
                matches.append(name)

            if self.orbit_prefilter:
                matches = self._filter_by_orbit_overlap(matches, min_lon, max_lon, min_lat, max_lat)

            print(f"Found {len(matches)} H-SAF {self.product.upper()} file(s) in window.")
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

        print(f"Downloaded {len(downloaded)} H-SAF {self.product.upper()} file(s).")
        return downloaded

    def _filter_by_orbit_overlap(
        self, matches: list[str], min_lon: float, max_lon: float, min_lat: float, max_lat: float,
    ) -> list[str]:
        """
        Drop filenames whose embedded satellite/sensing-window orbit
        prediction shows no overlap with the requested bbox (+ margin) --
        see orbit_coverage.orbit_overlaps_bbox. Fails open per-file: an
        unparseable filename, unregistered satellite, or any prediction
        failure keeps the file, never drops it.
        """
        from ..core.orbit_coverage import orbit_overlaps_bbox

        kept = []
        dropped = 0
        for name in matches:
            satellite = _parse_satellite(name)
            start = _parse_sensing_start(name)
            end = _parse_sensing_end(name)
            if satellite is None or start is None or end is None:
                kept.append(name)
                continue
            if orbit_overlaps_bbox(satellite, start, end, min_lon, max_lon, min_lat, max_lat):
                kept.append(name)
            else:
                dropped += 1
        if dropped:
            print(f"Orbit pre-filter: skipped {dropped} file(s) with no predicted overlap.")
        return kept

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
        description="Download H-SAF ASCAT SSM NRT data from the H-SAF FTP server.",
    )
    p.add_argument("--product", choices=sorted(_PRODUCT_PATHS), default="h122")
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
        build_output_dir(start, end, min_lon, max_lon, min_lat, max_lat) / "hsaf_ascat_ssm"
    )

    dl = HSAFDownloader(
        output_dir=output_dir,
        dry_run=args.dry_run,
        username=args.username,
        password=args.password,
        product=args.product,
        orbit_prefilter=args.orbit_prefilter,
    )
    dl.download(
        min_lon=min_lon, max_lon=max_lon,
        min_lat=min_lat, max_lat=max_lat,
        start=start, end=end,
    )


if __name__ == "__main__":
    main()
