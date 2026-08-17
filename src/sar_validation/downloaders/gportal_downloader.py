"""
Download AMSR2 soil-moisture products from JAXA G-Portal via SFTP.

NASA Earthdata's AMSR2 soil-moisture coverage (NSIDC-0451 / its
replacement AU_Land_NRT_R02, see earthdata_soil_moisture_downloader.py)
is frozen at 2025-09-01 (NSIDC stopped processing AMSR Unified data
sets) -- there is no earthaccess path for any date after that. JAXA
distributes AMSR2 (their own instrument, on GCOM-W1) directly with much
lower latency via SFTP.

Host/port/auth confirmed directly from the G-Portal (General) User's
manual, section 3.3.4 "How to download using SFTP": host
ftp.gportal.jaxa.jp, port 2051, protocol SFTP, account+password
authentication (no SSH key registration available for this account).
Directory layout confirmed from the same manual, section 3.1.1
"Directory structure":
  standard/[Project]/[Satellite.Sensor]/[Product Name]/[Version]/[Year]/[Month]/
  nrt/[Project]/[Satellite.Sensor]/[Product Name]/  (flat, ~1 week retention)

The manual is JAXA-generic and never names AMSR2's soil-moisture product
directory directly (it illustrates the pattern with GPM examples only)
-- this downloader therefore *discovers* the right directory by listing
and pattern-matching directory names, logging every directory it finds
along the way, rather than hardcoding a guessed path.

Library usage::

    from sar_validation.downloaders.gportal_downloader import GPortalAMSR2Downloader
    dl = GPortalAMSR2Downloader(output_dir=Path("data/run1/amsr_ssm"))
    dl.download(min_lon=-10, max_lon=10, min_lat=40, max_lat=55,
                start="2026-07-01", end="2026-07-02")

CLI usage::

    python -m sar_validation.downloaders.gportal_downloader \\
        --min-lon -10 --max-lon 10 --min-lat 40 --max-lat 55 \\
        --start 2026-07-01 --end 2026-07-02
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

from .base import authenticate_gportal, build_output_dir, normalize_datetime

logger = logging.getLogger(__name__)

__all__ = ["GPortalAMSR2Downloader"]

HOST = "ftp.gportal.jaxa.jp"
PORT = 2051

#: A transient connection-level failure (observed live: "Error reading SSH
#: protocol banner" -- the TCP handshake succeeds but the server closes or
#: goes silent before sending its SSH banner) gets one retry after a brief
#: backoff, rather than aborting this whole best-effort fallback source
#: outright. A bare reconnect moments later has reliably succeeded when this
#: was observed, so a short retry is worth it before giving up.
_CONNECT_MAX_ATTEMPTS = 2
_CONNECT_RETRY_BACKOFF_SECONDS = 2.0

_TOP_LEVEL_DIRS = ("standard", "nrt")
_SENSOR_NAME_PATTERN = re.compile(r"amsr2|gcom-w", re.IGNORECASE)
# "sm" must stand on its own as a token (e.g. "L3.SM_STD"), not be part of a
# longer word -- \b won't do this because "_" counts as a word character in
# Python's default \w, so "SM_STD" has no \b between "M" and "_". Use
# explicit letter-based lookarounds instead so "SM" followed by "_"/"."/digits
# still counts as a standalone token while "SMALL"/"OSMOSIS" etc. don't match.
_PRODUCT_NAME_PATTERN = re.compile(r"(?<![a-z])sm(?![a-z])|soil|smc", re.IGNORECASE)
_FILENAME_DATE_RE = re.compile(r"(\d{8})")
# G-Portal's standard/ tree mixes daily granules ("..._01D_...") with
# whole-month composite files ("..._01M_...") in the same Year/Month
# listing -- both can carry an embedded date that matches the requested
# window below, since a monthly file's date uses day="00" as a
# placeholder for "the whole month" (e.g. "20260100"), which the plain
# lexicographic start_date <= date <= end_date comparison can't tell
# apart from a real day once collocation-tolerance padding pushes the
# window's start into the previous month. This downloader/
# DataTreeConverter.from_amsr_ssm's whole pipeline is built for daily L3
# grids only -- a monthly file has a different HDF5 group layout (no
# "Time Information" group) that from_amsr_ssm can't parse, confirmed
# live: it falls through to the unrelated NSIDC-0451 branch and logs
# "Missing vsm/longitude/latitude field(s)" before being silently
# dropped.
_NON_DAILY_AGGREGATION_RE = re.compile(r"_\d{2}M_")


def _connect_with_retry(username: str, password: str):
    """
    Open a socket to G-Portal's SFTP server and authenticate, retrying up
    to :data:`_CONNECT_MAX_ATTEMPTS` times (with a
    :data:`_CONNECT_RETRY_BACKOFF_SECONDS` pause between attempts) on a
    transient connection-level failure -- ``paramiko.SSHException``
    (covers "Error reading SSH protocol banner"), ``OSError``, or
    ``EOFError``. Any other exception (e.g. a genuine auth failure)
    propagates immediately, unretried.

    Returns
    -------
    (paramiko.Transport, paramiko.SFTPClient)
    """
    import socket
    import time

    import paramiko

    for attempt in range(1, _CONNECT_MAX_ATTEMPTS + 1):
        transport = None
        try:
            sock = socket.create_connection((HOST, PORT), timeout=30)
            transport = paramiko.Transport(sock)
            transport.connect(username=username, password=password)
            sftp = paramiko.SFTPClient.from_transport(transport)
            return transport, sftp
        except Exception as exc:
            if transport is not None:
                transport.close()
            retryable = isinstance(exc, (paramiko.SSHException, OSError, EOFError))
            if retryable and attempt < _CONNECT_MAX_ATTEMPTS:
                logger.warning(
                    "G-Portal SFTP connect attempt %d/%d failed (%s) — retrying in %.0fs…",
                    attempt, _CONNECT_MAX_ATTEMPTS, exc, _CONNECT_RETRY_BACKOFF_SECONDS,
                )
                time.sleep(_CONNECT_RETRY_BACKOFF_SECONDS)
                continue
            raise
    raise AssertionError("unreachable")  # pragma: no cover


class GPortalAMSR2Downloader:
    """
    Download AMSR2 soil-moisture products from JAXA G-Portal via SFTP.

    Parameters
    ----------
    output_dir : Path
        Directory to save downloaded files.
    dry_run : bool
        If True and credentials are already configured, connect and report
        real matching file availability without downloading anything; if
        credentials are not configured, print a message to set them up
        instead of prompting interactively (a dry run never blocks on
        input).
    force_download : bool
        If True, re-download files even if already present in output_dir.
    username, password : str, optional
        G-Portal credentials. If omitted, resolved via authenticate_gportal().
    allow_prompt : bool
        Passed through to ``authenticate_gportal()``. Defaults to True,
        preserving today's interactive behavior for direct/CLI use of this
        downloader (see ``main()``). Automated callers -- e.g. the
        orchestrator's AMSR2 G-Portal fallback, which runs unattended --
        must pass False so a missing credential raises instead of hanging
        on an interactive password prompt.
    orbit_prefilter : bool
        When True (default), drop files whose embedded date's whole-day
        window shows no predicted orbit overlap with the requested bbox
        (see orbit_coverage.orbit_overlaps_bbox) before downloading them.
    """

    def __init__(
        self,
        output_dir: Path,
        dry_run: bool = False,
        force_download: bool = False,
        username: Optional[str] = None,
        password: Optional[str] = None,
        allow_prompt: bool = True,
        orbit_prefilter: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.force_download = force_download
        self._username = username
        self._password = password
        self._allow_prompt = allow_prompt
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
        Discover and download AMSR2 soil-moisture files whose embedded
        date falls in [start, end].

        min_lon/max_lon/min_lat/max_lat are accepted for interface
        consistency with every other downloader but not used for
        server-side filtering -- SFTP has no spatial query. Domain
        cropping is a converter-layer concern (out of scope here; see
        the design doc's "Out of scope" section).
        """
        start_dt = normalize_datetime(start)
        end_dt = normalize_datetime(end)

        if self.dry_run:
            # Report real file availability when credentials are already
            # configured -- but never prompt for one during a "dry" run
            # (allow_prompt=False regardless of self._allow_prompt): a
            # dry-run must never block on interactive input.
            try:
                username, password = authenticate_gportal(
                    self._username, self._password, allow_prompt=False,
                )
            except RuntimeError:
                print(
                    "[DRY RUN] G-Portal credentials are not configured -- run "
                    "`sar-validate --set-credential gportal` (or set "
                    "GPORTAL_USERNAME / GPORTAL_PASSWORD) to see which AMSR2 "
                    f"soil-moisture files are available for [{start_dt}, {end_dt}]."
                )
                return []
        else:
            username, password = authenticate_gportal(
                self._username, self._password, allow_prompt=self._allow_prompt,
            )

        transport = None
        sftp = None
        try:
            # paramiko.Transport.__init__/.connect() accept no timeout of
            # their own -- an unresponsive server would otherwise hang the
            # caller indefinitely. _connect_with_retry opens the socket
            # with a timeout first and hands it to Transport, mirroring
            # the same fix already applied to smos_downloader.py's
            # ftplib.FTP_TLS(..., timeout=60) -- and retries once on a
            # transient connection-level failure (e.g. "Error reading SSH
            # protocol banner", observed live against this exact server).
            transport, sftp = _connect_with_retry(username, password)
            product_dirs = self._discover_product_directory(sftp)
            if not product_dirs:
                return []
            # Try every confidently-matched product directory in order
            # (standard/ before nrt/, per _discover_product_directory), not
            # just the first: the product directory itself typically
            # exists under both trees on a real account, but nrt/'s
            # ~1-week-retention tree may hold recent files not yet
            # propagated to standard/'s Year/Month archive. Returning as
            # soon as standard/ comes up empty would make nrt/ -- the
            # whole reason this fallback exists -- structurally
            # unreachable in its own motivating scenario (a recent date
            # request that standard/ hasn't caught up to yet).
            downloaded: list[Path] = []
            for product_dir in product_dirs:
                downloaded = self._download_from_product_directory(
                    sftp, product_dir, start_dt, end_dt,
                    min_lon, max_lon, min_lat, max_lat,
                )
                if downloaded:
                    return downloaded
            return downloaded
        finally:
            if sftp is not None:
                sftp.close()
            if transport is not None:
                transport.close()

    def _discover_product_directory(self, sftp) -> list[str]:
        """
        Search standard/ and nrt/ for a directory two levels down whose
        name matches an AMSR2/soil-moisture heuristic.

        Returns every confidently-matched product directory found, in
        top-level order (``standard`` before ``nrt``) -- not just the
        first match. The product directory itself typically exists under
        both trees on a real account; it's specific *files* for a recent
        date that may only be in nrt/'s ~1-week-retention tree, not yet
        propagated to standard/'s Year/Month archive. A caller that stops
        at the first match would only ever consult nrt/ when standard/
        has no AMSR2 product directory at all -- which isn't true on a
        real account -- making the nrt/ fallback unreachable in the exact
        scenario (a recent date past standard/'s archive lag) it exists
        to serve. See ``download()`` for how callers use this list.

        Logs every directory name seen so a real run against a real
        account leaves a usable trail even when the heuristic doesn't
        match anywhere.
        """
        found_listings: dict[str, list[str]] = {}
        product_dirs: list[str] = []

        for top in _TOP_LEVEL_DIRS:
            product_dirs.extend(self._discover_in_top(sftp, top, found_listings))

        if not product_dirs:
            print(
                "G-Portal: could not confidently identify the AMSR2 soil-moisture "
                "product directory. Directories found:"
            )
            for path, names in found_listings.items():
                print(f"  {path}: {names}")

        return product_dirs

    def _discover_in_top(
        self, sftp, top: str, found_listings: dict[str, list[str]],
    ) -> list[str]:
        """
        Return every confidently-matched AMSR2 soil-moisture product
        directory found under top-level tree *top* (``"standard"`` or
        ``"nrt"``) -- not just the first.

        A real G-Portal account can list MULTIPLE sensor directories
        under one top-level tree that match ``_SENSOR_NAME_PATTERN``,
        and not all of them are real AMSR2 data: confirmed on a real
        account, ``standard/AQUA/AQUA.AMSR-E_AMSR2Format`` matches the
        "amsr2" pattern via its own literal filename component, but it's
        AMSR-E -- a different, retired-since-2011 instrument, reformatted
        to look like AMSR2's file layout for cross-instrument continuity
        -- and its ``L3.SMC_10`` archive only spans 2002-2011. Since the
        server lists ``AQUA`` before the genuine ``GCOM-W`` project,
        stopping at the first sensor match made
        ``standard/GCOM-W/GCOM-W.AMSR2`` (which does have current data)
        structurally unreachable for any date after 2011. Returning every
        match instead lets ``download()``'s existing "try each until one
        yields files" loop fall through the decoy to the real sensor.

        Populates *found_listings* (mutated in place) with every
        directory listing seen along the way, for
        ``_discover_product_directory``'s "no match anywhere" diagnostic.
        """
        try:
            sensors_parent_names = sftp.listdir(top)
        except IOError:
            return []
        found_listings[top] = sensors_parent_names
        logger.info("G-Portal: %s/ contains %s", top, sensors_parent_names)

        product_dirs: list[str] = []
        for project in sensors_parent_names:
            project_path = f"{top}/{project}"
            try:
                sensor_names = sftp.listdir(project_path)
            except IOError:
                continue
            found_listings[project_path] = sensor_names
            logger.info("G-Portal: %s contains %s", project_path, sensor_names)

            matching_sensors = [s for s in sensor_names if _SENSOR_NAME_PATTERN.search(s)]
            for sensor in matching_sensors:
                sensor_path = f"{project_path}/{sensor}"
                try:
                    product_names = sftp.listdir(sensor_path)
                except IOError:
                    continue
                found_listings[sensor_path] = product_names
                logger.info("G-Portal: %s contains %s", sensor_path, product_names)

                matching_products = [p for p in product_names if _PRODUCT_NAME_PATTERN.search(p)]
                if matching_products:
                    product_path = f"{sensor_path}/{matching_products[0]}"
                    print(f"G-Portal: discovered AMSR2 soil-moisture product directory: {product_path}")
                    product_dirs.append(product_path)
        return product_dirs

    def _download_from_product_directory(
        self, sftp, product_dir: str, start_dt: str, end_dt: str,
        min_lon: float, max_lon: float, min_lat: float, max_lat: float,
    ) -> list[Path]:
        is_nrt = product_dir.startswith("nrt/")

        if is_nrt:
            try:
                filenames = sftp.listdir(product_dir)
            except IOError:
                return []
            candidates = [(product_dir, name) for name in filenames]
        else:
            candidates = self._standard_tree_candidates(sftp, product_dir, start_dt, end_dt)

        start_date = start_dt[:10].replace("-", "")
        end_date = end_dt[:10].replace("-", "")

        matches = []
        for dir_path, name in candidates:
            if _NON_DAILY_AGGREGATION_RE.search(name):
                continue
            m = _FILENAME_DATE_RE.search(name)
            if m and start_date <= m.group(1) <= end_date:
                matches.append((dir_path, name))

        if self.orbit_prefilter:
            matches = self._filter_by_orbit_overlap(matches, min_lon, max_lon, min_lat, max_lat)

        print(f"Found {len(matches)} AMSR2 file(s) in window.")
        if not matches:
            return []

        if self.dry_run:
            print("[DRY RUN] Would download:")
            for _dir_path, name in matches:
                print(f"  {name}")
            return []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []
        for dir_path, name in matches:
            local_path = self.output_dir / name
            if not self.force_download and local_path.exists():
                print(f"  Already downloaded: {local_path}")
                downloaded.append(local_path)
                continue
            remote_path = f"{dir_path}/{name}"
            print(f"  Downloading {remote_path} …")
            sftp.get(remote_path, str(local_path))
            downloaded.append(local_path)

        print(f"Downloaded {len(downloaded)} AMSR2 file(s).")
        return downloaded

    def _filter_by_orbit_overlap(
        self, matches: list[tuple[str, str]], min_lon: float, max_lon: float, min_lat: float, max_lat: float,
    ) -> list[tuple[str, str]]:
        """Drop (dir_path, name) entries whose embedded date's whole-day
        window shows no predicted orbit overlap with the requested bbox
        -- see orbit_coverage.orbit_overlaps_bbox. Every entry here
        already matched _FILENAME_DATE_RE by construction (see
        _download_from_product_directory's matches-building loop), so
        re-matching it is guaranteed to succeed.

        Filenames only embed a date, not a time-of-day, so the whole day
        [00:00:00Z, 23:59:59Z] is used as the sensing window -- expected
        to rarely filter anything out, since AMSR2's near-global daily
        coverage means most days overlap most bboxes. Kept for
        consistency with the other two orbit-prefiltered sources."""
        from datetime import datetime, timedelta, timezone

        from ..core.orbit_coverage import orbit_overlaps_bbox

        kept = []
        dropped = 0
        for dir_path, name in matches:
            match = _FILENAME_DATE_RE.search(name)
            assert match is not None  # matches was already filtered to only entries where this matched
            day_start = datetime.strptime(match.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)
            day_end = day_start + timedelta(hours=23, minutes=59, seconds=59)
            if orbit_overlaps_bbox("gcom-w1", day_start, day_end, min_lon, max_lon, min_lat, max_lat):
                kept.append((dir_path, name))
            else:
                dropped += 1
        if dropped:
            print(f"Orbit pre-filter: skipped {dropped} file(s) with no predicted overlap.")
        return kept

    def _standard_tree_candidates(
        self, sftp, product_dir: str, start_dt: str, end_dt: str,
    ) -> list[tuple[str, str]]:
        """
        standard/ nests Version/Year/Month below the product directory.
        Version is unknown ahead of time (per the manual's diagram) --
        list whatever's there and descend into every version found,
        restricting Year/Month to the requested window.
        """
        start_year = int(start_dt[:4])
        end_year = int(end_dt[:4])
        start_month = int(start_dt[5:7])
        end_month = int(end_dt[5:7])

        candidates: list[tuple[str, str]] = []
        try:
            versions = sftp.listdir(product_dir)
        except IOError:
            return candidates

        for version in versions:
            version_path = f"{product_dir}/{version}"
            try:
                years = sftp.listdir(version_path)
            except IOError:
                continue
            for year in years:
                if not year.isdigit() or not (start_year <= int(year) <= end_year):
                    continue
                year_path = f"{version_path}/{year}"
                try:
                    months = sftp.listdir(year_path)
                except IOError:
                    continue
                for month in months:
                    if not month.isdigit():
                        continue
                    if int(year) == start_year and int(month) < start_month:
                        continue
                    if int(year) == end_year and int(month) > end_month:
                        continue
                    month_path = f"{year_path}/{month}"
                    try:
                        files = sftp.listdir(month_path)
                    except IOError:
                        continue
                    candidates.extend((month_path, name) for name in files)
        return candidates


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download AMSR2 soil-moisture data from JAXA G-Portal via SFTP.",
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
        build_output_dir(start, end, min_lon, max_lon, min_lat, max_lat) / "amsr_ssm_gportal"
    )

    dl = GPortalAMSR2Downloader(
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
