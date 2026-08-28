"""
Download AMSR2 soil-moisture products from JAXA G-Portal via SFTP.

NASA Earthdata's AMSR2 soil-moisture coverage (NSIDC-0451)
is frozen at 2025-09-01. JAXA distributes AMSR2 via SFTP.

Host, port, and authentication are specified in the G-Portal
(General) User's Manual, section 3.3.4 ("How to download using
SFTP"): host ``ftp.gportal.jaxa.jp``, port 2051, protocol SFTP,
account-and-password authentication (no SSH key registration is
available for this account). Directory layout is specified in the
same manual, section 3.1.1 ("Directory structure"):

  standard/[Project]/[Satellite.Sensor]/[Product Name]/[Version]/[Year]/[Month]/
  nrt/[Project]/[Satellite.Sensor]/[Product Name]/  (flat, ~1 week retention)

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
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from .base import authenticate_gportal, build_output_dir, normalize_datetime

logger = logging.getLogger(__name__)

__all__ = ["GPortalAMSR2Downloader"]

HOST = "ftp.gportal.jaxa.jp"
PORT = 2051

#: A transient connection-level failure ("Error reading SSH protocol banner" 
#: -- the TCP handshake succeeds but the server closes or goes silent before 
#: sending its SSH banner) gets one retry after a brief backoff.
_CONNECT_MAX_ATTEMPTS = 2
_CONNECT_RETRY_BACKOFF_SECONDS = 2.0

_TOP_LEVEL_DIRS = ("standard", "nrt")
_SENSOR_NAME_PATTERN = re.compile(r"amsr2|gcom-w", re.IGNORECASE)
_PRODUCT_NAME_PATTERN = re.compile(r"(?<![a-z])sm(?![a-z])|soil|smc", re.IGNORECASE)
#: AMSR-E's own reformatted-to-look-like-AMSR2 archive
#: (standard/AQUA/AQUA.AMSR-E_AMSR2Format) matches _SENSOR_NAME_PATTERN
#: via the literal "AMSR2" substring in its own directory name, even
#: though it is a different, retired instrument -- see _discover_in_top's
#: own docstring. This pattern identifies that decoy specifically, so it
#: can be excluded once the requested window is entirely after AMSR-E's
#: real data record ends.
_AMSR_E_DECOY_PATTERN = re.compile(r"amsr-e", re.IGNORECASE)
#: AMSR-E's antenna rotation mechanism failed on this date, ending its
#: operational data record.
_AMSR_E_RETIREMENT_DATE = "2011-10-04"
_FILENAME_DATE_RE = re.compile(r"(\d{8})")
# G-Portal's standard/ tree mixes daily granules ("..._01D_...") with
# whole-month composite files ("..._01M_...") in the same Year/Month
# listing. This downloader and DataTreeConverter.from_amsr_ssm's whole 
# pipeline are built for daily L3 grids only.
_NON_DAILY_AGGREGATION_RE = re.compile(r"_\d{2}M_")

#: list_candidates_dry's own shared cache of the RAW (i.e. not yet
#: filtered to an exact [start, end] day range) candidate list a
#: directory discovery + tree walk produces, keyed by
#: (start year-month, end year-month) rather than the exact requested
#: start/end -- SFTP has no server-side spatial or fine-grained temporal
#: query, only a Year/Month directory structure (see
#: _standard_tree_candidates), so any two calls whose requested window
#: falls within the same calendar month(s) need the identical
#: connection + directory walk; only the final per-file date filter
#: (cheap, local, applied by list_candidates_dry itself after a cache
#: hit or miss alike) differs per exact call. Without this,
#: --dry-collocation-detail's own per-footprint exhaustive scan
#: (_predict_orbit_corridor_source) opens a brand new SFTP connection
#: and re-walks the whole product-directory tree once per SAR
#: footprint -- for a recipe with more than a handful of footprints,
#: that dominates total runtime by a wide margin, and each footprint's
#: own connect attempt is an independent chance to hit a transient SFTP
#: failure.
_list_candidates_cache: "dict[tuple, list[tuple[str, str]]]" = {}

#: One lock per cache key (created lazily, guarded by
#: _list_candidates_locks_guard), not one lock for the whole cache --
#: mirrors InSituDownloader._fetch_stations_dry's identical pattern (see
#: its own module-level comment): the first caller for a given
#: year-month key holds that key's own lock for the duration of the real
#: SFTP work; concurrent callers for the SAME key block only on each
#: other, never on a caller whose window falls in different months.
_list_candidates_locks: "dict[tuple, threading.Lock]" = {}
_list_candidates_locks_guard = threading.Lock()

#: _discover_product_directory's own result (which top-level
#: Project/Sensor/Product directories under standard/ and nrt/ actually
#: hold AMSR2 data) does not depend on the exact requested start/end at
#: all, only on whether the AMSR-E decoy should be excluded (see
#: _AMSR_E_DECOY_PATTERN) -- it is otherwise a property of the SFTP
#: server's static directory tree, not of any particular query. Keyed by
#: exclude_amsr_e (so at most two distinct results are ever cached, not
#: one per exact query) and cached separately from _list_candidates_cache
#: (which is query-dependent, via its year-month key): the several
#: sequential listdir() round trips _discover_in_top makes to walk that
#: tree happen at most once per process per exclude_amsr_e value, even
#: across calls whose requested windows fall in different calendar
#: months and so would otherwise each pay for their own rediscovery of
#: the same, unchanging directory structure.
_product_dirs_cache: "dict[bool, list[str]]" = {}
_product_dirs_lock = threading.Lock()


def _get_list_candidates_lock(cache_key: tuple) -> "threading.Lock":
    with _list_candidates_locks_guard:
        lock = _list_candidates_locks.get(cache_key)
        if lock is None:
            lock = threading.Lock()
            _list_candidates_locks[cache_key] = lock
        return lock


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
        server-side filtering, since SFTP has no spatial query.
        """
        start_dt = normalize_datetime(start)
        end_dt = normalize_datetime(end)

        if self.dry_run:
            # Only report real file availability when credentials are 
            # already configured.
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
            # their own, meaning an unresponsive server would hang
            # indefinitely. _connect_with_retry opens the socket with a 
            # timeout first and hands it to Transport, and retries once on
            # a transient connection-level failure (e.g. "Error reading SSH
            # protocol banner").
            transport, sftp = _connect_with_retry(username, password)
            exclude_amsr_e = start_dt[:10] > _AMSR_E_RETIREMENT_DATE
            product_dirs = self._discover_product_directory(sftp, exclude_amsr_e=exclude_amsr_e)
            if not product_dirs:
                return []
            # Try every confidently-matched product directory in order
            # (standard/ before nrt/) -- see _discover_product_directory for why.
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

    def list_candidates_dry(
        self, min_lon: float, max_lon: float, min_lat: float, max_lat: float, start: str, end: str,
    ) -> "list[tuple[str, datetime, datetime]]":
        """(remote_path, day_start, day_end) for every AMSR2 file whose
        embedded date falls in [start, end] -- the same SFTP directory
        discovery and date-matching logic download()/
        _download_from_product_directory use, without the orbit
        prefilter (_filter_by_orbit_overlap) and without fetching
        anything. min_lon/max_lon/min_lat/max_lat are accepted for
        interface consistency but unused here for the same reason
        download() doesn't use them for server-side filtering (SFTP has
        no spatial query) -- geographic refinement is the caller's job
        (see dry_collocation._predict_orbit_corridor_source, which this
        feeds).

        Filenames only embed a date, not a time-of-day, so the whole UTC
        day is used as each match's sensing window, mirroring
        _filter_by_orbit_overlap's identical whole-day construction.

        Always authenticates with allow_prompt=False -- like download()'s
        own dry-run branch, a prediction call must never block on an
        interactive password prompt.

        The actual SFTP connect + directory discovery + tree walk is
        shared across every caller whose start/end fall in the same
        calendar month(s), via _list_candidates_cache -- see that
        cache's own module-level comment. Only the final per-file exact-
        date filter below runs on every call, uncached, since it is cheap
        and local once the raw listing is in hand.
        """
        from datetime import timedelta, timezone

        start_dt = normalize_datetime(start)
        end_dt = normalize_datetime(end)

        cache_key = (start_dt[:7], end_dt[:7])
        lock = _get_list_candidates_lock(cache_key)
        with lock:
            raw_candidates = _list_candidates_cache.get(cache_key)
            if raw_candidates is None:
                raw_candidates = self._discover_raw_candidates(start_dt, end_dt)
                _list_candidates_cache[cache_key] = raw_candidates

        start_date = start_dt[:10].replace("-", "")
        end_date = end_dt[:10].replace("-", "")

        candidates: "list[tuple[str, datetime, datetime]]" = []
        for dir_path, name in raw_candidates:
            if _NON_DAILY_AGGREGATION_RE.search(name):
                continue
            m = _FILENAME_DATE_RE.search(name)
            if not (m and start_date <= m.group(1) <= end_date):
                continue
            try:
                day_start = datetime.strptime(m.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            day_end = day_start + timedelta(hours=23, minutes=59, seconds=59)
            candidates.append((f"{dir_path}/{name}", day_start, day_end))
        return candidates

    def _discover_raw_candidates(self, start_dt: str, end_dt: str) -> "list[tuple[str, str]]":
        """The actual SFTP connect + product-directory discovery + tree
        walk list_candidates_dry shares via its own cache -- returns the
        (dir_path, filename) pairs found in the first product directory
        (in _discover_product_directory's own standard-before-nrt order)
        that has any, restricted only by _standard_tree_candidates' own
        coarse Year/Month bounds (the exact per-day filter is
        list_candidates_dry's job, applied after a cache hit or miss
        alike). Mirrors download()'s own "try each until one yields
        files" loop, so nrt/'s ~1-week-retention tree is only ever
        walked when standard/'s archive has nothing for this cached
        window -- for a window straddling a Year/Month boundary that
        gap could in principle miss a day standard/ lacks but nrt/ has,
        the same coarse-window tradeoff download() itself already
        accepts. Product-directory discovery itself is a second,
        separately-memoized step -- see _product_dirs_cache's own
        module-level comment for why."""
        username, password = authenticate_gportal(self._username, self._password, allow_prompt=False)

        transport = None
        sftp = None
        try:
            transport, sftp = _connect_with_retry(username, password)
            exclude_amsr_e = start_dt[:10] > _AMSR_E_RETIREMENT_DATE
            with _product_dirs_lock:
                cached_dirs = _product_dirs_cache.get(exclude_amsr_e)
                if cached_dirs is None:
                    cached_dirs = self._discover_product_directory(sftp, exclude_amsr_e=exclude_amsr_e)
                    _product_dirs_cache[exclude_amsr_e] = cached_dirs
                product_dirs = cached_dirs

            for product_dir in product_dirs:
                is_nrt = product_dir.startswith("nrt/")
                if is_nrt:
                    try:
                        filenames = sftp.listdir(product_dir)
                    except IOError:
                        continue
                    dir_candidates = [(product_dir, name) for name in filenames]
                else:
                    dir_candidates = self._standard_tree_candidates(sftp, product_dir, start_dt, end_dt)
                if dir_candidates:
                    print(
                        f"G-Portal: {product_dir} has {len(dir_candidates)} candidate(s) "
                        f"for {start_dt[:7]}..{end_dt[:7]} -- stopping here, not querying "
                        "any other discovered directory."
                    )
                    return dir_candidates
                print(f"G-Portal: {product_dir} has no candidates for {start_dt[:7]}..{end_dt[:7]} -- trying next.")
            return []
        finally:
            if sftp is not None:
                sftp.close()
            if transport is not None:
                transport.close()

    def _discover_product_directory(self, sftp, exclude_amsr_e: bool = False) -> list[str]:
        """
        Search standard/ and nrt/ for a directory two levels down whose
        name matches an AMSR2/soil-moisture heuristic.

        Returns every confidently-matched product directory found, in
        top-level order (``standard`` before ``nrt``), not just the first
        match, since a recent date's files may only be in nrt/'s ~1-week
        -retention tree, not yet propagated to standard/'s Year/Month archive.
        See ``download()`` for how callers use this list.

        exclude_amsr_e drops the AMSR-E decoy directory from the result --
        see _discover_in_top's own docstring for what it is and why a
        caller whose requested window is entirely after
        _AMSR_E_RETIREMENT_DATE passes True here, to skip listing a
        directory that can never hold real data for that window anyway.

        Logs every directory name seen, so a real run leaves a usable trail
        even when the heuristic matches nothing.
        """
        found_listings: dict[str, list[str]] = {}
        product_dirs: list[str] = []

        for top in _TOP_LEVEL_DIRS:
            product_dirs.extend(self._discover_in_top(sftp, top, found_listings, exclude_amsr_e))

        if not product_dirs:
            print(
                "G-Portal: could not confidently identify the AMSR2 soil-moisture "
                "product directory. Directories found:"
            )
            for path, names in found_listings.items():
                print(f"  {path}: {names}")

        return product_dirs

    def _discover_in_top(
        self, sftp, top: str, found_listings: dict[str, list[str]], exclude_amsr_e: bool = False,
    ) -> list[str]:
        """
        Return every confidently-matched AMSR2 soil-moisture product
        directory found under top-level tree *top* (``"standard"`` or
        ``"nrt"``) -- not just the first.

        A real G-Portal account can list multiple sensor directories
        under one top-level tree that match ``_SENSOR_NAME_PATTERN``,
        without all of them being real AMSR2 data: e.g.
        ``standard/AQUA/AQUA.AMSR-E_AMSR2Format`` matches via its own
        literal filename component, but it is AMSR-E (a retired, different
        instrument, reformatted to look like AMSR2's file layout), listed
        before the genuine ``standard/GCOM-W/GCOM-W.AMSR2``. When
        exclude_amsr_e is False (the default), this decoy is still
        returned -- ``download()``'s "try each until one yields files"
        loop falls through it to the real sensor -- so a caller with a
        genuinely historical (pre-retirement) window can still reach it.
        exclude_amsr_e=True drops it before it is even listed, since a
        caller whose window is entirely after
        _AMSR_E_RETIREMENT_DATE already knows it can never hold real
        data there.

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
            if exclude_amsr_e:
                matching_sensors = [s for s in matching_sensors if not _AMSR_E_DECOY_PATTERN.search(s)]
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
        """
        Drop (dir_path, name) entries whose embedded date's whole-day window
        shows no predicted orbit overlap with the requested bbox -- see
        orbit_coverage.orbit_overlaps_bbox. Filenames embed only a date, not
        a time of day, so the whole day [00:00:00Z, 23:59:59Z] is used as
        the sensing window (kept for consistency with the other two
        orbit-prefiltered sources; rarely filters anything out, given
        AMSR2's near-global daily coverage).

        _FILENAME_DATE_RE matches an 8-digit run, not a valid calendar date
        -- e.g. "20260231" passes the earlier lexicographic date-range
        filter but is not real. strptime is wrapped in its own try/except:
        on a ValueError, the file is kept unfiltered (fail-open) rather than
        aborting the whole download over one malformed filename.
        """
        from datetime import datetime, timedelta, timezone

        from ..core.orbit_coverage import orbit_overlaps_bbox

        kept = []
        dropped = 0
        for dir_path, name in matches:
            match = _FILENAME_DATE_RE.search(name)
            assert match is not None  # matches was already filtered to only entries where this matched
            try:
                day_start = datetime.strptime(match.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)
            except ValueError:
                kept.append((dir_path, name))
                continue
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
        list whatever is there and descend into every version found,
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
