"""
Download moored and drifting buoy wind, wave, and current observations
from the WMO Global Telecommunication System (GTS), via ECMWF's MARS
archive.

Requires MARS access, with credentials read from ``~/.ecmwfapirc`` (the
standard MARS/Web API key file).

Library usage::

    from sar_validation.downloaders.gts_buoy_downloader import GTSBuoyDownloader
    dl = GTSBuoyDownloader(output_dir=Path("data/run1/gts_buoy"))
    paths = dl.download(min_lon=-10, max_lon=10, min_lat=40, max_lat=55,
                         start="2026-08-30T00:00:00", end="2026-08-30T23:59:00")

CLI usage::

    python -m sar_validation.downloaders.gts_buoy_downloader \\
        --min-lon -10 --max-lon 10 --min-lat 40 --max-lat 55 \\
        --start 2026-08-30T00:00:00 --end 2026-08-30T23:59:00
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from .base import build_output_dir, normalize_datetime, run_with_timeout

logger = logging.getLogger(__name__)

__all__ = ["GTSBuoyDownloader", "main"]

#: MARS obstype value for BUFR moored buoys (181) and BUFR drifting buoys
#: (182). Combined into one request -- narrowing further with an ``ident``
#: filter does not work for this obstype (confirmed against a live MARS
#: request: an ident-filtered pull returned zero matches for a station
#: known to be present in the unfiltered result for the same day).
_OBSTYPE = "181/182"

#: How long to wait for a single day's MARS request before giving up on
#: it. MARS provides no way to cancel an in-flight request, so this only
#: bounds how long this process waits -- the abandoned request keeps
#: running on ECMWF's side regardless.
_MARS_REQUEST_TIMEOUT_SECONDS = 360


class GTSBuoyDownloader:
    """
    Download GTS buoy wind, wave, and current observations via MARS, one BUFR file per
    calendar day covering the requested window.

    Parameters
    ----------
    output_dir : Path
        Directory to save downloaded BUFR files.
    dry_run : bool
        If True, log what would be downloaded without calling MARS.
    force_download : bool
        If True, re-download a day's BUFR file even if it already exists
        on disk, rather than skipping it. Lets a truncated or corrupted
        file left behind by an interrupted process be re-fetched instead
        of requiring manual deletion.
    """

    def __init__(
        self, output_dir: Path, dry_run: bool = False, force_download: bool = False,
    ) -> None:
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
        """
        Download one BUFR file per calendar day in ``[start, end]``.

        MARS's ``date``/``time``/``range`` request shape is day-based, not
        an arbitrary start/end span, so a window spanning multiple days is
        split into one request per day here, each covering that day's full
        00:00-23:59. *min_lon*/*max_lon*/*min_lat*/*max_lat* are accepted
        for interface symmetry with the other downloaders and recorded in
        log output, but MARS's buoy obstype has no bbox request facet --
        the whole day's global buoy set is fetched and cropped to the
        recipe bounds later, during DataTree conversion (matching how the
        radiometer downloader handles RSS's globally-gridded products).

        Returns
        -------
        list[Path]
            Paths to the downloaded (or already-cached) BUFR files, one
            per requested day.
        """
        window_start = datetime.fromisoformat(normalize_datetime(start))
        window_end = datetime.fromisoformat(normalize_datetime(end))

        if not self.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        downloaded: list[Path] = []
        day = window_start.date()
        while day <= window_end.date():
            target = self._bufr_path_for_day(day)

            if not self.force_download and target.exists():
                logger.info("  %s: already present (%s), skipping.", day.isoformat(), target.name)
                downloaded.append(target)
                day += timedelta(days=1)
                continue

            if self.dry_run:
                logger.info(
                    "  [dry-run] would download GTS buoy obs for %s "
                    "(region lon [%.2f,%.2f] lat [%.2f,%.2f])",
                    day.isoformat(), min_lon, max_lon, min_lat, max_lat,
                )
                day += timedelta(days=1)
                continue

            request = self._build_request(day)
            print(f"  Downloading GTS buoy obs for {day.isoformat()} …")
            try:
                completed = run_with_timeout(
                    lambda: self._execute_mars_request(request, target),
                    _MARS_REQUEST_TIMEOUT_SECONDS,
                )
            except Exception:
                target.unlink(missing_ok=True)
                raise
            if not completed:
                target.unlink(missing_ok=True)
                logger.warning(
                    "GTS buoy obs for %s timed out after %d minute(s) waiting on "
                    "MARS; abandoning the remaining day(s) in this window for "
                    "this run.",
                    day.isoformat(), _MARS_REQUEST_TIMEOUT_SECONDS // 60,
                )
                break
            downloaded.append(target)
            day += timedelta(days=1)

        return downloaded

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bufr_path_for_day(self, day: date) -> Path:
        return self.output_dir / f"gts_buoy_{day.strftime('%Y%m%d')}.bufr"

    def _build_request(self, day: date) -> dict:
        return {
            "class": "od",
            "type": "ob",
            "stream": "oper",
            "obstype": _OBSTYPE,
            "date": day.strftime("%Y%m%d"),
            "time": "0000",
            "range": "1439",
        }

    def _execute_mars_request(self, request: dict, target: Path) -> None:
        """Submit *request* to MARS and stream the result to *target*.

        The only place ``ecmwfapi`` is imported or touched -- tests
        monkeypatch this method instead of mocking the network client.
        ``target`` is passed only as ``ECMWFService.execute``'s own
        second argument, which controls where the completed result is
        downloaded to locally -- it must not also be a key inside
        *request* itself, since that dictionary is submitted to MARS as
        the retrieval request, and MARS's request-language parser
        requires a ``target`` value to be quoted; an unquoted path
        submitted this way fails immediately at its first "/" character.
        """
        from ecmwfapi import ECMWFService  # noqa: PLC0415 -- optional dependency, imported lazily

        server = ECMWFService("mars")
        server.execute(request, str(target))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download GTS buoy wind, wave, and current observations via MARS."
    )
    p.add_argument("--min-lon", type=float, required=True)
    p.add_argument("--max-lon", type=float, required=True)
    p.add_argument("--min-lat", type=float, required=True)
    p.add_argument("--max-lat", type=float, required=True)
    p.add_argument("--start", required=True, help="Start date/datetime (ISO-8601).")
    p.add_argument("--end", required=True, help="End date/datetime (ISO-8601, inclusive).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory (default: data/<timerange>_<bounds>/gts_buoy).")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=None)
    out_dir = Path(args.output_dir) if args.output_dir else (
        build_output_dir(args.start, args.end, args.min_lon, args.max_lon,
                         args.min_lat, args.max_lat) / "gts_buoy"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    dl = GTSBuoyDownloader(output_dir=out_dir, dry_run=args.dry_run)
    dl.download(
        min_lon=args.min_lon, max_lon=args.max_lon,
        min_lat=args.min_lat, max_lat=args.max_lat,
        start=args.start, end=args.end,
    )


if __name__ == "__main__":
    main()
