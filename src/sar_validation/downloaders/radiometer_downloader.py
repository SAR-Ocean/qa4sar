"""
Download satellite microwave radiometer ocean-wind data from Remote Sensing
Systems (RSS) over public HTTPS (``https://data.remss.com/``) — no account
needed.

Radiometers measure ocean surface **wind speed** (and, for polarimetric
sensors like WindSat, wind **direction**). RSS distributes each mission as a
daily gridded product **already resampled to a common 0.25° global grid**, so
one file per sensor per day covers the whole globe. Two passes (ascending /
descending) are stored per file, and every grid cell carries its own
measurement time.

Format landscape 
-------------------------------------------------
- **AMSR2** publishes modern **NetCDF** L3 files:
    ``amsr2/ocean/L3/v08.2/daily/{YYYY}/RSS_AMSR2_ocean_L3_daily_{date}_v08.2.nc``
  (recent days appear first under ``.../daily/rt/..._v08.2-rt.nc``).
- **GMI, SSMIS (F16/F17/F18), WindSat, AMSR-E** are distributed only as RSS
  **binary bytemaps** (gzipped flat arrays), a different format that needs a
  dedicated reader.

This module downloads both formats over HTTPS: AMSR2 as **NetCDF**, and GMI /
SSMIS (f16/f17/f18) / WindSat as RSS **binary bytemaps** (``.gz``, decoded by
:mod:`sar_validation.downloaders._rss_bytemap`; WindSat additionally provides
wind direction). The per-sensor :data:`SENSORS` table carries a ``format`` tag
and the URL templates for each.

Library usage::

    from sar_validation.downloaders.radiometer_downloader import RadiometerDownloader
    dl = RadiometerDownloader(output_dir=Path("data/run1/radiometer"))
    paths = dl.download(min_lon=-10, max_lon=5, min_lat=50, max_lat=62,
                        start="2024-06-01", end="2024-06-02")

CLI usage::

    python -m sar_validation.downloaders.radiometer_downloader \\
        --min-lon -10 --max-lon 5 --min-lat 50 --max-lat 62 \\
        --start 2024-06-01 --end 2024-06-02
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import requests

from .base import build_output_dir, normalize_datetime

__all__ = ["RadiometerDownloader", "SENSORS"]

BASE_URL = "https://data.remss.com"

# ---------------------------------------------------------------------------
# Per-sensor configuration.
#
# ``format`` selects the download/parse path: "netcdf" sensors are downloaded
# here directly; "bytemap" sensors are declared for completeness but skipped
# until the binary-bytemap reader lands (fast-follow).
#
# The ``{Y}``/``{m}``/``{d}`` placeholders are filled per day. ``url_path`` is
# relative to BASE_URL; the daily file lives at ``<url_path>/<file>``. Some
# products also expose a flat real-time folder for the most recent days
# (``rt_url_path``/``rt_file``), tried as a fallback when the dated file 404s.
# ---------------------------------------------------------------------------
SENSORS: dict[str, dict] = {
    "amsr2": {
        "format": "netcdf",
        "platform": "GCOM-W1",
        "has_direction": False,
        "availability_start": "2012-07-02T00:00:00",
        "url_path": "amsr2/ocean/L3/v08.2/daily/{Y}",
        "file": "RSS_AMSR2_ocean_L3_daily_{Y}-{m}-{d}_v08.2.nc",
        # Recent days are published here first, flat (no year subfolder).
        "rt_url_path": "amsr2/ocean/L3/v08.2/daily/rt",
        "rt_file": "RSS_AMSR2_ocean_L3_daily_{Y}-{m}-{d}_v08.2-rt.nc",
    },
    # --- Binary-bytemap sensors (RSS .gz; decoded by _rss_bytemap.py) ---
    # Daily files live under y{Y}/m{m}/ with a YYYYMMDD stamp in the name.
    "gmi": {
        "format": "bytemap", "has_direction": False,
        "availability_start": "2014-03-04T00:00:00",
        "url_path": "gmi/bmaps_v08.2/y{Y}/m{m}",
        "file": "f35_{Y}{m}{d}v8.2.gz",
    },
    "ssmis_f16": {
        "format": "bytemap", "has_direction": False,
        "availability_start": "2003-10-26T00:00:00",
        "url_path": "ssmi/f16/bmaps_v07/y{Y}/m{m}",
        "file": "f16_{Y}{m}{d}v7.gz",
    },
    "ssmis_f17": {
        "format": "bytemap", "has_direction": False,
        "availability_start": "2006-11-04T00:00:00",
        "url_path": "ssmi/f17/bmaps_v07/y{Y}/m{m}",
        "file": "f17_{Y}{m}{d}v7.gz",
    },
    "ssmis_f18": {
        "format": "bytemap", "has_direction": False,
        "availability_start": "2009-10-18T00:00:00",
        "url_path": "ssmi/f18/bmaps_v07/y{Y}/m{m}",
        "file": "f18_{Y}{m}{d}v7.gz",
    },
    "windsat": {
        "format": "bytemap", "has_direction": True,
        "availability_start": "2003-02-05T00:00:00",
        "url_path": "windsat/bmaps_v07.0.1/y{Y}/m{m}",
        "file": "wsat_{Y}{m}{d}v7.0.1.gz",
    },
}

#: Sensors this module can download (those with a configured URL template).
SUPPORTED_SENSORS = [s for s, cfg in SENSORS.items() if cfg.get("url_path")]


class RadiometerDownloader:
    """
    Download RSS radiometer daily gridded ocean-wind NetCDF files over HTTPS.

    Parameters
    ----------
    output_dir : Path
        Directory to save downloaded NetCDF files.
    dry_run : bool
        If True, print the sensor/date/URL matrix without downloading.
    """

    def __init__(
        self,
        output_dir: Path,
        dry_run: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run

    def download(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start: str,
        end: str,
        sensors: Optional[Iterable[str]] = None,
    ) -> list[Path]:
        """
        Download every daily radiometer file for the requested sensors that
        falls within the ``[start, end]`` window.

        The files are global 0.25° grids; the spatial bounds are accepted for
        interface symmetry with the other downloaders and recorded in the
        dry-run output, but the whole daily file is fetched and cropped to the
        recipe bounds later, during DataTree conversion.

        Parameters
        ----------
        sensors : iterable of str, optional
            Restrict to these sensor keys (see :data:`SENSORS`). None (default)
            means all currently-supported (NetCDF) sensors.

        Returns
        -------
        list[Path]
            Paths to the downloaded NetCDF files.
        """
        start_dt = normalize_datetime(start)
        end_dt = normalize_datetime(end)

        requested = list(sensors) if sensors else list(SUPPORTED_SENSORS)
        unknown = [s for s in requested if s not in SENSORS]
        if unknown:
            print(f"  WARNING: unknown radiometer sensor(s) {unknown}; skipping them.")
        requested = [s for s in requested if s in SENSORS]

        if not self.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        downloaded: list[Path] = []

        for sensor in requested:
            cfg = SENSORS[sensor]
            if not cfg.get("url_path"):
                print(f"  Skipping {sensor}: no download URL configured for this sensor.")
                continue

            avail_start = cfg.get("availability_start")
            if avail_start and end_dt < avail_start:
                print(
                    f"  Skipping {sensor}: requested window ends {end_dt}, before "
                    f"{sensor} availability starts ({avail_start})."
                )
                continue

            downloaded.extend(
                self._download_sensor(
                    sensor, cfg, start_dt, end_dt,
                    min_lon, max_lon, min_lat, max_lat,
                )
            )

        print(f"Downloaded {len(downloaded)} radiometer file(s).")
        return downloaded

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _download_sensor(
        self,
        sensor: str,
        cfg: dict,
        start_dt: str,
        end_dt: str,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
    ) -> list[Path]:
        """Download every daily file for a single NetCDF sensor."""
        avail_start = cfg.get("availability_start")
        eff_start_dt = max(start_dt, avail_start) if avail_start else start_dt

        day = datetime.fromisoformat(eff_start_dt).date()
        last = datetime.fromisoformat(end_dt).date()

        out: list[Path] = []
        while day <= last:
            fields = {"Y": f"{day.year:04d}", "m": f"{day.month:02d}", "d": f"{day.day:02d}"}
            # Primary (final) URL, then real-time fallback for recent days.
            candidates = [
                (
                    f"{BASE_URL}/{cfg['url_path'].format(**fields)}/{cfg['file'].format(**fields)}",
                    cfg["file"].format(**fields),
                )
            ]
            if cfg.get("rt_url_path"):
                candidates.append(
                    (
                        f"{BASE_URL}/{cfg['rt_url_path'].format(**fields)}/{cfg['rt_file'].format(**fields)}",
                        cfg["rt_file"].format(**fields),
                    )
                )

            if self.dry_run:
                print(
                    f"[DRY RUN] Would download {sensor} {day.isoformat()} "
                    f"(region lon [{min_lon},{max_lon}] lat [{min_lat},{max_lat}]):\n"
                    f"  {candidates[0][0]}"
                    + (f"\n  (fallback) {candidates[1][0]}" if len(candidates) > 1 else "")
                )
                day += timedelta(days=1)
                continue

            path = self._fetch_first(sensor, day, candidates)
            if path is not None:
                out.append(path)
            day += timedelta(days=1)

        return out

    def _fetch_first(self, sensor: str, day, candidates: list[tuple[str, str]]) -> Optional[Path]:
        """Try each candidate URL in order; save the first that exists."""
        for url, filename in candidates:
            dest = self.output_dir / filename
            if dest.exists():
                print(f"  {sensor} {day.isoformat()}: already present ({dest.name}), skipping.")
                return dest
            try:
                with requests.get(url, stream=True, timeout=120) as resp:
                    if resp.status_code == 404:
                        continue   # try the next candidate (e.g. rt fallback)
                    resp.raise_for_status()
                    print(f"  Downloading {sensor} {day.isoformat()} → {filename} …")
                    with open(dest, "wb") as fh:
                        for chunk in resp.iter_content(chunk_size=1 << 20):
                            if chunk:
                                fh.write(chunk)
                print(f"  Saved to {dest}")
                return dest
            except Exception as exc:
                # Clean up a partial file so a re-run retries cleanly.
                if dest.exists():
                    dest.unlink()
                print(f"  ERROR downloading {url}: {exc}")
                return None

        print(f"  {sensor} {day.isoformat()}: no file available (404), skipped.")
        return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download RSS radiometer ocean-wind NetCDF data over HTTPS.",
    )
    p.add_argument("--params-file", metavar="FILE")
    p.add_argument("--min-lon", type=float)
    p.add_argument("--max-lon", type=float)
    p.add_argument("--min-lat", type=float)
    p.add_argument("--max-lat", type=float)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument(
        "--sensors",
        default=None,
        help=f"Comma-separated sensor keys (default: all supported = "
             f"{','.join(SUPPORTED_SENSORS)}). See SENSORS for all keys.",
    )
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
        build_output_dir(start, end, min_lon, max_lon, min_lat, max_lat) / "radiometer"
    )

    sensors = (
        [s.strip() for s in args.sensors.split(",")]
        if args.sensors else None
    )

    dl = RadiometerDownloader(
        output_dir=output_dir,
        dry_run=args.dry_run,
    )
    dl.download(
        min_lon=min_lon, max_lon=max_lon,
        min_lat=min_lat, max_lat=max_lat,
        start=start, end=end,
        sensors=sensors,
    )


if __name__ == "__main__":
    main()
