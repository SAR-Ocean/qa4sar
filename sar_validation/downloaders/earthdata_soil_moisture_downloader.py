"""
Download soil-moisture products from NASA Earthdata via ``earthaccess``.

Covers both AMSR-E/AMSR2 (NSIDC-0451, "Daily Global Land Parameters
Derived from AMSR-E and AMSR2") and SMAP (SPL2SMP_E, "SMAP Enhanced L2
Radiometer Half-Orbit 9 km EASE-Grid Soil Moisture") with one class,
parameterized by dataset short_name — both are searchable through the
same ``earthaccess.search_data()``/``earthaccess.download()`` calls.

Credentials: ``earthaccess.login()`` resolves NASA Earthdata Login
credentials via its own standard convention (``EARTHDATA_USERNAME``/
``EARTHDATA_PASSWORD`` env vars, then ``~/.netrc``) — no bespoke
credentials file for this toolbox.

Library usage::

    from sar_validation.downloaders.earthdata_soil_moisture_downloader import EarthdataSoilMoistureDownloader
    dl = EarthdataSoilMoistureDownloader(dataset="SPL2SMP_E", version="006",
                                          output_dir=Path("data/run1/smap_ssm"))
    dl.download(min_lon=-10, max_lon=10, min_lat=40, max_lat=55,
                start="2026-01-01", end="2026-01-02")

CLI usage::

    python -m sar_validation.downloaders.earthdata_soil_moisture_downloader \\
        --dataset SPL2SMP_E --version 006 \\
        --min-lon -10 --max-lon 10 --min-lat 40 --max-lat 55 \\
        --start 2026-01-01 --end 2026-01-02
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .base import build_output_dir, normalize_datetime

__all__ = ["EarthdataSoilMoistureDownloader"]


class EarthdataSoilMoistureDownloader:
    """
    Download NASA Earthdata soil-moisture granules via ``earthaccess``.

    Parameters
    ----------
    dataset : str
        Earthdata ``short_name`` (e.g. ``"NSIDC-0451"`` or ``"SPL2SMP_E"``).
    output_dir : Path
        Directory to save downloaded files.
    version : str, optional
        Dataset version filter (recommended when multiple versions are
        public — avoids returning every version's granules at once).
    dry_run : bool
        If True, print what would be searched without actually downloading.
    """

    def __init__(
        self,
        dataset: str,
        output_dir: Path,
        version: Optional[str] = None,
        dry_run: bool = False,
    ) -> None:
        self.dataset = dataset
        self.version = version
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
    ) -> list[Path]:
        """
        Search and download granules intersecting the given region/time window.

        Returns
        -------
        list[Path]
            Paths to the downloaded files.
        """
        start_dt = normalize_datetime(start)
        end_dt   = normalize_datetime(end)

        if self.dry_run:
            print(
                f"[DRY RUN] Would search earthaccess for {self.dataset} "
                f"(version={self.version})\n"
                f"  Region: lon [{min_lon},{max_lon}] lat [{min_lat},{max_lat}]\n"
                f"  Time:   {start_dt} -> {end_dt}\n"
                f"  Output: {self.output_dir}"
            )
            return []

        import earthaccess

        earthaccess.login()
        results = earthaccess.search_data(
            short_name=self.dataset,
            version=self.version,
            bounding_box=(min_lon, min_lat, max_lon, max_lat),
            temporal=(start_dt, end_dt),
        )
        print(f"Found {len(results)} {self.dataset} granule(s).")
        if not results:
            return []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        downloaded = earthaccess.download(results, str(self.output_dir))
        paths = [Path(p) for p in downloaded]
        print(f"Downloaded {len(paths)} {self.dataset} file(s).")
        return paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download soil-moisture data from NASA Earthdata via earthaccess.",
    )
    p.add_argument("--params-file", metavar="FILE")
    p.add_argument("--dataset", required=True, help='Earthdata short_name, e.g. "NSIDC-0451" or "SPL2SMP_E"')
    p.add_argument("--version", default=None)
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
        build_output_dir(start, end, min_lon, max_lon, min_lat, max_lat) / args.dataset.lower()
    )

    dl = EarthdataSoilMoistureDownloader(
        dataset=args.dataset,
        version=args.version,
        output_dir=output_dir,
        dry_run=args.dry_run,
    )
    dl.download(
        min_lon=min_lon, max_lon=max_lon,
        min_lat=min_lat, max_lat=max_lat,
        start=start, end=end,
    )


if __name__ == "__main__":
    main()
