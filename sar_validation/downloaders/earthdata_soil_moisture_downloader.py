"""
Download soil-moisture products from NASA Earthdata via ``earthaccess``.

Covers both AMSR-E/AMSR2 (NSIDC-0451, "Daily Global Land Parameters
Derived from AMSR-E and AMSR2") and SMAP (SPL2SMP_E, "SMAP Enhanced L2
Radiometer Half-Orbit 9 km EASE-Grid Soil Moisture") with one class,
parameterized by dataset short_name — both are searchable through the
same ``earthaccess.search_data()``/``earthaccess.download()`` calls.

Credentials: ``base.authenticate_earthdata()`` resolves NASA Earthdata
Login credentials (explicit args > ``EARTHDATA_USERNAME``/
``EARTHDATA_PASSWORD`` env vars > OS keyring, migrating a legacy
``~/.netrc`` entry into the keyring on first use) before calling
``earthaccess.login()`` — see ``sar-validate --set-credential earthdata``.

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
from typing import List, Optional, Sequence, Tuple, Union

from .base import build_output_dir, normalize_datetime

__all__ = ["EarthdataSoilMoistureDownloader"]


def _describe_granule(granule: object) -> str:
    """One-line ``name  (size)`` summary of an ``earthaccess.DataGranule``
    for dry-run/found-granule listings. Falls back to ``str(granule)`` for
    anything that doesn't expose ``data_links()``/``size()`` (e.g. a plain
    string used as a test double)."""
    try:
        links = granule.data_links()  # type: ignore[attr-defined]
        name = links[0].rsplit("/", 1)[-1] if links else str(granule)
    except AttributeError:
        name = str(granule)
    try:
        size_str = f"{granule.size():.1f} MB"  # type: ignore[attr-defined]
    except AttributeError:
        size_str = "size unknown"
    return f"{name}  ({size_str})"


class EarthdataSoilMoistureDownloader:
    """
    Download NASA Earthdata soil-moisture granules via ``earthaccess``.

    Parameters
    ----------
    dataset : str or sequence of (short_name, version) tuples
        A single Earthdata ``short_name`` (e.g. ``"NSIDC-0451"`` or
        ``"SPL2SMP_E"``), paired with *version* below -- or a list of
        ``(short_name, version)`` candidate pairs to search and merge
        results from. The latter is for a mission whose data has moved
        between CMR collections over time with no temporal overlap (e.g.
        NISAR SME2's beta -> provisional product-maturity transition) --
        a single requested time window might need either collection, or
        even straddle both, and CMR has no "try these short_names in
        order" query of its own.
    output_dir : Path
        Directory to save downloaded files.
    version : str, optional
        Dataset version filter (recommended when multiple versions are
        public — avoids returning every version's granules at once).
        Ignored when *dataset* is a list of ``(short_name, version)``
        pairs (each pair carries its own version).
    dry_run : bool
        If True, still search (so the found granules can be listed) but
        skip the actual download.
    """

    def __init__(
        self,
        dataset: Union[str, Sequence[Tuple[str, Optional[str]]]],
        output_dir: Path,
        version: Optional[str] = None,
        dry_run: bool = False,
    ) -> None:
        if isinstance(dataset, str):
            self._candidates: List[Tuple[str, Optional[str]]] = [(dataset, version)]
        else:
            self._candidates = list(dataset)
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
        Search and download granules intersecting the given region/time
        window, across every ``(short_name, version)`` candidate, merging
        the results into one combined list.

        Returns
        -------
        list[Path]
            Paths to the downloaded files.
        """
        start_dt = normalize_datetime(start)
        end_dt   = normalize_datetime(end)

        import earthaccess

        from .base import authenticate_earthdata

        authenticate_earthdata()

        all_results = []
        for short_name, version in self._candidates:
            results = earthaccess.search_data(
                short_name=short_name,
                version=version,
                bounding_box=(min_lon, min_lat, max_lon, max_lat),
                temporal=(start_dt, end_dt),
            )
            print(f"Found {len(results)} {short_name} granule(s).")
            for granule in results:
                print(f"  {_describe_granule(granule)}")
            all_results.extend(results)

        self.found_count = len(all_results)

        if self.dry_run:
            print(f"[DRY RUN] Would download {len(all_results)} granule(s) to {self.output_dir}")
            return []

        if not all_results:
            return []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        downloaded = earthaccess.download(all_results, str(self.output_dir))
        paths = [Path(p) for p in downloaded]
        print(f"Downloaded {len(paths)} file(s).")
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
