"""
Download ASCAT Soil Moisture (12.5 km Swath Grid) products from EUMETSAT
via EUMDAC.

Collection EO:EUM:DAT:METOP:SOMO12 ("ASCAT Soil Moisture at 12.5 km Swath
Grid in NRT - Metop") stopped receiving new products on 2026-07-15 — this
downloader is intended for HISTORICAL data only (requests for dates after
that cutoff will simply return zero products; there is no special-casing
for the cutoff here). Reuses the same EUMDAC credentials as the existing
ASCAT wind downloader (scatterometer_downloader.py).

Library usage::

    from sar_validation.downloaders.ascat_soil_moisture_downloader import ASCATSoilMoistureDownloader
    dl = ASCATSoilMoistureDownloader(output_dir=Path("data/run1/ascat_ssm"))
    dl.download(min_lon=-20, max_lon=0, min_lat=35, max_lat=60,
                start="2026-01-01", end="2026-01-02")

CLI usage::

    python -m sar_validation.downloaders.ascat_soil_moisture_downloader \\
        --min-lon -20 --max-lon 0 --min-lat 35 --max-lat 60 \\
        --start 2026-01-01 --end 2026-01-02
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from .base import authenticate_eumdac, build_output_dir, normalize_datetime, split_antimeridian_bbox

__all__ = ["ASCATSoilMoistureDownloader", "COLLECTION_ID"]

COLLECTION_ID = "EO:EUM:DAT:METOP:SOMO12"
# Real SOMO12 product IDs use EUMETSAT's short satellite codes (e.g.
# ASCA_SMR_02_M01_20240102204500Z_...), never the literal strings
# "metopb"/"metopc" that the OSI-104 wind collection's product IDs use.
SATELLITES    = ["m01", "m02", "m03"]


class ASCATSoilMoistureDownloader:
    """
    Download ASCAT Soil Moisture 12.5 km Swath Grid products from EUMETSAT.

    Parameters
    ----------
    output_dir : Path
        Directory to save downloaded files.
    dry_run : bool
        If True, print what would be downloaded without actually downloading.
    username, password : str, optional
        EUMDAC credentials. If omitted, resolved from environment / credentials file.
    """

    def __init__(
        self,
        output_dir: Path,
        dry_run: bool = False,
        username: Optional[str] = None,
        password: Optional[str] = None,
        force_download: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self._username = username
        self._password = password
        self._token = None
        self.force_download = force_download

    def _get_token(self):
        if self._token is None:
            self._token = authenticate_eumdac(self._username, self._password)
        return self._token

    def list_candidates_dry(
        self, min_lon: float, max_lon: float, min_lat: float, max_lat: float, start: str, end: str,
    ) -> "list[tuple[str, datetime, datetime]]":
        """(product_id, sensing_start, sensing_end) for every EUMDAC
        SOMO12 product download() would find in [start, end] -- the same
        collection.search() call, deduplicated the same way, without
        fetching any product. This is a real network/authentication call
        (not a dry_run shortcut, same as download() itself).
        """
        import eumdac

        token = self._get_token()
        datastore = eumdac.DataStore(token)
        collection = datastore.get_collection(COLLECTION_ID)

        start_dt = normalize_datetime(start)
        end_dt = normalize_datetime(end)

        seen_ids: set[str] = set()
        candidates: "list[tuple[str, datetime, datetime]]" = []
        for lo, hi in split_antimeridian_bbox(min_lon, max_lon):
            bbox = f"{lo},{min_lat},{hi},{max_lat}"
            for product in collection.search(bbox=bbox, dtstart=start_dt, dtend=end_dt):
                key = str(product)
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                candidates.append((key, product.sensing_start, product.sensing_end))
        return candidates

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
        Download ASCAT SSM products that intersect the given region and time window.

        Returns
        -------
        list[Path]
            Paths to the downloaded files.
        """
        try:
            import eumdac
        except ImportError as exc:
            raise ImportError(
                "eumdac is required for ASCAT soil-moisture downloads.\n"
                "Install it with:  pip install eumdac"
            ) from exc

        start_dt = normalize_datetime(start)
        end_dt   = normalize_datetime(end)
        windows = split_antimeridian_bbox(min_lon, max_lon)

        if self.dry_run:
            for lo, hi in windows:
                print(
                    f"[DRY RUN] Would download ASCAT SSM data\n"
                    f"  Region: lon [{lo},{hi}] lat [{min_lat},{max_lat}]\n"
                    f"  Time:   {start_dt} -> {end_dt}\n"
                    f"  Output: {self.output_dir}"
                )
            return []

        token = self._get_token()
        datastore = eumdac.DataStore(token)
        collection = datastore.get_collection(COLLECTION_ID)

        seen_ids: set[str] = set()
        products = []
        for lo, hi in windows:
            bbox = f"{lo},{min_lat},{hi},{max_lat}"
            for product_id in collection.search(bbox=bbox, dtstart=start_dt, dtend=end_dt):
                key = str(product_id)
                if key not in seen_ids:
                    seen_ids.add(key)
                    products.append(product_id)

        print(f"Found {len(products)} ASCAT SSM products.")
        if not products:
            return []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []

        for product_id in products:
            product_str = str(product_id)
            if not any(sat in product_str.lower() for sat in SATELLITES):
                continue

            if not self.force_download and any(
                product_str in f.name for f in self.output_dir.glob("*") if f.is_file()
            ):
                print(f"  Already downloaded: {product_id}")
                continue

            try:
                product = datastore.get_product(
                    product_id=product_id,  # type: ignore[arg-type]
                    collection_id=COLLECTION_ID,
                )
                with product.open() as fsrc:
                    out_path = self.output_dir / fsrc.name
                    print(f"  Downloading {fsrc.name} …")
                    with open(out_path, "wb") as fdst:
                        shutil.copyfileobj(fsrc, fdst)

                if out_path.suffix == ".zip":
                    with zipfile.ZipFile(out_path, "r") as zf:
                        names = zf.namelist()
                        zf.extractall(self.output_dir)
                    out_path.unlink()
                    extracted = [self.output_dir / name for name in names]
                    downloaded.extend(extracted)
                    print(f"  Unzipped to {self.output_dir}")
                else:
                    downloaded.append(out_path)
                    print(f"  Saved to {out_path}")

            except Exception as exc:
                print(f"  ERROR downloading {product_id}: {exc}")

        print(f"Downloaded {len(downloaded)} ASCAT SSM file(s).")
        return downloaded


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download ASCAT Soil Moisture data from EUMETSAT EUMDAC.",
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
        build_output_dir(start, end, min_lon, max_lon, min_lat, max_lat) / "ascat_ssm"
    )

    dl = ASCATSoilMoistureDownloader(
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
