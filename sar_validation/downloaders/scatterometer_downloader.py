"""
Download ASCAT scatterometer wind data from EUMETSAT via EUMDAC.

Currently supports near-real-time data from MetOp-B and MetOp-C
(EUMETSAT OSI-SAF, collection EO:EUM:DAT:METOP:OSI-104).

NOTE: Historical scatterometer data for MetOp-A, HY-2B, HY-2C and
other satellites is not yet supported. See TODO section below.

Library usage::

    from sar_validation.downloaders.scatterometer_downloader import ScatterometerDownloader
    dl = ScatterometerDownloader(output_dir=Path("data/run1/osi_saf"))
    dl.download(min_lon=-20, max_lon=0, min_lat=35, max_lat=60,
                start="2026-01-01", end="2026-01-02")

CLI usage::

    python -m sar_validation.downloaders.scatterometer_downloader \\
        --min-lon -20 --max-lon 0 --min-lat 35 --max-lat 60 \\
        --start 2026-01-01 --end 2026-01-02
"""

from __future__ import annotations

# TODO: Historical scatterometer data
#
# The current implementation is limited to near-real-time data from
# MetOp-B and MetOp-C via EUMETSAT EUMDAC (collection OSI-104).
#
# For historical data and additional satellites the following options
# should be evaluated and implemented:
#
# - MetOp-A (historical): EUMDAC archive (same collection, earlier dates)
# - HY-2B / HY-2C: NSOAS FTP  (ftp://ftp.nsoas.org.cn/)
#   or EUMETSAT re-processed archive
# - NASA PO.DAAC (OPeNDAP, no FTP): covers ASCAT, QuikSCAT, RapidScat
#   https://podaac.jpl.nasa.gov/
# - REMSS: https://www.remss.com/
#
# Until this is implemented, historic data must be obtained manually.

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional

from .base import authenticate_eumdac, normalize_datetime, build_output_dir

__all__ = ["ScatterometerDownloader"]

COLLECTION_ID = "EO:EUM:DAT:METOP:OSI-104"
SATELLITES    = ["metopb", "metopc"]


class ScatterometerDownloader:
    """
    Download ASCAT coastal-wind products from EUMETSAT OSI-SAF.

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
    ) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self._username = username
        self._password = password
        self._token = None

    def _get_token(self):
        if self._token is None:
            self._token = authenticate_eumdac(self._username, self._password)
        return self._token

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
        Download ASCAT products that intersect the given region and time window.

        Returns
        -------
        list[Path]
            Paths to the downloaded (and unzipped) files.
        """
        try:
            import eumdac
        except ImportError as exc:
            raise ImportError(
                "eumdac is required for scatterometer downloads.\n"
                "Install it with:  pip install eumdac"
            ) from exc

        start_dt = normalize_datetime(start)
        end_dt   = normalize_datetime(end)
        bbox     = f"{min_lon},{min_lat},{max_lon},{max_lat}"

        if self.dry_run:
            print(
                f"[DRY RUN] Would download ASCAT data\n"
                f"  Region: lon [{min_lon},{max_lon}] lat [{min_lat},{max_lat}]\n"
                f"  Time:   {start_dt} → {end_dt}\n"
                f"  Output: {self.output_dir}"
            )
            return []

        token = self._get_token()
        datastore = eumdac.DataStore(token)
        collection = datastore.get_collection(COLLECTION_ID)

        products = list(
            collection.search(bbox=bbox, dtstart=start_dt, dtend=end_dt)
        )
        print(f"Found {len(products)} ASCAT products.")
        if not products:
            return []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []

        for product_id in products:
            product_str = str(product_id)
            if not any(sat in product_str.lower() for sat in SATELLITES):
                continue

            try:
                product = datastore.get_product(
                    # eumdac's search yields product objects that get_product
                    # accepts here; its stub types product_id as str.
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
                        zf.extractall(self.output_dir)
                    out_path.unlink()
                    print(f"  Unzipped to {self.output_dir}")
                else:
                    downloaded.append(out_path)
                    print(f"  Saved to {out_path}")

            except Exception as exc:
                print(f"  ERROR downloading {product_id}: {exc}")

        print(f"Downloaded {len(downloaded)} ASCAT file(s).")
        return downloaded


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download ASCAT scatterometer data from EUMETSAT EUMDAC.",
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
        build_output_dir(start, end, min_lon, max_lon, min_lat, max_lat) / "osi_saf_winds"
    )

    dl = ScatterometerDownloader(
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
