"""
Download Sentinel-1 L2_OCN products from Copernicus Dataspace (CDSE).

Can be used as a library::

    from sar_validation.downloaders.sentinel1_l2_ocn_downloader import SARDownloader
    dl = SARDownloader(output_dir=Path("data/run1/S1_L2_OCN"))
    dl.download(min_lon=-20, max_lon=0, min_lat=35, max_lat=60,
                start="2026-01-01", end="2026-01-02")

Or from the command line::

    python -m sar_validation.downloaders.sentinel1_l2_ocn_downloader \\
        --min-lon -20 --max-lon 0 --min-lat 35 --max-lat 60 \\
        --start 2026-01-01 --end 2026-01-02 --download-indices all
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

from .base import (
    CopernicusODataClient,
    authenticate_cdse,
    build_output_dir,
    normalize_datetime,
    split_antimeridian_bbox,
)

__all__ = ["SARDownloader"]

VALID_MODES = {"WV", "SM", "IW", "EW"}


def _parse_modes(mode_str: str) -> list[str]:
    """Parse a comma-separated mode string. Returns [] for 'all'."""
    if not mode_str or mode_str.lower() == "all":
        return []
    modes = [m.strip().upper() for m in mode_str.split(",")]
    invalid = [m for m in modes if m not in VALID_MODES]
    if invalid:
        raise ValueError(
            f"Invalid SAR mode(s): {', '.join(invalid)}. "
            f"Valid modes: {', '.join(sorted(VALID_MODES))}"
        )
    return modes


class SARDownloader:
    """
    Download Sentinel-1 L2_OCN products from Copernicus Dataspace.

    Parameters
    ----------
    output_dir : Path
        Directory to save downloaded files.
    dry_run : bool
        If True, print what would be downloaded without actually downloading.
    username, password : str, optional
        CDSE credentials. If omitted, resolved from environment / credentials file.
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
        self._client: Optional[CopernicusODataClient] = None
        self.force_download = force_download

    def _get_client(self) -> CopernicusODataClient:
        if self._client is None:
            user, pwd = authenticate_cdse(self._username, self._password)
            self._client = CopernicusODataClient(user, pwd)
        return self._client

    def query(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start: str,
        end: str,
        modes: Optional[list[str]] = None,
        top: int = 100,
    ) -> pd.DataFrame:
        """
        Query available Sentinel-1 L2_OCN products.

        Returns a DataFrame with columns:
            Id, Name, ContentDate_Start, ContentDate_End, ContentLength_GB, Online
        """
        start_norm = normalize_datetime(start) + ".000Z"
        end_norm   = normalize_datetime(end)   + ".000Z"

        client = self._get_client()
        frames = []
        for lo, hi in split_antimeridian_bbox(min_lon, max_lon):
            records = client.query_products(
                collection="SENTINEL-1",
                product_type="OCN",
                start_date=start_norm,
                end_date=end_norm,
                min_lon=lo,
                max_lon=hi,
                min_lat=min_lat,
                max_lat=max_lat,
                top=top,
            )
            frames.append(pd.DataFrame(records))
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if df.empty:
            return df
        df = df.drop_duplicates(subset="Id", keep="first").reset_index(drop=True)

        # Filter by mode if specified
        if modes:
            pattern = "^S1[ABCD]_(" + "|".join(modes) + ")_"
            df = df[df["Name"].str.match(pattern)].reset_index(drop=True)

        return df

    def download(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start: str,
        end: str,
        modes: Optional[list[str]] = None,
        limit: Optional[int] = None,
        top: int = 100,
    ) -> list[Path]:
        """
        Query and download Sentinel-1 L2_OCN products.

        Parameters
        ----------
        modes : list[str], optional
            SAR modes to download (e.g., ["IW", "EW"]). None means all.
        limit : int, optional
            Maximum number of products to download.

        Returns
        -------
        list[Path]
            Paths to the downloaded files.
        """
        df = self.query(
            min_lon=min_lon, max_lon=max_lon,
            min_lat=min_lat, max_lat=max_lat,
            start=start, end=end, modes=modes, top=top,
        )

        if df.empty:
            print("No Sentinel-1 L2_OCN products found.")
            return []

        print(f"Found {len(df)} products.")
        for _, row in df.iterrows():
            print(f"  {row['Name']}  ({row['ContentLength_GB']:.2f} GB)  online={row['Online']}")

        if limit is not None:
            df = df.head(limit)
            print(f"Limiting to {limit} product(s).")

        if self.dry_run:
            print(f"[DRY RUN] Would download {len(df)} product(s) to {self.output_dir}")
            return []

        client = self._get_client()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []

        for i, (_, row) in enumerate(df.iterrows(), start=1):
            product_name = row["Name"]
            if not self.force_download and (self.output_dir / product_name).exists():
                print(f"[{i}/{len(df)}] Already downloaded: {product_name}")
                continue
            print(f"[{i}/{len(df)}] Downloading {product_name} …")
            try:
                path = client.download_product(row["Id"], self.output_dir, product_name)
                # Unzip if needed
                if path.suffix == ".zip":
                    with zipfile.ZipFile(path, "r") as zf:
                        zf.extractall(self.output_dir)
                    path.unlink()
                    print(f"  Unzipped to {self.output_dir}")
                else:
                    downloaded.append(path)
                    print(f"  Saved to {path}")
            except Exception as exc:
                print(f"  ERROR: {exc}")

        return downloaded


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download Sentinel-1 L2_OCN products from Copernicus Dataspace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m sar_validation.downloaders.sentinel1_l2_ocn_downloader \\
      --min-lon -20 --max-lon 0 --min-lat 35 --max-lat 60 \\
      --start 2026-01-01 --end 2026-01-02 --download-indices all

  python -m sar_validation.downloaders.sentinel1_l2_ocn_downloader \\
      --params-file params.json --mode IW,EW --limit 5
        """,
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
    p.add_argument("--mode", type=str, help="SAR mode(s): IW,EW,WV,SM or 'all'")
    p.add_argument("--top", type=int, default=100)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--download-indices", metavar="INDICES",
                   help="'all' or comma-separated indices")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--save-params", metavar="FILE")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    # Load from params file
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

    if args.save_params:
        params_out = dict(
            start_datetime=normalize_datetime(start),
            end_datetime=normalize_datetime(end),
            minimum_longitude=float(min_lon),
            maximum_longitude=float(max_lon),
            minimum_latitude=float(min_lat),
            maximum_latitude=float(max_lat),
        )
        with open(args.save_params, "w") as f:
            json.dump(params_out, f, indent=2)
        print(f"Parameters saved to {args.save_params}")

    modes = _parse_modes(args.mode) if args.mode else None

    output_dir = Path(args.output_dir) if args.output_dir else (
        build_output_dir(start, end, min_lon, max_lon, min_lat, max_lat) / "S1_L2_OCN"
    )

    # Determine indices / limit
    limit = args.limit
    if args.download_indices and args.download_indices.lower() != "all":
        indices = [int(i.strip()) for i in args.download_indices.split(",")]
        limit = len(indices)  # simplification: download first N matching

    dl = SARDownloader(
        output_dir=output_dir,
        dry_run=args.dry_run,
        username=args.username,
        password=args.password,
    )
    dl.download(
        min_lon=min_lon, max_lon=max_lon,
        min_lat=min_lat, max_lat=max_lat,
        start=start, end=end,
        modes=modes, limit=limit, top=args.top,
    )


if __name__ == "__main__":
    main()
