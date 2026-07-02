"""
Download Copernicus Marine in-situ ocean observation data.

Downloads from the CMEMS in-situ dataset:
    cmems_obs-ins_glo_phybgcwav_mynrt_na_irr

Covers moorings (MO), drifting buoys (DB), ferryboxes (FB),
tide gauges (TG), and autonomous drifters (AD).

Library usage::

    from sar_validation.downloaders.insitu_downloader import InSituDownloader
    dl = InSituDownloader(output_dir=Path("data/run1/copernicus_insitu"))
    path = dl.download(
        min_lon=-20, max_lon=0, min_lat=35, max_lat=60,
        start="2026-01-01", end="2026-01-02",
        source_types=["mooring", "buoy"],
    )

CLI usage::

    python -m sar_validation.downloaders.insitu_downloader \\
        --min-lon -20 --max-lon 0 --min-lat 35 --max-lat 60 \\
        --start 2026-01-01 --end 2026-01-02
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from .base import normalize_datetime, is_date_recent, build_output_dir

__all__ = ["InSituDownloader"]

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

DATASET_ID = "cmems_obs-ins_glo_phybgcwav_mynrt_na_irr"
ALL_VARIABLES = ["WSPD", "WDIR", "VAVH", "VGHS", "VHM0", "HCDT", "HCSP", "EWCT", "NSCT"]

# Mapping from recipe source types to Copernicus platform codes
SOURCE_TYPE_TO_PLATFORM = {
    "mooring":     "MO",
    "buoy":        "DB",
    "ferrybox":    "FB",
    "drifter":     "DB",
    "tidal_gauge": "TG",
}


def _resolve_platform_codes(source_types: list[str]) -> list[str]:
    """Map source-type names to Copernicus platform codes (deduplicated)."""
    codes: list[str] = []
    for st in source_types:
        code = SOURCE_TYPE_TO_PLATFORM.get(st.lower())
        if code is None:
            raise ValueError(
                f"Unknown source type '{st}'. "
                f"Valid types: {', '.join(sorted(SOURCE_TYPE_TO_PLATFORM))}"
            )
        if code not in codes:
            codes.append(code)
    return codes


def _build_csv_filename(
    min_lon: float, max_lon: float,
    min_lat: float, max_lat: float,
    start_dt: str, end_dt: str,
    min_depth: float, max_depth: float,
) -> str:
    """Return the filename that copernicusmarine creates for the subset call."""
    lon_sfx = lambda v: "W" if v < 0 else "E"
    lat_sfx = lambda v: "S" if v < 0 else "N"
    vars_str = "-".join(ALL_VARIABLES)
    start_d = start_dt.split("T")[0]
    end_d   = end_dt.split("T")[0]
    date_str = start_d if start_d == end_d else f"{start_d}-{end_d}"
    depth_str = f"{abs(min_depth):.2f}-{abs(max_depth):.2f}m"
    return (
        f"{DATASET_ID}_{vars_str}_"
        f"{abs(min_lon):.2f}{lon_sfx(min_lon)}-{abs(max_lon):.2f}{lon_sfx(max_lon)}_"
        f"{abs(min_lat):.2f}{lat_sfx(min_lat)}-{abs(max_lat):.2f}{lat_sfx(max_lat)}_"
        f"{depth_str}_{date_str}.csv"
    )


class InSituDownloader:
    """
    Download in-situ observations from Copernicus Marine.

    Parameters
    ----------
    output_dir : Path
        Directory to save downloaded CSV files.
    dry_run : bool
        If True, print what would be downloaded without actually downloading.
    min_depth, max_depth : float
        Depth range for the query (metres; negative = below sea surface).
    """

    def __init__(
        self,
        output_dir: Path,
        dry_run: bool = False,
        min_depth: float = -20.0,
        max_depth: float = 20.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.min_depth = min_depth
        self.max_depth = max_depth

    def download(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start: str,
        end: str,
        source_types: Optional[list[str]] = None,
        dataset_part: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Download in-situ observations and save to a CSV.

        Parameters
        ----------
        source_types : list[str], optional
            Filter by platform type(s): mooring, buoy, ferrybox, drifter, tidal_gauge.
            None or empty list means keep all platform types.
        dataset_part : str, optional
            Which dataset part to use: "monthly" (historical) or "latest" (recent).
            If None, auto-detects based on whether end_date is within 30 days.

        Returns
        -------
        Path or None
            Path to the downloaded CSV, or None in dry-run mode.
        """
        try:
            import copernicusmarine
        except ImportError as exc:
            raise ImportError(
                "copernicusmarine is required for in-situ downloads.\n"
                "Install it with:  pip install copernicusmarine"
            ) from exc

        start_dt = normalize_datetime(start)
        end_dt   = normalize_datetime(end)

        expected_filename = _build_csv_filename(
            min_lon, max_lon, min_lat, max_lat,
            start_dt, end_dt, self.min_depth, self.max_depth,
        )
        dest_path = self.output_dir / expected_filename

        if self.dry_run:
            print(
                f"[DRY RUN] Would download in-situ data to:\n  {dest_path}"
            )
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect dataset_part if not provided
        if dataset_part is None:
            dataset_part = "latest" if is_date_recent(end_dt) else "monthly"

        # Run the copernicusmarine subset call (downloads to CWD by default)
        print(f"Downloading in-situ data …")
        print(f"  Region: lon [{min_lon}, {max_lon}] lat [{min_lat}, {max_lat}]")
        print(f"  Time:   {start_dt} → {end_dt}")
        print(f"  Depth:  {self.min_depth} to {self.max_depth} m")
        print(f"  Dataset: {dataset_part}")

        # Try initial dataset_part, with fallback if data not available
        try:
            self._download_with_part(
                copernicusmarine,
                dataset_part,
                min_lon, max_lon, min_lat, max_lat,
                start_dt, end_dt,
            )
        except Exception as e:
            error_msg = str(e)
            # Check if error is about data exceeding coordinates (date outside available range)
            if "exceed the dataset coordinates" in error_msg or "out of bounds" in error_msg.lower():
                # Try the opposite dataset_part
                alt_dataset_part = "monthly" if dataset_part == "latest" else "latest"
                print(f"  Retrying with dataset_part='{alt_dataset_part}' due to: {error_msg[:100]}…")
                try:
                    self._download_with_part(
                        copernicusmarine,
                        alt_dataset_part,
                        min_lon, max_lon, min_lat, max_lat,
                        start_dt, end_dt,
                    )
                    dataset_part = alt_dataset_part
                except Exception as e2:
                    # Both failed, raise the second error
                    raise e2
            else:
                # Not a data availability error, re-raise original
                raise

        # Move the file (copernicusmarine writes it to CWD) to our output_dir
        if Path(expected_filename).exists():
            shutil.move(str(expected_filename), str(dest_path))
            print(f"  Saved to {dest_path}")
        elif dest_path.exists():
            print(f"  Already at {dest_path}")
        else:
            # Try to find a recently-created CSV in CWD
            candidates = sorted(Path(".").glob(f"{DATASET_ID}*.csv"), key=os.path.getmtime, reverse=True)
            if candidates:
                shutil.move(str(candidates[0]), str(dest_path))
                print(f"  Saved to {dest_path}")
            else:
                raise FileNotFoundError(
                    f"copernicusmarine download completed but output CSV not found.\n"
                    f"Expected: {expected_filename}"
                )

        # Apply platform-type filter
        if source_types:
            platform_codes = _resolve_platform_codes(source_types)
            if platform_codes:
                df = pd.read_csv(dest_path)
                if "platform_type" in df.columns:
                    df = df[df["platform_type"].isin(platform_codes)]
                    df.to_csv(dest_path, index=False)
                    print(f"  Filtered to {len(df)} rows ({', '.join(source_types)})")

        return dest_path

    def _download_with_part(
        self,
        copernicusmarine,
        dataset_part: str,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start_dt: str,
        end_dt: str,
    ) -> None:
        """Internal helper to run copernicusmarine.subset() with a specific dataset_part."""
        copernicusmarine.subset(
            dataset_id=DATASET_ID,
            dataset_part=dataset_part,
            variables=ALL_VARIABLES,
            minimum_longitude=min_lon,
            maximum_longitude=max_lon,
            minimum_latitude=min_lat,
            maximum_latitude=max_lat,
            start_datetime=start_dt,
            end_datetime=end_dt,
            minimum_depth=self.min_depth,
            maximum_depth=self.max_depth,
            force_download=False,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download Copernicus Marine in-situ observations.",
    )
    p.add_argument("--params-file", metavar="FILE")
    p.add_argument("--min-lon", type=float)
    p.add_argument("--max-lon", type=float)
    p.add_argument("--min-lat", type=float)
    p.add_argument("--max-lat", type=float)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--min-depth", type=float, default=-20.0)
    p.add_argument("--max-depth", type=float, default=20.0)
    p.add_argument(
        "--source-types",
        help="Comma-separated: mooring,buoy,ferrybox,drifter,tidal_gauge",
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if args.params_file:
        with open(args.params_file) as f:
            params = json.load(f)
        min_lon   = params["minimum_longitude"]
        max_lon   = params["maximum_longitude"]
        min_lat   = params["minimum_latitude"]
        max_lat   = params["maximum_latitude"]
        start     = params["start_datetime"]
        end       = params["end_datetime"]
        min_depth = params.get("minimum_depth", -20.0)
        max_depth = params.get("maximum_depth", 20.0)
    else:
        for attr in ("min_lon", "max_lon", "min_lat", "max_lat", "start", "end"):
            if getattr(args, attr) is None:
                print(f"Error: --{attr.replace('_','-')} is required (or use --params-file)")
                sys.exit(1)
        min_lon, max_lon = args.min_lon, args.max_lon
        min_lat, max_lat = args.min_lat, args.max_lat
        start, end = args.start, args.end
        min_depth, max_depth = args.min_depth, args.max_depth

    output_dir = Path(args.output_dir) if args.output_dir else (
        build_output_dir(start, end, min_lon, max_lon, min_lat, max_lat)
        / "copernicus_insitu"
    )

    source_types = (
        [s.strip() for s in args.source_types.split(",")]
        if args.source_types else None
    )

    dl = InSituDownloader(
        output_dir=output_dir,
        dry_run=args.dry_run,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    dl.download(
        min_lon=min_lon, max_lon=max_lon,
        min_lat=min_lat, max_lat=max_lat,
        start=start, end=end,
        source_types=source_types,
    )


if __name__ == "__main__":
    main()
