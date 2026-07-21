"""
Download delayed-mode (multi-year) in-situ current observations from
Copernicus Marine — ADCP moorings, Argo floats, drifters, and gliders.

Data source: INSITU_GLO_PHY_UV_DISCRETE_MY_013_044 — the same parent product
``hf_radar_historical_downloader.py`` uses for its ``radar-total`` dataset
part. Each instrument type here lives under its own sibling dataset_id and,
unlike the radar-total grid (which is not subsettable server-side — see that
file's docstring), IS subsettable directly via ``copernicusmarine.subset()``,
so this downloader follows the same direct output_directory/output_filename
call shape as ``hf_radar_downloader.py``.

Only EWCT/NSCT (eastward/northward current components) are downloaded, per
the recipe's request for current-only data from these instruments.

Delayed-mode data typically isn't finalized until 6-24 months after
acquisition, so ``download()`` short-circuits (skip + log) for any request
whose end date is younger than ``_MIN_AGE_DAYS`` (182, ~6 months) — the same
threshold used by ``hf_radar_historical_downloader``.

Library usage::

    from sar_validation.downloaders.insitu_currents_historical_downloader import (
        InSituCurrentsHistoricalDownloader,
    )
    dl = InSituCurrentsHistoricalDownloader(instrument="adcp", output_dir=Path("data/run1/adcp_historical"))
    dl.download(min_lon=-20, max_lon=0, min_lat=35, max_lat=60,
                start="2024-01-01", end="2024-01-02")

CLI usage::

    python -m sar_validation.downloaders.insitu_currents_historical_downloader \\
        --instrument adcp \\
        --min-lon -20 --max-lon 0 --min-lat 35 --max-lat 60 \\
        --start 2024-01-01 --end 2024-01-02
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .base import (
    build_output_dir,
    copernicus_marine_download_kwargs,
    normalize_datetime,
    split_antimeridian_bbox,
)

logger = logging.getLogger(__name__)

__all__ = ["InSituCurrentsHistoricalDownloader"]

_DATASET_IDS = {
    "adcp":    "cmems_obs-ins_glo_phy-cur_my_adcp_irr",
    "argo":    "cmems_obs-ins_glo_phy-cur_my_argo_irr",
    "drifter": "cmems_obs-ins_glo_phy-cur_my_drifter_PT1H",
    "glider":  "cmems_obs-ins_glo_phy-cur_my_glider_irr",
}
_VARIABLES = ["EWCT", "NSCT"]
_MIN_AGE_DAYS = 182


def _parse_iso_dt(s: str) -> datetime:
    """Convert ISO date string from normalize_datetime to timezone-aware UTC datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=timezone.utc)


class InSituCurrentsHistoricalDownloader:
    """
    Download delayed-mode current observations for one instrument type from
    Copernicus Marine (product 013_044).

    Parameters
    ----------
    instrument : str
        One of "adcp", "argo", "drifter", "glider".
    output_dir : Path
        Directory to save downloaded CSVs.
    dry_run : bool
        If True, print what would be downloaded without actually downloading.
    min_depth, max_depth : float
        Depth range for the query (metres; negative = below sea surface).
    """

    def __init__(
        self,
        instrument: str,
        output_dir: Path,
        dry_run: bool = False,
        min_depth: float = -20.0,
        max_depth: float = 20.0,
        force_download: bool = False,
    ) -> None:
        if instrument not in _DATASET_IDS:
            raise ValueError(
                f"Unknown instrument '{instrument}'. Valid: {sorted(_DATASET_IDS)}"
            )
        self.instrument = instrument
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.min_depth = min_depth
        self.max_depth = max_depth
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
        start_dt = normalize_datetime(start)
        end_dt = normalize_datetime(end)

        cutoff = datetime.now(timezone.utc) - timedelta(days=_MIN_AGE_DAYS)
        if _parse_iso_dt(end_dt) > cutoff:
            logger.warning(
                "Skipping %s delayed-mode currents download: end date %s is less "
                "than %d days old. This archive lags real-time by 6-24 months. "
                "Re-run after %s.",
                self.instrument, end_dt, _MIN_AGE_DAYS, cutoff.strftime("%Y-%m-%d"),
            )
            return []

        windows = split_antimeridian_bbox(min_lon, max_lon)
        downloaded: list[Path] = []
        for i, (win_min_lon, win_max_lon) in enumerate(windows):
            suffix = f"_w{i}" if len(windows) > 1 else ""
            path = self._download_window(
                win_min_lon, win_max_lon, min_lat, max_lat, start_dt, end_dt, suffix,
            )
            if path is not None:
                downloaded.append(path)
        return downloaded

    def _download_window(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start_dt: str,
        end_dt: str,
        filename_suffix: str,
    ) -> Optional[Path]:
        dataset_id = _DATASET_IDS[self.instrument]
        start_d = start_dt.split("T")[0]
        end_d = end_dt.split("T")[0]
        date_str = start_d if start_d == end_d else f"{start_d}-{end_d}"
        filename = f"{dataset_id}_{date_str}{filename_suffix}.csv"
        dest_path = self.output_dir / filename

        if self.dry_run:
            print(
                f"[DRY RUN] Would download {self.instrument} delayed-mode currents "
                f"(dataset_id='{dataset_id}') to:\n  {dest_path}"
            )
            return None

        if not self.force_download and dest_path.exists():
            print(f"  Already downloaded: {dest_path}")
            return dest_path

        try:
            import copernicusmarine
        except ImportError as exc:
            raise ImportError(
                "copernicusmarine is required for delayed-mode currents downloads.\n"
                "Install it with:  pip install copernicusmarine"
            ) from exc

        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {self.instrument} delayed-mode currents …")
        print(f"  Dataset: {dataset_id}")
        print(f"  Region: lon [{min_lon}, {max_lon}] lat [{min_lat}, {max_lat}]")
        print(f"  Time:   {start_dt} → {end_dt}")
        print(f"  Depth:  {self.min_depth} to {self.max_depth} m")

        copernicusmarine.subset(
            dataset_id=dataset_id,
            variables=_VARIABLES,
            minimum_longitude=min_lon,
            maximum_longitude=max_lon,
            minimum_latitude=min_lat,
            maximum_latitude=max_lat,
            start_datetime=start_dt,
            end_datetime=end_dt,
            minimum_depth=self.min_depth,
            maximum_depth=self.max_depth,
            output_directory=str(dest_path.parent),
            output_filename=dest_path.name,
            **copernicus_marine_download_kwargs(self.force_download),
        )

        if not dest_path.exists():
            raise FileNotFoundError(
                f"copernicusmarine download completed but produced no file for "
                f"{self.instrument} in [{start_dt}, {end_dt}] (dataset_id='{dataset_id}')."
            )

        print(f"  Saved to {dest_path}")
        return dest_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download delayed-mode Copernicus Marine current observations.",
    )
    p.add_argument("--instrument", required=True, choices=sorted(_DATASET_IDS))
    p.add_argument("--params-file", metavar="FILE")
    p.add_argument("--min-lon", type=float)
    p.add_argument("--max-lon", type=float)
    p.add_argument("--min-lat", type=float)
    p.add_argument("--max-lat", type=float)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--min-depth", type=float, default=-20.0)
    p.add_argument("--max-depth", type=float, default=20.0)
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
        / f"{args.instrument}_historical"
    )

    dl = InSituCurrentsHistoricalDownloader(
        instrument=args.instrument,
        output_dir=output_dir,
        dry_run=args.dry_run,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    dl.download(
        min_lon=min_lon, max_lon=max_lon,
        min_lat=min_lat, max_lat=max_lat,
        start=start, end=end,
    )


if __name__ == "__main__":
    main()
