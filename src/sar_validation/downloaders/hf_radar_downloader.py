"""
Download HF-radar surface-current *grids* from Copernicus Marine.

Data source: INSITU_GLO_PHYBGCWAV_DISCRETE_MYNRT_013_030
    Dataset ID: cmems_obs-ins_glo_phybgcwav_mynrt_na_irr
    dataset_part: "<latest|monthly>-radar-total--<Region>"

The dataset's plain "latest"/"monthly" parts are a *sparse per-platform*
in-situ feed (moorings/buoys/drifters/ferrybox/tide gauges) that carries no
HF-radar rows at all — verified empty for any bbox/time/variable combination
tried on 2026-07-15. HF-radar current data is delivered separately, as a
regular (time, lat, lon) grid per named coastal region, via
"<latest|monthly>-radar-total--<Region>" dataset_parts. This downloader
resolves the request bbox to one of those named regions and subsets that
grid directly.

Library usage::

    from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader
    dl = HFRadarDownloader(output_dir=Path("data/run1/hf_radar"))
    dl.download(min_lon=-90, max_lon=-60, min_lat=30, max_lat=40,
                start="2026-06-05", end="2026-06-06")

CLI usage::

    python -m sar_validation.downloaders.hf_radar_downloader \\
        --min-lon -90 --max-lon -60 --min-lat 30 --max-lat 40 \\
        --start 2026-06-05 --end 2026-06-06
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from ._hf_radar_regions import HFR_REGIONS, resolve_hfr_region
from .base import (
    build_output_dir,
    copernicus_marine_download_kwargs,
    is_date_recent,
    normalize_datetime,
    split_antimeridian_bbox,
)

__all__ = ["HFRadarDownloader"]

DATASET_ID = "cmems_obs-ins_glo_phybgcwav_mynrt_na_irr"


def _build_filename(region: str, start_dt: str, end_dt: str) -> str:
    start_d = start_dt.split("T")[0]
    end_d = end_dt.split("T")[0]
    date_str = start_d if start_d == end_d else f"{start_d}-{end_d}"
    return f"{DATASET_ID}_radar-total_{region}_{date_str}.nc"


class HFRadarDownloader:
    """
    Download a Copernicus Marine HF-radar current grid for the region that
    overlaps the request bbox.

    Parameters
    ----------
    output_dir : Path
        Directory to save the downloaded NetCDF.
    dry_run : bool
        If True, print what would be downloaded without actually downloading.
    min_depth, max_depth : float
        Accepted for interface compatibility with the orchestrator's
        recipe-level depth-resolution machinery. The HF-radar-total grid has
        no depth axis (it's a fixed near-surface radar measurement), so
        these are unused.
    """

    def __init__(
        self,
        output_dir: Path,
        dry_run: bool = False,
        min_depth: float = -2.0,
        max_depth: float = 2.0,
        force_download: bool = False,
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
        start_dt = normalize_datetime(start)
        end_dt = normalize_datetime(end)
        windows = split_antimeridian_bbox(min_lon, max_lon)

        downloaded: list[Path] = []
        last_error: Optional[ValueError] = None
        resolved_any = False
        for i, (win_min_lon, win_max_lon) in enumerate(windows):
            suffix = f"_w{i}" if len(windows) > 1 else ""
            try:
                region = resolve_hfr_region(win_min_lon, win_max_lon, min_lat, max_lat)
            except ValueError as exc:
                if len(windows) == 1:
                    raise
                last_error = exc
                continue
            resolved_any = True
            path = self._download_region_window(
                region, win_min_lon, win_max_lon, min_lat, max_lat,
                start_dt, end_dt, suffix,
            )
            if path is not None:
                downloaded.append(path)

        if not resolved_any and last_error is not None:
            raise last_error
        return downloaded

    def check_availability_dry(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start: str,
        end: str,
    ) -> bool:
        """
        Whether the Copernicus Marine HF-radar grid for the region
        overlapping this bbox has any data in [start, end], without writing
        anything to disk.

        Resolves the same region + dataset_part ``_download_region_window``
        would. Unlike that method, this makes no dataset_part fallback
        retry attempt (a footprint sitting right at the ``latest``/``monthly``
        boundary could in principle get a less accurate answer than a real
        download would) -- the same known, deliberately out-of-scope gap
        ``InSituDownloader.check_availability_dry`` also leaves.

        Uses ``copernicusmarine.open_dataset()`` -- a lazy, server-side
        bbox/time-filtered xarray open with no local file written -- rather
        than ``subset()``'s download-to-disk call. The
        ``"<latest|monthly>-radar-total--<Region>"`` dataset_parts are
        genuine gridded NetCDF products, unlike the plain ``"latest"``/
        ``"monthly"`` in-situ dataset_parts under this same ``DATASET_ID``
        that ``InSituDownloader.check_availability_dry`` targets instead
        (which needed ``read_dataframe()`` -- see that method's own
        docstring for why ``open_dataset()`` doesn't work there).

        Returns False when no known region overlaps the bbox at all.
        """
        import copernicusmarine

        try:
            region = resolve_hfr_region(min_lon, max_lon, min_lat, max_lat)
        except ValueError:
            return False

        start_dt = normalize_datetime(start)
        end_dt = normalize_datetime(end)
        use_latest = HFR_REGIONS[region]["has_latest"] and is_date_recent(end_dt)
        dataset_part = f"{'latest' if use_latest else 'monthly'}-radar-total--{region}"

        ds = copernicusmarine.open_dataset(
            dataset_id=DATASET_ID,
            dataset_part=dataset_part,
            minimum_longitude=min_lon,
            maximum_longitude=max_lon,
            minimum_latitude=min_lat,
            maximum_latitude=max_lat,
            start_datetime=start_dt,
            end_datetime=end_dt,
        )
        return "time" in ds.sizes and ds.sizes["time"] > 0

    def _download_region_window(
        self,
        region: str,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start_dt: str,
        end_dt: str,
        filename_suffix: str,
    ) -> Optional[Path]:
        use_latest = HFR_REGIONS[region]["has_latest"] and is_date_recent(end_dt)
        dataset_part = f"{'latest' if use_latest else 'monthly'}-radar-total--{region}"
        filename = _build_filename(region, start_dt, end_dt)
        if filename_suffix:
            filename = filename.replace(".nc", f"{filename_suffix}.nc")
        dest_path = self.output_dir / filename

        if self.dry_run:
            print(
                f"[DRY RUN] Would download Copernicus HF-radar grid for region "
                f"'{region}' (dataset_part='{dataset_part}') to:\n  {dest_path}"
            )
            return None

        try:
            import copernicusmarine
        except ImportError as exc:
            raise ImportError(
                "copernicusmarine is required for HF radar downloads.\n"
                "Install it with:  pip install copernicusmarine"
            ) from exc

        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("Downloading Copernicus HF-radar surface-current grid …")
        print(f"  Region: {region}")
        print(f"  BBox:   lon [{min_lon}, {max_lon}] lat [{min_lat}, {max_lat}]")
        print(f"  Time:   {start_dt} → {end_dt}")
        print(f"  Dataset part: {dataset_part}")

        try:
            self._subset_with_part(
                copernicusmarine, dataset_part,
                min_lon, max_lon, min_lat, max_lat,
                start_dt, end_dt, dest_path,
            )
        except Exception as e:
            error_msg = str(e)
            if use_latest and (
                "exceed the dataset coordinates" in error_msg
                or "out of bounds" in error_msg.lower()
            ):
                dataset_part = f"monthly-radar-total--{region}"
                print(f"  Retrying with dataset_part='{dataset_part}' due to: {error_msg[:120]}…")
                self._subset_with_part(
                    copernicusmarine, dataset_part,
                    min_lon, max_lon, min_lat, max_lat,
                    start_dt, end_dt, dest_path,
                )
            else:
                raise

        if not dest_path.exists():
            raise FileNotFoundError(
                f"Copernicus HF-radar grid download completed but produced no "
                f"file for region '{region}' in [{start_dt}, {end_dt}] "
                f"(dataset_part='{dataset_part}')."
            )

        print(f"  Saved to {dest_path}")
        return dest_path

    def _subset_with_part(
        self, copernicusmarine, dataset_part,
        min_lon, max_lon, min_lat, max_lat,
        start_dt, end_dt, dest_path,
    ) -> None:
        # No `variables=` filter: omitting it makes copernicusmarine return
        # every variable in the dataset_part (verified live — 14 vars for
        # *-radar-total--<Region>, including EWCS/NSCS standard deviations
        # and all *_QC/QCflag fields), so the converter (Task 3) always has
        # the full ancillary set to pick from on disk.
        copernicusmarine.subset(
            dataset_id=DATASET_ID,
            dataset_part=dataset_part,
            minimum_longitude=min_lon,
            maximum_longitude=max_lon,
            minimum_latitude=min_lat,
            maximum_latitude=max_lat,
            start_datetime=start_dt,
            end_datetime=end_dt,
            output_directory=str(dest_path.parent),
            output_filename=dest_path.name,
            **copernicus_marine_download_kwargs(self.force_download),
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download a Copernicus Marine HF-radar current grid.",
    )
    p.add_argument("--params-file", metavar="FILE")
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
        build_output_dir(start, end, min_lon, max_lon, min_lat, max_lat) / "hf_radar"
    )

    dl = HFRadarDownloader(output_dir=output_dir, dry_run=args.dry_run)
    dl.download(
        min_lon=min_lon, max_lon=max_lon,
        min_lat=min_lat, max_lat=max_lat,
        start=start, end=end,
    )


if __name__ == "__main__":
    main()
