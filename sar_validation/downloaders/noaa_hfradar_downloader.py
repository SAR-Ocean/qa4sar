"""
Download NOAA HFRnet gridded surface currents (Real-Time Velocities, RTV).

Backend for Phase 3a: ERDDAP griddap NetCDF subset (recent ~3-month window).
The THREDDS/OPeNDAP archive backend (2012–present) is Phase 3b.

Data source: NOAA/UCSD HFRnet Regional/National RTV, distributed via ERDDAP
griddap on coastwatch.pfeg.noaa.gov. Variables ``water_u``/``water_v`` carry CF
standard names ``surface_eastward/northward_sea_water_velocity``.

Reference for the SAR-vs-HF-radar comparison this feeds:
Martin, Gommenginger, Jacob & Staneva (2022), RSE 268:112758.

CLI usage::

    python -m sar_validation.downloaders.noaa_hfradar_downloader \\
        --min-lon -125 --max-lon -119 --min-lat 33 --max-lat 38 \\
        --start 2024-05-01 --end 2024-05-02 --resolution 6 --dry-run
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from .base import normalize_datetime, split_antimeridian_bbox

__all__ = [
    "NOAAHFRadarDownloader",
    "select_erddap_dataset",
    "build_erddap_subset_url",
    "select_backend",
    "clamp_to_region_bbox",
]

# ERDDAP griddap host serving the UCSD HFRnet RTV datasets.
ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
# ERDDAP keeps a rolling ~3-month window; older dates need the THREDDS archive
# (Phase 3b).
ERDDAP_WINDOW_DAYS = 90
DEFAULT_RESOLUTION_KM = 6

# Region bounding boxes (lon_min, lon_max, lat_min, lat_max) and their
# resolution → ERDDAP dataset-id maps. Non-CONUS regions (Hawaii, Alaska,
# PR/USVI, Great Lakes) are deferred (design §6) and raise a clear error.
_REGIONS = {
    "US_WEST": {
        "bbox": (-130.36, -115.8056, 30.25, 49.99204),
        "datasets": {1: "ucsdHfrW1", 2: "ucsdHfrW2", 6: "ucsdHfrW6"},
    },
    "US_EAST_GULF": {
        "bbox": (-97.88385, -60.0, 22.0, 46.0),
        "datasets": {1: "ucsdHfrE1", 6: "ucsdHfrE6"},
    },
}


def _bbox_center(min_lon, max_lon, min_lat, max_lat):
    return (min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0


def _match_region(min_lon, max_lon, min_lat, max_lat) -> tuple[str, Dict]:
    """Find the ``_REGIONS`` entry whose bbox contains the request's center point.

    Raises ``ValueError`` if no configured region contains it.
    """
    clon, clat = _bbox_center(min_lon, max_lon, min_lat, max_lat)
    for name, cfg in _REGIONS.items():
        lo, hi, la, ha = cfg["bbox"]
        if lo <= clon <= hi and la <= clat <= ha:
            return name, cfg
    raise ValueError(
        "No ERDDAP HF-radar dataset for bbox center "
        f"({clon:.2f}, {clat:.2f}). Phase 3a supports US West and US "
        "East/Gulf coasts; other regions arrive in later phases."
    )


def select_erddap_dataset(min_lon, max_lon, min_lat, max_lat, resolution_km: int) -> str:
    """Choose the ERDDAP RTV dataset id from the request bbox and resolution.

    Raises ``ValueError`` if the region is outside the supported CONUS coasts
    or the requested resolution is unavailable for that region.
    """
    name, cfg = _match_region(min_lon, max_lon, min_lat, max_lat)
    datasets = cfg["datasets"]
    if resolution_km not in datasets:
        raise ValueError(
            f"resolution {resolution_km} km not available for region "
            f"{name}; available: {sorted(datasets)} km"
        )
    return datasets[resolution_km]


def clamp_to_region_bbox(min_lon, max_lon, min_lat, max_lat) -> tuple[float, float, float, float]:
    """Clamp a request bbox to the matched region's known grid extent.

    ERDDAP griddap rejects (HTTP 404) any axis constraint that starts or ends
    outside the dataset's actual coordinate axis, rather than clipping it —
    so a recipe bbox that only partially overlaps a region (e.g. extending
    past its southern edge) must be clamped here before the URL is built, not
    left for the server to reject outright.
    """
    _, cfg = _match_region(min_lon, max_lon, min_lat, max_lat)
    lo, hi, la, ha = cfg["bbox"]
    return (
        max(min_lon, lo), min(max_lon, hi),
        max(min_lat, la), min(max_lat, ha),
    )


def build_erddap_subset_url(
    dataset_id: str, min_lon, max_lon, min_lat, max_lat, start: str, end: str
) -> str:
    """Build an ERDDAP griddap NetCDF-subset URL for water_u/water_v.

    Dimension order is ``[time][latitude][longitude]``; ERDDAP accepts value
    selectors ``[(min):(max)]`` and returns the enclosing grid subset
    server-side. This URL is also the intended seed for a future
    granule-search / dry-collocation feature.
    """
    t0 = normalize_datetime(start)
    t1 = normalize_datetime(end)
    # ERDDAP wants explicit Z-suffixed ISO timestamps.
    t0 = t0 if t0.endswith("Z") else t0 + "Z"
    t1 = t1 if t1.endswith("Z") else t1 + "Z"
    sel = (
        f"[({t0}):({t1})]"
        f"[({float(min_lat)}):({float(max_lat)})]"
        f"[({float(min_lon)}):({float(max_lon)})]"
    )
    query = f"water_u{sel},water_v{sel}"
    return f"{ERDDAP_BASE}/{dataset_id}.nc?{query}"


def select_backend(end: str) -> str:
    """Pick the download backend from the requested end date.

    Phase 3a implements only the ERDDAP griddap backend (recent window). Dates
    older than ``ERDDAP_WINDOW_DAYS`` require the THREDDS archive backend, which
    is delivered in Phase 3b.
    """
    end_dt = datetime.fromisoformat(normalize_datetime(end))
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    age_days = (now_utc - end_dt).days
    if age_days > ERDDAP_WINDOW_DAYS:
        raise NotImplementedError(
            f"end date {end} is older than the ERDDAP ~{ERDDAP_WINDOW_DAYS}-day "
            "window; the THREDDS/OPeNDAP archive backend is Phase 3b."
        )
    return "erddap"


class NOAAHFRadarDownloader:
    """Download NOAA HFRnet gridded RTV currents via ERDDAP griddap.

    Parameters
    ----------
    output_dir : Path
        Directory to save the downloaded NetCDF.
    dry_run : bool
        If True, print the subset URL and return None without fetching.
    resolution_km : int
        Grid resolution (1/2/6 km); default 6 km for robust coverage.
    """

    def __init__(self, output_dir: Path, dry_run: bool = False,
                 resolution_km: int = DEFAULT_RESOLUTION_KM,
                 force_download: bool = False) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.resolution_km = resolution_km
        self.force_download = force_download

    def download(self, min_lon, max_lon, min_lat, max_lat,
                 start: str, end: str) -> list[Path]:
        windows = split_antimeridian_bbox(min_lon, max_lon)
        downloaded: list[Path] = []
        last_error: Optional[ValueError] = None
        resolved_any = False
        for i, (win_min_lon, win_max_lon) in enumerate(windows):
            suffix = f"_w{i}" if len(windows) > 1 else ""
            try:
                path = self._download_window(
                    win_min_lon, win_max_lon, min_lat, max_lat, start, end, suffix,
                )
            except ValueError as exc:
                if len(windows) == 1:
                    raise
                last_error = exc
                continue
            resolved_any = True
            if path is not None:
                downloaded.append(path)

        if not resolved_any and last_error is not None:
            raise last_error
        return downloaded

    def _download_window(
        self, min_lon, max_lon, min_lat, max_lat, start: str, end: str, filename_suffix: str,
    ) -> Optional[Path]:
        backend = select_backend(end)  # raises if archive (Phase 3b) needed
        dataset_id = select_erddap_dataset(
            min_lon, max_lon, min_lat, max_lat, self.resolution_km
        )
        min_lon, max_lon, min_lat, max_lat = clamp_to_region_bbox(
            min_lon, max_lon, min_lat, max_lat
        )
        start_d = normalize_datetime(start).split("T")[0]
        end_d = normalize_datetime(end).split("T")[0]
        date_str = start_d if start_d == end_d else f"{start_d}_{end_d}"
        out_path = self.output_dir / f"{dataset_id}_{self.resolution_km}km_{date_str}{filename_suffix}.nc"
        url = build_erddap_subset_url(
            dataset_id, min_lon, max_lon, min_lat, max_lat, start, end
        )

        if self.dry_run:
            print(f"[dry-run] NOAA HF-radar ({backend}) would download:\n  {url}")
            return None

        if not self.force_download and out_path.exists():
            print(f"  Already downloaded: {out_path}")
            return out_path

        self.output_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(out_path))
        return out_path


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Download NOAA HFRnet RTV currents (ERDDAP).")
    p.add_argument("--min-lon", type=float, required=True)
    p.add_argument("--max-lon", type=float, required=True)
    p.add_argument("--min-lat", type=float, required=True)
    p.add_argument("--max-lat", type=float, required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION_KM,
                   choices=[1, 2, 6])
    p.add_argument("--output-dir", default="data/hfr_noaa")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    dl = NOAAHFRadarDownloader(
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
        resolution_km=args.resolution,
    )
    out = dl.download(
        args.min_lon, args.max_lon, args.min_lat, args.max_lat,
        args.start, args.end,
    )
    if out is not None:
        print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
