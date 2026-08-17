"""
Download NOAA HFRnet gridded surface currents (Real-Time Velocities, RTV).

Backend for the recent ~90-day ERDDAP griddap window. Dates older than that
are served by the THREDDS archive backend
(noaa_hfradar_thredds_downloader.py); hf_radar_us_downloader.py's waterfall
tries this module first, then THREDDS, then Copernicus.

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
from typing import Optional

from ._noaa_hfr_regions import _resolution_token, match_noaa_hfr_region
from .base import normalize_datetime, prefer_ipv4_dns, split_antimeridian_bbox

__all__ = [
    "NOAAHFRadarDownloader",
    "select_erddap_dataset",
    "build_erddap_subset_url",
    "select_backend",
    "clamp_to_region_bbox",
]

# ERDDAP griddap host serving the UCSD HFRnet RTV datasets.
ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
# ERDDAP keeps a rolling ~3-month window; older dates need the THREDDS archive.
ERDDAP_WINDOW_DAYS = 90
DEFAULT_RESOLUTION_KM = 6


def select_erddap_dataset(min_lon, max_lon, min_lat, max_lat, resolution_km: float) -> str:
    """Choose the ERDDAP RTV dataset id from the request bbox and resolution.

    Raises ``ValueError`` if the region is outside NOAA's coverage, has no
    ERDDAP dataset at all (Great Lakes/Gulf of Alaska), or doesn't offer
    the requested resolution.
    """
    name, region = match_noaa_hfr_region(min_lon, max_lon, min_lat, max_lat)
    datasets = region["erddap_datasets"]
    if datasets is None:
        raise ValueError(
            f"region {name} has no ERDDAP dataset at all (THREDDS-only region); "
            "use the THREDDS backend instead."
        )
    if resolution_km not in datasets:
        available = ", ".join(_resolution_token(r) for r in sorted(datasets))
        raise ValueError(
            f"resolution {_resolution_token(resolution_km)} not available for region "
            f"{name}; available: {available}"
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
    _, region = match_noaa_hfr_region(min_lon, max_lon, min_lat, max_lat)
    lo, hi, la, ha = region["bbox"]
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

    Dates within ERDDAP_WINDOW_DAYS use ERDDAP; older dates require the
    THREDDS archive backend (noaa_hfradar_thredds_downloader.py), which
    hf_radar_us_downloader.py's waterfall tries next.
    """
    end_dt = datetime.fromisoformat(normalize_datetime(end))
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    age_days = (now_utc - end_dt).days
    if age_days > ERDDAP_WINDOW_DAYS:
        raise NotImplementedError(
            f"end date {end} is older than the ERDDAP ~{ERDDAP_WINDOW_DAYS}-day "
            "window; use the THREDDS archive backend."
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
    resolution_km : float
        Grid resolution (0.5/1/2/6 km, region-dependent); default 6 km for
        robust coverage.
    """

    def __init__(self, output_dir: Path, dry_run: bool = False,
                 resolution_km: float = DEFAULT_RESOLUTION_KM,
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
        backend = select_backend(end)  # raises if the THREDDS archive backend is needed
        dataset_id = select_erddap_dataset(
            min_lon, max_lon, min_lat, max_lat, self.resolution_km
        )
        min_lon, max_lon, min_lat, max_lat = clamp_to_region_bbox(
            min_lon, max_lon, min_lat, max_lat
        )
        start_d = normalize_datetime(start).split("T")[0]
        end_d = normalize_datetime(end).split("T")[0]
        date_str = start_d if start_d == end_d else f"{start_d}_{end_d}"
        res_token = _resolution_token(self.resolution_km)
        out_path = self.output_dir / f"{dataset_id}_{res_token}_{date_str}{filename_suffix}.nc"
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
        with prefer_ipv4_dns(), urllib.request.urlopen(url, timeout=15) as resp:
            out_path.write_bytes(resp.read())
        return out_path


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Download NOAA HFRnet RTV currents (ERDDAP).")
    p.add_argument("--min-lon", type=float, required=True)
    p.add_argument("--max-lon", type=float, required=True)
    p.add_argument("--min-lat", type=float, required=True)
    p.add_argument("--max-lat", type=float, required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION_KM,
                   choices=[0.5, 1, 2, 6])
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
