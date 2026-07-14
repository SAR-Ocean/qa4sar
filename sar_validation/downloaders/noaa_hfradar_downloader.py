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
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from .base import normalize_datetime

__all__ = [
    "NOAAHFRadarDownloader",
    "select_erddap_dataset",
    "build_erddap_subset_url",
    "select_backend",
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
        "bbox": (-130.0, -116.0, 30.0, 50.0),
        "datasets": {1: "ucsdHfrW1", 2: "ucsdHfrW2", 6: "ucsdHfrW6"},
    },
    "US_EAST_GULF": {
        "bbox": (-98.0, -60.0, 22.0, 46.0),
        "datasets": {1: "ucsdHfrE1", 6: "ucsdHfrE6"},
    },
}


def _bbox_center(min_lon, max_lon, min_lat, max_lat):
    return (min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0


def select_erddap_dataset(min_lon, max_lon, min_lat, max_lat, resolution_km: int) -> str:
    """Choose the ERDDAP RTV dataset id from the request bbox and resolution.

    Raises ``ValueError`` if the region is outside the supported CONUS coasts
    or the requested resolution is unavailable for that region.
    """
    clon, clat = _bbox_center(min_lon, max_lon, min_lat, max_lat)
    for name, cfg in _REGIONS.items():
        lo, hi, la, ha = cfg["bbox"]
        if lo <= clon <= hi and la <= clat <= ha:
            datasets = cfg["datasets"]
            if resolution_km not in datasets:
                raise ValueError(
                    f"resolution {resolution_km} km not available for region "
                    f"{name}; available: {sorted(datasets)} km"
                )
            return datasets[resolution_km]
    raise ValueError(
        "No ERDDAP HF-radar dataset for bbox center "
        f"({clon:.2f}, {clat:.2f}). Phase 3a supports US West and US "
        "East/Gulf coasts; other regions arrive in later phases."
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
    end_norm = normalize_datetime(end).rstrip("Z")
    end_dt = datetime.fromisoformat(end_norm)
    age_days = (datetime.utcnow() - end_dt).days
    if age_days > ERDDAP_WINDOW_DAYS:
        raise NotImplementedError(
            f"end date {end} is older than the ERDDAP ~{ERDDAP_WINDOW_DAYS}-day "
            "window; the THREDDS/OPeNDAP archive backend is Phase 3b."
        )
    return "erddap"


class NOAAHFRadarDownloader:
    """See Task 4 for the full implementation."""

    def __init__(self, output_dir: Path, dry_run: bool = False,
                 resolution_km: int = DEFAULT_RESOLUTION_KM) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.resolution_km = resolution_km
