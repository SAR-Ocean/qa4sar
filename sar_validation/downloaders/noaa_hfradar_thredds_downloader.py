"""
Download NOAA HFRnet gridded surface currents from the NCEI THREDDS archive
(2006-present, published a few weeks behind real-time).

Complements noaa_hfradar_downloader.py's ERDDAP backend (rolling ~90-day
window): hf_radar_us_downloader.py's waterfall tries ERDDAP first, then
this module, then Copernicus.

Catalog layout: thredds-ocean/catalog/ioos/hfradar/rtv/{YYYY}/{YYYYMM}/{REGION}/
One file per hour per resolution. Filename format changed on 2025-07-01
(verified live, bisected month-by-month):
  - Before: {YYYYMMDDHHMM}_hfr_{region}_{res}_rtv_uwls_{SIO|NDBC|25hr_average_SIO}.nc
    -- only the _NDBC variant matches ERDDAP's own pipeline (same
    institution/creator_name); _SIO/_25hr_average_SIO are different products
    and are never used.
  - From 2025-07-01: rtv-{region}-{res}-uwls_v1r0_hfr_s{start}_e{end}_c{created}.nc
    -- one unified file per hour/resolution, no provider suffix.

Wire variable names are u/v (both eras) with the same CF standard_names as
ERDDAP's water_u/water_v -- renamed on read so
DataTreeConverter.from_hf_radar_grid's existing defaults need no change.

Library usage::

    from sar_validation.downloaders.noaa_hfradar_thredds_downloader import (
        NOAATHREDDSHFRadarDownloader,
    )
    dl = NOAATHREDDSHFRadarDownloader(output_dir=Path("data/run1/hfr_noaa"))
    dl.download(min_lon=-125, max_lon=-119, min_lat=33, max_lat=38,
                start="2024-01-31", end="2024-01-31")
"""

from __future__ import annotations

import logging
import re
import shutil
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr

from ._noaa_hfr_regions import _resolution_token, match_noaa_hfr_region
from .base import months_touched, normalize_datetime, prefer_ipv4_dns, split_antimeridian_bbox

logger = logging.getLogger(__name__)

__all__ = ["NOAATHREDDSHFRadarDownloader"]

THREDDS_BASE = "https://www.ncei.noaa.gov/thredds-ocean"
_CATALOG_NS = "{http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0}"

_OLD_PATTERN = re.compile(
    r"^(?P<ts>\d{12})_hfr_[a-z]+_(?P<res>\d+km|500m)_rtv_uwls_NDBC\.nc$"
)
_NEW_PATTERN = re.compile(
    r"^rtv-[a-z]+-(?P<res>\d+km|500m)-uwls_v1r0_hfr_s(?P<ts>\d{15})_e\d{15}_c\d{15}\.nc$"
)


def _list_thredds_granules(
    catalog_xml_text: str, resolution_km: float, start: datetime, end: datetime,
    end_exclusive: bool = False,
) -> List[Tuple[datetime, str]]:
    """Parse one month's THREDDS catalog.xml and return (timestamp, urlPath)
    for every granule at *resolution_km* whose timestamp falls in
    [start, end] (or [start, end) if *end_exclusive* is True). Handles both
    filename eras; only the pre-2025-07-01 era's _NDBC variant is matched
    (_SIO/_25hr_average_SIO are excluded).

    *end_exclusive* is for callers that have already widened *end* to the
    start of the day after a date-only bound (see _download_window) and need
    a strict upper bound so that widening doesn't pull in the next day's
    granules too. Direct full-precision-datetime callers should leave this
    False to keep the inclusive <= semantics on both bounds."""
    token = _resolution_token(resolution_km)
    root = ET.fromstring(catalog_xml_text)
    results: List[Tuple[datetime, str]] = []
    for ds_elem in root.iter(f"{_CATALOG_NS}dataset"):
        name = ds_elem.get("name")
        url_path = ds_elem.get("urlPath")
        if not name or not url_path:
            continue
        m = _OLD_PATTERN.match(name)
        if m:
            if m.group("res") != token:
                continue
            ts = datetime.strptime(m.group("ts"), "%Y%m%d%H%M")
        else:
            m = _NEW_PATTERN.match(name)
            if not m or m.group("res") != token:
                continue
            ts = datetime.strptime(m.group("ts")[:12], "%Y%m%d%H%M")
        end_ok = ts < end if end_exclusive else ts <= end
        if start <= ts and end_ok:
            results.append((ts, url_path))
    return sorted(results)


class NOAATHREDDSHFRadarDownloader:
    """Download a NOAA HF-radar current grid from the NCEI THREDDS archive.

    Parameters
    ----------
    output_dir : Path
        Directory to save the merged NetCDF.
    dry_run : bool
        If True, print what would be fetched and return [] without network calls.
    resolution_km : float
        Grid resolution (0.5/1/2/6 km depending on region); default 6.
    """

    def __init__(
        self, output_dir: Path, dry_run: bool = False,
        resolution_km: float = 6, force_download: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.resolution_km = resolution_km
        self.force_download = force_download

    def download(self, min_lon, max_lon, min_lat, max_lat, start: str, end: str) -> list[Path]:
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
        region_name, region = match_noaa_hfr_region(min_lon, max_lon, min_lat, max_lat)
        thredds_code = region["thredds_code"]
        start_dt = datetime.fromisoformat(normalize_datetime(start))
        end_dt = datetime.fromisoformat(normalize_datetime(end))
        token = _resolution_token(self.resolution_km)

        # A date-only end (e.g. "2026-06-30") normalizes to that date's
        # midnight, but hourly granules are timestamped all day long -- widen
        # the matching bound to cover the whole day. end_dt itself stays
        # un-expanded since it's also used below for the output filename.
        is_end_date_only = len(end.strip().rstrip("Z")) == 10  # YYYY-MM-DD
        end_dt_for_matching = end_dt + timedelta(days=1) if is_end_date_only else end_dt

        if self.dry_run:
            print(
                f"[dry-run] NOAA THREDDS HF-radar ({region_name}, {token}) would search "
                f"{THREDDS_BASE}/catalog/ioos/hfradar/rtv/{{YYYY}}/{{YYYYMM}}/{thredds_code}/"
                f"catalog.xml for [{start_dt}, {end_dt_for_matching}]"
            )
            return None

        logger.info(
            "hf_radar_thredds: resolved region=%s resolution=%s, searching "
            "catalogs for [%s, %s]", region_name, token, start_dt, end_dt_for_matching,
        )

        granules: List[Tuple[datetime, str]] = []
        any_catalog_fetched = False
        # Walk calendar months using the un-expanded end_dt: the expanded
        # end_dt_for_matching bound only widens which hours within a day are
        # matched, it never changes which month's catalog folder holds them.
        for year, month in months_touched(start_dt, end_dt):
            catalog_url = (
                f"{THREDDS_BASE}/catalog/ioos/hfradar/rtv/{year}/{year}{month:02d}/"
                f"{thredds_code}/catalog.xml"
            )
            try:
                with prefer_ipv4_dns(), urllib.request.urlopen(catalog_url, timeout=15) as resp:
                    text = resp.read().decode()
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    continue
                raise
            any_catalog_fetched = True
            granules.extend(
                _list_thredds_granules(
                    text, self.resolution_km, start_dt, end_dt_for_matching,
                    end_exclusive=is_end_date_only,
                )
            )

        logger.info("hf_radar_thredds: %d granule(s) matched", len(granules))

        if not granules:
            if not any_catalog_fetched:
                raise ValueError(
                    f"No THREDDS archive data for {region_name} in "
                    f"[{start_dt}, {end_dt_for_matching}] at any resolution -- either "
                    "before this region's coverage starts, or in the not-yet-published "
                    "tail (THREDDS lags real-time by a few weeks)."
                )
            return None

        granules.sort()
        out_path = self.output_dir / (
            f"{thredds_code}_{token}_thredds_"
            f"{start_dt.date()}_{end_dt.date()}{filename_suffix}.nc"
        )
        if not self.force_download and out_path.exists():
            print(f"  Already downloaded: {out_path}")
            return out_path

        self.output_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = self.output_dir / f".thredds_tmp{filename_suffix}"
        tmp_dir.mkdir(exist_ok=True)
        try:
            granule_paths = []
            for n, (ts, url_path) in enumerate(granules, start=1):
                logger.info(
                    "hf_radar_thredds: downloading granule %d/%d", n, len(granules),
                )
                dest = tmp_dir / Path(url_path).name
                with prefer_ipv4_dns(), urllib.request.urlopen(
                    f"{THREDDS_BASE}/fileServer/{url_path}", timeout=15
                ) as resp:
                    dest.write_bytes(resp.read())
                granule_paths.append(dest)

            datasets = [
                xr.open_dataset(p).rename({"u": "water_u", "v": "water_v"})
                for p in granule_paths
            ]
            merged = xr.concat(datasets, dim="time").sortby("time")
            # Trim to the requested bbox using integer-index selection rather
            # than .where(..., drop=True): .where() broadcasts its condition
            # (and thus the whole dataset) across every variable, including
            # non-gridded ones like time_bnds(time, nv) and scalar CF
            # metadata variables (wgs84, processing_parameters,
            # radial_metadata) that don't have lat/lon dims at all -- those
            # get spuriously broadcast to full lat/lon grids and upcast
            # (e.g. int8 -> float32) by .where()'s NaN-filling machinery.
            # .isel() only touches the lat/lon dimensions and leaves every
            # other variable untouched.
            lat_idx = np.nonzero(
                (merged.lat.values >= min_lat) & (merged.lat.values <= max_lat)
            )[0]
            lon_idx = np.nonzero(
                (merged.lon.values >= min_lon) & (merged.lon.values <= max_lon)
            )[0]
            if lat_idx.size == 0 or lon_idx.size == 0:
                raise ValueError(
                    f"THREDDS granules for {region_name} matched the search window "
                    f"but none of their grid points fall within the requested bbox "
                    f"[{min_lon}, {max_lon}, {min_lat}, {max_lat}] -- the region's "
                    f"native grid at this resolution likely doesn't cover this sub-area."
                )
            merged = merged.isel(lat=lat_idx, lon=lon_idx)
            merged.to_netcdf(out_path)
            for ds in datasets:
                ds.close()
            for p in granule_paths:
                p.unlink()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return out_path
