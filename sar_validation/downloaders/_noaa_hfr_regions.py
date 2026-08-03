"""
Shared NOAA HF-radar region table.

Single source of truth for the 6 regions NOAA's HF-radar network covers
(ERDDAP's rolling ~90-day window and/or NCEI THREDDS archive back to 2006):
bbox, THREDDS folder code, ERDDAP dataset-id map (None where ERDDAP has no
dataset at all), and the set of resolutions THREDDS serves. Consumed by
noaa_hfradar_downloader.py (ERDDAP), noaa_hfradar_thredds_downloader.py
(THREDDS), hf_radar_us_downloader.py (the ERDDAP->THREDDS->Copernicus
waterfall), and cli.py (currents template auto-source-selection). See
docs/superpowers/specs/2026-07-31-hf-radar-us-thredds-archive-and-region-expansion-design.md
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Optional, Tuple, TypedDict

__all__ = [
    "NoaaHfrRegion",
    "NOAA_HFR_REGIONS",
    "match_noaa_hfr_region",
    "region_bbox_overlaps",
    "finest_resolution_km",
]


class NoaaHfrRegion(TypedDict):
    bbox: Tuple[float, float, float, float]      # (min_lon, max_lon, min_lat, max_lat)
    thredds_code: str                              # THREDDS folder name, e.g. "USWC"
    erddap_datasets: Optional[Dict[float, str]]    # resolution_km -> ERDDAP dataset id, or None
    thredds_resolutions_km: FrozenSet[float]       # resolution_km values THREDDS serves
    default_resolution_km: float                   # used when a request doesn't override


NOAA_HFR_REGIONS: Dict[str, NoaaHfrRegion] = {
    "US_WEST": {
        "bbox": (-130.36, -115.8056, 30.25, 49.99204),
        "thredds_code": "USWC",
        "erddap_datasets": {1: "ucsdHfrW1", 2: "ucsdHfrW2", 6: "ucsdHfrW6", 0.5: "ucsdHfrW500"},
        "thredds_resolutions_km": frozenset({0.5, 1, 2, 6}),
        "default_resolution_km": 6,
    },
    # Must precede US_EAST_GULF in this dict: US_GREAT_LAKES's small bbox
    # center also falls inside US_EAST_GULF's much larger bbox (lon
    # -97.88..-60, lat 22..46 -- nearly all of the US East Coast/Gulf),
    # and match_noaa_hfr_region() below is a first-match-wins center-point
    # lookup over insertion order. No other pair of regions' bboxes
    # overlap (verified pairwise across all 6), so this is the only
    # order-dependent entry in this table.
    "US_GREAT_LAKES": {
        "bbox": (-85.3587, -84.16428, 45.62711, 46.060886),
        "thredds_code": "GLNA",
        "erddap_datasets": None,
        "thredds_resolutions_km": frozenset({0.5, 1, 2, 6}),
        "default_resolution_km": 6,
    },
    "US_EAST_GULF": {
        "bbox": (-97.88385, -60.0, 22.0, 46.0),
        "thredds_code": "USEGC",
        "erddap_datasets": {1: "ucsdHfrE1", 2: "ucsdHfrE2", 6: "ucsdHfrE6"},
        "thredds_resolutions_km": frozenset({1, 2, 6}),
        "default_resolution_km": 6,
    },
    "US_HAWAII": {
        "bbox": (-163.1444, -151.9565, 16.2204, 24.91688),
        "thredds_code": "USHI",
        "erddap_datasets": {1: "ucsdHfrH1"},
        "thredds_resolutions_km": frozenset({1, 2, 6}),
        "default_resolution_km": 1,
    },
    "US_PRVI": {
        "bbox": (-70.5, -61.0242, 14.5, 21.99766),
        "thredds_code": "PRVI",
        "erddap_datasets": {2: "ucsdHfrP2", 6: "ucsdHfrP6"},
        "thredds_resolutions_km": frozenset({2, 6}),
        "default_resolution_km": 6,
    },
    "US_GULF_OF_ALASKA": {
        "bbox": (-167.0, -123.83641, 50.01798, 61.99266),
        "thredds_code": "GAK",
        "erddap_datasets": None,
        "thredds_resolutions_km": frozenset({2, 6}),
        "default_resolution_km": 6,
    },
}


def _bbox_center(min_lon, max_lon, min_lat, max_lat):
    return (min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0


def match_noaa_hfr_region(min_lon, max_lon, min_lat, max_lat) -> Tuple[str, NoaaHfrRegion]:
    """Return the NOAA_HFR_REGIONS entry whose bbox contains the request's
    center point. Raises ValueError if no region contains it.

    First-match-wins over NOAA_HFR_REGIONS' insertion order: this only
    matters for the one pair of regions whose bboxes overlap
    (US_GREAT_LAKES/US_EAST_GULF -- see the comment on US_GREAT_LAKES's
    entry above), where the more specific region must be listed first.
    """
    clon, clat = _bbox_center(min_lon, max_lon, min_lat, max_lat)
    for name, region in NOAA_HFR_REGIONS.items():
        lo, hi, la, ha = region["bbox"]
        if lo <= clon <= hi and la <= clat <= ha:
            return name, region
    raise ValueError(
        f"No NOAA HF-radar region for bbox center ({clon:.2f}, {clat:.2f}). "
        f"Known regions: {sorted(NOAA_HFR_REGIONS)}"
    )


def region_bbox_overlaps(region: NoaaHfrRegion, min_lon, max_lon, min_lat, max_lat) -> bool:
    """True if the request bbox overlaps the region's bbox at all (not just
    contains its center) -- a deliberately more permissive test than
    match_noaa_hfr_region, used only for the CLI template's
    source-selection heuristic, never for download-time routing."""
    lo, hi, la, ha = region["bbox"]
    lon_overlap = min(max_lon, hi) - max(min_lon, lo)
    lat_overlap = min(max_lat, ha) - max(min_lat, la)
    return lon_overlap > 0 and lat_overlap > 0


def finest_resolution_km(region: NoaaHfrRegion) -> float:
    """min() over the union of erddap_datasets' keys (if any) and
    thredds_resolutions_km -- the smallest (finest) resolution_km reachable
    for this region on any backend."""
    available = set(region["thredds_resolutions_km"])
    if region["erddap_datasets"] is not None:
        available |= set(region["erddap_datasets"])
    return min(available)


def _resolution_token(resolution_km: float) -> str:
    """Map a resolution_km value to the token NOAA's filenames/dataset ids
    use on the wire: "500m" for 0.5, "{N}km" for a whole number."""
    if resolution_km == 0.5:
        return "500m"
    return f"{int(resolution_km)}km"
