"""
Predict collocation between a recipe's SAR data and its configured
validation sources, before downloading anything from either side.

See docs/superpowers/specs/2026-08-18-dry-collocation-design.md.

This module is built up across four sequential implementation plans:
  1. orbit_coverage.py foundation (orbit_overlap_windows, _point_in_polygon)
  2. This file's SarFootprint model + SAR-side discovery (this plan)
  3. Per-validation-source-type predicates (SourcePrediction, predict_source)
  4. Orchestrator/CLI wiring (predict_collocation, CollocationReport)
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from ..downloaders import radarsat2_wind_downloader as _rs2
from ..downloaders._hf_radar_regions import HFR_REGIONS
from ..downloaders._noaa_hfr_regions import NOAA_HFR_REGIONS
from ..downloaders.altimeter_downloader import AltimeterDownloader
from ..downloaders.ascat_soil_moisture_downloader import ASCATSoilMoistureDownloader
from ..downloaders.base import (
    CopernicusODataClient,
    authenticate_cdse,
    months_touched,
    normalize_datetime,
    prefer_ipv4_dns,
    split_antimeridian_bbox,
)
from ..downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader
from ..downloaders.earthdata_soil_moisture_downloader import EarthdataSoilMoistureDownloader
from ..downloaders.era5_downloader import ERA5Downloader
from ..downloaders.hf_radar_downloader import HFRadarDownloader
from ..downloaders.hf_radar_historical_downloader import HFRadarHistoricalDownloader
from ..downloaders.hf_radar_us_downloader import HFRadarUSDownloader
from ..downloaders.hsaf_downloader import HSAFDownloader
from ..downloaders.hsaf_downloader import _parse_satellite as _hsaf_parse_satellite
from ..downloaders.hycom_downloader import _HYCOM_MIN_DATE, HycomDownloader, _resolve_hycom_segments
from ..downloaders.ismn_downloader import ISMNDownloader
from ..downloaders.noaa_hfradar_downloader import NOAAHFRadarDownloader
from ..downloaders.radiometer_downloader import RadiometerDownloader
from ..downloaders.scatterometer_downloader import ScatterometerDownloader
from ..downloaders.scatterometer_ftp_downloader import ScatterometerFTPDownloader
from ..downloaders.smos_downloader import SMOSDownloader
from . import orbit_coverage
from .collocation import _haversine_distance
from .datatree_converter import DataTreeConverter
from .orbit_coverage import _point_in_bbox
from .orchestrator import _resolve_temporal_padding_minutes

logger = logging.getLogger(__name__)

__all__ = [
    "SarFootprint",
    "Verdict",
    "SourcePrediction",
    "CollocationReport",
    "predict_source",
    "predict_collocation",
    "render_console_table",
    "report_to_json",
]

#: Registered Sentinel-1 orbit specs (orbit_coverage.SATELLITE_ORBIT_SPECS)
#: to try per CLMS SSM tile-day candidate -- a tile could have been
#: covered by any of the three, so every one is tried and their matched
#: windows are unioned (see _discover_clms_ssm_footprints_dry).
_CLMS_SSM_SATELLITES = ("sentinel-1a", "sentinel-1b", "sentinel-1c")

_WV_MODE_RE = re.compile(r"^S1[A-D]_WV_")


@dataclass(frozen=True)
class SarFootprint:
    """One SAR granule's real (or, for CLMS SSM's dry-search path,
    orbit-predicted) footprint + acquisition time window.

    kind="polygon": Sentinel-1 OCN (non-WV) / NISAR SME2 -- polygon holds
    real vertices when the source's catalog provides them (RADARSAT-2
    doesn't -- see this plan's "Correction found during planning" note --
    so its footprints leave polygon=None despite kind="polygon").

    kind="wv_points": Sentinel-1 WV mode -- points holds one (lat, lon)
    per sparse ~20x20km vignette; polygon is always None.

    kind="orbit_swath": Sentinel-1 CLMS SSM -- bbox is the predicted (dry
    path) or real non-NaN-pixel (from-downloaded path) overpass extent
    within the product's much larger nominal tile; polygon is always None.
    """

    kind: Literal["polygon", "wv_points", "orbit_swath"]
    bbox: Tuple[float, float, float, float]  # (min_lon, max_lon, min_lat, max_lat)
    polygon: Optional[List[Tuple[float, float]]]  # (lat, lon) vertices, kind="polygon" only
    points: Optional[List[Tuple[float, float]]]  # (lat, lon) per vignette, kind="wv_points" only
    sensing_start: datetime
    sensing_end: datetime
    source_file: str


def _query_sentinel1_ocn_dry(cfg) -> "list[dict]":
    """Query CDSE for Sentinel-1 OCN products matching cfg's bbox/window,
    without downloading anything -- reuses SARDownloader's exact
    collection/product_type values (see sentinel1_l2_ocn_downloader.py)."""
    username, password = authenticate_cdse()
    client = CopernicusODataClient(username, password)
    # CDSE's OData $filter rejects anything but full ISO-8601 with a Z
    # suffix -- same normalization the real downloader applies (see
    # SARDownloader.query in sentinel1_l2_ocn_downloader.py) before this
    # value ever reaches query_products.
    start_norm = normalize_datetime(cfg.temporal_bounds.start) + ".000Z"
    end_norm = normalize_datetime(cfg.temporal_bounds.end) + ".000Z"
    return client.query_products(
        collection="SENTINEL-1",
        product_type="OCN",
        start_date=start_norm,
        end_date=end_norm,
        min_lon=cfg.geographic_bounds.min_lon,
        max_lon=cfg.geographic_bounds.max_lon,
        min_lat=cfg.geographic_bounds.min_lat,
        max_lat=cfg.geographic_bounds.max_lat,
    )


def _fallback_bbox_from_cfg(cfg) -> "tuple[float, float, float, float]":
    """The recipe's own requested bbox, used as the conservative
    fallback when a CDSE record's GeoFootprint is missing/malformed.
    cfg is a full RecipeConfig (bbox nested under .geographic_bounds)."""
    bounds = cfg.geographic_bounds
    return (bounds.min_lon, bounds.max_lon, bounds.min_lat, bounds.max_lat)


def _polygon_and_bbox_from_geofootprint(
    geofootprint: "Optional[dict]", fallback_bbox_fn: "Callable[[], tuple[float, float, float, float]]",
) -> "tuple[Optional[list[tuple[float, float]]], tuple[float, float, float, float]]":
    """(polygon, bbox) from a CDSE GeoFootprint GeoJSON object -- GeoJSON
    coordinates are [lon, lat] pairs; SarFootprint's own convention is
    (lat, lon), so they're swapped here. GeoJSON rings repeat their first
    vertex to close the loop -- that trailing duplicate is dropped so
    SarFootprint.polygon holds one entry per distinct vertex. Falls back
    to (None, fallback_bbox_fn()) when geofootprint is missing/malformed
    -- never raises, since a missing footprint must degrade to "less
    precise", not "drop this granule". fallback_bbox_fn is only called
    (lazily) when actually needed, since deriving it may require
    attributes the caller's cfg doesn't always carry (e.g. when
    GeoFootprint is present, cfg's own bbox is never consulted)."""
    if not geofootprint or geofootprint.get("type") != "Polygon":
        return None, fallback_bbox_fn()
    try:
        ring = geofootprint["coordinates"][0]
        if len(ring) > 1 and ring[0] == ring[-1]:
            ring = ring[:-1]
        polygon = [(lat, lon) for lon, lat in ring]
        lons = [lon for lon, _lat in ring]
        lats = [lat for _lon, lat in ring]
        bbox = (min(lons), max(lons), min(lats), max(lats))
        return polygon, bbox
    except (KeyError, IndexError, TypeError, ValueError):
        return None, fallback_bbox_fn()


def _vignette_points_and_bbox_from_geofootprint(
    geofootprint: "Optional[dict]",
    bounds,
    fallback_bbox_fn: "Callable[[], tuple[float, float, float, float]]",
) -> "tuple[list[tuple[float, float]], tuple[float, float, float, float]]":
    """(points, bbox) from a CDSE WV-mode GeoFootprint. CDSE catalogs an
    entire WV pass as one product, not one vignette per catalog entry --
    GeoFootprint's type is "MultiPolygon" and its "coordinates" list
    already holds one small ~20-30km quad ring per vignette, directly in
    the catalog search response. No manifest.safe fetch via CDSE's
    /Products({id})/Nodes(...) endpoint is needed. Each
    ``coordinates[i]`` entry is always ``[exterior_ring]`` -- a
    single-element list holding that vignette's one ring of [lon, lat]
    pairs (no interior holes, no flatter variant) -- so the ring is read
    via a fixed ``polygon_entry[0]`` descent, mirroring
    _polygon_and_bbox_from_geofootprint's ``coordinates[0]`` pattern.

    Each ring's centroid becomes one point: the mean of its deduped
    unique [lon, lat] vertices (GeoJSON rings repeat their first vertex
    to close the loop, so that trailing duplicate is dropped first --
    mirroring _polygon_and_bbox_from_geofootprint's convention).

    A WV product's overall catalog envelope only guarantees it TOUCHES
    the recipe's requested bbox (see query_products) -- real WV products
    can span very large geographic extents (a single product can span
    hundreds of km along-track with 100+ vignettes) over open ocean,
    so most of a product's vignettes are typically geographically
    irrelevant to a realistically-sized recipe bbox. Each centroid is
    therefore filtered against *bounds* (a GeographicBounds-shaped object
    with min_lon/max_lon/min_lat/max_lat attributes -- normally
    cfg.geographic_bounds) via the same antimeridian-aware
    orbit_coverage._point_in_bbox convention used everywhere else in this
    codebase (bounds.min_lon > bounds.max_lon signals a bbox wrapping the
    antimeridian). bbox is the enclosing box over the SURVIVING
    (in-bounds) vignette centroids only, not over the raw vertices or the
    unfiltered centroid set, since points -- not polygon -- is what a
    wv_points footprint is compared against downstream.

    Falls back to ([], fallback_bbox_fn()) when geofootprint is
    missing/malformed, contains zero usable vignette rings, or every
    vignette centroid falls outside *bounds* -- never raises, since a
    missing/broken/out-of-region footprint must degrade to "less
    precise" (an empty points list is still a valid wv_points footprint),
    not "drop this granule"."""
    if not geofootprint or geofootprint.get("type") != "MultiPolygon":
        return [], fallback_bbox_fn()
    try:
        points: "list[tuple[float, float]]" = []
        for polygon_entry in geofootprint["coordinates"]:
            ring = polygon_entry[0]
            vertices = ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring
            if not vertices:
                continue
            lons = [v[0] for v in vertices]
            lats = [v[1] for v in vertices]
            lat, lon = sum(lats) / len(lats), sum(lons) / len(lons)
            if _point_in_bbox(lat, lon, bounds.min_lon, bounds.max_lon, bounds.min_lat, bounds.max_lat):
                points.append((lat, lon))
        if not points:
            return [], fallback_bbox_fn()
        centroid_lons = [lon for _lat, lon in points]
        centroid_lats = [lat for lat, _lon in points]
        bbox = (min(centroid_lons), max(centroid_lons), min(centroid_lats), max(centroid_lats))
        return points, bbox
    except (KeyError, IndexError, TypeError, ValueError):
        return [], fallback_bbox_fn()


def _discover_sentinel1_wv_footprints_dry(cfg) -> "list[SarFootprint]":
    """WV-mode Sentinel-1 footprints from a dry CDSE catalog search --
    see _vignette_points_and_bbox_from_geofootprint for why CDSE catalogs
    an entire WV pass (many vignettes) as one product, with per-vignette
    geometry already in the catalog response, and for why each vignette
    centroid is filtered against cfg.geographic_bounds before being kept
    (a WV product's own catalog envelope only guarantees it touches the
    recipe's requested bbox, not that most -- or any -- of its individual
    vignettes fall inside it). Non-WV granules are excluded here -- see
    _discover_sentinel1_ocn_footprints_dry."""
    records = _query_sentinel1_ocn_dry(cfg)
    footprints = []
    for record in records:
        if not _WV_MODE_RE.match(record["Name"]):
            continue
        points, bbox = _vignette_points_and_bbox_from_geofootprint(
            record.get("GeoFootprint"), cfg.geographic_bounds, lambda: _fallback_bbox_from_cfg(cfg),
        )
        footprints.append(
            SarFootprint(
                kind="wv_points",
                bbox=bbox,
                polygon=None,
                points=points,
                sensing_start=datetime.fromisoformat(record["ContentDate_Start"].replace("Z", "+00:00")),
                sensing_end=datetime.fromisoformat(record["ContentDate_End"].replace("Z", "+00:00")),
                source_file=record["Name"],
            )
        )
    return footprints


def _list_radarsat2_candidates_dry(cfg) -> "list[tuple[str, datetime]]":
    """(url_path, sensing_time) pairs for every RADARSAT-2 candidate in
    cfg's window, via the same THREDDS catalog.xml listing
    radarsat2_wind_downloader.py's real download() already does --
    filename-embedded center point only, no NCML fetch yet. Mirrors
    RADARSAT2WindDownloader._download_window's month-loop exactly (one
    catalog.xml fetch per touched (year, month), 404 == "no catalog for
    this month, skip", any other HTTP error re-raised).

    Like RADARSAT2WindDownloader.download(), an antimeridian-crossing
    cfg.geographic_bounds (min_lon > max_lon, per GeographicBounds'
    convention) is pre-split via split_antimeridian_bbox into one or two
    non-crossing windows, each searched independently (its own
    months-touched catalog.xml loop) with the results concatenated --
    _query_sentinel1_ocn_dry's sibling doesn't need this since CDSE's own
    query_products handles antimeridian-crossing bboxes server-side."""
    bounds = cfg.geographic_bounds
    start_dt = datetime.fromisoformat(normalize_datetime(cfg.temporal_bounds.start))
    end_dt = datetime.fromisoformat(normalize_datetime(cfg.temporal_bounds.end))

    candidates: "list[tuple[str, datetime]]" = []
    for win_min_lon, win_max_lon in split_antimeridian_bbox(bounds.min_lon, bounds.max_lon):
        for year, month in months_touched(start_dt, end_dt):
            catalog_url = f"{_rs2.THREDDS_BASE}/catalog/sar-winds/radarsat2/{year}/{month:02d}/catalog.xml"
            try:
                with prefer_ipv4_dns(), urllib.request.urlopen(catalog_url, timeout=15) as resp:
                    text = resp.read().decode()
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    continue
                raise
            for ts, url_path, _lon, _lat in _rs2._list_radarsat2_granules(
                text, start_dt, end_dt,
                win_min_lon, win_max_lon, bounds.min_lat, bounds.max_lat,
            ):
                candidates.append((url_path, ts))
    return candidates


def _radarsat2_ncml_bbox(url_path: str) -> "Optional[tuple[float, float, float, float]]":
    """(min_lon, max_lon, min_lat, max_lat) from the granule's NCML
    metadata, or None if unavailable -- fetches the NCML document
    exactly as RADARSAT2WindDownloader._passes_ncml_check already does
    (same URL template, same prefer_ipv4_dns() + urlopen call shape),
    then delegates the parse to radarsat2_wind_downloader._parse_ncml_bbox.
    Unlike _passes_ncml_check, this never fails open to True/False -- it
    has no requested bbox to compare against here, it just reports what
    the metadata says (or None on any fetch/parse failure), leaving the
    "fall back to the recipe's own bbox" decision to the caller."""
    try:
        with prefer_ipv4_dns(), urllib.request.urlopen(
            f"{_rs2.THREDDS_BASE}/ncml/{url_path}", timeout=15
        ) as resp:
            text = resp.read().decode()
    except urllib.error.URLError:
        # Covers HTTPError too (it subclasses URLError).
        return None
    return _rs2._parse_ncml_bbox(text)


def _discover_radarsat2_footprints_dry(cfg) -> "list[SarFootprint]":
    """RADARSAT-2 footprints from a dry THREDDS/NCML search. polygon is
    always None -- RADARSAT-2's NCML metadata only exposes a bounding
    box (geospatial_lon_min/max, geospatial_lat_min/max), never real
    vertices, unlike CDSE's GeoFootprint (see this plan's correction
    note at the top of this document). Falls back to the recipe's own
    requested bbox (cfg.geographic_bounds) when a candidate's NCML bbox
    is unavailable -- cfg is only consulted for that fallback lazily,
    mirroring _polygon_and_bbox_from_geofootprint's fallback_bbox_fn
    convention elsewhere in this module, since cfg need not carry
    geographic_bounds at all when every candidate's NCML check succeeds."""
    footprints = []
    for url_path, sensing_time in _list_radarsat2_candidates_dry(cfg):
        bbox = _radarsat2_ncml_bbox(url_path) or _fallback_bbox_from_cfg(cfg)
        footprints.append(
            SarFootprint(
                kind="polygon", bbox=bbox, polygon=None, points=None,
                sensing_start=sensing_time, sensing_end=sensing_time,
                source_file=url_path,
            )
        )
    return footprints


def _search_nisar_sme2_dry(cfg) -> "list[dict]":
    """earthaccess/CMR granule search for NISAR SME2, no download --
    searches BOTH (short_name, version) candidates in
    sar_sources.NISAR_SME2_CANDIDATES and merges their results, mirroring
    EarthdataSoilMoistureDownloader.download()'s own per-candidate search
    loop (earthdata_soil_moisture_downloader.py). NISAR SME2's underlying
    CMR collection changed mid-mission with no temporal overlap between its
    beta and provisional product-maturity levels (see sar_sources.py's own
    comment on NISAR_SME2_CANDIDATES) -- searching only the first candidate
    would silently miss real granules whenever cfg's window falls (even
    partially) in the provisional-only range."""
    import earthaccess

    from ..downloaders.base import authenticate_earthdata
    from .sar_sources import NISAR_SME2_CANDIDATES

    authenticate_earthdata()

    bounds = cfg.geographic_bounds
    all_results: "list[dict]" = []
    for short_name, version in NISAR_SME2_CANDIDATES:
        all_results.extend(
            earthaccess.search_data(
                short_name=short_name,
                version=version,
                bounding_box=(bounds.min_lon, bounds.min_lat, bounds.max_lon, bounds.max_lat),
                temporal=(cfg.temporal_bounds.start, cfg.temporal_bounds.end),
            )
        )
    return all_results


def _discover_nisar_sme2_footprints_dry(cfg) -> "list[SarFootprint]":
    """NISAR SME2 footprints from a dry earthaccess/CMR search across both
    beta/provisional candidates -- see _search_nisar_sme2_dry.

    This collection's granules publish GPolygons with real vertex
    geometry (unlike RADARSAT-2's NCML, which is bbox-only), so that path
    is the primary one here, dropping a ring's repeated closing vertex
    when present (CMR's GPolygons close their ring the same way GeoJSON
    does). A BoundingRectangles-only fallback is kept for defensiveness
    (a different/future NISAR collection, or an edge-case granule, might
    lack GPolygons), with the recipe's own requested bbox as the final
    fallback for that tier -- the same "degrade, don't drop" convention
    used throughout this module. Unlike CDSE's optional GeoFootprint,
    CMR's UMM-G schema guarantees every granule carries TemporalExtent
    and SpatialExtent.HorizontalSpatialDomain.Geometry, so those outer
    lookups are not defensively wrapped -- but the
    GPolygons-vs-BoundingRectangles parsing within Geometry is wrapped in
    a per-granule try/except, matching
    _polygon_and_bbox_from_geofootprint's convention: a malformed
    Geometry (missing "Boundary"/"Points"/coordinate keys) degrades that
    one granule to the cfg-bbox fallback instead of aborting discovery
    for the whole batch."""
    footprints = []
    for granule in _search_nisar_sme2_dry(cfg):
        umm = granule["umm"]
        geom = umm["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]
        try:
            gpolygons = geom.get("GPolygons")
            if gpolygons:
                pts = gpolygons[0]["Boundary"]["Points"]
                if len(pts) > 1 and (pts[0]["Latitude"], pts[0]["Longitude"]) == (
                    pts[-1]["Latitude"], pts[-1]["Longitude"]
                ):
                    pts = pts[:-1]
                polygon = [(p["Latitude"], p["Longitude"]) for p in pts]
                lons = [p["Longitude"] for p in pts]
                lats = [p["Latitude"] for p in pts]
                bbox = (min(lons), max(lons), min(lats), max(lats))
            elif "BoundingRectangles" in geom:
                polygon = None
                rect = geom["BoundingRectangles"][0]
                bbox = (
                    rect["WestBoundingCoordinate"], rect["EastBoundingCoordinate"],
                    rect["SouthBoundingCoordinate"], rect["NorthBoundingCoordinate"],
                )
            else:
                polygon = None
                bbox = _fallback_bbox_from_cfg(cfg)
        except (KeyError, IndexError, TypeError, ValueError):
            polygon = None
            bbox = _fallback_bbox_from_cfg(cfg)
        temporal = umm["TemporalExtent"]["RangeDateTime"]
        footprints.append(
            SarFootprint(
                kind="polygon",
                bbox=bbox,
                polygon=polygon,
                points=None,
                sensing_start=datetime.fromisoformat(temporal["BeginningDateTime"].replace("Z", "+00:00")),
                sensing_end=datetime.fromisoformat(temporal["EndingDateTime"].replace("Z", "+00:00")),
                source_file=granule["meta"]["native-id"],
            )
        )
    return footprints


def _discover_sentinel1_ocn_footprints_dry(cfg) -> "list[SarFootprint]":
    """Non-WV (IW/EW/SM) Sentinel-1 OCN footprints from a dry CDSE
    catalog search. WV-mode granules are excluded here -- see
    _discover_sentinel1_wv_footprints_dry."""
    records = _query_sentinel1_ocn_dry(cfg)
    footprints = []
    for record in records:
        if _WV_MODE_RE.match(record["Name"]):
            continue
        polygon, bbox = _polygon_and_bbox_from_geofootprint(
            record.get("GeoFootprint"), lambda: _fallback_bbox_from_cfg(cfg),
        )
        footprints.append(
            SarFootprint(
                kind="polygon",
                bbox=bbox,
                polygon=polygon,
                points=None,
                sensing_start=datetime.fromisoformat(record["ContentDate_Start"].replace("Z", "+00:00")),
                sensing_end=datetime.fromisoformat(record["ContentDate_End"].replace("Z", "+00:00")),
                source_file=record["Name"],
            )
        )
    return footprints


def _query_clms_ssm_dry(cfg) -> "list[dict]":
    """CDSE CLMS catalog search for Sentinel-1 SSM tiles, no download --
    reuses SoilMoistureDownloader.query's exact dataset_identifier (see
    sentinel1_soil_moisture_downloader.py). output_dir is a required
    constructor argument there but is never consulted by query() itself
    (only download() touches it), so a harmless placeholder is passed.
    query() returns a pd.DataFrame; it is converted to list[dict] here so
    every discovery function in this module shares the same
    record["field"] access pattern."""
    from ..downloaders.sentinel1_soil_moisture_downloader import SoilMoistureDownloader

    bounds = cfg.geographic_bounds
    dl = SoilMoistureDownloader(output_dir=Path("."), dry_run=True)
    df = dl.query(
        min_lon=bounds.min_lon, max_lon=bounds.max_lon,
        min_lat=bounds.min_lat, max_lat=bounds.max_lat,
        start=cfg.temporal_bounds.start, end=cfg.temporal_bounds.end,
    )
    return df.to_dict("records")


def _discover_clms_ssm_footprints_dry(cfg) -> "list[SarFootprint]":
    """CLMS SSM's own catalog footprint is the tile's whole nominal
    region (e.g. all of Europe), not real per-day coverage -- propagate
    Sentinel-1's own orbit across the tile's day (via
    orbit_coverage.orbit_overlap_windows) to predict which part of the
    tile was actually overpassed, producing kind="orbit_swath"
    footprints (polygon is always None -- only a predicted bbox is
    available here, never real vertices).

    Each candidate tile could have been covered by any of Sentinel-1A/B/C
    (_CLMS_SSM_SATELLITES), so all three are tried and their matched
    windows unioned -- deduplicated by (start, end) value, since more
    than one satellite reporting the exact same window (or a fail-open
    "whole day" window from more than one of them) must not produce one
    redundant footprint per satellite."""
    records = _query_clms_ssm_dry(cfg)
    bounds = cfg.geographic_bounds
    bbox = (bounds.min_lon, bounds.max_lon, bounds.min_lat, bounds.max_lat)

    footprints = []
    for record in records:
        start = datetime.fromisoformat(record["ContentDate_Start"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(record["ContentDate_End"].replace("Z", "+00:00"))

        matched_windows: "set[tuple[datetime, datetime]]" = set()
        for satellite in _CLMS_SSM_SATELLITES:
            windows = orbit_coverage.orbit_overlap_windows(
                satellite, start, end,
                bounds.min_lon, bounds.max_lon, bounds.min_lat, bounds.max_lat,
            )
            matched_windows.update(windows)

        for w_start, w_end in sorted(matched_windows):
            footprints.append(
                SarFootprint(
                    kind="orbit_swath",
                    bbox=bbox,
                    polygon=None,
                    points=None,
                    sensing_start=w_start,
                    sensing_end=w_end,
                    source_file=record["Name"],
                )
            )
    return footprints


def _clms_ssm_footprint_from_downloaded(tif_path: "Path") -> "Optional[SarFootprint]":
    """Real non-NaN pixel extent from an already-downloaded CLMS SSM
    GeoTIFF -- more accurate than orbit propagation since the real data
    is already on disk. Returns None when the GeoTIFF is missing
    (mirrors DataTreeConverter.from_sar_l3_ssm_geotiff's own "file not
    found" contract) or when every pixel is NaN (a tile with zero valid
    retrievals for the day has no real extent to report)."""
    ds = DataTreeConverter.from_sar_l3_ssm_geotiff(tif_path)
    if ds is None:
        return None

    valid = ds["sarSSM"].notnull()
    if not bool(valid.any()):
        return None

    valid_lon = ds["lon"].where(valid)
    valid_lat = ds["lat"].where(valid)
    bbox = (
        float(valid_lon.min()), float(valid_lon.max()),
        float(valid_lat.min()), float(valid_lat.max()),
    )
    sensing = pd.Timestamp(ds["time"].values).to_pydatetime()
    return SarFootprint(
        kind="orbit_swath",
        bbox=bbox,
        polygon=None,
        points=None,
        sensing_start=sensing,
        sensing_end=sensing,
        source_file=str(tif_path),
    )


#: One dry-search discovery entry per SAR_SOURCES key (core/sar_sources.py).
#: "sentinel1_l2_ocn" combines both the non-WV and WV-mode discovery
#: functions since a recipe's Sentinel-1 OCN data can include either or
#: both -- their results are concatenated into one list.
_DRY_DISCOVERY_BY_SOURCE = {
    "sentinel1_l2_ocn": lambda cfg: (
        _discover_sentinel1_ocn_footprints_dry(cfg) + _discover_sentinel1_wv_footprints_dry(cfg)
    ),
    "sentinel1_clms_ssm": _discover_clms_ssm_footprints_dry,
    "nisar_sme2": _discover_nisar_sme2_footprints_dry,
    "radarsat2": _discover_radarsat2_footprints_dry,
}


def discover_sar_footprints_dry(sar_data_spec, cfg) -> "list[SarFootprint]":
    """Dispatch to the right SAR source's dry-search discovery function,
    keyed by sar_data_spec.source (same key SAR_SOURCES in
    core/sar_sources.py uses)."""
    discover_fn = _DRY_DISCOVERY_BY_SOURCE.get(sar_data_spec.source)
    if discover_fn is None:
        raise ValueError(
            f"discover_sar_footprints_dry: unknown SAR source {sar_data_spec.source!r} "
            f"-- expected one of {sorted(_DRY_DISCOVERY_BY_SOURCE)}."
        )
    return discover_fn(cfg)


def sar_footprints_from_downloaded(sar_files, sar_source_spec, product_type: str) -> "list[SarFootprint]":
    """Footprints from already-downloaded, already-converted SAR files --
    used by a real run's inline (non-dry) prediction path. sar_source_spec
    is the matching entry from SAR_SOURCES (core/sar_sources.py).
    product_type (the recipe's cfg.variable, e.g. "wind"/"waves"/
    "currents"/"soil_moisture") is threaded straight through to
    sar_source_spec.convert(path, product_type), mirroring
    orchestrator.py's own _compute_sar_scene_times pattern -- required
    (no default) since only the caller knows the real value, and passing
    a placeholder would silently break sentinel1_l2_ocn's convert(),
    which branches on it.

    "sentinel1_clms_ssm" keeps its own dedicated path
    (_clms_ssm_footprint_from_downloaded), since CLMS SSM tiles need
    real non-NaN pixel extent, not just a converted Dataset's overall
    bbox. Every other SAR_SOURCES key (Sentinel-1 OCN, NISAR SME2,
    RADARSAT-2) shares one generic path below: each source's convert()
    already normalizes its native format into a (lon, lat, time)
    xarray.Dataset with either a WV-mode "point" dimension or a
    grid-mode (y, x) shape -- see DataTreeConverter.from_sar_l2_ocn_safe/
    from_nisar_sme2/from_radarsat2_wind -- so no further per-source
    branching is needed here."""
    if sar_source_spec.key == "sentinel1_clms_ssm":
        footprints = []
        for path in sar_files:
            fp = _clms_ssm_footprint_from_downloaded(path)
            if fp is not None:
                footprints.append(fp)
        return footprints

    footprints = []
    for path in sar_files:
        # Everything for one file -- both convert() itself AND the
        # geometry/time extraction that follows it -- lives in this one
        # try/except: a converted Dataset missing lon/lat raises KeyError,
        # and a non-WV source's "time" value can be a length-1 array
        # rather than a true scalar (see the pd.to_datetime(np.atleast_1d(
        # ...)) note below) -- either must skip just THIS file, not
        # propagate out of sar_footprints_from_downloaded and hit
        # _collocation_predictions()'s blanket except Exception, which
        # would silently disable gating for the ENTIRE run instead of
        # just dropping one bad file.
        try:
            ds = sar_source_spec.convert(path, product_type)
            if ds is None:
                continue
            lon_min, lon_max = float(ds["lon"].min()), float(ds["lon"].max())
            lat_min, lat_max = float(ds["lat"].min()), float(ds["lat"].max())
            is_wv = "point" in ds.dims and "y" not in ds.dims  # matches collocation.py's own is_wv_mode check
            if is_wv:
                lats = ds["lat"].values.tolist()
                lons = ds["lon"].values.tolist()
                points = list(zip(lats, lons))
                # A WV product's "time" coord holds one value per vignette
                # (dims=["point"]) -- a real product bundles ~16 of them at
                # DIFFERENT acquisition times (see
                # DataTreeConverter.from_sar_l2_ocn_wv_safe), so it's an
                # array, not a scalar; sensing_start/sensing_end span its
                # full range rather than assuming a single .item()-able value.
                times = pd.to_datetime(ds["time"].values)
                sensing_start_ts, sensing_end_ts = times.min(), times.max()
                kind: "Literal['polygon', 'wv_points']" = "wv_points"
            else:
                # pd.to_datetime(np.atleast_1d(...)), not pd.Timestamp(...)
                # or numpy's own .item(): a numpy datetime64 scalar's
                # .item() returns a raw int (nanoseconds since epoch), not
                # a datetime -- every predicate below does datetime
                # arithmetic (subtraction/comparison/.isoformat()) on
                # sensing_start/sensing_end, so an int here makes every
                # single prediction raise and fall back to "unknown",
                # which never satisfies _should_skip_for_collocation's
                # verdict == "none-predicted" check -- silently disabling
                # the entire skip-gating feature for every real (non-dry)
                # run. pd.Timestamp(...) additionally raises TypeError
                # outright if "time" happens to be a length-1 array rather
                # than a true 0-d scalar -- np.atleast_1d normalizes that
                # the same defensive way orchestrator.py's own
                # _compute_sar_scene_times already does for these exact
                # same converted files.
                idx = pd.to_datetime(np.atleast_1d(ds["time"].values))
                sensing_start_ts, sensing_end_ts = idx.min(), idx.max()
                kind = "polygon"

            # A resolved NaT (e.g. a NISAR SME2 granule with a missing/
            # unparseable zeroDopplerStartTime -- see
            # datatree_converter.py) must not become a footprint at all:
            # an unfiltered NaT can make _predict_global_composite's
            # day-range loop produce zero days, a false "none-predicted"
            # verdict that wrongly SKIPS a real download -- the one
            # fail-closed risk in this feature, and the only place in
            # this module that must NOT fail toward inclusion.
            if pd.isna(sensing_start_ts) or pd.isna(sensing_end_ts):
                logger.debug("sar_footprints_from_downloaded: NaT sensing time for %s, skipping", path)
                continue

            if kind == "wv_points":
                footprints.append(SarFootprint(
                    kind="wv_points", bbox=(lon_min, lon_max, lat_min, lat_max), polygon=None, points=points,
                    sensing_start=sensing_start_ts.to_pydatetime(), sensing_end=sensing_end_ts.to_pydatetime(),
                    source_file=str(path),
                ))
            else:
                footprints.append(SarFootprint(
                    kind="polygon", bbox=(lon_min, lon_max, lat_min, lat_max), polygon=None, points=None,
                    sensing_start=sensing_start_ts.to_pydatetime(), sensing_end=sensing_end_ts.to_pydatetime(),
                    source_file=str(path),
                ))
        except Exception:
            logger.debug("sar_footprints_from_downloaded: failed for %s", path, exc_info=True)
            continue
    return footprints


Verdict = Literal["collocated", "none-predicted", "unknown"]


@dataclass(frozen=True)
class SourcePrediction:
    """The predicted collocation outcome for one configured validation
    source against the recipe's discovered SAR footprints.

    matched_windows and matched_stations default to empty lists rather
    than None -- callers can always iterate them without a None check --
    but a mutable default can't be given directly on a dataclass field,
    so they're seeded in __post_init__ instead (frozen-safe via
    object.__setattr__, since normal attribute assignment is blocked on
    a frozen dataclass)."""

    source_type: str
    bucket: str
    verdict: Verdict
    detail: str
    message: Optional[str] = None
    matched_windows: List[Tuple[datetime, datetime]] = None  # type: ignore[assignment]
    matched_stations: List[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.matched_windows is None:
            object.__setattr__(self, "matched_windows", [])
        if self.matched_stations is None:
            object.__setattr__(self, "matched_stations", [])


@dataclass(frozen=True)
class CollocationReport:
    """The full dry-collocation result for one recipe: one
    SourcePrediction per configured validation source, plus the SAR-side
    footprint count they were all evaluated against."""

    recipe_path: str
    sar_footprint_count: int
    predictions: List[SourcePrediction]


#: Populated incrementally by each per-validation-source-type predicate:
#: maps a source_type string (matching cfg's own recipe source-type keys)
#: to a predicate function `(source, cfg, sar_footprints) -> SourcePrediction`.
_PREDICATES: "dict[str, Callable[..., SourcePrediction]]" = {}


def _predicate_accepts_stop_on_first_match(predicate: "Callable[..., SourcePrediction]") -> bool:
    """Whether *predicate* declares a stop_on_first_match keyword argument
    (directly, or via **kwargs) -- only the orbit-corridor/catalog-precise
    bucket predicates (and the thin per-source_type wrappers registered
    over them) do. This lets predict_source thread the real-run gating
    path's early-exit request (see DataOrchestrator._collocation_predictions)
    through to exactly those predicates, leaving every other predicate's
    3-argument call signature -- and any test that monkeypatches one of
    the shared bucket predicates with a fixed-arity fake -- untouched."""
    try:
        params = inspect.signature(predicate).parameters
    except (TypeError, ValueError):
        return False
    return "stop_on_first_match" in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def predict_source(
    source, cfg, sar_footprints: "list[SarFootprint]", *, stop_on_first_match: bool = False,
) -> SourcePrediction:
    """Dispatch to the right bucket's predicate for source.source_type.
    An unrecognized source_type is itself an "unknown" verdict -- never
    raises, since the report loop calls this once per configured
    validation source and must not crash on one bad/future entry.

    stop_on_first_match (default False, used by the --dry-collocation
    preview path): requests that a predicate stop probing as soon as one
    confirmed collocation is found, rather than exhaustively checking
    every SAR footprint -- see DataOrchestrator._collocation_predictions,
    the real-run gating path this exists for. Only forwarded to predicates
    that actually declare the parameter (see
    _predicate_accepts_stop_on_first_match); every other predicate keeps
    running its own default (exhaustive) behavior regardless of this
    flag, since a plain yes/no predicate already short-circuits its own
    footprint loop on the first hit (see e.g. _predict_insitu)."""
    predicate = _PREDICATES.get(source.source_type)
    if predicate is None:
        return SourcePrediction(
            source_type=source.source_type, bucket="unregistered", verdict="unknown",
            detail=f"No dry-collocation predicate registered for source_type={source.source_type!r}.",
        )
    if stop_on_first_match and _predicate_accepts_stop_on_first_match(predicate):
        return predicate(source, cfg, sar_footprints, stop_on_first_match=True)
    return predicate(source, cfg, sar_footprints)


def predict_collocation(
    cfg, sar_footprints: "list[SarFootprint]", recipe_path: str = "", *, stop_on_first_match: bool = False,
) -> CollocationReport:
    """Predict collocation for every configured validation source in
    cfg.validation_sources against sar_footprints. One source's predicate
    raising is caught and turned into an "unknown" verdict for that
    source alone -- never aborts the whole report.

    stop_on_first_match is forwarded to predict_source per source -- see
    its docstring."""
    predictions = []
    for source in cfg.validation_sources:
        try:
            predictions.append(
                predict_source(source, cfg, sar_footprints, stop_on_first_match=stop_on_first_match)
            )
        except Exception:
            logger.debug("predict_collocation: predict_source failed for %s", source.source_type, exc_info=True)
            predictions.append(
                SourcePrediction(
                    source_type=source.source_type, bucket="unregistered", verdict="unknown",
                    detail=f"Prediction raised an exception for {source.source_type}.",
                )
            )
    return CollocationReport(
        recipe_path=recipe_path, sar_footprint_count=len(sar_footprints), predictions=predictions,
    )


def render_console_table(report: CollocationReport) -> str:
    """A simple fixed-width table -- source_type | bucket | verdict |
    detail -- one row per prediction, no external dependency (no
    tabulate/rich) since every other CLI output in this codebase is
    plain print()."""
    header = f"{'source':<24}{'bucket':<18}{'verdict':<16}detail"
    lines = [header, "-" * len(header)]
    for p in report.predictions:
        lines.append(f"{p.source_type:<24}{p.bucket:<18}{p.verdict:<16}{p.detail}")
        if p.message:
            lines.append(f"{'':<24}{'':<18}{'':<16}  -> {p.message}")
    return "\n".join(lines)


def report_to_json(report: CollocationReport) -> str:
    """JSON serialization of report -- datetimes become ISO-8601 strings."""
    def _default(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Not JSON serializable: {obj!r}")

    payload = {
        "recipe_path": report.recipe_path,
        "sar_footprint_count": report.sar_footprint_count,
        "predictions": [
            {
                "source_type": p.source_type, "bucket": p.bucket, "verdict": p.verdict,
                "detail": p.detail, "message": p.message,
                "matched_windows": p.matched_windows, "matched_stations": p.matched_stations,
            }
            for p in report.predictions
        ],
    }
    return json.dumps(payload, default=_default, indent=2)


def _point_in_footprint(
    lat: float, lon: float, footprint: "SarFootprint", wv_search_radius_km: float = 14.0,
) -> bool:
    """Whether (lat, lon) falls inside footprint's real shape -- not just
    its bbox. kind="polygon": real point-in-polygon via
    orbit_coverage._point_in_polygon when a polygon is available (the
    same helper the orbit-corridor bucket's fine refinement uses -- one
    implementation, not two), falling back to the bbox check when
    polygon is None (RADARSAT-2). kind="wv_points": within
    wv_search_radius_km of any vignette point -- a coarser,
    distance-based check since a vignette is a small area, not a single
    point boundary; callers with a real recipe should pass
    cfg.collocation.sar_footprint_radius_km rather than relying on this
    default. kind="orbit_swath": bbox only, since that kind's own
    geometry is already an approximation with nothing more precise to
    test against."""
    if footprint.kind == "polygon":
        if footprint.polygon is not None:
            return orbit_coverage._point_in_polygon(lat, lon, footprint.polygon)
        return _point_in_bbox(lat, lon, footprint.bbox[0], footprint.bbox[1], footprint.bbox[2], footprint.bbox[3])
    if footprint.kind == "wv_points":
        if not footprint.points:
            return False
        for vlat, vlon in footprint.points:
            if _haversine_distance(lon, lat, vlon, vlat) <= wv_search_radius_km:
                return True
        return False
    # kind == "orbit_swath"
    return _point_in_bbox(lat, lon, footprint.bbox[0], footprint.bbox[1], footprint.bbox[2], footprint.bbox[3])


def _bbox_overlaps_footprint(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float, footprint: "SarFootprint",
) -> bool:
    """Whether the given bbox (e.g. an HF-radar region's own coverage
    extent) overlaps footprint -- a coarser area-vs-area check than
    _point_in_footprint's point-vs-shape test, for ground sources whose
    real unit is itself an area (a radar grid region), not a point
    (a station). Always a plain bbox-vs-bbox intersection test against
    footprint.bbox, regardless of kind -- including kind="polygon",
    where footprint.bbox is a superset of the real polygon, so this can
    only over-predict overlap, never miss a real one (the same
    fail-toward-inclusion property every bbox check in this module has).
    It does not add precision beyond the coarse bbox pass for
    kind="polygon" footprints specifically -- callers that need that
    precision for a single point should use _point_in_footprint instead;
    this function exists for area-shaped ground sources, where a
    per-point check doesn't apply."""
    f_min_lon, f_max_lon, f_min_lat, f_max_lat = footprint.bbox
    return not (max_lon < f_min_lon or min_lon > f_max_lon or max_lat < f_min_lat or min_lat > f_max_lat)


def _to_naive_utc(d: datetime) -> datetime:
    """*d* as a naive UTC datetime -- every timestamp this module compares
    is UTC, whether or not it happens to carry an explicit tzinfo (a
    SarFootprint's own sensing_start/sensing_end are not guaranteed
    timezone-aware, and neither are the various per-source timestamps
    compared against them, e.g. ISMN's .stm-parsed station date ranges).
    An aware value is converted to UTC and then has its tzinfo stripped;
    an already-naive value passes through unchanged. Used by
    _windows_overlap so two real timestamps of differing tz-awareness can
    still be compared without raising."""
    return d.astimezone(timezone.utc).replace(tzinfo=None) if d.tzinfo is not None else d


def _windows_overlap(
    a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime,
) -> bool:
    """Whether time windows [a_start, a_end] and [b_start, b_end] overlap
    at all (inclusive of touching endpoints). Each bound is normalized via
    _to_naive_utc first, since callers routinely mix a tz-aware SarFootprint
    bound against a naive per-source one (or vice versa)."""
    a_start, a_end, b_start, b_end = (
        _to_naive_utc(a_start), _to_naive_utc(a_end), _to_naive_utc(b_start), _to_naive_utc(b_end),
    )
    return a_start <= b_end and b_start <= a_end


def _predict_orbit_corridor_source(
    source,
    cfg,
    sar_footprints: "list[SarFootprint]",
    *,
    satellite_resolver: "Callable[[str], str]",
    list_candidates_dry: "Callable[..., list[tuple[str, datetime, datetime]]]",
    source_type: str,
    stop_on_first_match: bool = False,
) -> SourcePrediction:
    """Shared predicate for every orbit-corridor-bucket source (ASCAT SSM
    via H-SAF/EUMDAC, HY-2B/HY-2C/Oceansat-3, AMSR2, SMOS): a coarse
    dry-listing per footprint's padded time window, then
    orbit_overlap_windows as the fine refinement. Each candidate's own
    satellite is resolved independently via satellite_resolver rather
    than assuming one fixed satellite for the whole call, since a single
    listing can span multiple satellites. Refinement windows are
    intersected against the footprint's own sensing window padded by
    cfg's resolved collocation time tolerance for source_type.

    A wv_points footprint is checked per-vignette (one zero-width-bbox
    orbit_overlap_windows call per point), never against one enclosing
    box, since a vignette is a small area far smaller than the
    footprint's overall bbox.

    stop_on_first_match: the --dry-collocation preview path (the default,
    False) keeps scanning every footprint/candidate/point so matched_windows
    accumulates a real count for the report's own "N matched window(s)"
    detail text. The real-run gating path (DataOrchestrator._collocation_
    predictions, via predict_collocation(..., stop_on_first_match=True))
    only ever needs a yes/no verdict, so it opts into stopping at the
    first confirmed match instead -- this is what bounds this predicate's
    live-probe cost (list_candidates_dry / orbit_overlap_windows calls) on
    the default path of every real recipe run.
    """
    if not sar_footprints:
        return SourcePrediction(
            source_type=source_type, bucket="orbit-corridor", verdict="unknown",
            detail="No SAR footprints supplied -- cannot predict.",
        )

    tolerance = timedelta(minutes=_resolve_temporal_padding_minutes(cfg, source_type))
    matched_windows: "list[tuple[datetime, datetime]]" = []

    for footprint in sar_footprints:
        if stop_on_first_match and matched_windows:
            break
        padded_start = footprint.sensing_start - tolerance
        padded_end = footprint.sensing_end + tolerance
        try:
            candidates = list_candidates_dry(
                footprint.bbox[0], footprint.bbox[1], footprint.bbox[2], footprint.bbox[3],
                padded_start.isoformat(), padded_end.isoformat(),
            )
        except Exception:
            logger.debug("_predict_orbit_corridor_source: list_candidates_dry failed", exc_info=True)
            return SourcePrediction(
                source_type=source_type, bucket="orbit-corridor", verdict="unknown",
                detail=f"Candidate listing failed for {source_type}.",
            )

        points: "list[Optional[tuple[float, float]]]" = (
            list(footprint.points or []) if footprint.kind == "wv_points" else [None]
        )
        for candidate_name, cand_start, cand_end in candidates:
            if stop_on_first_match and matched_windows:
                break
            try:
                satellite = satellite_resolver(candidate_name)
            except Exception:
                logger.debug("_predict_orbit_corridor_source: satellite_resolver failed", exc_info=True)
                matched_windows.append((cand_start, cand_end))  # fail open
                continue
            for point in points:
                if stop_on_first_match and matched_windows:
                    break
                if point is None:
                    target_bbox = footprint.bbox
                    target_polygon = footprint.polygon
                else:
                    lat, lon = point
                    target_bbox = (lon, lon, lat, lat)
                    target_polygon = None
                try:
                    windows = orbit_coverage.orbit_overlap_windows(
                        satellite, cand_start, cand_end,
                        target_bbox[0], target_bbox[1], target_bbox[2], target_bbox[3],
                        polygon=target_polygon,
                    )
                except Exception:
                    logger.debug("_predict_orbit_corridor_source: orbit_overlap_windows failed", exc_info=True)
                    windows = [(cand_start, cand_end)]  # fail open
                for w_start, w_end in windows:
                    if _windows_overlap(w_start, w_end, padded_start, padded_end):
                        matched_windows.append((w_start, w_end))
                        if stop_on_first_match:
                            break

    if matched_windows:
        return SourcePrediction(
            source_type=source_type, bucket="orbit-corridor", verdict="collocated",
            detail=f"{len(matched_windows)} matched window(s) across {len(sar_footprints)} SAR footprint(s).",
            matched_windows=matched_windows,
        )
    return SourcePrediction(
        source_type=source_type, bucket="orbit-corridor", verdict="none-predicted",
        detail=f"No predicted overlap across {len(sar_footprints)} SAR footprint(s).",
    )


def _predict_catalog_precise_source(
    source,
    cfg,
    sar_footprints: "list[SarFootprint]",
    *,
    list_candidates_dry: "Callable[..., list[tuple[str, datetime, datetime]]]",
    source_type: str,
    tolerance_source_types: "Optional[tuple[str, ...]]" = None,
    stop_on_first_match: bool = False,
) -> SourcePrediction:
    """Shared predicate for every catalog-precise-bucket source (ASCAT
    winds, SMAP/AMSR2 via NASA Earthdata/CMR, altimeter via Copernicus
    Marine): the coarse listing IS the geometrically-precise answer,
    since these sources already run a real bbox-filtered server-side
    catalog search -- unlike the orbit-corridor bucket, no fine
    refinement against a predicted orbit swath is needed or performed.

    A wv_points footprint is queried per-vignette (one bbox-filtered
    listing call per point, matching that vignette's own tiny extent),
    never against one enclosing box, mirroring
    _predict_orbit_corridor_source's identical per-vignette handling.

    tolerance_source_types overrides which key(s) resolve the time
    tolerance via _resolve_temporal_padding_minutes, for a caller whose
    source_type doesn't itself carry a DEFAULT_LAYER_TYPE_SPECS entry
    (e.g. altimeter, keyed there as "altimeter_1hz"/"altimeter_5hz").
    Defaults to (source_type,), matching every other caller.

    stop_on_first_match: see _predict_orbit_corridor_source's identical
    parameter -- False (the default) preserves the --dry-collocation
    preview path's exhaustive matched_windows count; the real-run gating
    path opts into True to bound this predicate's live listing-call cost.
    """
    if not sar_footprints:
        return SourcePrediction(
            source_type=source_type, bucket="catalog-precise", verdict="unknown",
            detail="No SAR footprints supplied -- cannot predict.",
        )

    tolerance = timedelta(
        minutes=_resolve_temporal_padding_minutes(cfg, *(tolerance_source_types or (source_type,)))
    )
    matched_windows: "list[tuple[datetime, datetime]]" = []

    for footprint in sar_footprints:
        if stop_on_first_match and matched_windows:
            break
        padded_start = footprint.sensing_start - tolerance
        padded_end = footprint.sensing_end + tolerance
        points: "list[Optional[tuple[float, float]]]" = (
            list(footprint.points or []) if footprint.kind == "wv_points" else [None]
        )
        for point in points:
            if stop_on_first_match and matched_windows:
                break
            bbox = footprint.bbox if point is None else (point[1], point[1], point[0], point[0])
            try:
                candidates = list_candidates_dry(
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    padded_start.isoformat(), padded_end.isoformat(),
                )
            except Exception:
                logger.debug("_predict_catalog_precise_source: listing failed", exc_info=True)
                return SourcePrediction(
                    source_type=source_type, bucket="catalog-precise", verdict="unknown",
                    detail=f"Candidate listing failed for {source_type}.",
                )
            for _name, cand_start, cand_end in candidates:
                matched_windows.append((cand_start, cand_end))
                if stop_on_first_match:
                    break

    if matched_windows:
        return SourcePrediction(
            source_type=source_type, bucket="catalog-precise", verdict="collocated",
            detail=f"{len(matched_windows)} candidate(s) across {len(sar_footprints)} SAR footprint(s).",
            matched_windows=matched_windows,
        )
    return SourcePrediction(
        source_type=source_type, bucket="catalog-precise", verdict="none-predicted",
        detail=f"No candidates found across {len(sar_footprints)} SAR footprint(s).",
    )


def _hsaf_list_candidates_dry(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float, start: str, end: str,
) -> "list[tuple[str, datetime, datetime]]":
    """Thin wrapper: constructs an HSAFDownloader (default product h122,
    matching orchestrator.py's own default) and calls its
    list_candidates_dry. output_dir is never written to by
    list_candidates_dry, so a harmless placeholder is safe here."""
    dl = HSAFDownloader(output_dir=Path("/dev/null"), product="h122")
    return dl.list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end)


#: Mirrors orchestrator.py's own `_ASCAT_COVERAGE_CUTOFF` exactly -- the
#: EUMDAC SOMO12 collection stopped receiving new products on this date.
#: Kept as a separate module-level constant (not imported from
#: orchestrator.py) to avoid coupling the two cutoff values to a single
#: import site -- both must be updated together if EUMETSAT's cutoff
#: ever moves.
_ASCAT_COVERAGE_CUTOFF = "2025-07-15"

#: EUMETSAT's SOMO12 product-ID satellite codes -> orbit_coverage.py's
#: SATELLITE_ORBIT_SPECS keys. EUMETSAT's numbering is launch-order-based,
#: not letter-order-based: M01=MetOp-B, M02=MetOp-A, M03=MetOp-C. MetOp-A
#: has no SATELLITE_ORBIT_SPECS entry (decommissioned) -- mapped here
#: anyway so a real "m02" product still reaches orbit_overlap_windows,
#: which fails open (falls back to the candidate's own coarse window) on
#: an unrecognized satellite key rather than raising.
_EUMDAC_ASCAT_SATELLITE_MAP = {"m01": "metop-b", "m02": "metop-a", "m03": "metop-c"}


def _hsaf_satellite_resolver(filename: str) -> str:
    """satellite_resolver adapter for H-SAF: hsaf_downloader._parse_satellite
    returns Optional[str] (None for an unparseable filename), while
    _predict_orbit_corridor_source's satellite_resolver contract wants a
    str -- an unrecognized/unresolvable key ("unknown") still reaches
    orbit_overlap_windows, which itself fails open (whole candidate
    window kept) on any key absent from SATELLITE_ORBIT_SPECS, matching
    _filter_by_orbit_overlap's own fail-open handling of the same case."""
    return _hsaf_parse_satellite(filename) or "unknown"


def _parse_eumdac_ascat_satellite(product_id: str) -> str:
    """Best-effort satellite key for a SOMO12 product ID; falls back to
    'metop-b' (never raises) for an unrecognized ID."""
    lowered = product_id.lower()
    for code, sat in _EUMDAC_ASCAT_SATELLITE_MAP.items():
        if code in lowered:
            return sat
    return "metop-b"


def _eumdac_ascat_ssm_list_candidates_dry(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float, start: str, end: str,
) -> "list[tuple[str, datetime, datetime]]":
    """Thin wrapper: constructs an ASCATSoilMoistureDownloader and calls
    its list_candidates_dry. output_dir is never written to by
    list_candidates_dry, so a harmless placeholder is safe here."""
    dl = ASCATSoilMoistureDownloader(output_dir=Path("/dev/null"))
    return dl.list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end)


def _predict_ascat_ssm(
    source, cfg, sar_footprints: "list[SarFootprint]", *, stop_on_first_match: bool = False,
) -> SourcePrediction:
    """Combined predicate for source_type="ascat_ssm", mirroring
    orchestrator.py's _download_ascat_ssm branching: EUMDAC's SOMO12
    archive covers footprints on/before _ASCAT_COVERAGE_CUTOFF, H-SAF's
    rolling on-line archive covers roughly the last 60 days. A
    footprint's own sensing time (padded by the resolved collocation
    time tolerance) determines which branch(es) actually apply, matching
    real download behavior rather than always checking both. Collocated
    if either applicable branch predicts collocated; a footprint that
    falls in the genuine gap between the two archives (neither branch
    applies) is "unknown" rather than a confident "none-predicted",
    since that gap is a coverage limitation, not evidence of absence.

    stop_on_first_match: forwarded to each branch's own
    _predict_orbit_corridor_source call (see its docstring) -- and, when
    True, also skips the H-SAF branch entirely once the EUMDAC branch
    alone already confirmed "collocated", since a real-run gating caller
    only needs one confirmed hit across either archive.
    """
    tolerance = timedelta(minutes=_resolve_temporal_padding_minutes(cfg, "ascat_ssm"))
    hsaf_window_start = (datetime.now(timezone.utc).date() - timedelta(days=60)).isoformat()

    eumdac_footprints = [
        fp for fp in sar_footprints
        if (fp.sensing_start - tolerance).date().isoformat() <= _ASCAT_COVERAGE_CUTOFF
    ]
    hsaf_footprints = [
        fp for fp in sar_footprints
        if (fp.sensing_end + tolerance).date().isoformat() >= hsaf_window_start
    ]

    def _orbit_corridor(
        footprints: "list[SarFootprint]",
        satellite_resolver: "Callable[[str], str]",
        list_candidates_dry: "Callable[..., list[tuple[str, datetime, datetime]]]",
    ) -> SourcePrediction:
        # Explicit branching rather than a **kwargs spread: mypy can't
        # verify a spread dict's value type against every one of
        # _predict_orbit_corridor_source's differently-typed keyword
        # parameters, and passing stop_on_first_match=False unconditionally
        # would break the several existing tests that monkeypatch
        # _predict_orbit_corridor_source with a fixed-arity fake lacking
        # that parameter.
        if stop_on_first_match:
            return _predict_orbit_corridor_source(
                source, cfg, footprints,
                satellite_resolver=satellite_resolver, list_candidates_dry=list_candidates_dry,
                source_type="ascat_ssm", stop_on_first_match=True,
            )
        return _predict_orbit_corridor_source(
            source, cfg, footprints,
            satellite_resolver=satellite_resolver, list_candidates_dry=list_candidates_dry,
            source_type="ascat_ssm",
        )

    predictions: "list[SourcePrediction]" = []
    if eumdac_footprints:
        predictions.append(_orbit_corridor(
            eumdac_footprints, _parse_eumdac_ascat_satellite, _eumdac_ascat_ssm_list_candidates_dry,
        ))
    already_confirmed = stop_on_first_match and any(p.verdict == "collocated" for p in predictions)
    if hsaf_footprints and not already_confirmed:
        predictions.append(_orbit_corridor(
            hsaf_footprints, _hsaf_satellite_resolver, _hsaf_list_candidates_dry,
        ))

    if not predictions:
        return SourcePrediction(
            source_type="ascat_ssm", bucket="orbit-corridor", verdict="unknown",
            detail=(
                f"No SAR footprint falls within EUMDAC's historical archive "
                f"(through {_ASCAT_COVERAGE_CUTOFF}) or H-SAF's rolling "
                f"on-line archive (since {hsaf_window_start})."
            ),
        )
    if any(p.verdict == "collocated" for p in predictions):
        matched = [w for p in predictions for w in (p.matched_windows or [])]
        return SourcePrediction(
            source_type="ascat_ssm", bucket="orbit-corridor", verdict="collocated",
            detail=" / ".join(p.detail for p in predictions),
            matched_windows=matched,
        )
    if any(p.verdict == "unknown" for p in predictions):
        return SourcePrediction(
            source_type="ascat_ssm", bucket="orbit-corridor", verdict="unknown",
            detail=" / ".join(p.detail for p in predictions),
        )
    return SourcePrediction(
        source_type="ascat_ssm", bucket="orbit-corridor", verdict="none-predicted",
        detail=" / ".join(p.detail for p in predictions),
    )


_PREDICATES["ascat_ssm"] = _predict_ascat_ssm


def _scatterometer_ftp_list_candidates_dry(
    satellite: str,
    min_lon: float, max_lon: float, min_lat: float, max_lat: float, start: str, end: str,
) -> "list[tuple[str, datetime, datetime]]":
    """Thin wrapper: constructs a ScatterometerFTPDownloader for *satellite*
    and calls its list_candidates_dry. output_dir is never written to by
    list_candidates_dry, so a harmless placeholder is safe here."""
    dl = ScatterometerFTPDownloader(satellite=satellite, output_dir=Path("/dev/null"))
    return dl.list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end)


def _hy2b_list_candidates_dry(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float, start: str, end: str,
) -> "list[tuple[str, datetime, datetime]]":
    return _scatterometer_ftp_list_candidates_dry("hy2b", min_lon, max_lon, min_lat, max_lat, start, end)


def _hy2c_list_candidates_dry(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float, start: str, end: str,
) -> "list[tuple[str, datetime, datetime]]":
    return _scatterometer_ftp_list_candidates_dry("hy2c", min_lon, max_lon, min_lat, max_lat, start, end)


def _oceansat3_list_candidates_dry(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float, start: str, end: str,
) -> "list[tuple[str, datetime, datetime]]":
    return _scatterometer_ftp_list_candidates_dry("oceansat3", min_lon, max_lon, min_lat, max_lat, start, end)


def _predict_scatterometer_hy2b(
    source, cfg, sar_footprints: "list[SarFootprint]", *, stop_on_first_match: bool = False,
) -> SourcePrediction:
    """Orbit-corridor predicate for source_type="scatterometer_hy2b" --
    HY-2B's OSI-SAF wind FTP listing (see scatterometer_ftp_downloader.py)
    only ever contains HY-2B's own data, so satellite_resolver is a fixed
    constant rather than parsed per candidate. stop_on_first_match is
    forwarded straight through -- see _predict_orbit_corridor_source's
    docstring."""
    return _predict_orbit_corridor_source(
        source, cfg, sar_footprints,
        satellite_resolver=lambda name: "hy2b",
        list_candidates_dry=_hy2b_list_candidates_dry,
        source_type="scatterometer_hy2b",
        stop_on_first_match=stop_on_first_match,
    )


def _predict_scatterometer_hy2c(
    source, cfg, sar_footprints: "list[SarFootprint]", *, stop_on_first_match: bool = False,
) -> SourcePrediction:
    """Orbit-corridor predicate for source_type="scatterometer_hy2c" --
    see _predict_scatterometer_hy2b."""
    return _predict_orbit_corridor_source(
        source, cfg, sar_footprints,
        satellite_resolver=lambda name: "hy2c",
        list_candidates_dry=_hy2c_list_candidates_dry,
        source_type="scatterometer_hy2c",
        stop_on_first_match=stop_on_first_match,
    )


def _predict_scatterometer_oceansat3(
    source, cfg, sar_footprints: "list[SarFootprint]", *, stop_on_first_match: bool = False,
) -> SourcePrediction:
    """Orbit-corridor predicate for source_type="scatterometer_oceansat3"
    -- see _predict_scatterometer_hy2b."""
    return _predict_orbit_corridor_source(
        source, cfg, sar_footprints,
        satellite_resolver=lambda name: "oceansat3",
        list_candidates_dry=_oceansat3_list_candidates_dry,
        source_type="scatterometer_oceansat3",
        stop_on_first_match=stop_on_first_match,
    )


_PREDICATES["scatterometer_hy2b"] = _predict_scatterometer_hy2b
_PREDICATES["scatterometer_hy2c"] = _predict_scatterometer_hy2c
_PREDICATES["scatterometer_oceansat3"] = _predict_scatterometer_oceansat3


def _smos_list_candidates_dry(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float, start: str, end: str,
) -> "list[tuple[str, datetime, datetime]]":
    """Thin wrapper: constructs an SMOSDownloader and calls its
    list_candidates_dry. output_dir is never written to by
    list_candidates_dry (except for a defensive debug-HTML dump when
    OADS's own response is unparseable, wrapped in its own try/except),
    so a harmless placeholder is safe here."""
    dl = SMOSDownloader(output_dir=Path("/dev/null"))
    return dl.list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end)


def _predict_smos_ssm(
    source, cfg, sar_footprints: "list[SarFootprint]", *, stop_on_first_match: bool = False,
) -> SourcePrediction:
    """Orbit-corridor predicate for source_type="smos_ssm" -- SMOS's OADS
    listing (see smos_downloader.py) only ever contains SMOS's own data,
    so satellite_resolver is a fixed constant rather than parsed per
    candidate."""
    return _predict_orbit_corridor_source(
        source, cfg, sar_footprints,
        satellite_resolver=lambda name: "smos",
        list_candidates_dry=_smos_list_candidates_dry,
        source_type="smos_ssm",
        stop_on_first_match=stop_on_first_match,
    )


_PREDICATES["smos_ssm"] = _predict_smos_ssm


def _scatterometer_list_candidates_dry(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float, start: str, end: str,
) -> "list[tuple[str, datetime, datetime]]":
    """Thin wrapper: constructs a ScatterometerDownloader (EUMDAC OSI-104
    ASCAT winds) and calls its list_candidates_dry. output_dir is never
    written to by list_candidates_dry, so a harmless placeholder is safe
    here."""
    dl = ScatterometerDownloader(output_dir=Path("/dev/null"))
    return dl.list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end)


def _predict_scatterometer(
    source, cfg, sar_footprints: "list[SarFootprint]", *, stop_on_first_match: bool = False,
) -> SourcePrediction:
    """Catalog-precise predicate for source_type="scatterometer" (ASCAT
    winds via EUMDAC) -- EUMDAC's own collection.search(bbox=...) is
    already a real geometrically-precise server-side query."""
    return _predict_catalog_precise_source(
        source, cfg, sar_footprints,
        list_candidates_dry=_scatterometer_list_candidates_dry,
        source_type="scatterometer",
        stop_on_first_match=stop_on_first_match,
    )


_PREDICATES["scatterometer"] = _predict_scatterometer


def _smap_ssm_list_candidates_dry(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float, start: str, end: str,
) -> "list[tuple[str, datetime, datetime]]":
    """Thin wrapper: constructs an EarthdataSoilMoistureDownloader for
    SMAP's SPL2SMP_E dataset (matching orchestrator.py's
    _download_smap_ssm) and calls its list_candidates_dry."""
    dl = EarthdataSoilMoistureDownloader(dataset="SPL2SMP_E", version="006", output_dir=Path("/dev/null"))
    return dl.list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end)


def _predict_smap_ssm(
    source, cfg, sar_footprints: "list[SarFootprint]", *, stop_on_first_match: bool = False,
) -> SourcePrediction:
    """Catalog-precise predicate for source_type="smap_ssm" -- NASA
    Earthdata/CMR's earthaccess.search_data(bounding_box=...) is already
    a real geometrically-precise server-side query."""
    return _predict_catalog_precise_source(
        source, cfg, sar_footprints,
        list_candidates_dry=_smap_ssm_list_candidates_dry,
        source_type="smap_ssm",
        stop_on_first_match=stop_on_first_match,
    )


_PREDICATES["smap_ssm"] = _predict_smap_ssm


def _altimeter_list_candidates_dry(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float, start: str, end: str,
) -> "list[tuple[str, datetime, datetime]]":
    """Thin wrapper: constructs an AltimeterDownloader and calls its
    list_candidates_dry (every mission/frequency it covers, default
    selection -- matching orchestrator.py's _download_altimeter default
    of frequencies=["1hz"] would require plumbing the recipe's own
    download_kwargs through here, which dry-collocation prediction has
    no access to; querying every mission/frequency is the conservative
    (fail-toward-inclusion) choice instead)."""
    dl = AltimeterDownloader(output_dir=Path("/dev/null"))
    return dl.list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end)


def _predict_altimeter(
    source, cfg, sar_footprints: "list[SarFootprint]", *, stop_on_first_match: bool = False,
) -> SourcePrediction:
    """Catalog-precise predicate for source_type="altimeter" -- Copernicus
    Marine's along-track datasets are opened bbox/time-filtered via
    copernicusmarine.open_dataset(), already a real geometrically-precise
    server-side query.

    tolerance_source_types passes ("altimeter_1hz", "altimeter_5hz")
    rather than the bare "altimeter" source_type, mirroring
    orchestrator.py's _download_altimeter -- DEFAULT_LAYER_TYPE_SPECS has
    no "altimeter" entry, only per-frequency ones (both 180 minutes), so
    using the bare key would silently fall back to the generic 30-minute
    default."""
    return _predict_catalog_precise_source(
        source, cfg, sar_footprints,
        list_candidates_dry=_altimeter_list_candidates_dry,
        source_type="altimeter",
        tolerance_source_types=("altimeter_1hz", "altimeter_5hz"),
        stop_on_first_match=stop_on_first_match,
    )


_PREDICATES["altimeter"] = _predict_altimeter


#: Mirrors orchestrator.py's own `_NSIDC_0451_CUTOFF` exactly -- NASA
#: Earthdata's NSIDC-0451 AMSR2 soil-moisture dataset stopped being
#: updated after this date, with AU_Land taking over afterward. Kept as
#: a separate module-level constant (not imported from orchestrator.py)
#: to avoid coupling the two cutoff values to a single import site --
#: both must be updated together if NSIDC's cutoff ever moves.
_NSIDC_0451_CUTOFF = "2023-12-31"


def _earthdata_amsr_ssm_list_candidates_dry(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float, start: str, end: str,
) -> "list[tuple[str, datetime, datetime]]":
    """Thin wrapper: constructs an EarthdataSoilMoistureDownloader for
    whichever AMSR2 CMR dataset covers *end*'s date (NSIDC-0451 or its
    AU_Land replacement), mirroring orchestrator.py's own
    _download_amsr_ssm dataset-cutoff selection, and calls its
    list_candidates_dry."""
    # Compare only the date portion -- end is a full ISO datetime (this
    # wrapper's own caller always passes a padded_end.isoformat() with a
    # time-of-day), while _NSIDC_0451_CUTOFF is a bare date. A naive
    # full-string comparison would misclassify any timestamp ON the
    # cutoff day itself (e.g. "2023-12-31T06:00:00" > "2023-12-31"
    # lexicographically, even though both are the same calendar day).
    dataset = "NSIDC-0451" if end[:10] <= _NSIDC_0451_CUTOFF else "AU_Land"
    dl = EarthdataSoilMoistureDownloader(dataset=dataset, output_dir=Path("/dev/null"))
    return dl.list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end)


def _gportal_amsr_ssm_list_candidates_dry(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float, start: str, end: str,
) -> "list[tuple[str, datetime, datetime]]":
    """Thin wrapper: constructs a GPortalAMSR2Downloader and calls its
    list_candidates_dry. Imported lazily (not at module top) since
    paramiko -- G-Portal's SFTP dependency -- is optional and this
    module must import cleanly without it."""
    from ..downloaders.gportal_downloader import GPortalAMSR2Downloader

    dl = GPortalAMSR2Downloader(output_dir=Path("/dev/null"))
    return dl.list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end)


def _predict_amsr_ssm(
    source, cfg, sar_footprints: "list[SarFootprint]", *, stop_on_first_match: bool = False,
) -> SourcePrediction:
    """Combined predicate for source_type="amsr_ssm", mirroring
    orchestrator.py's _download_amsr_ssm: NASA Earthdata/CMR is checked
    first, JAXA G-Portal (SFTP) is a real second source that often
    succeeds when Earthdata doesn't. Unlike ascat_ssm, both branches
    apply to every footprint -- there is no date-based eligibility
    split, since Earthdata's coverage cutoff only changes which CMR
    dataset is queried (see _earthdata_amsr_ssm_list_candidates_dry), it
    never makes the whole branch inapplicable. Collocated if either
    branch predicts collocated. bucket is "catalog-precise" (matching
    this source's own registration) even though the G-Portal branch
    internally reuses the orbit-corridor bucket's shared predicate --
    the label describes the source, not which internal path happened to
    answer.

    stop_on_first_match is forwarded to each branch's own shared-predicate
    call (see _predict_orbit_corridor_source's docstring) and, when True,
    also skips the G-Portal branch entirely once Earthdata alone already
    confirmed "collocated" -- a real-run gating caller only needs one
    confirmed hit across either source.
    """
    # Explicit branching rather than a **kwargs spread: mypy can't verify
    # a spread dict's value type against every one of these shared
    # predicates' differently-typed keyword parameters (e.g.
    # tolerance_source_types: Optional[Tuple[str, ...]]), and passing
    # stop_on_first_match=False unconditionally would break the several
    # existing tests that monkeypatch _predict_catalog_precise_source /
    # _predict_orbit_corridor_source with a fixed-arity fake lacking that
    # parameter.
    if stop_on_first_match:
        earthdata_prediction = _predict_catalog_precise_source(
            source, cfg, sar_footprints,
            list_candidates_dry=_earthdata_amsr_ssm_list_candidates_dry,
            source_type="amsr_ssm", stop_on_first_match=True,
        )
    else:
        earthdata_prediction = _predict_catalog_precise_source(
            source, cfg, sar_footprints,
            list_candidates_dry=_earthdata_amsr_ssm_list_candidates_dry,
            source_type="amsr_ssm",
        )
    if stop_on_first_match and earthdata_prediction.verdict == "collocated":
        return earthdata_prediction
    if stop_on_first_match:
        gportal_prediction = _predict_orbit_corridor_source(
            source, cfg, sar_footprints,
            satellite_resolver=lambda name: "gcom-w1",
            list_candidates_dry=_gportal_amsr_ssm_list_candidates_dry,
            source_type="amsr_ssm", stop_on_first_match=True,
        )
    else:
        gportal_prediction = _predict_orbit_corridor_source(
            source, cfg, sar_footprints,
            satellite_resolver=lambda name: "gcom-w1",
            list_candidates_dry=_gportal_amsr_ssm_list_candidates_dry,
            source_type="amsr_ssm",
        )

    predictions = [earthdata_prediction, gportal_prediction]
    if any(p.verdict == "collocated" for p in predictions):
        matched = [w for p in predictions for w in (p.matched_windows or [])]
        return SourcePrediction(
            source_type="amsr_ssm", bucket="catalog-precise", verdict="collocated",
            detail=" / ".join(p.detail for p in predictions),
            matched_windows=matched,
        )
    if any(p.verdict == "unknown" for p in predictions):
        return SourcePrediction(
            source_type="amsr_ssm", bucket="catalog-precise", verdict="unknown",
            detail=" / ".join(p.detail for p in predictions),
        )
    return SourcePrediction(
        source_type="amsr_ssm", bucket="catalog-precise", verdict="none-predicted",
        detail=" / ".join(p.detail for p in predictions),
    )


_PREDICATES["amsr_ssm"] = _predict_amsr_ssm


def _build_ismn_downloader(cfg) -> ISMNDownloader:
    """Thin wrapper (a separate function so tests can monkeypatch it
    directly, matching this module's own convention): constructs an
    ISMNDownloader. output_dir is never written to by
    station_date_ranges_dry, so a harmless placeholder is safe here."""
    return ISMNDownloader(output_dir=Path("/dev/null"))


def _predict_ismn(source, cfg, sar_footprints: "list[SarFootprint]") -> SourcePrediction:
    """Predicate for source_type="ismn", the first ground/point-bucket
    source: station_date_ranges_dry's own bbox argument is only a coarse
    pre-filter (it has no notion of a footprint's real shape), so
    _point_in_footprint is applied per matched station as the fine
    refinement -- a station inside a footprint's bbox but outside its
    true (e.g. rotated) polygon must not count as collocated. "unknown"
    (not "none-predicted") whenever no local ISMN archive is found at
    all, since ISMN has no download API and the absence of an archive is
    a missing precondition, not evidence that no station would match.

    station_date_ranges_dry rescans the local ISMN archive's .stm files
    from scratch on every call, so it is called exactly once here --
    against the bounding envelope (union) of every supplied footprint's
    own bbox, rather than once per footprint -- and the result is reused
    for every footprint's own (cheap, in-memory) fine refinement below.
    The union bbox can only widen the coarse pre-filter relative to a
    per-footprint call, never narrow it, so every station a per-footprint
    call would have found is still present here; _point_in_footprint
    still restricts each footprint's own matches to its real shape, so
    the set of stations ultimately matched per footprint is unchanged.
    For a single footprint the union bbox is exactly that footprint's own
    bbox.
    """
    if not sar_footprints:
        return SourcePrediction(
            source_type="ismn", bucket="ground-point", verdict="unknown",
            detail="No SAR footprints supplied -- cannot predict.",
        )

    dl = _build_ismn_downloader(cfg)
    tolerance = timedelta(minutes=_resolve_temporal_padding_minutes(cfg, "ismn"))

    union_min_lon = min(fp.bbox[0] for fp in sar_footprints)
    union_max_lon = max(fp.bbox[1] for fp in sar_footprints)
    union_min_lat = min(fp.bbox[2] for fp in sar_footprints)
    union_max_lat = max(fp.bbox[3] for fp in sar_footprints)

    ranges = dl.station_date_ranges_dry(union_min_lon, union_max_lon, union_min_lat, union_max_lat)
    if ranges is None:
        return SourcePrediction(
            source_type="ismn", bucket="ground-point", verdict="unknown",
            detail="No local ISMN archive found.",
            message=(
                "ISMN has no download API -- download an export once from "
                "https://ismn.earth/en/dataviewer/ and place it under "
                "data/_archive_cache/ismn (see README's ISMN credentials section)."
            ),
        )

    all_stations_matched: "set[str]" = set()
    for footprint in sar_footprints:
        padded_start = footprint.sensing_start - tolerance
        padded_end = footprint.sensing_end + tolerance
        for dir_prefix, (lat, lon, earliest, latest) in ranges.items():
            # ranges was pre-filtered against the union bbox above, which
            # is only ever a superset of this footprint's own bbox --
            # _point_in_footprint is still the fine refinement: a station
            # inside the footprint's bbox but outside its true (e.g.
            # rotated) polygon must not count as collocated, and a station
            # only inside another footprint's bbox (not this one's) must
            # not count here either.
            if not _point_in_footprint(lat, lon, footprint, cfg.collocation.sar_footprint_radius_km):
                continue
            if _windows_overlap(earliest, latest, padded_start, padded_end):
                all_stations_matched.add(dir_prefix)

    if all_stations_matched:
        return SourcePrediction(
            source_type="ismn", bucket="ground-point", verdict="collocated",
            detail=f"{len(all_stations_matched)} station(s) with data in the predicted window(s).",
            matched_stations=sorted(all_stations_matched),
        )
    return SourcePrediction(
        source_type="ismn", bucket="ground-point", verdict="none-predicted",
        detail="No ISMN station has data within any predicted SAR footprint window.",
    )


_PREDICATES["ismn"] = _predict_ismn


def _predict_insitu(source, cfg, sar_footprints: "list[SarFootprint]") -> SourcePrediction:
    """Predicate for the five real Copernicus Marine in-situ source
    types (mooring, buoy, drifter, ferrybox, tidal_gauge) -- see
    orchestrator.py's _INSITU_TYPES. Unlike those five types' real
    (non-dry) download path, which batches every requested platform type
    into one InSituDownloader.download(source_types=[...]) call,
    predict_source is called once per individual validation source, so
    this predicate filters its own check_availability_dry call down to
    just the single source.source_type it was invoked for.

    This bucket stays bbox-only (no _point_in_footprint refinement):
    check_availability_dry is a single aggregate "does any data exist"
    boolean from the Copernicus Marine service itself, with no per-station
    coordinates to check further -- unlike ISMN's local station index.
    """
    if not sar_footprints:
        return SourcePrediction(
            source_type=source.source_type, bucket="ground-point", verdict="unknown",
            detail="No SAR footprints supplied -- cannot predict.",
        )

    from ..downloaders.insitu_downloader import InSituDownloader

    dl = InSituDownloader(output_dir=Path("."))
    tolerance = timedelta(minutes=_resolve_temporal_padding_minutes(cfg, source.source_type))
    any_available = False

    for footprint in sar_footprints:
        padded_start = footprint.sensing_start - tolerance
        padded_end = footprint.sensing_end + tolerance
        try:
            if dl.check_availability_dry(
                footprint.bbox[0], footprint.bbox[1], footprint.bbox[2], footprint.bbox[3],
                padded_start.isoformat(), padded_end.isoformat(),
                source_types=[source.source_type],
            ):
                any_available = True
                break  # one confirmed hit is enough -- see _predict_model_source's identical short-circuit
        except Exception:
            logger.debug("_predict_insitu: availability check failed", exc_info=True)
            return SourcePrediction(
                source_type=source.source_type, bucket="ground-point", verdict="unknown",
                detail="In-situ availability check failed.",
            )

    verdict: Verdict = "collocated" if any_available else "none-predicted"
    return SourcePrediction(
        source_type=source.source_type, bucket="ground-point", verdict=verdict,
        detail=f"In-situ data {'found' if any_available else 'not found'} in predicted window(s).",
    )


for _insitu_type in ("mooring", "buoy", "drifter", "ferrybox", "tidal_gauge"):
    _PREDICATES[_insitu_type] = _predict_insitu


def _predict_insitu_currents_historical(source, cfg, sar_footprints: "list[SarFootprint]") -> SourcePrediction:
    """Predicate for the four real delayed-mode in-situ currents source
    types (adcp_historical, argo_historical, drifter_historical,
    glider_historical) -- see orchestrator.py's
    _download_currents_historical. There is no single source_type
    "insitu_currents_historical": unlike _predict_insitu's shared
    InSituDownloader instance (filtered per-call via
    source_types=[...]), instrument is a required *constructor* argument
    here, so each call builds its own InSituCurrentsHistoricalDownloader
    instance, with instrument derived by stripping the "_historical"
    suffix off source.source_type (e.g. "adcp_historical" -> "adcp").

    Delayed-mode data isn't finalized until
    insitu_currents_historical_downloader._MIN_AGE_DAYS (182 days, ~6
    months) after acquisition -- download() itself skips (no network
    call) for any window younger than that. A footprint younger than the
    cutoff is excluded from the availability check entirely and, if
    every supplied footprint is too recent, the verdict is "unknown"
    (not "none-predicted"): a too-recent footprint says nothing about
    whether data would eventually appear once the archive catches up.
    Age is compared via a bare date string (mirroring _predict_ascat_ssm's
    own _ASCAT_COVERAGE_CUTOFF comparison) rather than raw datetimes, since
    a SarFootprint's sensing_start/sensing_end are not guaranteed
    timezone-aware.

    This bucket stays bbox-only (no _point_in_footprint refinement) for
    the same reason _predict_insitu does: check_availability_dry is a
    single aggregate "does any data exist" boolean, with no per-platform
    coordinates to check further.
    """
    from ..downloaders.insitu_currents_historical_downloader import (
        _MIN_AGE_DAYS,
        InSituCurrentsHistoricalDownloader,
    )

    instrument = source.source_type.removesuffix("_historical")
    tolerance = timedelta(minutes=_resolve_temporal_padding_minutes(cfg, source.source_type))
    cutoff_date = (datetime.now(timezone.utc) - timedelta(days=_MIN_AGE_DAYS)).date().isoformat()

    eligible_footprints = [
        fp for fp in sar_footprints
        if (fp.sensing_end + tolerance).date().isoformat() <= cutoff_date
    ]
    if not eligible_footprints:
        return SourcePrediction(
            source_type=source.source_type, bucket="ground-point", verdict="unknown",
            detail=(
                f"No SAR footprint is old enough for the delayed-mode currents "
                f"archive (data lags real-time by {_MIN_AGE_DAYS} days)."
            ),
        )

    dl = InSituCurrentsHistoricalDownloader(instrument=instrument, output_dir=Path("."))
    any_available = False

    for footprint in eligible_footprints:
        padded_start = footprint.sensing_start - tolerance
        padded_end = footprint.sensing_end + tolerance
        try:
            if dl.check_availability_dry(
                footprint.bbox[0], footprint.bbox[1], footprint.bbox[2], footprint.bbox[3],
                padded_start.isoformat(), padded_end.isoformat(),
            ):
                any_available = True
                break  # one confirmed hit is enough -- see _predict_model_source's identical short-circuit
        except Exception:
            logger.debug(
                "_predict_insitu_currents_historical: availability check failed", exc_info=True,
            )
            return SourcePrediction(
                source_type=source.source_type, bucket="ground-point", verdict="unknown",
                detail="Delayed-mode currents availability check failed.",
            )

    verdict: Verdict = "collocated" if any_available else "none-predicted"
    return SourcePrediction(
        source_type=source.source_type, bucket="ground-point", verdict=verdict,
        detail=f"Delayed-mode currents data {'found' if any_available else 'not found'} in predicted window(s).",
    )


for _currents_historical_type in (
    "adcp_historical", "argo_historical", "drifter_historical", "glider_historical",
):
    _PREDICATES[_currents_historical_type] = _predict_insitu_currents_historical


def _hf_radar_candidate_regions(
    regions_table: "dict", footprint: "SarFootprint",
) -> "list[tuple[str, tuple[float, float, float, float]]]":
    """(region_name, intersected_bbox) for every entry in *regions_table*
    (HFR_REGIONS or NOAA_HFR_REGIONS -- both share a "bbox" field per
    entry) whose own bbox genuinely overlaps *footprint*'s real shape, per
    _bbox_overlaps_footprint -- an area-vs-area refinement over a naive
    "does the recipe's own bbox intersect the region's bbox" comparison,
    so a footprint sitting only in one corner of a large multi-region
    country still gets attributed to the right specific region(s), not
    just whichever region happens to have the largest overlap with the
    recipe's own (typically much larger) requested bbox.

    intersected_bbox is entirely inside the matched region's own bbox
    (clamped on all four bounds), so passing it on to that region's own
    resolve_hfr_region/match_noaa_hfr_region call (inside the downloader's
    check_availability_dry) is guaranteed to resolve back to this exact
    region, never a differently-overlapping neighbor.
    """
    f_min_lon, f_max_lon, f_min_lat, f_max_lat = footprint.bbox
    candidates: "list[tuple[str, tuple[float, float, float, float]]]" = []
    for name, region in regions_table.items():
        r_min_lon, r_max_lon, r_min_lat, r_max_lat = region["bbox"]
        if not _bbox_overlaps_footprint(r_min_lon, r_max_lon, r_min_lat, r_max_lat, footprint):
            continue
        intersected = (
            max(f_min_lon, r_min_lon), min(f_max_lon, r_max_lon),
            max(f_min_lat, r_min_lat), min(f_max_lat, r_max_lat),
        )
        candidates.append((name, intersected))
    return candidates


def _predict_hf_radar(
    source,
    cfg,
    sar_footprints: "list[SarFootprint]",
    *,
    check_availability_dry: "Callable[..., bool]",
    regions_table: "dict",
    source_type: str,
) -> SourcePrediction:
    """Shared predicate for the three HF-radar source_types with a real
    per-region grid table of their own: "hf_radar" (Copernicus Marine
    NRT), "hf_radar_historical" (Copernicus Marine delayed-mode), and
    "hf_radar_noaa" (NOAA ERDDAP). "hf_radar_us" (the ERDDAP->THREDDS->
    Copernicus waterfall) has its own _predict_hf_radar_us instead, since
    it delegates to the other four downloaders' own check_availability_dry
    rather than owning a single region table itself.

    HF-radar's real unit is a whole grid region, not a point station --
    unlike ISMN/in-situ (_predict_ismn/_predict_insitu), the coarse pass
    here is a per-region _bbox_overlaps_footprint area check
    (_hf_radar_candidate_regions) *before* ever calling
    check_availability_dry, rather than a raw footprint-bbox query
    followed by a point-in-shape refinement -- this is what lets a
    footprint sitting only in one corner of a large multi-region country
    still get attributed to the right specific region(s), rather than
    whichever region the recipe's own (typically much larger) nominal
    bbox happens to overlap most.

    All four real HF-radar source_types resolve their time tolerance via
    the same "hf_radar_grid" key (not their own source_type string) --
    a single shared tolerance-lookup key across all four, not a
    per-source_type one, matching orchestrator.py's own
    _padded_temporal_bounds("hf_radar_grid") call sites.
    """
    if not sar_footprints:
        return SourcePrediction(
            source_type=source_type, bucket="ground-point", verdict="unknown",
            detail="No SAR footprints supplied -- cannot predict.",
        )

    tolerance = timedelta(minutes=_resolve_temporal_padding_minutes(cfg, "hf_radar_grid"))
    any_available = False

    for footprint in sar_footprints:
        padded_start = footprint.sensing_start - tolerance
        padded_end = footprint.sensing_end + tolerance
        for _region_name, (r_min_lon, r_max_lon, r_min_lat, r_max_lat) in _hf_radar_candidate_regions(
            regions_table, footprint,
        ):
            try:
                if check_availability_dry(
                    r_min_lon, r_max_lon, r_min_lat, r_max_lat,
                    padded_start.isoformat(), padded_end.isoformat(),
                ):
                    any_available = True
                    break  # one confirmed hit is enough -- see _predict_model_source's identical short-circuit
            except Exception:
                logger.debug("_predict_hf_radar: availability check failed", exc_info=True)
                return SourcePrediction(
                    source_type=source_type, bucket="ground-point", verdict="unknown",
                    detail=f"HF-radar availability check failed for {source_type}.",
                )
        if any_available:
            break

    verdict: Verdict = "collocated" if any_available else "none-predicted"
    return SourcePrediction(
        source_type=source_type, bucket="ground-point", verdict=verdict,
        detail=f"HF-radar data {'found' if any_available else 'not found'} in predicted window(s).",
    )


def _hf_radar_copernicus_check_dry(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float, start: str, end: str,
) -> bool:
    """Thin wrapper: constructs an HFRadarDownloader and calls its
    check_availability_dry. output_dir is never written to by that
    method, so a harmless placeholder is safe here."""
    dl = HFRadarDownloader(output_dir=Path("/dev/null"))
    return dl.check_availability_dry(min_lon, max_lon, min_lat, max_lat, start, end)


def _predict_hf_radar_copernicus(source, cfg, sar_footprints: "list[SarFootprint]") -> SourcePrediction:
    """Predicate for source_type="hf_radar" (Copernicus Marine NRT grid)."""
    return _predict_hf_radar(
        source, cfg, sar_footprints,
        check_availability_dry=_hf_radar_copernicus_check_dry,
        regions_table=HFR_REGIONS,
        source_type="hf_radar",
    )


_PREDICATES["hf_radar"] = _predict_hf_radar_copernicus


def _hf_radar_historical_check_dry(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float, start: str, end: str,
) -> bool:
    """Thin wrapper: constructs an HFRadarHistoricalDownloader and calls
    its check_availability_dry. output_dir is never written to by that
    method, so a harmless placeholder is safe here."""
    dl = HFRadarHistoricalDownloader(output_dir=Path("/dev/null"))
    return dl.check_availability_dry(min_lon, max_lon, min_lat, max_lat, start, end)


def _predict_hf_radar_historical(source, cfg, sar_footprints: "list[SarFootprint]") -> SourcePrediction:
    """Predicate for source_type="hf_radar_historical" (Copernicus Marine
    delayed-mode archive)."""
    return _predict_hf_radar(
        source, cfg, sar_footprints,
        check_availability_dry=_hf_radar_historical_check_dry,
        regions_table=HFR_REGIONS,
        source_type="hf_radar_historical",
    )


_PREDICATES["hf_radar_historical"] = _predict_hf_radar_historical


def _hf_radar_noaa_check_dry(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float, start: str, end: str,
) -> bool:
    """Thin wrapper: constructs a NOAAHFRadarDownloader and calls its
    check_availability_dry. output_dir is never written to by that
    method, so a harmless placeholder is safe here."""
    dl = NOAAHFRadarDownloader(output_dir=Path("/dev/null"))
    return dl.check_availability_dry(min_lon, max_lon, min_lat, max_lat, start, end)


def _predict_hf_radar_noaa(source, cfg, sar_footprints: "list[SarFootprint]") -> SourcePrediction:
    """Predicate for source_type="hf_radar_noaa" (NOAA ERDDAP griddap)."""
    return _predict_hf_radar(
        source, cfg, sar_footprints,
        check_availability_dry=_hf_radar_noaa_check_dry,
        regions_table=NOAA_HFR_REGIONS,
        source_type="hf_radar_noaa",
    )


_PREDICATES["hf_radar_noaa"] = _predict_hf_radar_noaa


def _hf_radar_us_check_dry(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float, start: str, end: str,
) -> bool:
    """Thin wrapper: constructs an HFRadarUSDownloader and calls its
    check_availability_dry. output_dir is never written to by that
    method, so a harmless placeholder is safe here."""
    dl = HFRadarUSDownloader(output_dir=Path("/dev/null"))
    return dl.check_availability_dry(min_lon, max_lon, min_lat, max_lat, start, end)


def _predict_hf_radar_us(source, cfg, sar_footprints: "list[SarFootprint]") -> SourcePrediction:
    """Predicate for source_type="hf_radar_us" (the ERDDAP->THREDDS->
    Copernicus waterfall). Unlike _predict_hf_radar's three sibling
    registrations, this delegates directly to HFRadarUSDownloader.
    check_availability_dry per footprint's own (un-refined) bbox -- no
    local per-region-table iteration here, since that delegator's own
    check_availability_dry already resolves whichever single NOAA or
    Copernicus region applies via its own internal waterfall (mirroring
    download()'s own region resolution), the same way each of the other
    four sources resolves its own region internally.
    """
    if not sar_footprints:
        return SourcePrediction(
            source_type="hf_radar_us", bucket="ground-point", verdict="unknown",
            detail="No SAR footprints supplied -- cannot predict.",
        )

    tolerance = timedelta(minutes=_resolve_temporal_padding_minutes(cfg, "hf_radar_grid"))
    any_available = False

    for footprint in sar_footprints:
        padded_start = footprint.sensing_start - tolerance
        padded_end = footprint.sensing_end + tolerance
        try:
            if _hf_radar_us_check_dry(
                footprint.bbox[0], footprint.bbox[1], footprint.bbox[2], footprint.bbox[3],
                padded_start.isoformat(), padded_end.isoformat(),
            ):
                any_available = True
                break  # one confirmed hit is enough -- see _predict_model_source's identical short-circuit
        except Exception:
            logger.debug("_predict_hf_radar_us: availability check failed", exc_info=True)
            return SourcePrediction(
                source_type="hf_radar_us", bucket="ground-point", verdict="unknown",
                detail="HF-radar availability check failed for hf_radar_us.",
            )

    verdict: Verdict = "collocated" if any_available else "none-predicted"
    return SourcePrediction(
        source_type="hf_radar_us", bucket="ground-point", verdict=verdict,
        detail=f"HF-radar data {'found' if any_available else 'not found'} in predicted window(s).",
    )


_PREDICATES["hf_radar_us"] = _predict_hf_radar_us


def _predict_global_composite(
    source,
    cfg,
    sar_footprints: "list[SarFootprint]",
    *,
    check_exists_dry: "Callable[[date], bool]",
    source_type: str,
) -> SourcePrediction:
    """Shared predicate for the global-composite bucket (RSS radiometer,
    CDS SSM): both sources publish one daily global-coverage file, so
    spatial refinement isn't meaningful -- a footprint's own bbox/polygon
    is never consulted, only which calendar day(s) its (padded) sensing
    window touches. Collocated if any of those days has data available
    from *check_exists_dry*.

    Each footprint's sensing window is padded by the resolved collocation
    time tolerance before being converted to calendar days -- mirroring
    the real download path's own padding (see
    ``orchestrator._padded_temporal_bounds``) -- since a footprint near a
    UTC day boundary can pull real data from an adjacent day once that
    padding is applied, which checking only the footprint's own unpadded
    ``sensing_start`` date would miss.
    """
    if not sar_footprints:
        return SourcePrediction(
            source_type=source_type, bucket="global-composite", verdict="unknown",
            detail="No SAR footprints supplied -- cannot predict.",
        )

    tolerance = timedelta(minutes=_resolve_temporal_padding_minutes(cfg, source_type))

    any_exists = False
    for footprint in sar_footprints:
        padded_start = footprint.sensing_start - tolerance
        padded_end = footprint.sensing_end + tolerance
        days: "list[date]" = []
        d = padded_start.date()
        while d <= padded_end.date():
            days.append(d)
            d += timedelta(days=1)

        for day in days:
            try:
                if check_exists_dry(day):
                    any_exists = True
                    break  # one confirmed hit is enough -- see _predict_model_source's identical short-circuit
            except Exception:
                logger.debug("_predict_global_composite: existence check failed", exc_info=True)
                return SourcePrediction(
                    source_type=source_type, bucket="global-composite", verdict="unknown",
                    detail=f"Existence check failed for {source_type}.",
                )
        if any_exists:
            break

    verdict: Verdict = "collocated" if any_exists else "none-predicted"
    return SourcePrediction(
        source_type=source_type, bucket="global-composite", verdict=verdict,
        detail=f"{source_type} data {'found' if any_exists else 'not found'} for the SAR footprint day(s).",
    )


def _radiometer_check_dry(source, day: date) -> bool:
    """Thin wrapper: constructs a RadiometerDownloader and calls its
    check_exists_dry, restricted to source.download_kwargs["sensors"] when
    present (mirroring orchestrator.py's _download_radiometer, which passes
    that same key straight through to download()). output_dir is never
    written to by check_exists_dry, so a harmless placeholder is safe here."""
    dl = RadiometerDownloader(output_dir=Path("/dev/null"))
    sensors = source.download_kwargs.get("sensors")
    return dl.check_exists_dry(day, sensors=sensors)


def _cds_ssm_check_dry(source, day: date) -> bool:
    """Thin wrapper: constructs a CDSSoilMoistureDownloader for
    source.download_kwargs["product_type"] (default "active", mirroring
    orchestrator.py's own _download_cds_ssm default) and calls its
    check_availability_dry. output_dir is never written to by
    check_availability_dry, so a harmless placeholder is safe here."""
    product_type = source.download_kwargs.get("product_type", "active")
    dl = CDSSoilMoistureDownloader(product_type=product_type, output_dir=Path("/dev/null"))
    return dl.check_availability_dry(day)


def _predict_radiometer(source, cfg, sar_footprints: "list[SarFootprint]") -> SourcePrediction:
    """Predicate for source_type="radiometer" (RSS microwave radiometer
    daily gridded ocean-wind product)."""
    return _predict_global_composite(
        source, cfg, sar_footprints,
        check_exists_dry=lambda day: _radiometer_check_dry(source, day),
        source_type="radiometer",
    )


def _predict_cds_ssm(source, cfg, sar_footprints: "list[SarFootprint]") -> SourcePrediction:
    """Predicate for source_type="cds_ssm" (C3S CDS daily gridded satellite
    soil moisture product)."""
    return _predict_global_composite(
        source, cfg, sar_footprints,
        check_exists_dry=lambda day: _cds_ssm_check_dry(source, day),
        source_type="cds_ssm",
    )


_PREDICATES["radiometer"] = _predict_radiometer
_PREDICATES["cds_ssm"] = _predict_cds_ssm


#: Footprints whose sensing_start falls within this many days of "now"
#: get an extra live-probe check before a model source's temporal-
#: coverage-window verdict is trusted as "collocated" -- a model's most
#: recent granule(s) may not be published yet even though the date is
#: nominally within its documented coverage window.
_MODEL_RECENT_PROBE_WINDOW_DAYS = 30


def _predict_model_source(
    source,
    cfg,
    sar_footprints: "list[SarFootprint]",
    *,
    coverage_start: "Optional[datetime]",
    coverage_end: "Optional[datetime]",
    source_type: str,
    live_probe: "Optional[Callable[[SarFootprint], bool]]" = None,
) -> SourcePrediction:
    """Models (ERA5, HYCOM): no spatial check -- global/regional grid
    coverage is assumed. Only a temporal-coverage-window check against
    the source's documented availability boundary, plus (when
    *live_probe* is given) a follow-up live check for any footprint
    within _MODEL_RECENT_PROBE_WINDOW_DAYS of today -- a model's most
    recent data may not be published yet even though the window is
    nominally within coverage. A footprint's own recency is compared via
    a bare calendar date (not raw datetime subtraction), since a
    SarFootprint's sensing_start/sensing_end are not guaranteed
    timezone-aware.

    A single footprint confirmed collocated (either not recent, or
    recent and live_probe returned True) is enough for an overall
    "collocated" verdict, even if other footprints' probes failed or
    came back empty -- only when EVERY in-coverage footprint is both
    recent and unconfirmed does the verdict fall back to "unknown"
    (couldn't confirm yet, not evidence of absence), and a live-probe
    exception is likewise "unknown", never a false "none-predicted".
    """
    if not sar_footprints:
        return SourcePrediction(
            source_type=source_type, bucket="model", verdict="unknown",
            detail="No SAR footprints supplied -- cannot predict.",
        )

    recent_cutoff = (
        datetime.now(timezone.utc) - timedelta(days=_MODEL_RECENT_PROBE_WINDOW_DAYS)
    ).date()

    any_confirmed = False
    any_recent_unconfirmed = False
    probe_failed = False

    for footprint in sar_footprints:
        # _to_naive_utc: coverage_start/coverage_end are plain module-level
        # constants (naive) while a real footprint's sensing_start/sensing_end
        # are typically tz-aware -- normalize both sides before comparing to
        # avoid "can't compare offset-naive and offset-aware datetimes".
        if coverage_start is not None and _to_naive_utc(footprint.sensing_end) < _to_naive_utc(coverage_start):
            continue
        if coverage_end is not None and _to_naive_utc(footprint.sensing_start) > _to_naive_utc(coverage_end):
            continue

        is_recent = footprint.sensing_start.date() >= recent_cutoff
        if not is_recent or live_probe is None:
            any_confirmed = True
            continue

        try:
            if live_probe(footprint):
                any_confirmed = True
                break
            else:
                any_recent_unconfirmed = True
        except Exception:
            logger.debug("_predict_model_source: live probe failed for %s", source_type, exc_info=True)
            probe_failed = True

    if any_confirmed:
        return SourcePrediction(
            source_type=source_type, bucket="model", verdict="collocated",
            detail=f"SAR footprint window(s) within {source_type}'s documented coverage.",
        )
    if probe_failed:
        return SourcePrediction(
            source_type=source_type, bucket="model", verdict="unknown",
            detail=f"Live coverage probe failed for {source_type}.",
        )
    if any_recent_unconfirmed:
        return SourcePrediction(
            source_type=source_type, bucket="model", verdict="unknown",
            detail=(
                f"SAR footprint window(s) within {source_type}'s documented coverage, "
                f"but {source_type} has not yet published data for the most recent "
                f"footprint(s) (live probe found no granule)."
            ),
        )
    return SourcePrediction(
        source_type=source_type, bucket="model", verdict="none-predicted",
        detail=f"SAR footprint window(s) outside {source_type}'s documented coverage.",
    )


def _hycom_live_probe(footprint: "SarFootprint") -> bool:
    """Whether HyCOM's live OPeNDAP dataset already has a granule for
    footprint's own sensing window -- reuses HycomDownloader.has_coverage
    (metadata-only, no full grid load) per dataset-segment touched by the
    window (see _resolve_hycom_segments).

    _resolve_hycom_segments compares its arguments against the naive
    _HYCOM_MIN_DATE/_HYCOM_CUTOVER_DATE constants (matching the real
    download path's own always-naive windows -- see
    orchestrator._padded_temporal_bounds); a real SarFootprint's
    sensing_start/sensing_end is typically tz-aware, so it is normalized
    via _to_naive_utc first to avoid "can't compare offset-naive and
    offset-aware datetimes"."""
    segments = _resolve_hycom_segments(
        _to_naive_utc(footprint.sensing_start), _to_naive_utc(footprint.sensing_end),
    )
    dl = HycomDownloader(output_dir=Path("/dev/null"))
    clip_at_cutover = len(segments) == 2
    return any(
        dl.has_coverage(dataset_key, seg_start, seg_end, clip_at_cutover)
        for dataset_key, seg_start, seg_end in segments
    )


def _era5_live_probe(cfg, footprint: "SarFootprint") -> bool:
    """Whether ERA5's live CDS catalogue already reports data covering
    footprint's own sensing day -- reuses ERA5Downloader.check_availability_dry
    (a fast, unauthenticated catalogue-extent lookup, never a real
    cdsapi.Client.retrieve() job), against whichever CDS dataset cfg's
    own configured variable maps to."""
    dl = ERA5Downloader(variable=cfg.variable, output_dir=Path("/dev/null"))
    return dl.check_availability_dry(footprint.sensing_start.date())


_PREDICATES["hycom"] = lambda source, cfg, sar_footprints: _predict_model_source(
    source, cfg, sar_footprints,
    coverage_start=_HYCOM_MIN_DATE, coverage_end=None, source_type="hycom",
    live_probe=_hycom_live_probe,
)
_PREDICATES["era5"] = lambda source, cfg, sar_footprints: _predict_model_source(
    source, cfg, sar_footprints,
    coverage_start=None, coverage_end=None, source_type="era5",
    live_probe=lambda footprint: _era5_live_probe(cfg, footprint),
)
