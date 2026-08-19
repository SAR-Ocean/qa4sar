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

import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple

import pandas as pd

from ..downloaders import radarsat2_wind_downloader as _rs2
from ..downloaders.ascat_soil_moisture_downloader import ASCATSoilMoistureDownloader
from ..downloaders.base import (
    CopernicusODataClient,
    authenticate_cdse,
    months_touched,
    normalize_datetime,
    prefer_ipv4_dns,
    split_antimeridian_bbox,
)
from ..downloaders.hsaf_downloader import HSAFDownloader
from ..downloaders.hsaf_downloader import _parse_satellite as _hsaf_parse_satellite
from . import orbit_coverage
from .collocation import _haversine_distance
from .datatree_converter import DataTreeConverter
from .orbit_coverage import _point_in_bbox
from .orchestrator import _resolve_temporal_padding_minutes

logger = logging.getLogger(__name__)

__all__ = ["SarFootprint", "Verdict", "SourcePrediction", "CollocationReport", "predict_source"]

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
    return client.query_products(
        collection="SENTINEL-1",
        product_type="OCN",
        start_date=cfg.temporal_bounds.start,
        end_date=cfg.temporal_bounds.end,
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


def sar_footprints_from_downloaded(sar_files, sar_source_spec) -> "list[SarFootprint]":
    """Footprints from already-downloaded, already-converted SAR files --
    used by a real run's inline (non-dry) prediction path. sar_source_spec
    is the matching entry from SAR_SOURCES (core/sar_sources.py).

    Only "sentinel1_clms_ssm" is wired here so far -- the other three
    SAR_SOURCES keys (Sentinel-1 OCN, RADARSAT-2, NISAR SME2) raise
    NotImplementedError below. This is a deliberately deferred gap, not
    an oversight: wiring their from-downloaded paths is left to whichever
    later plan first needs it, since nothing in this plan's own tests
    exercises it for those three sources."""
    if sar_source_spec.key == "sentinel1_clms_ssm":
        footprints = []
        for path in sar_files:
            fp = _clms_ssm_footprint_from_downloaded(path)
            if fp is not None:
                footprints.append(fp)
        return footprints
    raise NotImplementedError(
        f"sar_footprints_from_downloaded: {sar_source_spec.key!r} not yet wired -- "
        f"non-CLMS-SSM sources reuse SAR_SOURCES[...].convert(...) directly the same "
        f"way _compute_sar_scene_times does; add that dispatch branch here when wiring "
        f"the corresponding source into a later plan."
    )


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


def predict_source(source, cfg, sar_footprints: "list[SarFootprint]") -> SourcePrediction:
    """Dispatch to the right bucket's predicate for source.source_type.
    An unrecognized source_type is itself an "unknown" verdict -- never
    raises, since the report loop calls this once per configured
    validation source and must not crash on one bad/future entry."""
    predicate = _PREDICATES.get(source.source_type)
    if predicate is None:
        return SourcePrediction(
            source_type=source.source_type, bucket="unregistered", verdict="unknown",
            detail=f"No dry-collocation predicate registered for source_type={source.source_type!r}.",
        )
    return predicate(source, cfg, sar_footprints)


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


def _windows_overlap(
    a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime,
) -> bool:
    """Whether time windows [a_start, a_end] and [b_start, b_end] overlap
    at all (inclusive of touching endpoints)."""
    return a_start <= b_end and b_start <= a_end


def _predict_orbit_corridor_source(
    source,
    cfg,
    sar_footprints: "list[SarFootprint]",
    *,
    satellite_resolver: "Callable[[str], str]",
    list_candidates_dry: "Callable[..., list[tuple[str, datetime, datetime]]]",
    source_type: str,
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
    """
    if not sar_footprints:
        return SourcePrediction(
            source_type=source_type, bucket="orbit-corridor", verdict="unknown",
            detail="No SAR footprints supplied -- cannot predict.",
        )

    tolerance = timedelta(minutes=_resolve_temporal_padding_minutes(cfg, source_type))
    matched_windows: "list[tuple[datetime, datetime]]" = []

    for footprint in sar_footprints:
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
            try:
                satellite = satellite_resolver(candidate_name)
            except Exception:
                logger.debug("_predict_orbit_corridor_source: satellite_resolver failed", exc_info=True)
                matched_windows.append((cand_start, cand_end))  # fail open
                continue
            for point in points:
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


def _predict_ascat_ssm(source, cfg, sar_footprints: "list[SarFootprint]") -> SourcePrediction:
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

    predictions: "list[SourcePrediction]" = []
    if eumdac_footprints:
        predictions.append(_predict_orbit_corridor_source(
            source, cfg, eumdac_footprints,
            satellite_resolver=_parse_eumdac_ascat_satellite,
            list_candidates_dry=_eumdac_ascat_ssm_list_candidates_dry,
            source_type="ascat_ssm",
        ))
    if hsaf_footprints:
        predictions.append(_predict_orbit_corridor_source(
            source, cfg, hsaf_footprints,
            satellite_resolver=_hsaf_satellite_resolver,
            list_candidates_dry=_hsaf_list_candidates_dry,
            source_type="ascat_ssm",
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
