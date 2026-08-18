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

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Literal, Optional, Tuple

from ..downloaders.base import CopernicusODataClient, authenticate_cdse
from .orbit_coverage import _point_in_bbox

__all__ = ["SarFootprint"]

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
    """(points, bbox) from a CDSE WV-mode GeoFootprint. CONFIRMED LIVE
    2026-08-18 (curl against the real CDSE OData API): CDSE catalogs an
    entire WV pass as ONE product, not one vignette per catalog entry --
    GeoFootprint's type is "MultiPolygon" and its "coordinates" list
    already holds one small ~20-30km quad ring per vignette (125-145 per
    product in three live samples inspected), directly in the catalog
    search response. No manifest.safe fetch via CDSE's
    /Products({id})/Nodes(...) endpoint is needed. Each
    ``coordinates[i]`` entry is always ``[exterior_ring]`` -- a
    single-element list holding that vignette's one ring of [lon, lat]
    pairs (no interior holes, no flatter variant) -- CONFIRMED against a
    live payload, so the ring is read via a fixed ``polygon_entry[0]``
    descent, mirroring _polygon_and_bbox_from_geofootprint's
    ``coordinates[0]`` pattern.

    Each ring's centroid becomes one point: the mean of its deduped
    unique [lon, lat] vertices (GeoJSON rings repeat their first vertex
    to close the loop, so that trailing duplicate is dropped first --
    mirroring _polygon_and_bbox_from_geofootprint's convention).

    A WV product's overall catalog envelope only guarantees it TOUCHES
    the recipe's requested bbox (see query_products) -- real WV products
    can span very large geographic extents (confirmed live: one real
    product spanned ~700km along-track, 145 vignettes) over open ocean,
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
    see _vignette_points_and_bbox_from_geofootprint for the confirmed-live
    finding that CDSE catalogs an entire WV pass (many vignettes) as one
    product, with per-vignette geometry already in the catalog response,
    and for why each vignette centroid is filtered against
    cfg.geographic_bounds before being kept (a WV product's own catalog
    envelope only guarantees it touches the recipe's requested bbox, not
    that most -- or any -- of its individual vignettes fall inside it).
    Non-WV granules are excluded here -- see
    _discover_sentinel1_ocn_footprints_dry (Task 3)."""
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


def _discover_sentinel1_ocn_footprints_dry(cfg) -> "list[SarFootprint]":
    """Non-WV (IW/EW/SM) Sentinel-1 OCN footprints from a dry CDSE
    catalog search. WV-mode granules are excluded here -- see
    _discover_sentinel1_wv_footprints_dry (Task 4)."""
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
