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

from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional, Tuple

__all__ = ["SarFootprint"]


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
