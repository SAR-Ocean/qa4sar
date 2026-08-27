"""
Collocation algorithms — step 3 of the validation pipeline.

Four collocation types are supported (see README.md's "Collocation
types" table):

  Point vs. Point  — mooring/buoy/ferrybox/drifter/tidal gauge vs. SAR
                     WV-mode vignettes (plain average, no distance
                     weighting)
  Point vs. Layer  — mooring/buoy/ferrybox/drifter/tidal gauge/HF radar
                     vs. SAR grid 
  Layer vs. Layer  — scatterometer/altimeter/radiometer/HF-radar grid/
                     satellite soil moisture vs. SAR grid
  Model vs. Layer  — ERA5/HYCOM (gridded background field) vs. SAR
                     grid, via bilinear spatial + temporal interpolation

Point vs. Point and Point vs. Layer are both produced by
``PointLayerCollocation`` (the ``collocation_type`` label on each
result distinguishes them); ``LayerLayerCollocation`` implements
Layer vs. Layer; ``ModelLayerCollocation`` (model_collocation.py)
implements Model vs. Layer.

:func:`run_collocation` is the entry point: it dispatches each
validation source in a recipe's DataTree to the matching type and
saves the combined result to ``collocation_results.nc``.
"""


from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    # scipy is only needed when collocation actually runs and is imported
    # lazily inside those functions; bind cKDTree here purely so the type
    # annotations below resolve.
    from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

__all__ = [
    "CollocatedPoint",
    "PointLayerCollocation",
    "LayerLayerCollocation",
    "run_collocation",
    "_detect_collocation_type",
    "_resolve_layer_type",
    "LAYER_DATA_TYPES",
    "LAYER_SOURCE_PATHS",
    "SSM_SATELLITE_LAYER_TYPES",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CollocatedPoint:
    """
    One matched pair between a SAR grid cell and a validation observation.
    """

    # SAR side
    sar_lon:  float
    sar_lat:  float
    sar_time: datetime
    sar_data: Dict[str, float]   # variable_name → value

    # Validation side
    val_lon:  float
    val_lat:  float
    val_time: datetime
    val_data: Dict[str, float]   # variable_name → value

    # Quality / offset metrics
    spatial_distance_km:        float
    temporal_distance_minutes:  float

    # Provenance
    val_source: str              # e.g. "mooring", "buoy", "scatterometer"
    val_id: Optional[str] = None
    collocation_type: str = "point_vs_layer"  # point_vs_point | point_vs_layer | layer_vs_layer

    # Pixel indices — the SAR pixel and scene this observation was matched to
    sar_y_idx: int = 0
    sar_x_idx: int = 0
    sar_scene_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sar_lon":                   self.sar_lon,
            "sar_lat":                   self.sar_lat,
            "sar_time":                  self.sar_time.isoformat(),
            "sar_data":                  self.sar_data,
            "val_lon":                   self.val_lon,
            "val_lat":                   self.val_lat,
            "val_time":                  self.val_time.isoformat(),
            "val_data":                  self.val_data,
            "spatial_distance_km":       self.spatial_distance_km,
            "temporal_distance_minutes": self.temporal_distance_minutes,
            "val_source":                self.val_source,
            "val_id":                    self.val_id,
            "collocation_type":          self.collocation_type,
            "sar_y_idx":                 self.sar_y_idx,
            "sar_x_idx":                 self.sar_x_idx,
            "sar_scene_name":            self.sar_scene_name,
        }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _haversine_distance(
    lon1: float, lat1: float,
    lon2: float, lat2: float,
) -> float:
    """Great-circle distance in km (scalar inputs)."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    )
    return R * 2.0 * np.arcsin(np.sqrt(a))


def _project_currents_to_radial(ewct: float, nsct: float, heading_deg: float) -> float:
    """
    Project an eastward/northward current onto the SAR radial (line-of-sight).

    The SAR line-of-sight is the range direction, perpendicular to the platform
    heading ``rvlHeading`` (azimuth/along-track), hence the ``- 90``. The result
    is the quantity compared against the L2 OCN ``rvlRadVel`` product
    (``rvlRadVel_projection``).

    Typed here for this module's own scalar-per-row usage; the underlying 
    arithmetic is generic over numpy arrays too via broadcasting.

    Reference: Martin, Gommenginger, Jacob & Staneva (2022), RSE 268:112758.
    """
    heading_rad = np.radians(heading_deg - 90.0)
    return ewct * np.cos(heading_rad) + nsct * np.sin(heading_rad)


def _haversine_distance_grid(
    lon1: float, lat1: float,
    grid_lon: np.ndarray, grid_lat: np.ndarray,
) -> np.ndarray:
    """
    Great-circle distance from a scalar point to every cell in a 2-D grid (km).
    """
    R = 6371.0
    dlat = np.radians(grid_lat - lat1)
    dlon = np.radians(grid_lon - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(grid_lat)) * np.sin(dlon / 2.0) ** 2
    )
    return R * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _lonlat_to_unit_xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """
    Convert longitude/latitude (degrees) to Cartesian coordinates on the unit
    sphere, shape ``(..., 3)``.

    Chord length between two such points is a monotonic function of their
    great-circle angular separation, so a Euclidean nearest-neighbour search
    (e.g. via ``scipy.spatial.cKDTree``) on these coordinates gives the same
    result as a Haversine nearest-neighbour search, at KD-tree speed.
    """
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.stack([x, y, z], axis=-1)


def _to_datetime_array(time_array) -> np.ndarray:
    """
    Normalise heterogeneous time inputs to an object array of Python datetimes.
    """
    if not isinstance(time_array, (list, tuple, np.ndarray)):
        time_array = [time_array]
    result = []
    for t in time_array:
        if isinstance(t, datetime):
            result.append(t)
        elif isinstance(t, (np.datetime64, pd.Timestamp)):
            result.append(pd.Timestamp(t).to_pydatetime())
        else:
            result.append(t)
    return np.asarray(result, dtype=object)


# ---------------------------------------------------------------------------
# Distance-weighting functions for SAR aggregation
# ---------------------------------------------------------------------------

def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normalize *weights* to sum to 1.0; unchanged if the sum is 0.
    """
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        weights = weights / weight_sum
    return weights


def _gaussian_weights(distances: np.ndarray, sigma_km: float) -> np.ndarray:
    """
    Compute normalized Gaussian distance weights.
    
    Formula: w_i = exp(-d_i^2 / (2*sigma^2)), normalized so sum = 1.0
    
    Parameters
    ----------
    distances : np.ndarray
        Array of distances in km.
    sigma_km : float
        Standard deviation in km.
    
    Returns
    -------
    np.ndarray
        Normalized weights, same shape as distances.
    """
    weights = np.exp(-(distances ** 2) / (2 * sigma_km ** 2))
    return _normalize_weights(weights)


def _inverse_distance_weights(distances: np.ndarray, power: float = 2.0) -> np.ndarray:
    """
    Compute normalized inverse-distance weights.
    
    Formula: w_i = 1 / (d_i^power), normalized so sum = 1.0
    
    Parameters
    ----------
    distances : np.ndarray
        Array of distances in km. Must be > 0 for all elements.
    power : float
        Exponent for inverse distance (default: 2.0).
    
    Returns
    -------
    np.ndarray
        Normalized weights, same shape as distances.
    """
    # Avoid division by zero
    safe_distances = np.maximum(distances, 1e-6)
    weights = 1.0 / (safe_distances ** power)
    return _normalize_weights(weights)


def _linear_weights(distances: np.ndarray, max_distance_km: float) -> np.ndarray:
    """
    Compute normalized linear-decay distance weights.
    
    Formula: w_i = max(0, 1 - d_i / max_dist), normalized so sum = 1.0
    
    Parameters
    ----------
    distances : np.ndarray
        Array of distances in km.
    max_distance_km : float
        Maximum distance at which weight becomes zero.
    
    Returns
    -------
    np.ndarray
        Normalized weights, same shape as distances.
    """
    weights = np.maximum(0.0, 1.0 - distances / max_distance_km)
    return _normalize_weights(weights)


def _equal_weights(distances: np.ndarray) -> np.ndarray:
    """
    Compute uniform weights (all cells equally weighted).
    
    Parameters
    ----------
    distances : np.ndarray
        Array of distances in km (only used for shape).
    
    Returns
    -------
    np.ndarray
        Uniform weights, same shape as distances.
    """
    n = len(distances)
    if n > 0:
        return np.ones(n) / n
    return np.array([])


# ---------------------------------------------------------------------------
# 1. Point vs. Layer
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Source-type dispatch helpers
# ---------------------------------------------------------------------------

# data_type attribute values that indicate a gridded layer source
LAYER_DATA_TYPES = {
    "scatterometer", "altimeter", "hf_radar", "hf_radar_grid", "radiometer",
    "scatterometer_ssm", "radiometer_ssm",
    "cds_ssm",
    # ERA5 nodes never reach _detect_collocation_type (they lack a
    # "point" dimension and are pulled out by run_collocation's own
    # model_sources scan before that check runs) -- these entries exist
    # purely so _canonical_source_order()'s registered-set check (in
    # visualization.py) includes era5's val_source labels for permanent
    # palette slots. 
    "era5_wind", "era5_waves", "era5_soil_moisture",
    # HYCOM never reaches _detect_collocation_type either (see the ERA5 
    # comment above).
    "hycom",
}
# path-fragment fallbacks when attributes are absent
LAYER_SOURCE_PATHS = {
    "osi_saf_winds", "scatterometer", "altimeter", "hf_radar",
    "hf_radar_grid", "hfr_noaa", "radiometer", "ascat_ssm", "amsr_ssm", "smap_ssm",
    "smos_ssm", "cds_ssm",
}


def _model_source_type(data_type: str) -> Optional[str]:
    """
    Map a gridded "model" validation Dataset's ``data_type`` attribute to
    the ``source_type`` key used in the recipe's ``validation_sources``
    (and therefore in ``source_type_overrides``) -- e.g. ``"era5_wind"``
    -> ``"era5"``, ``"hycom"`` -> ``"hycom"``. Returns ``None`` if
    *data_type* is not a recognized model source.

    ERA5 spans three ``data_type`` values (one per ``variable``: 
    wind/waves/soil_moisture), all sharing the recipe `source_type` 
    literal ``"era5"`` -- hence the prefix match. HYCOM only ever 
    produces ``data_type="hycom"``, an exact match.
    """
    if data_type.startswith("era5_"):
        return "era5"
    if data_type == "hycom":
        return "hycom"
    return None


def _detect_collocation_type(val_ds: "xr.Dataset", source_path: str) -> str:
    """
    Infer the appropriate collocation class name from a validation Dataset.

    Checks (in order):
    1. ``data_type`` attribute      → layer or default
    2. Source path fragment         → layer or default
    """
    data_type = val_ds.attrs.get("data_type", "").lower()

    if data_type in LAYER_DATA_TYPES:
        return "layer_vs_layer"
    for fragment in LAYER_SOURCE_PATHS:
        if fragment in source_path.lower():
            return "layer_vs_layer"
    return "point_vs_layer"


class PointLayerCollocation:
    """
    Match fixed-point (or slowly-moving) validation observations with a
    gridded SAR layer using distance-weighted spatial aggregation.

    Aggregation approach
    --------------------
    For each validation observation:

    1. **SAR Aggregation**: Find all SAR grid cells within `aggregation_window_km`
       (circular radius). Compute a distance-weighted average of SAR variables
       using the selected weighting method (Gaussian, inverse-distance, linear, or equal).

    2. **Validation matching**: Each validation observation is matched as-is
       (no temporal averaging) to any SAR acquisition within
       `time_tolerance_minutes`.

    3. **Output**: Single `CollocatedPoint` per validation observation with
       aggregated SAR mean vs. the raw validation observation.

    Typical use cases: moorings, in-situ buoys.

    Parameters
    ----------
    spatial_tolerance_km : float
        Maximum great-circle distance for pre-filtering (legacy parameter,
        now superseded by `aggregation_window_km`).
    time_tolerance_minutes : int
        Maximum absolute time difference for matching SAR acquisitions.
    interpolation_method : str
        Retained for API compatibility; currently unused (aggregation is fixed).
    aggregation_window_km : float
        Circular radius (km) around each validation point for SAR aggregation.
        Default: 5.0 km.
    validation_temporal_averaging_minutes : int
        Unused by `PointLayerCollocation` (kept for API compatibility and used
        by subclasses such as `LayerLayerCollocation`). Validation observations
        are matched using their own raw value, not averaged.
    distance_weighting : str
        Distance weighting method for SAR aggregation:
        - ``"gaussian"`` — Gaussian kernel (default)
        - ``"inverse_distance"`` — Inverse distance (1/d^2)
        - ``"linear"`` — Linear decay
        - ``"equal"`` — Uniform weights
    gaussian_sigma_km : float
        Standard deviation (km) for Gaussian weighting. Default: 5.0 km.
    dedup_nearest_in_time : bool
        When False (default), every validation observation within
        ``time_tolerance_minutes`` of a SAR acquisition produces its own
        `CollocatedPoint` — used by scatterometer/buoy/mooring sources, 
        where repeated readings at one location are distinct passes/looks 
        worth comparing independently. When True, only the reading closest 
        in time per (station, SAR acquisition) is kept — for sources like 
        ISMN, which report far more densely (hourly) than the SAR revisit 
        time, so a wide ``time_tolerance_minutes`` (needed to tolerate 
        reporting gaps) would otherwise multiply one physical station into 
        dozens of near-duplicate collocations of the same slowly-evolving 
        quantity.
    """

    #: Collocation type label stored on each CollocatedPoint result.
    collocation_type: str = "point_vs_layer"

    def __init__(
        self,
        spatial_tolerance_km: float = 12.5, #spatial tolerance from Abderrahim et al. 2019
        time_tolerance_minutes: int = 30, #buoys vs SAR interval from Abderrahim et al. 2019
        interpolation_method: str = "nearest",
        aggregation_window_km: float = 5.0,
        validation_temporal_averaging_minutes: int = 30,
        distance_weighting: str = "gaussian",
        gaussian_sigma_km: float = 5.0,
        emit_diagnostics: bool = False,
        dedup_nearest_in_time: bool = False,
    ) -> None:
        self.spatial_tolerance_km = spatial_tolerance_km
        self.time_tolerance_minutes = time_tolerance_minutes
        self.interpolation_method = interpolation_method
        self.aggregation_window_km = aggregation_window_km
        self.validation_temporal_averaging_minutes = validation_temporal_averaging_minutes
        self.distance_weighting = distance_weighting
        self.gaussian_sigma_km = gaussian_sigma_km
        self.dedup_nearest_in_time = dedup_nearest_in_time

        if interpolation_method not in ("nearest", "linear", "cubic"):
            raise ValueError(
                f"Unknown interpolation_method '{interpolation_method}'. "
                "Use 'nearest', 'linear', or 'cubic'."
            )
        if distance_weighting not in ("gaussian", "inverse_distance", "linear", "equal"):
            raise ValueError(
                f"Unknown distance_weighting '{distance_weighting}'. "
                "Use 'gaussian', 'inverse_distance', 'linear', or 'equal'."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collocate(
        self,
        sar_data: Dict[str, np.ndarray],
        sar_lon: np.ndarray,
        sar_lat: np.ndarray,
        sar_time: np.ndarray,
        val_data: pd.DataFrame,
        val_source: str,
        sar_scene_name: str = "",
        grid_tree: Optional[Tuple["cKDTree", np.ndarray, int]] = None,
    ) -> List[CollocatedPoint]:
        """
        Match validation observations to the SAR grid using distance-weighted aggregation.

        Algorithm
        ---------
        For each validation observation:

        1. Find all SAR cells within ``aggregation_window_km`` (circular radius).
        2. Compute distance-weighted average of SAR variables using ``distance_weighting`` method.
        3. Use the validation observation's own raw value (no temporal averaging).
        4. Create single `CollocatedPoint` with aggregated SAR vs. raw validation value.

        Parameters
        ----------
        sar_data : dict
            SAR variables as 3-D arrays with shape ``(time, y, x)``.
        sar_lon, sar_lat : np.ndarray
            SAR coordinate grids, shape ``(y, x)``.
        sar_time : array-like
            SAR acquisition times, shape ``(time,)``.
        val_data : pd.DataFrame
            Validation data with columns ``lon``, ``lat``, ``time``, and
            any number of variable columns.
        val_source : str
            Label for the validation source (e.g. ``"buoy"``).
        sar_scene_name : str
            Name of the SAR scene node in the DataTree.
        grid_tree : tuple, optional
            Pre-built ``(tree, flat_idx, n_x)`` from :meth:`_build_grid_tree`
            for this scene's ``(sar_lon, sar_lat)`` grid. ``run_collocation``
            builds this once per SAR scene and passes it to every
            validation source matched against that scene, since rebuilding
            a KD-tree over a large grid (e.g. CLMS Surface Soil Moisture's
            ~28M-cell 1 km Europe grid) per validation file is expensive.
            Built here if omitted (e.g. when calling this method directly).

        Returns
        -------
        list[CollocatedPoint]
            List of collocated matches (one per validation observation).
        """
        from datetime import timedelta as _td

        sar_times = _to_datetime_array(sar_time)
        collocations: List[CollocatedPoint] = []

        # Pre-filters: eliminate validation rows that cannot match
        # Use spatial_tolerance_km for the initial bounding box; dividing
        # by 100 converts km to degrees (conservatively).
        deg_buf = self.spatial_tolerance_km / 100.0
        lon_min = float(np.nanmin(sar_lon)) - deg_buf # sar_lon is in degrees
        lon_max = float(np.nanmax(sar_lon)) + deg_buf
        lat_min = float(np.nanmin(sar_lat)) - deg_buf # sar_lat is in degrees
        lat_max = float(np.nanmax(sar_lat)) + deg_buf

        spatial_mask = (
            (val_data["lon"] >= lon_min) & (val_data["lon"] <= lon_max) &
            (val_data["lat"] >= lat_min) & (val_data["lat"] <= lat_max)
        )
        val_data_filtered = val_data[spatial_mask].copy()

        if val_data_filtered.empty:
            logger.debug("No validation data within spatial bounds")
            return collocations

        # Temporal pre-filter
        t_min = min(sar_times) - _td(minutes=self.time_tolerance_minutes)
        t_max = max(sar_times) + _td(minutes=self.time_tolerance_minutes)
        if hasattr(t_min, "tzinfo") and t_min.tzinfo is not None:
            t_min = t_min.replace(tzinfo=None)
            t_max = t_max.replace(tzinfo=None)

        val_times_pd = pd.to_datetime(val_data_filtered["time"].values)
        if val_times_pd.tz is not None:
            val_times_pd = val_times_pd.tz_localize(None)

        temporal_mask = (val_times_pd >= t_min) & (val_times_pd <= t_max)
        val_data_filtered = val_data_filtered[temporal_mask]

        if val_data_filtered.empty:
            logger.debug("No validation data within temporal window")
            return collocations

        logger.debug(
            "Pre-filters kept %d validation rows (spatial + temporal)",
            len(val_data_filtered),
        )

        # Identify numeric columns (for aggregation)
        numeric_cols = [
            col for col in val_data_filtered.columns
            if col not in {"lon", "lat", "time", "platform_id", "platform_type"} and
            pd.api.types.is_numeric_dtype(val_data_filtered[col])
        ]

        # Time-sorted, per-platform groups used to forward-fill a NaN
        # reading: when a validation observation is missing a value for
        # some column, look at THAT SAME PLATFORM's later observations
        # (still within ``time_tolerance_minutes`` of the original
        # reading) for the next one that has a valid value -- e.g. a
        # mooring's own wind sensor had a brief gap, use its own next
        # reading. Scoped to platform_id (when the column exists) rather
        # than searched across the whole spatial+temporal window: without
        # this, a validation point whose own platform never reports a
        # given variable at all (e.g. a tidal gauge, which only measures
        # water level, never wind) would silently borrow a DIFFERENT
        # nearby platform's own reading of that variable just because it
        # happens to fall within the same time tolerance -- corrupting
        # the collocation with a real value from the wrong instrument
        # rather than correctly leaving it unmatched.
        has_platform_id = "platform_id" in val_data_filtered.columns
        if has_platform_id:
            _platform_groups: Dict[Any, pd.DataFrame] = {
                platform_id: group.sort_values("time").reset_index(drop=True)
                for platform_id, group in val_data_filtered.groupby("platform_id", sort=False)
            }
        else:
            _platform_groups = {None: val_data_filtered.sort_values("time").reset_index(drop=True)}

        _group_times: Dict[Any, np.ndarray] = {}
        _next_valid_idx: Dict[Tuple[Any, str], np.ndarray] = {}

        def _next_valid_value(col: str, after_time, platform_id: Any = None) -> Optional[float]:
            group = _platform_groups.get(platform_id)
            if group is None or group.empty or col not in group.columns:
                return None
            if platform_id not in _group_times:
                group_times = pd.to_datetime(group["time"].values)
                if group_times.tz is not None:
                    group_times = group_times.tz_localize(None)
                _group_times[platform_id] = group_times.values.astype("datetime64[ns]")
            group_times_ns = _group_times[platform_id]
            n_group = len(group)
            key = (platform_id, col)
            if key not in _next_valid_idx:
                valid = group[col].notna().values
                idx = np.where(valid, np.arange(n_group), n_group)
                _next_valid_idx[key] = np.minimum.accumulate(idx[::-1])[::-1]
            after_ns = np.datetime64(pd.Timestamp(after_time))
            pos = int(np.searchsorted(group_times_ns, after_ns, side="right"))
            if pos >= n_group:
                return None
            j = int(_next_valid_idx[key][pos])
            if j >= n_group:
                return None
            gap_min = (group_times_ns[j] - after_ns) / np.timedelta64(1, "m")
            if gap_min > self.time_tolerance_minutes:
                return None
            return float(group[col].values[j])

        # KD-tree over the SAR grid cells (unit-sphere Cartesian coordinates,
        # see _lonlat_to_unit_xyz): each validation point then queries only
        # its local neighbourhood instead of computing a Haversine distance
        # to every grid cell. Reuse the caller's pre-built tree when given
        # (see the `grid_tree` parameter); only build one here as a fallback
        # for direct calls.
        if grid_tree is None:
            grid_tree = self._build_grid_tree(sar_lon, sar_lat)

        # dedup_nearest_in_time (see class docstring): keep only the reading
        # closest in time per (station, SAR time), keyed by platform_id when
        # available, else by the validation point's own coordinates.
        best_matches: Dict[Tuple[Any, int], "CollocatedPoint"] = {}

        # Process each validation observation
        for idx, val_row in val_data_filtered.iterrows():
            v_lon = float(val_row["lon"])
            v_lat = float(val_row["lat"])
            v_time = _to_datetime_array([val_row["time"]])[0]
            v_platform_id = val_row["platform_id"] if has_platform_id else None

            # Find nearby SAR cells within aggregation window
            nearby_cells_with_dist = self._nearby_cells_with_distances(
                v_lon, v_lat, sar_lon, sar_lat, self.aggregation_window_km,
                grid_tree=grid_tree,
            )

            if not nearby_cells_with_dist:
                logger.debug(
                    "No SAR cells within %.1f km of validation point (%.2f, %.2f)",
                    self.aggregation_window_km, v_lon, v_lat
                )
                continue

            # Find nearby SAR times
            nearby_t_idx = self._nearby_times(v_time, sar_times)
            if not nearby_t_idx:
                logger.debug(
                    "No SAR times within %d minutes of validation time",
                    self.time_tolerance_minutes
                )
                continue

            # Process each nearby SAR time
            for t_idx in nearby_t_idx:
                # Compute aggregated SAR values
                sar_aggregated = self._compute_aggregated_sar_value(
                    nearby_cells_with_dist,
                    sar_data,
                    t_idx,
                    weighting_method=self.distance_weighting,
                    sigma_km=self.gaussian_sigma_km,
                    agg_window_km=self.aggregation_window_km,
                )

                if not sar_aggregated:
                    logger.debug("No valid SAR values at t_idx=%d", t_idx)
                    continue

                # Use the validation observation's own raw value (nearest in
                # time by construction, since it is matched as-is rather than
                # averaged with neighboring observations). If a column is NaN
                # for this exact reading, forward-fill it from the next
                # observation still within the temporal tolerance window
                # (e.g. a mooring sensor gap) instead of leaving it missing.
                val_aggregated = {}
                for col in numeric_cols:
                    raw_val = val_row[col]
                    if pd.notna(raw_val):
                        val_aggregated[col] = float(raw_val)
                        continue
                    filled_val = _next_valid_value(col, v_time, v_platform_id)
                    if filled_val is not None:
                        val_aggregated[col] = filled_val

                if not val_aggregated:
                    logger.debug("No valid validation values")
                    continue

                # Compute SAR position (center of aggregation window or closest cell)
                closest_idx = np.argmin([d for _, _, d in nearby_cells_with_dist])
                y_idx, x_idx, _ = nearby_cells_with_dist[closest_idx]
                s_lon = float(sar_lon[y_idx, x_idx])
                s_lat = float(sar_lat[y_idx, x_idx])
                s_time = sar_times[t_idx]

                # Compute distances from validation point to closest SAR cell
                spatial_dist = _haversine_distance(v_lon, v_lat, s_lon, s_lat)
                temporal_dist = abs((v_time - s_time).total_seconds() / 60.0)

                # =========== RVL Projection ===========
                # Project validation currents to radial velocity if applicable
                if (
                    "rvlRadVel" in sar_aggregated
                    and "rvlHeading" in sar_data
                    and "EWCT" in val_aggregated
                    and "NSCT" in val_aggregated
                ):
                    try:
                        # Get average rvlHeading value
                        nearby_headings = np.array([
                            sar_data["rvlHeading"][t_idx, y, x]
                            for y, x, _ in nearby_cells_with_dist
                        ])
                        valid_headings = nearby_headings[~np.isnan(nearby_headings)]
                        if len(valid_headings) > 0:
                            val_aggregated["rvlRadVel_projection"] = _project_currents_to_radial(
                                float(val_aggregated["EWCT"]),
                                float(val_aggregated["NSCT"]),
                                float(np.nanmean(valid_headings)),
                            )
                    except (KeyError, ValueError, TypeError) as e:
                        logger.debug("RVL projection failed: %s", e)

                # Prefer the observation's own platform type (e.g. a combined
                # in-situ CSV mixing moorings and buoys) over the single
                # val_source passed in for the whole pass, when available.
                row_platform_type = val_row.get("platform_type")
                point_val_source = (
                    row_platform_type
                    if isinstance(row_platform_type, str) and row_platform_type
                    else val_source
                )

                # Create CollocatedPoint. When dedup_nearest_in_time is set,
                # keep only the closest-in-time match per (station, SAR
                # time) — see best_matches above.
                point = CollocatedPoint(
                    sar_lon=s_lon,
                    sar_lat=s_lat,
                    sar_time=s_time,
                    sar_data=sar_aggregated,
                    val_lon=v_lon,
                    val_lat=v_lat,
                    val_time=v_time,
                    val_data=val_aggregated,
                    spatial_distance_km=spatial_dist,
                    temporal_distance_minutes=temporal_dist,
                    val_source=point_val_source,
                    val_id=val_row.get("platform_id"),
                    collocation_type=self.collocation_type,
                    sar_y_idx=y_idx,
                    sar_x_idx=x_idx,
                    sar_scene_name=sar_scene_name,
                )

                if not self.dedup_nearest_in_time:
                    collocations.append(point)
                    continue

                station_key = point.val_id if pd.notna(point.val_id) else (v_lon, v_lat)
                dedup_key = (station_key, t_idx)
                existing = best_matches.get(dedup_key)
                if existing is None or temporal_dist < existing.temporal_distance_minutes:
                    best_matches[dedup_key] = point

        if self.dedup_nearest_in_time:
            collocations = list(best_matches.values())

        logger.info(
            "%s: found %d matches from %d validation observations (source=%s)",
            self.__class__.__name__, len(collocations), len(val_data_filtered), val_source,
        )
        return collocations

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _nearby_cells(
        self,
        lon: float, lat: float,
        grid_lon: np.ndarray, grid_lat: np.ndarray,
    ) -> List[Tuple[int, int]]:
        distances = _haversine_distance_grid(lon, lat, grid_lon, grid_lat)
        ys, xs = np.where(distances <= self.spatial_tolerance_km)
        return list(zip(ys.tolist(), xs.tolist()))

    def _nearby_times(
        self,
        target: datetime,
        time_array: np.ndarray,
    ) -> List[int]:
        times_ns = pd.to_datetime(list(time_array))
        if times_ns.tz is not None:
            times_ns = times_ns.tz_localize(None)
        target_ns = np.datetime64(pd.Timestamp(target))
        distances = np.abs(
            (times_ns.values.astype("datetime64[ns]") - target_ns)
            / np.timedelta64(1, "m")
        ).astype(float)
        return np.where(distances <= self.time_tolerance_minutes)[0].tolist()

    @staticmethod
    def _build_grid_tree(
        grid_lon: np.ndarray, grid_lat: np.ndarray,
    ) -> Tuple["cKDTree", np.ndarray, int]:
        """
        Build a KD-tree over the finite cells of a 2-D coordinate grid.

        Returns
        -------
        (tree, flat_idx, n_x)
            ``tree`` is a cKDTree over unit-sphere Cartesian coordinates,
            ``flat_idx`` maps tree indices back to flat grid indices (NaN
            cells are excluded), ``n_x`` is the grid's x-dimension size.
        """
        from scipy.spatial import cKDTree

        lon_flat = grid_lon.ravel()
        lat_flat = grid_lat.ravel()
        flat_idx = np.flatnonzero(np.isfinite(lon_flat) & np.isfinite(lat_flat))
        tree = cKDTree(_lonlat_to_unit_xyz(lon_flat[flat_idx], lat_flat[flat_idx]))
        return tree, flat_idx, grid_lon.shape[1]

    def _nearby_cells_with_distances(
        self,
        lon: float, lat: float,
        grid_lon: np.ndarray, grid_lat: np.ndarray,
        max_distance_km: float,
        grid_tree: Optional[Tuple["cKDTree", np.ndarray, int]] = None,
    ) -> List[Tuple[int, int, float]]:
        """
        Find SAR cells within max_distance_km and return with distances.

        When *grid_tree* (from :meth:`_build_grid_tree`) is given, only the
        KD-tree neighbourhood of the point is examined; otherwise a Haversine
        distance is computed to every grid cell.

        Returns
        -------
        list of (y_idx, x_idx, distance_km) tuples
        """
        if grid_tree is not None:
            tree, flat_idx, n_x = grid_tree
            R = 6371.0
            # Chord length equivalent of the great-circle search radius: a
            # point is within max_distance_km along the sphere iff its chord
            # distance is within this radius, so the ball query is exact.
            chord_radius = 2.0 * np.sin(max_distance_km / (2.0 * R))
            cand = tree.query_ball_point(
                _lonlat_to_unit_xyz(np.array([lon]), np.array([lat]))[0],
                r=chord_radius,
            )
            if not cand:
                return []
            # Ascending flat order matches the row-major order np.where
            # produced before, keeping argmin tie-breaks identical.
            sel = np.sort(flat_idx[np.asarray(cand)])
            dists = _haversine_distance_grid(
                lon, lat, grid_lon.ravel()[sel], grid_lat.ravel()[sel]
            )
            keep = dists <= max_distance_km
            return [
                (int(f) // n_x, int(f) % n_x, float(d))
                for f, d in zip(sel[keep], dists[keep])
            ]

        distances = _haversine_distance_grid(lon, lat, grid_lon, grid_lat)
        ys, xs = np.where(distances <= max_distance_km)
        result = [(y, x, distances[y, x]) for y, x in zip(ys.tolist(), xs.tolist())]
        return result

    def _compute_aggregated_sar_value(
        self,
        nearby_cells_with_dist: List[Tuple[int, int, float]],
        sar_data: Dict[str, np.ndarray],
        t_idx: int,
        weighting_method: str = "gaussian",
        sigma_km: float = 2.0,
        agg_window_km: float = 5.0,
    ) -> Dict[str, float]:
        """
        Compute distance-weighted average of SAR values over nearby cells.
        
        Parameters
        ----------
        nearby_cells_with_dist : list of (y, x, distance_km)
            Cells and their distances from the validation point.
        sar_data : dict
            SAR variables as 3-D arrays with shape (time, y, x).
        t_idx : int
            Time index to use.
        weighting_method : str
            "gaussian", "inverse_distance", "linear", or "equal".
        sigma_km : float
            Gaussian sigma or threshold for linear weighting.
        agg_window_km : float
            Aggregation window radius (used for linear weighting).
        
        Returns
        -------
        dict
            {var_name: aggregated_value} with aggregated SAR values.
        """
        if not nearby_cells_with_dist:
            return {}

        # Extract distances and cell indices
        cell_indices = [(y, x) for y, x, _ in nearby_cells_with_dist]
        distances_arr = np.array([d for _, _, d in nearby_cells_with_dist])

        # Compute weights
        if weighting_method == "gaussian":
            weights = _gaussian_weights(distances_arr, sigma_km)
        elif weighting_method == "inverse_distance":
            weights = _inverse_distance_weights(distances_arr, power=2.0)
        elif weighting_method == "linear":
            weights = _linear_weights(distances_arr, agg_window_km)
        elif weighting_method == "equal":
            weights = _equal_weights(distances_arr)
        else:
            logger.warning(f"Unknown weighting method '{weighting_method}', using equal weights")
            weights = _equal_weights(distances_arr)

        # Compute weighted average for each variable
        result = {}
        for var_name, var_arr in sar_data.items():
            values = np.array([var_arr[t_idx, y, x] for y, x in cell_indices])
            valid_mask = ~np.isnan(values)

            if not np.any(valid_mask):
                # All NaN, skip this variable
                continue

            # Use only valid values and renormalize weights
            valid_values = values[valid_mask]
            valid_weights = weights[valid_mask]
            valid_weights = valid_weights / np.sum(valid_weights)  # renormalize

            aggregated = np.sum(valid_values * valid_weights)
            if not np.isnan(aggregated):
                result[var_name] = float(aggregated)

        return result


# ---------------------------------------------------------------------------
# High-level pipeline helper (step 3)
# ---------------------------------------------------------------------------

def _merge_collocation_kwargs(
    global_kwargs: Dict[str, Any],
    per_source_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge global collocation settings with per-source overrides.

    Parameters
    ----------
    global_kwargs : dict
        Global collocation parameters from recipe.config.collocation.
    per_source_kwargs : dict
        Per-source overrides from validation_sources[i].collocation_kwargs.

    Returns
    -------
    dict
        Merged kwargs with per_source values overriding globals.
    """
    merged = dict(global_kwargs)
    merged.update(per_source_kwargs)
    return merged


def _resolve_layer_type(val_ds: "xr.Dataset", val_name: str, layer_vs_layer_specs: dict) -> str:
    """
    Resolve the layer_vs_layer_specs lookup key for one validation source.

    Starts from the dataset's ``data_type`` attribute (set by the
    converter), falling back to path-fragment inference when that attribute
    is absent. Refines ambiguous data_types into their more specific
    per-variant keys:

    - "altimeter" -> "altimeter_<frequency>" (1 Hz vs 5 Hz have very
      different along-track spacing).
    - "radiometer" -> "radiometer_<sensor>" when that sensor has its own
      spec (all RSS sensors share the same 0.25 deg grid by default, but
      stay individually tunable).
    - "scatterometer" -> "scatterometer_hy2b"/"_hy2c"/"_oceansat3" when the
      node's path names one of those FTP-sourced satellites — otherwise
      ASCAT/EUMDAC's plain "scatterometer" (12.5 km) applies. This
      distinction cannot come from ``data_type`` alone: every scatterometer
      file (ASCAT or HY-2/Oceansat-3) is stamped with the same
      ``data_type="scatterometer"`` by ``from_scatterometer_nc``.
    - "radiometer_ssm" -> "<sensor>_ssm" (e.g. "amsr_ssm"/"smap_ssm"/
      "smos_ssm") when that sensor has its own spec -- AMSR-E/2, SMAP, and
      SMOS soil moisture all share the generic ``data_type="radiometer_ssm"``
      at conversion time; sensor-specific refinement happens here.
    """
    layer_type = val_ds.attrs.get("data_type", "").lower()
    path_parts = val_name.lower().split("/")

    if not layer_type:
        if "scatterometer" in path_parts:
            layer_type = "scatterometer"
        elif "osi_saf_winds" in path_parts or "winds" in path_parts:
            layer_type = "scatterometer"
        elif "altimeter" in path_parts:
            layer_type = "altimeter"
        elif "radiometer" in path_parts:
            layer_type = "radiometer"
        elif "hf_radar" in path_parts:
            layer_type = "hf_radar"
        elif "hfr_noaa" in path_parts or "hf_radar_grid" in path_parts:
            layer_type = "hf_radar_grid"
        elif "amsr_ssm" in path_parts:
            layer_type = "amsr_ssm"
        elif "smap_ssm" in path_parts:
            layer_type = "smap_ssm"
        elif "smos_ssm" in path_parts:
            layer_type = "smos_ssm"

    if layer_type == "scatterometer":
        for variant in ("scatterometer_hy2b", "scatterometer_hy2c", "scatterometer_oceansat3"):
            if variant in path_parts:
                layer_type = variant
                break
    elif layer_type == "altimeter":
        freq = val_ds.attrs.get("frequency", "1hz").lower()
        layer_type = f"altimeter_{freq}"
    elif layer_type == "radiometer":
        sensor = val_ds.attrs.get("sensor", "").lower()
        if sensor and f"radiometer_{sensor}" in layer_vs_layer_specs:
            layer_type = f"radiometer_{sensor}"
    elif layer_type == "radiometer_ssm":
        # AMSR-E/2 and SMAP soil moisture share data_type="radiometer_ssm";
        # refine to the sensor-specific spec key (amsr_ssm / smap_ssm),
        # mirroring the radiometer_<sensor> refinement above.
        sensor = val_ds.attrs.get("sensor", "").lower()
        sensor_key = f"{sensor}_ssm"
        if sensor and sensor_key in layer_vs_layer_specs:
            layer_type = sensor_key

    return layer_type


def _apply_hf_radar_resolution_override(
    layer_type: str, val_ds: "xr.Dataset", merged_kwargs: dict, recipe_layer_type_specs: dict,
) -> None:
    """
    Override merged_kwargs["aggregation_window_km"] with the node's own
    hfr_resolution_km (stamped by from_hf_radar_grid from the file's actual
    lat/lon spacing) -- unless the recipe explicitly set
    aggregation_window_km for this layer_type itself, which always wins.
    A no-op for any layer_type other than "hf_radar_grid", or when the
    node has no hfr_resolution_km attr (e.g. it was not computable).
    """
    if layer_type != "hf_radar_grid":
        return
    hfr_resolution_km = val_ds.attrs.get("hfr_resolution_km")
    if hfr_resolution_km is None:
        return
    if "aggregation_window_km" in recipe_layer_type_specs.get(layer_type, {}):
        return
    merged_kwargs["aggregation_window_km"] = hfr_resolution_km


def _apply_ascat_resolution_override(
    layer_type: str, val_ds: "xr.Dataset", merged_kwargs: dict, recipe_layer_type_specs: dict,
) -> None:
    """
    Override merged_kwargs["aggregation_window_km"] with the node's own
    ascat_resolution_km (stamped by from_ascat_ssm/from_hsaf_ssm from the
    file's own filename -- H-SAF's H29 is 12.5km, H122 is 6.25km, and
    EUMDAC/SOMO12 filenames never encode a resolution so they fall back to
    12.5km) -- unless the recipe explicitly set aggregation_window_km for
    this layer_type itself, which always wins. A no-op for any layer_type
    other than "scatterometer_ssm", or when the node has no
    ascat_resolution_km attr. Structurally identical to
    _apply_hf_radar_resolution_override.
    """
    if layer_type != "scatterometer_ssm":
        return
    ascat_resolution_km = val_ds.attrs.get("ascat_resolution_km")
    if ascat_resolution_km is None:
        return
    if "aggregation_window_km" in recipe_layer_type_specs.get(layer_type, {}):
        return
    merged_kwargs["aggregation_window_km"] = ascat_resolution_km


#: layer_vs_layer types that receive pre-collocation ±time_tolerance
#: temporal averaging (soil-moisture satellite sources) instead of the
#: default "keep every reading" behaviour -- see run_collocation and
#: _average_within_sar_tolerance below.
SSM_SATELLITE_LAYER_TYPES = {
    "scatterometer_ssm", "radiometer_ssm", "amsr_ssm", "smap_ssm", "smos_ssm",
    "cds_ssm",
}


def _snap_to_grid(values: np.ndarray, step_deg: float) -> np.ndarray:
    """
    Round coordinates to the nearest multiple of ``step_deg``.

    Gives repeated readings of "the same" underlying satellite grid cell an
    identical grouping key even when the source reports lon/lat as a
    continuously-varying swath (e.g. AMSR2's AU_Land half-orbit format)
    rather than a fixed grid -- a no-op for sources already on a fixed
    grid, since repeated cells already share identical lon/lat.
    """
    return np.round(values / step_deg) * step_deg


def _average_within_sar_tolerance(
    val_df: pd.DataFrame,
    sar_times: List[datetime],
    group_cols: List[str],
    time_tolerance_minutes: float,
) -> pd.DataFrame:
    """
    Collapse *val_df* to one row per (group_cols..., matched SAR time) by
    averaging every numeric column across all readings within
    ``time_tolerance_minutes`` of that SAR time.

    Each reading is assigned to its single *nearest* SAR time (via
    ``pd.merge_asof(direction="nearest")``), not every SAR time within
    tolerance -- correct as long as consecutive SAR times are spaced more
    than ``2 * time_tolerance_minutes`` apart, true for soil moisture's
    daily scenes with a 12h tolerance. Readings outside tolerance of every
    SAR time are dropped. The output's ``time`` column is set to the matched 
    SAR time, not the mean of the grouped readings' own times, so downstream
    temporal-distance calculations see ~0 for these pre-averaged rows.

    Parameters
    ----------
    val_df : pd.DataFrame
        Must have ``lon``, ``lat``, ``time`` columns plus any number of
        variable columns. May also have ``platform_id``/``platform_type``.
    sar_times : list[datetime]
        Every SAR scene's acquisition time in this collocation run.
    group_cols : list[str]
        Column(s) identifying "the same physical location" -- e.g.
        ``["platform_id"]`` for ISMN stations (or ``["lon", "lat"]`` when
        no platform_id column exists), or ``["_snap_lon", "_snap_lat"]``
        for gridded satellite sources (see ``_snap_to_grid``).
    time_tolerance_minutes : float
        The collocation pass's own time tolerance.

    Returns
    -------
    pd.DataFrame
        One row per (group_cols..., matched SAR time), with numeric
        columns averaged and ``lon``/``lat`` set to the group's mean
        coordinate (unchanged if ``lon``/``lat`` are themselves the
        group_cols).
    """
    if val_df.empty or not sar_times:
        return val_df.iloc[0:0]

    working = val_df.copy()
    working["_val_time_ns"] = pd.to_datetime(working["time"]).values.astype("datetime64[ns]")
    # Validation-side timestamps can legitimately be NaT; merge_asof requires
    # non-null keys, so drop them here too, matching the SAR-side handling.
    working = working[working["_val_time_ns"].notna()].reset_index(drop=True)
    if working.empty:
        return val_df.iloc[0:0]
    working = working.sort_values("_val_time_ns").reset_index(drop=True)

    sar_times_ns = pd.to_datetime(sorted(sar_times)).values.astype("datetime64[ns]")
    sar_df = pd.DataFrame({"_sar_time_ns": sar_times_ns})

    merged = pd.merge_asof(
        working, sar_df,
        left_on="_val_time_ns", right_on="_sar_time_ns",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=time_tolerance_minutes),
    )
    merged = merged.dropna(subset=["_sar_time_ns"]).reset_index(drop=True)
    if merged.empty:
        return merged.drop(columns=["_val_time_ns", "_sar_time_ns"])

    numeric_cols = [
        col for col in merged.columns
        if col not in {"lon", "lat", "time", "platform_id", "platform_type",
                       "_val_time_ns", "_sar_time_ns", *group_cols}
        and pd.api.types.is_numeric_dtype(merged[col])
    ]

    agg: Dict[str, Any] = {col: "mean" for col in numeric_cols}
    if "lon" not in group_cols:
        agg["lon"] = "mean"
    if "lat" not in group_cols:
        agg["lat"] = "mean"
    if "platform_id" in merged.columns and "platform_id" not in group_cols:
        agg["platform_id"] = "first"
    if "platform_type" in merged.columns and "platform_type" not in group_cols:
        agg["platform_type"] = "first"

    grouped = merged.groupby(group_cols + ["_sar_time_ns"], as_index=False).agg(agg)
    grouped["time"] = grouped["_sar_time_ns"]
    return grouped.drop(columns=["_sar_time_ns"])


def _merge_sibling_ssm_nodes(
    buckets: Dict[str, Dict[str, Any]],
    source_metadata: Dict[str, Dict[str, Any]],
    layer_vs_layer_specs: dict,
) -> None:
    """
    Collapse sibling per-file SSM satellite nodes sharing the same source
    into a single logical node before averaging or collocation.

    Each downloaded file becomes its own datatree node, so e.g. an
    ascending and a descending overpass end up as two separate nodes. Left
    unmerged, they would be averaged independently instead of blended within
    one time-tolerance window, defeating the purpose of overpass blending.
    Mutates the input buckets and metadata in place, keeping one
    representative node per source.
    """
    sources = buckets.get("layer_vs_layer")
    if not sources:
        return

    groups: Dict[str, List[str]] = {}
    for val_name, val_ds in sources.items():
        layer_type = _resolve_layer_type(val_ds, val_name, layer_vs_layer_specs)
        if layer_type not in SSM_SATELLITE_LAYER_TYPES:
            continue
        source_type = source_metadata.get(val_name, {}).get("source_type", val_name)
        groups.setdefault(source_type, []).append(val_name)

    for val_names in groups.values():
        if len(val_names) < 2:
            continue
        ordered = sorted(val_names)
        representative = ordered[0]
        merged_df = pd.concat(
            [sources[name].to_dataframe().reset_index(drop=True) for name in ordered],
            ignore_index=True,
        )
        source_metadata[representative]["_merged_raw_df"] = merged_df
        for name in ordered[1:]:
            del sources[name]
            source_metadata.pop(name, None)


def _distance_weights(
    distances_km: np.ndarray,
    method: str,
    sigma_km: float,
    radius_km: float,
) -> np.ndarray:
    """Dispatch to the same weighting kernels used for SAR aggregation."""
    if method == "gaussian":
        return _gaussian_weights(distances_km, sigma_km)
    if method == "inverse_distance":
        return _inverse_distance_weights(distances_km, power=2.0)
    if method == "linear":
        return _linear_weights(distances_km, radius_km)
    return _equal_weights(distances_km)


def _collocate_wv_points(
    sar_lons: np.ndarray,
    sar_lats: np.ndarray,
    sar_times: np.ndarray,
    sar_point_vars: Dict[str, np.ndarray],
    val_data: pd.DataFrame,
    val_source: str,
    footprint_radius_km: float,
    time_tolerance_minutes: float,
    distance_weighting: str,
    gaussian_sigma_km: float,
    collocation_type: str,
    sar_scene_name: str = "",
) -> List[CollocatedPoint]:
    """
    SAR-point-anchored collocation for sparse WV-mode OSW vignettes.

    Each Sentinel-1 WV vignette is a single point representing a ~20×20 km
    footprint, and consecutive vignettes are ~200 km apart. Rather than
    requiring validation data within a few km of the vignette *centre* (as the
    grid-oriented matchers do when a WV point is faked into a 1×1 grid), this
    gathers every validation observation within ``footprint_radius_km`` and
    ``time_tolerance_minutes`` of each vignette and aggregates them into a
    single match anchored on the SAR point.

    Parameters
    ----------
    sar_lons, sar_lats, sar_times : np.ndarray
        Per-vignette coordinates/times for one SAR scene, shape ``(n_points,)``.
    sar_point_vars : dict
        SAR variables as ``(n_points,)`` arrays (e.g. ``{"oswHs": ...}``).
    val_data : pd.DataFrame
        Validation observations with ``lon``, ``lat``, ``time`` and any number
        of variable columns.
    footprint_radius_km : float
        Search radius around each vignette (≈ footprint half-diagonal).
    time_tolerance_minutes : float
        Maximum absolute time difference for a validation obs to contribute.
    distance_weighting : str
        Aggregation weighting: ``"equal"`` (plain average, used for in-situ),
        ``"gaussian"``, ``"inverse_distance"`` or ``"linear"``.
    collocation_type : str
        Label stored on each result (``"point_vs_point"`` for in-situ,
        ``"point_vs_layer"`` for altimeter/scatterometer layers).

    Returns
    -------
    list[CollocatedPoint]
        One match per vignette that had at least one contributing observation.
    """
    from scipy.spatial import cKDTree

    collocations: List[CollocatedPoint] = []
    if val_data.empty:
        return collocations

    numeric_cols = [
        col for col in val_data.columns
        if col not in {"lon", "lat", "time", "platform_id", "platform_type"} and
        pd.api.types.is_numeric_dtype(val_data[col])
    ]
    if not numeric_cols:
        return collocations

    val_lons = val_data["lon"].values.astype(float)
    val_lats = val_data["lat"].values.astype(float)
    val_times_pd = pd.to_datetime(val_data["time"].values)
    if val_times_pd.tz is not None:
        val_times_pd = val_times_pd.tz_localize(None)
    val_times_np = val_times_pd.values.astype("datetime64[ns]")
    has_platform_type = "platform_type" in val_data.columns
    has_platform_id = "platform_id" in val_data.columns

    # KD-tree over validation points; query_ball_point with the chord-length
    # equivalent of footprint_radius_km returns every obs inside the footprint.
    R = 6371.0
    tree = cKDTree(_lonlat_to_unit_xyz(val_lons, val_lats))
    chord_radius = 2.0 * np.sin(footprint_radius_km / (2.0 * R))

    for i in range(len(sar_lons)):
        s_lon = float(sar_lons[i])
        s_lat = float(sar_lats[i])
        if not (np.isfinite(s_lon) and np.isfinite(s_lat)):
            continue

        # SAR variables for this vignette (skip if all NaN)
        sar_aggregated = {
            var: float(arr[i]) for var, arr in sar_point_vars.items()
            if np.isfinite(arr[i])
        }
        if not sar_aggregated:
            continue

        idx = tree.query_ball_point(
            _lonlat_to_unit_xyz(np.array([s_lon]), np.array([s_lat]))[0],
            r=chord_radius,
        )
        if not idx:
            continue
        idx = np.asarray(idx)

        s_time_np = np.datetime64(pd.Timestamp(sar_times[i]))
        dt_min = np.abs(
            (val_times_np[idx] - s_time_np) / np.timedelta64(1, "m")
        ).astype(float)
        tmask = dt_min <= time_tolerance_minutes
        if not np.any(tmask):
            continue
        idx = idx[tmask]
        dt_min = dt_min[tmask]

        dists = _haversine_distance_grid(s_lon, s_lat, val_lons[idx], val_lats[idx])
        weights = _distance_weights(
            dists, distance_weighting, gaussian_sigma_km, footprint_radius_km
        )

        val_aggregated: Dict[str, float] = {}
        for col in numeric_cols:
            vals = val_data[col].values[idx].astype(float)
            valid = np.isfinite(vals)
            if not np.any(valid):
                continue
            w = weights[valid]
            wsum = float(np.sum(w))
            agg = float(np.mean(vals[valid])) if wsum == 0 else float(np.sum(vals[valid] * w) / wsum)
            if np.isfinite(agg):
                val_aggregated[col] = agg
        if not val_aggregated:
            continue

        # Project the in-situ current vector (EWCT/NSCT) onto the SAR radial
        # look direction so it can be compared against rvlRadVel — mirrors the
        # grid collocation path. Here, rvlHeading is a scalar per vignette point.
        if (
            "rvlRadVel" in sar_aggregated
            and "rvlHeading" in sar_aggregated
            and "EWCT" in val_aggregated
            and "NSCT" in val_aggregated
        ):
            val_aggregated["rvlRadVel_projection"] = _project_currents_to_radial(
                float(val_aggregated["EWCT"]),
                float(val_aggregated["NSCT"]),
                float(sar_aggregated["rvlHeading"]),
            )

        nearest = int(np.argmin(dists))
        near_row = idx[nearest]
        point_val_source = val_source
        if has_platform_type:
            pt = val_data["platform_type"].values[near_row]
            if isinstance(pt, str) and pt:
                point_val_source = pt

        collocations.append(
            CollocatedPoint(
                sar_lon=s_lon,
                sar_lat=s_lat,
                sar_time=pd.Timestamp(sar_times[i]).to_pydatetime(),
                sar_data=sar_aggregated,
                val_lon=float(val_lons[near_row]),
                val_lat=float(val_lats[near_row]),
                val_time=_to_datetime_array([val_data["time"].values[near_row]])[0],
                val_data=val_aggregated,
                spatial_distance_km=float(dists[nearest]),
                temporal_distance_minutes=float(np.min(dt_min)),
                val_source=point_val_source,
                val_id=(val_data["platform_id"].values[near_row] if has_platform_id else None),
                collocation_type=collocation_type,
                sar_y_idx=0,
                sar_x_idx=i,
                sar_scene_name=sar_scene_name,
            )
        )

    logger.info(
        "WV-point collocation [%s]: %d match(es) from %d vignette(s) (source=%s)",
        collocation_type, len(collocations), len(sar_lons), val_source,
    )
    return collocations


def run_collocation(
    recipe,
    datatree: "xr.DataTree",
    base_dir: Union[str, Path],
    emit_diagnostics: bool = False,
    layer_vs_layer_collocation_method: str = "cell-averaging",
    filename_suffix: str = "",
) -> Optional["xr.Dataset"]:
    """
    Collocate SAR and validation nodes in *datatree* and save the combined
    results to ``<base_dir>/collocation_results<filename_suffix>.nc``.

    Each validation source is auto-assigned to one of the four collocation
    types described in this module's docstring (point_vs_point,
    point_vs_layer, layer_vs_layer, model_vs_layer), based on its
    ``platform_type`` / ``data_type`` attribute and whether the matching SAR
    node is gridded or WV-mode (see :func:`_detect_collocation_type`).

    The collocation parameters (tolerances, interpolation method) are taken
    from ``recipe.config.collocation`` with per-source overrides applied
    from ``validation_sources[i].collocation_kwargs``.

    DataTree layout expected
    ------------------------
    - ``/sar/<scene-name>``              — one node per downloaded SAR file
    - ``/validation/<source-name>``      — point-observation Datasets
    (may be nested one level deeper for satellite products)

    Each SAR node must have ``lon`` and ``lat`` as 2-D ``(y, x)`` coordinates
    and a scalar ``time`` coordinate.  Each validation node must use a flat
    ``point`` dimension with ``lon``, ``lat``, and ``time`` coordinates.

    Parameters
    ----------
    recipe : Recipe
        Recipe object; its ``config.collocation`` field provides tolerances
        and the interpolation method.
    datatree : xr.DataTree
        DataTree produced by :func:`DataTreeConverter.convert_downloaded_data`.
    base_dir : str or Path
        Directory where ``collocation_results.nc`` will be written.
    emit_diagnostics : bool
        If True, emit detailed diagnostic logging for scatterometer collocation
        and generate diagnostic visualization plots.
    layer_vs_layer_collocation_method : str
        Collocation method for layer-vs-layer (scatterometer) data:
        'individual' matches each SAR pixel to the closest scatterometer point (allowing reuse), or
        'cell-averaging' aggregates SAR pixels within aggregation_window_km around each
        (already-gridded) scatterometer point.
    filename_suffix : str
        Appended to the output filename stem, e.g. ``"_individual"`` →
        ``collocation_results_individual.nc``. Lets two collocation methods
        be run for the same recipe without overwriting each other.

    Returns
    -------
    xr.Dataset or None
        Dataset of collocated pairs, or None if no matches were found.
    """
    from pathlib import Path as _Path

    from .datatree_converter import DataTreeConverter
    from .model_collocation import ModelLayerCollocation

    base_dir = _Path(base_dir)
    coll_cfg = recipe.config.collocation

    _COLLOC_CLASSES = {
        "point_vs_layer": PointLayerCollocation,
        "layer_vs_layer": LayerLayerCollocation,
    }

    # Build a map of source_type to collocation_kwargs for per-source overrides
    source_type_overrides: Dict[str, Dict[str, Any]] = {}
    for src in recipe.config.validation_sources:
        if src.collocation_kwargs:
            source_type_overrides[src.source_type] = src.collocation_kwargs

    # Convert global collocation config to dict for easy merging
    # Use point_vs_layer as the base for global defaults
    pvl_cfg = coll_cfg.point_vs_layer
    global_coll_kwargs = {
        "spatial_tolerance_km": pvl_cfg.spatial_tolerance_km,
        "time_tolerance_minutes": pvl_cfg.time_tolerance_minutes,
        "interpolation_method": pvl_cfg.interpolation_method,
        "aggregation_window_km": pvl_cfg.aggregation_window_km,
        "validation_temporal_averaging_minutes": pvl_cfg.validation_temporal_averaging_minutes,
        "distance_weighting": pvl_cfg.distance_weighting,
        "gaussian_sigma_km": pvl_cfg.gaussian_sigma_km,
        "emit_diagnostics": emit_diagnostics,
    }
    
    # Start from the built-in defaults so recipes with no layer_vs_layer
    # section still get sensible per-source values, then merge recipe
    # overrides per-key (not a top-level dict replace) so overriding one
    # field of a layer_type spec does not drop that layer_type's other
    # defaults.
    from .recipe import DEFAULT_LAYER_TYPE_SPECS
    layer_vs_layer_specs = dict(DEFAULT_LAYER_TYPE_SPECS)
    if coll_cfg.layer_vs_layer is not None:
        for key, spec in coll_cfg.layer_vs_layer.layer_type_specs.items():
            layer_vs_layer_specs[key] = {**layer_vs_layer_specs.get(key, {}), **spec}

    recipe_layer_type_specs = (
        coll_cfg.layer_vs_layer.layer_type_specs if coll_cfg.layer_vs_layer is not None else {}
    )

    # Collect SAR nodes (one Dataset per SAR scene)
    sar_scenes: Dict[str, Any] = {}
    if "sar" in datatree.children:
        for name, node in datatree["sar"].children.items():
            sar_scenes[name] = node.to_dataset()

    if not sar_scenes:
        logger.warning("No SAR nodes found in DataTree — nothing to collocate.")
        return None

    # Collect validation nodes (flatten up to two levels deep) and bucket
    # each by its auto-detected collocation type, tracking source metadata.
    buckets: Dict[str, Dict[str, Any]] = {t: {} for t in _COLLOC_CLASSES}
    source_metadata: Dict[str, Dict[str, Any]] = {}  # source_name -> {source_type, colloc_kwargs}
    # DataFrame view of each validation node, built once here rather than
    # once per (SAR scene × source) pair inside the scene loop below.
    val_dfs: Dict[str, pd.DataFrame] = {}

    if "validation" in datatree.children:
        for name, node in datatree["validation"].children.items():
            ds = node.to_dataset()
            if "point" in ds.dims and len(ds.data_vars) > 0:
                ctype = _detect_collocation_type(ds, name)
                buckets[ctype][name] = ds
                # Try to infer source_type from the first matching validation source config
                source_type = ds.attrs.get("platform_type", name.split("/")[-1])
                source_metadata[name] = {
                    "source_type": source_type,
                    "colloc_kwargs": source_type_overrides.get(source_type, {}),
                }
            # One level deeper (e.g. validation/osi_saf_winds/<file>)
            for subname, subnode in node.children.items():
                sub_ds = subnode.to_dataset()
                if "point" in sub_ds.dims and len(sub_ds.data_vars) > 0:
                    path = f"{name}/{subname}"
                    ctype = _detect_collocation_type(sub_ds, path)
                    buckets[ctype][path] = sub_ds
                    # Use the top-level name as source_type hint
                    source_type = name
                    source_metadata[path] = {
                        "source_type": source_type,
                        "colloc_kwargs": source_type_overrides.get(source_type, {}),
                    }

    # Gridded "model" sources (ERA5, HYCOM) -- kept as raw, native
    # (time, lat, lon) Datasets rather than flattened into `buckets`/
    # `val_dfs` like every other validation source, since
    # ModelLayerCollocation interpolates the field directly onto SAR pixel
    # locations at collocation time instead of matching against
    # pre-existing rows. Detected via _model_source_type() rather than a
    # "point" dimension check.
    model_sources: Dict[str, "xr.Dataset"] = {}
    model_source_metadata: Dict[str, Dict[str, Any]] = {}
    if "validation" in datatree.children:
        for name, node in datatree["validation"].children.items():
            ds = node.to_dataset()
            model_source_type = _model_source_type(ds.attrs.get("data_type", ""))
            if model_source_type is not None:
                model_sources[name] = ds
                model_source_metadata[name] = {
                    "colloc_kwargs": source_type_overrides.get(model_source_type, {}),
                }
            # One level deeper: DataTreeConverter nests ERA5 at
            # "validation/era5/era5" like other satellite-product sources, so its
            # data_type attr lives on the leaf node below the (attr-less) group node.
            for subname, subnode in node.children.items():
                sub_ds = subnode.to_dataset()
                sub_model_source_type = _model_source_type(sub_ds.attrs.get("data_type", ""))
                if sub_model_source_type is not None:
                    path = f"{name}/{subname}"
                    model_sources[path] = sub_ds
                    model_source_metadata[path] = {
                        "colloc_kwargs": source_type_overrides.get(sub_model_source_type, {}),
                    }

    _merge_sibling_ssm_nodes(buckets, source_metadata, layer_vs_layer_specs)

    total_sources = sum(len(v) for v in buckets.values()) + len(model_sources)
    if total_sources == 0:
        logger.warning("No validation nodes with 'point' dimension found — nothing to collocate.")
        return None

    # np.atleast_1d normalises the scalar time (grid-mode) vs. (point,)
    # array (WV-mode) into one iterable. Scenes with no time coordinate, or
    # individual NaT entries, are skipped rather than raised here -- a NaT
    # would otherwise reach merge_asof as a null merge key, which pandas
    # rejects.
    sar_scene_times: List[datetime] = [
        pd.Timestamp(t).to_pydatetime()
        for ds in sar_scenes.values()
        if "time" in ds.coords
        for t in np.atleast_1d(ds["time"].values)
        if not pd.isna(t)
    ]

    for ctype, sources in buckets.items():
        for val_name, val_ds in sources.items():
            merged_raw_df = source_metadata.get(val_name, {}).get("_merged_raw_df")
            df = (
                merged_raw_df if merged_raw_df is not None
                else val_ds.to_dataframe().reset_index(drop=True)
            )
            source_label = val_ds.attrs.get("platform_type", val_name.split("/")[-1])

            if ctype == "point_vs_layer" and source_label == "ismn":
                # Average every ISMN reading within ±time_tolerance_minutes
                # of each SAR scene into one station-day value, instead of
                # picking the single nearest (always-nighttime, since S1
                # SSM scenes are stamped at midnight) reading. 
                per_source_kwargs = source_metadata.get(val_name, {}).get("colloc_kwargs", {})
                merged_kwargs = _merge_collocation_kwargs(global_coll_kwargs, per_source_kwargs)
                group_cols = ["platform_id"] if "platform_id" in df.columns else ["lon", "lat"]
                df = _average_within_sar_tolerance(
                    df, sar_scene_times, group_cols,
                    merged_kwargs["time_tolerance_minutes"],
                )
            elif ctype == "layer_vs_layer":
                layer_type = _resolve_layer_type(val_ds, val_name, layer_vs_layer_specs)
                if layer_type in SSM_SATELLITE_LAYER_TYPES:
                    # Blend every overpass of the same grid cell within ±time_tolerance
                    # into one day value (same treatment as ISMN above), which also shrinks
                    # soil-moisture's ~100k-point scale. Km-based sources are snapped to
                    # aggregation_window_km below; AMSR2's fixed-lattice G-Portal format is
                    # grouped by its raw lon/lat instead, since repeated readings already
                    # share exact coordinates.
                    spec = layer_vs_layer_specs.get(layer_type, {})
                    time_tol = spec.get(
                        "time_tolerance_minutes", global_coll_kwargs["time_tolerance_minutes"]
                    )
                    native_grid_deg = val_ds.attrs.get("native_grid_deg")
                    if native_grid_deg is not None:
                        # A degree-native fixed grid (e.g. AMSR2
                        # G-Portal's 0.1x0.1 EQR grid) reports identical
                        # lon/lat for repeated readings of the same cell --
                        # no snapping needed, group on the raw coordinates
                        # directly. Rounding this grid's cell centres
                        # (which sit at exact half-step offsets, e.g.
                        # 9.05/9.15/9.25/...) to a 0.1deg step would put
                        # every centre exactly on a tie, and NumPy's
                        # round-half-to-even resolves ties inconsistently,
                        # silently merging ~1 in 5 pairs of genuinely
                        # adjacent native cells.
                        df = _average_within_sar_tolerance(
                            df, sar_scene_times, ["lon", "lat"], time_tol,
                        )
                    else:
                        # Km-based (roughly equal-area) grids: a degree of
                        # longitude covers less physical distance than a
                        # degree of latitude away from the equator, by a
                        # factor of cos(latitude) -- widen the longitude
                        # step accordingly so the snap footprint is
                        # isotropic in physical km at any latitude.
                        lon_vals = df["lon"].to_numpy(dtype=float)
                        lat_vals = df["lat"].to_numpy(dtype=float)
                        agg_km = spec.get("aggregation_window_km", 25.0)
                        if (
                            layer_type == "scatterometer_ssm"
                            and "aggregation_window_km" not in recipe_layer_type_specs.get(layer_type, {})
                        ):
                            ascat_resolution_km = val_ds.attrs.get("ascat_resolution_km")
                            if ascat_resolution_km is not None:
                                agg_km = ascat_resolution_km
                        lat_step = agg_km / 111.0
                        mean_lat = float(np.nanmean(lat_vals)) if lat_vals.size else 0.0
                        lon_step = agg_km / (111.0 * max(np.cos(np.radians(mean_lat)), 1e-6))
                        df["_snap_lon"] = _snap_to_grid(lon_vals, lon_step)
                        df["_snap_lat"] = _snap_to_grid(lat_vals, lat_step)
                        df = _average_within_sar_tolerance(
                            df, sar_scene_times, ["_snap_lon", "_snap_lat"], time_tol,
                        )
                        df = df.drop(columns=["_snap_lon", "_snap_lat"], errors="ignore")

            val_dfs[val_name] = df

    for ctype, sources in buckets.items():
        if sources:
            logger.info(
                "Collocation pass '%s': %d source(s): %s",
                ctype, len(sources), list(sources),
            )

    all_collocations: List[CollocatedPoint] = []

    for sar_name, sar_ds in sar_scenes.items():
        if "lon" not in sar_ds.coords or "lat" not in sar_ds.coords:
            logger.warning("SAR node '%s' missing lon/lat coordinates — skipping.", sar_name)
            continue
        if "time" not in sar_ds.coords:
            logger.warning("SAR node '%s' missing time coordinate — skipping.", sar_name)
            continue

        # Detect SAR data type: grid (y, x) or point-based (WV mode)
        is_wv_mode = "point" in sar_ds.dims and "y" not in sar_ds.dims

        if is_wv_mode:
            # =========== WV MODE (SAR-footprint-anchored) ===========
            # Each vignette is a single point standing for a ~20×20 km
            # footprint, and vignettes are ~200 km apart — far too sparse to
            # match by requiring validation data within a few km of the point
            # centre. Instead, anchor on each vignette and aggregate every
            # validation obs within the footprint radius (see
            # _collocate_wv_points).
            sar_lons = sar_ds["lon"].values   # (point,)
            sar_lats = sar_ds["lat"].values   # (point,)
            sar_times = sar_ds["time"].values # (point,) as datetime64

            sar_point_vars = {
                var: sar_ds[var].values  # (point,)
                for var in sar_ds.data_vars
                if sar_ds[var].dims == ("point",)
            }

            if not sar_point_vars:
                logger.warning("SAR node '%s' (WV mode) has no point variables — skipping.", sar_name)
                continue

            n_points = len(sar_lons)
            logger.info("SAR node '%s' is WV mode with %d vignette point(s)", sar_name, n_points)

            for ctype, sources in buckets.items():
                if not sources:
                    continue
                for val_name, val_ds in sources.items():
                    per_source_kwargs = source_metadata.get(val_name, {}).get("colloc_kwargs", {})
                    merged_kwargs = _merge_collocation_kwargs(global_coll_kwargs, per_source_kwargs)

                    # Footprint radius (per-source override wins over recipe default)
                    footprint_radius_km = per_source_kwargs.get(
                        "sar_footprint_radius_km", coll_cfg.sar_footprint_radius_km
                    )

                    if ctype == "layer_vs_layer":
                        # Altimeter/scatterometer: sampled as a layer at the
                        # vignette with distance-weighted aggregation and the
                        # layer's own time tolerance. Resolve the layer type
                        # exactly as the grid path does.
                        layer_type = _resolve_layer_type(val_ds, val_name, layer_vs_layer_specs)
                        if layer_type in layer_vs_layer_specs:
                            merged_kwargs.update(layer_vs_layer_specs[layer_type])
                        _apply_hf_radar_resolution_override(
                            layer_type, val_ds, merged_kwargs, recipe_layer_type_specs,
                        )
                        collocation_type = "point_vs_layer"
                        distance_weighting = merged_kwargs.get("distance_weighting", "equal")
                    else:
                        # In-situ: plain average over the obs inside the footprint.
                        collocation_type = "point_vs_point"
                        distance_weighting = "equal"

                    df = val_dfs[val_name]
                    source_label = val_ds.attrs.get("platform_type", val_name.split("/")[-1])

                    matches = _collocate_wv_points(
                        sar_lons=sar_lons,
                        sar_lats=sar_lats,
                        sar_times=sar_times,
                        sar_point_vars=sar_point_vars,
                        val_data=df,
                        val_source=source_label,
                        footprint_radius_km=footprint_radius_km,
                        time_tolerance_minutes=merged_kwargs.get("time_tolerance_minutes", 30),
                        distance_weighting=distance_weighting,
                        gaussian_sigma_km=merged_kwargs.get("gaussian_sigma_km", 5.0),
                        collocation_type=collocation_type,
                        sar_scene_name=sar_name,
                    )
                    all_collocations.extend(matches)
                    logger.info(
                        "SAR '%s' × validation '%s' [%s]: %d match(es)",
                        sar_name, val_name, collocation_type, len(matches),
                    )

            # Gridded "model" source (e.g. ERA5, HYCOM); WV vignettes are 
            # sparse SAR-anchor points, so ModelLayerCollocation.collocate_points 
            # always interpolates the model directly at each vignette regardless
            # of the recipe's chosen method.
            for val_name, val_ds in model_sources.items():
                per_source_kwargs = model_source_metadata.get(val_name, {}).get("colloc_kwargs", {})
                layer_type = val_ds.attrs.get("data_type", "era5")
                model_kwargs = dict(layer_vs_layer_specs.get(layer_type, {}))
                model_kwargs.update(per_source_kwargs)
                model_colloc = ModelLayerCollocation(
                    method=model_kwargs.get("method", "cell-averaging"),
                    temporal_method=model_kwargs.get("temporal_method", "hyperbolic"),
                )
                source_label = val_ds.attrs.get("platform_type", val_name.split("/")[-1])
                matches = model_colloc.collocate_points(
                    sar_point_vars=sar_point_vars,
                    sar_lons=sar_lons, sar_lats=sar_lats, sar_times=sar_times,
                    model_ds=val_ds, val_source=source_label, sar_scene_name=sar_name,
                )
                all_collocations.extend(matches)
                logger.info(
                    "SAR '%s' × validation '%s' [model_vs_layer/points]: %d match(es)",
                    sar_name, val_name, len(matches),
                )

        else:
            # =========== IW/EW MODE (Grid-based) ===========
            sar_lon = sar_ds["lon"].values           # (y, x)
            sar_lat = sar_ds["lat"].values           # (y, x)
            acq_time = pd.Timestamp(sar_ds["time"].values).to_pydatetime()
            sar_time_arr = np.array([acq_time])

            # Expand each variable to (1, y, x) for collocate()
            sar_data_3d: Dict[str, np.ndarray] = {
                var: sar_ds[var].values[np.newaxis, :, :]
                for var in sar_ds.data_vars
                if sar_ds[var].dims == ("y", "x")
            }

            if not sar_data_3d:
                logger.warning("SAR node '%s' has no (y, x) variables — skipping.", sar_name)
                continue

            # Built once per SAR scene and reused across every matched validation
            # source (both point_vs_layer and layer_vs_layer resolve to
            # PointLayerCollocation.collocate(), which accepts a pre-built tree) --
            # rebuilding per validation file is pathological for many small
            # per-source files against one large grid (e.g. ISMN's per-station
            # files against a multi-million-cell grid).
            grid_tree = PointLayerCollocation._build_grid_tree(sar_lon, sar_lat)

            # Pass 1/2: layer_vs_layer and point_vs_layer buckets (model
            # sources are handled separately below as pass 3).
            for ctype, sources in buckets.items():
                if not sources:
                    continue
                for val_name, val_ds in sources.items():
                    # Apply per-source kwargs overrides
                    per_source_kwargs = source_metadata.get(val_name, {}).get("colloc_kwargs", {})
                    merged_kwargs = _merge_collocation_kwargs(global_coll_kwargs, per_source_kwargs)

                    df = val_dfs[val_name]
                    source_label = val_ds.attrs.get("platform_type", val_name.split("/")[-1])

                    # For layer_vs_layer, apply layer-type-specific specs from recipe
                    if ctype == "layer_vs_layer" and layer_vs_layer_specs:
                        layer_type = _resolve_layer_type(val_ds, val_name, layer_vs_layer_specs)

                        if layer_type in layer_vs_layer_specs:
                            merged_kwargs.update(layer_vs_layer_specs[layer_type])
                            logger.info(
                                "Applying layer_vs_layer specs for '%s': %s",
                                layer_type, layer_vs_layer_specs[layer_type],
                            )
                        _apply_hf_radar_resolution_override(
                            layer_type, val_ds, merged_kwargs, recipe_layer_type_specs,
                        )
                        _apply_ascat_resolution_override(
                            layer_type, val_ds, merged_kwargs, recipe_layer_type_specs,
                        )

                        # Add layer-vs-layer collocation method
                        merged_kwargs["method"] = layer_vs_layer_collocation_method

                    colloc = _COLLOC_CLASSES[ctype](**merged_kwargs)

                    matches = colloc.collocate(
                        sar_data=sar_data_3d,
                        sar_lon=sar_lon,
                        sar_lat=sar_lat,
                        sar_time=sar_time_arr,
                        val_data=df,
                        val_source=source_label,
                        sar_scene_name=sar_name,
                        grid_tree=grid_tree,
                    )
                    all_collocations.extend(matches)
                    logger.info(
                        "SAR '%s' × validation '%s' [%s]: %d match(es)",
                        sar_name, val_name, ctype, len(matches),
                    )

            # Gridded model sources (e.g. ERA5 HYCOM) 
            for val_name, val_ds in model_sources.items():
                per_source_kwargs = model_source_metadata.get(val_name, {}).get("colloc_kwargs", {})
                layer_type = val_ds.attrs.get("data_type", "era5")
                model_kwargs = dict(layer_vs_layer_specs.get(layer_type, {}))
                model_kwargs.update(per_source_kwargs)
                model_colloc = ModelLayerCollocation(
                    method=model_kwargs.get("method", "cell-averaging"),
                    temporal_method=model_kwargs.get("temporal_method", "hyperbolic"),
                    time_tolerance_minutes=model_kwargs.get("time_tolerance_minutes", 60),
                    aggregation_window_km=model_kwargs.get("aggregation_window_km", 12.5),
                    distance_weighting=model_kwargs.get("distance_weighting", "equal"),
                    gaussian_sigma_km=model_kwargs.get("gaussian_sigma_km", 12.5),
                )
                source_label = val_ds.attrs.get("platform_type", val_name.split("/")[-1])
                matches = model_colloc.collocate(
                    sar_data=sar_data_3d, sar_lon=sar_lon, sar_lat=sar_lat,
                    sar_time=sar_time_arr, model_ds=val_ds,
                    val_source=source_label, sar_scene_name=sar_name,
                )
                all_collocations.extend(matches)
                logger.info(
                    "SAR '%s' × validation '%s' [model_vs_layer]: %d match(es)",
                    sar_name, val_name, len(matches),
                )

    if not all_collocations:
        logger.warning("Collocation complete — no matches found.")
        return None

    result_ds = DataTreeConverter.from_collocations(all_collocations)
    if result_ds is None:
        logger.warning(
            "Collocation produced no Dataset despite %d match(es).", len(all_collocations)
        )
        return None

    # Carry the datatree's CF metadata (standard_name/long_name/units) over
    # to the matched sar_*/val_* columns.
    from ._cf_metadata import annotate_collocation_ds
    annotate_collocation_ds(result_ds, datatree)

    out_path = base_dir / f"collocation_results{filename_suffix}.nc"
    result_ds.to_netcdf(out_path)
    logger.info(
        "Collocation results saved to %s (%d matches total)",
        out_path, len(all_collocations),
    )

    return result_ds


# ---------------------------------------------------------------------------
# 2. Layer vs. Layer
# ---------------------------------------------------------------------------

class LayerLayerCollocation(PointLayerCollocation):
    """
    Match a gridded layer_vs_layer validation product (scatterometer,
    altimeter, HF-radar grid, or satellite soil moisture) to a SAR layer, by
    aggregating SAR pixels around each validation point, via one of two
    methods selected by ``method``.

    Aggregation approach
    --------------------
    These sources are delivered pre-gridded — one observation per resolution
    cell (e.g. ASCAT's ~12.5x12.5 km wind-vector cell) — so each validation
    point already *is* its own cell; no spatial re-clustering is needed.
    ``method="cell-averaging"`` (default) therefore reuses the parent
    ``PointLayerCollocation.collocate()`` algorithm directly, per validation
    point:

    1. **SAR Aggregation**: For each validation point, finds all SAR grid
    cells within ``aggregation_window_km`` and computes a distance-weighted
    average of SAR variables.

    2. **Output**: One ``CollocatedPoint`` per validation point (per matching
    SAR time), with the aggregated SAR mean vs. the point's own raw value.

    ``method="individual"`` instead matches each SAR pixel to its nearest
    validation point via KD-tree search, allowing reuse; see
    :meth:`_collocate_individual`.

    Collocation parameters
    ----------------------
    The constructor defaults below are ASCAT-tuned; ``run_collocation``
    overrides them per layer type via ``DEFAULT_LAYER_TYPE_SPECS`` for every
    other supported source (altimeter, HF-radar, soil moisture, ...):

    - ``aggregation_window_km`` : 12.5 km (ASCAT scatterometer cell size)
    - ``time_tolerance_minutes`` : 180 min (±3 hours, per Abderrahim et al. 2019)
    - ``distance_weighting`` : "equal" (uniform weights across regular grid cells)
    - ``validation_temporal_averaging_minutes`` : 60 min (±1 hour window)
    """

    collocation_type: str = "layer_vs_layer"

    def __init__(
        self,
        spatial_tolerance_km: float = 12.5,
        time_tolerance_minutes: int = 180,
        interpolation_method: str = "nearest",
        aggregation_window_km: float = 12.5,
        validation_temporal_averaging_minutes: int = 60,
        distance_weighting: str = "equal",
        gaussian_sigma_km: float = 12.5,
        emit_diagnostics: bool = False,
        dedup_nearest_in_time: bool = False,
        method: str = "cell-averaging",
    ) -> None:
        """
        Initialize LayerLayerCollocation with scatterometer-optimized defaults.

        Parameters
        ----------
        spatial_tolerance_km : float
            Maximum distance for pre-filtering (12.5 km per ASCAT specification).
        time_tolerance_minutes : int
            Maximum time difference for matching (180 min per hal-04202202).
        interpolation_method : str
            Retained for compatibility; currently unused.
        aggregation_window_km : float
            SAR aggregation radius around each scatterometer cell (12.5 km).
        validation_temporal_averaging_minutes : int
            Half-width of temporal window for scatterometer averaging (60 min).
        distance_weighting : str
            SAR weighting method: "equal" (default), "gaussian", "inverse_distance", "linear".
        gaussian_sigma_km : float
            Gaussian sigma if using Gaussian weighting (12.5 km default).
        emit_diagnostics : bool
            If True, emit detailed per-cell diagnostic logging.
        dedup_nearest_in_time : bool
            When True, a validation point (keyed by platform_id if present,
            else its own (lon, lat)) contributes only its single
            closest-in-time reading per SAR time index, instead of one
            collocation per candidate reading inside the tolerance window.
            Default False (unchanged historical behaviour).
        method : str
            Collocation method: "cell-averaging" (clusters scatterometer into grid cells, aggregates SAR)
            or "individual" (matches each SAR pixel to closest scatterometer point, allowing reuse).
        """
        super().__init__(
            spatial_tolerance_km=spatial_tolerance_km,
            time_tolerance_minutes=time_tolerance_minutes,
            interpolation_method=interpolation_method,
            aggregation_window_km=aggregation_window_km,
            validation_temporal_averaging_minutes=validation_temporal_averaging_minutes,
            distance_weighting=distance_weighting,
            gaussian_sigma_km=gaussian_sigma_km,
            dedup_nearest_in_time=dedup_nearest_in_time,
        )
        self.method = method
        self.emit_diagnostics = emit_diagnostics
        logger.debug(
            "LayerLayerCollocation initialized: %d min temporal, %.1f km spatial, "
            "%s weighting, %.1f km aggregation window, diagnostics=%s",
            self.time_tolerance_minutes,
            self.spatial_tolerance_km,
            self.distance_weighting,
            self.aggregation_window_km,
            self.emit_diagnostics,
        )

    def _collocate_individual(
        self,
        sar_data: Dict[str, np.ndarray],
        sar_lon: np.ndarray,
        sar_lat: np.ndarray,
        sar_time: np.ndarray,
        val_data: pd.DataFrame,
        val_source: str,
        sar_scene_name: str = "",
    ) -> List[CollocatedPoint]:
        """
        Match each individual SAR pixel to its closest validation point (individual method).

        For each SAR grid cell at each time step:
        1. Find closest validation point (vectorized nearest-neighbour search via a
        KD-tree over unit-sphere Cartesian coordinates)
        2. Check spatial tolerance (within spatial_tolerance_km)
        3. Check temporal match (within time_tolerance_minutes)
        4. For gridded HF-radar (EWCT/NSCT) matched against SAR radial velocity
        (rvlRadVel/rvlHeading), project the current onto the SAR line-of-sight
        for comparability (see :func:`_project_currents_to_radial`)
        5. Create CollocatedPoint with SAR as anchor, validation point as matched value
        6. Validation points can be reused across multiple SAR cells

        Returns
        -------
        list[CollocatedPoint]
            List of collocated matches (one per matched SAR cell).
        """
        from datetime import timedelta as _td

        from scipy.spatial import cKDTree

        sar_times = _to_datetime_array(sar_time)
        collocations: List[CollocatedPoint] = []

        # Pre-filter scatterometer data: spatial and temporal bounds.
        # nanmin/nanmax: SAR grids commonly carry NaN lon/lat at masked or
        # edge cells, and plain min/max would propagate that NaN into every
        # bound, making the mask below all-False.
        deg_buf = self.spatial_tolerance_km / 55.0
        lon_min = float(np.nanmin(sar_lon)) - deg_buf
        lon_max = float(np.nanmax(sar_lon)) + deg_buf
        lat_min = float(np.nanmin(sar_lat)) - deg_buf
        lat_max = float(np.nanmax(sar_lat)) + deg_buf
        
        spatial_mask = (
            (val_data["lon"] >= lon_min) & (val_data["lon"] <= lon_max) &
            (val_data["lat"] >= lat_min) & (val_data["lat"] <= lat_max)
        )
        val_data_filtered = val_data[spatial_mask].copy()

        if val_data_filtered.empty:
            logger.debug("No scatterometer data within spatial bounds")
            return collocations

        # Temporal pre-filter
        t_min = min(sar_times) - _td(minutes=self.time_tolerance_minutes)
        t_max = max(sar_times) + _td(minutes=self.time_tolerance_minutes)
        if hasattr(t_min, "tzinfo") and t_min.tzinfo is not None:
            t_min = t_min.replace(tzinfo=None)
            t_max = t_max.replace(tzinfo=None)

        val_times_pd = pd.to_datetime(val_data_filtered["time"].values)
        if val_times_pd.tz is not None:
            val_times_pd = val_times_pd.tz_localize(None)

        temporal_mask = (val_times_pd >= t_min) & (val_times_pd <= t_max)
        val_data_filtered = val_data_filtered[temporal_mask]

        if val_data_filtered.empty:
            logger.debug("No scatterometer data within temporal window")
            return collocations

        logger.debug(
            "Pre-filters kept %d scatterometer points (spatial + temporal)",
            len(val_data_filtered),
        )

        # Identify numeric columns in validation data
        val_numeric_cols = [
            col for col in val_data_filtered.columns
            if col not in {"lon", "lat", "time", "platform_id"} and
            pd.api.types.is_numeric_dtype(val_data_filtered[col])
        ]

        # Pre-extract scatterometer coordinates and times for vectorized distance computation
        scat_lons = val_data_filtered["lon"].values.astype(float)
        scat_lats = val_data_filtered["lat"].values.astype(float)
        scat_times_pd = pd.to_datetime(val_data_filtered["time"].values)
        if scat_times_pd.tz is not None:
            scat_times_pd = scat_times_pd.tz_localize(None)
        scat_times_np = scat_times_pd.values.astype("datetime64[ns]")
        scat_times_objs = np.array([t.to_pydatetime() if hasattr(t, 'to_pydatetime') else t
                                    for t in scat_times_pd], dtype=object)

        # Build a KD-tree over validation points in unit-sphere Cartesian
        # coordinates once.
        R = 6371.0
        scat_tree = cKDTree(_lonlat_to_unit_xyz(scat_lons, scat_lats))

        sar_grid_y, sar_grid_x = sar_lon.shape
        sar_lon_flat = sar_lon.ravel()
        sar_lat_flat = sar_lat.ravel()
        # cKDTree.query rejects NaN query points outright, and SAR grids can
        # carry NaN lon/lat at masked/edge cells independently of the data
        # variables, so this must be excluded up front alongside has_data_mask.
        has_coord_mask = np.isfinite(sar_lon_flat) & np.isfinite(sar_lat_flat)
        var_names = list(sar_data.keys())

        # Process each SAR time
        rejected_spatial = 0
        rejected_temporal = 0
        rejected_no_data = 0

        for t_idx, sar_t in enumerate(sar_times):
            # Stack all SAR variables for this time step: (n_vars, n_cells)
            values_stack = np.stack(
                [sar_data[var][t_idx].ravel() for var in var_names], axis=0
            )
            has_data_mask = ~np.all(np.isnan(values_stack), axis=0) & has_coord_mask
            rejected_no_data += int(np.sum(~has_data_mask))

            candidate_cells = np.where(has_data_mask)[0]
            if candidate_cells.size == 0:
                continue

            # Vectorized nearest-scatterometer-point search for every
            # SAR cell that has at least one valid variable.
            chord_dist, closest_scat_idx = scat_tree.query(
                _lonlat_to_unit_xyz(sar_lon_flat[candidate_cells], sar_lat_flat[candidate_cells]),
                k=1,
            )
            distances_km = R * 2.0 * np.arcsin(np.clip(chord_dist, 0.0, 2.0) / 2.0)

            spatial_ok = distances_km <= self.spatial_tolerance_km
            rejected_spatial += int(np.sum(~spatial_ok))
            if self.emit_diagnostics and np.any(~spatial_ok):
                for cell_idx, dist in zip(candidate_cells[~spatial_ok], distances_km[~spatial_ok]):
                    y_idx, x_idx = divmod(int(cell_idx), sar_grid_x)
                    logger.debug(
                        "SAR cell (y=%d, x=%d) at (%.3f°, %.3f°): REJECTED spatial (dist=%.2f km > %.1f km)",
                        y_idx, x_idx, sar_lon_flat[cell_idx], sar_lat_flat[cell_idx],
                        dist, self.spatial_tolerance_km,
                    )

            if not np.any(spatial_ok):
                continue

            spatial_cells = candidate_cells[spatial_ok]
            spatial_distances = distances_km[spatial_ok]
            spatial_scat_idx = closest_scat_idx[spatial_ok]

            # Vectorized temporal check for the spatially-valid candidates.
            sar_t_np = np.datetime64(sar_t)
            time_diff_min = np.abs(
                (scat_times_np[spatial_scat_idx] - sar_t_np) / np.timedelta64(1, "m")
            ).astype(float)
            temporal_ok = time_diff_min <= self.time_tolerance_minutes
            rejected_temporal += int(np.sum(~temporal_ok))
            if self.emit_diagnostics and np.any(~temporal_ok):
                for cell_idx, diff in zip(spatial_cells[~temporal_ok], time_diff_min[~temporal_ok]):
                    y_idx, x_idx = divmod(int(cell_idx), sar_grid_x)
                    logger.debug(
                        "SAR cell (y=%d, x=%d): REJECTED temporal (time_diff=%.1f min > %d min)",
                        y_idx, x_idx, diff, self.time_tolerance_minutes,
                    )

            if not np.any(temporal_ok):
                continue

            matched_cells = spatial_cells[temporal_ok]
            matched_distance = spatial_distances[temporal_ok]
            matched_time_diff = time_diff_min[temporal_ok]
            matched_scat_idx = spatial_scat_idx[temporal_ok]

            # Only the already-matched cells reach this per-row loop now,
            # instead of every (SAR pixel, scatterometer point) pair.
            for k, cell_idx in enumerate(matched_cells):
                y_idx, x_idx = divmod(int(cell_idx), sar_grid_x)

                sar_aggregated = {
                    var: float(values_stack[v_idx, cell_idx])
                    for v_idx, var in enumerate(var_names)
                    if not np.isnan(values_stack[v_idx, cell_idx])
                }

                scat_idx = int(matched_scat_idx[k])
                closest_row = val_data_filtered.iloc[scat_idx]
                val_aggregated = {
                    col: float(closest_row[col])
                    for col in val_numeric_cols
                    if pd.notna(closest_row[col])
                }

                if not val_aggregated:
                    rejected_no_data += 1
                    if self.emit_diagnostics:
                        logger.debug(
                            "SAR cell (y=%d, x=%d): REJECTED (no valid scatterometer values)",
                            y_idx, x_idx,
                        )
                    continue

                # Project currents onto the SAR line-of-sight so gridded
                # HF-radar (EWCT/NSCT) is comparable to rvlRadVel — the
                # cell-averaging path does this via PointLayerCollocation, but
                # the SAR-anchor 'individual' path must do it explicitly.
                if (
                    "rvlRadVel" in sar_aggregated
                    and "rvlHeading" in sar_aggregated
                    and "EWCT" in val_aggregated
                    and "NSCT" in val_aggregated
                ):
                    val_aggregated["rvlRadVel_projection"] = _project_currents_to_radial(
                        float(val_aggregated["EWCT"]),
                        float(val_aggregated["NSCT"]),
                        float(sar_aggregated["rvlHeading"]),
                    )

                sar_cell_lon = float(sar_lon_flat[cell_idx])
                sar_cell_lat = float(sar_lat_flat[cell_idx])
                closest_scat_lon = float(scat_lons[scat_idx])
                closest_scat_lat = float(scat_lats[scat_idx])
                closest_scat_time = scat_times_objs[scat_idx]

                if self.emit_diagnostics:
                    logger.debug(
                        "SAR cell (%.3f°, %.3f°) MATCHED to scatterometer (%.3f°, %.3f°) "
                        "at distance=%.2f km, time_diff=%.1f min",
                        sar_cell_lon, sar_cell_lat, closest_scat_lon, closest_scat_lat,
                        matched_distance[k], matched_time_diff[k],
                    )

                # Create CollocatedPoint with SAR as anchor
                collocations.append(
                    CollocatedPoint(
                        sar_lon=sar_cell_lon,
                        sar_lat=sar_cell_lat,
                        sar_time=sar_t,
                        sar_data=sar_aggregated,
                        val_lon=closest_scat_lon,
                        val_lat=closest_scat_lat,
                        val_time=closest_scat_time,
                        val_data=val_aggregated,
                        spatial_distance_km=float(matched_distance[k]),
                        temporal_distance_minutes=float(matched_time_diff[k]),
                        val_source=val_source,
                        val_id=None,  # Scatterometer points don't have IDs
                        collocation_type=self.collocation_type,
                        sar_y_idx=y_idx,
                        sar_x_idx=x_idx,
                        sar_scene_name=sar_scene_name,
                    )
                )

        if self.emit_diagnostics:
            logger.info(
                "[SUMMARY] LayerLayerCollocation (SAR-anchor): %d matches from SAR grid. "
                "Rejected: %d spatial, %d temporal, %d no-data.",
                len(collocations), rejected_spatial, rejected_temporal, rejected_no_data,
            )
            if collocations:
                spatial_dists = [c.spatial_distance_km for c in collocations]
                temporal_dists = [c.temporal_distance_minutes for c in collocations]
                logger.info(
                    "[DISTANCES] Matched pairs: spatial=[%.2f, %.2f, %.2f] km (min/median/max), "
                    "temporal=[%.1f, %.1f, %.1f] min (min/median/max)",
                    np.min(spatial_dists), np.median(spatial_dists), np.max(spatial_dists),
                    np.min(temporal_dists), np.median(temporal_dists), np.max(temporal_dists),
                )

        logger.info(
            "%s (individual/SAR-anchor method): found %d matches from SAR grid (source=%s)",
            self.__class__.__name__, len(collocations), val_source,
        )
        return collocations

    def collocate(
        self,
        sar_data: Dict[str, np.ndarray],
        sar_lon: np.ndarray,
        sar_lat: np.ndarray,
        sar_time: np.ndarray,
        val_data: pd.DataFrame,
        val_source: str,
        sar_scene_name: str = "",
        grid_tree: Optional[Tuple["cKDTree", np.ndarray, int]] = None,
    ) -> List[CollocatedPoint]:
        """
        Match a layer_vs_layer validation source to SAR using selected 
        collocation method.

        Dispatches to either individual point-to-point or cell-averaging methods
        based on self.method setting. ``grid_tree`` (see
        ``PointLayerCollocation.collocate``) is only used by the
        cell-averaging path, which delegates to that method; the individual
        (SAR-anchor) path builds its own KD-tree over the validation points
        instead, so it ignores this parameter.

        Parameters
        ----------
        sar_data : dict
            SAR variables as 3-D arrays with shape ``(time, y, x)``.
        sar_lon, sar_lat : np.ndarray
            SAR coordinate grids, shape ``(y, x)``.
        sar_time : array-like
            SAR acquisition times, shape ``(time,)``.
        val_data : pd.DataFrame
            Validation data with columns ``lon``, ``lat``, ``time``, and variables.
        val_source : str
            Label for validation source.
        sar_scene_name : str
            Name of SAR scene node in DataTree.

        Returns
        -------
        list[CollocatedPoint]
            List of collocated matches.
        """
        if self.method == "individual":
            return self._collocate_individual(
                sar_data, sar_lon, sar_lat, sar_time, val_data, val_source, sar_scene_name
            )
        else:
            # Default to cell-averaging
            return self._collocate_cell_averaging(
                sar_data, sar_lon, sar_lat, sar_time, val_data, val_source, sar_scene_name,
                grid_tree=grid_tree,
            )

    def _collocate_cell_averaging(
        self,
        sar_data: Dict[str, np.ndarray],
        sar_lon: np.ndarray,
        sar_lat: np.ndarray,
        sar_time: np.ndarray,
        val_data: pd.DataFrame,
        val_source: str,
        sar_scene_name: str = "",
        grid_tree: Optional[Tuple["cKDTree", np.ndarray, int]] = None,
    ) -> List[CollocatedPoint]:
        """
        Match a layer_vs_layer validation source to SAR grid using spatial aggregation
        (cell-averaging method).

        Layer_vs_layer validation sources already represent their own resolution
        cell or footprint (e.g. a scatterometer wind-vector cell, an along-track
        altimeter point) — no spatial clustering is needed. This reuses
        ``PointLayerCollocation.collocate()`` directly: for each
        validation point, find all SAR cells within
        ``aggregation_window_km``, average them (distance-weighted or
        equal), and match against the point's own raw value.

        Parameters
        ----------
        sar_data : dict
            SAR variables as 3-D arrays with shape ``(time, y, x)``.
        sar_lon, sar_lat : np.ndarray
            SAR coordinate grids, shape ``(y, x)``.
        sar_time : array-like
            SAR acquisition times, shape ``(time,)``.
        val_data : pd.DataFrame
            Validation data with columns ``lon``, ``lat``, ``time``, and
            any number of variable columns.
        val_source : str
            Label for validation source.
        sar_scene_name : str
            Name of SAR scene node in DataTree.

        Returns
        -------
        list[CollocatedPoint]
            List of collocated matches (one per validation point, per
            matching SAR time).
        """
        return PointLayerCollocation.collocate(
            self, sar_data, sar_lon, sar_lat, sar_time,
            val_data, val_source, sar_scene_name,
            grid_tree=grid_tree,
        )
