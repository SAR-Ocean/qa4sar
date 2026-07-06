"""
Collocation algorithms — step 3 of the validation pipeline.

Three collocation geometries are supported:

1. ``PointLayerCollocation``      — fixed / slow-moving point vs. SAR grid  ✅
2. ``TrajectoryLayerCollocation`` — moving trajectory vs. SAR grid          🚧
3. ``LayerLayerCollocation``      — gridded product vs. SAR grid            🚧
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "CollocatedPoint",
    "PointLayerCollocation",
    "TrajectoryLayerCollocation",
    "LayerLayerCollocation",
    "run_collocation",
    "_detect_collocation_type",
    "TRAJECTORY_PLATFORM_TYPES",
    "LAYER_DATA_TYPES",
    "LAYER_SOURCE_PATHS",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CollocatedPoint:
    """One matched pair between a SAR grid cell and a validation observation."""

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
    collocation_type: str = "point_vs_layer"  # point_vs_layer | trajectory_vs_layer | layer_vs_layer

    # Pixel indices — used by patch_extractor to retrieve a spatial neighbourhood
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


def _haversine_distance_grid(
    lon1: float, lat1: float,
    grid_lon: np.ndarray, grid_lat: np.ndarray,
) -> np.ndarray:
    """Great-circle distance from a scalar point to every cell in a 2-D grid (km)."""
    R = 6371.0
    dlat = np.radians(grid_lat - lat1)
    dlon = np.radians(grid_lon - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(grid_lat)) * np.sin(dlon / 2.0) ** 2
    )
    return R * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _to_datetime_array(time_array) -> np.ndarray:
    """Normalise heterogeneous time inputs to an object array of Python datetimes."""
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
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        weights = weights / weight_sum
    return weights


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
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        weights = weights / weight_sum
    return weights


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
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        weights = weights / weight_sum
    return weights


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

# platform_type attribute values that indicate a moving trajectory source
TRAJECTORY_PLATFORM_TYPES = {"ferrybox", "fb", "drifter", "ad"}
# data_type attribute values that indicate a gridded layer source
LAYER_DATA_TYPES = {"scatterometer"}
# path-fragment fallbacks when attributes are absent
LAYER_SOURCE_PATHS = {"osi_saf_winds", "scatterometer"}


def _detect_collocation_type(val_ds: "xr.Dataset", source_path: str) -> str:
    """
    Infer the appropriate collocation class name from a validation Dataset.

    Checks (in order):
    1. ``platform_type`` attribute  → trajectory or default
    2. ``data_type`` attribute      → layer or default
    3. Source path fragment         → layer or default
    """
    platform_type = val_ds.attrs.get("platform_type", "").lower()
    data_type = val_ds.attrs.get("data_type", "").lower()

    if platform_type in TRAJECTORY_PLATFORM_TYPES:
        return "trajectory_vs_layer"
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

    2. **Validation Aggregation**: Temporally average validation observations
       within ±`validation_temporal_averaging_minutes` around each observation.

    3. **Output**: Single `CollocatedPoint` per validation observation with
       aggregated SAR mean vs. aggregated validation mean.

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
        Half-width (minutes) of temporal window for validation data averaging.
        Observations within ±this window are averaged. Default: 30 min.
    distance_weighting : str
        Distance weighting method for SAR aggregation:
        - ``"gaussian"`` — Gaussian kernel (default)
        - ``"inverse_distance"`` — Inverse distance (1/d^2)
        - ``"linear"`` — Linear decay
        - ``"equal"`` — Uniform weights
    gaussian_sigma_km : float
        Standard deviation (km) for Gaussian weighting. Default: 5.0 km.
    """

    #: Collocation type label stored on each CollocatedPoint result.
    collocation_type: str = "point_vs_layer"

    def __init__(
        self,
        spatial_tolerance_km: float = 12.5, #based on teh 12.5 km spatial tolerance used in Abderrahim et al. 2019
        time_tolerance_minutes: int = 30, #based on the 30 min interval for buoys vs SAR in Abderrahim et al. 2019
        interpolation_method: str = "nearest",
        aggregation_window_km: float = 5.0,
        validation_temporal_averaging_minutes: int = 30,
        distance_weighting: str = "gaussian",
        gaussian_sigma_km: float = 5.0,
        emit_diagnostics: bool = False,
    ) -> None:
        self.spatial_tolerance_km = spatial_tolerance_km
        self.time_tolerance_minutes = time_tolerance_minutes
        self.interpolation_method = interpolation_method
        self.aggregation_window_km = aggregation_window_km
        self.validation_temporal_averaging_minutes = validation_temporal_averaging_minutes
        self.distance_weighting = distance_weighting
        self.gaussian_sigma_km = gaussian_sigma_km

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
    ) -> List[CollocatedPoint]:
        """
        Match validation observations to the SAR grid using distance-weighted aggregation.

        Algorithm
        ---------
        For each validation observation:

        1. Find all SAR cells within ``aggregation_window_km`` (circular radius).
        2. Compute distance-weighted average of SAR variables using ``distance_weighting`` method.
        3. Temporally average validation observations within ±``validation_temporal_averaging_minutes``.
        4. Create single `CollocatedPoint` with aggregated SAR vs. aggregated validation.

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

        Returns
        -------
        list[CollocatedPoint]
            List of collocated matches (one per validation observation).
        """
        from datetime import timedelta as _td

        sar_times = _to_datetime_array(sar_time)
        collocations: List[CollocatedPoint] = []

        # Pre-filters: eliminate validation rows that cannot match
        # Use spatial_tolerance_km for initial bounding box
        deg_buf = self.aggregation_window_km / 55.0  # Use aggregation window for pre-filter
        lon_min = float(sar_lon.min()) - deg_buf
        lon_max = float(sar_lon.max()) + deg_buf
        lat_min = float(sar_lat.min()) - deg_buf
        lat_max = float(sar_lat.max()) + deg_buf

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
            if col not in {"lon", "lat", "time", "platform_id"} and
            pd.api.types.is_numeric_dtype(val_data_filtered[col])
        ]

        # Process each validation observation
        for idx, val_row in val_data_filtered.iterrows():
            v_lon = float(val_row["lon"])
            v_lat = float(val_row["lat"])
            v_time = _to_datetime_array([val_row["time"]])[0]

            # Find nearby SAR cells within aggregation window
            nearby_cells_with_dist = self._nearby_cells_with_distances(
                v_lon, v_lat, sar_lon, sar_lat, self.aggregation_window_km
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

                # Aggregate validation observations temporally
                val_aggregated = self._average_validation_observations(
                    val_data_filtered,
                    v_time,
                    self.validation_temporal_averaging_minutes,
                    numeric_cols,
                )

                # If no temporal aggregation found, use current observation
                if not val_aggregated:
                    val_aggregated = {
                        col: float(val_row[col])
                        for col in numeric_cols
                        if pd.notna(val_row[col])
                    }

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
                            heading_deg = np.nanmean(valid_headings)
                            heading_rad = np.radians(float(heading_deg) - 90.0)
                            ewct = float(val_aggregated["EWCT"])
                            nsct = float(val_aggregated["NSCT"])
                            radial_vel = ewct * np.cos(heading_rad) + nsct * np.sin(heading_rad)
                            val_aggregated["rvlRadVel_projection"] = radial_vel
                    except (KeyError, ValueError, TypeError) as e:
                        logger.debug("RVL projection failed: %s", e)

                # Create CollocatedPoint
                collocations.append(
                    CollocatedPoint(
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
                        val_source=val_source,
                        val_id=val_row.get("platform_id"),
                        collocation_type=self.collocation_type,
                        sar_y_idx=y_idx,
                        sar_x_idx=x_idx,
                        sar_scene_name=sar_scene_name,
                    )
                )

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
        distances = np.array(
            [abs((t - target).total_seconds() / 60.0) for t in time_array]
        )
        return np.where(distances <= self.time_tolerance_minutes)[0].tolist()

    def _nearby_cells_with_distances(
        self,
        lon: float, lat: float,
        grid_lon: np.ndarray, grid_lat: np.ndarray,
        max_distance_km: float,
    ) -> List[Tuple[int, int, float]]:
        """
        Find SAR cells within max_distance_km and return with distances.
        
        Returns
        -------
        list of (y_idx, x_idx, distance_km) tuples
        """
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

    def _average_validation_observations(
        self,
        val_data: pd.DataFrame,
        center_time: datetime,
        temporal_window_minutes: int,
        numeric_cols: List[str],
    ) -> Dict[str, float]:
        """
        Average validation observations within a temporal window around center_time.
        
        Parameters
        ----------
        val_data : pd.DataFrame
            Validation data with columns: lon, lat, time, and numeric variables.
        center_time : datetime
            Center time for the window.
        temporal_window_minutes : int
            Half-width of window (window is ±temporal_window_minutes from center).
        numeric_cols : list of str
            Column names to average.
        
        Returns
        -------
        dict
            {col: averaged_value} for all numeric_cols.
        """
        from datetime import timedelta as _td

        t_min = center_time - _td(minutes=temporal_window_minutes)
        t_max = center_time + _td(minutes=temporal_window_minutes)

        # Handle timezone-aware datetimes
        if hasattr(center_time, "tzinfo") and center_time.tzinfo is not None:
            t_min = t_min.replace(tzinfo=None)
            t_max = t_max.replace(tzinfo=None)

        val_times_pd = pd.to_datetime(val_data["time"].values)
        if val_times_pd.tz is not None:
            val_times_pd = val_times_pd.tz_localize(None)

        temporal_mask = (val_times_pd >= t_min) & (val_times_pd <= t_max)
        subset = val_data[temporal_mask]

        if subset.empty:
            # No observations in window, return empty
            return {}

        result = {}
        for col in numeric_cols:
            if col in subset.columns:
                values = subset[col].values
                valid_mask = pd.notna(values)
                if np.any(valid_mask):
                    result[col] = float(np.nanmean(values[valid_mask]))

        return result


# ---------------------------------------------------------------------------
# High-level pipeline helper (step 3)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper to load RVL data on-demand for currents validation
# ---------------------------------------------------------------------------

def _load_rvl_for_collocation(
    sar_datasets: Dict[str, Any],
    base_dir: Path,
) -> None:
    """
    Load RVL (Radial Velocity Linesight) data from WV mode SAFE files and
    add to SAR datasets for currents validation.

    Creates separate dataset entries for RVL data (named `{scene}_rvl`)
    so it can be processed independently.

    Modifies sar_datasets in-place to add RVL datasets where available.

    Parameters
    ----------
    sar_datasets : dict
        Mapping of SAR scene names to Datasets (will be modified)
    base_dir : Path
        Base directory containing S1_L2_OCN/ subdirectory
    """
    from .datatree_converter import DataTreeConverter

    sar_dir = base_dir / "S1_L2_OCN"
    if not sar_dir.exists():
        return

    # Scan for WV mode SAFE directories and extract RVL
    for safe_dir in sorted(d for d in sar_dir.iterdir()
                          if d.is_dir() and d.suffix == ".SAFE"):
        safe_name = safe_dir.name.upper()
        if "WV" not in safe_name:
            continue

        try:
            rvl_ds = DataTreeConverter._extract_rvl_from_wv_safe(safe_dir)
            if rvl_ds is not None:
                # Add as a separate dataset entry with _rvl suffix
                sar_datasets[f"{safe_dir.name}_rvl"] = rvl_ds
                logger.info("Loaded RVL data for SAR scene %s", safe_dir.name)
        except Exception as e:
            logger.debug("Could not load RVL for %s: %s", safe_dir.name, e)


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


def run_collocation(
    recipe,
    datatree: "xr.DataTree",
    base_dir: Union[str, Path],
    emit_diagnostics: bool = False,
) -> Optional["xr.Dataset"]:
    """
    Run all three collocation passes between SAR and validation nodes in
    *datatree* and save the combined results to
    ``<base_dir>/collocation_results.nc``.

    Pass order
    ----------
    1. **point_vs_layer**      — moorings, buoys, tidal gauges, HF radar
    2. **trajectory_vs_layer** — ferryboxes, drifters
    3. **layer_vs_layer**      — scatterometer swaths, OSI-SAF winds

    Each validation source is auto-assigned to a pass based on its
    ``platform_type`` / ``data_type`` Dataset attribute (see
    :func:`_detect_collocation_type`).

    The collocation parameters (tolerances, interpolation method) are taken
    from ``recipe.config.collocation`` with per-source overrides applied
    from ``validation_sources[i].collocation_kwargs``.

    DataTree layout expected
    ------------------------
    - ``/sar/<scene-name>``              — one node per SAR SAFE file
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

    Returns
    -------
    xr.Dataset or None
        Dataset of collocated pairs, or None if no matches were found.
    """
    import xarray as xr
    from pathlib import Path as _Path
    from .datatree_converter import DataTreeConverter

    base_dir = _Path(base_dir)
    coll_cfg = recipe.config.collocation

    _COLLOC_CLASSES = {
        "point_vs_layer":       PointLayerCollocation,
        "trajectory_vs_layer":  TrajectoryLayerCollocation,
        "layer_vs_layer":       LayerLayerCollocation,
    }

    # Build a map of source_type to collocation_kwargs for per-source overrides
    source_type_overrides: Dict[str, Dict[str, Any]] = {}
    for src in recipe.config.validation_sources:
        if src.collocation_kwargs:
            source_type_overrides[src.source_type] = src.collocation_kwargs

    # Convert global collocation config to dict for easy merging
    global_coll_kwargs = {
        "spatial_tolerance_km": coll_cfg.spatial_tolerance_km,
        "time_tolerance_minutes": coll_cfg.time_tolerance_minutes,
        "interpolation_method": coll_cfg.interpolation_method,
        "aggregation_window_km": coll_cfg.aggregation_window_km,
        "validation_temporal_averaging_minutes": coll_cfg.validation_temporal_averaging_minutes,
        "distance_weighting": coll_cfg.distance_weighting,
        "gaussian_sigma_km": coll_cfg.gaussian_sigma_km,
        "emit_diagnostics": emit_diagnostics,
    }

    # Collect SAR nodes (one Dataset per SAR scene)
    sar_scenes: Dict[str, Any] = {}
    if "sar" in datatree.children:
        for name, node in datatree["sar"].children.items():
            sar_scenes[name] = node.to_dataset()

    if not sar_scenes:
        logger.warning("No SAR nodes found in DataTree — nothing to collocate.")
        return None

    # Load RVL data on-demand for currents validation
    _load_rvl_for_collocation(sar_scenes, base_dir)

    # Collect validation nodes (flatten up to two levels deep) and bucket
    # each by its auto-detected collocation type, tracking source metadata.
    buckets: Dict[str, Dict[str, Any]] = {t: {} for t in _COLLOC_CLASSES}
    source_metadata: Dict[str, Dict[str, Any]] = {}  # source_name -> {source_type, colloc_kwargs}

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

    total_sources = sum(len(v) for v in buckets.values())
    if total_sources == 0:
        logger.warning("No validation nodes with 'point' dimension found — nothing to collocate.")
        return None

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
            # =========== WV MODE (Point-based) ===========
            # Extract point measurements and iterate over each point
            sar_lons = sar_ds["lon"].values   # (point,)
            sar_lats = sar_ds["lat"].values   # (point,)
            sar_times = sar_ds["time"].values # (point,) as datetime64

            # Get all point variables
            sar_point_vars = {
                var: sar_ds[var].values  # (point,)
                for var in sar_ds.data_vars
                if sar_ds[var].dims == ("point",)
            }

            if not sar_point_vars:
                logger.warning("SAR node '%s' (WV mode) has no point variables — skipping.", sar_name)
                continue

            # For each SAR point, convert to 1×1 grid format and collocate
            n_points = len(sar_lons)
            logger.info("SAR node '%s' is WV mode with %d point measurements", sar_name, n_points)

            for point_idx in range(n_points):
                # Extract single point and reshape to 1×1 grid
                lon_1x1 = np.array([[sar_lons[point_idx]]])
                lat_1x1 = np.array([[sar_lats[point_idx]]])
                acq_time = pd.Timestamp(sar_times[point_idx]).to_pydatetime()
                sar_time_arr = np.array([acq_time])

                # Create 1×1 grid for each variable
                sar_data_3d: Dict[str, np.ndarray] = {
                    var: np.array([[[sar_point_vars[var][point_idx]]]])
                    for var in sar_point_vars
                }

                # Run the three passes in order
                for ctype, sources in buckets.items():
                    if not sources:
                        continue
                    for val_name, val_ds in sources.items():
                        # Apply per-source kwargs overrides
                        per_source_kwargs = source_metadata.get(val_name, {}).get("colloc_kwargs", {})
                        merged_kwargs = _merge_collocation_kwargs(global_coll_kwargs, per_source_kwargs)

                        colloc = _COLLOC_CLASSES[ctype](**merged_kwargs)

                        df = val_ds.to_dataframe().reset_index(drop=True)
                        source_label = val_ds.attrs.get("platform_type", val_name.split("/")[-1])

                        matches = colloc.collocate(
                            sar_data=sar_data_3d,
                            sar_lon=lon_1x1,
                            sar_lat=lat_1x1,
                            sar_time=sar_time_arr,
                            val_data=df,
                            val_source=source_label,
                            sar_scene_name=sar_name,
                        )
                        all_collocations.extend(matches)

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

            # Run the three passes in order
            for ctype, sources in buckets.items():
                if not sources:
                    continue
                for val_name, val_ds in sources.items():
                    # Apply per-source kwargs overrides
                    per_source_kwargs = source_metadata.get(val_name, {}).get("colloc_kwargs", {})
                    merged_kwargs = _merge_collocation_kwargs(global_coll_kwargs, per_source_kwargs)

                    colloc = _COLLOC_CLASSES[ctype](**merged_kwargs)

                    df = val_ds.to_dataframe().reset_index(drop=True)
                    source_label = val_ds.attrs.get("platform_type", val_name.split("/")[-1])

                    matches = colloc.collocate(
                        sar_data=sar_data_3d,
                        sar_lon=sar_lon,
                        sar_lat=sar_lat,
                        sar_time=sar_time_arr,
                        val_data=df,
                        val_source=source_label,
                        sar_scene_name=sar_name,
                    )
                    all_collocations.extend(matches)
                    logger.info(
                        "SAR '%s' × validation '%s' [%s]: %d match(es)",
                        sar_name, val_name, ctype, len(matches),
                    )

    if not all_collocations:
        logger.warning("Collocation complete — no matches found.")
        return None

    result_ds = DataTreeConverter.from_collocations(all_collocations)

    out_path = base_dir / "collocation_results.nc"
    result_ds.to_netcdf(out_path)
    logger.info(
        "Collocation results saved to %s (%d matches total)",
        out_path, len(all_collocations),
    )

    # Step 4a: optional SAR patch extraction
    patch_size = getattr(coll_cfg, "patch_size", 0)
    if patch_size and patch_size > 0:
        from .patch_extractor import run_patch_extraction
        run_patch_extraction(result_ds, datatree, patch_size, base_dir)

    return result_ds


# ---------------------------------------------------------------------------
# 2. Trajectory vs. Layer
# ---------------------------------------------------------------------------

class TrajectoryLayerCollocation(PointLayerCollocation):
    """
    Match a moving trajectory (ferrybox, drifter) to a SAR layer.

    Each observation is matched independently: its individual ``lon``, ``lat``,
    and ``time`` are checked against the SAR grid.  This is equivalent to
    ``PointLayerCollocation`` but produces results labelled
    ``collocation_type="trajectory_vs_layer"``.

    Typical use cases: ferrybox transects, drifting buoys.
    """

    collocation_type: str = "trajectory_vs_layer"


# ---------------------------------------------------------------------------
# 3. Layer vs. Layer
# ---------------------------------------------------------------------------

class LayerLayerCollocation(PointLayerCollocation):
    """
    Match a gridded validation product (e.g. ASCAT scatterometer swath) to a SAR
    layer by aggregating SAR pixels within each scatterometer wind vector cell.

    Grid-aware aggregation approach
    --------------------------------
    Unlike the parent ``PointLayerCollocation``, this class:

    1. **Detects scatterometer grid structure** — reconstructs spatial cells from 
       flattened scatterometer data using adaptive clustering based on point spacing.
    
    2. **SAR Aggregation**: For each scatterometer cell, finds all SAR grid cells 
       within ``aggregation_window_km`` (e.g., 12.5 km for ASCAT) and computes a 
       distance-weighted average of SAR variables.
    
    3. **Validation Aggregation**: Temporally averages scatterometer observations 
       within ±``validation_temporal_averaging_minutes`` around each cell center.
    
    4. **Output**: Single ``CollocatedPoint`` per scatterometer cell with aggregated 
       SAR mean vs. aggregated scatterometer mean.

    Typical use cases: ASCAT scatterometer swaths (12.5×12.5 km cells), OSI-SAF wind products.

    Collocation parameters
    ----------------------
    See parent class, but LayerLayerCollocation provides different defaults optimized 
    for scatterometer-SAR comparison:

    - ``aggregation_window_km`` : 12.5 km (ASCAT scatterometer cell size)
    - ``time_tolerance_minutes`` : 180 min (±3 hours, per hal-04202202)
    - ``distance_weighting`` : "equal" (uniform weights across regular grid cells)
    - ``validation_temporal_averaging_minutes`` : 60 min (±1 hour window)

    Reference
    ---------
    Collocation methodology follows:
    - Abderrahim et al. (2019) — paper hal-04202202
      "Validation of Sentinel-1 wind products against scatterometer measurements"
    - Spatial tolerance: 12.5 km (ASCAT cell size)
    - Temporal tolerance: 180 minutes (3-hour match window)
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
        """
        super().__init__(
            spatial_tolerance_km=spatial_tolerance_km,
            time_tolerance_minutes=time_tolerance_minutes,
            interpolation_method=interpolation_method,
            aggregation_window_km=aggregation_window_km,
            validation_temporal_averaging_minutes=validation_temporal_averaging_minutes,
            distance_weighting=distance_weighting,
            gaussian_sigma_km=gaussian_sigma_km,
        )
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

    def _infer_scatterometer_grid(
        self,
        val_data: pd.DataFrame,
        linkage_method: str = "complete",
        distance_threshold_percentile: float = 60.0,
    ) -> Dict[int, List[int]]:
        """
        Reconstruct scatterometer grid structure using hierarchical clustering.

        ASCAT scatterometer data is typically delivered as a 2D swath (e.g., 3168×82 cells)
        but converted to a flat ``point`` dimension. This method reconstructs the grid
        by detecting natural spatial clusters using hierarchical clustering on
        all scatterometer points.

        Algorithm
        ----------
        1. Compute pairwise Haversine distances between all scatterometer points.
        2. Run hierarchical clustering (complete linkage) on the distance matrix.
        3. Estimate cell size from distance histogram (typically 12.5 km for ASCAT).
        4. Cut dendrogram at threshold ≈ 0.6 × median_distance to group nearby points into cells.
        5. Return dictionary mapping cell ID → list of point row indices.

        Parameters
        ----------
        val_data : pd.DataFrame
            Validation data with columns ``lon``, ``lat``, ``time``, and numeric variables.
        linkage_method : str
            Hierarchical clustering method: "complete", "average", "single", "ward".
        distance_threshold_percentile : float
            Percentile of distance distribution to use as clustering threshold (60th percentile default).

        Returns
        -------
        dict
            Mapping: cell_id (int) → list of point row indices (list of int).
            If clustering fails, returns {0: list(range(len(val_data)))} (all points in one cell).
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist

        n_points = len(val_data)
        if n_points < 2:
            # Single point or empty → treat as one cell
            if self.emit_diagnostics:
                logger.info("[GRID] Received %d point(s); treating as single cell", n_points)
            return {0: list(range(n_points))}

        # Extract coordinates
        lons = val_data["lon"].values
        lats = val_data["lat"].values

        try:
            # Compute pairwise Haversine distances
            if self.emit_diagnostics:
                logger.info("[GRID] Computing pairwise distances for %d scatterometer points...", n_points)
            distances = []
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    d = _haversine_distance(lons[i], lats[i], lons[j], lats[j])
                    distances.append(d)

            if not distances:
                if self.emit_diagnostics:
                    logger.info("[GRID] No distances computed; treating as single cell")
                logger.warning("No distances computed; falling back to single cell.")
                return {0: list(range(n_points))}

            distances = np.array(distances)

            # Estimate threshold from distance histogram
            # ASCAT cell size is ~12.5 km; use percentile of distances as threshold
            threshold = np.percentile(distances, distance_threshold_percentile)
            if self.emit_diagnostics:
                logger.info(
                    "[GRID] Distance distribution: min=%.2f km, max=%.2f km, median=%.2f km, "
                    "%d-percentile=%.2f km (threshold)",
                    distances.min(), distances.max(), np.median(distances), 
                    int(distance_threshold_percentile), threshold,
                )

            # Run hierarchical clustering
            if self.emit_diagnostics:
                logger.info("[GRID] Running hierarchical clustering (linkage='%s')...", linkage_method)
            z = linkage(distances, method=linkage_method)
            cluster_ids = fcluster(z, threshold, criterion="distance")

            # Map cluster IDs to point indices
            cells: Dict[int, List[int]] = {}
            for point_idx, cluster_id in enumerate(cluster_ids):
                if cluster_id not in cells:
                    cells[cluster_id] = []
                cells[cluster_id].append(point_idx)

            if self.emit_diagnostics:
                cell_sizes = [len(indices) for indices in cells.values()]
                logger.info(
                    "[GRID] Grid inference SUCCESS: detected %d cells from %d points. "
                    "Cell sizes: min=%d, max=%d, mean=%.1f",
                    len(cells), n_points, min(cell_sizes), max(cell_sizes), np.mean(cell_sizes),
                )
            else:
                logger.info(
                    "Grid inference: detected %d cells from %d scatterometer points",
                    len(cells), n_points,
                )
            return cells

        except Exception as e:
            if self.emit_diagnostics:
                logger.info("[GRID] Grid inference FAILED: %s; using all points as single cell.", type(e).__name__)
            logger.warning("Grid inference failed (%s); using all points as single cell.", e)
            return {0: list(range(n_points))}

    def collocate(
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
        Match scatterometer cells to SAR grid using spatial/temporal aggregation.

        Algorithm
        ---------
        1. Infer scatterometer grid structure (group points into cells).
        2. For each scatterometer cell:
           a. Find all SAR cells within aggregation_window_km.
           b. Average SAR variables (distance-weighted or equal).
           c. Temporally average scatterometer observations within ±validation_temporal_averaging_minutes.
           d. Create CollocatedPoint with aggregated SAR vs. aggregated scatterometer.

        Parameters
        ----------
        sar_data : dict
            SAR variables as 3-D arrays with shape ``(time, y, x)``.
        sar_lon, sar_lat : np.ndarray
            SAR coordinate grids, shape ``(y, x)``.
        sar_time : array-like
            SAR acquisition times, shape ``(time,)``.
        val_data : pd.DataFrame
            Scatterometer data with columns ``lon``, ``lat``, ``time``, and
            any number of variable columns.
        val_source : str
            Label for validation source (e.g. ``"scatterometer"``).
        sar_scene_name : str
            Name of SAR scene node in DataTree.

        Returns
        -------
        list[CollocatedPoint]
            List of collocated matches (one per scatterometer cell).
        """
        from datetime import timedelta as _td

        sar_times = _to_datetime_array(sar_time)
        collocations: List[CollocatedPoint] = []

        # Pre-filter: spatial and temporal bounds
        deg_buf = self.aggregation_window_km / 55.0
        lon_min = float(sar_lon.min()) - deg_buf
        lon_max = float(sar_lon.max()) + deg_buf
        lat_min = float(sar_lat.min()) - deg_buf
        lat_max = float(sar_lat.max()) + deg_buf

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

        # **NEW**: Infer scatterometer grid structure
        cells = self._infer_scatterometer_grid(val_data_filtered)
        logger.info("LayerLayerCollocation: processing %d scatterometer cells", len(cells))

        # Identify numeric columns
        numeric_cols = [
            col for col in val_data_filtered.columns
            if col not in {"lon", "lat", "time", "platform_id"} and
            pd.api.types.is_numeric_dtype(val_data_filtered[col])
        ]

        # Process each scatterometer cell
        matches_found = 0
        rejected_spatial = 0
        rejected_temporal = 0
        rejected_no_data = 0
        
        for cell_id, point_indices in cells.items():
            # Get scatterometer observations in this cell
            cell_subset = val_data_filtered.iloc[point_indices]

            # Compute cell center (spatial mean)
            c_lon = float(cell_subset["lon"].mean())
            c_lat = float(cell_subset["lat"].mean())
            # Use first observation time as reference (all should be close due to single pass)
            c_time = _to_datetime_array([cell_subset["time"].iloc[0]])[0]

            if self.emit_diagnostics:
                logger.info(
                    "[CELL %d] Center: (%.3f°, %.3f°), Time: %s, Observations: %d",
                    cell_id, c_lon, c_lat, c_time.isoformat(), len(cell_subset),
                )

            # Find nearby SAR cells within aggregation window
            nearby_cells_with_dist = self._nearby_cells_with_distances(
                c_lon, c_lat, sar_lon, sar_lat, self.aggregation_window_km
            )

            if not nearby_cells_with_dist:
                rejected_spatial += 1
                if self.emit_diagnostics:
                    logger.info(
                        "[CELL %d] REJECTED: No SAR cells within %.1f km",
                        cell_id, self.aggregation_window_km,
                    )
                logger.debug(
                    "No SAR cells within %.1f km of scatterometer cell (%.2f, %.2f)",
                    self.aggregation_window_km, c_lon, c_lat,
                )
                continue

            if self.emit_diagnostics:
                distances_km = [d for _, _, d in nearby_cells_with_dist]
                logger.info(
                    "[CELL %d] Found %d SAR cells within %.1f km: distances=[%.2f, %.2f, ..., %.2f] km",
                    cell_id, len(nearby_cells_with_dist), self.aggregation_window_km,
                    min(distances_km), np.median(distances_km), max(distances_km),
                )

            # Find nearby SAR times
            nearby_t_idx = self._nearby_times(c_time, sar_times)
            if not nearby_t_idx:
                rejected_temporal += 1
                if self.emit_diagnostics:
                    logger.info(
                        "[CELL %d] REJECTED: No SAR times within ±%d minutes",
                        cell_id, self.time_tolerance_minutes,
                    )
                logger.debug(
                    "No SAR times within %d minutes of scatterometer cell time",
                    self.time_tolerance_minutes,
                )
                continue

            if self.emit_diagnostics:
                sar_times_nearby = [sar_times[idx] for idx in nearby_t_idx]
                logger.info(
                    "[CELL %d] Found %d SAR times within ±%d minutes: %s",
                    cell_id, len(nearby_t_idx), self.time_tolerance_minutes,
                    ", ".join(t.isoformat() for t in sar_times_nearby),
                )

            # Process each nearby SAR time
            for t_idx in nearby_t_idx:
                # Compute aggregated SAR values over nearby cells
                sar_aggregated = self._compute_aggregated_sar_value(
                    nearby_cells_with_dist,
                    sar_data,
                    t_idx,
                    weighting_method=self.distance_weighting,
                    sigma_km=self.gaussian_sigma_km,
                    agg_window_km=self.aggregation_window_km,
                )

                if not sar_aggregated:
                    rejected_no_data += 1
                    if self.emit_diagnostics:
                        logger.info("[CELL %d] t_idx=%d: REJECTED: No valid SAR values", cell_id, t_idx)
                    logger.debug("No valid SAR values at t_idx=%d", t_idx)
                    continue

                # Aggregate scatterometer observations temporally within window
                val_aggregated = self._average_validation_observations(
                    cell_subset,
                    c_time,
                    self.validation_temporal_averaging_minutes,
                    numeric_cols,
                )

                # If no temporal aggregation, use all values in cell averaged
                if not val_aggregated:
                    val_aggregated = {}
                    for col in numeric_cols:
                        values = cell_subset[col].values
                        valid_mask = pd.notna(values)
                        if np.any(valid_mask):
                            val_aggregated[col] = float(np.nanmean(values[valid_mask]))

                if not val_aggregated:
                    rejected_no_data += 1
                    if self.emit_diagnostics:
                        logger.info("[CELL %d] t_idx=%d: REJECTED: No valid scatterometer values", cell_id, t_idx)
                    logger.debug("No valid scatterometer values in cell")
                    continue

                # Use closest SAR cell for position/indices
                closest_idx = np.argmin([d for _, _, d in nearby_cells_with_dist])
                y_idx, x_idx, _ = nearby_cells_with_dist[closest_idx]
                s_lon = float(sar_lon[y_idx, x_idx])
                s_lat = float(sar_lat[y_idx, x_idx])
                s_time = sar_times[t_idx]

                # Compute distances
                spatial_dist = _haversine_distance(c_lon, c_lat, s_lon, s_lat)
                temporal_dist = abs((c_time - s_time).total_seconds() / 60.0)

                if self.emit_diagnostics:
                    logger.info(
                        "[CELL %d] t_idx=%d: MATCHED! Spatial dist=%.2f km, Temporal dist=%.1f min",
                        cell_id, t_idx, spatial_dist, temporal_dist,
                    )

                matches_found += 1

                # Create CollocatedPoint
                collocations.append(
                    CollocatedPoint(
                        sar_lon=s_lon,
                        sar_lat=s_lat,
                        sar_time=s_time,
                        sar_data=sar_aggregated,
                        val_lon=c_lon,
                        val_lat=c_lat,
                        val_time=c_time,
                        val_data=val_aggregated,
                        spatial_distance_km=spatial_dist,
                        temporal_distance_minutes=temporal_dist,
                        val_source=val_source,
                        val_id=None,  # Scatterometer cells don't have IDs
                        collocation_type=self.collocation_type,
                        sar_y_idx=y_idx,
                        sar_x_idx=x_idx,
                        sar_scene_name=sar_scene_name,
                    )
                )

        if self.emit_diagnostics:
            logger.info(
                "[SUMMARY] LayerLayerCollocation: %d matches from %d cells (spatial_tol=%.1f km, "
                "temporal_tol=%d min). Rejected: %d spatial, %d temporal, %d no-data.",
                len(collocations), len(cells), self.spatial_tolerance_km, self.time_tolerance_minutes,
                rejected_spatial, rejected_temporal, rejected_no_data,
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
            "%s: found %d matches from %d scatterometer cells (source=%s)",
            self.__class__.__name__, len(collocations), len(cells), val_source,
        )
        return collocations
