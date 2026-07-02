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
        Standard deviation (km) for Gaussian weighting. Default: 2.0 km.
    """

    #: Collocation type label stored on each CollocatedPoint result.
    collocation_type: str = "point_vs_layer"

    def __init__(
        self,
        spatial_tolerance_km: float = 50.0,
        time_tolerance_minutes: int = 60,
        interpolation_method: str = "nearest",
        aggregation_window_km: float = 5.0,
        validation_temporal_averaging_minutes: int = 30,
        distance_weighting: str = "gaussian",
        gaussian_sigma_km: float = 2.0,
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


def run_collocation(
    recipe,
    datatree: "xr.DataTree",
    base_dir: Union[str, Path],
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
    from ``recipe.config.collocation``.

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
    # each by its auto-detected collocation type.
    buckets: Dict[str, Dict[str, Any]] = {t: {} for t in _COLLOC_CLASSES}

    if "validation" in datatree.children:
        for name, node in datatree["validation"].children.items():
            ds = node.to_dataset()
            if "point" in ds.dims and len(ds.data_vars) > 0:
                ctype = _detect_collocation_type(ds, name)
                buckets[ctype][name] = ds
            # One level deeper (e.g. validation/osi_saf_winds/<file>)
            for subname, subnode in node.children.items():
                sub_ds = subnode.to_dataset()
                if "point" in sub_ds.dims and len(sub_ds.data_vars) > 0:
                    path = f"{name}/{subname}"
                    ctype = _detect_collocation_type(sub_ds, path)
                    buckets[ctype][path] = sub_ds

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
                    colloc = _COLLOC_CLASSES[ctype](
                        spatial_tolerance_km=coll_cfg.spatial_tolerance_km,
                        time_tolerance_minutes=coll_cfg.time_tolerance_minutes,
                        interpolation_method=coll_cfg.interpolation_method,
                        aggregation_window_km=coll_cfg.aggregation_window_km,
                        validation_temporal_averaging_minutes=coll_cfg.validation_temporal_averaging_minutes,
                        distance_weighting=coll_cfg.distance_weighting,
                        gaussian_sigma_km=coll_cfg.gaussian_sigma_km,
                    )
                    for val_name, val_ds in sources.items():
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
                colloc = _COLLOC_CLASSES[ctype](
                    spatial_tolerance_km=coll_cfg.spatial_tolerance_km,
                    time_tolerance_minutes=coll_cfg.time_tolerance_minutes,
                    interpolation_method=coll_cfg.interpolation_method,
                    aggregation_window_km=coll_cfg.aggregation_window_km,
                    validation_temporal_averaging_minutes=coll_cfg.validation_temporal_averaging_minutes,
                    distance_weighting=coll_cfg.distance_weighting,
                    gaussian_sigma_km=coll_cfg.gaussian_sigma_km,
                )
                for val_name, val_ds in sources.items():
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
    Match a gridded validation product (e.g. scatterometer swath) to a SAR
    layer cell-by-cell.

    The validation dataset is flattened to individual (lon, lat, time)
    observations before matching — each grid cell is treated as an
    independent point and checked against the SAR grid with the same
    Haversine + temporal tolerance logic as ``PointLayerCollocation``.

    Results are labelled ``collocation_type="layer_vs_layer"``.

    Typical use cases: ASCAT scatterometer swaths, OSI-SAF wind products.
    """

    collocation_type: str = "layer_vs_layer"
