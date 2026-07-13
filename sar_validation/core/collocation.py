"""
Collocation algorithms — step 3 of the validation pipeline.

Two collocation geometries are supported:

1. ``PointLayerCollocation`` — fixed / slow-moving point vs. SAR grid  ✅
2. ``LayerLayerCollocation`` — gridded product vs. SAR grid            🚧
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
    "LayerLayerCollocation",
    "run_collocation",
    "_detect_collocation_type",
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

# data_type attribute values that indicate a gridded layer source
LAYER_DATA_TYPES = {"scatterometer", "altimeter", "hf_radar", "radiometer"}
# path-fragment fallbacks when attributes are absent
LAYER_SOURCE_PATHS = {"osi_saf_winds", "scatterometer", "altimeter", "hf_radar", "radiometer"}


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
            if col not in {"lon", "lat", "time", "platform_id", "platform_type"} and
            pd.api.types.is_numeric_dtype(val_data_filtered[col])
        ]

        # Time-sorted view used to forward-fill a NaN reading: when a
        # validation observation is missing a value for some column, look at
        # later observations (still within ``time_tolerance_minutes`` of the
        # original reading) for the next one that has a valid value.
        val_data_sorted = val_data_filtered.sort_values("time").reset_index(drop=True)
        sorted_times_ns = pd.to_datetime(val_data_sorted["time"].values)
        if sorted_times_ns.tz is not None:
            sorted_times_ns = sorted_times_ns.tz_localize(None)
        sorted_times_ns = sorted_times_ns.values.astype("datetime64[ns]")
        n_sorted = len(val_data_sorted)

        # Per column: index of the first row at-or-after each position that
        # holds a valid (non-NaN) value, so each lookup is a binary search
        # instead of an O(n) scan.
        _next_valid_idx: Dict[str, np.ndarray] = {}

        def _next_valid_value(col: str, after_time) -> Optional[float]:
            if col not in _next_valid_idx:
                valid = val_data_sorted[col].notna().values
                idx = np.where(valid, np.arange(n_sorted), n_sorted)
                _next_valid_idx[col] = np.minimum.accumulate(idx[::-1])[::-1]
            after_ns = np.datetime64(pd.Timestamp(after_time))
            pos = int(np.searchsorted(sorted_times_ns, after_ns, side="right"))
            if pos >= n_sorted:
                return None
            j = int(_next_valid_idx[col][pos])
            if j >= n_sorted:
                return None
            gap_min = (sorted_times_ns[j] - after_ns) / np.timedelta64(1, "m")
            if gap_min > self.time_tolerance_minutes:
                return None
            return float(val_data_sorted[col].values[j])

        # KD-tree over the SAR grid cells (unit-sphere Cartesian coordinates,
        # see _lonlat_to_unit_xyz) built once per scene: each validation
        # point then queries only its local neighbourhood instead of
        # computing a Haversine distance to every grid cell.
        grid_tree = self._build_grid_tree(sar_lon, sar_lat)

        # Process each validation observation
        for idx, val_row in val_data_filtered.iterrows():
            v_lon = float(val_row["lon"])
            v_lat = float(val_row["lat"])
            v_time = _to_datetime_array([val_row["time"]])[0]

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
                    filled_val = _next_valid_value(col, v_time)
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
                            heading_deg = np.nanmean(valid_headings)
                            heading_rad = np.radians(float(heading_deg) - 90.0)
                            ewct = float(val_aggregated["EWCT"])
                            nsct = float(val_aggregated["NSCT"])
                            radial_vel = ewct * np.cos(heading_rad) + nsct * np.sin(heading_rad)
                            val_aggregated["rvlRadVel_projection"] = radial_vel
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
                        val_source=point_val_source,
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
    SAR-point-anchored collocation for sparse WV-mode OSW imagettes.

    Each Sentinel-1 WV imagette is a single point representing a ~20×20 km
    footprint, and consecutive imagettes are ~200 km apart. Rather than
    requiring validation data within a few km of the imagette *centre* (as the
    grid-oriented matchers do when a WV point is faked into a 1×1 grid), this
    gathers every validation observation within ``footprint_radius_km`` and
    ``time_tolerance_minutes`` of each imagette and aggregates them into a
    single match anchored on the SAR point.

    Parameters
    ----------
    sar_lons, sar_lats, sar_times : np.ndarray
        Per-imagette coordinates/times for one SAR scene, shape ``(n_points,)``.
    sar_point_vars : dict
        SAR variables as ``(n_points,)`` arrays (e.g. ``{"oswHs": ...}``).
    val_data : pd.DataFrame
        Validation observations with ``lon``, ``lat``, ``time`` and any number
        of variable columns.
    footprint_radius_km : float
        Search radius around each imagette (≈ footprint half-diagonal).
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
        One match per imagette that had at least one contributing observation.
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

        # SAR variables for this imagette (skip if all NaN)
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
        "WV-point collocation [%s]: %d match(es) from %d imagette(s) (source=%s)",
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
    Run both collocation passes between SAR and validation nodes in
    *datatree* and save the combined results to
    ``<base_dir>/collocation_results<filename_suffix>.nc``.

    Pass order
    ----------
    1. **point_vs_layer** — moorings, buoys, tidal gauges, HF radar, ferryboxes, drifters
    2. **layer_vs_layer** — scatterometer swaths, OSI-SAF winds

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
    import xarray as xr
    from pathlib import Path as _Path
    from .datatree_converter import DataTreeConverter

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
    
    # Layer-vs-layer specs: start from the built-in defaults (so recipes that
    # declare no layer_vs_layer section at all still get sensible per-source
    # aggregation windows), then let any recipe-level overrides win per-key.
    from .recipe import DEFAULT_LAYER_TYPE_SPECS
    layer_vs_layer_specs = dict(DEFAULT_LAYER_TYPE_SPECS)
    if coll_cfg.layer_vs_layer is not None:
        layer_vs_layer_specs.update(coll_cfg.layer_vs_layer.layer_type_specs)

    # Collect SAR nodes (one Dataset per SAR scene)
    sar_scenes: Dict[str, Any] = {}
    if "sar" in datatree.children:
        for name, node in datatree["sar"].children.items():
            sar_scenes[name] = node.to_dataset()

    if not sar_scenes:
        logger.warning("No SAR nodes found in DataTree — nothing to collocate.")
        return None

    # Load RVL (radial velocity) data on-demand — only for currents recipes.
    # RVL is the currents observable; loading it for a wind/waves recipe would
    # add spurious {scene}_rvl SAR nodes that get collocated against the
    # in-situ/altimeter data instead of the intended OWI/OSW measurement.
    if str(getattr(recipe.config, "variable", "")).lower() == "currents":
        _load_rvl_for_collocation(sar_scenes, base_dir)

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

    total_sources = sum(len(v) for v in buckets.values())
    if total_sources == 0:
        logger.warning("No validation nodes with 'point' dimension found — nothing to collocate.")
        return None

    for sources in buckets.values():
        for val_name, val_ds in sources.items():
            val_dfs[val_name] = val_ds.to_dataframe().reset_index(drop=True)

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
            # Each imagette is a single point standing for a ~20×20 km
            # footprint, and imagettes are ~200 km apart — far too sparse to
            # match by requiring validation data within a few km of the point
            # centre. Instead, anchor on each imagette and aggregate every
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
            logger.info("SAR node '%s' is WV mode with %d imagette point(s)", sar_name, n_points)

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
                        # imagette with distance-weighted aggregation and the
                        # layer's own time tolerance. Resolve the layer type
                        # exactly as the grid path does.
                        layer_type = val_ds.attrs.get("data_type", "").lower()
                        if not layer_type:
                            path_parts = val_name.lower().split("/")
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
                        if layer_type == "altimeter":
                            freq = val_ds.attrs.get("frequency", "1hz").lower()
                            layer_type = f"altimeter_{freq}"
                        elif layer_type == "radiometer":
                            # Per-sensor specs (radiometer_amsr2, …); all RSS
                            # products share the 0.25° grid but stay individually
                            # tunable. Falls back to bare 'radiometer' if unset.
                            sensor = val_ds.attrs.get("sensor", "").lower()
                            if sensor and f"radiometer_{sensor}" in layer_vs_layer_specs:
                                layer_type = f"radiometer_{sensor}"
                        if layer_type in layer_vs_layer_specs:
                            merged_kwargs.update(layer_vs_layer_specs[layer_type])
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

                    # For layer_vs_layer, apply layer-type-specific specs from recipe
                    if ctype == "layer_vs_layer" and layer_vs_layer_specs:
                        layer_type = val_ds.attrs.get("data_type", "").lower()
                        if not layer_type:
                            # Fallback: infer from path (e.g., "osi_saf_winds/...", "scatterometer", "altimeter")
                            path_parts = val_name.lower().split("/")
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

                        if layer_type == "altimeter":
                            # Altimeter's aggregation window depends on
                            # along-track resolution, which differs 5x
                            # between the 1 Hz and 5 Hz products.
                            freq = val_ds.attrs.get("frequency", "1hz").lower()
                            layer_type = f"altimeter_{freq}"
                        elif layer_type == "radiometer":
                            # Per-sensor specs (radiometer_amsr2, …). All RSS
                            # radiometers share the 0.25° grid, so the specs
                            # default alike, but each stays tunable. Falls back
                            # to bare 'radiometer' when the sensor is unknown.
                            sensor = val_ds.attrs.get("sensor", "").lower()
                            if sensor and f"radiometer_{sensor}" in layer_vs_layer_specs:
                                layer_type = f"radiometer_{sensor}"

                        if layer_type in layer_vs_layer_specs:
                            merged_kwargs.update(layer_vs_layer_specs[layer_type])
                            logger.info(
                                "Applying layer_vs_layer specs for '%s': %s",
                                layer_type, layer_vs_layer_specs[layer_type],
                            )
                        
                        # Add layer-vs-layer collocation method
                        merged_kwargs["method"] = layer_vs_layer_collocation_method

                    colloc = _COLLOC_CLASSES[ctype](**merged_kwargs)

                    df = val_dfs[val_name]
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
    Match a gridded validation product (e.g. ASCAT scatterometer swath) to a SAR
    layer by aggregating SAR pixels within each scatterometer wind vector cell.

    Aggregation approach
    --------------------
    ASCAT/OSI-SAF scatterometer products are delivered pre-gridded — one
    observation per wind-vector cell (WVC), already ~12.5×12.5 km — so each
    scatterometer point already *is* its own cell; no spatial re-clustering
    is needed. ``cell-averaging`` therefore reuses the parent
    ``PointLayerCollocation.collocate()`` algorithm directly, per
    scatterometer point:

    1. **SAR Aggregation**: For each scatterometer point, finds all SAR grid
       cells within ``aggregation_window_km`` (e.g., 12.5 km for ASCAT) and
       computes a distance-weighted average of SAR variables.

    2. **Output**: One ``CollocatedPoint`` per scatterometer point (per
       matching SAR time), with the aggregated SAR mean vs. the point's own
       raw value.

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
        Match each individual SAR pixel to the closest scatterometer point (individual method).

        For each SAR grid cell at each time step:
        1. Find closest scatterometer point (vectorized nearest-neighbour search
           via a KD-tree over unit-sphere Cartesian coordinates)
        2. Check spatial tolerance (within spatial_tolerance_km)
        3. Check temporal match (within time_tolerance_minutes)
        4. Create CollocatedPoint with SAR as anchor, scatterometer as matched value
        5. Scatterometer points can be reused across multiple SAR cells

        Returns
        -------
        list[CollocatedPoint]
            List of collocated matches (one per matched SAR cell).
        """
        from datetime import timedelta as _td
        from scipy.spatial import cKDTree

        sar_times = _to_datetime_array(sar_time)
        collocations: List[CollocatedPoint] = []

        # Pre-filter scatterometer data: spatial and temporal bounds
        deg_buf = self.spatial_tolerance_km / 55.0
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

        # Build a KD-tree over scatterometer points in unit-sphere Cartesian
        # coordinates once: Euclidean nearest-neighbour there is equivalent to
        # great-circle nearest-neighbour, so per-time-step matching becomes a
        # single vectorized query instead of an O(pixels x scat_points)
        # Python-level Haversine loop.
        R = 6371.0
        scat_tree = cKDTree(_lonlat_to_unit_xyz(scat_lons, scat_lats))

        sar_grid_y, sar_grid_x = sar_lon.shape
        sar_lon_flat = sar_lon.ravel()
        sar_lat_flat = sar_lat.ravel()
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
            has_data_mask = ~np.all(np.isnan(values_stack), axis=0)
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
    ) -> List[CollocatedPoint]:
        """
        Match scatterometer to SAR using selected collocation method.

        Dispatches to either individual point-to-point or cell-averaging methods
        based on self.method setting.

        Parameters
        ----------
        sar_data : dict
            SAR variables as 3-D arrays with shape ``(time, y, x)``.
        sar_lon, sar_lat : np.ndarray
            SAR coordinate grids, shape ``(y, x)``.
        sar_time : array-like
            SAR acquisition times, shape ``(time,)``.
        val_data : pd.DataFrame
            Scatterometer data with columns ``lon``, ``lat``, ``time``, and variables.
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
                sar_data, sar_lon, sar_lat, sar_time, val_data, val_source, sar_scene_name
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
    ) -> List[CollocatedPoint]:
        """
        Match scatterometer points to SAR grid using spatial aggregation
        (cell-averaging method).

        ASCAT/OSI-SAF scatterometer products are delivered pre-gridded — one
        observation per wind-vector cell — so each scatterometer point
        already *is* its own cell; no clustering is needed. This reuses
        ``PointLayerCollocation.collocate()`` directly: for each
        scatterometer point, find all SAR cells within
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
            Scatterometer data with columns ``lon``, ``lat``, ``time``, and
            any number of variable columns.
        val_source : str
            Label for validation source (e.g. ``"scatterometer"``).
        sar_scene_name : str
            Name of SAR scene node in DataTree.

        Returns
        -------
        list[CollocatedPoint]
            List of collocated matches (one per scatterometer point, per
            matching SAR time).
        """
        return PointLayerCollocation.collocate(
            self, sar_data, sar_lon, sar_lat, sar_time,
            val_data, val_source, sar_scene_name,
        )
