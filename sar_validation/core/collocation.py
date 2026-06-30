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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "CollocatedPoint",
    "PointLayerCollocation",
    "TrajectoryLayerCollocation",
    "LayerLayerCollocation",
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
# 1. Point vs. Layer
# ---------------------------------------------------------------------------

class PointLayerCollocation:
    """
    Match fixed-point (or slowly-moving) validation observations with a
    gridded SAR layer.

    Typical use cases: moorings, in-situ buoys.

    Parameters
    ----------
    spatial_tolerance_km : float
        Maximum great-circle distance between the validation point and the
        nearest SAR grid cell to accept a match.
    time_tolerance_minutes : int
        Maximum absolute time difference.
    interpolation_method : str
        How to extract the SAR value at the validation location:
        - ``"nearest"``  — use the closest grid cell (default)
        - ``"linear"``   — bilinear interpolation (TODO)
        - ``"cubic"``    — bicubic interpolation (TODO)
    """

    def __init__(
        self,
        spatial_tolerance_km: float = 50.0,
        time_tolerance_minutes: int = 60,
        interpolation_method: str = "nearest",
    ) -> None:
        self.spatial_tolerance_km = spatial_tolerance_km
        self.time_tolerance_minutes = time_tolerance_minutes
        self.interpolation_method = interpolation_method

        if interpolation_method not in ("nearest", "linear", "cubic"):
            raise ValueError(
                f"Unknown interpolation_method '{interpolation_method}'. "
                "Use 'nearest', 'linear', or 'cubic'."
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
    ) -> List[CollocatedPoint]:
        """
        Match validation point observations to the SAR grid.

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
            any number of variable columns.  An optional ``platform_id``
            column is used to populate ``CollocatedPoint.val_id``.
        val_source : str
            Label for the validation source (e.g. ``"buoy"``).

        Returns
        -------
        list[CollocatedPoint]
        """
        sar_times = _to_datetime_array(sar_time)
        collocations: List[CollocatedPoint] = []

        for _, val_row in val_data.iterrows():
            v_lon = float(val_row["lon"])
            v_lat = float(val_row["lat"])
            v_time = _to_datetime_array([val_row["time"]])[0]

            nearby_cells = self._nearby_cells(v_lon, v_lat, sar_lon, sar_lat)
            if not nearby_cells:
                continue

            nearby_t_idx = self._nearby_times(v_time, sar_times)
            if len(nearby_t_idx) == 0:
                continue

            for t_idx in nearby_t_idx:
                for y_idx, x_idx in nearby_cells:
                    # Extract SAR values at this pixel
                    sar_point: Dict[str, float] = {}
                    for var_name, var_arr in sar_data.items():
                        value = var_arr[t_idx, y_idx, x_idx]
                        if not np.isnan(value):
                            sar_point[var_name] = float(value)

                    if not sar_point:
                        continue   # all NaN at this pixel

                    s_lon = float(sar_lon[y_idx, x_idx])
                    s_lat = float(sar_lat[y_idx, x_idx])
                    s_time = sar_times[t_idx]

                    spatial_dist  = _haversine_distance(v_lon, v_lat, s_lon, s_lat)
                    temporal_dist = abs(
                        (v_time - s_time).total_seconds() / 60.0
                    )

                    if (
                        spatial_dist  <= self.spatial_tolerance_km
                        and temporal_dist <= self.time_tolerance_minutes
                    ):
                        val_point = {
                            col: float(val_row[col])
                            for col in val_data.columns
                            if col not in {"lon", "lat", "time", "platform_id"}
                            and pd.notna(val_row[col])
                        }
                        collocations.append(
                            CollocatedPoint(
                                sar_lon=s_lon,
                                sar_lat=s_lat,
                                sar_time=s_time,
                                sar_data=sar_point,
                                val_lon=v_lon,
                                val_lat=v_lat,
                                val_time=v_time,
                                val_data=val_point,
                                spatial_distance_km=spatial_dist,
                                temporal_distance_minutes=temporal_dist,
                                val_source=val_source,
                                val_id=val_row.get("platform_id"),
                            )
                        )

        logger.info(
            "PointLayerCollocation: found %d matches (source=%s)",
            len(collocations), val_source,
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


# ---------------------------------------------------------------------------
# 2. Trajectory vs. Layer  (stub)
# ---------------------------------------------------------------------------

class TrajectoryLayerCollocation:
    """
    Match a moving trajectory (ferrybox, drifter) to a SAR layer.

    Not yet implemented.  The collocation must account for the platform
    moving during the SAR overpass; the matching logic should interpolate
    the trajectory to the SAR acquisition time.
    """

    def collocate(self, *args, **kwargs):
        raise NotImplementedError(
            "TrajectoryLayerCollocation is not yet implemented."
        )


# ---------------------------------------------------------------------------
# 3. Layer vs. Layer  (stub)
# ---------------------------------------------------------------------------

class LayerLayerCollocation:
    """
    Match two gridded products (e.g. scatterometer vs. SAR).

    Not yet implemented.  Key considerations:
      - Resample both grids to a common resolution before matching.
      - The temporal tolerance applies to the centre of each grid's
        acquisition window.
      - Store per-cell spatial and temporal offsets.
    """

    def collocate(self, *args, **kwargs):
        raise NotImplementedError(
            "LayerLayerCollocation is not yet implemented."
        )
