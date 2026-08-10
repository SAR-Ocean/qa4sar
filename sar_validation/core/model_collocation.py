"""
Collocation between SAR and gridded background-field ("model") validation
sources -- currently ERA5 reanalysis. Ported from the proven bilinear
spatial + nearest-hour/hyperbolic temporal interpolation method in
``relevant_code_for_toolbox/s1_ocn_nwp_coloc/collocate_nwp_to_sat.py``,
adapted to this toolbox's ``collocation.py`` architecture. See
docs/superpowers/specs/2026-08-06-era5-model-validation-design.md and
docs/design-choices.md's ERA5 section for the full rationale, including why
this method is NOT extended to the existing observational layer_vs_layer
sources (scatterometer/altimeter/radiometer/hf_radar_grid/satellite SSM).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from .collocation import CollocatedPoint, PointLayerCollocation

logger = logging.getLogger(__name__)

__all__ = ["ModelLayerCollocation", "build_spatial_interpolator"]


# ---------------------------------------------------------------------------
# Spatial interpolation
# ---------------------------------------------------------------------------

def build_spatial_interpolator(
    lat_ax: np.ndarray, lon_ax: np.ndarray, field: np.ndarray,
) -> RegularGridInterpolator:
    """
    Build a bilinear ``RegularGridInterpolator`` over one ERA5 regional
    grid slice, shape ``(n_lat, n_lon)``.

    Unlike the reference NWP script's ``build_interpolator``, this
    intentionally applies NO longitude wrap-around padding. That padding
    exists there to handle the antimeridian for a GLOBAL NWP grid; ERA5
    downloads in this toolbox always request a small regional bounding box
    (see ``era5_downloader.py``), so padding a small regional extract with
    data from its own opposite edge would fabricate values -- the box
    isn't periodic. A recipe bbox that crosses the antimeridian is instead
    handled upstream, before this function ever runs: the downloader
    requests two non-crossing windows and the converter stitches them into
    one contiguous (if >180-valued) axis -- see Task 14 and
    docs/design-choices.md. This function itself doesn't need to know the
    difference; it just needs a monotonically increasing *lon_ax*,
    whatever its numeric range.

    *lat_ax*/*lon_ax* must be strictly monotonically increasing (true for
    every ERA5 CDS download in this toolbox). Returns NaN (via
    ``bounds_error=False, fill_value=np.nan``) for any query point outside
    the grid's coverage.
    """
    return RegularGridInterpolator(
        (lat_ax, lon_ax), field, method="linear", bounds_error=False, fill_value=np.nan,
    )


# ---------------------------------------------------------------------------
# Antimeridian support
# ---------------------------------------------------------------------------

def _normalize_query_lon(query_lon: np.ndarray, lon_ax: np.ndarray) -> np.ndarray:
    """
    Remap query longitudes to match an antimeridian-stitched ERA5 grid's
    axis (see ``DataTreeConverter._stitch_antimeridian_window_files``),
    whose lon axis may extend past 180 degrees. SAR pixel longitudes
    always use the standard -180..180 convention; only the stitched west
    window's original (negative) values were shifted by +360 to build
    that axis, so any query longitude below 0 needs the same +360 shift
    to land in the grid's actual coordinate range. No-op (returns
    *query_lon* unchanged) when the grid was not stitched
    (``lon_ax.max() <= 180``).
    """
    if lon_ax.max() <= 180.0:
        return query_lon
    return np.where(query_lon < 0, query_lon + 360.0, query_lon)


def _wrap_lon_to_pm180(lon: float) -> float:
    """Wrap a longitude value back into the standard -180..180 convention
    -- used when reporting an ERA5 native cell centre that came from an
    antimeridian-stitched grid axis extending past 180."""
    return lon - 360.0 if lon > 180.0 else lon


# ---------------------------------------------------------------------------
# Temporal interpolation
# ---------------------------------------------------------------------------

def _derive_wind_wspd_wdir(values: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Derive ``WSPD``/``WDIR`` from already-interpolated ``u10``/``v10``
    wind components, replacing them in the returned dict.

    This MUST run only on FINAL, already spatially/temporally interpolated
    component values -- never before interpolation. ``WDIR`` is a
    CIRCULAR quantity (0 degrees and 360 degrees are the same direction);
    interpolating a pre-derived direction as an ordinary linear/hyperbolic
    scalar produces wrong answers whenever the true value crosses the
    0/360 seam (e.g. blending 359 and 1 degrees naively yields ~180, not
    ~0). Deriving WSPD/WDIR from u10/v10 AFTER interpolation sidesteps
    this entirely, since u10/v10 themselves are ordinary (non-circular)
    scalar components that interpolate correctly. See C1 in
    docs/design-choices.md.

    No-op (returns *values* unchanged) unless both ``"u10"`` and
    ``"v10"`` are present -- a self-describing gate, since only wind data
    ever carries these keys (waves/soil_moisture Datasets are untouched).

    ``u10``/``v10`` are removed from the returned dict -- only ``WSPD``/
    ``WDIR`` remain, matching the output contract every other wind
    validation source (and every downstream consumer -- _variable_map.py,
    statistics.py) already expects.
    """
    if "u10" not in values or "v10" not in values:
        return values
    u10 = values["u10"]
    v10 = values["v10"]
    wspd = np.hypot(u10, v10)
    # Meteorological "from" direction, clockwise from north -- the same
    # convention every other WDIR column in this codebase uses.
    wdir = (270.0 - np.degrees(np.arctan2(v10, u10))) % 360.0
    out = {k: v for k, v in values.items() if k not in ("u10", "v10")}
    out["WSPD"] = wspd
    out["WDIR"] = wdir
    return out


def _hyperbolic_interp(
    val1: np.ndarray, val2: np.ndarray, val3: np.ndarray, t_prime: Union[float, np.ndarray],
) -> np.ndarray:
    """
    KNMI quadratic temporal interpolation through three equally-spaced
    (1-hour apart) values -- ported from
    ``collocate_nwp_to_sat.py``'s ``_quadratic_interp``.

    *val1*/*val2*/*val3* are the (already spatially-resolved) field values
    at ``t2 - 1h`` / ``t2`` / ``t2 + 1h``. ``t_prime = (t_obs - t2) / 1h``,
    in ``[0, 1)``.
    """
    a = (val3 + val1 - 2.0 * val2) / 2.0
    b = (val3 - val1) / 2.0
    c = val2
    return a * t_prime**2 + b * t_prime + c


# ---------------------------------------------------------------------------
# Shared point-interpolation core
# ---------------------------------------------------------------------------

def _model_values_at_points(
    lons: np.ndarray, lats: np.ndarray, times: np.ndarray,
    era5_ds: xr.Dataset, temporal_method: str,
) -> Dict[str, np.ndarray]:
    """
    Bilinear-spatial + nearest-hour/hyperbolic-temporal interpolate every
    model variable in *era5_ds* at each of ``len(lons)`` query points
    ``(lons[i], lats[i], times[i])``.

    Efficient for the common case where many points share the same (or
    very few distinct) observation times -- always true for a SAR scene,
    whose whole IW/EW grid shares one scalar acquisition time, or whose
    WV-mode imagettes share only a handful of per-imagette times: each
    hour's spatial interpolator is built once and queried in one
    vectorized batch per group of points sharing that hour-bracket,
    instead of rebuilding an interpolator per point.

    Returns
    -------
    dict[str, np.ndarray]
        ``{var_name: (n,) array}``, NaN at points where the model has no
        data (outside the downloaded grid, or no bracketing hour
        available).
    """
    n = len(lons)
    model_vars: list[str] = [str(v) for v in era5_ds.data_vars]
    era5_times = pd.to_datetime(era5_ds["time"].values).to_numpy()
    lat_ax = era5_ds["lat"].values
    lon_ax = era5_ds["lon"].values
    lons = _normalize_query_lon(np.asarray(lons, dtype=float), lon_ax)

    out: Dict[str, np.ndarray] = {var: np.full(n, np.nan, dtype=np.float64) for var in model_vars}

    times_np = pd.to_datetime(times).to_numpy()
    valid_mask = np.isfinite(lons) & np.isfinite(lats)
    unique_times = np.unique(times_np[valid_mask]) if np.any(valid_mask) else np.array([], dtype=times_np.dtype)

    # Interpolation-contamination guard (companion to the cell-averaging
    # land-skip above): land_sea_mask ("lsm", present only for wind -- see
    # DataTreeConverter.from_era5) is time-invariant, so it's bilinearly
    # interpolated ONCE here (no per-hour rebuild needed, unlike the
    # per-variable interpolators below) at every query point. A point
    # whose interpolated lsm exceeds 0.5 is close enough to a land grid
    # cell that its bilinearly-interpolated ERA5 wind value is itself
    # meaningfully blended with land-physics wind -- masked to NaN for
    # every model variable, same threshold/rationale as the cell-averaging
    # skip. `None` (not just all-zero) when era5_ds has no lsm at all, so
    # waves/soil_moisture/pre-fix wind Datasets are unaffected.
    land_mask = None
    if "lsm" in era5_ds.variables and np.any(valid_mask):
        lsm_interp = build_spatial_interpolator(lat_ax, lon_ax, era5_ds["lsm"].values)
        lsm_at_points = np.full(n, np.nan, dtype=np.float64)
        lsm_at_points[valid_mask] = lsm_interp(
            np.column_stack([lats[valid_mask], lons[valid_mask]])
        )
        land_mask = np.isfinite(lsm_at_points) & (lsm_at_points > 0.5)

    interp_cache: Dict[Tuple[str, int], RegularGridInterpolator] = {}

    def _get_interp(var: str, hour_idx: int) -> RegularGridInterpolator:
        key = (var, hour_idx)
        if key not in interp_cache:
            field = era5_ds[var].isel(time=hour_idx).values
            interp_cache[key] = build_spatial_interpolator(lat_ax, lon_ax, field)
        return interp_cache[key]

    for t in unique_times:
        group_mask = valid_mask & (times_np == t)
        group_pts = np.column_stack([lats[group_mask], lons[group_mask]])

        floor_hour = t.astype("datetime64[h]")
        if temporal_method == "nearest":
            hour_idxs = [int(np.argmin(np.abs(era5_times - t)))]
        else:
            idx2 = int(np.searchsorted(era5_times, floor_hour))
            if idx2 >= len(era5_times) or era5_times[idx2] != floor_hour:
                idx2 -= 1
            if idx2 < 1 or idx2 + 1 >= len(era5_times):
                continue  # no bracketing hour for this time group -- leave NaN
            hour_idxs = [idx2 - 1, idx2, idx2 + 1]

        for var in model_vars:
            values_at_hours = [_get_interp(var, h)(group_pts) for h in hour_idxs]
            if temporal_method == "nearest":
                blended = values_at_hours[0]
            else:
                t2 = era5_times[hour_idxs[1]]
                t_prime = (t - t2) / np.timedelta64(1, "h")
                blended = _hyperbolic_interp(
                    values_at_hours[0], values_at_hours[1], values_at_hours[2],
                    np.full(group_pts.shape[0], t_prime, dtype=float),
                )
            out[var][group_mask] = blended

    if land_mask is not None and np.any(land_mask):
        for var in model_vars:
            out[var][land_mask] = np.nan

    # Derive WSPD/WDIR from the now-FINAL, interpolated u10/v10 values --
    # must happen here, AFTER all spatial/temporal interpolation above, not
    # before (see C1 fix / _derive_wind_wspd_wdir's docstring). No-op for
    # waves/soil_moisture (no u10/v10 keys).
    return _derive_wind_wspd_wdir(out)


# ---------------------------------------------------------------------------
# ModelLayerCollocation
# ---------------------------------------------------------------------------

class ModelLayerCollocation:
    """
    Collocate a gridded background-field model source (currently only
    ERA5) against a SAR scene, using bilinear spatial interpolation plus
    nearest-hour or hyperbolic (KNMI quadratic) temporal interpolation.

    Unlike ``PointLayerCollocation``/``LayerLayerCollocation``, this class
    consumes the validation source as a raw GRIDDED ``xr.Dataset`` (dims:
    ``time``, ``lat``, ``lon``), not a flattened point ``DataFrame`` -- the
    whole point is to interpolate the model field onto arbitrary SAR
    locations, not to match against pre-existing rows.

    Parameters
    ----------
    method : str
        ``"individual"`` -- interpolate ERA5 directly onto every SAR
        pixel/point (dense). ``"cell-averaging"`` -- one match per ERA5
        native grid cell, averaging the SAR pixels within it (sparse,
        model-scale). Only affects grid-mode (IW/EW) scenes;
        :meth:`collocate_points` (WV mode) always uses direct
        interpolation regardless of this setting.
    temporal_method : str
        ``"nearest"`` or ``"hyperbolic"`` (KNMI quadratic).
    time_tolerance_minutes, aggregation_window_km, distance_weighting,
    gaussian_sigma_km : see ``PointLayerCollocation`` -- only used by the
        ``"cell-averaging"`` grid-mode path (Task 9).
    """

    collocation_type: str = "model_vs_layer"

    def __init__(
        self,
        method: str = "cell-averaging",
        temporal_method: str = "hyperbolic",
        time_tolerance_minutes: int = 60,
        aggregation_window_km: float = 12.5,
        distance_weighting: str = "equal",
        gaussian_sigma_km: float = 12.5,
    ) -> None:
        if method not in ("individual", "cell-averaging"):
            raise ValueError(f"Unknown method {method!r}. Use 'individual' or 'cell-averaging'.")
        if temporal_method not in ("nearest", "hyperbolic"):
            raise ValueError(f"Unknown temporal_method {temporal_method!r}. Use 'nearest' or 'hyperbolic'.")
        self.method = method
        self.temporal_method = temporal_method
        self.time_tolerance_minutes = time_tolerance_minutes
        self.aggregation_window_km = aggregation_window_km
        self.distance_weighting = distance_weighting
        self.gaussian_sigma_km = gaussian_sigma_km

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collocate(
        self,
        sar_data: Dict[str, np.ndarray],
        sar_lon: np.ndarray,
        sar_lat: np.ndarray,
        sar_time: np.ndarray,
        era5_ds: xr.Dataset,
        val_source: str,
        sar_scene_name: str = "",
    ) -> List[CollocatedPoint]:
        """Grid-mode (IW/EW) SAR scene. *sar_lon*/*sar_lat* shape ``(y, x)``;
        *sar_data* values shape ``(1, y, x)``; *sar_time* shape ``(1,)``.
        Dispatches to :attr:`method`."""
        if self.method == "individual":
            return self._collocate_individual_grid(
                sar_data, sar_lon, sar_lat, sar_time, era5_ds, val_source, sar_scene_name,
            )
        return self._collocate_cell_averaging_grid(
            sar_data, sar_lon, sar_lat, sar_time, era5_ds, val_source, sar_scene_name,
        )

    def collocate_points(
        self,
        sar_point_vars: Dict[str, np.ndarray],
        sar_lons: np.ndarray,
        sar_lats: np.ndarray,
        sar_times: np.ndarray,
        era5_ds: xr.Dataset,
        val_source: str,
        sar_scene_name: str = "",
    ) -> List[CollocatedPoint]:
        """
        WV-mode (sparse imagette points) SAR scene, all arrays shape
        ``(n_points,)``. Always interpolates ERA5 directly at each point
        regardless of :attr:`method` -- WV imagettes are already sparse
        SAR-anchor points (~200 km apart), so there is no dense SAR grid
        within one ERA5 cell to aggregate the way cell-averaging does for
        grid-mode scenes; interpolating ERA5 exactly at each point is the
        natural match here. See docs/design-choices.md.
        """
        times_np = pd.to_datetime(sar_times).to_numpy()
        model_values = _model_values_at_points(sar_lons, sar_lats, times_np, era5_ds, self.temporal_method)

        results: List[CollocatedPoint] = []
        for i in range(len(sar_lons)):
            if not (np.isfinite(sar_lons[i]) and np.isfinite(sar_lats[i])):
                continue
            val_point = {var: float(arr[i]) for var, arr in model_values.items() if np.isfinite(arr[i])}
            if not val_point:
                continue
            sar_point = {var: float(arr[i]) for var, arr in sar_point_vars.items() if np.isfinite(arr[i])}
            if not sar_point:
                continue
            obs_time = pd.Timestamp(times_np[i]).to_pydatetime()
            results.append(CollocatedPoint(
                sar_lon=float(sar_lons[i]), sar_lat=float(sar_lats[i]), sar_time=obs_time,
                sar_data=sar_point,
                val_lon=float(sar_lons[i]), val_lat=float(sar_lats[i]), val_time=obs_time,
                val_data=val_point,
                spatial_distance_km=0.0, temporal_distance_minutes=0.0,
                val_source=val_source, val_id=None,
                collocation_type=self.collocation_type,
                sar_y_idx=0, sar_x_idx=i,
                sar_scene_name=sar_scene_name,
            ))
        return results

    # ------------------------------------------------------------------
    # Individual (grid mode)
    # ------------------------------------------------------------------

    def _collocate_individual_grid(
        self,
        sar_data: Dict[str, np.ndarray],
        sar_lon: np.ndarray,
        sar_lat: np.ndarray,
        sar_time: np.ndarray,
        era5_ds: xr.Dataset,
        val_source: str,
        sar_scene_name: str,
    ) -> List[CollocatedPoint]:
        ny, nx = sar_lon.shape
        lons_flat = sar_lon.ravel()
        lats_flat = sar_lat.ravel()
        obs_time = pd.Timestamp(np.atleast_1d(sar_time)[0]).to_pydatetime()
        times_flat = np.full(lons_flat.shape, np.datetime64(obs_time), dtype="datetime64[ns]")

        model_values = _model_values_at_points(lons_flat, lats_flat, times_flat, era5_ds, self.temporal_method)

        results: List[CollocatedPoint] = []
        for flat_idx in range(lons_flat.size):
            if not (np.isfinite(lons_flat[flat_idx]) and np.isfinite(lats_flat[flat_idx])):
                continue
            val_point = {
                var: float(arr[flat_idx]) for var, arr in model_values.items() if np.isfinite(arr[flat_idx])
            }
            if not val_point:
                continue
            sar_point = {
                var: float(sar_data[var][0].ravel()[flat_idx])
                for var in sar_data
                if np.isfinite(sar_data[var][0].ravel()[flat_idx])
            }
            if not sar_point:
                continue
            y_idx, x_idx = divmod(flat_idx, nx)
            results.append(CollocatedPoint(
                sar_lon=float(lons_flat[flat_idx]), sar_lat=float(lats_flat[flat_idx]), sar_time=obs_time,
                sar_data=sar_point,
                val_lon=float(lons_flat[flat_idx]), val_lat=float(lats_flat[flat_idx]), val_time=obs_time,
                val_data=val_point,
                spatial_distance_km=0.0, temporal_distance_minutes=0.0,
                val_source=val_source, val_id=None,
                collocation_type=self.collocation_type,
                sar_y_idx=y_idx, sar_x_idx=x_idx,
                sar_scene_name=sar_scene_name,
            ))
        return results

    # ------------------------------------------------------------------
    # Cell-averaging (grid mode)
    # ------------------------------------------------------------------

    def _collocate_cell_averaging_grid(
        self,
        sar_data: Dict[str, np.ndarray],
        sar_lon: np.ndarray,
        sar_lat: np.ndarray,
        sar_time: np.ndarray,
        era5_ds: xr.Dataset,
        val_source: str,
        sar_scene_name: str,
    ) -> List[CollocatedPoint]:
        """
        For each native ERA5 grid cell overlapping the SAR scene:
        interpolate ERA5 TEMPORALLY (nearest-hour or hyperbolic) to the
        scene's acquisition time -- no spatial interpolation is needed
        here, since the match point IS the grid's own native cell centre
        -- then average every SAR pixel within ``aggregation_window_km``
        of that cell centre, reusing the exact same distance-weighted
        aggregation machinery ``PointLayerCollocation`` already uses for
        every other layer source (``_nearby_cells_with_distances`` +
        ``_compute_aggregated_sar_value``).
        """
        obs_time = pd.Timestamp(np.atleast_1d(sar_time)[0]).to_pydatetime()
        obs_np = np.datetime64(obs_time)
        era5_times = pd.to_datetime(era5_ds["time"].values).to_numpy()
        floor_hour = obs_np.astype("datetime64[h]")

        if self.temporal_method == "nearest":
            hour_idxs = [int(np.argmin(np.abs(era5_times - obs_np)))]
        else:
            idx2 = int(np.searchsorted(era5_times, floor_hour))
            if idx2 >= len(era5_times) or era5_times[idx2] != floor_hour:
                idx2 -= 1
            if idx2 < 1 or idx2 + 1 >= len(era5_times):
                logger.warning(
                    "ModelLayerCollocation: no bracketing ERA5 hour for scene '%s' at %s "
                    "-- skipping cell-averaging pass.", sar_scene_name, obs_time,
                )
                return []
            hour_idxs = [idx2 - 1, idx2, idx2 + 1]

        lat_ax = era5_ds["lat"].values
        lon_ax = era5_ds["lon"].values
        lon2d, lat2d = np.meshgrid(lon_ax, lat_ax)

        # land_sea_mask ("lsm", present only for wind -- see
        # DataTreeConverter.from_era5 / era5_downloader.py): a (lat, lon)
        # coordinate, not a data_var, so `era5_ds.data_vars` below never
        # sees it. Guarded with getattr/`.get` semantics via `in
        # era5_ds.variables` so waves/soil_moisture Datasets (which never
        # had lsm added) fall through unchanged -- no land-skip applied
        # when there's no data to skip on.
        lsm_grid = era5_ds["lsm"].values if "lsm" in era5_ds.variables else None

        model_vars: List[str] = [str(v) for v in era5_ds.data_vars]
        cell_values: Dict[str, np.ndarray] = {}
        for var in model_vars:
            fields = [era5_ds[var].isel(time=h).values for h in hour_idxs]
            if self.temporal_method == "nearest":
                cell_values[var] = fields[0]
            else:
                t2 = era5_times[hour_idxs[1]]
                t_prime = float((obs_np - t2) / np.timedelta64(1, "h"))
                cell_values[var] = _hyperbolic_interp(fields[0], fields[1], fields[2], t_prime)

        # Derive WSPD/WDIR from the now-FINAL, per-cell interpolated
        # u10/v10 values -- see C1 fix / _derive_wind_wspd_wdir's
        # docstring for why this must happen AFTER temporal interpolation,
        # not before. No-op for waves/soil_moisture. `model_vars` is
        # refreshed to match the (possibly renamed) keys so the val_point
        # comprehension below iterates the right set.
        cell_values = _derive_wind_wspd_wdir(cell_values)
        model_vars = list(cell_values.keys())

        helper = PointLayerCollocation(
            aggregation_window_km=self.aggregation_window_km,
            distance_weighting=self.distance_weighting,
            gaussian_sigma_km=self.gaussian_sigma_km,
        )
        grid_tree = PointLayerCollocation._build_grid_tree(sar_lon, sar_lat)

        results: List[CollocatedPoint] = []
        n_lat, n_lon = lat2d.shape
        for cy in range(n_lat):
            for cx in range(n_lon):
                # Cheap early skip, before any SAR-pixel aggregation work:
                # a cell whose own center is over land (lsm > 0.5, standard
                # ECMWF/oceanographic threshold) is skipped entirely, even
                # if valid ocean SAR pixels exist nearby -- ERA5's wind
                # field uses different surface-roughness/friction physics
                # over land vs. sea, so a land grid point's wind isn't
                # comparable to SAR ocean wind retrieval regardless of
                # proximity to the coast.
                if lsm_grid is not None and lsm_grid[cy, cx] > 0.5:
                    continue
                cell_lon = _wrap_lon_to_pm180(float(lon2d[cy, cx]))
                cell_lat = float(lat2d[cy, cx])
                val_point = {
                    var: float(cell_values[var][cy, cx])
                    for var in model_vars if np.isfinite(cell_values[var][cy, cx])
                }
                if not val_point:
                    continue

                nearby = helper._nearby_cells_with_distances(
                    cell_lon, cell_lat, sar_lon, sar_lat, self.aggregation_window_km, grid_tree=grid_tree,
                )
                if not nearby:
                    continue

                sar_aggregated = helper._compute_aggregated_sar_value(
                    nearby, sar_data, t_idx=0,
                    weighting_method=self.distance_weighting,
                    sigma_km=self.gaussian_sigma_km,
                    agg_window_km=self.aggregation_window_km,
                )
                if not sar_aggregated:
                    continue

                mean_dist = float(np.mean([d for _, _, d in nearby]))
                results.append(CollocatedPoint(
                    sar_lon=cell_lon, sar_lat=cell_lat, sar_time=obs_time,
                    sar_data=sar_aggregated,
                    val_lon=cell_lon, val_lat=cell_lat, val_time=obs_time,
                    val_data=val_point,
                    spatial_distance_km=mean_dist, temporal_distance_minutes=0.0,
                    val_source=val_source, val_id=None,
                    collocation_type=self.collocation_type,
                    sar_y_idx=cy, sar_x_idx=cx,
                    sar_scene_name=sar_scene_name,
                ))
        return results
