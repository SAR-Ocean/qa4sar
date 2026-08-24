"""
Collocation between SAR and gridded background-field ("model") validation
sources -- currently ERA5 reanalysis and HYCOM. Implements bilinear
spatial + nearest-hour/hyperbolic temporal interpolation methods and keeps
the models gridded (in contrast to the flattened approach in layer_vs_layer 
collocation).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

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
    Build a bilinear ``RegularGridInterpolator`` over one model grid
    slice (ERA5 or HYCOM), shape ``(n_lat, n_lon)``.

    This function itself does not need to know the model source. It needs 
    a monotonically increasing *lon_ax*. ERA5 downloads in this toolbox 
    always request a small regional bounding box (see ``era5_downloader.py``); 
    a recipe bbox that crosses the antimeridian is handled upstream, by 
    requesting two non-crossing windows and stitching them into one 
    contiguous (if >180-valued) axis -- see docs/design-choices.md. HYCOM
    does not need this as it is served over a native 0-360 longitude grid,

    *lat_ax*/*lon_ax* must be strictly monotonically increasing. This
    function does not re-sort either axis itself. For ERA5, CDS always 
    returns latitude DESCENDING (north -> south); ``DataTreeConverter.from_era5``
    establishes the ascending invariant via an explicit ``raw.sortby("lat")``. 
    For HYCOM, THREDDS already provides latitude ascending, no sortby needed.
    Longitude is naturally ascending already, including the antimeridian-
    stitched ERA5 case (see below). Returns NaN (via ``bounds_error=False, 
    fill_value=np.nan``) for any query point outside the grid's coverage.
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
    scalar components that interpolate correctly. See docs/design-choices.md
    §5.7 ("ERA5 model validation") and §6 ("Circular variables") for the
    full rationale.

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


def _derive_currents_radial_projection(
    values: Dict[str, Any], heading_deg: Any,
) -> Dict[str, Any]:
    """
    Add ``rvlRadVel_projection`` to *values*, computed by projecting 
    the ``EWCT``/``NSCT`` (eastward/northward current) components onto 
    the SAR line-of-sight, given platform heading *heading_deg* (degrees;
    a scalar or an array matching *values*).

    Returns *values* unchanged unless both ``"EWCT"`` and ``"NSCT"`` are
    present, and *heading_deg* is not ``None``; HYCOM-family currents 
    Datasets carry these keys, so wind, waves, and soil moisture model
    Datasets are unaffected. 

    Uses the same ``_project_currents_to_radial`` formula applied to every
    other currents validation source (e.g. HF-radar, in-situ), rather than
    reimplementing it. Unlike :func:`_derive_wind_wspd_wdir`, ``EWCT``/
    ``NSCT`` are retained in the output.

    *heading_deg* must be supplied by the caller from the SAR side; see the 
    three collocation paths in ``ModelLayerCollocation``.
    """
    if heading_deg is None or "EWCT" not in values or "NSCT" not in values:
        return values
    from .collocation import _project_currents_to_radial
    out = dict(values)
    out["rvlRadVel_projection"] = _project_currents_to_radial(
        values["EWCT"], values["NSCT"], heading_deg,
    )
    return out


def _hyperbolic_interp(
    val1: np.ndarray, val2: np.ndarray, val3: np.ndarray, t_prime: Union[float, np.ndarray],
) -> np.ndarray:
    """
    Quadratic (KNMI-style) temporal interpolation through three
    equally-spaced values. 
    
    This function is spacing-agnostic: it has no notion of hours or any 
    other physical unit, only the normalized offset *t_prime*. The caller
    is responsible for normalizing *t_prime* against the actual measured 
    spacing between its three bracket points -- 1 hour for ERA5's hourly 
    granules, 3 hours for HYCOM's 3-hourly granules. See
    ``_model_values_at_points`` and
    ``ModelLayerCollocation._collocate_cell_averaging_grid``, and
    :func:`_regular_bracket_gap`, which both callers use to derive that
    spacing rather than assuming it).

    *val1*, *val2*, and *val3* are the already spatially-resolved field 
    values at ``t2 - dt``, ``t2``, and ``t2 + dt``, respectively, for 
    whatever bracket spacing ``dt`` the caller normalized against. 
    ``t_prime = (t_obs - t2) / dt``, in ``[0, 1)``.
    """
    a = (val3 + val1 - 2.0 * val2) / 2.0
    b = (val3 - val1) / 2.0
    c = val2
    return a * t_prime**2 + b * t_prime + c


def _regular_bracket_gap(
    model_times: np.ndarray, hour_idxs: List[int],
) -> Optional[np.timedelta64]:
    """
    Return the spacing between the three bracket points
    ``model_times[hour_idxs]`` (``[idx2 - 1, idx2, idx2 + 1]``) used by
    :func:`_hyperbolic_interp`'s callers to normalize ``t_prime`` -- or
    ``None`` if the backward gap (``t2 - t1``) and forward gap
    (``t3 - t2``) differ.

    :func:`_hyperbolic_interp`'s quadratic formula assumes its three
    samples are equally spaced; both ERA5 data (hourly) and HYCOM data 
    (3-hourly) satisfy this in the common case. However, HYCOM data could
    contain data gaps, meaning that there is no single spacing to 
    normalize ``t_prime`` against. Returning ``None`` in these cases, thus
    resulting in NaN to avoid wrong use of the interpolation method. 
    """
    forward = model_times[hour_idxs[2]] - model_times[hour_idxs[1]]
    backward = model_times[hour_idxs[1]] - model_times[hour_idxs[0]]
    if forward != backward:
        return None
    return forward


# ---------------------------------------------------------------------------
# Shared point-interpolation core
# ---------------------------------------------------------------------------

def _model_values_at_points(
    lons: np.ndarray, lats: np.ndarray, times: np.ndarray,
    model_ds: xr.Dataset, temporal_method: str,
) -> Dict[str, np.ndarray]:
    """
    Bilinear-spatial + nearest-hour/hyperbolic-temporal interpolate every
    model variable in *model_ds* at each of ``len(lons)`` query points
    ``(lons[i], lats[i], times[i])``.

    Efficient when many points share the same (or few distinct) 
    observation times -- always true for a SAR scene, whose IW/EW grid 
    shares one acquisition time and whose WV-mode vignettes share only a 
    handful: each hour's spatial interpolator is built once and queried 
    in a vectorized batch per group time, rather than rebuilt per point.

    Returns
    -------
    dict[str, np.ndarray]
        ``{var_name: (n,) array}``, NaN at points where the model has no
        data (outside the downloaded grid, or no bracketing hour
        available).
    """
    n = len(lons)
    model_vars: list[str] = [str(v) for v in model_ds.data_vars]
    model_times = pd.to_datetime(model_ds["time"].values).to_numpy()
    lat_ax = model_ds["lat"].values
    lon_ax = model_ds["lon"].values
    lons = _normalize_query_lon(np.asarray(lons, dtype=float), lon_ax)

    out: Dict[str, np.ndarray] = {var: np.full(n, np.nan, dtype=np.float64) for var in model_vars}

    times_np = pd.to_datetime(times).to_numpy()
    valid_mask = np.isfinite(lons) & np.isfinite(lats)
    unique_times = np.unique(times_np[valid_mask]) if np.any(valid_mask) else np.array([], dtype=times_np.dtype)

    # land_sea_mask ("lsm", present only for ERA5 wind) is time-invariant
    # and is therefore interpolated once here, rather than rebuilt per hour
    # as the per-variable interpolators below are. A point whose
    # interpolated lsm value exceeds 0.5 is considered close enough to
    # land that its bilinearly-interpolated wind value is contaminated by
    # land-physics wind; such points are masked to NaN for every model
    # variable, mirroring the cell-averaging land-skip above. The mask is
    # `None`, rather than all-zero, when model_ds has no lsm variable,
    # leaving waves and soil-moisture Datasets unaffected.

    land_mask = None
    if "lsm" in model_ds.variables and np.any(valid_mask):
        lsm_interp = build_spatial_interpolator(lat_ax, lon_ax, model_ds["lsm"].values)
        lsm_at_points = np.full(n, np.nan, dtype=np.float64)
        lsm_at_points[valid_mask] = lsm_interp(
            np.column_stack([lats[valid_mask], lons[valid_mask]])
        )
        land_mask = np.isfinite(lsm_at_points) & (lsm_at_points > 0.5)

    interp_cache: Dict[Tuple[str, int], RegularGridInterpolator] = {}

    def _get_interp(var: str, hour_idx: int) -> RegularGridInterpolator:
        key = (var, hour_idx)
        if key not in interp_cache:
            field = model_ds[var].isel(time=hour_idx).values
            interp_cache[key] = build_spatial_interpolator(lat_ax, lon_ax, field)
        return interp_cache[key]

    for t in unique_times:
        group_mask = valid_mask & (times_np == t)
        group_pts = np.column_stack([lats[group_mask], lons[group_mask]])

        floor_hour = t.astype("datetime64[h]")
        if temporal_method == "nearest":
            hour_idxs = [int(np.argmin(np.abs(model_times - t)))]
        else:
            idx2 = int(np.searchsorted(model_times, floor_hour))
            if idx2 >= len(model_times) or model_times[idx2] != floor_hour:
                idx2 -= 1
            if idx2 < 1 or idx2 + 1 >= len(model_times):
                continue  # no bracketing hour for this time group -- leave NaN
            hour_idxs = [idx2 - 1, idx2, idx2 + 1]

        t_prime: Optional[float] = None
        if temporal_method != "nearest":
            t2 = model_times[hour_idxs[1]]
            gap = _regular_bracket_gap(model_times, hour_idxs)
            if gap is None:
                logger.debug(
                    "ModelLayerCollocation: irregular bracket spacing around %s "
                    "(backward gap != forward gap) -- leaving this time group NaN "
                    "instead of fabricating an interpolated value.", t2,
                )
                continue  # skip this whole time group -- leave NaN
            t_prime = (t - t2) / gap

        for var in model_vars:
            values_at_hours = [_get_interp(var, h)(group_pts) for h in hour_idxs]
            if temporal_method == "nearest":
                blended = values_at_hours[0]
            else:
                blended = _hyperbolic_interp(
                    values_at_hours[0], values_at_hours[1], values_at_hours[2],
                    np.full(group_pts.shape[0], t_prime, dtype=float),
                )
            out[var][group_mask] = blended

    if land_mask is not None and np.any(land_mask):
        for var in model_vars:
            out[var][land_mask] = np.nan

    # Derive WSPD/WDIR from the now-final, interpolated u10/v10 values.
    # This must occur after all spatial/temporal interpolation above, not
    # before -- see _derive_wind_wspd_wdir's docstring for why. A no-op
    # for waves/soil-moisture, which carry no u10/v10 keys.

    return _derive_wind_wspd_wdir(out)


# ---------------------------------------------------------------------------
# ModelLayerCollocation
# ---------------------------------------------------------------------------

class ModelLayerCollocation:
    """
    Collocate a gridded background-field model source (ERA5 or HYCOM) 
    against a SAR scene, using bilinear spatial interpolation together
    with nearest-hour or hyperbolic (KNMI quadratic) temporal interpolation.

    Unlike ``PointLayerCollocation``/``LayerLayerCollocation``, this class
    consumes the validation source as a raw gridded ``xr.Dataset`` (dims:
    ``time``, ``lat``, ``lon``), rather than a flattened point ``DataFrame``:
    its purpose is to interpolate the model field onto arbitrary SAR
    locations, not to match against pre-existing rows.

    Parameters
    ----------
    method : str
        ``"individual"`` interpolates the model directly onto every SAR
        pixel or point (dense). ``"cell-averaging"`` produces one match
        per native model grid cell, averaging the SAR pixels within it 
        (sparse, model-scale). This only affects grid-mode (IW/EW) scenes;
        :meth:`collocate_points` (WV mode) always uses direct
        interpolation regardless of this setting.
    temporal_method : str
        ``"nearest"`` or ``"hyperbolic"`` (KNMI quadratic).
    time_tolerance_minutes, aggregation_window_km, distance_weighting,
    gaussian_sigma_km : see ``PointLayerCollocation``; used only by the
        ``"cell-averaging"`` grid-mode path.
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
        model_ds: xr.Dataset,
        val_source: str,
        sar_scene_name: str = "",
    ) -> List[CollocatedPoint]:
        """Grid-mode (IW/EW) SAR scene. *sar_lon*/*sar_lat* shape ``(y, x)``;
        *sar_data* values shape ``(1, y, x)``; *sar_time* shape ``(1,)``.
        Dispatches to :attr:`method`."""
        if self.method == "individual":
            return self._collocate_individual_grid(
                sar_data, sar_lon, sar_lat, sar_time, model_ds, val_source, sar_scene_name,
            )
        return self._collocate_cell_averaging_grid(
            sar_data, sar_lon, sar_lat, sar_time, model_ds, val_source, sar_scene_name,
        )

    def collocate_points(
        self,
        sar_point_vars: Dict[str, np.ndarray],
        sar_lons: np.ndarray,
        sar_lats: np.ndarray,
        sar_times: np.ndarray,
        model_ds: xr.Dataset,
        val_source: str,
        sar_scene_name: str = "",
    ) -> List[CollocatedPoint]:
        """
        WV-mode (sparse vignette points) SAR scene; all arrays have shape
        ``(n_points,)``. The model is always interpolated directly at each
        point regardless of :attr:`method`: WV vignettes are already sparse
        SAR-anchor points (~200 km apart), so there is no dense SAR grid 
        within a single model cell to aggregate, as cell-averaging does 
        for grid-mode scenes. Interpolating the model directly at each 
        point is the appropriate approach here. See docs/design-choices.md.
        """
        times_np = pd.to_datetime(sar_times).to_numpy()
        model_values = _model_values_at_points(sar_lons, sar_lats, times_np, model_ds, self.temporal_method)
        if "rvlHeading" in sar_point_vars:
            model_values = _derive_currents_radial_projection(
                model_values, sar_point_vars["rvlHeading"],
            )

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
        model_ds: xr.Dataset,
        val_source: str,
        sar_scene_name: str,
    ) -> List[CollocatedPoint]:
        ny, nx = sar_lon.shape
        lons_flat = sar_lon.ravel()
        lats_flat = sar_lat.ravel()
        obs_time = pd.Timestamp(np.atleast_1d(sar_time)[0]).to_pydatetime()
        times_flat = np.full(lons_flat.shape, np.datetime64(obs_time), dtype="datetime64[ns]")

        model_values = _model_values_at_points(lons_flat, lats_flat, times_flat, model_ds, self.temporal_method)
        if "rvlHeading" in sar_data:
            model_values = _derive_currents_radial_projection(
                model_values, sar_data["rvlHeading"][0].ravel(),
            )

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
        model_ds: xr.Dataset,
        val_source: str,
        sar_scene_name: str,
    ) -> List[CollocatedPoint]:
        """
        For each native model grid cell overlapping the SAR scene, the model
        is interpolated temporally (nearest-hour or hyperbolic) to the
        scene's acquisition time; spatial interpolation is required, since
        the match point is the grid cell's own native centre. Every SAR pixel 
        within ``aggregation_window_km`` of that cell centre is then averaged, 
        reusing the exact same distance-weighted aggregation machinery that
        ``PointLayerCollocation`` uses for every other layer source. See
        (``_nearby_cells_with_distances`` + ``_compute_aggregated_sar_value``).
        """
        obs_time = pd.Timestamp(np.atleast_1d(sar_time)[0]).to_pydatetime()
        obs_np = np.datetime64(obs_time)
        model_times = pd.to_datetime(model_ds["time"].values).to_numpy()
        floor_hour = obs_np.astype("datetime64[h]")

        if self.temporal_method == "nearest":
            hour_idxs = [int(np.argmin(np.abs(model_times - obs_np)))]
        else:
            idx2 = int(np.searchsorted(model_times, floor_hour))
            if idx2 >= len(model_times) or model_times[idx2] != floor_hour:
                idx2 -= 1
            if idx2 < 1 or idx2 + 1 >= len(model_times):
                logger.warning(
                    "ModelLayerCollocation: no bracketing model hour for scene '%s' at %s "
                    "-- skipping cell-averaging pass.", sar_scene_name, obs_time,
                )
                return []
            hour_idxs = [idx2 - 1, idx2, idx2 + 1]

        bracket_gap: Optional[np.timedelta64] = None
        if self.temporal_method == "hyperbolic":
            bracket_gap = _regular_bracket_gap(model_times, hour_idxs)
            if bracket_gap is None:
                logger.warning(
                    "ModelLayerCollocation: irregular bracket spacing around %s for "
                    "scene '%s' (backward gap != forward gap) -- skipping "
                    "cell-averaging pass instead of fabricating an interpolated value.",
                    model_times[hour_idxs[1]], sar_scene_name,
                )
                return []

        lat_ax = model_ds["lat"].values
        lon_ax = model_ds["lon"].values
        lon2d, lat2d = np.meshgrid(lon_ax, lat_ax)

        # land_sea_mask ("lsm", present only for ERA5 wind -- see
        # DataTreeConverter.from_era5 / era5_downloader.py) is a (lat, lon)
        # coordinate rather than a data_var, so it is never included in
        # `model_ds.data_vars` below. Its absence from waves and
        # soil-moisture Datasets means no land-skip is applied to those
        # sources.
        lsm_grid = model_ds["lsm"].values if "lsm" in model_ds.variables else None

        model_vars: List[str] = [str(v) for v in model_ds.data_vars]
        cell_values: Dict[str, np.ndarray] = {}
        for var in model_vars:
            fields = [model_ds[var].isel(time=h).values for h in hour_idxs]
            if self.temporal_method == "nearest":
                cell_values[var] = fields[0]
            else:
                t2 = model_times[hour_idxs[1]]
                t_prime = float((obs_np - t2) / bracket_gap)
                cell_values[var] = _hyperbolic_interp(fields[0], fields[1], fields[2], t_prime)

        # Derive WSPD/WDIR from the now-final, per-cell interpolated u10/v10
        # values -- see _derive_wind_wspd_wdir's docstring for why this must
        # occur after temporal interpolation, not before. A no-op for
        # waves/soil-moisture. `model_vars` is refreshed to reflect the
        # (possibly renamed) keys, so the val_point comprehension below
        # iterates the correct set.
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
                # An early, inexpensive skip performed before any SAR-pixel
                # aggregation work: a cell whose own centre lies over land (lsm > 0.5,
                # the standard ECMWF/oceanographic threshold) is skipped entirely,
                # even if valid ocean SAR pixels exist nearby. ERA5's wind field uses
                # different surface-roughness and friction physics over land than
                # over sea, so a land grid point's wind value is not comparable to a
                # SAR ocean wind retrieval, regardless of its proximity to the coast.
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
                val_point = _derive_currents_radial_projection(
                    val_point, sar_aggregated.get("rvlHeading"),
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
