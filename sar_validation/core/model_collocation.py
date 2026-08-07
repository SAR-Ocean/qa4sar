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
from typing import Dict, List, Tuple

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
# Temporal interpolation
# ---------------------------------------------------------------------------

def _hyperbolic_interp(
    val1: np.ndarray, val2: np.ndarray, val3: np.ndarray, t_prime: np.ndarray,
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

    out: Dict[str, np.ndarray] = {var: np.full(n, np.nan, dtype=np.float64) for var in model_vars}

    times_np = pd.to_datetime(times).to_numpy()
    valid_mask = np.isfinite(lons) & np.isfinite(lats)
    unique_times = np.unique(times_np[valid_mask]) if np.any(valid_mask) else np.array([], dtype=times_np.dtype)

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

    return out
