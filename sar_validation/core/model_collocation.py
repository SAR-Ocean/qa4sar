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
