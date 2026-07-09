"""
Mapping from recipe ``variable`` field to (sar_var, val_var) pairs.

This is the single source of truth used by both ``statistics.py`` and
``visualization.py``.  The SAR variable names follow the ``owi*`` convention
from Sentinel-1 L2 OCN products; the validation variable names match the
column headers in Copernicus Marine in-situ CSV files and standard CF names
used by the scatterometer / altimeter converters.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

__all__ = ["VARIABLE_PAIRS", "CIRCULAR_VAL_VARS", "infer_variable_pairs", "filter_variable_pairs"]

# ---------------------------------------------------------------------------
# (sar_variable_name, validation_variable_name) pairs per validated quantity
# ---------------------------------------------------------------------------

VARIABLE_PAIRS: dict[str, List[Tuple[str, str]]] = {
    "wind": [
        ("owiWindSpeed",     "WSPD"),
        ("owiWindDirection", "WDIR"),
    ],
    "currents": [
        ("rvlRadVel", "rvlRadVel_projection"),  # RVL radial velocity (scalar) vs projected validation currents
    ],
    "waves": [
        ("oswHs",                       "VHM0"),  # WV mode point measurement (1×1)
        ("owiHs",    "VHM0"),  # IW/EW mode OWI grid (fallback)
    ],
}

# ---------------------------------------------------------------------------
# Validation-variable codes that are circular (angular, degrees in [0, 360)).
# ``statistics.compute_statistics`` uses this to switch to wrap-aware bias /
# RMSE / correlation instead of plain linear arithmetic.
# ---------------------------------------------------------------------------

CIRCULAR_VAL_VARS: set[str] = {"WDIR"}


def infer_variable_pairs(variable: str) -> List[Tuple[str, str]]:
    """
    Return the list of ``(sar_var, val_var)`` pairs for *variable*.

    Parameters
    ----------
    variable : str
        Value of ``recipe.config.variable`` — one of ``"wind"``,
        ``"currents"``, or ``"waves"``.

    Returns
    -------
    list[tuple[str, str]]
        Pairs of ``(sar_variable_name, validation_variable_name)``.

    Raises
    ------
    KeyError
        If *variable* is not recognised.
    """
    if variable not in VARIABLE_PAIRS:
        known = ", ".join(f"'{k}'" for k in VARIABLE_PAIRS)
        raise KeyError(
            f"Unknown recipe variable '{variable}'. Known values: {known}."
        )
    return VARIABLE_PAIRS[variable]


def filter_variable_pairs(
    recipe,
    collocation_ds,
) -> List[Tuple[str, str]]:
    """
    Filter variable pairs based on recipe swath mode and available variables
    in the collocation dataset.

    For "waves" variable type, this function:
    1. Detects which wave validation parameters are available (VHM0, VAVH, VGHS, etc.)
    2. For WV mode: generates pairs with oswHs only
    3. For IW/EW mode: generates pairs for both oswHs and owiSignificantWaveHeight
    4. Filters to only pairs where both SAR and validation variables exist

    For other variable types (wind, currents):
    - Filters to only pairs where both variables exist in the collocation data

    Parameters
    ----------
    recipe : Recipe
        Recipe object with config.variable and config.sar_data.swath_mode
    collocation_ds : xr.Dataset
        Collocation dataset with variables named ``sar_<var>`` and ``val_<var>``

    Returns
    -------
    list[tuple[str, str]]
        Filtered pairs where both variables exist in the collocation data.
    """
    variable = recipe.config.variable
    base_pairs = infer_variable_pairs(variable)
    swath_modes = recipe.config.sar_data.swath_mode or ["IW", "EW"]
    is_wv_only = set(swath_modes) == {"WV"}

    # For waves: expand to all available wave validation parameters
    if variable == "waves":
        # Wave validation parameter candidates (in preferred order)
        wave_val_params = ["VHM0", "VAVH", "VGHS", "VAVH_UNFILTERED"]
        
        # Determine which SAR variables to use based on swath mode
        if is_wv_only:
            # WV mode: only use oswHs
            sar_vars = ["oswHs"]
        else:
            # IW/EW or mixed mode: prefer oswHs if available, but also try owiSignificantWaveHeight
            sar_vars = ["oswHs", "owiSignificantWaveHeight"]
        
        # Generate all combinations of available SAR and validation pairs
        pairs = []
        for sar_var in sar_vars:
            for val_param in wave_val_params:
                pairs.append((sar_var, val_param))
    else:
        pairs = base_pairs.copy()

    # Filter to only pairs where both variables actually exist in the collocation dataset
    valid_pairs = []
    for sar_var, val_var in pairs:
        sar_col = f"sar_{sar_var}"
        val_col = f"val_{val_var}"
        if sar_col in collocation_ds and val_col in collocation_ds:
            valid_pairs.append((sar_var, val_var))

    return valid_pairs
