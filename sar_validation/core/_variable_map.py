"""
Mapping from recipe ``variable`` field to (sar_var, val_var) pairs.

This is the single source of truth used by both ``statistics.py`` and
``visualization.py``.  The SAR variable names follow the ``owi*`` convention
from Sentinel-1 L2 OCN products; the validation variable names match the
column headers in Copernicus Marine in-situ CSV files and standard CF names
used by the scatterometer / altimeter converters.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

__all__ = [
    "VARIABLE_PAIRS", "CIRCULAR_VAL_VARS", "infer_variable_pairs",
    "filter_variable_pairs", "circular_diff_deg",
]

# ---------------------------------------------------------------------------
# (sar_variable_name, validation_variable_name) pairs per validated quantity
# ---------------------------------------------------------------------------

VARIABLE_PAIRS: dict[str, List[Tuple[str, str]]] = {
    "wind": [
        # WSPD covers in-situ, scatterometer AND altimeter wind speed: the
        # converters rename the raw product names (OSI-SAF ``wind_speed``,
        # CMEMS altimeter ``WIND_SPEED``) to this single canonical code so
        # all sources land in one comparison section.
        ("owiWindSpeed",     "WSPD"),
        ("owiWindDirection", "WDIR"),
    ],
    "currents": [
        ("rvlRadVel", "rvlRadVel_projection"),  # RVL radial velocity (scalar) vs projected validation currents
    ],
    "waves": [
        ("oswTotalHs",                  "VHM0"),  # WV mode: integrated total significant wave height
        ("oswHs",                       "VHM0"),  # WV mode partition Hs (legacy fallback)
        ("owiHs",    "VHM0"),  # IW/EW mode OWI grid (fallback)
    ],
}

# ---------------------------------------------------------------------------
# Validation-variable codes that are circular (angular, degrees in [0, 360)).
# ``statistics.compute_statistics`` uses this to switch to wrap-aware bias /
# RMSE / correlation instead of plain linear arithmetic.
# ---------------------------------------------------------------------------

CIRCULAR_VAL_VARS: set[str] = {"WDIR"}


def circular_diff_deg(a, b):
    """Wrapped angular difference a-b in (-180, 180], for degree-valued circular variables."""
    return ((np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0) % 360.0) - 180.0


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
    Filter variable pairs based on the recipe's variable type and available
    variables in the collocation dataset.

    For "waves" variable type, this function:
    1. Detects which wave validation parameters are available (VHM0, VAVH, VGHS, etc.)
    2. Picks the primary SAR wave variable by fallback (oswTotalHs, else
       oswHs, based on which column actually exists in collocation_ds), and
       additionally includes owiSignificantWaveHeight whenever that column
       exists and has at least one non-NaN value
    3. Filters to only pairs where both SAR and validation variables exist

    For other variable types (wind, currents):
    - Filters to only pairs where both variables exist in the collocation data

    Parameters
    ----------
    recipe : Recipe
        Recipe object with config.variable
    collocation_ds : xr.Dataset
        Collocation dataset with variables named ``sar_<var>`` and ``val_<var>``

    Returns
    -------
    list[tuple[str, str]]
        Filtered pairs where both variables exist in the collocation data.
    """
    variable = recipe.config.variable
    base_pairs = infer_variable_pairs(variable)

    # For waves: expand to all available wave validation parameters
    if variable == "waves":
        # Wave validation parameter candidates (in preferred order)
        wave_val_params = ["VHM0", "VAVH", "VGHS"]

        # Primary SAR wave-height variable: single-winner fallback driven by
        # which sar_<name> column actually exists in collocation_ds — NOT by
        # recipe.config.sar_data.swath_mode, since a recipe can request
        # multiple modes (e.g. [WV, SM]) while the downloader only ends up
        # returning scenes for one of them. Using the requested mode to pick
        # candidates caused real WV-only results to be silently dropped when
        # a mixed mode was requested (see
        # docs/superpowers/specs/2026-07-16-wave-sar-variable-fallback-design.md).
        primary_candidates = ["oswTotalHs", "oswHs"]
        primary_var = next(
            (v for v in primary_candidates if f"sar_{v}" in collocation_ds),
            None,
        )

        sar_vars = [primary_var] if primary_var is not None else []

        # owiSignificantWaveHeight is additive, not a fallback: it's a
        # genuinely different measurement (IW/EW grid product), so when it
        # actually carries data it gets its own statistics alongside the
        # primary variable rather than replacing it. In every real product
        # seen so far this column is either absent or entirely NaN, in which
        # case it must NOT be selected.
        owi_col = "sar_owiSignificantWaveHeight"
        if owi_col in collocation_ds and bool(collocation_ds[owi_col].notnull().any()):
            sar_vars.append("owiSignificantWaveHeight")

        # Generate all combinations of the selected SAR variable(s) and
        # available validation pairs
        pairs = []
        for sv in sar_vars:
            for val_param in wave_val_params:
                pairs.append((sv, val_param))
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
