"""
Mapping from recipe ``variable`` field to (sar_var, val_var) pairs.

This is the single source of truth used by both ``statistics.py`` and
``visualization.py``.

sar_var names originate from Sentinel-1 L2 OCN's own product field
names (``owi*``, ``osw*``, ``rvlRadVel``), but every SAR source's
converter normalizes its output to these same canonical codes at
conversion time -- e.g. RADARSAT-2 wind is renamed to
``owiWindSpeed``, and NISAR SME2 soil moisture to ``sarSSM`` --
rather than each source keeping its own raw field name:
  wind          : ``owiWindSpeed``/``owiWindDirection``
  waves         : ``osw*``
  currents      : ``rvlRadVel``
  soil_moisture : ``sarSSM`` 

val_var names are likewise the codes every validation converter
normalizes its own source to, not literal column headers from any
one source -- e.g. CF-like codes (``WSPD``/``WDIR``), the CMEMS
wave-height code (``VHM0``), a derived projection
(``rvlRadVel_projection``), or a generic descriptive code
(``SOIL_MOISTURE``).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

__all__ = [
    "VARIABLE_PAIRS", "CIRCULAR_VAL_VARS", "infer_variable_pairs",
    "filter_variable_pairs", "circular_diff_deg",
    "WAVE_HEIGHT_VAL_VARS", "WAVE_HEIGHT_MERGED_VAL_VAR", "merge_wave_height_columns",
]

# ---------------------------------------------------------------------------
# (sar_variable_name, validation_variable_name) pairs per validated quantity
# ---------------------------------------------------------------------------

VARIABLE_PAIRS: dict[str, List[Tuple[str, str]]] = {
    "wind": [
        # WSPD covers in-situ, scatterometer and altimeter wind speed: the
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
    "soil_moisture": [
        ("sarSSM", "SOIL_MOISTURE"),
    ],
}

# ---------------------------------------------------------------------------
# Validation-variable codes that are circular (angular, degrees in [0, 360)).
# ``statistics.compute_statistics`` uses this to switch to wrap-aware bias /
# RMSE / correlation instead of plain linear arithmetic.
# ---------------------------------------------------------------------------

CIRCULAR_VAL_VARS: set[str] = {"WDIR"}

# ---------------------------------------------------------------------------
# Wave-height validation codes, merged into one combined comparison group
# ---------------------------------------------------------------------------

#: Wave-height validation codes that get merged into one combined "SWH"
#: comparison group, in precedence order (matches from_insitu_csv's own
#: per-row precedence). VHM0 (spectral Hm0, from ERA5/some in-situ platforms) 
#: and VAVH (time-domain H1/3, from altimeters/some in-situ platforms) 
#: are correlated-but-distinct estimators of the same physical quantity.
#: Merging them avoids a separate report section per code, some of which
#: would otherwise show all-NaN geographic panels for collocation types
#: that do not use that section's variable.
WAVE_HEIGHT_VAL_VARS: Tuple[str, ...] = ("VHM0", "VAVH", "VGHS")

#: Canonical merged val_var code standing in for any WAVE_HEIGHT_VAL_VARS
#: member.
WAVE_HEIGHT_MERGED_VAL_VAR = "SWH"


def merge_wave_height_columns(collocation_ds) -> bool:
    """
    Combine ``val_VHM0``/``val_VAVH``/``val_VGHS`` into one
    ``val_SWH`` column in place, plus a ``val_var_code`` companion
    column recording which of the three each row's value came from.

    At most one of the three is ever populated for a given row:
    ``from_insitu_csv`` already nulls out all but the
    highest-precedence one, and every other converter only ever
    produces one of the three in the first place. So merging is just
    taking whichever one is non-null per row, in
    :data:`WAVE_HEIGHT_VAL_VARS` precedence order.

    Returns False and makes no changes if none of
    :data:`WAVE_HEIGHT_VAL_VARS` are present in *collocation_ds*.
    Returns True without redoing the merge if ``val_SWH`` already
    exists, so the function is safe to call more than once on the
    same Dataset.

    Returns
    -------
    bool
        True if ``val_SWH``/``val_var_code`` were (just now, or already)
        present.
    """
    merged_col = f"val_{WAVE_HEIGHT_MERGED_VAL_VAR}"
    if merged_col in collocation_ds:
        return True

    present = [c for c in WAVE_HEIGHT_VAL_VARS if f"val_{c}" in collocation_ds]
    if not present:
        return False

    n = collocation_ds.sizes.get("collocation", 0)
    combined = np.full(n, np.nan)
    var_code = np.full(n, "", dtype=object)
    attrs: dict = {}
    for code in present:
        da = collocation_ds[f"val_{code}"]
        col = np.asarray(da.values, dtype=float)
        unclaimed = var_code == ""
        mask = unclaimed & ~np.isnan(col)
        combined[mask] = col[mask]
        var_code[mask] = code
        if not attrs and da.attrs:
            attrs = dict(da.attrs)

    collocation_ds[merged_col] = ("collocation", combined)
    if attrs:
        long_name = attrs.get("long_name", "significant wave height")
        collocation_ds[merged_col].attrs = {
            **attrs,
            "long_name": f"{long_name} (combined {'/'.join(present)} — see val_var_code)",
        }
    collocation_ds["val_var_code"] = ("collocation", var_code)
    collocation_ds["val_var_code"].attrs = {
        "long_name": "originating wave-height validation code for this row "
                      f"({'/'.join(WAVE_HEIGHT_VAL_VARS)})",
    }
    return True


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
    1. Merges every available wave-height validation parameter (VHM0,
       VAVH, VGHS) into one combined ``val_SWH`` column, in place on
       *collocation_ds* (see :func:`merge_wave_height_columns`), so a
       recipe with e.g. both ERA5 (VHM0) and altimeter (VAVH) gets one
       comparison group instead of a separate report section per code
    2. Picks the primary SAR wave variable by fallback (oswTotalHs, else
       oswHs), and additionally includes owiSignificantWaveHeight whenever 
       that column exists and has at least one non-NaN value
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

    # For waves: merge every available wave-height validation parameter
    # (VHM0/VAVH/VGHS) into one combined comparison group instead of a
    # separate section per code.
    if variable == "waves":
        merge_wave_height_columns(collocation_ds)

        # Primary SAR wave-height variable: single-winner fallback driven by
        # which sar_<name> column exists in collocation_ds. 
        primary_candidates = ["oswTotalHs", "oswHs"]
        primary_var = next(
            (v for v in primary_candidates if f"sar_{v}" in collocation_ds),
            None,
        )

        sar_vars = [primary_var] if primary_var is not None else []

        # owiSignificantWaveHeight is additive, not a fallback: it is a
        # different measurement (IW/EW grid product) in principle not provided.
        # When it carries data it gets its own statistics alongside the
        # primary variable rather than replacing it. 
        owi_col = "sar_owiSignificantWaveHeight"
        if owi_col in collocation_ds and bool(collocation_ds[owi_col].notnull().any()):
            sar_vars.append("owiSignificantWaveHeight")

        pairs = [(sv, WAVE_HEIGHT_MERGED_VAL_VAR) for sv in sar_vars]
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
