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

__all__ = ["VARIABLE_PAIRS", "infer_variable_pairs"]

# ---------------------------------------------------------------------------
# (sar_variable_name, validation_variable_name) pairs per validated quantity
# ---------------------------------------------------------------------------

VARIABLE_PAIRS: dict[str, List[Tuple[str, str]]] = {
    "wind": [
        ("owiWindSpeed",     "WSPD"),
        ("owiWindDirection", "WDIR"),
    ],
    "currents": [
        ("owiEastwardCurrent",   "EWCT"),
        ("owiNorthwardCurrent",  "NSCT"),
    ],
    "waves": [
        ("owiSignificantWaveHeight", "VHM0"),
    ],
}


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
