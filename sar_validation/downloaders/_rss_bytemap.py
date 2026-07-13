"""
Decoder for RSS binary bytemap radiometer products (GMI, SSMIS, WindSat).

Remote Sensing Systems distributes these daily ocean products as gzipped
``uint8`` "bytemaps" on a 0.25° global grid — a different format from the
AMSR2 NetCDF handled elsewhere. RSS ships Python-2 read routines for them;
this module is a compact, Python-3 re-implementation of the same format
(no RSS code copied), driven by a per-sensor layout table.

Format (confirmed against RSS's read routines *and* empirically against real
GMI / SSMIS / WindSat files):

- Uncompressed layout is ``(npass=2, nvar, nlat=720, nlon=1440)`` uint8, with
  ascending/descending as the two passes.
- Grid: lon ``0.125..359.875`` (0–360 convention), lat ``-89.875..89.875``
  (south→north; index 0 is the southernmost row — no flip needed).
- Physical value = ``byte * scale + offset``.
- Byte values ``>= 251`` are special/missing codes (land, sea-ice, coast,
  rain-flagged, no-observation, bad) and decode to ``NaN``.

The reader decodes every variable generically; the converter
(:func:`DataTreeConverter.from_radiometer_bytemap`) keeps only wind speed,
wind direction (WindSat) and the per-cell time.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

__all__ = ["BYTEMAP_LAYOUT", "read_rss_bytemap"]

NPASS, NLAT, NLON = 2, 720, 1440

# Grid cell-centre coordinates (identical across all RSS 0.25° bytemaps).
LON_1D = np.arange(NLON) * 0.25 + 0.125    # 0.125 .. 359.875
LAT_1D = np.arange(NLAT) * 0.25 - 89.875   # -89.875 .. 89.875

# Byte codes at or above this value are special/missing (masked to NaN).
MISSING_CODE_MIN = 251

# ---------------------------------------------------------------------------
# Per-sensor variable layout.
#
# ``vars`` is the ordered list of (name, scale, offset) for the file's
# geophysical variables (physical = byte*scale + offset). ``wind`` / ``wdir`` /
# ``time`` name the variables the converter consumes; ``time_unit`` is the unit
# of the time-of-day variable ("hours" for fractional GMT hours, "minutes" for
# minutes since midnight — WindSat). ``wdir`` is None for sensors without a
# wind-direction retrieval.
#
# Scale/offsets and variable order are taken from RSS's read routines:
#   GMI     — gmi/support_v08.2/python/gmi_daily_v8.py
#   SSMIS   — ssmi/ssmi_support/python/ssmis_daily_v7.py
#   WindSat — windsat/support_v07.0.1/python/windsat_daily_v7.py
# ---------------------------------------------------------------------------
BYTEMAP_LAYOUT: Dict[str, dict] = {
    "gmi": {
        "vars": [
            ("time",   0.1,  0.0),
            ("sst",    0.15, -3.0),
            ("windLF", 0.2,  0.0),
            ("windMF", 0.2,  0.0),
            ("vapor",  0.3,  0.0),
            ("cloud",  0.01, -0.05),
            ("rain",   0.1,  0.0),
        ],
        "wind": "windLF", "wdir": None, "time": "time", "time_unit": "hours",
    },
    # SSMIS f16/f17/f18 share one layout (medium-frequency wind only).
    "ssmis_f16": None,  # filled below
    "ssmis_f17": None,
    "ssmis_f18": None,
    "windsat": {
        "vars": [
            ("mingmt", 6.0,  0.0),
            ("sst",    0.15, -3.0),
            ("w-lf",   0.2,  0.0),
            ("w-mf",   0.2,  0.0),
            ("vapor",  0.3,  0.0),
            ("cloud",  0.01, -0.05),
            ("rain",   0.1,  0.0),
            ("w-aw",   0.2,  0.0),
            ("wdir",   1.5,  0.0),
        ],
        "wind": "w-lf", "wdir": "wdir", "time": "mingmt", "time_unit": "minutes",
    },
}

_SSMIS_LAYOUT = {
    "vars": [
        ("time",    0.1,  0.0),
        ("wspd_mf", 0.2,  0.0),
        ("vapor",   0.3,  0.0),
        ("cloud",   0.01, -0.05),
        ("rain",    0.1,  0.0),
    ],
    "wind": "wspd_mf", "wdir": None, "time": "time", "time_unit": "hours",
}
for _f in ("ssmis_f16", "ssmis_f17", "ssmis_f18"):
    BYTEMAP_LAYOUT[_f] = _SSMIS_LAYOUT


def read_rss_bytemap(
    path: Union[str, Path],
    sensor: str,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Decode an RSS bytemap ``.gz`` file into physical-valued arrays.

    Parameters
    ----------
    path : str or Path
        Path to the gzipped bytemap file.
    sensor : str
        Sensor key present in :data:`BYTEMAP_LAYOUT` (``gmi``, ``ssmis_f16``,
        ``ssmis_f17``, ``ssmis_f18``, ``windsat``).

    Returns
    -------
    (decoded, lon, lat)
        ``decoded`` maps each variable name to a ``(npass, nlat, nlon)`` float
        array with special/missing codes set to NaN; ``lon`` (0–360) and
        ``lat`` (−90→90) are the 1-D cell-centre coordinates.

    Raises
    ------
    KeyError
        If *sensor* is not in :data:`BYTEMAP_LAYOUT`.
    ValueError
        If the decompressed size doesn't match the expected variable count.
    """
    if sensor not in BYTEMAP_LAYOUT:
        raise KeyError(
            f"Unknown bytemap sensor '{sensor}'. Known: {sorted(BYTEMAP_LAYOUT)}"
        )
    layout = BYTEMAP_LAYOUT[sensor]
    variables = layout["vars"]
    nvar = len(variables)

    with gzip.open(path, "rb") as fh:
        raw = np.frombuffer(fh.read(), dtype=np.uint8)

    expected = NPASS * nvar * NLAT * NLON
    if raw.size != expected:
        raise ValueError(
            f"{path}: expected {expected} bytes for {sensor} "
            f"({nvar} vars × {NPASS} passes × {NLAT} × {NLON}), got {raw.size}."
        )

    arr = raw.reshape(NPASS, nvar, NLAT, NLON).astype(np.float64)

    decoded: Dict[str, np.ndarray] = {}
    for i, (name, scale, offset) in enumerate(variables):
        band = arr[:, i]  # (npass, nlat, nlon)
        decoded[name] = np.where(band >= MISSING_CODE_MIN, np.nan, band * scale + offset)

    return decoded, LON_1D.copy(), LAT_1D.copy()
