"""
Download HyCOM ocean-current model data (water_u/water_v, surface level)
via THREDDS OPeNDAP.

Two HyCOM datasets are wired in, auto-selected by date -- see
docs/superpowers/specs/2026-08-10-hycom-currents-validation-design.md:

- ESPC-D-V02 (2024-08-10 -> present): one continuous OPeNDAP dataset per
  component, https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/u3z (+ v3z).
- GOFS 3.1 Analysis, GLBy0.08/expt_93.0 (2018-12-04 -> 2024-09-04): one
  COMBINED water_u+water_v dataset PER CALENDAR YEAR,
  https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z/{year}.

HyCOM coverage in this toolbox starts 2018-12-04 -- GOFS 3.1 Analysis is
fragmented across several expt_9X.X sub-experiments with unconfirmed
pre-2018-12-04 boundaries; rather than guess at an unverified dataset
path (the exact mistake docs/design-choices.md sections 8.11/10 already
warn against), only the one continuously-verified dataset (expt_93.0) is
wired in. A recipe window ending before 2018-12-04 is a clear error, not
a silent no-op.

Depth is always the surface level (depth=0.0, nearest-match) -- SAR,
HF-radar, and this toolbox's in-situ currents sources are all treated as
near-surface measurements.

Library usage::

    from sar_validation.downloaders.hycom_downloader import HycomDownloader
    dl = HycomDownloader(output_dir=Path("data/run1/hycom"))
    dl.download(min_lon=-10, max_lon=10, min_lat=40, max_lat=55,
                start="2025-01-01T00:00:00", end="2025-01-02T00:00:00")

CLI usage::

    python -m sar_validation.downloaders.hycom_downloader \\
        --min-lon -10 --max-lon 10 --min-lat 40 --max-lat 55 \\
        --start 2025-01-01T00:00:00 --end 2025-01-02T00:00:00
"""

from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

__all__ = ["_resolve_hycom_segments"]

#: First date this toolbox has verified HyCOM coverage for -- the start of
#: GLBy0.08/expt_93.0, the one continuously-verified GOFS 3.1 Analysis
#: dataset. See module docstring.
_HYCOM_MIN_DATE = datetime(2018, 12, 4)

#: ESPC-D-V02 is preferred from this date onward (it's the more current
#: model); GOFS 3.1 Analysis (expt_93.0) covers everything strictly
#: before it, down to _HYCOM_MIN_DATE.
_HYCOM_CUTOVER_DATE = datetime(2024, 8, 10)

#: Grid padding (degrees) beyond the recipe bbox on every edge, so
#: bilinear spatial interpolation at collocation time never has to
#: extrapolate at a SAR scene's edges -- one native grid cell (1/12 deg).
_GRID_PAD_DEG = 1.0 / 12.0


def _resolve_hycom_segments(
    window_start: datetime, window_end: datetime,
) -> list[tuple[str, datetime, datetime]]:
    """
    Split [*window_start*, *window_end*] into at most two
    ``(dataset_key, seg_start, seg_end)`` segments at
    :data:`_HYCOM_CUTOVER_DATE`, where ``dataset_key`` is
    ``"gofs31_930"`` or ``"espc_d_v02"``.

    The window is clamped at :data:`_HYCOM_MIN_DATE`: a window starting
    before it but ending at/after it is silently trimmed to
    ``[_HYCOM_MIN_DATE, window_end]`` (natural "coverage starts here"
    behaviour, matching how this toolbox already pads/clips other
    sources' windows). A window ending before it raises ``ValueError``.
    """
    if window_end < _HYCOM_MIN_DATE:
        raise ValueError(
            "HyCOM coverage for this toolbox starts 2018-12-04 (GOFS 3.1 "
            "Analysis, GLBy0.08/expt_93.0)."
        )
    start = max(window_start, _HYCOM_MIN_DATE)

    if window_end <= _HYCOM_CUTOVER_DATE:
        return [("gofs31_930", start, window_end)]
    if start >= _HYCOM_CUTOVER_DATE:
        return [("espc_d_v02", start, window_end)]
    return [
        ("gofs31_930", start, _HYCOM_CUTOVER_DATE),
        ("espc_d_v02", _HYCOM_CUTOVER_DATE, window_end),
    ]
