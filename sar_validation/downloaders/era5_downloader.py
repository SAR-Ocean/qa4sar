"""
Download ERA5 reanalysis data (wind, waves, soil moisture) via the
Copernicus Climate Data Store (CDS) ``cdsapi`` client.

Uses the gridded area/bbox CDS datasets -- NOT the single-point
``-timeseries`` datasets -- since the collocation method in
``sar_validation.core.model_collocation`` needs a spatial grid to
bilinearly interpolate over (see
docs/superpowers/specs/2026-08-06-era5-model-validation-design.md).

Credentials are read automatically from ``~/.cdsapirc`` (the standard CDS
API key file), the same file already used by
``cds_soil_moisture_downloader.py``. No OS-keyring wiring needed.

Library usage::

    from sar_validation.downloaders.era5_downloader import ERA5Downloader
    dl = ERA5Downloader(variable="wind", output_dir=Path("data/run1/era5"))
    dl.download(min_lon=-10, max_lon=10, min_lat=40, max_lat=55,
                start="2026-07-12T18:00:00", end="2026-07-12T23:00:00")

CLI usage::

    python -m sar_validation.downloaders.era5_downloader \\
        --variable wind \\
        --min-lon -10 --max-lon 10 --min-lat 40 --max-lat 55 \\
        --start 2026-07-12T18:00:00 --end 2026-07-12T23:00:00
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

__all__ = ["_hours_needed_for_day"]

#: Margin (hours) added on both sides of the recipe's own padded temporal
#: window before it's clipped to a calendar day, giving the hyperbolic
#: method room to find 3 consecutive bracketing hours even at the window's
#: edges.
_HOUR_BUFFER = 2


def _hours_needed_for_day(
    day: date,
    window_start: datetime,
    window_end: datetime,
    buffer_hours: int = _HOUR_BUFFER,
) -> list[int]:
    """
    Return the sorted list of hour-of-day integers (0-23) needed for *day*
    to cover ``[window_start - buffer_hours, window_end + buffer_hours]``.

    Downloads only these hours instead of always requesting the full 24 --
    avoids over-downloading for narrow wind/wave recipe windows, while wide
    soil-moisture windows still naturally end up requesting most/all hours
    of their interior days. No per-variable special-casing needed: the
    same rule produces the right answer for both cases.
    """
    padded_start = window_start - timedelta(hours=buffer_hours)
    padded_end = window_end + timedelta(hours=buffer_hours)

    day_start = datetime(day.year, day.month, day.day, 0)
    day_end = datetime(day.year, day.month, day.day, 23)

    lo = max(day_start, padded_start.replace(minute=0, second=0, microsecond=0))
    hi = min(day_end, padded_end.replace(minute=0, second=0, microsecond=0))

    if lo > hi:
        return []
    return list(range(lo.hour, hi.hour + 1))
