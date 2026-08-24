"""
Download ERA5 reanalysis data (wind, waves, soil moisture) via the
Copernicus Climate Data Store (CDS) ``cdsapi`` client.

Uses the gridded area/bbox CDS datasets needed for bilinear interpolation
in ``sar_validation.core.model_collocation``.

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

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal, Optional

from .base import build_output_dir, normalize_datetime, split_antimeridian_bbox

logger = logging.getLogger(__name__)

__all__ = ["_hours_needed_for_day", "ERA5Downloader", "main"]

Era5Variable = Literal["wind", "waves", "soil_moisture"]

#: CDS dataset identifier per ERA5 variable (wind/waves vs soil_moisture
#: come from different datasets).
_CDS_DATASET_BY_VARIABLE: dict[Era5Variable, str] = {
    "wind": "reanalysis-era5-single-levels",
    "waves": "reanalysis-era5-single-levels",
    "soil_moisture": "reanalysis-era5-land",
}

#: Base URL for the CDS catalogue's collection-metadata endpoint, used by
#: check_availability_dry. Mirrors cds_soil_moisture_downloader.py's own
#: ``_CDS_API_URL`` -- kept as a separate module-level constant since the
#: two downloaders are otherwise independent.
_CDS_API_URL = "https://cds.climate.copernicus.eu/api"

#: CDS variable name(s) per ERA5 variable. The ERA5-Land soil moisture
#: variable is named "..._layer_1" (not "..._level_1") in the live CDS
#: API's variable enum. "land_sea_mask" ("lsm" on the wire) is requested
#: alongside u10/v10 for wind only, in the SAME CDS call. It is used as
#: a masking input by ModelLayerCollocation to skip ERA5 grid cells whose
#: center is land, because ERA5's wind field is computed with different
#: surface-roughness/friction physics over land vs. sea, making a land
#: grid point's wind not meaningfully comparable to SAR ocean wind
#: retrieval. "land_sea_mask" is not needed for waves, since a downloaded
#: era5_waves_*.nc already has swh NaN'd over land.
_CDS_VARIABLE_NAMES_BY_VARIABLE: dict[Era5Variable, list[str]] = {
    "wind": ["10m_u_component_of_wind", "10m_v_component_of_wind", "land_sea_mask"],
    "waves": ["significant_height_of_combined_wind_waves_and_swell"],
    "soil_moisture": ["volumetric_soil_water_layer_1"],
}

#: Grid padding (degrees) per variable, used to ensure bilinear
#: interpolation never extrapolates at a SAR scene's edges. ERA5 Land has
#: 0.1° resolution; ERA5 single-levels (wind, waves) have 0.25° resolution.
_GRID_PAD_DEG_BY_VARIABLE: dict[Era5Variable, float] = {
    "wind": 0.25,
    "waves": 0.25,
    "soil_moisture": 0.1,
}

#: Default margin (hours) added on both sides of the requested temporal
#: window before it is clipped to a calendar day, giving the hyperbolic
#: method room to find 3 consecutive bracketing hours even at the window's
#: edges (2x ERA5's 1-hourly cadence, this mirrors hycom_downloader.py's
#: _BRACKET_BUFFER_HOURS derivation). Used only as
#: :meth:`ERA5Downloader.__init__`'s ``time_tolerance_minutes`` default;
#: a recipe-driven run passes the orchestrator's own resolved
#: ``DEFAULT_LAYER_TYPE_SPECS["era5_<variable>"]["time_tolerance_minutes"]``
#: instead.
_HOUR_BUFFER = 2


def _hours_needed_for_day(
    day: date,
    window_start: datetime,
    window_end: datetime,
    buffer_hours: float = _HOUR_BUFFER,
) -> list[int]:
    """
    Return the sorted list of hour-of-day integers (0-23) needed for *day*
    to cover ``[window_start - buffer_hours, window_end + buffer_hours]``.

    Downloads only these hours instead of always requesting the full 24 --
    avoids over-downloading for narrow wind/wave recipe windows, while wide
    soil-moisture windows still naturally end up requesting most/all hours
    of their interior days.
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


class ERA5Downloader:
    """
    Download ERA5 reanalysis data via ``cdsapi``, using the gridded
    area/bbox datasets so the result can feed bilinear spatial
    interpolation (see module docstring).

    Parameters
    ----------
    variable : str
        One of ``"wind"``, ``"waves"``, ``"soil_moisture"``.
    output_dir : Path
        Directory to save downloaded NetCDF files.
    dry_run : bool
        If True, log what would be downloaded without calling the CDS API.
    time_tolerance_minutes : float
        Hour-buffer margin (see :func:`_hours_needed_for_day`) used when
        widening the requested window before it's clipped to a calendar
        day. Defaults to :data:`_HOUR_BUFFER` converted to minutes; a
        recipe-driven run passes the orchestrator's own resolved value
        (``DEFAULT_LAYER_TYPE_SPECS["era5_<variable>"]["time_tolerance_minutes"]``,
        or a recipe override) instead.
    """

    def __init__(
        self, variable: Era5Variable, output_dir: Path, dry_run: bool = False,
        time_tolerance_minutes: float = _HOUR_BUFFER * 60,
    ) -> None:
        if variable not in _CDS_DATASET_BY_VARIABLE:
            raise ValueError(
                f"variable must be one of {sorted(_CDS_DATASET_BY_VARIABLE)}; got {variable!r}"
            )
        self.variable = variable
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.time_tolerance_minutes = time_tolerance_minutes

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def download(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start: str,
        end: str,
    ) -> list[Path]:
        """
        Download daily ERA5 files covering every calendar day whose
        needed-hour range (see :func:`_hours_needed_for_day`) is non-empty
        within [*start*, *end*]. A bbox crossing the antimeridian
        (``min_lon > max_lon``) is split into two non-crossing windows via
        :func:`~sar_validation.downloaders.base.split_antimeridian_bbox`,
        each downloaded to its own ``_w0``/``_w1``-suffixed file per day;
        :meth:`~sar_validation.core.datatree_converter.DataTreeConverter.from_era5`
        stitches them back into one contiguous grid.

        Parameters
        ----------
        min_lon, max_lon, min_lat, max_lat : float
            Geographic bounds -- padded internally by one native grid cell
            (see :meth:`_build_area`).
        start, end : str
            ISO-8601 date or datetime strings.

        Returns
        -------
        list[Path]
            Paths to downloaded (or already-cached) NetCDF files.
        """
        window_start = datetime.fromisoformat(normalize_datetime(start))
        window_end = datetime.fromisoformat(normalize_datetime(end))

        self.output_dir.mkdir(parents=True, exist_ok=True)

        windows = split_antimeridian_bbox(min_lon, max_lon)
        multi_window = len(windows) > 1

        downloaded: list[Path] = []
        buffer = timedelta(hours=self.time_tolerance_minutes / 60.0)
        day = (window_start - buffer).date()
        end_day = (window_end + buffer).date()
        while day <= end_day:
            hours = _hours_needed_for_day(
                day, window_start, window_end, buffer_hours=self.time_tolerance_minutes / 60.0,
            )
            if not hours:
                day += timedelta(days=1)
                continue

            for win_idx, (win_min_lon, win_max_lon) in enumerate(windows):
                idx_arg = win_idx if multi_window else None
                nc_path = self._nc_path_for_day(day, idx_arg)
                if nc_path.exists():
                    logger.info(
                        "  %s (window %s): already present (%s), skipping.",
                        day.isoformat(), idx_arg, nc_path.name,
                    )
                    downloaded.append(nc_path)
                    continue

                if self.dry_run:
                    logger.info(
                        "  [dry-run] would download ERA5 %s for %s window [%.2f, %.2f] (hours %s)",
                        self.variable, day.isoformat(), win_min_lon, win_max_lon, hours,
                    )
                    continue

                result = self._download_day(day, hours, win_min_lon, win_max_lon, min_lat, max_lat, idx_arg)
                if result is not None:
                    downloaded.append(result)

            day += timedelta(days=1)

        return downloaded

    def check_availability_dry(self, day: date) -> bool:
        """
        Whether ERA5 has published data for *day*, without downloading it.

        Queries the CDS catalogue's collection-metadata endpoint (via
        ``ecmwf.datastores.Client.get_collection`` -- a ``cdsapi``
        dependency) for this variable's dataset (see
        :data:`_CDS_DATASET_BY_VARIABLE`) real, live temporal coverage
        extent (``begin_datetime``/``end_datetime``) and checks whether
        *day* falls within it. Never submits a real
        ``cdsapi.Client.retrieve()`` processing job -- the catalogue
        endpoint is a fast, unauthenticated, sub-second metadata lookup
        instead.

        Any failure to determine the extent (missing dependency, network
        error, unexpected response shape) is raised rather than
        swallowed into ``False`` -- callers must treat an exception here
        as "couldn't determine", never as "no data".

        Returns
        -------
        bool
            True if *day* falls within the dataset's live-queried
            begin/end temporal extent.
        """
        try:
            import ecmwf.datastores  # noqa: PLC0415 — optional dependency, imported lazily
        except ImportError as exc:
            logger.debug("check_availability_dry: ecmwf.datastores unavailable: %s", exc)
            raise ImportError(
                "ecmwf-datastores-client (installed automatically alongside cdsapi) is "
                "required for ERA5 availability checks. Install it with: "
                "pip install 'sar-l2-validation-toolbox[era5]'"
            ) from exc

        dataset = _CDS_DATASET_BY_VARIABLE[self.variable]
        try:
            client = ecmwf.datastores.Client(url=_CDS_API_URL)
            collection = client.get_collection(dataset)
            begin = collection.begin_datetime
            end = collection.end_datetime
        except Exception:
            logger.debug(
                "check_availability_dry: catalogue lookup failed for %s (%s)",
                dataset, day.isoformat(), exc_info=True,
            )
            raise

        if begin is None or end is None:
            logger.debug(
                "check_availability_dry: catalogue reported no usable temporal extent "
                "for %s (begin=%r, end=%r)", dataset, begin, end,
            )
            raise RuntimeError(
                f"CDS catalogue returned no usable temporal extent for {dataset!r} "
                f"(begin={begin!r}, end={end!r})."
            )

        return begin.date() <= day <= end.date()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _nc_path_for_day(self, day: date, window_idx: Optional[int] = None) -> Path:
        suffix = f"_w{window_idx}" if window_idx is not None else ""
        return self.output_dir / f"era5_{self.variable}_{day.strftime('%Y%m%d')}{suffix}.nc"

    def _build_area(self, min_lon: float, max_lon: float, min_lat: float, max_lat: float) -> list[float]:
        """
        CDS ``area`` facet: ``[north, west, south, east]``, padded by one
        native grid cell so bilinear interpolation never extrapolates at a
        SAR scene's edges -- clipped at +/-180 so a window that already
        touches the antimeridian (from split_antimeridian_bbox) is never
        padded past it into an invalid CDS area value.
        """
        pad = _GRID_PAD_DEG_BY_VARIABLE[self.variable]
        west = max(min_lon - pad, -180.0)
        east = min(max_lon + pad, 180.0)
        return [max_lat + pad, west, min_lat - pad, east]

    def _build_request(
        self, day: date, hours: list[int],
        min_lon: float, max_lon: float, min_lat: float, max_lat: float,
    ) -> dict:
        """Build the cdsapi request dict for a single *day*."""
        request: dict = {
            "variable": _CDS_VARIABLE_NAMES_BY_VARIABLE[self.variable],
            "year": [str(day.year)],
            "month": [f"{day.month:02d}"],
            "day": [f"{day.day:02d}"],
            "time": [f"{h:02d}:00" for h in hours],
            "area": self._build_area(min_lon, max_lon, min_lat, max_lat),
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        # reanalysis-era5-land has no product_type facet; the atmospheric
        # single-levels dataset requires it (reanalysis vs ensemble members).
        if _CDS_DATASET_BY_VARIABLE[self.variable] != "reanalysis-era5-land":
            request["product_type"] = ["reanalysis"]
        return request

    def _download_day(
        self, day: date, hours: list[int],
        min_lon: float, max_lon: float, min_lat: float, max_lat: float,
        window_idx: Optional[int] = None,
    ) -> Optional[Path]:
        """Download one day's ERA5 file (one antimeridian-split window, if
        *window_idx* is given). Returns the NC path, or ``None`` on
        failure."""
        try:
            import cdsapi  # noqa: PLC0415 — optional dependency, imported lazily
        except ImportError as exc:
            raise ImportError(
                "cdsapi is required for ERA5 downloads. Install it with: "
                "pip install 'sar-l2-validation-toolbox[era5]'"
            ) from exc

        nc_path = self._nc_path_for_day(day, window_idx)
        request = self._build_request(day, hours, min_lon, max_lon, min_lat, max_lat)
        dataset = _CDS_DATASET_BY_VARIABLE[self.variable]

        # print(), not just logger.info(): the CDS request below can take a
        # long time and the CLI's root logger defaults to WARNING (cli.py).
        window_suffix = f" (window {window_idx})" if window_idx is not None else ""
        print(f"  Downloading ERA5 {self.variable} for {day.isoformat()}{window_suffix} …")
        try:
            client = cdsapi.Client(quiet=True)
            client.retrieve(dataset, request).download(str(nc_path))
        except Exception as exc:  # noqa: BLE001 — cdsapi raises broad exceptions
            logger.warning("  %s (window %s): CDS download failed: %s", day.isoformat(), window_idx, exc)
            nc_path.unlink(missing_ok=True)
            return None
        return nc_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download ERA5 reanalysis data.")
    p.add_argument("--variable", choices=["wind", "waves", "soil_moisture"], required=True)
    p.add_argument("--min-lon", type=float, required=True)
    p.add_argument("--max-lon", type=float, required=True)
    p.add_argument("--min-lat", type=float, required=True)
    p.add_argument("--max-lat", type=float, required=True)
    p.add_argument("--start", required=True, help="Start date/datetime (ISO-8601).")
    p.add_argument("--end", required=True, help="End date/datetime (ISO-8601, inclusive).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory (default: data/<timerange>_<bounds>/era5).")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    out_dir = Path(args.output_dir) if args.output_dir else (
        build_output_dir(args.start, args.end, args.min_lon, args.max_lon,
                         args.min_lat, args.max_lat) / "era5"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    dl = ERA5Downloader(variable=args.variable, output_dir=out_dir, dry_run=args.dry_run)
    dl.download(
        min_lon=args.min_lon, max_lon=args.max_lon,
        min_lat=args.min_lat, max_lat=args.max_lat,
        start=args.start, end=args.end,
    )


if __name__ == "__main__":
    main()
