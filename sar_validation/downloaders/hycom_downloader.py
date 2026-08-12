"""
Download HyCOM ocean-current model data (water_u/water_v, surface level)
via THREDDS OPeNDAP.

- ESPC-D-V02 (2024-08-10 -> present): one continuous OPeNDAP dataset per
  component, https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/u3z (+ v3z).
- GOFS 3.1 Analysis, GLBy0.08/expt_93.0 (2018-12-04 -> 2024-09-04): one
  COMBINED water_u+water_v dataset PER CALENDAR YEAR,
  https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z/{year}.

HyCOM coverage in this toolbox starts 2018-12-04. A recipe window ending 
before 2018-12-04 is a clear error, not a silent no-op.

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

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .base import build_output_dir, normalize_datetime

if TYPE_CHECKING:
    import xarray as xr

logger = logging.getLogger(__name__)

__all__ = ["HycomDownloader", "_resolve_hycom_segments", "main"]

#: First date this toolbox has verified HyCOM coverage for -- the start of
#: GLBy0.08/expt_93.0, the one continuously-verified GOFS 3.1 Analysis
#: dataset.
_HYCOM_MIN_DATE = datetime(2018, 12, 4)

#: ESPC-D-V02 is preferred from this date onward (it's the more current
#: model); GOFS 3.1 Analysis (expt_93.0) covers everything strictly
#: before it, down to _HYCOM_MIN_DATE.
_HYCOM_CUTOVER_DATE = datetime(2024, 8, 10)

#: Grid padding (degrees) beyond the recipe bbox on every edge, so
#: bilinear spatial interpolation at collocation time never has to
#: extrapolate at a SAR scene's edges -- one native grid cell (1/12 deg).
_GRID_PAD_DEG = 1.0 / 12.0

#: Both real HyCOM THREDDS datasets (GOFS 3.1 / expt_93.0 and ESPC-D-V02) 
#: carry a ``tau`` (forecast lead-time) data variable whose ``units`` 
#: attribute is the CF-noncompliant string "hours since analysis". As 
#: ``tau`` is never consumed downstream (only water_u/water_v/time/lat
#: /lon/depth are used), it is excluded.
_DROP_VARS = ["tau"]

#: HyCOM granule cadence (hours) -- both real datasets (ESPC-D-V02 and
#: GOFS 3.1 Analysis / expt_93.0) emit one synoptic snapshot every 3
#: hours (00, 03, 06, ... UTC)
_CADENCE_HOURS = 3

#: Margin (hours), on top of whatever [seg_start, seg_end] window this
#: downloader is handed, added on BOTH sides before building the actual
#: OPeNDAP ``time=slice(...)`` request -- mirrors
#: ``era5_downloader._HOUR_BUFFER``, sized to HyCOM's 3-hourly (not
#: ERA5's 1-hourly) cadence, and applied by this downloader itself since
#: the generic orchestrator/recipe-level padding
#: (``DEFAULT_LAYER_TYPE_SPECS["hycom"]["time_tolerance_minutes"]``) is
#: not read by ``ModelLayerCollocation`` and is not sized for
#: bracket-finding at all.
#:
#: Derivation (traced against ``ModelLayerCollocation._model_values_at_
#: points``'s ``floor_hour``/``searchsorted``/``idx2`` logic, not just
#: estimated): for an observation time T, ``floor_hour`` truncates T down
#: to the nearest HOUR (not cadence) boundary, and the bracket centre
#: ``t2`` is the largest actual granule <= ``floor_hour``. Since HyCOM
#: granules fall on hour boundaries that are multiples of the cadence,
#: ``floor_hour - t2`` is at most ``cadence - 1`` hours, and
#: ``T - floor_hour`` is less than 1 hour, so ``T - t2 < cadence`` always
#: (e.g. cadence=3h, T=11:59:59, granules at 09:00/12:00 -> t2=09:00,
#: T - t2 = 2h59m59s < 3h). The bracket needed is
#: ``[t2 - cadence, t2, t2 + cadence]``, so in the worst case the
#: EARLIEST granule needed is up to just-under ``2 * cadence`` before T
#: (``t2 - cadence``, with ``t2`` up to just-under ``cadence`` before T),
#: and the LATEST granule needed is up to exactly ``cadence`` after T
#: (when T lands exactly on a granule, ``t2 == T``). A symmetric
#: ``2 * cadence`` buffer on both sides covers this worst case at every
#: recipe-window edge (slightly generous on the "after" side, which only
#: strictly needs ``1 * cadence`` -- simple, safe, harmless extra data).
_BRACKET_BUFFER_HOURS = 2 * _CADENCE_HOURS

#: Tiny margin subtracted from :data:`_HYCOM_CUTOVER_DATE` when clipping
#: the ``gofs31_930`` segment's buffered request (see
#: :meth:`HycomDownloader._buffered_bounds`) so its ``.sel(time=slice
#: (..., stop))`` excludes the cutover instant itself -- a plain
#: ``stop = _HYCOM_CUTOVER_DATE`` would still match a granule landing
#: exactly ON the cutover (a real possibility since granules fall on
#: cadence-hour boundaries and the cutover is midnight), which is
#: reserved for ``espc_d_v02`` (see module docstring: "ESPC-D-V02 is
#: preferred from this date onward"). Far smaller than
#: :data:`_CADENCE_HOURS` (3h = 10800s), so it can never accidentally
#: exclude/include the wrong granule.
_CUTOVER_CLIP_EPSILON = timedelta(seconds=1)


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


def _select_hycom_lon_window(
    merged: "xr.Dataset", west: float, east: float, time_lat_sel: dict,
) -> "xr.Dataset":
    """
    Select *merged*'s ``lon``/``time``/``lat`` window, converting *west*/
    *east* (this toolbox's standard -180..180 convention, already padded
    by :data:`_GRID_PAD_DEG`) into HyCOM's REAL native 0-360 ``lon``
    convention first.
    """
    import xarray as xr

    west_360 = west % 360.0
    east_360 = east % 360.0

    if west_360 <= east_360:
        return merged.sel(lon=slice(west_360, east_360), **time_lat_sel)

    seg_a = merged.sel(lon=slice(west_360, 360.0), **time_lat_sel)
    seg_b = merged.sel(lon=slice(0.0, east_360), **time_lat_sel)
    seg_b = seg_b.assign_coords(lon=seg_b["lon"] + 360.0)
    combined = xr.concat([seg_a, seg_b], dim="lon")
    return combined.drop_duplicates("lon", keep="first")


class HycomDownloader:
    """
    Download HyCOM surface current velocity (water_u/water_v, depth=0.0)
    via THREDDS OPeNDAP, using xarray's lazy ``.sel()`` subsetting so only
    the requested bbox/time/depth slice is pulled over the network -- see
    module docstring for the two datasets this spans.

    Parameters
    ----------
    output_dir : Path
        Directory to save downloaded NetCDF files.
    dry_run : bool
        If True, probe each resolved dataset's ``time`` coordinate only
        (cheap -- no full u/v grid load) and report real day-level
        coverage without downloading.
    """

    def __init__(self, output_dir: Path, dry_run: bool = False) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run

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
        Download one combined NetCDF per HyCOM dataset-segment touched by
        [*start*, *end*] (see :func:`_resolve_hycom_segments`).

        Parameters
        ----------
        min_lon, max_lon, min_lat, max_lat : float
            Geographic bounds -- padded internally by one native grid
            cell (see :data:`_GRID_PAD_DEG`).
        start, end : str
            ISO-8601 date or datetime strings.

        Returns
        -------
        list[Path]
            Paths to downloaded (or already-cached) NetCDF files.
        """
        window_start = datetime.fromisoformat(normalize_datetime(start))
        window_end = datetime.fromisoformat(normalize_datetime(end))
        segments = _resolve_hycom_segments(window_start, window_end)
        # Two segments only ever happens for a window straddling
        # _HYCOM_CUTOVER_DATE (see _resolve_hycom_segments) -- only THEN
        # do the two segments' buffered requests risk overlapping each
        # other in real-world time, so only then must _buffered_bounds
        # clip at the cutover (see its docstring). A single-segment
        # window must NOT be clipped -- there is no other segment/file
        # to conflict with, so clipping would just needlessly shrink the
        # bracket buffer near the cutover for no benefit.
        clip_at_cutover = len(segments) == 2

        self.output_dir.mkdir(parents=True, exist_ok=True)

        downloaded: list[Path] = []
        for dataset_key, seg_start, seg_end in segments:
            nc_path = self._nc_path_for_segment(dataset_key, seg_start, seg_end)
            if nc_path.exists():
                logger.info("  %s: already present (%s), skipping.", dataset_key, nc_path.name)
                downloaded.append(nc_path)
                continue

            if self.dry_run:
                self._probe_coverage(dataset_key, seg_start, seg_end, clip_at_cutover)
                continue

            result = self._download_segment(
                dataset_key, seg_start, seg_end, min_lon, max_lon, min_lat, max_lat,
                clip_at_cutover,
            )
            if result is not None:
                downloaded.append(result)

        return downloaded

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _nc_path_for_segment(self, dataset_key: str, seg_start: datetime, seg_end: datetime) -> Path:
        return self.output_dir / (
            f"hycom_{dataset_key}_{seg_start:%Y%m%dT%H%M%S}_{seg_end:%Y%m%dT%H%M%S}.nc"
        )

    @staticmethod
    def _buffered_bounds(
        dataset_key: str, seg_start: datetime, seg_end: datetime, clip_at_cutover: bool = False,
    ) -> tuple[datetime, datetime]:
        """Widen [*seg_start*, *seg_end*] by :data:`_BRACKET_BUFFER_HOURS`
        on both sides for the actual OPeNDAP request.

        Note: the *nc_path* filename (see :meth:`_nc_path_for_segment`)
        deliberately stays keyed on the UNBUFFERED *seg_start*/*seg_end*
        -- only the actual network request widens.

        *clip_at_cutover* (set by :meth:`download` only when a window
        straddles :data:`_HYCOM_CUTOVER_DATE`, i.e. two segments are both
        in play for this call) additionally clips the widened bounds so
        they never cross INTO the other dataset's own preferred date
        range: ``gofs31_930``'s buffered ``stop`` never reaches
        :data:`_HYCOM_CUTOVER_DATE` or beyond, and ``espc_d_v02``'s
        buffered ``start`` never goes before it.

        Without this, for a straddling window the ``gofs31_930``
        segment's buffered request (widened forward) and the
        ``espc_d_v02`` segment's buffered request (widened backward)
        genuinely overlap in real-world time -- both real HyCOM datasets
        (GOFS 3.1 Analysis / expt_93.0, verified through 2024-09-04, and
        ESPC-D-V02) carry real, but DIFFERENT, independently-run model
        data for the same dates in that overlap window, so
        ``DataTreeConverter.from_hycom``'s ``xr.concat(..., dim="time")``
        would produce duplicate, non-monotonic timestamps that
        ``model_collocation.py``'s ``np.searchsorted``-based bracket
        search has no defined/correct behaviour for -- effectively a
        coin flip over which model's value is actually used for a scene
        near the boundary, silently violating this toolbox's stated
        ESPC-D-V02-preferred-at/after-cutover rule.
        """
        buffered_start = max(
            seg_start - timedelta(hours=_BRACKET_BUFFER_HOURS), _HYCOM_MIN_DATE,
        )
        buffered_end = seg_end + timedelta(hours=_BRACKET_BUFFER_HOURS)
        if clip_at_cutover:
            if dataset_key == "gofs31_930":
                buffered_end = min(buffered_end, _HYCOM_CUTOVER_DATE - _CUTOVER_CLIP_EPSILON)
            elif dataset_key == "espc_d_v02":
                buffered_start = max(buffered_start, _HYCOM_CUTOVER_DATE)
        return buffered_start, buffered_end

    def _dodsc_urls(self, dataset_key: str, seg_start: datetime, seg_end: datetime) -> dict[str, str]:
        """
        Map of ``{label: dodsC url}`` to open for this segment.

        ``espc_d_v02`` -> one continuous dataset per component (``"u"``,
        ``"v"`` labels). ``gofs31_930`` -> one combined water_u+water_v
        dataset per calendar year touched by [*seg_start*, *seg_end*]
        (``"uv_<year>"`` labels) -- see module docstring for why these two
        datasets have different shapes on the wire.
        """
        if dataset_key == "espc_d_v02":
            base = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02"
            return {"u": f"{base}/u3z", "v": f"{base}/v3z"}

        base = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        years = range(seg_start.year, seg_end.year + 1)
        return {f"uv_{year}": f"{base}/{year}" for year in years}

    def _probe_coverage(
        self, dataset_key: str, seg_start: datetime, seg_end: datetime, clip_at_cutover: bool = False,
    ) -> None:
        """Open each URL for this segment and report which requested
        days have granules in the live dataset's ``time`` coordinate 
        (`_buffered_bounds`-widened window) -- no full u/v grid load. 
        Analogous to ``noaa_hfradar_thredds_downloader.py``'s 
        ``catalog.xml`` probe. Used for the ``--dry-run`` CLI option.
        """
        import numpy as np
        import xarray as xr

        buffered_start, buffered_end = self._buffered_bounds(
            dataset_key, seg_start, seg_end, clip_at_cutover,
        )

        for label, url in self._dodsc_urls(dataset_key, buffered_start, buffered_end).items():
            try:
                remote = xr.open_dataset(url, drop_variables=_DROP_VARS)
            except Exception as exc:  # noqa: BLE001 — remote OPeNDAP errors are broad
                logger.warning("  [dry-run] %s (%s): could not open %s: %s", dataset_key, label, url, exc)
                continue
            times = remote.time.values
            remote.close()
            in_window = times[
                (times >= np.datetime64(buffered_start)) & (times <= np.datetime64(buffered_end))
            ]
            logger.info(
                "  [dry-run] %s (%s): %d granule(s) in [%s, %s]",
                dataset_key, label, len(in_window), buffered_start, buffered_end,
            )

    def _download_segment(
        self,
        dataset_key: str,
        seg_start: datetime,
        seg_end: datetime,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        clip_at_cutover: bool = False,
    ) -> Optional[Path]:
        """Download one dataset-segment's water_u/water_v subset, combine
        into one NetCDF. Returns the NC path, or ``None`` on failure.
        """
        import xarray as xr

        west = min_lon - _GRID_PAD_DEG
        east = max_lon + _GRID_PAD_DEG
        south = min_lat - _GRID_PAD_DEG
        north = max_lat + _GRID_PAD_DEG

        nc_path = self._nc_path_for_segment(dataset_key, seg_start, seg_end)
        buffered_start, buffered_end = self._buffered_bounds(
            dataset_key, seg_start, seg_end, clip_at_cutover,
        )
        urls = self._dodsc_urls(dataset_key, buffered_start, buffered_end)

        # A real OPeNDAP fetch below (xr.open_dataset + .load()) can take a
        # while over the network with zero output in between -- without
        # this, a slow segment looks indistinguishable from a hang. Mirrors
        # cds_soil_moisture_downloader._download_day's
        # "requesting CDS ... SSM" notice before its own comparably slow
        # network request.
        logger.info(
            "  %s: requesting HyCOM %s for [%s, %s] ...",
            dataset_key, "/".join(sorted(urls)), buffered_start, buffered_end,
        )

        opened: dict[str, xr.Dataset] = {}
        try:
            try:
                for label, url in urls.items():
                    opened[label] = xr.open_dataset(url, drop_variables=_DROP_VARS)

                if dataset_key == "espc_d_v02":
                    merged = xr.merge([opened["u"][["water_u"]], opened["v"][["water_v"]]])
                else:
                    per_year = [opened[label][["water_u", "water_v"]] for label in sorted(opened)]
                    merged = per_year[0] if len(per_year) == 1 else xr.concat(per_year, dim="time")

                time_lat_sel = {"time": slice(buffered_start, buffered_end), "lat": slice(south, north)}
                subset = _select_hycom_lon_window(merged, west, east, time_lat_sel)
                subset = subset.sel(depth=0.0, method="nearest")
                subset = subset.load()
            finally:
                for ds in opened.values():
                    ds.close()
        except Exception as exc:  # noqa: BLE001 — remote OPeNDAP errors are broad
            logger.warning("  %s: HyCOM OPeNDAP download failed: %s", dataset_key, exc)
            return None

        subset.to_netcdf(nc_path)
        return nc_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download HyCOM surface current data.")
    p.add_argument("--min-lon", type=float, required=True)
    p.add_argument("--max-lon", type=float, required=True)
    p.add_argument("--min-lat", type=float, required=True)
    p.add_argument("--max-lat", type=float, required=True)
    p.add_argument("--start", required=True, help="Start date/datetime (ISO-8601).")
    p.add_argument("--end", required=True, help="End date/datetime (ISO-8601, inclusive).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory (default: data/<timerange>_<bounds>/hycom).")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    out_dir = Path(args.output_dir) if args.output_dir else (
        build_output_dir(args.start, args.end, args.min_lon, args.max_lon,
                         args.min_lat, args.max_lat) / "hycom"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    dl = HycomDownloader(output_dir=out_dir, dry_run=args.dry_run)
    dl.download(
        min_lon=args.min_lon, max_lon=args.max_lon,
        min_lat=args.min_lat, max_lat=args.max_lat,
        start=args.start, end=args.end,
    )


if __name__ == "__main__":
    main()
