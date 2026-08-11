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

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .base import build_output_dir, normalize_datetime

logger = logging.getLogger(__name__)

__all__ = ["HycomDownloader", "_resolve_hycom_segments", "main"]

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

        self.output_dir.mkdir(parents=True, exist_ok=True)

        downloaded: list[Path] = []
        for dataset_key, seg_start, seg_end in segments:
            nc_path = self._nc_path_for_segment(dataset_key, seg_start, seg_end)
            if nc_path.exists():
                logger.info("  %s: already present (%s), skipping.", dataset_key, nc_path.name)
                downloaded.append(nc_path)
                continue

            if self.dry_run:
                self._probe_coverage(dataset_key, seg_start, seg_end)
                continue

            result = self._download_segment(
                dataset_key, seg_start, seg_end, min_lon, max_lon, min_lat, max_lat,
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

    def _dodsc_urls(self, dataset_key: str, seg_start: datetime, seg_end: datetime) -> dict[str, str]:
        """
        Map of ``{label: dodsC url}`` to open for this segment.

        ``espc_d_v02`` -> one continuous dataset per component (``"u"``,
        ``"v"`` labels). ``gofs31_930`` -> one COMBINED water_u+water_v
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

    def _probe_coverage(self, dataset_key: str, seg_start: datetime, seg_end: datetime) -> None:
        """Lazily open each URL for this segment and report which
        requested days actually have granules in the live dataset's
        ``time`` coordinate -- no full u/v grid load. Analogous to
        ``noaa_hfradar_thredds_downloader.py``'s ``catalog.xml`` probe."""
        import numpy as np
        import xarray as xr

        for label, url in self._dodsc_urls(dataset_key, seg_start, seg_end).items():
            try:
                remote = xr.open_dataset(url)
            except Exception as exc:  # noqa: BLE001 — remote OPeNDAP errors are broad
                logger.warning("  [dry-run] %s (%s): could not open %s: %s", dataset_key, label, url, exc)
                continue
            times = remote.time.values
            remote.close()
            in_window = times[(times >= np.datetime64(seg_start)) & (times <= np.datetime64(seg_end))]
            logger.info(
                "  [dry-run] %s (%s): %d granule(s) in [%s, %s]",
                dataset_key, label, len(in_window), seg_start, seg_end,
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
    ) -> Optional[Path]:
        """Download one dataset-segment's water_u/water_v subset, combine
        into one NetCDF. Returns the NC path, or ``None`` on failure."""
        import xarray as xr

        west = min_lon - _GRID_PAD_DEG
        east = max_lon + _GRID_PAD_DEG
        south = min_lat - _GRID_PAD_DEG
        north = max_lat + _GRID_PAD_DEG

        nc_path = self._nc_path_for_segment(dataset_key, seg_start, seg_end)
        urls = self._dodsc_urls(dataset_key, seg_start, seg_end)

        opened: dict[str, xr.Dataset] = {}
        try:
            try:
                for label, url in urls.items():
                    opened[label] = xr.open_dataset(url)

                if dataset_key == "espc_d_v02":
                    merged = xr.merge([opened["u"][["water_u"]], opened["v"][["water_v"]]])
                else:
                    per_year = [opened[label][["water_u", "water_v"]] for label in sorted(opened)]
                    merged = per_year[0] if len(per_year) == 1 else xr.concat(per_year, dim="time")

                subset = merged.sel(
                    time=slice(seg_start, seg_end),
                    lat=slice(south, north),
                    lon=slice(west, east),
                ).sel(depth=0.0, method="nearest")
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
