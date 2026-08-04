"""
Download C3S CDS satellite soil moisture products via the ``cdsapi`` library.

Supports the three sensor-class variants of the
``satellite-soil-moisture`` dataset:

* ``"active"``   — ASCAT multi-scatterometer composite, 0.25°, daily, [%]
* ``"passive"``  — Multi-radiometer composite, 0.25°, daily, [m³ m⁻³]
* ``"combined"`` — Merged active + passive, 0.25°, daily, [m³ m⁻³]

Credentials are read automatically from ``~/.cdsapirc`` (the standard
Copernicus CDS API key file). No OS-keyring wiring is needed — this
differs from other downloaders intentionally, because CDS uses its own
standard credential file. See README §Credentials for registration
instructions.

No server-side bbox filtering — the CDS delivers a global 0.25° grid per
request; spatial sub-setting is performed downstream by
``DataTreeConverter.from_c3s_ssm()``.

The downloader requests each day independently so partially-complete
time ranges succeed without skipping any available days, and individual
day failures can be logged without aborting the whole batch.

Library usage::

    from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader
    dl = CDSSoilMoistureDownloader(product_type="active",
                                   output_dir=Path("data/run1/cds_ssm"))
    dl.download(min_lon=-10, max_lon=10, min_lat=40, max_lat=55,
                start="2026-01-01", end="2026-01-02")

CLI usage::

    python -m sar_validation.downloaders.cds_soil_moisture_downloader \\
        --product-type active \\
        --min-lon -10 --max-lon 10 --min-lat 40 --max-lat 55 \\
        --start 2026-01-01 --end 2026-01-02
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Literal, Optional

from .base import build_output_dir, normalize_datetime

logger = logging.getLogger(__name__)

__all__ = ["CDSSoilMoistureDownloader"]

#: CDS dataset identifier for the satellite soil moisture product.
_CDS_DATASET = "satellite-soil-moisture"

#: Latest CDS product version.  Update when CDS publishes a new version.
_CDS_VERSION = "v202505"

ProductType = Literal["active", "passive", "combined"]


class CDSSoilMoistureDownloader:
    """
    Download C3S CDS satellite soil moisture products via ``cdsapi``.

    Parameters
    ----------
    product_type : str
        Sensor-class variant: ``"active"``, ``"passive"``, or
        ``"combined"``.
    output_dir : Path
        Directory to save downloaded NetCDF files.
    dry_run : bool
        If True, log what would be downloaded without calling the CDS API.
    version : str, optional
        CDS product version string (default: :data:`_CDS_VERSION`).
    """

    def __init__(
        self,
        product_type: ProductType,
        output_dir: Path,
        dry_run: bool = False,
        version: Optional[str] = None,
    ) -> None:
        if product_type not in ("active", "passive", "combined"):
            raise ValueError(
                f"product_type must be 'active', 'passive', or 'combined'; got {product_type!r}"
            )
        self.product_type = product_type
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self._version = version or _CDS_VERSION

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
        Download daily CDS soil moisture files for the date range
        [*start*, *end*) (end-exclusive, matching the convention used
        across the toolbox).

        Parameters
        ----------
        min_lon, max_lon, min_lat, max_lat : float
            Geographic bounds (informational only — not passed to CDS;
            spatial filtering happens downstream).
        start, end : str
            ISO-8601 date or datetime strings.

        Returns
        -------
        list[Path]
            Paths to downloaded (extracted) NetCDF files.
        """
        start_date = _parse_date(normalize_datetime(start))
        end_date = _parse_date(normalize_datetime(end))

        self.output_dir.mkdir(parents=True, exist_ok=True)

        downloaded: list[Path] = []
        day = start_date
        while day < end_date:
            nc_path = self._nc_path_for_day(day)
            if nc_path.exists():
                logger.info("  %s: already present (%s), skipping.", day.isoformat(), nc_path.name)
                downloaded.append(nc_path)
                day += timedelta(days=1)
                continue

            if self.dry_run:
                logger.info("  [dry-run] would download CDS %s SSM for %s", self.product_type, day.isoformat())
                day += timedelta(days=1)
                continue

            nc_path_or_none = self._download_day(day)
            if nc_path_or_none is not None:
                downloaded.append(nc_path_or_none)
            day += timedelta(days=1)

        return downloaded

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _nc_path_for_day(self, day: date) -> Path:
        """Return the expected output NC path for *day*."""
        return self.output_dir / f"c3s_ssm_{self.product_type}_{day.strftime('%Y%m%d')}.nc"

    def _build_request(self, day: date) -> dict:
        """Build the cdsapi request dict for a single *day*."""
        # The CDS API requires month/day as zero-padded strings.
        return {
            "variable": ["surface_soil_moisture_volumetric"],
            "type_of_sensor": [self.product_type],
            "time_aggregation": ["daily"],
            "year": [str(day.year)],
            "month": [f"{day.month:02d}"],
            "day": [f"{day.day:02d}"],
            # Use ICDR for recent/NRT data; CDR is available for older dates
            # but ICDR requests succeed even for historical dates (the server
            # returns CDR data when ICDR is not yet available for that day).
            "type_of_record": ["icdr"],
            "version": [self._version],
        }

    def _download_day(self, day: date) -> Optional[Path]:
        """
        Download one day's CDS product and extract the NetCDF from the
        returned zip archive.  Returns the extracted NC path, or ``None``
        on failure.
        """
        try:
            import cdsapi  # noqa: PLC0415 — optional dependency, imported lazily
        except ImportError as exc:
            raise ImportError(
                "cdsapi is required for CDS soil moisture downloads. "
                "Install it with: pip install 'sar-l2-validation-toolbox[soil_moisture]'"
            ) from exc

        zip_path = self.output_dir / f"c3s_ssm_{self.product_type}_{day.strftime('%Y%m%d')}.zip"
        request = self._build_request(day)

        logger.info("  %s: requesting CDS %s SSM …", day.isoformat(), self.product_type)
        try:
            client = cdsapi.Client(quiet=True)
            client.retrieve(_CDS_DATASET, request).download(str(zip_path))
        except Exception as exc:  # noqa: BLE001 — cdsapi raises broad exceptions
            logger.warning("  %s: CDS download failed: %s", day.isoformat(), exc)
            zip_path.unlink(missing_ok=True)
            return None

        nc_path = self._extract_nc(zip_path, day)
        zip_path.unlink(missing_ok=True)
        return nc_path

    def _extract_nc(self, zip_path: Path, day: date) -> Optional[Path]:
        """Extract the first .nc file from *zip_path* to :attr:`output_dir`."""
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
                if not nc_names:
                    logger.warning("  %s: zip contains no .nc files: %s", day.isoformat(), zip_path.name)
                    return None
                # Rename to a stable, predictable filename.
                dest = self._nc_path_for_day(day)
                with zf.open(nc_names[0]) as src, dest.open("wb") as dst:
                    dst.write(src.read())
            logger.info("  %s: saved %s", day.isoformat(), dest.name)
            return dest
        except zipfile.BadZipFile as exc:
            logger.warning("  %s: bad zip from CDS: %s", day.isoformat(), exc)
            return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download C3S CDS satellite soil moisture products.",
    )
    p.add_argument("--product-type", choices=["active", "passive", "combined"],
                   required=True, help="Sensor-class variant.")
    p.add_argument("--min-lon", type=float, required=True)
    p.add_argument("--max-lon", type=float, required=True)
    p.add_argument("--min-lat", type=float, required=True)
    p.add_argument("--max-lat", type=float, required=True)
    p.add_argument("--start", required=True, help="Start date (ISO-8601).")
    p.add_argument("--end", required=True, help="End date (ISO-8601, exclusive).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory (default: data/<timerange>_<bounds>/cds_ssm).")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--version", default=None,
                   help=f"CDS product version (default: {_CDS_VERSION}).")
    return p


def _parse_date(iso_str: str) -> date:
    """Parse ISO datetime string and return just the date part."""
    return date.fromisoformat(iso_str[:10])


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    out_dir = Path(args.output_dir) if args.output_dir else (
        build_output_dir(args.start, args.end, args.min_lon, args.max_lon,
                         args.min_lat, args.max_lat) / "cds_ssm"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    dl = CDSSoilMoistureDownloader(
        product_type=args.product_type,
        output_dir=out_dir,
        dry_run=args.dry_run,
        version=args.version,
    )
    dl.download(
        min_lon=args.min_lon,
        max_lon=args.max_lon,
        min_lat=args.min_lat,
        max_lat=args.max_lat,
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()
