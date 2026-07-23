"""
DataOrchestrator — step 1 of the validation pipeline.

Reads a Recipe and drives all data downloads:
  - SAR L2_OCN data
  - In-situ observations (moorings, buoys, ferryboxes, …)
  - HF radar
  - Scatterometer (ASCAT MetOp-B/C)
  - Altimeter (along-track SWH/wind, Copernicus Marine L3)
  - Radiometer (RSS daily gridded ocean winds, e.g. AMSR2)
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from ..downloaders.base import build_output_dir
from .recipe import Recipe

logger = logging.getLogger(__name__)

__all__ = ["DataOrchestrator"]

# In-situ platform types handled by the InSituDownloader
_INSITU_TYPES = {"mooring", "buoy", "drifter", "ferrybox", "tidal_gauge"}

# Delayed-mode ("historical") source_types, dispatched before any NRT
# source so their actual results can inform whether the NRT counterpart is
# still needed. hf_radar_historical/adcp_historical/argo_historical/
# glider_historical have no NRT pairing entry below (glider/adcp/argo have
# no NRT counterpart at all; hf_radar's pairing is expressed via
# _HISTORICAL_FIRST_PAIRS instead of this set).
_HISTORICAL_FIRST_TYPES = {
    "hf_radar_historical", "adcp_historical", "argo_historical",
    "drifter_historical", "glider_historical",
}

# NRT source_type -> its delayed-mode counterpart. When the historical side
# is present in the recipe and produced at least one file for this run's
# window, the NRT side is skipped (hf_radar) or excluded from the batched
# NRT in-situ call (drifter) rather than re-downloading the same physical
# observations from a lower-QC pipeline.
_HISTORICAL_FIRST_PAIRS = {
    "hf_radar": "hf_radar_historical",
    "drifter": "drifter_historical",
}

# Delayed-mode in-situ current instruments that share a single combined
# "no data" message (see _report_combined_currents_status) instead of each
# logging its own -- fixed order controls the message's word order.
_CURRENTS_INSTRUMENT_TYPES = (
    "adcp_historical", "argo_historical", "drifter_historical", "glider_historical",
)


class DataOrchestrator:
    """
    Orchestrate all downloads for a single validation run.

    Parameters
    ----------
    recipe : Recipe
        Recipe object that specifies what to download.
    dry_run : bool
        If True, print download commands without executing them.
    """

    def __init__(self, recipe: Recipe, dry_run: bool = False, force_download: bool = False) -> None:
        self.recipe   = recipe
        self.dry_run  = dry_run
        self.force_download = force_download
        self.base_dir = self._setup_base_dir()
        self.metadata: Dict[str, Any] = {
            "recipe_name": recipe.config.name,
            "variable":    recipe.config.variable,
            "created":     datetime.now().isoformat(),
            "geographic_bounds": recipe.config.geographic_bounds.to_dict(),
            "temporal_bounds":   recipe.config.temporal_bounds.to_dict(),
            "downloads": {},
            "errors":    [],
            # Non-failure, user-facing observations (e.g. "no data found for
            # this window") -- distinct from "errors", which
            # _is_already_downloaded (cli.py) treats as "something went
            # wrong, don't skip re-download next time". A notice must never
            # trigger that.
            "notices":   [],
        }

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_base_dir(self) -> Path:
        cfg = self.recipe.config
        b = cfg.geographic_bounds
        t = cfg.temporal_bounds

        if cfg.output_dir:
            base = Path(cfg.output_dir)
        else:
            base = build_output_dir(
                t.start, t.end,
                b.min_lon, b.max_lon, b.min_lat, b.max_lat,
            )

        if not self.dry_run:
            base.mkdir(parents=True, exist_ok=True)
            logger.info("Base directory: %s", base)
        return base

    def _cleanup_if_empty(self, out_dir: Path) -> None:
        """Remove out_dir if the download produced no files anywhere under
        it (including nested, otherwise-empty subdirectories). Directories
        that already contain at least one file are left untouched."""
        if out_dir.exists() and not any(p.is_file() for p in out_dir.rglob("*")):
            shutil.rmtree(out_dir)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def download_all(self) -> bool:
        """
        Execute all downloads according to the recipe.

        Returns
        -------
        bool
            True if all downloads completed without errors.
        """
        logger.info("Starting download run: %s", self.recipe.config.name)

        ok = True

        # 1. SAR data
        if not self._download_sar():
            ok = False

        # 2. Delayed-mode ("*_historical") sources first. hf_radar and the
        # NRT in-situ batch (below) consult file_count from these results
        # to avoid re-downloading the same physical observations from the
        # NRT feed once the historical/reprocessed data already covers
        # this run's window.
        historical_had_data: Dict[str, bool] = {}
        for source in self.recipe.config.validation_sources:
            if source.source_type not in _HISTORICAL_FIRST_TYPES:
                continue
            if not self._dispatch_source(source):
                ok = False
            file_count = self.metadata["downloads"].get(
                source.source_type, {}
            ).get("file_count", 0)
            historical_had_data[source.source_type] = file_count > 0

        self._report_combined_currents_status()

        # 3. Group in-situ sources and download as one batch, dropping any
        # NRT type whose paired historical source already covered this
        # window (in practice, only "drifter" is ever a pair key here).
        insitu_sources = [
            s for s in self.recipe.config.validation_sources
            if s.source_type in _INSITU_TYPES
        ]
        source_types = [
            s.source_type for s in insitu_sources
            if not historical_had_data.get(_HISTORICAL_FIRST_PAIRS.get(s.source_type, ""))
        ]
        if source_types:
            # Use the most permissive depth window across the in-situ
            # sources actually being requested (excludes any source dropped
            # above, so an excluded source's depth override can't widen a
            # batch it's no longer part of).
            min_depth = min(
                s.resolved_min_depth for s in insitu_sources
                if s.source_type in source_types
            )
            max_depth = max(
                s.resolved_max_depth for s in insitu_sources
                if s.source_type in source_types
            )
            if not self._download_insitu(source_types, min_depth, max_depth):
                ok = False
        elif insitu_sources:
            logger.info(
                "Skipping NRT in-situ batch: every requested platform type "
                "is covered by a historical source for this window."
            )

        # 4. Other sources one by one
        for source in self.recipe.config.validation_sources:
            if source.source_type in _INSITU_TYPES:
                continue   # handled above
            if source.source_type in _HISTORICAL_FIRST_TYPES:
                continue   # already dispatched in step 2
            paired_historical = _HISTORICAL_FIRST_PAIRS.get(source.source_type)
            if paired_historical and historical_had_data.get(paired_historical):
                self.metadata["downloads"][source.source_type] = {
                    "status": "skipped",
                    "reason": f"covered by {paired_historical}",
                }
                logger.info(
                    "Skipping %s: covered by %s for this window.",
                    source.source_type, paired_historical,
                )
                continue
            if not self._dispatch_source(source):
                ok = False

        if not self.dry_run:
            self._save_metadata()
        return ok

    # ------------------------------------------------------------------
    # Per-source downloaders
    # ------------------------------------------------------------------

    def _download_sar(self) -> bool:
        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        temp   = cfg.temporal_bounds

        if cfg.sar_data.product_level == "L3_SSM":
            from ..downloaders.soil_moisture_downloader import SoilMoistureDownloader

            out_dir = self.base_dir / "S1_L3_SSM"
            try:
                ssm_dl = SoilMoistureDownloader(
                    output_dir=out_dir,
                    dry_run=self.dry_run,
                    force_download=self.force_download,
                )
                paths = ssm_dl.download(
                    min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                    min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                    start=temp.start,      end=temp.end,
                )
                self._cleanup_if_empty(out_dir)
                self.metadata["downloads"]["sar"] = {
                    "status": "dry_run" if self.dry_run else "success",
                    "files":  [str(p) for p in paths],
                }
                return True
            except Exception as exc:
                msg = f"SAR (CLMS SSM) download failed: {exc}"
                logger.error(msg)
                self.metadata["errors"].append(msg)
                self.metadata["downloads"]["sar"] = {"status": "failed", "error": msg}
                return False

        from ..downloaders.sar_downloader import SARDownloader

        out_dir = self.base_dir / "S1_L2_OCN"

        try:
            dl = SARDownloader(
                output_dir=out_dir,
                dry_run=self.dry_run,
                force_download=self.force_download,
            )
            paths = dl.download(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=temp.start,      end=temp.end,
                modes=cfg.sar_data.swath_mode or None,
                limit=cfg.sar_data.max_downloads,
            )
            self._cleanup_if_empty(out_dir)
            self.metadata["downloads"]["sar"] = {
                "status": "dry_run" if self.dry_run else "success",
                "files":  [str(p) for p in paths],
            }
            return True
        except Exception as exc:
            msg = f"SAR download failed: {exc}"
            logger.error(msg)
            self.metadata["errors"].append(msg)
            self.metadata["downloads"]["sar"] = {"status": "failed", "error": msg}
            return False

    def _download_insitu(
        self,
        source_types: list[str],
        min_depth: float,
        max_depth: float,
    ) -> bool:
        from ..downloaders.insitu_downloader import InSituDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        temp   = cfg.temporal_bounds

        out_dir = self.base_dir / "copernicus_insitu"

        try:
            dl = InSituDownloader(
                output_dir=out_dir,
                dry_run=self.dry_run,
                min_depth=min_depth,
                max_depth=max_depth,
                force_download=self.force_download,
            )
            dl.download(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=temp.start,       end=temp.end,
                source_types=source_types,
            )
            self._cleanup_if_empty(out_dir)
            self.metadata["downloads"]["insitu"] = {
                "status":       "dry_run" if self.dry_run else "success",
                "source_types": source_types,
            }
            return True
        except Exception as exc:
            msg = f"In-situ download failed ({source_types}): {exc}"
            logger.error(msg)
            self.metadata["errors"].append(msg)
            self.metadata["downloads"]["insitu"] = {
                "status": "failed", "error": msg, "source_types": source_types
            }
            return False

    def _report_combined_currents_status(self) -> None:
        """Log one combined warning for the four delayed-mode in-situ
        current instruments (adcp/argo/drifter/glider) when every one of
        them present in this recipe produced zero files, instead of each
        instrument logging its own near-identical warning."""
        if self.dry_run:
            return
        attempted = [
            t for t in _CURRENTS_INSTRUMENT_TYPES
            if any(s.source_type == t for s in self.recipe.config.validation_sources)
        ]
        if not attempted:
            return
        all_empty = all(
            self.metadata["downloads"].get(t, {}).get("status") != "failed"
            and self.metadata["downloads"].get(t, {}).get("file_count", 0) == 0
            for t in attempted
        )
        if all_empty:
            names = ", ".join(t.replace("_historical", "") for t in attempted)
            msg = f"No delayed-mode in-situ current data found ({names}) for this window."
            logger.warning(msg)
            self.metadata["notices"].append(msg)

    def _dispatch_source(self, source) -> bool:
        handlers = {
            "scatterometer": self._download_scatterometer,
            "scatterometer_hy2b": self._download_scatterometer_hy2b,
            "scatterometer_hy2c": self._download_scatterometer_hy2c,
            "scatterometer_oceansat3": self._download_scatterometer_oceansat3,
            "hf_radar":      self._download_hf_radar,
            "hf_radar_noaa": self._download_noaa_hfradar,
            "hf_radar_historical": self._download_hf_radar_historical,
            "adcp_historical": self._download_adcp_historical,
            "argo_historical": self._download_argo_historical,
            "drifter_historical": self._download_drifter_historical,
            "glider_historical": self._download_glider_historical,
            "altimeter":     self._download_altimeter,
            "radiometer":    self._download_radiometer,
            "ismn":          self._download_ismn,
        }
        handler = handlers.get(source.source_type)
        if handler is None:
            msg = f"No downloader for source_type '{source.source_type}'"
            logger.warning(msg)
            self.metadata["errors"].append(msg)
            return False
        return handler(source)

    def _download_scatterometer(self, source) -> bool:
        from ..downloaders.scatterometer_downloader import ScatterometerDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        temp   = cfg.temporal_bounds
        out_dir = self.base_dir / "osi_saf_winds"

        try:
            dl = ScatterometerDownloader(
                output_dir=out_dir, dry_run=self.dry_run, force_download=self.force_download,
            )
            dl.download(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=temp.start, end=temp.end,
            )
            self._cleanup_if_empty(out_dir)
            self.metadata["downloads"]["scatterometer"] = {
                "status": "dry_run" if self.dry_run else "success",
            }
            return True
        except Exception as exc:
            msg = f"Scatterometer download failed: {exc}"
            logger.error(msg)
            self.metadata["errors"].append(msg)
            self.metadata["downloads"]["scatterometer"] = {
                "status": "failed", "error": msg
            }
            return False

    def _download_scatterometer_ftp(self, source, satellite: str) -> bool:
        from ..downloaders.scatterometer_ftp_downloader import ScatterometerFTPDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        temp   = cfg.temporal_bounds
        out_dir = self.base_dir / f"scatterometer_{satellite}"

        try:
            dl = ScatterometerFTPDownloader(
                satellite=satellite,
                output_dir=out_dir, dry_run=self.dry_run, force_download=self.force_download,
            )
            dl.download(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=temp.start, end=temp.end,
            )
            self._cleanup_if_empty(out_dir)
            self.metadata["downloads"][f"scatterometer_{satellite}"] = {
                "status": "dry_run" if self.dry_run else "success",
            }
            return True
        except Exception as exc:
            msg = f"{satellite} FTP scatterometer download failed: {exc}"
            logger.error(msg)
            self.metadata["errors"].append(msg)
            self.metadata["downloads"][f"scatterometer_{satellite}"] = {
                "status": "failed", "error": msg
            }
            return False

    def _download_scatterometer_hy2b(self, source) -> bool:
        return self._download_scatterometer_ftp(source, "hy2b")

    def _download_scatterometer_hy2c(self, source) -> bool:
        return self._download_scatterometer_ftp(source, "hy2c")

    def _download_scatterometer_oceansat3(self, source) -> bool:
        return self._download_scatterometer_ftp(source, "oceansat3")

    def _download_hf_radar(self, source) -> bool:
        from ..downloaders.hf_radar_downloader import HFRadarDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        temp   = cfg.temporal_bounds
        out_dir = self.base_dir / "hf_radar"

        try:
            dl = HFRadarDownloader(
                output_dir=out_dir,
                dry_run=self.dry_run,
                min_depth=source.resolved_min_depth,
                max_depth=source.resolved_max_depth,
                force_download=self.force_download,
            )
            dl.download(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=temp.start, end=temp.end,
            )
            self._cleanup_if_empty(out_dir)
            self.metadata["downloads"]["hf_radar"] = {
                "status": "dry_run" if self.dry_run else "success",
            }
            return True
        except Exception as exc:
            msg = f"HF radar download failed: {exc}"
            logger.error(msg)
            self.metadata["errors"].append(msg)
            self.metadata["downloads"]["hf_radar"] = {"status": "failed", "error": msg}
            return False

    def _download_noaa_hfradar(self, source) -> bool:
        from ..downloaders.noaa_hfradar_downloader import (
            DEFAULT_RESOLUTION_KM,
            NOAAHFRadarDownloader,
        )

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        temp   = cfg.temporal_bounds
        out_dir = self.base_dir / "hfr_noaa"
        # Resolution is an optional per-source override, forwarded via the
        # established ValidationDataSource.download_kwargs channel.
        resolution_km = int(source.download_kwargs.get("resolution_km", DEFAULT_RESOLUTION_KM))

        try:
            dl = NOAAHFRadarDownloader(
                output_dir=out_dir,
                dry_run=self.dry_run,
                resolution_km=resolution_km,
                force_download=self.force_download,
            )
            dl.download(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=temp.start, end=temp.end,
            )
            self._cleanup_if_empty(out_dir)
            self.metadata["downloads"]["hf_radar_noaa"] = {
                "status": "dry_run" if self.dry_run else "success",
            }
            return True
        except Exception as exc:
            msg = f"NOAA HF-radar download failed: {exc}"
            logger.error(msg)
            self.metadata["errors"].append(msg)
            self.metadata["downloads"]["hf_radar_noaa"] = {"status": "failed", "error": msg}
            return False

    def _download_hf_radar_historical(self, source) -> bool:
        from ..downloaders.hf_radar_historical_downloader import HFRadarHistoricalDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        temp   = cfg.temporal_bounds
        out_dir = self.base_dir / "hf_radar_historical"

        try:
            dl = HFRadarHistoricalDownloader(
                output_dir=out_dir, dry_run=self.dry_run, force_download=self.force_download,
            )
            downloaded = dl.download(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=temp.start, end=temp.end,
            ) or []
            self._cleanup_if_empty(out_dir)
            self.metadata["downloads"]["hf_radar_historical"] = {
                "status": "dry_run" if self.dry_run else "success",
                "file_count": len(downloaded),
            }
            return True
        except Exception as exc:
            msg = f"HF radar historical download failed: {exc}"
            logger.error(msg)
            self.metadata["errors"].append(msg)
            self.metadata["downloads"]["hf_radar_historical"] = {"status": "failed", "error": msg}
            return False

    def _download_currents_historical(self, source, instrument: str) -> bool:
        from ..downloaders.insitu_currents_historical_downloader import (
            InSituCurrentsHistoricalDownloader,
        )

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        temp   = cfg.temporal_bounds
        out_dir = self.base_dir / f"{instrument}_historical"

        try:
            dl = InSituCurrentsHistoricalDownloader(
                instrument=instrument,
                output_dir=out_dir,
                dry_run=self.dry_run,
                min_depth=source.resolved_min_depth,
                max_depth=source.resolved_max_depth,
                force_download=self.force_download,
            )
            downloaded = dl.download(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=temp.start, end=temp.end,
            ) or []
            self._cleanup_if_empty(out_dir)
            self.metadata["downloads"][f"{instrument}_historical"] = {
                "status": "dry_run" if self.dry_run else "success",
                "file_count": len(downloaded),
            }
            return True
        except Exception as exc:
            msg = f"{instrument} delayed-mode currents download failed: {exc}"
            logger.error(msg)
            self.metadata["errors"].append(msg)
            self.metadata["downloads"][f"{instrument}_historical"] = {
                "status": "failed", "error": msg
            }
            return False

    def _download_adcp_historical(self, source) -> bool:
        return self._download_currents_historical(source, "adcp")

    def _download_argo_historical(self, source) -> bool:
        return self._download_currents_historical(source, "argo")

    def _download_drifter_historical(self, source) -> bool:
        return self._download_currents_historical(source, "drifter")

    def _download_glider_historical(self, source) -> bool:
        return self._download_currents_historical(source, "glider")

    # Altimeter download frequencies, keyed by recipe variable. Wind never
    # needs 5 Hz (no WIND_SPEED there, and 5x the point density for no
    # benefit); waves uses both since VAVH/VAVH_UNFILTERED exist at both.
    _ALTIMETER_FREQUENCIES_BY_VARIABLE = {
        "wind":  ["1hz"],
        "waves": ["1hz", "5hz"],
    }

    def _download_altimeter(self, source) -> bool:
        from ..downloaders.altimeter_downloader import AltimeterDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        temp   = cfg.temporal_bounds
        out_dir = self.base_dir / "altimeter"

        try:
            dl = AltimeterDownloader(
                output_dir=out_dir, dry_run=self.dry_run, force_download=self.force_download,
            )
            kwargs = {
                "frequencies": self._ALTIMETER_FREQUENCIES_BY_VARIABLE.get(
                    cfg.variable, ["1hz", "5hz"]
                ),
            }
            kwargs.update(source.download_kwargs)   # recipe-level override wins
            paths = dl.download(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=temp.start, end=temp.end,
                **kwargs,
            )
            self._cleanup_if_empty(out_dir)
            self.metadata["downloads"]["altimeter"] = {
                "status": "dry_run" if self.dry_run else "success",
                "files":  [str(p) for p in paths],
            }
            return True
        except Exception as exc:
            msg = f"Altimeter download failed: {exc}"
            logger.error(msg)
            self.metadata["errors"].append(msg)
            self.metadata["downloads"]["altimeter"] = {"status": "failed", "error": msg}
            return False

    def _download_radiometer(self, source) -> bool:
        from ..downloaders.radiometer_downloader import RadiometerDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        temp   = cfg.temporal_bounds
        out_dir = self.base_dir / "radiometer"

        try:
            dl = RadiometerDownloader(output_dir=out_dir, dry_run=self.dry_run)
            kwargs = dict(source.download_kwargs)   # e.g. {"sensors": ["amsr2"]}
            paths = dl.download(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=temp.start, end=temp.end,
                **kwargs,
            )
            self._cleanup_if_empty(out_dir)
            self.metadata["downloads"]["radiometer"] = {
                "status": "dry_run" if self.dry_run else "success",
                "files":  [str(p) for p in paths],
            }
            return True
        except Exception as exc:
            msg = f"Radiometer download failed: {exc}"
            logger.error(msg)
            self.metadata["errors"].append(msg)
            self.metadata["downloads"]["radiometer"] = {"status": "failed", "error": msg}
            return False

    def _download_ismn(self, source) -> bool:
        from ..downloaders.ismn_downloader import ISMNDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        temp   = cfg.temporal_bounds
        out_dir = self.base_dir / "ismn"

        try:
            dl = ISMNDownloader(output_dir=out_dir, dry_run=self.dry_run)
            paths = dl.download(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=temp.start, end=temp.end,
                min_depth=source.resolved_min_depth,
                max_depth=source.resolved_max_depth,
                archive_path=source.download_kwargs.get("ismn_archive_path"),
            )
            self._cleanup_if_empty(out_dir)
            if self.dry_run:
                status = "dry_run"
            elif paths:
                status = "success"
            else:
                # Not a failure -- the manually-downloaded archive just
                # isn't there yet (see ISMNDownloader's printed portal
                # instructions). Reporting "success" here would hide that
                # zero files were actually collected.
                status = "awaiting_manual_archive"
            self.metadata["downloads"]["ismn"] = {
                "status": status,
                "files":  [str(p) for p in paths],
            }
            return True
        except Exception as exc:
            msg = f"ISMN selection failed: {exc}"
            logger.error(msg)
            self.metadata["errors"].append(msg)
            self.metadata["downloads"]["ismn"] = {"status": "failed", "error": msg}
            return False

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _save_metadata(self) -> None:
        meta_path = self.base_dir / "download_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)
        logger.info("Metadata saved to %s", meta_path)
