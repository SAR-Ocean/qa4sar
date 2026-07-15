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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .recipe import Recipe, RecipeConfig, GeographicBounds, TemporalBounds
from ..downloaders.base import build_output_dir

logger = logging.getLogger(__name__)

__all__ = ["DataOrchestrator"]

# In-situ platform types handled by the InSituDownloader
_INSITU_TYPES = {"mooring", "buoy", "drifter", "ferrybox", "tidal_gauge"}


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

    def __init__(self, recipe: Recipe, dry_run: bool = False) -> None:
        self.recipe   = recipe
        self.dry_run  = dry_run
        self.base_dir = self._setup_base_dir()
        self.metadata: Dict[str, Any] = {
            "recipe_name": recipe.config.name,
            "variable":    recipe.config.variable,
            "created":     datetime.now().isoformat(),
            "geographic_bounds": recipe.config.geographic_bounds.to_dict(),
            "temporal_bounds":   recipe.config.temporal_bounds.to_dict(),
            "downloads": {},
            "errors":    [],
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

        # 2. Group in-situ sources and download as one batch
        insitu_sources = [
            s for s in self.recipe.config.validation_sources
            if s.source_type in _INSITU_TYPES
        ]
        if insitu_sources:
            source_types = [s.source_type for s in insitu_sources]
            # Use the most permissive depth window across all in-situ sources
            min_depth = min(s.resolved_min_depth for s in insitu_sources)
            max_depth = max(s.resolved_max_depth for s in insitu_sources)
            if not self._download_insitu(source_types, min_depth, max_depth):
                ok = False

        # 3. Other sources one by one
        for source in self.recipe.config.validation_sources:
            if source.source_type in _INSITU_TYPES:
                continue   # handled above
            if not self._dispatch_source(source):
                ok = False

        if not self.dry_run:
            self._save_metadata()
        return ok

    # ------------------------------------------------------------------
    # Per-source downloaders
    # ------------------------------------------------------------------

    def _download_sar(self) -> bool:
        from ..downloaders.sar_downloader import SARDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        temp   = cfg.temporal_bounds

        out_dir = self.base_dir / "S1_L2_OCN"

        try:
            dl = SARDownloader(
                output_dir=out_dir,
                dry_run=self.dry_run,
            )
            paths = dl.download(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=temp.start,      end=temp.end,
                modes=cfg.sar_data.swath_mode or None,
                limit=cfg.sar_data.max_downloads,
            )
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
            )
            dl.download(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=temp.start,       end=temp.end,
                source_types=source_types,
            )
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

    def _dispatch_source(self, source) -> bool:
        handlers = {
            "scatterometer": self._download_scatterometer,
            "hf_radar":      self._download_hf_radar,
            "hf_radar_noaa": self._download_noaa_hfradar,
            "altimeter":     self._download_altimeter,
            "radiometer":    self._download_radiometer,
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
            dl = ScatterometerDownloader(output_dir=out_dir, dry_run=self.dry_run)
            dl.download(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=temp.start, end=temp.end,
            )
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
            )
            dl.download(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=temp.start, end=temp.end,
            )
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
            NOAAHFRadarDownloader,
            DEFAULT_RESOLUTION_KM,
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
            )
            dl.download(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=temp.start, end=temp.end,
            )
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
            dl = AltimeterDownloader(output_dir=out_dir, dry_run=self.dry_run)
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

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _save_metadata(self) -> None:
        meta_path = self.base_dir / "download_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)
        logger.info("Metadata saved to %s", meta_path)
