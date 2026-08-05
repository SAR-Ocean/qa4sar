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
from typing import Any, Dict, Optional

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

#: AMSR-E/2's combined coverage across NSIDC-0451 (ends 2023-12-31) and its
#: replacement AU_Land (frozen 2025-09-01, per NSIDC's "AMSR2 SIPS stopped
#: processing AMSR Unified data sets" notice; confirmed 2026-07-31 against
#: NASA's live CMR catalog -- AU_Land's newest real granule is dated
#: 2025-08-31T23:19) -- the later of the two, since _download_amsr_ssm
#: switches datasets at 2023-12-31 (see Task 7). ISO date string, compared
#: against the recipe's temporal_bounds.end.
#:
#: NOTE: "AU_Land_NRT_R02" (a differently-versioned, near-real-time variant
#: of this collection) was used here until 2026-07-31 -- confirmed against
#: CMR to hold only 3 granules total, all dated 2020-10-19 (an abandoned
#: test batch), so every real recipe request against it silently returned
#: 0 files and fell through to the G-Portal SFTP fallback. "AU_Land" (no
#: suffix) is the real, actively-populated 237k+-granule archival
#: collection and is what's actually queried now.
_AMSR_COVERAGE_CUTOFF = "2025-09-01"

#: NSIDC-0451's own coverage ends here; AU_Land picks up from roughly this
#: point through _AMSR_COVERAGE_CUTOFF.
_NSIDC_0451_CUTOFF = "2023-12-31"

#: EUMETSAT's ASCAT NRT dissemination access this toolbox relies on
#: stopped being populated for recent dates as of this cutoff -- a
#: request ending after it returning 0 products is expected, not an
#: error, and gets the same "coverage cutoff" notice AMSR2 gets above.
_ASCAT_COVERAGE_CUTOFF = "2025-07-15"

# ASCAT's collocation spec lives under its data_type tag "scatterometer_ssm"
# in DEFAULT_LAYER_TYPE_SPECS, not its own source_type "ascat_ssm" -- see
# that dict's comment and collocation.py's _resolve_layer_type. Every other
# soil-moisture/layer source_type matches its DEFAULT_LAYER_TYPE_SPECS key
# directly, so only this one needs an alias.
_TOLERANCE_LOOKUP_ALIASES = {"ascat_ssm": "scatterometer_ssm"}


def _resolve_temporal_padding_minutes(cfg, *source_types: str) -> float:
    """
    Largest collocation time-tolerance, in minutes, that applies to any of
    *source_types* -- mirrors collocation.py's own resolution order
    (per-source override, then layer_vs_layer.layer_type_specs, then the
    point_vs_layer default/validation_temporal_averaging_minutes) closely
    enough to safely pad a *download* request window, without needing a
    live datatree/node (this runs before anything has been downloaded or
    converted).

    Used to pad each downloader's requested start/end so every SAR scene's
    real collocation window is fully covered by downloaded data -- even
    the first/last scene in a multi-day request, whose window extends past
    the literal requested start/end. Without this, a source whose download
    stops exactly at the requested boundary silently starves the outermost
    SAR scenes of validation data relative to scenes further from the
    range's edges (see the collocation-diagnostics plot's day-1/day-3 vs.
    day-2 asymmetry this was written to fix).
    """
    from .recipe import DEFAULT_LAYER_TYPE_SPECS

    coll_cfg = cfg.collocation
    layer_specs = dict(DEFAULT_LAYER_TYPE_SPECS)
    if coll_cfg.layer_vs_layer is not None:
        layer_specs.update(coll_cfg.layer_vs_layer.layer_type_specs)
    pvl = coll_cfg.point_vs_layer
    fallback = max(pvl.time_tolerance_minutes, pvl.validation_temporal_averaging_minutes)

    source_overrides = {
        s.source_type: s.collocation_kwargs["time_tolerance_minutes"]
        for s in cfg.validation_sources
        if s.collocation_kwargs and "time_tolerance_minutes" in s.collocation_kwargs
    }

    tolerances = []
    for st in source_types:
        if st in source_overrides:
            tolerances.append(float(source_overrides[st]))
            continue
        key = _TOLERANCE_LOOKUP_ALIASES.get(st, st)
        tolerances.append(float(layer_specs[key]["time_tolerance_minutes"]) if key in layer_specs else fallback)
    return max(tolerances) if tolerances else fallback


def _padded_temporal_bounds(cfg, *source_types: str) -> "tuple[str, str]":
    """(start, end) ISO strings, padded symmetrically by
    :func:`_resolve_temporal_padding_minutes` on each side of
    ``cfg.temporal_bounds`` -- for passing to a downloader's own
    ``start``/``end`` arguments. Does not mutate ``cfg.temporal_bounds``
    itself, since other logic (output folder naming, coverage-cutoff
    comparisons, metadata) must keep using the literal requested range.
    """
    import pandas as pd

    pad = pd.Timedelta(minutes=_resolve_temporal_padding_minutes(cfg, *source_types))
    temp = cfg.temporal_bounds
    start = (pd.Timestamp(temp.start) - pad).isoformat()
    end = (pd.Timestamp(temp.end) + pad).isoformat()
    return start, end


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
        self._previous_downloads: Dict[str, Any] = self._load_previous_downloads()
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

    def _run_download(
        self, key: str, out_dir: Path, build_dl, build_kwargs, error_label: str,
        *, result_to_metadata=None,
    ) -> bool:
        """Shared skeleton for the simple ``_download_*`` handlers: build the
        downloader, call ``.download()``, clean up an empty output dir, and
        record success/failure metadata under *key*.

        *build_dl* is a zero-arg callable returning the constructed
        downloader. *build_kwargs* is a zero-arg callable returning the
        ``download()`` keyword-argument dict. *result_to_metadata*, if
        given, maps ``(download_result, downloader) -> dict`` merged into
        the success entry; default is ``{"files": [str(p) for p in result]}``.
        """
        try:
            dl = build_dl()
            download_kwargs = build_kwargs()
            result = dl.download(**download_kwargs)
            self._cleanup_if_empty(out_dir)
            entry: Dict[str, Any] = {"status": "dry_run" if self.dry_run else "success"}
            if result_to_metadata is not None:
                entry.update(result_to_metadata(result, dl))
            else:
                entry["files"] = [str(p) for p in (result or [])]
            self.metadata["downloads"][key] = entry
            return True
        except Exception as exc:
            msg = f"{error_label} download failed: {exc}"
            logger.error(msg)
            self.metadata["errors"].append(msg)
            self.metadata["downloads"][key] = {"status": "failed", "error": msg}
            return False

    def _load_previous_downloads(self) -> Dict[str, Any]:
        """Read the ``downloads`` section of a prior run's
        ``download_metadata.json`` in ``self.base_dir``, if present.

        Used by :meth:`_already_succeeded` so a rerun triggered by one
        source's failure (e.g. SMOS) doesn't force every other,
        already-succeeded source (e.g. ASCAT) to re-authenticate and
        re-dispatch — see design-choices.md's per-source gating fix.
        """
        meta_path = self.base_dir / "download_metadata.json"
        if not meta_path.exists():
            return {}
        try:
            with open(meta_path) as f:
                return json.load(f).get("downloads", {})
        except Exception:
            return {}

    def _already_succeeded(self, source_type: str) -> bool:
        """True if *source_type* succeeded in the previous run recorded in
        ``self._previous_downloads`` and ``force_download`` isn't set."""
        if self.force_download:
            return False
        prev = self._previous_downloads.get(source_type)
        if prev is None:
            return False
        return prev.get("status") == "success"

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
        if self._already_succeeded("sar"):
            self.metadata["downloads"]["sar"] = self._previous_downloads["sar"]
            logger.info("Skipping SAR download: already succeeded in a previous run.")
        elif not self._download_sar():
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
            if self._already_succeeded(source.source_type):
                self.metadata["downloads"][source.source_type] = self._previous_downloads[source.source_type]
                logger.info(
                    "Skipping %s: already succeeded in a previous run.",
                    source.source_type,
                )
                continue
            if not self._dispatch_source(source):
                ok = False

        self._report_combined_hf_radar_us_status()
        self._report_combined_hf_radar_status()

        if not self.dry_run:
            self._save_metadata()
        return ok

    # ------------------------------------------------------------------
    # Per-source downloaders
    # ------------------------------------------------------------------

    def _download_sar(self) -> bool:
        from .sar_sources import SAR_SOURCES

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        temp   = cfg.temporal_bounds
        spec   = SAR_SOURCES[cfg.sar_data.source]
        out_dir = self.base_dir / spec.output_subdir

        return self._run_download(
            "sar", out_dir,
            lambda: spec.build_downloader(out_dir, self.dry_run, self.force_download),
            lambda: dict(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=temp.start,      end=temp.end,
                **spec.extra_download_kwargs(cfg.sar_data),
            ),
            f"SAR ({cfg.sar_data.source})",
        )

    def _download_insitu(
        self,
        source_types: list[str],
        min_depth: float,
        max_depth: float,
    ) -> bool:
        from ..downloaders.insitu_downloader import InSituDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        pad_start, pad_end = _padded_temporal_bounds(cfg, *source_types)

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
                start=pad_start,        end=pad_end,
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

    def _report_combined_hf_radar_us_status(self) -> None:
        """Log one combined 'no data' notice for hf_radar_us, naming which
        backends were tried, instead of the silent zero-message outcome
        today's per-source status recording produces for an empty result."""
        if self.dry_run:
            return
        entry = self.metadata["downloads"].get("hf_radar_us")
        if entry is None or entry.get("status") == "failed":
            return
        if entry.get("file_count", 0) > 0:
            return
        backends = entry.get("attempted_backends") or []
        msg = f"No US HF-radar data found (tried {', '.join(backends)}) for this window."
        logger.warning(msg)
        self.metadata["notices"].append(msg)

    def _report_combined_hf_radar_status(self) -> None:
        """Log one combined 'no data' notice for the non-US hf_radar /
        hf_radar_historical pair, mirroring _report_combined_currents_status."""
        if self.dry_run:
            return
        attempted = [
            t for t in ("hf_radar", "hf_radar_historical")
            if any(s.source_type == t for s in self.recipe.config.validation_sources)
        ]
        if not attempted:
            return
        total_files = 0
        any_non_failed = False
        for t in attempted:
            entry = self.metadata["downloads"].get(t, {})
            if entry.get("status") == "failed":
                continue
            any_non_failed = True
            total_files += entry.get("file_count", 0)
        if not any_non_failed or total_files > 0:
            return
        msg = "No HF-radar data found (hf_radar/hf_radar_historical) for this window."
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
            "hf_radar_us":   self._download_hf_radar_us,
            "adcp_historical": self._download_adcp_historical,
            "argo_historical": self._download_argo_historical,
            "drifter_historical": self._download_drifter_historical,
            "glider_historical": self._download_glider_historical,
            "altimeter":     self._download_altimeter,
            "radiometer":    self._download_radiometer,
            "ismn":          self._download_ismn,
            "ascat_ssm":     self._download_ascat_ssm,
            "amsr_ssm":      self._download_amsr_ssm,
            "smap_ssm":      self._download_smap_ssm,
            "smos_ssm":      self._download_smos_ssm,
            "cds_ssm":       self._download_cds_ssm,
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
        # Fixed literal, not source.source_type: this handler is only ever
        # dispatched for source_type "scatterometer" (see _dispatch_source),
        # and some existing tests call it directly with source=None.
        pad_start, pad_end = _padded_temporal_bounds(cfg, "scatterometer")
        out_dir = self.base_dir / "osi_saf_winds"

        return self._run_download(
            "scatterometer", out_dir,
            lambda: ScatterometerDownloader(
                output_dir=out_dir, dry_run=self.dry_run, force_download=self.force_download,
            ),
            lambda: dict(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=pad_start, end=pad_end,
            ),
            "Scatterometer",
            result_to_metadata=lambda result, dl: {},
        )

    def _download_ascat_ssm(self, source) -> bool:
        from ..downloaders.ascat_soil_moisture_downloader import ASCATSoilMoistureDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        pad_start, pad_end = _padded_temporal_bounds(cfg, source.source_type)
        out_dir = self.base_dir / "ascat_ssm"

        ok = self._run_download(
            "ascat_ssm", out_dir,
            lambda: ASCATSoilMoistureDownloader(
                output_dir=out_dir, dry_run=self.dry_run, force_download=self.force_download,
            ),
            lambda: dict(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=pad_start, end=pad_end,
            ),
            "ASCAT SSM",
        )
        if (
            ok and not self.dry_run
            and not self.metadata["downloads"].get("ascat_ssm", {}).get("files")
            and cfg.temporal_bounds.end > _ASCAT_COVERAGE_CUTOFF
        ):
            self.metadata["notices"].append(
                f"ASCAT: requested range ends {cfg.temporal_bounds.end}, after this "
                f"source's known coverage cutoff ({_ASCAT_COVERAGE_CUTOFF}) — "
                f"0 products found (expected, not an error)."
            )
        return ok

    def _download_earthdata_ssm(
        self, source, dataset: str, version: Optional[str], out_subdir: str,
    ) -> bool:
        from ..downloaders.earthdata_soil_moisture_downloader import EarthdataSoilMoistureDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        pad_start, pad_end = _padded_temporal_bounds(cfg, source.source_type)
        out_dir = self.base_dir / out_subdir

        try:
            dl = EarthdataSoilMoistureDownloader(
                dataset=dataset, version=version, output_dir=out_dir, dry_run=self.dry_run,
            )
            paths = dl.download(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=pad_start, end=pad_end,
            )
            self._cleanup_if_empty(out_dir)
            self.metadata["downloads"][out_subdir] = {
                "status": "dry_run" if self.dry_run else "success",
                "files":  [str(p) for p in paths],
            }
            return True
        except Exception as exc:
            msg = f"{dataset} download failed: {exc}"
            logger.error(msg)
            self.metadata["errors"].append(msg)
            self.metadata["downloads"][out_subdir] = {"status": "failed", "error": msg}
            return False

    def _download_amsr_ssm(self, source) -> bool:
        temp = self.recipe.config.temporal_bounds
        if temp.end <= _NSIDC_0451_CUTOFF:
            dataset = "NSIDC-0451"
        else:
            dataset = "AU_Land"
        ok = self._download_earthdata_ssm(source, dataset=dataset, version=None, out_subdir="amsr_ssm")
        if ok and not self.metadata["downloads"].get("amsr_ssm", {}).get("files"):
            # NASA Earthdata alone returning 0 files is not yet the final
            # word -- G-Portal (below) is a real second source, and often
            # succeeds when Earthdata doesn't. Only note the coverage
            # cutoff once *both* have been tried and both found nothing
            # (see _try_gportal_amsr_fallback) -- surfacing it here
            # unconditionally previously fired even when G-Portal went on
            # to find real files, confirmed against a real recipe run.
            self._try_gportal_amsr_fallback()
        return ok

    def _try_gportal_amsr_fallback(self) -> None:
        """
        Best-effort fallback when NASA Earthdata's AMSR2 coverage
        (frozen at _AMSR_COVERAGE_CUTOFF) returns zero files: try JAXA's
        own G-Portal SFTP archive for the same window. A failure here
        (missing credentials, discovery failure, connection error) is
        recorded as a notice, not an error -- this is a best-effort
        second attempt, not a required source.
        """
        from ..downloaders.gportal_downloader import GPortalAMSR2Downloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        pad_start, pad_end = _padded_temporal_bounds(cfg, "amsr_ssm")
        out_dir = self.base_dir / "amsr_ssm"

        try:
            dl = GPortalAMSR2Downloader(
                output_dir=out_dir, dry_run=self.dry_run, force_download=self.force_download,
                allow_prompt=False,
            )
            paths = dl.download(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=pad_start, end=pad_end,
            )
            self._cleanup_if_empty(out_dir)
            entry = self.metadata["downloads"].setdefault("amsr_ssm", {})
            entry["files"] = [str(p) for p in paths]
            entry["status"] = "dry_run" if self.dry_run else "success"
            entry["gportal_fallback"] = True
            if not paths and not self.dry_run:
                # 0 files here is a normal, non-error outcome (e.g. no AMSR2
                # coverage for this window at all) -- but reported entirely
                # silently, "status": "success" with 0 files is otherwise
                # indistinguishable from a genuine download, and no notice
                # exists to explain it. Both NASA Earthdata AND G-Portal
                # have now been tried and both found nothing, so this is
                # the right (and only) point to mention the known
                # coverage-cutoff explanation, if it applies.
                temp = cfg.temporal_bounds
                if temp.end > _AMSR_COVERAGE_CUTOFF:
                    self.metadata["notices"].append(
                        f"AMSR-E/2: requested range ends {temp.end}, after this "
                        f"source's known coverage cutoff ({_AMSR_COVERAGE_CUTOFF}) — "
                        f"0 granules found from NASA Earthdata or the G-Portal "
                        f"fallback (expected, not an error)."
                    )
                else:
                    self.metadata["notices"].append(
                        "AMSR2: G-Portal fallback also found 0 files in window — "
                        "no AMSR2 soil-moisture data available for this run from "
                        "either source."
                    )
        except Exception as exc:
            msg = f"G-Portal AMSR2 fallback failed: {exc}"
            logger.warning(msg)
            self.metadata["notices"].append(msg)

    def _download_smap_ssm(self, source) -> bool:
        return self._download_earthdata_ssm(source, dataset="SPL2SMP_E", version="006", out_subdir="smap_ssm")

    def _download_smos_ssm(self, source) -> bool:
        from ..downloaders.smos_downloader import SMOSDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        pad_start, pad_end = _padded_temporal_bounds(cfg, source.source_type)
        out_dir = self.base_dir / "smos_ssm"

        return self._run_download(
            "smos_ssm", out_dir,
            lambda: SMOSDownloader(output_dir=out_dir, dry_run=self.dry_run),
            lambda: dict(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=pad_start, end=pad_end,
            ),
            "SMOS SSM",
        )

    def _download_cds_ssm(self, source) -> bool:
        from ..downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        pad_start, pad_end = _padded_temporal_bounds(cfg, source.source_type)
        out_dir = self.base_dir / "cds_ssm"
        product_type = source.download_kwargs.get("product_type", "active")

        return self._run_download(
            "cds_ssm", out_dir,
            lambda: CDSSoilMoistureDownloader(
                product_type=product_type,
                output_dir=out_dir, dry_run=self.dry_run,
            ),
            lambda: dict(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=pad_start, end=pad_end,
            ),
            f"C3S CDS SSM ({product_type})",
        )

    def _download_scatterometer_ftp(self, source, satellite: str) -> bool:
        from ..downloaders.scatterometer_ftp_downloader import ScatterometerFTPDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        pad_start, pad_end = _padded_temporal_bounds(cfg, source.source_type)
        out_dir = self.base_dir / f"scatterometer_{satellite}"

        return self._run_download(
            f"scatterometer_{satellite}", out_dir,
            lambda: ScatterometerFTPDownloader(
                satellite=satellite,
                output_dir=out_dir, dry_run=self.dry_run, force_download=self.force_download,
            ),
            lambda: dict(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=pad_start, end=pad_end,
            ),
            f"{satellite} FTP scatterometer",
            result_to_metadata=lambda result, dl: {},
        )

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
        pad_start, pad_end = _padded_temporal_bounds(cfg, "hf_radar_grid")
        out_dir = self.base_dir / "hf_radar"

        return self._run_download(
            "hf_radar", out_dir,
            lambda: HFRadarDownloader(
                output_dir=out_dir,
                dry_run=self.dry_run,
                min_depth=source.resolved_min_depth,
                max_depth=source.resolved_max_depth,
                force_download=self.force_download,
            ),
            lambda: dict(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=pad_start, end=pad_end,
            ),
            "HF radar",
            result_to_metadata=lambda result, dl: {"file_count": len(result or [])},
        )

    def _download_noaa_hfradar(self, source) -> bool:
        from ..downloaders.noaa_hfradar_downloader import (
            DEFAULT_RESOLUTION_KM,
            NOAAHFRadarDownloader,
        )

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        pad_start, pad_end = _padded_temporal_bounds(cfg, "hf_radar_grid")
        out_dir = self.base_dir / "hfr_noaa"
        # Resolution is an optional per-source override, forwarded via the
        # established ValidationDataSource.download_kwargs channel.
        resolution_km = int(source.download_kwargs.get("resolution_km", DEFAULT_RESOLUTION_KM))

        return self._run_download(
            "hf_radar_noaa", out_dir,
            lambda: NOAAHFRadarDownloader(
                output_dir=out_dir, dry_run=self.dry_run,
                resolution_km=resolution_km, force_download=self.force_download,
            ),
            lambda: dict(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=pad_start, end=pad_end,
            ),
            "NOAA HF-radar",
            result_to_metadata=lambda result, dl: {},
        )

    def _download_hf_radar_historical(self, source) -> bool:
        from ..downloaders.hf_radar_historical_downloader import HFRadarHistoricalDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        pad_start, pad_end = _padded_temporal_bounds(cfg, "hf_radar_grid")
        out_dir = self.base_dir / "hf_radar_historical"

        return self._run_download(
            "hf_radar_historical", out_dir,
            lambda: HFRadarHistoricalDownloader(
                output_dir=out_dir, dry_run=self.dry_run, force_download=self.force_download,
            ),
            lambda: dict(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=pad_start, end=pad_end,
            ),
            "HF radar historical",
            result_to_metadata=lambda result, dl: {"file_count": len(result or [])},
        )

    def _download_hf_radar_us(self, source) -> bool:
        from ..downloaders.hf_radar_us_downloader import HFRadarUSDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        pad_start, pad_end = _padded_temporal_bounds(cfg, "hf_radar_grid")
        # Resolution is an optional per-source override, forwarded via the
        # established ValidationDataSource.download_kwargs channel. None
        # (the recipe didn't set it) lets HFRadarUSDownloader auto-pick the
        # matched region's own default; "finest" and explicit floats are
        # also forwarded as-is. Ignored on the Copernicus fallback path.
        resolution_km = source.download_kwargs.get("resolution_km")

        try:
            dl = HFRadarUSDownloader(
                output_dir=self.base_dir,
                dry_run=self.dry_run,
                resolution_km=resolution_km,
                force_download=self.force_download,
            )
            downloaded = dl.download(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=pad_start, end=pad_end,
            ) or []
            for subdir in ("hfr_noaa", "hf_radar", "hf_radar_historical"):
                self._cleanup_if_empty(self.base_dir / subdir)
            self.metadata["downloads"]["hf_radar_us"] = {
                "status": "dry_run" if self.dry_run else "success",
                "file_count": len(downloaded),
                "backend": dl.resolved_backend,
                "attempted_backends": dl.attempted_backends,
            }
            return True
        except Exception as exc:
            msg = f"US HF-radar download failed: {exc}"
            logger.error(msg)
            self.metadata["errors"].append(msg)
            self.metadata["downloads"]["hf_radar_us"] = {"status": "failed", "error": msg}
            return False

    def _download_currents_historical(self, source, instrument: str) -> bool:
        from ..downloaders.insitu_currents_historical_downloader import (
            InSituCurrentsHistoricalDownloader,
        )

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        pad_start, pad_end = _padded_temporal_bounds(cfg, source.source_type)
        out_dir = self.base_dir / f"{instrument}_historical"

        return self._run_download(
            f"{instrument}_historical", out_dir,
            lambda: InSituCurrentsHistoricalDownloader(
                instrument=instrument,
                output_dir=out_dir,
                dry_run=self.dry_run,
                min_depth=source.resolved_min_depth,
                max_depth=source.resolved_max_depth,
                force_download=self.force_download,
            ),
            lambda: dict(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=pad_start, end=pad_end,
            ),
            f"{instrument} delayed-mode currents",
            result_to_metadata=lambda result, dl: {"file_count": len(result or [])},
        )

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
        # DEFAULT_LAYER_TYPE_SPECS keys altimeter by frequency
        # ("altimeter_1hz"/"altimeter_5hz"), not the bare "altimeter"
        # source_type -- pass both so the padding lookup finds their
        # (equal, 180min) tolerance regardless of which frequency this
        # recipe's variable actually requests.
        pad_start, pad_end = _padded_temporal_bounds(cfg, "altimeter_1hz", "altimeter_5hz")
        out_dir = self.base_dir / "altimeter"
        kwargs = {
            "frequencies": self._ALTIMETER_FREQUENCIES_BY_VARIABLE.get(
                cfg.variable, ["1hz", "5hz"]
            ),
        }
        kwargs.update(source.download_kwargs)   # recipe-level override wins

        return self._run_download(
            "altimeter", out_dir,
            lambda: AltimeterDownloader(
                output_dir=out_dir, dry_run=self.dry_run, force_download=self.force_download,
            ),
            lambda: dict(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=pad_start, end=pad_end,
                **kwargs,
            ),
            "Altimeter",
        )

    def _download_radiometer(self, source) -> bool:
        from ..downloaders.radiometer_downloader import RadiometerDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        pad_start, pad_end = _padded_temporal_bounds(cfg, source.source_type)
        out_dir = self.base_dir / "radiometer"
        kwargs = dict(source.download_kwargs)   # e.g. {"sensors": ["amsr2"]}

        return self._run_download(
            "radiometer", out_dir,
            lambda: RadiometerDownloader(output_dir=out_dir, dry_run=self.dry_run),
            lambda: dict(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=pad_start, end=pad_end,
                **kwargs,
            ),
            "Radiometer",
        )

    def _download_ismn(self, source) -> bool:
        from ..downloaders.ismn_downloader import ISMNDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        pad_start, pad_end = _padded_temporal_bounds(cfg, source.source_type)
        out_dir = self.base_dir / "ismn"

        try:
            dl = ISMNDownloader(output_dir=out_dir, dry_run=self.dry_run)
            paths = dl.download(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=pad_start, end=pad_end,
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
