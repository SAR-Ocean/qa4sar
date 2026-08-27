"""
DataOrchestrator -- step 1 of the validation pipeline.

Reads a Recipe and drives every download it requires: the recipe's
SAR source (see sar_sources.SAR_SOURCES) and its validation sources --
in-situ observations, HF radar, scatterometer, altimeter, radiometer,
and model reanalysis for wind/waves/currents; in-situ stations and
satellite retrievals for soil moisture. See README.md for the full
list of supported sources.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..downloaders.base import build_output_dir
from .recipe import Recipe

logger = logging.getLogger(__name__)

__all__ = ["DataOrchestrator"]

# In-situ platform types handled by the InSituDownloader
_INSITU_TYPES = {"mooring", "buoy", "drifter", "ferrybox", "tidal_gauge"}

# Delayed-mode ("historical") source_types, dispatched before any NRT
# source such that its results can inform whether the NRT counterpart is
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
#: replacement AU_Land (frozen 2025-09-01) ends 2025-08-31.
_AMSR_COVERAGE_CUTOFF = "2025-09-01"

#: NSIDC-0451's own coverage ends here; AU_Land picks up from roughly this
#: point through _AMSR_COVERAGE_CUTOFF.
_NSIDC_0451_CUTOFF = "2023-12-31"

#: EUMETSAT's ASCAT NRT dissemination access ended on 2025-07-15, not 
#: providing any data for later days. The H-SAF ASCAT SSM product takes
#: over this demand (only for the last 60 days).
_ASCAT_COVERAGE_CUTOFF = "2025-07-15"

# ASCAT's EUMDAC and H-SAF downloaders both stamp
# data_type="scatterometer_ssm" (not "ascat_ssm"), so its collocation
# spec is keyed under that data_type in DEFAULT_LAYER_TYPE_SPECS --
# see collocation.py's _resolve_layer_type. Every other source_type
# matches its DEFAULT_LAYER_TYPE_SPECS key directly.
_TOLERANCE_LOOKUP_ALIASES = {"ascat_ssm": "scatterometer_ssm"}


def _resolve_temporal_padding_minutes(cfg, *source_types: str) -> float:
    """
    Return the largest collocation time-tolerance, in minutes, across
    *source_types*. Mirrors collocation.py's own resolution order
    (per-source override, then layer_vs_layer.layer_type_specs, then the
    point_vs_layer default/validation_temporal_averaging_minutes) closely
    enough to safely pad a download request window, without requiring a
    live datatree or node -- this runs before any data has been
    downloaded or converted.

    Used to pad each downloader's requested start/end so every SAR
    scene's collocation window is fully covered, including the first and
    last scenes in a multi-day request, whose windows extend past the
    literal requested boundaries. Without this padding, a source whose
    download stops exactly at the requested boundary would leave the
    outermost SAR scenes with less validation data than scenes nearer
    the middle of the range.
    """
    from .recipe import DEFAULT_LAYER_TYPE_SPECS, min_safe_model_time_tolerance_minutes

    coll_cfg = cfg.collocation
    layer_specs = dict(DEFAULT_LAYER_TYPE_SPECS)
    if coll_cfg.layer_vs_layer is not None:
        # Per-key deep merge (not layer_specs.update(...), which would replace
        # a key's whole dict): a recipe overriding only one field for a given
        # layer_type (e.g. "method" for "era5_wind") must not silently lose
        # the rest of that key's defaults (e.g. "time_tolerance_minutes") in
        # the process. Mirrors datatree_converter.py's _build_subset_kwargs,
        # which already does this correctly.
        for key, spec in coll_cfg.layer_vs_layer.layer_type_specs.items():
            layer_specs[key] = {**layer_specs.get(key, {}), **spec}
    pvl = coll_cfg.point_vs_layer
    fallback = max(pvl.time_tolerance_minutes, pvl.validation_temporal_averaging_minutes)

    source_overrides = {
        s.source_type: s.collocation_kwargs["time_tolerance_minutes"]
        for s in cfg.validation_sources
        if s.collocation_kwargs and "time_tolerance_minutes" in s.collocation_kwargs
    }

    tolerances = []
    for st in source_types:
        # "era5" (the source_type every recipe actually declares) has no
        # DEFAULT_LAYER_TYPE_SPECS entry of its own -- only the
        # variable-specific "era5_wind"/"era5_waves"/"era5_soil_moisture" keys
        # do, matching the data_type DataTreeConverter.from_era5 stamps on
        # each node (visualization.py's own tolerance lookup reads that
        # per-node data_type directly, so it does not need this alias).
        # Omitting this special case would leave every ERA5 download using
        # the generic point_vs_layer tolerance instead of its own
        # bracket-safe value.
        key = "era5_" + cfg.variable if st == "era5" else _TOLERANCE_LOOKUP_ALIASES.get(st, st)
        if st in source_overrides:
            resolved = float(source_overrides[st])
        elif key in layer_specs:
            resolved = float(layer_specs[key]["time_tolerance_minutes"])
        else:
            resolved = fallback
        min_safe = min_safe_model_time_tolerance_minutes(key)
        if min_safe is not None and resolved < min_safe:
            logger.warning(
                "%s: time_tolerance_minutes=%s is below the %s-minute minimum "
                "that guarantees ModelLayerCollocation always finds a "
                "bracketing pair of granules for %r (cadence-derived, see "
                "recipe.MODEL_CADENCE_HOURS) -- SAR scenes near the edges of "
                "the requested window may end up with no matched model data.",
                cfg.name, resolved, min_safe, key,
            )
        tolerances.append(resolved)
    return max(tolerances) if tolerances else fallback


def _sar_entry_found_data(
    entry: Dict[str, Any],
    sar_dir: Optional[Path] = None,
    file_glob: Optional[str] = None,
) -> bool:
    """
    True if a recorded ``"sar"`` download-metadata entry represents at
    least one product found for the window, a failed attempt (whose data
    status is unknown and must not be treated as empty), or no recorded
    entry at all.

    Entries recorded before ``found_count`` existed can have an empty
    ``"files"`` list even though real products were found and downloaded:
    ``SARDownloader.download()`` (and its per-source siblings) only
    appends *newly* downloaded files to the list it returns -- a product
    that was already present on disk (skipped as a duplicate) is never
    appended. So a fully successful old-schema run where every matched
    product happened to already be cached ends up recording ``files: []``
    despite real data being present. For an entry with no ``found_count`` and an empty
    ``"files"`` list, check *sar_dir* directly (matching *file_glob*, e.g.
    ``SARSourceSpec.file_glob``) instead of trusting that ambiguous empty
    list -- callers that can't supply *sar_dir*/*file_glob* keep the old
    ``len(files)`` behaviour.

    When the disk check runs, it also backfills ``entry["found_count"]``
    and ``entry["files"]`` in place from what it actually found -- *entry*
    is typically the same dict a caller is about to save as part of its
    own metadata (e.g. ``download_all()``'s already-succeeded copy-forward
    path), so this heals a stale old-schema entry into a self-consistent
    one instead of leaving the gap to be silently re-read (and re-scanned)
    on every future run.
    """
    if not entry or entry.get("status") == "failed":
        return True
    if "found_count" in entry:
        return entry["found_count"] > 0
    if entry.get("files"):
        return True
    if sar_dir is not None and file_glob is not None and sar_dir.exists():
        found_files = sorted(sar_dir.rglob(file_glob))
        entry["found_count"] = len(found_files)
        entry["files"] = [str(p) for p in found_files]
        return len(found_files) > 0
    return False


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

    def __init__(
        self, recipe: Recipe, dry_run: bool = False, force_download: bool = False,
        download_all_in_bbox: bool = False,
    ) -> None:
        self.recipe   = recipe
        self.dry_run  = dry_run
        self.force_download = force_download
        self.download_all_in_bbox = download_all_in_bbox
        self.base_dir = self._setup_base_dir()
        self._previous_downloads: Dict[str, Any] = self._load_previous_downloads()
        self._previous_variable: Optional[str] = self._load_previous_variable()
        self.metadata: Dict[str, Any] = {
            "recipe_name": recipe.config.name,
            "variable":    recipe.config.variable,
            "created":     datetime.now().isoformat(),
            "geographic_bounds": recipe.config.geographic_bounds.to_dict(),
            "temporal_bounds":   recipe.config.temporal_bounds.to_dict(),
            "downloads": {},
            "errors":    [],
            # Non-failure, user-facing observations (e.g. "no data found for this
            # window"), distinct from "errors": cli.py's _is_already_downloaded
            # treats an error as "something went wrong, don't skip re-download
            # next time," a behavior a notice must never trigger.
            "notices":   [],
        }

        # Populated by _compute_sar_scene_times() (see download_all()) from
        # the downloaded SAR files' embedded timestamps, sorted ascending, and
        # consumed by _padded_temporal_bounds to narrow each validation
        # source's download window to the actual SAR scenes rather than the
        # recipe's full nominal temporal_bounds. Remains None until populated,
        # or on extraction failure; this is purely an optimization and must
        # never alter what a download would do in its absence.
        self._sar_scene_times: Optional[List[pd.Timestamp]] = None

        # Populated lazily by _collocation_predictions() (see download_all())
        # on first use, keyed by validation source_type -- None until then,
        # matching _sar_scene_times's own established laziness pattern.
        self._collocation_predictions_cache: Optional[Dict[str, Any]] = None

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
        """
        Remove out_dir if the download produced no files anywhere under
        it (including nested, otherwise-empty subdirectories). Directories
        that already contain at least one file are left untouched.
        """
        if out_dir.exists() and not any(p.is_file() for p in out_dir.rglob("*")):
            shutil.rmtree(out_dir)

    def _run_download(
        self, key: str, out_dir: Path, build_dl, windows, build_kwargs, error_label: str,
        *, result_to_metadata=None,
    ) -> bool:
        """
        Shared skeleton for the simple ``_download_*`` handlers: build the
        downloader once, call ``.download()`` once per window in *windows*
        (concatenating results across windows), clean up an empty output
        directory, and record success/failure metadata under *key*.

        *build_dl* is a zero-argument callable returning the constructed
        downloader, built once and reused across every window. *windows* is
        a list of ``(start, end)`` ISO-string pairs, e.g. from
        ``self._padded_temporal_bounds(...)``, or a single-element list for
        a caller that does not window (e.g. SAR itself). *build_kwargs* is a
        ``(start, end) -> dict`` callable returning that window's
        ``download()`` keyword arguments (only start/end vary across
        windows). *result_to_metadata*, if given, maps
        ``(merged_result, downloader) -> dict``, merged into the success
        entry; the default is ``{"files": [str(p) for p in merged_result]}``.

        A failure partway through *windows* does not discard results already
        produced by earlier windows -- the partial ``merged_result`` is
        still recorded under *key* -- but the run as a whole is always
        recorded as a failure if any window fails:
        ``{"status": "failed", "error": ..., "files": [...]}``, an error
        appended to ``self.metadata["errors"]``, and ``False`` returned.
        ``cli.py``'s ``_is_already_downloaded`` gates purely on
        ``metadata["errors"]`` being empty, so a partial failure must remain
        visible as an error, or the next run would silently skip
        re-downloading the failed window's data.
        """
        try:
            dl = build_dl()
        except Exception as exc:
            msg = f"{error_label} download failed: {exc}"
            logger.error(msg)
            self.metadata["errors"].append(msg)
            self.metadata["downloads"][key] = {"status": "failed", "error": msg}
            return False

        merged_result: list = []
        failure_exc: Optional[Exception] = None
        for start, end in windows:
            try:
                download_kwargs = build_kwargs(start, end)
                result = dl.download(**download_kwargs)
                merged_result.extend(result or [])
            except Exception as exc:
                failure_exc = exc
                break

        if failure_exc is not None:
            msg = f"{error_label} download failed: {failure_exc}"
            logger.error(msg)
            self.metadata["errors"].append(msg)
            entry: Dict[str, Any] = {"status": "failed", "error": msg}
            if result_to_metadata is not None:
                entry.update(result_to_metadata(merged_result, dl))
            else:
                entry["files"] = [str(p) for p in merged_result]
            self.metadata["downloads"][key] = entry
            return False

        self._cleanup_if_empty(out_dir)
        entry = {"status": "dry_run" if self.dry_run else "success"}
        if result_to_metadata is not None:
            entry.update(result_to_metadata(merged_result, dl))
        else:
            entry["files"] = [str(p) for p in merged_result]
        self.metadata["downloads"][key] = entry
        return True

    def _load_previous_downloads(self) -> Dict[str, Any]:
        """
        Read the ``downloads`` section of a prior run's
        ``download_metadata.json`` in ``self.base_dir``, if present.

        Used by :meth:`_already_succeeded` so a rerun triggered by one
        source's failure (e.g. SMOS) does not force every other,
        already-succeeded source (e.g. ASCAT) to re-authenticate and
        re-dispatch — see docs/design-choices.md.
        """
        meta_path = self.base_dir / "download_metadata.json"
        if not meta_path.exists():
            return {}
        try:
            with open(meta_path) as f:
                return json.load(f).get("downloads", {})
        except Exception:
            return {}

    def _load_previous_variable(self) -> Optional[str]:
        """
        Read the top-level ``variable`` field of a prior run's
        ``download_metadata.json`` in ``self.base_dir``, if present.

        Used by :meth:`_already_succeeded` to detect when two recipes sharing
        identical geographic/temporal bounds (and therefore the same
        ``base_dir``) request different ``recipe.config.variable`` values.
        ERA5's downloaded file depends on ``variable`` (each maps to a
        distinct CDS dataset/variable set), unlike SAR, whose L2 OCN product
        contains fields for every variable regardless of which recipe
        downloaded it. A stale ``era5: {"status": "success"}`` entry recorded
        under one variable must therefore not be trusted for another.
        """
        meta_path = self.base_dir / "download_metadata.json"
        if not meta_path.exists():
            return None
        try:
            with open(meta_path) as f:
                variable = json.load(f).get("variable")
            return variable if isinstance(variable, str) else None
        except Exception:
            return None

    def _already_succeeded(self, source_type: str) -> bool:
        """
        True if *source_type* succeeded in the previous run recorded in
        ``self._previous_downloads`` and ``force_download`` is not set.
        """
        if self.force_download:
            return False
        prev = self._previous_downloads.get(source_type)
        if prev is None:
            return False
        if source_type == "era5" and self._previous_variable != self.recipe.config.variable:
            return False
        return prev.get("status") == "success"

    def _sar_dir_and_glob(self) -> tuple:
        """(sar_dir, file_glob) for this recipe's SAR source -- passed to
        _sar_entry_found_data so it can check disk directly for old-schema
        metadata entries (see that function's docstring)."""
        from .sar_sources import SAR_SOURCES

        spec = SAR_SOURCES[self.recipe.config.sar_data.source]
        return self.base_dir / spec.output_subdir, spec.file_glob

    def previous_sar_data_found(self) -> bool:
        """
        True if a previous run's recorded SAR entry (if any) found at
        least one product for this window. Used by the CLI's "already
        downloaded" resume shortcut, which reuses a prior run's cached
        metadata without calling :meth:`download_all` again.
        """
        sar_dir, file_glob = self._sar_dir_and_glob()
        return _sar_entry_found_data(
            self._previous_downloads.get("sar", {}), sar_dir=sar_dir, file_glob=file_glob,
        )

    def _padded_temporal_bounds(self, *source_types: str) -> List[Tuple[str, str]]:
        """
        List of (start, end) ISO-string windows, padded symmetrically by
        :func:`_resolve_temporal_padding_minutes` on each side of
        ``cfg.temporal_bounds``, for passing to a downloader's ``start``/
        ``end`` arguments once per window. Does not mutate
        ``cfg.temporal_bounds``, since other logic (output folder naming, 
        coverage-cutoff comparisons, metadata) must continue using the
        literal requested range.

        When ``self._sar_scene_times`` has been populated (see
        ``_compute_sar_scene_times``, called from ``download_all`` once
        real SAR files are on disk), the single nominal-padded window is
        replaced by one or more narrower windows clustered around the
        real SAR scene(s): consecutive, sorted scene times separated by
        more than ``2 * pad`` start a new cluster, since that is exactly the
        point at which their own ``+-pad`` windows stop overlapping. Each 
        cluster's own [min, max] is padded by the same amount and clamped 
        to the nominal padded window on both ends. The result is therefore
        always a subset of the nominal-padded window's own span, never wider, 
        and falls back to the single-element list whenever scene times are
        unavailable.
        """
        cfg = self.recipe.config
        pad = pd.Timedelta(minutes=_resolve_temporal_padding_minutes(cfg, *source_types))
        temp = cfg.temporal_bounds
        nominal_pad_start = pd.Timestamp(temp.start) - pad
        nominal_pad_end   = pd.Timestamp(temp.end) + pad
        nominal_window = [(nominal_pad_start.isoformat(), nominal_pad_end.isoformat())]

        if not self._sar_scene_times:
            return nominal_window

        try:
            # Defensive copy, sorted locally: this method's correctness must not
            # depend on _compute_sar_scene_times having already sorted
            # self._sar_scene_times, which would otherwise be a fragile,
            # load-bearing cross-method contract. Does not mutate
            # self._sar_scene_times itself.
            scene_times = sorted(self._sar_scene_times)

            clusters: list[list[pd.Timestamp]] = [[scene_times[0]]]
            for t in scene_times[1:]:
                if t - clusters[-1][-1] > 2 * pad:
                    clusters.append([t])
                else:
                    clusters[-1].append(t)

            windows: list[tuple[str, str]] = []
            for cluster in clusters:
                cluster_pad_start = cluster[0] - pad
                cluster_pad_end   = cluster[-1] + pad
                start_ts = max(nominal_pad_start, cluster_pad_start)
                end_ts   = min(nominal_pad_end, cluster_pad_end)
                if start_ts <= end_ts:
                    windows.append((start_ts.isoformat(), end_ts.isoformat()))

            return windows or nominal_window
        except Exception:
            # _compute_sar_scene_times' own docstring guarantees this clustering
            # is "never allowed to block download_all()"; that guarantee must
            # hold here too, where self._sar_scene_times is actually read, not
            # just where it is populated. Any unexpected failure in the
            # clustering above therefore falls back to the same nominal window
            # used when no scene times are available.
            logger.warning(
                "_padded_temporal_bounds: clustering around SAR scene times "
                "failed, falling back to the nominal padded window.",
                exc_info=True,
            )
            return nominal_window

    def _compute_sar_scene_times(self) -> None:
        """
        Populate ``self._sar_scene_times`` (sorted ascending) from the
        just-downloaded SAR files' embedded timestamps, reusing each source's
        own ``.convert`` callable (see ``sar_sources.SAR_SOURCES``) rather
        than new parsing code. Works identically for any SAR source type,
        since ``SAR_SOURCES`` is keyed dynamically by
        ``cfg.sar_data.source``.

        Left as ``None`` on any failure, falling back to the
        nominal-window-only padding in :meth:`_padded_temporal_bounds`; this
        is purely an optimization and must never block ``download_all()`` or
        narrow a window incorrectly.
        """
        from .sar_sources import SAR_SOURCES

        try:
            cfg = self.recipe.config
            spec = SAR_SOURCES[cfg.sar_data.source]
            product_type = cfg.variable
            files = self.metadata["downloads"].get("sar", {}).get("files", [])

            times: list = []
            for f in files:
                try:
                    ds = spec.convert(Path(f), product_type)
                    if ds is None or "time" not in ds.coords:
                        continue
                    raw = ds.coords["time"].values
                    idx = pd.to_datetime(np.atleast_1d(raw))
                    if getattr(idx, "tz", None) is not None:
                        # Some SAR sources (e.g. NISAR SME2's
                        # from_nisar_sme2, which parses an ISO string with
                        # a UTC designator via pd.to_datetime) produce a
                        # timezone-aware index. Normalize to timezone-naive here, 
                        # at population time, matching _domain_filter's
                        # established pattern in datatree_converter.py, so
                        # every entry in self._sar_scene_times is
                        # comparable against the rest of this module's
                        # timezone-naive timestamps.
                        idx = idx.tz_localize(None)
                    times.extend(idx.tolist())
                except Exception as exc:
                    logger.debug("Could not extract scene time from %s: %s", f, exc)

            if times:
                self._sar_scene_times = sorted(times)
        except Exception:
            # This method's contract (see docstring) is "never allowed
            # to block download_all() or narrow a window incorrectly". 
            # The per-file convert() failures above already follow that, 
            # but the final sorted(times) call, and anything else
            # in this method, must also be guarded: any unexpected failure 
            # here must leave self._sar_scene_times as None rather than 
            # propagate and terminate the run.
            logger.warning(
                "_compute_sar_scene_times failed unexpectedly, falling "
                "back to the nominal temporal window.", exc_info=True,
            )
            self._sar_scene_times = None

    def _collocation_predictions(self) -> "Dict[str, Any]":
        """Predicted collocation per validation source_type, computed
        once (cached) from the real downloaded+converted SAR files --
        empty dict if download_all_in_bbox is set (gating disabled, never
        even compute this) or if footprint derivation/prediction itself
        fails (fail open -- an empty dict means every source's lookup
        below falls through to its own default, no gating)."""
        if self.download_all_in_bbox:
            return {}
        if self._collocation_predictions_cache is not None:
            return self._collocation_predictions_cache

        from .dry_collocation import predict_collocation, sar_footprints_from_downloaded
        from .sar_sources import SAR_SOURCES

        try:
            sar_entry = self.metadata["downloads"].get("sar", {})
            sar_files = [Path(f) for f in sar_entry.get("files", [])]
            sar_source_spec = SAR_SOURCES[self.recipe.config.sar_data.source]
            sar_footprints = sar_footprints_from_downloaded(
                sar_files, sar_source_spec, self.recipe.config.variable,
            )
            logger.info(
                "Predicting collocation for %d validation source(s)...",
                len(self.recipe.config.validation_sources),
            )
            # stop_on_first_match=True: this real-run gating path only ever
            # needs a yes/no verdict per source, unlike the --dry-collocation
            # preview path (predict_collocation's own default, False), which
            # needs every predicate's exhaustive matched_windows count for
            # its report. See predict_source's docstring.
            report = predict_collocation(self.recipe.config, sar_footprints, stop_on_first_match=True)
            self._collocation_predictions_cache = {p.source_type: p for p in report.predictions}
        except Exception:
            logger.debug(
                "_collocation_predictions: prediction failed, disabling "
                "gating for this run", exc_info=True,
            )
            self._collocation_predictions_cache = {}
        return self._collocation_predictions_cache

    def _should_skip_for_collocation(self, source_type: str) -> bool:
        """True only for a CONFIRMED none-predicted verdict -- unknown
        (including "no prediction computed at all") never skips, per this
        feature's fail-open contract. Short-circuits on download_all_in_bbox
        before even calling _collocation_predictions(), so gating truly
        never runs (not just "runs and returns nothing") when it's
        disabled."""
        if self.download_all_in_bbox:
            return False
        prediction = self._collocation_predictions().get(source_type)
        return prediction is not None and prediction.verdict == "none-predicted"

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

        sar_entry = self.metadata["downloads"].get("sar", {})
        sar_dir, file_glob = self._sar_dir_and_glob()
        sar_data_found = _sar_entry_found_data(sar_entry, sar_dir=sar_dir, file_glob=file_glob)
        self.metadata["sar_data_found"] = sar_data_found
        if not sar_data_found:
            msg = "No SAR data found for this window/region — skipping validation-data downloads."
            logger.warning(msg)
            self.metadata["notices"].append(msg)
            if not self.dry_run:
                self._save_metadata()
            return ok

        self._compute_sar_scene_times()

        # 2. Delayed-mode ("*_historical") sources first. hf_radar and the
        # NRT in-situ batch (below) consult file_count from these results
        # to avoid re-downloading the same physical observations from the
        # NRT feed once the historical/reprocessed data already covers
        # this run's window.
        historical_had_data: Dict[str, bool] = {}
        for source in self.recipe.config.validation_sources:
            if source.source_type not in _HISTORICAL_FIRST_TYPES:
                continue
            if self._should_skip_for_collocation(source.source_type):
                self.metadata["downloads"][source.source_type] = {
                    "status": "skipped", "reason": "no predicted collocation with SAR data",
                }
                logger.info("Skipping %s: no predicted collocation.", source.source_type)
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
            and not self._should_skip_for_collocation(s.source_type)
        ]
        if source_types:
            # Use the most permissive depth window across the in-situ
            # sources actually being requested (excludes any source dropped
            # above, so an excluded source's depth override cannot widen a
            # batch it is no longer part of).
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
            if self._should_skip_for_collocation(source.source_type):
                self.metadata["downloads"][source.source_type] = {
                    "status": "skipped", "reason": "no predicted collocation with SAR data",
                }
                logger.info("Skipping %s: no predicted collocation.", source.source_type)
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
            [(temp.start, temp.end)],
            lambda start, end: dict(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=start,            end=end,
                **spec.extra_download_kwargs(cfg.sar_data),
            ),
            f"SAR ({cfg.sar_data.source})",
            result_to_metadata=lambda result, dl: {
                "files": [str(p) for p in (result or [])],
                "found_count": getattr(dl, "found_count", len(result or [])),
            },
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
        windows = self._padded_temporal_bounds(*source_types)

        out_dir = self.base_dir / "copernicus_insitu"

        try:
            dl = InSituDownloader(
                output_dir=out_dir,
                dry_run=self.dry_run,
                min_depth=min_depth,
                max_depth=max_depth,
                force_download=self.force_download,
            )
            for start, end in windows:
                dl.download(
                    min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                    min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                    start=start,            end=end,
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
        """
        Log one combined warning for the four delayed-mode in-situ
        current instruments (adcp/argo/drifter/glider) when every one of
        them present in this recipe produced zero files, instead of each
        instrument logging its own near-identical warning.
        """
        if self.dry_run:
            return
        attempted = [
            t for t in _CURRENTS_INSTRUMENT_TYPES
            if any(s.source_type == t for s in self.recipe.config.validation_sources)
            # A "skipped" status means collocation-based gating never even
            # attempted this source_type's download -- it must not count
            # toward "we tried and got nothing", which this warning implies.
            and self.metadata["downloads"].get(t, {}).get("status") != "skipped"
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
        """
        Log one combined 'no data' notice for hf_radar_us, naming which
        backends were tried, instead of the silent zero-message outcome
        today's per-source status recording produces for an empty result.
        """
        if self.dry_run:
            return
        entry = self.metadata["downloads"].get("hf_radar_us")
        # "skipped" means collocation-based gating never even attempted
        # this download -- it must not be reported as "no data found",
        # which implies a real, empty download attempt.
        if entry is None or entry.get("status") in ("failed", "skipped"):
            return
        if entry.get("file_count", 0) > 0:
            return
        backends = entry.get("attempted_backends") or []
        msg = f"No US HF-radar data found (tried {', '.join(backends)}) for this window."
        logger.warning(msg)
        self.metadata["notices"].append(msg)

    def _report_combined_hf_radar_status(self) -> None:
        """
        Log one combined 'no data' notice for the non-US hf_radar /
        hf_radar_historical pair, mirroring _report_combined_currents_status.
        """
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
            # "skipped" (collocation-based gating never even attempted
            # this download) must not count toward "we tried and got
            # nothing" any more than "failed" does.
            if entry.get("status") in ("failed", "skipped"):
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
            "scatterometer_ascat": self._download_scatterometer_ascat,
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
            "era5":          self._download_era5,
            "hycom":         self._download_hycom,
        }
        handler = handlers.get(source.source_type)
        if handler is None:
            msg = f"No downloader for source_type '{source.source_type}'"
            logger.warning(msg)
            self.metadata["errors"].append(msg)
            return False
        return handler(source)

    def _download_scatterometer_ascat(self, source) -> bool:
        from ..downloaders.scatterometer_downloader import ScatterometerDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        # "scatterometer", not source.source_type ("scatterometer_ascat"):
        # this is the layer_type key (DEFAULT_LAYER_TYPE_SPECS in
        # recipe.py), not the recipe source_type -- ASCAT's own 12.5km
        # tolerance stays keyed "scatterometer" regardless of what the
        # recipe-facing source_type is named, since every one of the four
        # scatterometer source_types (ASCAT here, HY-2B/HY-2C/Oceansat-3
        # elsewhere) is stamped with the same shared
        # data_type="scatterometer" at conversion time (see
        # from_scatterometer_nc in datatree_converter.py) and
        # _resolve_layer_type only refines that into a more specific key
        # for the three FTP-sourced satellites, never for ASCAT. This
        # handler is only ever dispatched for source_type
        # "scatterometer_ascat" (see _dispatch_source), and some existing
        # tests call it directly with source=None.
        windows = self._padded_temporal_bounds("scatterometer")
        out_dir = self.base_dir / "osi_saf_winds"

        return self._run_download(
            "scatterometer_ascat", out_dir,
            lambda: ScatterometerDownloader(
                output_dir=out_dir, dry_run=self.dry_run, force_download=self.force_download,
            ),
            windows,
            lambda start, end: dict(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=start, end=end,
            ),
            "Scatterometer",
            result_to_metadata=lambda result, dl: {},
        )

    def _download_ascat_ssm(self, source) -> bool:
        from ..downloaders.ascat_soil_moisture_downloader import ASCATSoilMoistureDownloader
        from ..downloaders.hsaf_downloader import HSAFDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        windows = self._padded_temporal_bounds(source.source_type)
        eumdac_dir = self.base_dir / "ascat_ssm"
        hsaf_dir   = self.base_dir / "hsaf_ascat_ssm"

        # H-SAF's on-line archive (H122 default, H29 via
        # download_kwargs.hsaf_product) holds only a rolling
        # last-60-days window. Anything before that, down to the 
        # EUMDAC cutoff, is a coverage gap.
        today = datetime.now(timezone.utc).date()
        hsaf_window_start = (today - timedelta(days=60)).isoformat()

        # windows is sorted ascending; req_start/req_end are the overall 
        # requested span's bounds, used for the branch-level
        # attempted/coverage-cutoff logic. The per-window checks below
        # additionally skip any individual window failing entirely 
        # outside a branch's own coverage.
        req_start = windows[0][0][:10]
        req_end   = windows[-1][1][:10]

        eumdac_attempted = req_start <= _ASCAT_COVERAGE_CUTOFF
        hsaf_attempted = req_end >= hsaf_window_start

        files: list = []
        eumdac_ok = True
        hsaf_ok = True

        if eumdac_attempted:
            try:
                eumdac_dl = ASCATSoilMoistureDownloader(
                    output_dir=eumdac_dir, dry_run=self.dry_run,
                    force_download=self.force_download,
                )
                for start, end in windows:
                    if start[:10] > _ASCAT_COVERAGE_CUTOFF:
                        continue
                    files.extend(eumdac_dl.download(
                        min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                        min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                        start=start, end=end,
                    ) or [])
            except Exception as exc:
                eumdac_ok = False
                logger.error("ASCAT SSM (EUMDAC) download failed: %s", exc)

        if hsaf_attempted:
            try:
                hsaf_product = source.download_kwargs.get("hsaf_product", "h122")
                hsaf_dl = HSAFDownloader(
                    output_dir=hsaf_dir, dry_run=self.dry_run,
                    force_download=self.force_download,
                    product=hsaf_product,
                )
                for start, end in windows:
                    if end[:10] < hsaf_window_start:
                        continue
                    files.extend(hsaf_dl.download(
                        min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                        min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                        start=start, end=end,
                    ) or [])
            except Exception as exc:
                hsaf_ok = False
                logger.error("ASCAT SSM (H-SAF) download failed: %s", exc)

        self._cleanup_if_empty(eumdac_dir)
        self._cleanup_if_empty(hsaf_dir)

        # An unattempted branch (out of range) is not a failure; a hard
        # failure occurs only if every attempted branch fails. Since
        # EUMDAC/H-SAF coverage never overlaps, both are attempted only for
        # an unusually wide request. If exactly one attempted branch fails,
        # that is a partial failure, not a hard one: real data still exists,
        # so a notice is reported instead of a hard "download failed" (which
        # would incorrectly flag the report and trigger a full re-download
        # via _already_succeeded).
        attempted_ok_flags = [
            ok_flag for attempted, ok_flag in (
                (eumdac_attempted, eumdac_ok), (hsaf_attempted, hsaf_ok),
            ) if attempted
        ]
        hard_failure = bool(attempted_ok_flags) and not any(attempted_ok_flags)
        partial_failure = not hard_failure and not all(attempted_ok_flags)

        ok = not hard_failure
        self.metadata["downloads"]["ascat_ssm"] = {
            "status": "dry_run" if self.dry_run else ("success" if ok else "failed"),
            "files": [str(p) for p in files],
        }
        if hard_failure:
            self.metadata["errors"].append("ASCAT SSM download failed (see log)")
        elif partial_failure:
            self.metadata["notices"].append(
                "ASCAT: one of the EUMDAC/H-SAF branches failed for this "
                "request's date range while the other succeeded -- see log "
                "for details. Results may be incomplete for the failed "
                "branch's portion of the range."
            )

        if (
            ok and not files
            and req_start > _ASCAT_COVERAGE_CUTOFF and req_end < hsaf_window_start
        ):
            self.metadata["notices"].append(
                f"ASCAT: requested range [{req_start}, {req_end}] falls entirely in "
                f"the gap between the EUMDAC coverage cutoff ({_ASCAT_COVERAGE_CUTOFF}) "
                f"and H-SAF's rolling last-60-days on-line archive ({hsaf_window_start}) — "
                f"0 products found (expected, not an error). H-SAF's off-line/CDR "
                f"archive covers this gap but requires a manually-placed order, not "
                f"automated by this toolbox."
            )
        elif (
            ok and not self.dry_run and not files
            and req_end > _ASCAT_COVERAGE_CUTOFF
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
        windows = self._padded_temporal_bounds(source.source_type)
        out_dir = self.base_dir / out_subdir

        try:
            dl = EarthdataSoilMoistureDownloader(
                dataset=dataset, version=version, output_dir=out_dir, dry_run=self.dry_run,
            )
        except Exception as exc:
            msg = f"{dataset} download failed: {exc}"
            logger.error(msg)
            self.metadata["errors"].append(msg)
            self.metadata["downloads"][out_subdir] = {"status": "failed", "error": msg}
            return False

        # `paths` is declared before the loop, not inside it, so that if a
        # later window's download raises, the results already accumulated
        # from earlier windows in this same call are preserved rather than
        # lost when the loop breaks. Mirrors _download_ascat_ssm's own
        # "files" accumulator, which uses the same pattern.
        paths: list = []
        failure_exc: Optional[Exception] = None
        for start, end in windows:
            try:
                paths.extend(dl.download(
                    min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                    min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                    start=start, end=end,
                ) or [])
            except Exception as exc:
                failure_exc = exc
                break

        if failure_exc is not None:
            # Same partial-failure handling _run_download documents for
            # its own windows loop: preserve earlier windows' results,
            # but still record the run as a failure so cli.py's
            # _is_already_downloaded retries it.
            msg = f"{dataset} download failed: {failure_exc}"
            logger.error(msg)
            self.metadata["errors"].append(msg)
            self.metadata["downloads"][out_subdir] = {
                "status": "failed", "error": msg, "files": [str(p) for p in paths],
            }
            return False

        self._cleanup_if_empty(out_dir)
        self.metadata["downloads"][out_subdir] = {
            "status": "dry_run" if self.dry_run else "success",
            "files":  [str(p) for p in paths],
        }
        return True

    def _download_amsr_ssm(self, source) -> bool:
        temp = self.recipe.config.temporal_bounds
        if temp.end <= _NSIDC_0451_CUTOFF:
            dataset = "NSIDC-0451"
        else:
            dataset = "AU_Land"
        ok = self._download_earthdata_ssm(source, dataset=dataset, version=None, out_subdir="amsr_ssm")
        if ok and not self.metadata["downloads"].get("amsr_ssm", {}).get("files"):
            # If NASA Earthdata returns 0 files, G-Portal (below) is a 
            # second source, and often succeeds when Earthdata does not. 
            # Only note the coverage cutoff once *both* have been tried 
            # and both found nothing (see _try_gportal_amsr_fallback).
            self._try_gportal_amsr_fallback()
        return ok

    def _try_gportal_amsr_fallback(self) -> None:
        """
        Best-effort fallback when NASA Earthdata's AMSR2 coverage
        (frozen at _AMSR_COVERAGE_CUTOFF) returns zero files: try JAXA's
        own G-Portal SFTP archive for the same window. A failure here
        (missing credentials, discovery failure, connection error) is
        recorded as a notice, not an error as it is not a required source.
        """
        from ..downloaders.gportal_downloader import GPortalAMSR2Downloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        windows = self._padded_temporal_bounds("amsr_ssm")
        out_dir = self.base_dir / "amsr_ssm"

        try:
            dl = GPortalAMSR2Downloader(
                output_dir=out_dir, dry_run=self.dry_run, force_download=self.force_download,
                allow_prompt=False,
            )
            paths: list = []
            for start, end in windows:
                paths.extend(dl.download(
                    min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                    min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                    start=start, end=end,
                ) or [])
            self._cleanup_if_empty(out_dir)
            entry = self.metadata["downloads"].setdefault("amsr_ssm", {})
            entry["files"] = [str(p) for p in paths]
            entry["status"] = "dry_run" if self.dry_run else "success"
            entry["gportal_fallback"] = True
            if not paths and not self.dry_run:
                # 0 files is a normal, non-error outcome (e.g. no AMSR2 coverage for
                # this window), but "status": "success" with 0 files is otherwise
                # indistinguishable from a genuine download -- add a notice
                # explaining it, if the known coverage cutoff applies.
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
            # A fresh reconnect can fail on a purely transient blip even 
            # though real AMSR2 files from an earlier successful run 
            # already sit in out_dir. The pipeline then still has good data 
            # to use, so this must not be surfaced as a failure in the report. 
            existing_files = sorted(p for p in out_dir.glob("*") if p.is_file())
            if existing_files:
                logger.warning(
                    "G-Portal AMSR2 fallback failed (%s), but %d existing file(s) "
                    "already present in %s — reusing them, not reporting a failure.",
                    exc, len(existing_files), out_dir,
                )
                entry = self.metadata["downloads"].setdefault("amsr_ssm", {})
                entry["files"] = [str(p) for p in existing_files]
                entry["status"] = "success"
                entry["gportal_fallback"] = True
                return
            msg = f"G-Portal AMSR2 fallback failed: {exc}"
            logger.warning(msg)
            self.metadata["notices"].append(msg)

    def _download_smap_ssm(self, source) -> bool:
        return self._download_earthdata_ssm(source, dataset="SPL2SMP_E", version="006", out_subdir="smap_ssm")

    def _download_smos_ssm(self, source) -> bool:
        from ..downloaders.smos_downloader import SMOSDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        windows = self._padded_temporal_bounds(source.source_type)
        out_dir = self.base_dir / "smos_ssm"

        return self._run_download(
            "smos_ssm", out_dir,
            lambda: SMOSDownloader(output_dir=out_dir, dry_run=self.dry_run),
            windows,
            lambda start, end: dict(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=start, end=end,
            ),
            "SMOS SSM",
        )

    def _download_cds_ssm(self, source) -> bool:
        from ..downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        windows = self._padded_temporal_bounds(source.source_type)
        out_dir = self.base_dir / "cds_ssm"
        product_type = source.download_kwargs.get("product_type", "active")

        return self._run_download(
            "cds_ssm", out_dir,
            lambda: CDSSoilMoistureDownloader(
                product_type=product_type,
                output_dir=out_dir, dry_run=self.dry_run,
            ),
            windows,
            lambda start, end: dict(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=start, end=end,
            ),
            f"C3S CDS SSM ({product_type})",
        )

    def _download_era5(self, source) -> bool:
        from ..downloaders.era5_downloader import ERA5Downloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        temp   = cfg.temporal_bounds
        # ERA5Downloader performs its own bracket-margin widening internally
        # (via its time_tolerance_minutes parameter), driven by this same
        # resolved value. So the literal, unpadded recipe window is passed
        # here rather than a separately padded one.
        tolerance = _resolve_temporal_padding_minutes(cfg, source.source_type)
        out_dir = self.base_dir / "era5"

        return self._run_download(
            "era5", out_dir,
            lambda: ERA5Downloader(
                variable=cfg.variable,  # type: ignore[arg-type]
                output_dir=out_dir, dry_run=self.dry_run,
                time_tolerance_minutes=tolerance,
            ),
            [(temp.start, temp.end)],
            lambda start, end: dict(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=start, end=end,
            ),
            f"ERA5 ({cfg.variable})",
        )

    def _download_hycom(self, source) -> bool:
        from ..downloaders.hycom_downloader import HycomDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        temp   = cfg.temporal_bounds
        # HycomDownloader performs its own bracket-margin widening internally
        # (via its time_tolerance_minutes parameter), driven by this same
        # resolved value -- see _download_era5's identical rationale above.
        tolerance = _resolve_temporal_padding_minutes(cfg, source.source_type)
        out_dir = self.base_dir / "hycom"

        return self._run_download(
            "hycom", out_dir,
            lambda: HycomDownloader(
                output_dir=out_dir, dry_run=self.dry_run, time_tolerance_minutes=tolerance,
            ),
            [(temp.start, temp.end)],
            lambda start, end: dict(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=start, end=end,
            ),
            "HyCOM",
        )

    def _download_scatterometer_ftp(self, source, satellite: str) -> bool:
        from ..downloaders.scatterometer_ftp_downloader import ScatterometerFTPDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        windows = self._padded_temporal_bounds(source.source_type)
        out_dir = self.base_dir / f"scatterometer_{satellite}"

        return self._run_download(
            f"scatterometer_{satellite}", out_dir,
            lambda: ScatterometerFTPDownloader(
                satellite=satellite,
                output_dir=out_dir, dry_run=self.dry_run, force_download=self.force_download,
            ),
            windows,
            lambda start, end: dict(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=start, end=end,
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
        windows = self._padded_temporal_bounds("hf_radar_grid")
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
            windows,
            lambda start, end: dict(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=start, end=end,
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
        windows = self._padded_temporal_bounds("hf_radar_grid")
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
            windows,
            lambda start, end: dict(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=start, end=end,
            ),
            "NOAA HF-radar",
            result_to_metadata=lambda result, dl: {},
        )

    def _download_hf_radar_historical(self, source) -> bool:
        from ..downloaders.hf_radar_historical_downloader import HFRadarHistoricalDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        windows = self._padded_temporal_bounds("hf_radar_grid")
        out_dir = self.base_dir / "hf_radar_historical"

        return self._run_download(
            "hf_radar_historical", out_dir,
            lambda: HFRadarHistoricalDownloader(
                output_dir=out_dir, dry_run=self.dry_run, force_download=self.force_download,
            ),
            windows,
            lambda start, end: dict(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=start, end=end,
            ),
            "HF radar historical",
            result_to_metadata=lambda result, dl: {"file_count": len(result or [])},
        )

    def _download_hf_radar_us(self, source) -> bool:
        from ..downloaders.hf_radar_us_downloader import HFRadarUSDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        windows = self._padded_temporal_bounds("hf_radar_grid")
        # Resolution is an optional per-source override, forwarded via the
        # established ValidationDataSource.download_kwargs channel: None
        # (not set in the recipe) lets HFRadarUSDownloader auto-pick the
        # matched region's own default, while "finest" and explicit floats
        # are forwarded as-is. Ignored on the Copernicus fallback path.
        resolution_km = source.download_kwargs.get("resolution_km")

        try:
            dl = HFRadarUSDownloader(
                output_dir=self.base_dir,
                dry_run=self.dry_run,
                resolution_km=resolution_km,
                force_download=self.force_download,
            )
        except Exception as exc:
            msg = f"US HF-radar download failed: {exc}"
            logger.error(msg)
            self.metadata["errors"].append(msg)
            self.metadata["downloads"]["hf_radar_us"] = {"status": "failed", "error": msg}
            return False

        # Declared outside the per-window try (see _download_ascat_ssm's
        # own "files" accumulator for the same pattern this mirrors) so a
        # later window's exception cannot discard an earlier window's
        # already-collected results.
        downloaded: list = []
        failure_exc: Optional[Exception] = None
        for start, end in windows:
            try:
                downloaded.extend(dl.download(
                    min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                    min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                    start=start, end=end,
                ) or [])
            except Exception as exc:
                failure_exc = exc
                break

        if failure_exc is not None:
            # Same partial-failure handling _run_download documents for
            # its own windows loop: preserve earlier windows' results,
            # but still record the run as a failure so cli.py's
            # _is_already_downloaded retries it.
            msg = f"US HF-radar download failed: {failure_exc}"
            logger.error(msg)
            self.metadata["errors"].append(msg)
            self.metadata["downloads"]["hf_radar_us"] = {
                "status": "failed", "error": msg,
                "file_count": len(downloaded),
                "backend": dl.resolved_backend,
                "attempted_backends": dl.attempted_backends,
            }
            return False

        for subdir in ("hfr_noaa", "hf_radar", "hf_radar_historical"):
            self._cleanup_if_empty(self.base_dir / subdir)
        self.metadata["downloads"]["hf_radar_us"] = {
            "status": "dry_run" if self.dry_run else "success",
            "file_count": len(downloaded),
            "backend": dl.resolved_backend,
            "attempted_backends": dl.attempted_backends,
        }
        return True

    def _download_currents_historical(self, source, instrument: str) -> bool:
        from ..downloaders.insitu_currents_historical_downloader import (
            InSituCurrentsHistoricalDownloader,
        )

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        windows = self._padded_temporal_bounds(source.source_type)
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
            windows,
            lambda start, end: dict(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=start, end=end,
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

    # Altimeter download frequencies, keyed by recipe variable. This is
    # only a fallback for a recipe whose altimeter ValidationDataSource
    # omits "frequencies" (hand-edited recipes). Wind never needs 5 Hz
    # (no WIND_SPEED there); waves also defaults to 1 Hz, since 
    # co-downloading both by default over-samples SAR pixels near the 
    # ground track. A recipe can still opt into 5 Hz or both via 
    # --altimeter-freq at creation time, or by editing download_kwargs manually.
    _ALTIMETER_FREQUENCIES_BY_VARIABLE = {
        "wind":  ["1hz"],
        "waves": ["1hz"],
    }

    def _download_altimeter(self, source) -> bool:
        from ..downloaders.altimeter_downloader import AltimeterDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        # DEFAULT_LAYER_TYPE_SPECS keys altimeter by frequency
        # ("altimeter_1hz"/"altimeter_5hz"), not the bare "altimeter"
        # source_type; both are passed so the padding lookup finds their
        # (equal, 180min) tolerance regardless of which frequency this
        # recipe's variable requests.
        windows = self._padded_temporal_bounds("altimeter_1hz", "altimeter_5hz")
        out_dir = self.base_dir / "altimeter"
        kwargs = {
            "frequencies": self._ALTIMETER_FREQUENCIES_BY_VARIABLE.get(
                cfg.variable, ["1hz"]
            ),
        }
        kwargs.update(source.download_kwargs)   # recipe-level override wins

        return self._run_download(
            "altimeter", out_dir,
            lambda: AltimeterDownloader(
                output_dir=out_dir, dry_run=self.dry_run, force_download=self.force_download,
            ),
            windows,
            lambda start, end: dict(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=start, end=end,
                **kwargs,
            ),
            "Altimeter",
        )

    def _download_radiometer(self, source) -> bool:
        from ..downloaders.radiometer_downloader import RadiometerDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        windows = self._padded_temporal_bounds(source.source_type)
        out_dir = self.base_dir / "radiometer"
        kwargs = dict(source.download_kwargs)   # e.g. {"sensors": ["amsr2"]}

        return self._run_download(
            "radiometer", out_dir,
            lambda: RadiometerDownloader(output_dir=out_dir, dry_run=self.dry_run),
            windows,
            lambda start, end: dict(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=start, end=end,
                **kwargs,
            ),
            "Radiometer",
        )

    def _download_ismn(self, source) -> bool:
        from ..downloaders.ismn_downloader import ISMNDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        windows = self._padded_temporal_bounds(source.source_type)
        out_dir = self.base_dir / "ismn"

        try:
            dl = ISMNDownloader(output_dir=out_dir, dry_run=self.dry_run)
        except Exception as exc:
            msg = f"ISMN selection failed: {exc}"
            logger.error(msg)
            self.metadata["errors"].append(msg)
            self.metadata["downloads"]["ismn"] = {"status": "failed", "error": msg}
            return False

        # Declared outside the per-window try (see _download_ascat_ssm's
        # own "files" accumulator for the same pattern this mirrors) so a
        # later window's exception cannot discard an earlier window's
        # already-collected results.
        paths: list = []
        failure_exc: Optional[Exception] = None
        for start, end in windows:
            try:
                paths.extend(dl.download(
                    min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                    min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                    start=start, end=end,
                    min_depth=source.resolved_min_depth,
                    max_depth=source.resolved_max_depth,
                    archive_path=source.download_kwargs.get("ismn_archive_path"),
                ) or [])
            except Exception as exc:
                failure_exc = exc
                break

        if failure_exc is not None:
            # Same partial-failure handling _run_download documents for
            # its own windows loop: preserve earlier windows' results,
            # but still record the run as a failure so cli.py's
            # _is_already_downloaded retries it.
            msg = f"ISMN selection failed: {failure_exc}"
            logger.error(msg)
            self.metadata["errors"].append(msg)
            self.metadata["downloads"]["ismn"] = {
                "status": "failed", "error": msg, "files": [str(p) for p in paths],
            }
            return False

        self._cleanup_if_empty(out_dir)
        if self.dry_run:
            status = "dry_run"
        elif paths:
            status = "success"
        else:
            status = "awaiting_manual_archive"
        self.metadata["downloads"]["ismn"] = {
            "status": status,
            "files":  [str(p) for p in paths],
        }
        return True

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _save_metadata(self) -> None:
        meta_path = self.base_dir / "download_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)
        logger.info("Metadata saved to %s", meta_path)
