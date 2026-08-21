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
    from .recipe import DEFAULT_LAYER_TYPE_SPECS, min_safe_model_time_tolerance_minutes

    coll_cfg = cfg.collocation
    layer_specs = dict(DEFAULT_LAYER_TYPE_SPECS)
    if coll_cfg.layer_vs_layer is not None:
        # Per-key deep merge (not layer_specs.update(...), which would
        # replace a key's whole dict) -- a recipe overriding only e.g.
        # "method" for "era5_wind" must not silently lose the default's
        # "time_tolerance_minutes" in the process. Mirrors
        # datatree_converter.py's _build_subset_kwargs, which already does
        # this correctly.
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
        # "era5" (the ValidationDataSource.source_type actually used in
        # every recipe/template) has no DEFAULT_LAYER_TYPE_SPECS entry of
        # its own -- only the variable-specific "era5_wind"/"era5_waves"/
        # "era5_soil_moisture" keys do (matching the data_type each
        # DataTreeConverter.from_era5 node is actually stamped with; see
        # visualization.py's own tolerance lookup, which reads that
        # per-node data_type and so doesn't need this alias). Without this,
        # every era5 download silently fell back to the generic
        # point_vs_layer tolerance (30 min) instead of its own
        # (bracket-safe) value.
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
    """True if a recorded ``"sar"`` download-metadata entry represents at
    least one product found for the window, a failed download attempt
    (whose true "was there data" answer is unknown, so must not be
    treated as empty), or the absence of any recorded entry at all (no
    previous run to consult, so there's nothing to gate on).

    Entries recorded before ``found_count`` existed can have an empty
    ``"files"`` list even though real products were found and downloaded:
    ``SARDownloader.download()`` (and its per-source siblings) only
    appends *newly* downloaded files to the list it returns -- a product
    that was already present on disk (skipped as a duplicate) is never
    appended. So a fully successful old-schema run where every matched
    product happened to already be cached ends up recording ``files: []``
    despite real data being present -- confirmed live 2026-08-11: a
    ``currents_useastcoast.yaml`` run's copied-over ``download_metadata.json``
    (written before ``found_count`` existed, SAR genuinely downloaded but
    every product already cached from an earlier run) was misread as "no
    SAR data found," even though the SAR files were sitting right there in
    ``S1_L2_OCN/``. For an entry with no ``found_count`` and an empty
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
    on every future run."""
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
            # Non-failure, user-facing observations (e.g. "no data found for
            # this window") -- distinct from "errors", which
            # _is_already_downloaded (cli.py) treats as "something went
            # wrong, don't skip re-download next time". A notice must never
            # trigger that.
            "notices":   [],
        }

        # Populated by _compute_sar_scene_times() (see download_all()) from
        # the real downloaded SAR files' own embedded timestamps, sorted
        # ascending -- consumed by _padded_temporal_bounds to narrow every
        # validation source's download window to what the actual SAR
        # scene(s) need, not the recipe's full nominal temporal_bounds.
        # None until then, and stays None on any extraction failure -- this
        # is purely an optimization, never allowed to change what a
        # download would do in its absence.
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
        """Remove out_dir if the download produced no files anywhere under
        it (including nested, otherwise-empty subdirectories). Directories
        that already contain at least one file are left untouched."""
        if out_dir.exists() and not any(p.is_file() for p in out_dir.rglob("*")):
            shutil.rmtree(out_dir)

    def _run_download(
        self, key: str, out_dir: Path, build_dl, windows, build_kwargs, error_label: str,
        *, result_to_metadata=None,
    ) -> bool:
        """Shared skeleton for the simple ``_download_*`` handlers: build
        the downloader once, call ``.download()`` once per window in
        *windows* (concatenating results across windows -- a
        single-element windows list, the common case when SAR-scene
        clustering isn't in effect, makes exactly one call, identical to
        today's behavior), clean up an empty output dir, and record
        success/failure metadata under *key*.

        *build_dl* is a zero-arg callable returning the constructed
        downloader (built once, reused across every window). *windows* is
        a list of ``(start, end)`` ISO-string pairs, e.g. from
        ``self._padded_temporal_bounds(...)``, or a literal single-element
        list for a caller that doesn't window (e.g. SAR itself).
        *build_kwargs* is a ``(start, end) -> dict`` callable returning
        that window's ``download()`` keyword-argument dict (every other
        kwarg, e.g. bbox, stays the same across windows -- only
        start/end vary). *result_to_metadata*, if given, maps
        ``(merged_result, downloader) -> dict`` merged into the success
        entry; default is ``{"files": [str(p) for p in merged_result]}``.

        A failure partway through *windows* (network error, bad kwargs,
        etc.) does not discard whatever earlier windows already produced
        -- the partial ``merged_result`` is still recorded under *key*
        (so files already on disk stay tracked in metadata) -- but the
        run as a whole is always recorded as a FAILURE when any window
        fails, whether or not earlier windows already succeeded:
        ``{"status": "failed", "error": ..., "files": [...]}``, an error
        appended to ``self.metadata["errors"]``, and ``False`` returned.
        This is deliberate, not an oversight: ``cli.py``'s
        ``_is_already_downloaded`` gates purely on
        ``metadata["errors"]`` being empty, so a partial failure must
        stay visible as an error or the next run would silently skip
        re-downloading the failed window's data forever. See
        ``_download_ascat_ssm``'s own partial/hard-failure split for the
        same "preserve partial results, still fail the run" philosophy
        applied to its two named EUMDAC/H-SAF branches.
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

    def _load_previous_variable(self) -> Optional[str]:
        """Read the top-level ``variable`` field of a prior run's
        ``download_metadata.json`` in ``self.base_dir``, if present.

        Used by :meth:`_already_succeeded` to detect the case where two
        recipes with identical geographic/temporal bounds (so they share
        ``base_dir``) request *different* ``recipe.config.variable``
        values -- e.g. ``wind_era5.yaml`` and ``waves_era5.yaml`` in
        ``recipes/`` both cover the same bbox/window to reuse the SAR
        download. ERA5's downloaded file name/content depends on
        ``variable`` (``era5_<variable>_<day>.nc``, requesting a
        completely different CDS dataset/variable set), unlike SAR (whose
        L2 OCN product contains both wind- and wave-relevant OWI fields
        regardless of which recipe downloaded it) -- so a stale
        ``era5: {"status": "success"}`` recorded under one variable must
        NOT be trusted for another. Confirmed live 2026-08-07: running
        waves_era5.yaml right after wind_era5.yaml (same bbox/window)
        skipped the ERA5 download entirely, leaving zero era5 data in the
        DataTree for "waves".
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
        """True if *source_type* succeeded in the previous run recorded in
        ``self._previous_downloads`` and ``force_download`` isn't set."""
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
        """True if a previous run's recorded SAR entry (if any) found at
        least one product for this window. Used by the CLI's "already
        downloaded" resume shortcut, which reuses a prior run's cached
        metadata without calling :meth:`download_all` again."""
        sar_dir, file_glob = self._sar_dir_and_glob()
        return _sar_entry_found_data(
            self._previous_downloads.get("sar", {}), sar_dir=sar_dir, file_glob=file_glob,
        )

    def _padded_temporal_bounds(self, *source_types: str) -> List[Tuple[str, str]]:
        """List of (start, end) ISO-string windows, padded symmetrically by
        :func:`_resolve_temporal_padding_minutes` on each side of
        ``cfg.temporal_bounds`` -- for passing to a downloader's own
        ``start``/``end`` arguments, once per window. Does not mutate
        ``cfg.temporal_bounds`` itself, since other logic (output folder
        naming, coverage-cutoff comparisons, metadata) must keep using the
        literal requested range.

        When ``self._sar_scene_times`` has been populated (see
        ``_compute_sar_scene_times``, called from ``download_all`` once
        real SAR files are on disk), the single nominal-padded window is
        replaced by one *or more* narrower windows, clustered around the
        real SAR scene(s): consecutive (sorted) scene times separated by
        more than ``2 * pad`` start a new cluster -- that's exactly the
        point at which their own ``+-pad`` windows stop overlapping, so a
        genuine gap with nothing to collocate against sits between them.
        Each cluster's own [min, max] is padded by the same ``pad`` and
        clamped to the nominal padded window on both ends. The result can
        only ever be a subset of the nominal-padded window's own span --
        never wider -- and falls back to exactly today's single-window
        behavior (a one-element list) whenever scene times aren't
        available. See the design doc's Part 1.
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
            # Defensive copy, sorted locally -- this method's correctness
            # must not depend on _compute_sar_scene_times having already
            # sorted self._sar_scene_times (a load-bearing cross-method
            # contract otherwise, see also _download_ascat_ssm's
            # windows[0][0]/windows[-1][1] extraction). Does not mutate
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
            # _compute_sar_scene_times' own docstring promises "never
            # allowed to block download_all()" -- that contract must hold
            # here too, at the point self._sar_scene_times is actually
            # read, not just at the point it's populated. Any unexpected
            # failure in the clustering above (e.g. a scene time that
            # somehow escaped normalization) falls back to exactly the
            # same nominal window used when no scene times are available.
            logger.warning(
                "_padded_temporal_bounds: clustering around SAR scene times "
                "failed, falling back to the nominal padded window.",
                exc_info=True,
            )
            return nominal_window

    def _compute_sar_scene_times(self) -> None:
        """Populate self._sar_scene_times (sorted ascending) from the
        just-downloaded SAR files' real embedded timestamps, reusing each
        source's own .convert callable (see sar_sources.py's SAR_SOURCES
        registry) -- never new parsing code, and works identically for
        every SAR source type (sentinel1_l2_ocn, sentinel1_clms_ssm,
        nisar_sme2, radarsat2, or any future entry), since SAR_SOURCES is
        keyed dynamically by cfg.sar_data.source. Left as None (falls
        back to today's nominal-window-only padding, see
        _padded_temporal_bounds) on ANY failure -- this is purely an
        optimization, never allowed to block download_all() or narrow a
        window incorrectly.
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
                        # tz-AWARE index -- normalize to tz-naive here, at
                        # population time, matching _domain_filter's
                        # established pattern in datatree_converter.py, so
                        # every entry in self._sar_scene_times is
                        # comparable against the rest of this module's
                        # (tz-naive) timestamps.
                        idx = idx.tz_localize(None)
                    times.extend(idx.tolist())
                except Exception as exc:
                    logger.debug("Could not extract scene time from %s: %s", f, exc)

            if times:
                self._sar_scene_times = sorted(times)
        except Exception:
            # This method's own contract (see docstring) is "never
            # allowed to block download_all() or narrow a window
            # incorrectly" -- true for the per-file convert() failures
            # above, but the final sorted(times) call (and anything else
            # in this method) was previously NOT protected by any
            # try/except. Any unexpected failure here must leave
            # self._sar_scene_times as None (today's documented "scene
            # times unavailable" fallback), not propagate and kill the
            # whole run.
            logger.warning(
                "_compute_sar_scene_times failed unexpectedly -- falling "
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
            report = predict_collocation(self.recipe.config, sar_footprints)
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

    def _download_scatterometer(self, source) -> bool:
        from ..downloaders.scatterometer_downloader import ScatterometerDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        # Fixed literal, not source.source_type: this handler is only ever
        # dispatched for source_type "scatterometer" (see _dispatch_source),
        # and some existing tests call it directly with source=None.
        windows = self._padded_temporal_bounds("scatterometer")
        out_dir = self.base_dir / "osi_saf_winds"

        return self._run_download(
            "scatterometer", out_dir,
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
        # download_kwargs.hsaf_product) only ever holds a rolling
        # last-60-days window (confirmed by the user directly; the FTP
        # directories' own names, e.g. /h29/h29_cur_mon_nc/, are
        # misleading).
        # Anything before that, down to the EUMDAC cutoff, is a genuine
        # gap -- neither source can serve it (H-SAF's off-line/CDR
        # archive is out of scope, see design doc) -- and is surfaced as
        # a notice, not silently dropped.
        today = datetime.now(timezone.utc).date()
        hsaf_window_start = (today - timedelta(days=60)).isoformat()

        # windows is sorted ascending -- these are the overall requested
        # span's start/end dates, used for the same branch-level
        # attempted/coverage-cutoff logic as before Task 1. The per-window
        # date checks below additionally skip any individual window that
        # falls entirely outside a branch's own coverage.
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

        # A branch that wasn't even attempted (out of its date range) is
        # not a failure -- only treat this as a hard failure if every
        # branch that WAS attempted failed. If exactly one attempted
        # branch failed while another attempted branch succeeded (e.g. a
        # transient FTP hiccup on one side of an overlap-range request
        # spanning both the EUMDAC and H-SAF eras), that's a partial
        # failure: the run still has real, usable data from the
        # succeeding branch, so it must not be reported as a hard
        # "download failed" (which would wrongly red-flag the PDF report
        # cover page and force an unnecessary full re-download next run
        # via _already_succeeded). Surface it as a notice instead.
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
            # Unlike the generic "0 products found" notice below, this one
            # is a structural fact about the requested range -- neither
            # downloader branch above is even entered for a genuine gap
            # range -- so it fires regardless of dry_run, not just on real
            # runs that actually searched and came up empty.
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

        # Declared outside the per-window try (see _download_ascat_ssm's
        # own "files" accumulator for the same pattern this mirrors) so a
        # later window's exception can't discard an earlier window's
        # already-collected results.
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
            # Preserve whatever earlier windows already produced, but
            # still record this run as a FAILURE (not a partial success)
            # so cli.py's _is_already_downloaded (which gates purely on
            # metadata["errors"] being empty) retries the failed window's
            # data next run instead of silently treating this as clean.
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
            # A fresh reconnect can fail on a purely transient blip (see
            # _connect_with_retry) even though real AMSR2 files from an
            # earlier successful run already sit in out_dir -- the
            # pipeline still has good data to use, so this must not be
            # surfaced as a failure in the report. Confirmed against a
            # real run whose report showed this failure despite 4 real
            # .h5 files already present in that exact run's amsr_ssm/.
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
        # ERA5Downloader now does its OWN bracket-margin widening (see its
        # time_tolerance_minutes parameter), driven by this same resolved
        # value -- so the literal recipe window is passed here, not a
        # separately (and previously redundantly, since the downloader
        # used to also apply its own fixed _HOUR_BUFFER on top) padded one.
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
        # HycomDownloader now does its OWN bracket-margin widening (see its
        # time_tolerance_minutes parameter), driven by this same resolved
        # value -- see the identical rationale in _download_era5 above.
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
        except Exception as exc:
            msg = f"US HF-radar download failed: {exc}"
            logger.error(msg)
            self.metadata["errors"].append(msg)
            self.metadata["downloads"]["hf_radar_us"] = {"status": "failed", "error": msg}
            return False

        # Declared outside the per-window try (see _download_ascat_ssm's
        # own "files" accumulator for the same pattern this mirrors) so a
        # later window's exception can't discard an earlier window's
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
            # Preserve whatever earlier windows already produced, but
            # still record this run as a FAILURE (not a partial success)
            # so cli.py's _is_already_downloaded (which gates purely on
            # metadata["errors"] being empty) retries the failed window's
            # data next run instead of silently treating this as clean.
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
    # omits "frequencies" from download_kwargs -- cli.py's
    # _build_waves_config/_build_wind_config always set it explicitly, so
    # this only matters for hand-edited recipes. Wind never needs 5 Hz (no
    # WIND_SPEED there, and 5x the point density for no benefit); waves
    # defaults to 1 Hz too -- co-downloading both by default over-samples
    # SAR pixels near the ground track (see docs/design-choices.md §3.3).
    # A recipe can still opt into 5 Hz or both via --altimeter-freq at
    # creation time, or by editing download_kwargs directly.
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
        # source_type -- pass both so the padding lookup finds their
        # (equal, 180min) tolerance regardless of which frequency this
        # recipe's variable actually requests.
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
        # later window's exception can't discard an earlier window's
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
            # Preserve whatever earlier windows already produced, but
            # still record this run as a FAILURE (not a partial success)
            # so cli.py's _is_already_downloaded (which gates purely on
            # metadata["errors"] being empty) retries the failed window's
            # data next run instead of silently treating this as clean.
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

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _save_metadata(self) -> None:
        meta_path = self.base_dir / "download_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)
        logger.info("Metadata saved to %s", meta_path)
