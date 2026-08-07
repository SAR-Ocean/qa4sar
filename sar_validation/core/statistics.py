"""
Validation statistics — step 5a of the validation pipeline.

Computes per-source bias, RMSE, Pearson correlation, and scatter index
from the collocated pairs produced by step 3 (``collocation_results.nc``).
"""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from ._variable_map import CIRCULAR_VAL_VARS, circular_diff_deg, filter_variable_pairs

logger = logging.getLogger(__name__)

#: pytesmo.cdf_matching's own message when its minobs constraint reduces
#: nbins below what _cdf_match_sar_series/fit_sar_to_val_transform
#: requested -- an expected consequence of the small nbins cap both use,
#: not a sign of a problem, so both suppress it rather than let it reach
#: the CLI's console.
_BINS_RESIZED_MESSAGE = "The bins have been resized"

#: Minimum sample size for a reported correlation coefficient to be
#: treated as meaningful rather than a numerically-precise-looking
#: artifact of too few points. Pearson r is mathematically degenerate
#: for N=2 (any two points define a line, so r is always exactly +-1
#: regardless of how well the series actually agree) and empirically
#: still highly unstable through N=4. The Jammalamadaka-Sarma circular
#: correlation used for circular variables (e.g. wind direction) has
#: the identical degeneracy at N=2 and near-identical instability at
#: N=3/4 -- confirmed empirically: independent random samples at
#: N=3/4/5 give |r|>0.8 in ~40%/20%/13% of trials for both correlation
#: types, essentially interchangeably. Below this threshold,
#: correlation is reported as NaN instead of a spurious +-1 or a value
#: with no real statistical meaning; every other metric (bias, std,
#: rmse, scatter_index) is still computed and reported normally.
MIN_N_FOR_CORRELATION = 5

__all__ = [
    "compute_statistics",
    "compute_statistics_soil_moisture",
    "add_rescaled_sar_column",
    "fit_sar_to_val_transform",
    "save_statistics",
    "run_statistics",
    "run_statistics_native_units",
    "run_statistics_cds_ssm",
    "MIN_N_FOR_CORRELATION",
]


# ---------------------------------------------------------------------------
# Circular-statistics helpers (for angular variables such as wind direction)
# ---------------------------------------------------------------------------

def _circular_mean_deg(deg: np.ndarray) -> float:
    """Circular mean of angles given in degrees, wrapped to [0, 360)."""
    rad = np.radians(deg)
    mean_rad = np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))
    return float(np.degrees(mean_rad)) % 360.0


def _circular_corrcoef_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> float:
    """
    Jammalamadaka–Sarma circular-circular correlation coefficient.

    Returns NaN if either series has (numerically) zero angular spread
    around its circular mean.
    """
    a = np.radians(a_deg)
    b = np.radians(b_deg)
    a0 = np.radians(_circular_mean_deg(a_deg))
    b0 = np.radians(_circular_mean_deg(b_deg))
    sa = np.sin(a - a0)
    sb = np.sin(b - b0)
    denom = np.sqrt(np.sum(sa ** 2) * np.sum(sb ** 2))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(sa * sb) / denom)


# ---------------------------------------------------------------------------
# Extracted helpers for deduplication
# ---------------------------------------------------------------------------


def _missing_columns(collocation_ds: xr.Dataset, *cols: str) -> List[str]:
    """Names in *cols* absent from *collocation_ds*, in order."""
    return [c for c in cols if c not in collocation_ds]


def _group_by_columns(df, group_by: List[str]):
    """Group *df* by a single column, or by a synthetic ``_group`` column
    joining all of *group_by* with ``" | "`` when there's more than one."""
    if len(group_by) == 1:
        return df.groupby(group_by[0])
    df["_group"] = df[group_by].astype(str).agg(" | ".join, axis=1)
    return df.groupby("_group")


def _core_metrics(sar_vals: np.ndarray, val_vals: np.ndarray) -> dict:
    """bias/std/rmse/correlation/scatter_index for a non-circular
    (sar_vals, val_vals) pair. Shared by :func:`compute_statistics`'s
    non-circular branch and :func:`_soil_moisture_metrics` (which adds
    ubrmsd on top)."""
    diff = sar_vals - val_vals
    n = len(sar_vals)
    bias = float(np.mean(diff))
    std = float(np.std(diff, ddof=1)) if n > 1 else float("nan")
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mean_val = float(np.mean(val_vals))
    si = rmse / abs(mean_val) if abs(mean_val) > 1e-10 else float("nan")
    if n >= MIN_N_FOR_CORRELATION and np.std(sar_vals) > 0 and np.std(val_vals) > 0:
        corr = float(np.corrcoef(sar_vals, val_vals)[0, 1])
    else:
        corr = float("nan")
    return {"N": n, "bias": bias, "std": std, "rmse": rmse, "correlation": corr, "scatter_index": si}


def _assemble_stats_dataset(records, source_labels, sar_var: str, val_var: str, group_by: List[str]) -> xr.Dataset:
    """Build the per-source stats xr.Dataset shared by
    :func:`compute_statistics` and :func:`compute_statistics_soil_moisture`."""
    metrics = list(records[0].keys())
    return xr.Dataset(
        {
            metric: xr.DataArray(
                [r[metric] for r in records],
                dims=["source"],
                attrs={"sar_var": sar_var, "val_var": val_var},
            )
            for metric in metrics
        },
        coords={"source": source_labels},
        attrs={
            "sar_var":  sar_var,
            "val_var":  val_var,
            "group_by": ",".join(group_by),
        },
    )


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_statistics(
    collocation_ds: xr.Dataset,
    sar_var: str,
    val_var: str,
    group_by: Optional[List[str]] = None,
) -> Optional[xr.Dataset]:
    """
    Compute validation statistics for one (sar_var, val_var) pair.

    Metrics computed per group:

    * **N** — number of valid (non-NaN) collocated pairs
    * **bias** — mean(sar − val)
    * **std** — standard deviation of (sar − val)
    * **rmse** — root-mean-square error
    * **correlation** — Pearson r (or Jammalamadaka–Sarma circular correlation
      if val_var is circular); NaN if the group has fewer than
      :data:`MIN_N_FOR_CORRELATION` pairs, since both correlation types are
      mathematically degenerate at N=2 (any two points give exactly ±1,
      however poorly they actually agree) and empirically still highly
      unstable through N=4
    * **scatter_index** — RMSE / |mean(val)|  (dimensionless, NaN if mean_val ≈ 0)

    If ``val_var`` is a circular quantity (currently just ``"WDIR"`` — see
    :data:`~._variable_map.CIRCULAR_VAL_VARS`), the difference used above is
    the wrapped angular difference in ``(-180, 180]`` instead of a plain
    subtraction, ``std`` is the circular standard deviation, and
    ``correlation`` is the Jammalamadaka–Sarma circular-circular correlation
    coefficient rather than Pearson r.

    Parameters
    ----------
    collocation_ds : xr.Dataset
        Dataset produced by ``run_collocation``; must contain variables
        ``sar_<sar_var>`` and ``val_<val_var>`` along the ``collocation`` dim.
    sar_var : str
        Name of the SAR variable *without* the ``sar_`` prefix (e.g.
        ``"owiWindSpeed"``).
    val_var : str
        Name of the validation variable *without* the ``val_`` prefix (e.g.
        ``"WSPD"``).
    group_by : list[str], optional
        Dataset string variables to group by.  Defaults to
        ``["val_source"]``.

    Returns
    -------
    xr.Dataset or None
        Dataset with dimension ``source`` and data variables for each metric.
        Returns None if neither column is present in *collocation_ds*.
    """
    if group_by is None:
        group_by = ["val_source"]

    sar_col = f"sar_{sar_var}"
    val_col = f"val_{val_var}"

    missing = _missing_columns(collocation_ds, sar_col, val_col)
    if missing:
        logger.warning(
            "compute_statistics: variable(s) %s not found in collocation dataset — skipping.",
            missing,
        )
        return None

    # Convert to pandas for groupby convenience
    df = collocation_ds[[sar_col, val_col, *group_by]].to_dataframe()
    df = df.dropna(subset=[sar_col, val_col])

    if df.empty:
        logger.warning("compute_statistics: no valid pairs for %s vs %s.", sar_col, val_col)
        return None

    # Build group label by joining all group_by columns
    groups = _group_by_columns(df, group_by)

    is_circular = val_var in CIRCULAR_VAL_VARS

    records = []
    source_labels = []

    for label, grp in groups:
        sar_vals = grp[sar_col].values.astype(float)
        val_vals = grp[val_col].values.astype(float)
        n = len(sar_vals)

        if is_circular:
            # Wrapped angular difference, e.g. sar=359°, val=1° → diff=-2°
            # (not 358°), since direction is a circular quantity in [0, 360).
            diff = circular_diff_deg(sar_vals, val_vals)
            bias = ((_circular_mean_deg(diff) + 180.0) % 360.0) - 180.0
            if n > 1:
                diff_rad = np.radians(diff)
                resultant_length = np.hypot(np.mean(np.cos(diff_rad)), np.mean(np.sin(diff_rad)))
                if resultant_length > 0:
                    std = float(np.degrees(np.sqrt(-2.0 * np.log(resultant_length))))
                else:
                    std = float("nan")
            else:
                std = float("nan")
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            mean_val = _circular_mean_deg(val_vals)
            si = rmse / abs(mean_val) if abs(mean_val) > 1e-10 else float("nan")
            corr = (
                _circular_corrcoef_deg(sar_vals, val_vals)
                if n >= MIN_N_FOR_CORRELATION else float("nan")
            )
            record = {
                "N":             n,
                "bias":          bias,
                "std":           std,
                "rmse":          rmse,
                "correlation":   corr,
                "scatter_index": si,
            }
        else:
            record = _core_metrics(sar_vals, val_vals)

        records.append(record)
        source_labels.append(str(label))

    return _assemble_stats_dataset(records, source_labels, sar_var, val_var, group_by)


def _cdf_match_sar_series(
    sar_vals: np.ndarray, val_vals: np.ndarray,
) -> Optional[np.ndarray]:
    """
    CDF-match ``sar_vals`` onto ``val_vals``'s domain via
    ``pytesmo.scaling.scale``, returning the rescaled SAR series (same
    length/order as the input) — or None if the match degenerates.

    pytesmo is imported lazily (an optional ``soil_moisture`` extra, see
    pyproject.toml) so the rest of the toolbox never requires it.

    Returns
    -------
    np.ndarray or None
        None if CDF-matching itself degenerates (see ``nbins`` note below)
        rather than silently returning an all-NaN series.
    """
    from pytesmo.scaling import scale

    # pytesmo's own default (nbins=100, minobs=20) was confirmed against a
    # real collocation run to silently produce an all-NaN CDF-matched
    # series once nbins exceeds roughly minobs for a modestly-sized,
    # coarsely-quantized SAR sample (52 unique values across 1517 points):
    # nbins<=20 succeeded cleanly, nbins>=25 degenerated with a "Too few
    # percentiles for chosen k" warning. Since soil-moisture validation
    # runs are typically modest-sized and SAR SSM's own ~0.5%-step
    # quantization caps how much benefit finer percentile binning would
    # give anyway, cap nbins conservatively (still floored so tiny groups
    # get at least 2 bins) rather than inheriting pytesmo's fragile 100.
    nbins = max(2, min(10, len(sar_vals) // 20))

    df_pair = pd.DataFrame({"sar": sar_vals, "val": val_vals})
    with warnings.catch_warnings():
        # A direct, expected consequence of the small nbins cap above
        # combined with pytesmo's own minobs=20 default -- not a sign
        # anything went wrong, so don't let it leak to the CLI's console.
        warnings.filterwarnings(
            "ignore", message=re.escape(_BINS_RESIZED_MESSAGE), category=UserWarning,
        )
        scaled_df = scale(df_pair, method="cdf_match", reference_index=1, nbins=nbins)
    sar_rescaled = scaled_df["sar"].values

    if np.all(np.isnan(sar_rescaled)):
        logger.warning(
            "_cdf_match_sar_series: CDF-matching degenerated to all-NaN "
            "for a group of %d pairs (nbins=%d) — skipping.",
            len(sar_vals), nbins,
        )
        return None

    return sar_rescaled


def _soil_moisture_metrics(sar_rescaled: np.ndarray, val_vals: np.ndarray) -> dict:
    """
    bias/std/rmse/correlation/scatter_index/ubrmsd for an already-matched
    (sar_rescaled, val_vals) pair -- no CDF-matching performed here. Shared
    by :func:`_rescale_and_compute_soil_moisture_stats` (which CDF-matches
    first) and :func:`compute_statistics_soil_moisture`'s converted-group
    branch (already matched by :func:`_harmonize_percent_domain_sources`).
    """
    from pytesmo.metrics import ubrmsd

    record = _core_metrics(sar_rescaled, val_vals)
    record["ubrmsd"] = float(ubrmsd(sar_rescaled, val_vals)) if record["N"] > 1 else float("nan")
    return record


def _rescale_and_compute_soil_moisture_stats(
    sar_vals: np.ndarray, val_vals: np.ndarray,
) -> Optional[dict]:
    """
    Rescale the SAR series onto the ISMN series' domain via CDF-matching,
    then compute the same metrics as :func:`compute_statistics` on the
    rescaled pair, plus ubRMSD.

    Returns
    -------
    dict or None
        None if CDF-matching itself degenerates — see
        :func:`_cdf_match_sar_series`.
    """
    sar_rescaled = _cdf_match_sar_series(sar_vals, val_vals)
    if sar_rescaled is None:
        return None
    return _soil_moisture_metrics(sar_rescaled, val_vals)


def compute_statistics_soil_moisture(
    collocation_ds: xr.Dataset,
    sar_var: str,
    val_var: str,
    group_by: Optional[List[str]] = None,
) -> Optional[xr.Dataset]:
    """
    Soil-moisture variant of :func:`compute_statistics`.

    Before computing bias/RMSE/correlation, :func:`_harmonize_percent_domain_sources`
    converts any val_source sharing SAR's own raw units family (e.g. ASCAT's
    "%") into the reference source's (ISMN's) volumetric domain — for those
    groups, metrics are computed directly via :func:`_soil_moisture_metrics`
    (already matched). Every other group is CDF-matched onto its own domain
    as before, via :func:`_rescale_and_compute_soil_moisture_stats` — the
    satellite retrieval is rescaled to match the in-situ reference's dynamic
    range, not the reverse (matching standard soil-moisture validation
    practice, e.g. ESA CCI SM). A new ``ubrmsd`` field is added via
    ``pytesmo.metrics.ubrmsd`` either way.

    Same signature/return shape as :func:`compute_statistics` (with the
    added ``ubrmsd`` data variable), engaged only when
    ``recipe.config.variable == "soil_moisture"`` (see :func:`run_statistics`).
    """
    if group_by is None:
        group_by = ["val_source"]

    sar_col = f"sar_{sar_var}"
    val_col = f"val_{val_var}"

    missing = _missing_columns(collocation_ds, sar_col, val_col)
    if missing:
        logger.warning(
            "compute_statistics_soil_moisture: variable(s) %s not found in collocation dataset — skipping.",
            missing,
        )
        return None

    harmonized, converted_sources, _dropped = _harmonize_percent_domain_sources(collocation_ds, sar_var, val_var)

    df = harmonized[[sar_col, val_col, *group_by]].to_dataframe()
    df = df.dropna(subset=[sar_col, val_col])

    if df.empty:
        logger.warning("compute_statistics_soil_moisture: no valid pairs for %s vs %s.", sar_col, val_col)
        return None

    groups = _group_by_columns(df, group_by)

    records = []
    source_labels = []
    for label, grp in groups:
        sar_vals = grp[sar_col].values.astype(float)
        val_vals = grp[val_col].values.astype(float)
        if len(sar_vals) < 2:
            logger.warning(
                "compute_statistics_soil_moisture: group '%s' has <2 pairs — cannot CDF-match, skipping.",
                label,
            )
            continue
        record: Optional[dict] = None
        if str(label) in converted_sources:
            record = _soil_moisture_metrics(sar_vals, val_vals)
            records.append(record)
            source_labels.append(str(label))
        else:
            record = _rescale_and_compute_soil_moisture_stats(sar_vals, val_vals)
            if record is None:
                continue
            records.append(record)
            source_labels.append(str(label))

    if not records:
        return None

    return _assemble_stats_dataset(records, source_labels, sar_var, val_var, group_by)


def add_rescaled_sar_column(
    collocation_ds: xr.Dataset,
    sar_var: str,
    val_var: str,
    group_by: Optional[List[str]] = None,
) -> xr.Dataset:
    """
    Return a copy of *collocation_ds* with ``sar_<sar_var>``'s values
    replaced by their per-group CDF-matched equivalent (see
    :func:`compute_statistics_soil_moisture`), for use by plotting
    functions instead of the raw column.

    The raw SAR series and the validation series live in different
    physical domains (e.g. a relative SAR soil-saturation index vs. ISMN's
    absolute volumetric water content) — plotting them directly against
    each other (as the report's scatter/geographic/residual plots do) is
    not meaningful, even though the underlying statistics are already
    computed correctly on the rescaled pair internally. This makes the
    same rescaled values available to the plotting layer, so a report
    generated for ``variable == "soil_moisture"`` compares like with like.

    Before the per-group loop below, :func:`_harmonize_percent_domain_sources`
    converts any val_source sharing SAR's own raw units family (e.g. ASCAT's
    "%") into the reference source's (ISMN's) volumetric domain — those
    groups are then skipped in the loop below (already matched) rather than
    re-matched onto their own now-volumetric val values, which would
    double-apply CDF-matching.

    Points in a group too small to CDF-match, or whose CDF-matching
    degenerates (see :func:`_cdf_match_sar_series`), are left as NaN —
    matching the same per-group exclusion :func:`compute_statistics_soil_moisture`
    already applies when computing metrics.

    The returned column's ``units``/``long_name`` attrs are copied from
    ``val_<val_var>`` (the domain the values now live in), not the
    original ``sar_<sar_var>`` attrs.

    Parameters
    ----------
    collocation_ds : xr.Dataset
        Dataset produced by ``run_collocation``; unchanged (a copy is
        returned) if ``sar_<sar_var>``/``val_<val_var>`` are absent.
    sar_var, val_var : str
        Variable names without the ``sar_``/``val_`` prefix.
    group_by : list[str], optional
        Same grouping as :func:`compute_statistics_soil_moisture`. Defaults
        to ``["val_source"]``.

    Returns
    -------
    xr.Dataset
        Copy of *collocation_ds* with ``sar_<sar_var>`` rescaled in place.
    """
    if group_by is None:
        group_by = ["val_source"]

    sar_col = f"sar_{sar_var}"
    val_col = f"val_{val_var}"

    missing = _missing_columns(collocation_ds, sar_col, val_col)
    if missing:
        logger.warning(
            "add_rescaled_sar_column: variable(s) %s not found — returning unchanged.",
            missing,
        )
        return collocation_ds.copy(deep=True)

    out, converted_sources, _dropped = _harmonize_percent_domain_sources(collocation_ds, sar_var, val_var)

    # Ensure we have a copy if _harmonize_percent_domain_sources returned the original
    if out is collocation_ds:
        out = out.copy(deep=True)

    n = out.sizes.get("collocation", out[sar_col].size)
    rescaled = np.full(n, np.nan)

    df = out[[sar_col, val_col, *group_by]].to_dataframe()

    groups = _group_by_columns(df, group_by)

    for label, grp in groups:
        if str(label) in converted_sources:
            # Already matched onto the reference domain by
            # _harmonize_percent_domain_sources above -- re-matching here
            # would double-apply CDF-matching.
            valid_idx = grp[[sar_col, val_col]].dropna().index
            positions = df.index.get_indexer(valid_idx)
            rescaled[positions] = grp.loc[valid_idx, sar_col].values
            continue
        valid = grp[[sar_col, val_col]].dropna()
        if len(valid) < 2:
            continue
        sar_rescaled = _cdf_match_sar_series(
            valid[sar_col].values.astype(float), valid[val_col].values.astype(float),
        )
        if sar_rescaled is None:
            continue
        positions = df.index.get_indexer(valid.index)
        rescaled[positions] = sar_rescaled

    out[sar_col].values = rescaled
    out[sar_col].attrs = dict(out[val_col].attrs)
    return out


def fit_sar_to_val_transform(
    collocation_ds: xr.Dataset,
    sar_var: str,
    val_var: str,
):
    """
    Fit a CDF-matching transform from every valid collocated
    ``(sar_<sar_var>, val_<val_var>)`` pair (pooled across all groups —
    unlike :func:`add_rescaled_sar_column`'s per-``val_source`` rescaling,
    which is for statistics, not display) and return a callable that maps
    arbitrary SAR values (e.g. a full SAR scene's ``(y, x)`` grid, not just
    the collocated subset) into the validation series' domain.

    Before fitting, :func:`_harmonize_percent_domain_sources` converts any
    val_source sharing SAR's own raw units family (e.g. ASCAT's "%") into
    the reference source's (ISMN's) volumetric domain — but only its
    ``val_<val_var>`` *target* values are taken from that harmonized
    dataset here. Each row's ``sar_<sar_var>`` *input* stays the RAW,
    un-harmonized value from *collocation_ds*, since the fitted transform
    is meant to be applied later to a genuinely raw SAR scene field
    (always in SAR's own native units, e.g. percent) — not to a value
    that has already been run through one percent→volumetric conversion.
    Pairing a harmonized ``sar_col`` for ASCAT rows with a still-raw one
    for ISMN rows would pool two very different numeric scales as input
    to the same fit and skew the percentile binning; pooling raw percent
    and raw volumetric pairs with no harmonization at all (the original
    pre-fix behavior) was similarly nonsensical, just via the opposite
    column.

    Intended for plotting a SAR *field* (not just collocated points) in a
    validation variable's units — e.g. ``plot_geographic``'s background
    layer, which can't use :func:`add_rescaled_sar_column` directly since
    that only rescales points that have a paired validation value, and a
    background raster has values at every pixel, not just collocated ones.

    Returns
    -------
    callable or None
        ``transform(values: np.ndarray) -> np.ndarray`` (same shape as the
        input, NaN where the input was non-finite), or None if there
        aren't enough valid pairs to fit, or the fit itself fails (e.g.
        the same percentile-binning degeneration described in
        :func:`_cdf_match_sar_series` — this uses the lower-level
        ``pytesmo.cdf_matching.CDFMatching`` class directly rather than
        the ``scale()`` convenience wrapper, since ``scale()`` only
        transforms the columns of its own input DataFrame and has no
        "apply this fit to new data" mode).
    """
    from pytesmo.cdf_matching import CDFMatching

    sar_col = f"sar_{sar_var}"
    val_col = f"val_{val_var}"
    if sar_col not in collocation_ds or val_col not in collocation_ds:
        return None

    harmonized, _converted, _dropped = _harmonize_percent_domain_sources(collocation_ds, sar_var, val_var)

    # val_col: harmonized target (unified domain, e.g. ASCAT's proxy
    # volumetric value). sar_col: RAW input straight from collocation_ds,
    # never run through the harmonize step — see the docstring above.
    # collocation_ds and harmonized share the same "collocation"
    # dim/coord and row order (harmonize either returns collocation_ds
    # itself unchanged, or a deep=True copy with values overwritten
    # in-place, never reordered/filtered), so aligning by that shared
    # index is safe.
    df = harmonized[[val_col]].to_dataframe()
    df[sar_col] = collocation_ds[sar_col].to_dataframe()[sar_col]
    df = df.dropna()
    if len(df) < 2:
        return None

    nbins = max(2, min(10, len(df) // 20))
    matcher = CDFMatching(nbins=nbins, minobs=20)
    try:
        with warnings.catch_warnings():
            # See _cdf_match_sar_series -- same expected, benign resizing.
            warnings.filterwarnings(
                "ignore", message=re.escape(_BINS_RESIZED_MESSAGE), category=UserWarning,
            )
            matcher.fit(
                df[sar_col].values.astype(float), df[val_col].values.astype(float),
            )
    except Exception as exc:
        logger.warning("fit_sar_to_val_transform: CDF-matching fit failed: %s", exc)
        return None

    def _transform(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        flat = arr.ravel()
        out = np.full_like(flat, np.nan)
        valid = np.isfinite(flat)
        if valid.any():
            out[valid] = matcher.predict(flat[valid])
        return out.reshape(arr.shape)

    return _transform


def _harmonize_percent_domain_sources(
    collocation_ds: xr.Dataset,
    sar_var: str,
    val_var: str,
    reference_source: str = "ismn",
) -> tuple[xr.Dataset, set[str], set[str]]:
    """
    Convert every val_source sharing SAR's own raw units family (e.g.
    ASCAT's "%", the same domain as Sentinel-1 SSM's own retrieval) into
    reference_source's own domain (ISMN's volumetric fraction), so the
    CDF-matched report section shows every source on one consistent scale.

    Mechanism: SAR's own raw retrieval lives in the same physical domain as
    the sources being converted (see design-choices.md SS8.7), so the CDF-
    matching transform already fit for SAR-vs-reference_source pairs is
    equally valid applied to those sources' own raw values -- this avoids
    needing to collocate e.g. ASCAT against ISMN directly (they are never
    paired with each other, only each with SAR).

    val_source groups already sharing reference_source's own units family
    (e.g. ISMN/SMAP/SMOS, all volumetric) are returned untouched --
    downstream callers (add_rescaled_sar_column, compute_statistics_soil_moisture)
    handle those with their own normal per-group CDF-matching.

    "Needs converting" is detected via the existing _VAL_SOURCE_UNITS_FAMILY
    lookup (the same one run_statistics_native_units already uses), not by
    inspecting a per-row units companion -- so this function is a true no-op
    (returns collocation_ds itself, no copy) for every non-soil-moisture
    recipe and every soil-moisture recipe whose val_source labels aren't in
    that dict.

    Returns
    -------
    (harmonized_ds, converted_sources, dropped_sources)
        harmonized_ds : xr.Dataset
            collocation_ds unchanged (same object) if nothing needed
            converting; otherwise a deep copy with the converted sources'
            sar_<sar_var> and val_<val_var> values replaced, their
            val_units/val_long_name companion rows (if present) updated to
            reference_source's own units/long_name, and val_<val_var>'s
            column-level units attr collapsed from the "mixed — see
            val_units" sentinel to a real units string if every present
            source now shares one family.
        converted_sources : set[str]
            val_source labels that were rewritten. Empty if nothing needed
            converting, OR if conversion was attempted but reference_source
            was absent/too sparse to fit a transform (see below) -- in that
            case the to-be-converted sources' rows are set to NaN instead,
            logged, and excluded from converted_sources so callers don't
            try to skip re-matching a group whose rows are now all-NaN.
        dropped_sources : set[str]
            val_source labels that were identified as needing conversion
            (part of ``to_convert``) but couldn't be -- reference_source
            was absent, too sparse (< 2 valid collocated pairs), or its own
            CDF-matching fit raised. Their sar_<sar_var>/val_<val_var> rows
            are the same NaN'd-out rows excluded from converted_sources
            above; this set lets callers distinguish "this source needed no
            conversion at all" (empty dropped_sources, possibly non-empty
            converted_sources) from "this source WAS supposed to be
            converted but got dropped" (non-empty dropped_sources) --
            both cases otherwise look identical via converted_sources alone
            (empty). Always empty when to_convert is empty (the true no-op
            fast path) or when every source in to_convert was converted
            successfully.
    """
    sar_col = f"sar_{sar_var}"
    val_col = f"val_{val_var}"
    if sar_col not in collocation_ds or val_col not in collocation_ds or "val_source" not in collocation_ds:
        return collocation_ds, set(), set()

    sar_family = _normalize_units_family(collocation_ds[sar_col].attrs.get("units", ""))
    reference_family = _VAL_SOURCE_UNITS_FAMILY.get(reference_source)
    present_sources = set(str(s) for s in collocation_ds["val_source"].values)
    to_convert = {
        s for s in present_sources
        if _VAL_SOURCE_UNITS_FAMILY.get(s) is not None
        and _VAL_SOURCE_UNITS_FAMILY.get(s) == sar_family
        and _VAL_SOURCE_UNITS_FAMILY.get(s) != reference_family
    }
    if not to_convert:
        return collocation_ds, set(), set()

    def _drop_rows(reason: str) -> xr.Dataset:
        logger.warning(
            "_harmonize_percent_domain_sources: %s -- dropping %s from the "
            "CDF-matched section for this run (still available in native "
            "units). %s", reason, sorted(to_convert), reason,
        )
        out = collocation_ds.copy(deep=True)
        drop_mask = out["val_source"].isin(list(to_convert))
        out[sar_col] = out[sar_col].where(~drop_mask)
        out[val_col] = out[val_col].where(~drop_mask)
        return out

    if reference_source not in present_sources:
        return _drop_rows(f"reference source {reference_source!r} is absent"), set(), set(to_convert)

    df = collocation_ds[[sar_col, val_col, "val_source"]].to_dataframe()
    ref_mask = df["val_source"].astype(str) == reference_source
    ref_df = df.loc[ref_mask, [sar_col, val_col]].dropna()
    if len(ref_df) < 2:
        return (
            _drop_rows(f"reference source {reference_source!r} has < 2 valid collocated pairs"),
            set(),
            set(to_convert),
        )

    from pytesmo.cdf_matching import CDFMatching

    nbins = max(2, min(10, len(ref_df) // 20))
    matcher = CDFMatching(nbins=nbins, minobs=20)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=re.escape(_BINS_RESIZED_MESSAGE), category=UserWarning,
        )
        try:
            matcher.fit(
                ref_df[sar_col].values.astype(float), ref_df[val_col].values.astype(float),
            )
        except Exception as exc:
            return (
                _drop_rows(f"reference source {reference_source!r}'s CDF-matching fit failed: {exc}"),
                set(),
                set(to_convert),
            )

        out = collocation_ds.copy(deep=True)
        convert_mask = out["val_source"].isin(list(to_convert)).values
        sar_raw = out[sar_col].values.astype(float)
        val_raw = out[val_col].values.astype(float)
        valid = convert_mask & np.isfinite(sar_raw) & np.isfinite(val_raw)

        sar_out = sar_raw.copy()
        val_out = val_raw.copy()
        if valid.any():
            sar_out[valid] = matcher.predict(sar_raw[valid])
            val_out[valid] = matcher.predict(val_raw[valid])
    sar_out[convert_mask & ~valid] = np.nan
    val_out[convert_mask & ~valid] = np.nan
    out[sar_col].values = sar_out
    out[val_col].values = val_out

    if "val_units" in out:
        ref_row_idx = np.flatnonzero(np.array(out["val_source"].values, dtype=str) == reference_source)
        if len(ref_row_idx):
            new_units = str(out["val_units"].values[ref_row_idx[0]])
            val_units_arr = out["val_units"].values.copy()
            val_units_arr[convert_mask] = new_units
            out["val_units"].values = val_units_arr

            new_long_name = None
            if "val_long_name" in out:
                new_long_name = str(out["val_long_name"].values[ref_row_idx[0]])
                val_long_name_arr = out["val_long_name"].values.copy()
                val_long_name_arr[convert_mask] = new_long_name
                out["val_long_name"].values = val_long_name_arr

            if len(set(out["val_units"].values.tolist())) == 1:
                out[val_col].attrs["units"] = new_units
                out[sar_col].attrs["units"] = new_units
                if new_long_name is not None:
                    out[val_col].attrs["long_name"] = new_long_name

    return out, to_convert, set()


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_statistics(
    stats_ds: xr.Dataset,
    out_path: Union[str, Path],
) -> None:
    """
    Save *stats_ds* to both a NetCDF file and a CSV file.

    Parameters
    ----------
    stats_ds : xr.Dataset
        Output of :func:`compute_statistics`.
    out_path : str or Path
        Path for the ``.nc`` output.  A sibling ``.csv`` is also written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_ds.to_netcdf(out_path)
    logger.info("Statistics saved to %s", out_path)

    csv_path = out_path.with_suffix(".csv")
    stats_ds.to_dataframe().to_csv(csv_path)
    logger.info("Statistics CSV saved to %s", csv_path)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_statistics(
    collocation_ds: xr.Dataset,
    recipe,
    base_dir: Union[str, Path],
    filename_suffix: str = "",
) -> dict[str, xr.Dataset]:
    """
    Compute statistics for all variable pairs inferred from *recipe* and save
    results to ``<base_dir>/validation_statistics_<sar_var>_vs_<val_var><filename_suffix>.nc``.

    Parameters
    ----------
    collocation_ds : xr.Dataset
        Dataset produced by step 3 (``collocation_results.nc``).
    recipe : Recipe
        Recipe object; its ``config.variable`` field determines the
        (sar_var, val_var) pairs via :func:`~._variable_map.filter_variable_pairs`.
    base_dir : str or Path
        Directory where statistics files will be written.
    filename_suffix : str
        Appended to each output filename stem, e.g. ``"_individual"``. Lets
        statistics for two collocation methods coexist without overwriting
        each other.

    Returns
    -------
    dict[str, xr.Dataset]
        Mapping ``"<sar_var>_vs_<val_var>"`` → statistics Dataset for each pair.
    """
    base_dir = Path(base_dir)

    try:
        pairs = filter_variable_pairs(recipe, collocation_ds)
    except KeyError as exc:
        logger.error("run_statistics: %s", exc)
        return {}

    results = {}
    for sar_var, val_var in pairs:
        logger.info("Computing statistics: %s vs %s …", sar_var, val_var)

        # Exclude C3S CDS SSM rows: they are handled by run_statistics_cds_ssm
        # (a separate, non-CDF-matched pass) and must not enter the CDF-matched
        # section where their native units (%/m³m⁻³) would be treated as if
        # they were ISMN-equivalent volumetric fractions.
        if "val_source" in collocation_ds:
            cds_mask = collocation_ds["val_source"] != "cds_ssm"
            ds_for_stats = collocation_ds.isel(collocation=cds_mask.values)
        else:
            ds_for_stats = collocation_ds

        # Group by platform type (val_source, e.g. "mooring", "buoy",
        # "drifter", "scatterometer") rather than per-station (val_id), so
        # categories stay coarse and every platform type — including
        # scatterometer — gets its own row.
        if recipe.config.variable == "soil_moisture":
            stats_ds = compute_statistics_soil_moisture(ds_for_stats, sar_var, val_var,
                                                         group_by=["val_source"])
        else:
            stats_ds = compute_statistics(ds_for_stats, sar_var, val_var,
                                          group_by=["val_source"])
        if stats_ds is None:
            continue
        key = f"{sar_var}_vs_{val_var}"
        out_path = base_dir / f"validation_statistics_{key}{filename_suffix}.nc"
        save_statistics(stats_ds, out_path)
        results[key] = stats_ds

    if not results:
        logger.warning("run_statistics: no statistics produced (check variable names).")
    return results


# ---------------------------------------------------------------------------
# C3S CDS SSM statistics pass
# ---------------------------------------------------------------------------

def run_statistics_cds_ssm(
    collocation_ds: xr.Dataset,
    recipe,
    base_dir: Union[str, Path],
    filename_suffix: str = "",
) -> dict[str, xr.Dataset]:
    """
    Dedicated, non-CDF-matched statistics pass for C3S CDS satellite soil
    moisture (``val_source == "cds_ssm"``).

    Keeps only the rows whose ``val_source`` is ``"cds_ssm"``, then runs
    plain :func:`compute_statistics` (no CDF matching, no unit conversion)
    for each ``(sar_var, val_var)`` pair inferred from *recipe*.  Output is
    written to
    ``<base_dir>/validation_statistics_<sar_var>_vs_<val_var>_cds_ssm<filename_suffix>.nc/.csv``.

    No-op (returns ``{}``) when:
    - ``recipe.config.variable`` is not ``"soil_moisture"``
    - no ``"val_source"`` coordinate exists in *collocation_ds*
    - no rows with ``val_source == "cds_ssm"`` are present
    """
    base_dir = Path(base_dir)
    if recipe.config.variable != "soil_moisture":
        return {}
    if "val_source" not in collocation_ds:
        return {}

    cds_mask = collocation_ds["val_source"] == "cds_ssm"
    if not cds_mask.values.any():
        return {}

    cds_ds = collocation_ds.isel(collocation=cds_mask.values)

    try:
        pairs = filter_variable_pairs(recipe, cds_ds)
    except KeyError as exc:
        logger.error("run_statistics_cds_ssm: %s", exc)
        return {}

    results: dict[str, xr.Dataset] = {}
    for sar_var, val_var in pairs:
        logger.info("Computing C3S CDS SSM statistics: %s vs %s …", sar_var, val_var)
        stats_ds = compute_statistics(cds_ds, sar_var, val_var, group_by=["val_source"])
        if stats_ds is None:
            continue
        key = f"{sar_var}_vs_{val_var}"
        out_path = base_dir / f"validation_statistics_{key}_cds_ssm{filename_suffix}.nc"
        save_statistics(stats_ds, out_path)
        results[key] = stats_ds

    return results
# ---------------------------------------------------------------------------
# Native-units statistics pass (soil moisture only)
# ---------------------------------------------------------------------------

#: Unit family per validation-source platform type, used to gate the
#: native-units statistics pass (run_statistics_native_units). Keyed by
#: val_source, NOT read off the pooled val_<var> column — a single
#: collocation_results.nc column mixes multiple sources' raw values (e.g.
#: ASCAT's "%" alongside ISMN's "m3 m-3"), so a column-level units attr
#: cannot represent per-row units once they're pooled. See
#: design-choices.md §8.7.
_VAL_SOURCE_UNITS_FAMILY: dict[str, str] = {
    "ismn": "volumetric",
    "ascat_ssm": "percent_saturation",
    "amsr_ssm": "volumetric",
    "smap_ssm": "volumetric",
    "smos_ssm": "volumetric",
}


def _normalize_units_family(units: str) -> str:
    """Map a CF units string to a coarse family for the native-units gate."""
    u = (units or "").strip().lower()
    if u in ("%", "percent"):
        return "percent_saturation"
    if u in ("m3 m-3", "m3/m3", "cm3 cm-3", "cm3/cm3", "1"):
        return "volumetric"
    return u or "unknown"


def run_statistics_native_units(
    collocation_ds: xr.Dataset,
    recipe,
    base_dir: Union[str, Path],
    filename_suffix: str = "",
) -> dict[str, xr.Dataset]:
    """
    Second, non-CDF-matched statistics pass for soil_moisture recipes.

    Computes the plain, generic :func:`compute_statistics` (no rescaling)
    for each ``(sar_var, val_var)`` pair, then keeps only the ``source``
    entries (val_source platform types) whose unit family
    (:data:`_VAL_SOURCE_UNITS_FAMILY`) matches the SAR variable's own
    ``units`` attribute family. Written to
    ``validation_statistics_<sar_var>_vs_<val_var>_native_units<filename_suffix>.nc/.csv``,
    only when at least one matching-unit source is present for that pair.

    No-op (returns ``{}``) for any recipe whose ``config.variable`` is not
    ``"soil_moisture"``.
    """
    base_dir = Path(base_dir)
    if recipe.config.variable != "soil_moisture":
        return {}

    try:
        pairs = filter_variable_pairs(recipe, collocation_ds)
    except KeyError as exc:
        logger.error("run_statistics_native_units: %s", exc)
        return {}

    results: dict[str, xr.Dataset] = {}
    for sar_var, val_var in pairs:
        sar_col = f"sar_{sar_var}"
        if sar_col not in collocation_ds:
            continue
        sar_family = _normalize_units_family(collocation_ds[sar_col].attrs.get("units", ""))

        # Exclude cds_ssm from native-units pass: cds_ssm is handled by
        # run_statistics_cds_ssm.  It is not in _VAL_SOURCE_UNITS_FAMILY so
        # the matching-family filter below would drop it anyway, but an
        # explicit exclusion here prevents it appearing in intermediate stats.
        if "val_source" in collocation_ds:
            cds_mask = collocation_ds["val_source"] != "cds_ssm"
            ds_for_native = collocation_ds.isel(collocation=cds_mask.values)
        else:
            ds_for_native = collocation_ds

        stats_ds = compute_statistics(ds_for_native, sar_var, val_var, group_by=["val_source"])
        if stats_ds is None:
            continue

        matching = [
            s for s in stats_ds["source"].values
            if _VAL_SOURCE_UNITS_FAMILY.get(str(s)) == sar_family
        ]
        if not matching:
            continue
        stats_ds = stats_ds.sel(source=matching)

        key = f"{sar_var}_vs_{val_var}"
        out_path = base_dir / f"validation_statistics_{key}_native_units{filename_suffix}.nc"
        save_statistics(stats_ds, out_path)
        results[key] = stats_ds

    return results
