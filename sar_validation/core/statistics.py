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

__all__ = [
    "compute_statistics",
    "compute_statistics_soil_moisture",
    "add_rescaled_sar_column",
    "fit_sar_to_val_transform",
    "save_statistics",
    "run_statistics",
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
    * **correlation** — Pearson r (or Jammalamadaka–Sarma circular correlation if val_var is circular)
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

    missing = [c for c in (sar_col, val_col) if c not in collocation_ds]
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
    if len(group_by) == 1:
        groups = df.groupby(group_by[0])
    else:
        df["_group"] = df[group_by].astype(str).agg(" | ".join, axis=1)
        groups = df.groupby("_group")

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
            corr = _circular_corrcoef_deg(sar_vals, val_vals) if n > 1 else float("nan")
        else:
            diff = sar_vals - val_vals
            bias = float(np.mean(diff))
            std = float(np.std(diff, ddof=1)) if n > 1 else float("nan")
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            mean_val = float(np.mean(val_vals))
            si = rmse / abs(mean_val) if abs(mean_val) > 1e-10 else float("nan")

            if n > 1 and np.std(sar_vals) > 0 and np.std(val_vals) > 0:
                corr_mat = np.corrcoef(sar_vals, val_vals)
                corr = float(corr_mat[0, 1])
            else:
                corr = float("nan")

        records.append({
            "N":             n,
            "bias":          bias,
            "std":           std,
            "rmse":          rmse,
            "correlation":   corr,
            "scatter_index": si,
        })
        source_labels.append(str(label))

    metrics = list(records[0].keys())
    stats_ds = xr.Dataset(
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
    return stats_ds


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
    from pytesmo.metrics import ubrmsd

    sar_rescaled = _cdf_match_sar_series(sar_vals, val_vals)
    if sar_rescaled is None:
        return None

    diff = sar_rescaled - val_vals
    n = len(sar_rescaled)
    bias = float(np.mean(diff))
    std = float(np.std(diff, ddof=1)) if n > 1 else float("nan")
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mean_val = float(np.mean(val_vals))
    si = rmse / abs(mean_val) if abs(mean_val) > 1e-10 else float("nan")

    if n > 1 and np.std(sar_rescaled) > 0 and np.std(val_vals) > 0:
        corr = float(np.corrcoef(sar_rescaled, val_vals)[0, 1])
    else:
        corr = float("nan")

    ubrmsd_val = float(ubrmsd(sar_rescaled, val_vals)) if n > 1 else float("nan")

    return {
        "N": n, "bias": bias, "std": std, "rmse": rmse,
        "correlation": corr, "scatter_index": si, "ubrmsd": ubrmsd_val,
    }


def compute_statistics_soil_moisture(
    collocation_ds: xr.Dataset,
    sar_var: str,
    val_var: str,
    group_by: Optional[List[str]] = None,
) -> Optional[xr.Dataset]:
    """
    Soil-moisture variant of :func:`compute_statistics`.

    Before computing bias/RMSE/correlation, the SAR series is CDF-matched
    onto the ISMN series' domain via ``pytesmo.scaling.scale`` — the
    satellite retrieval is rescaled to match the in-situ reference's
    dynamic range, not the reverse (matching standard soil-moisture
    validation practice, e.g. ESA CCI SM). Metrics are then computed on the
    rescaled pair, plus a new ``ubrmsd`` field via
    ``pytesmo.metrics.ubrmsd``.

    Same signature/return shape as :func:`compute_statistics` (with the
    added ``ubrmsd`` data variable), engaged only when
    ``recipe.config.variable == "soil_moisture"`` (see :func:`run_statistics`).
    """
    if group_by is None:
        group_by = ["val_source"]

    sar_col = f"sar_{sar_var}"
    val_col = f"val_{val_var}"

    missing = [c for c in (sar_col, val_col) if c not in collocation_ds]
    if missing:
        logger.warning(
            "compute_statistics_soil_moisture: variable(s) %s not found in collocation dataset — skipping.",
            missing,
        )
        return None

    df = collocation_ds[[sar_col, val_col, *group_by]].to_dataframe()
    df = df.dropna(subset=[sar_col, val_col])

    if df.empty:
        logger.warning("compute_statistics_soil_moisture: no valid pairs for %s vs %s.", sar_col, val_col)
        return None

    if len(group_by) == 1:
        groups = df.groupby(group_by[0])
    else:
        df["_group"] = df[group_by].astype(str).agg(" | ".join, axis=1)
        groups = df.groupby("_group")

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
        record = _rescale_and_compute_soil_moisture_stats(sar_vals, val_vals)
        if record is None:
            continue
        records.append(record)
        source_labels.append(str(label))

    if not records:
        return None

    metrics = list(records[0].keys())
    stats_ds = xr.Dataset(
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
    return stats_ds


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
    out = collocation_ds.copy(deep=True)

    if sar_col not in out or val_col not in out:
        logger.warning(
            "add_rescaled_sar_column: variable(s) %s not found — returning unchanged.",
            [c for c in (sar_col, val_col) if c not in out],
        )
        return out

    n = out.sizes.get("collocation", out[sar_col].size)
    rescaled = np.full(n, np.nan)

    df = out[[sar_col, val_col, *group_by]].to_dataframe()

    if len(group_by) == 1:
        groups = df.groupby(group_by[0])
    else:
        df["_group"] = df[group_by].astype(str).agg(" | ".join, axis=1)
        groups = df.groupby("_group")

    for label, grp in groups:
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

    df = collocation_ds[[sar_col, val_col]].to_dataframe().dropna()
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

        # Group by platform type (val_source, e.g. "mooring", "buoy",
        # "drifter", "scatterometer") rather than per-station (val_id), so
        # categories stay coarse and every platform type — including
        # scatterometer — gets its own row.
        if recipe.config.variable == "soil_moisture":
            stats_ds = compute_statistics_soil_moisture(collocation_ds, sar_var, val_var,
                                                         group_by=["val_source"])
        else:
            stats_ds = compute_statistics(collocation_ds, sar_var, val_var,
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
