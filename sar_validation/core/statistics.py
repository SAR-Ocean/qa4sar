"""
Validation statistics — step 4b of the validation pipeline.

Computes per-source bias, RMSE, Pearson correlation, and scatter index
from the collocated pairs produced by step 3 (``collocation_results.nc``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import xarray as xr

from ._variable_map import infer_variable_pairs, filter_variable_pairs

logger = logging.getLogger(__name__)

__all__ = [
    "compute_statistics",
    "save_statistics",
    "run_statistics",
]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_statistics(
    collocation_ds: xr.Dataset,
    sar_var: str,
    val_var: str,
    group_by: List[str] = None,
) -> Optional[xr.Dataset]:
    """
    Compute validation statistics for one (sar_var, val_var) pair.

    Metrics computed per group:

    * **N** — number of valid (non-NaN) collocated pairs
    * **bias** — mean(sar − val)
    * **std** — standard deviation of (sar − val)
    * **rmse** — root-mean-square error
    * **correlation** — Pearson r
    * **scatter_index** — RMSE / |mean(val)|  (dimensionless, NaN if mean_val ≈ 0)

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

    records = []
    source_labels = []

    for label, grp in groups:
        sar_vals = grp[sar_col].values.astype(float)
        val_vals = grp[val_col].values.astype(float)
        n = len(sar_vals)
        diff = sar_vals - val_vals
        bias = float(np.mean(diff))
        std = float(np.std(diff, ddof=1)) if n > 1 else float("nan")
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mean_val = float(np.mean(val_vals))
        si = rmse / abs(mean_val) if abs(mean_val) > 1e-10 else float("nan")

        if n > 1:
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
) -> dict[str, xr.Dataset]:
    """
    Compute statistics for all variable pairs inferred from *recipe* and save
    results to ``<base_dir>/validation_statistics_<sar_var>_vs_<val_var>.nc``.

    Parameters
    ----------
    collocation_ds : xr.Dataset
        Dataset produced by step 3 (``collocation_results.nc``).
    recipe : Recipe
        Recipe object; its ``config.variable`` field is used to infer the
        (sar_var, val_var) pairs via :func:`~._variable_map.infer_variable_pairs`.
    base_dir : str or Path
        Directory where statistics files will be written.

    Returns
    -------
    dict[str, xr.Dataset]
        Mapping ``"<sar_var>_vs_<val_var>"`` → statistics Dataset for each pair.
    """
    base_dir = Path(base_dir)
    variable = recipe.config.variable

    try:
        pairs = filter_variable_pairs(recipe, collocation_ds)
    except KeyError as exc:
        logger.error("run_statistics: %s", exc)
        return {}

    results = {}
    for sar_var, val_var in pairs:
        logger.info("Computing statistics: %s vs %s …", sar_var, val_var)

        # Choose grouping dimension: per-station (val_id) when multiple unique
        # stations exist, otherwise fall back to per-source (val_source).
        group_by = ["val_source"]
        if "val_id" in collocation_ds.coords:
            unique_ids = [
                v for v in np.unique(collocation_ds["val_id"].values)
                if str(v) not in ("unknown", "nan", "")
            ]
            if len(unique_ids) > 1:
                group_by = ["val_id"]
                logger.info(
                    "Grouping statistics by val_id (%d unique stations)", len(unique_ids)
                )

        stats_ds = compute_statistics(collocation_ds, sar_var, val_var,
                                      group_by=group_by)
        if stats_ds is None:
            continue
        key = f"{sar_var}_vs_{val_var}"
        out_path = base_dir / f"validation_statistics_{key}.nc"
        save_statistics(stats_ds, out_path)
        results[key] = stats_ds

    if not results:
        logger.warning("run_statistics: no statistics produced (check variable names).")
    return results
