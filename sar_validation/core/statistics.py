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

from ._variable_map import CIRCULAR_VAL_VARS, infer_variable_pairs, filter_variable_pairs

logger = logging.getLogger(__name__)

__all__ = [
    "compute_statistics",
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
    group_by: List[str] = None,
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
            diff = ((sar_vals - val_vals + 180.0) % 360.0) - 180.0
            bias = ((_circular_mean_deg(diff) + 180.0) % 360.0) - 180.0
            if n > 1:
                diff_rad = np.radians(diff)
                resultant_length = np.hypot(np.mean(np.cos(diff_rad)), np.mean(np.sin(diff_rad)))
                std = float(np.degrees(np.sqrt(-2.0 * np.log(resultant_length)))) if resultant_length > 0 else float("nan")
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

        # Group by platform type (val_source, e.g. "mooring", "buoy",
        # "drifter", "scatterometer") rather than per-station (val_id), so
        # categories stay coarse and every platform type — including
        # scatterometer — gets its own row.
        stats_ds = compute_statistics(collocation_ds, sar_var, val_var,
                                      group_by=["val_source"])
        if stats_ds is None:
            continue
        key = f"{sar_var}_vs_{val_var}"
        out_path = base_dir / f"validation_statistics_{key}.nc"
        save_statistics(stats_ds, out_path)
        results[key] = stats_ds

    if not results:
        logger.warning("run_statistics: no statistics produced (check variable names).")
    return results
