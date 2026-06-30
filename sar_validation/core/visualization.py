"""
Visualization — step 5 of the validation pipeline.

Four public plot functions:

* :func:`plot_scatter`      — SAR vs. validation variable scatter plot
* :func:`plot_geographic`   — SAR field + collocated points, one subplot per SAR scene
* :func:`plot_statistics`   — bar chart of bias / RMSE / correlation per source
* :func:`plot_residuals`    — histogram / KDE of (SAR − validation) residuals

Plus a convenience wrapper:

* :func:`validation_report` — runs all four, infers variable pairs from the recipe,
  and saves PNG files to ``<out_dir>/plots/``

All functions accept an ``interactive=False`` keyword argument.  When
``interactive=True`` the function returns an hvplot / plotly / folium object
instead of a matplotlib Figure.  If the required optional library is not
installed a :class:`ImportError` is raised with a friendly installation hint.
"""

from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "plot_scatter",
    "plot_geographic",
    "plot_statistics",
    "plot_residuals",
    "validation_report",
]

# Colour palette used for validation sources (cycles if more sources than colours)
_SOURCE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _source_color_map(sources: List[str]) -> Dict[str, str]:
    return {s: _SOURCE_COLORS[i % len(_SOURCE_COLORS)] for i, s in enumerate(sorted(set(sources)))}


def _require(package: str, extra: str = "plot") -> None:
    """Raise a friendly ImportError if *package* is not installed."""
    try:
        __import__(package)
    except ImportError:
        raise ImportError(
            f"Package '{package}' is required for interactive plots. "
            f"Install it with:  pip install '{package}'  or  "
            f"pip install 'sar-l2-validation-toolbox[{extra}]'"
        ) from None


def _filter_by_scene(collocation_ds, scene_name: str):
    """Return rows where sar_scene_name matches *scene_name*."""
    import xarray as xr  # noqa: PLC0415

    if "sar_scene_name" not in collocation_ds:
        return collocation_ds   # old dataset without scene name — return all
    mask = collocation_ds["sar_scene_name"] == scene_name
    return collocation_ds.isel(collocation=mask)


def _sar_field(scene_ds, sar_var: str) -> Optional[np.ndarray]:
    """
    Return the (y, x) array for *sar_var* in *scene_ds*, or None if absent.
    The prefix ``sar_`` is stripped when looking up in the SAR scene dataset.
    """
    bare = sar_var.lstrip("owi")   # tolerate both "owiWindSpeed" and "WindSpeed"
    for candidate in (sar_var, f"owi{bare}"):
        if candidate in scene_ds.data_vars:
            return scene_ds[candidate].values
    return None


# ---------------------------------------------------------------------------
# 1. Scatter plot
# ---------------------------------------------------------------------------

def plot_scatter(
    collocation_ds,
    sar_var: str,
    val_var: str,
    *,
    by_source: bool = True,
    interactive: bool = False,
    ax=None,
):
    """
    Scatter plot of SAR vs. validation variable.

    Parameters
    ----------
    collocation_ds : xr.Dataset
        Step-3 collocations (``collocation_results.nc``).
    sar_var : str
        SAR variable name *without* ``sar_`` prefix (e.g. ``"owiWindSpeed"``).
    val_var : str
        Validation variable name *without* ``val_`` prefix (e.g. ``"WSPD"``).
    by_source : bool
        Colour points by ``val_source``.
    interactive : bool
        Return a plotly Figure instead of matplotlib.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into (static only).  A new figure is created if None.

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    sar_col = f"sar_{sar_var}"
    val_col = f"val_{val_var}"

    missing = [c for c in (sar_col, val_col) if c not in collocation_ds]
    if missing:
        warnings.warn(f"No valid data for {sar_col} vs {val_col}.")
        return None

    df = collocation_ds[[sar_col, val_col, "val_source"]].to_dataframe().dropna(
        subset=[sar_col, val_col]
    )

    if df.empty:
        warnings.warn(f"No valid data for {sar_col} vs {val_col}.")
        return None

    if interactive:
        _require("plotly")
        import plotly.express as px  # noqa: PLC0415

        fig = px.scatter(
            df, x=val_col, y=sar_col,
            color="val_source" if by_source else None,
            labels={val_col: val_var, sar_col: sar_var, "val_source": "Source"},
            title=f"{sar_var} vs {val_var}",
            opacity=0.7,
        )
        all_vals = np.concatenate([df[val_col].values, df[sar_col].values])
        vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        fig.add_scatter(x=[vmin, vmax], y=[vmin, vmax],
                        mode="lines", line=dict(color="black", dash="dash"),
                        name="1:1", showlegend=True)
        return fig

    import matplotlib.pyplot as plt  # noqa: PLC0415

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    sources = df["val_source"].unique().tolist()
    cmap = _source_color_map(sources)

    for src in sorted(sources):
        sub = df[df["val_source"] == src]
        label = src if by_source else None
        ax.scatter(sub[val_col], sub[sar_col], s=18, alpha=0.6,
                   color=cmap[src], label=label, rasterized=True)

    all_vals = np.concatenate([df[val_col].values, df[sar_col].values])
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1, label="1:1")

    # Annotate with N, bias, RMSE
    diff = df[sar_col].values - df[val_col].values
    n = len(diff)
    bias = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    corr = float(np.corrcoef(df[val_col].values, df[sar_col].values)[0, 1]) if n > 1 else float("nan")
    annotation = f"N={n}\nBias={bias:.3g}\nRMSE={rmse:.3g}\nr={corr:.3f}"
    ax.text(0.04, 0.96, annotation, transform=ax.transAxes,
            va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    ax.set_xlabel(val_var)
    ax.set_ylabel(sar_var)
    ax.set_title(f"{sar_var} vs {val_var}")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal", "box")
    ax.grid(True, linewidth=0.4)
    if by_source:
        ax.legend(fontsize=7, framealpha=0.7)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Geographic plot (one subplot per SAR scene)
# ---------------------------------------------------------------------------

def plot_geographic(
    datatree,
    collocation_ds,
    sar_var: str,
    *,
    ncols: int = 2,
    cmap: str = "viridis",
    interactive: bool = False,
):
    """
    Geographic overview: SAR field as background + collocated points overlaid.

    One subplot is produced per SAR scene in *datatree*.

    Parameters
    ----------
    datatree : xr.DataTree
        Step-2 DataTree (``datatree.nc``).
    collocation_ds : xr.Dataset
        Step-3 collocations (``collocation_results.nc``).
    sar_var : str
        SAR variable name *without* ``sar_`` prefix (e.g. ``"owiWindSpeed"``).
    ncols : int
        Number of subplot columns.
    cmap : str
        Matplotlib colourmap for the SAR field.
    interactive : bool
        Return a folium Map instead of matplotlib.

    Returns
    -------
    matplotlib.figure.Figure or folium.Map
    """
    sar_node = datatree.get("sar")
    if sar_node is None:
        raise ValueError("DataTree has no '/sar' group.")

    scene_names = list(sar_node.children.keys())
    if not scene_names:
        raise ValueError("No SAR scenes found in DataTree.")

    sar_col = f"sar_{sar_var}"
    val_sources = (
        collocation_ds["val_source"].values.tolist()
        if "val_source" in collocation_ds
        else []
    )
    source_cmap = _source_color_map(val_sources) if val_sources else {}

    if interactive:
        _require("folium")
        import folium  # noqa: PLC0415

        m = folium.Map(tiles="CartoDB positron")
        bounds_list = []

        for scene_name in scene_names:
            scene_ds = sar_node[scene_name].to_dataset()
            if "lon" not in scene_ds.coords or "lat" not in scene_ds.coords:
                continue

            lon2d = scene_ds["lon"].values
            lat2d = scene_ds["lat"].values
            bounds_list.append([
                [float(lat2d.min()), float(lon2d.min())],
                [float(lat2d.max()), float(lon2d.max())],
            ])

            group = folium.FeatureGroup(name=scene_name)
            sub_coll = _filter_by_scene(collocation_ds, scene_name)

            if "collocation" in sub_coll.dims and sub_coll.sizes["collocation"] > 0:
                df_pts = sub_coll[["sar_lon", "sar_lat", "val_source"]].to_dataframe()
                for _, row in df_pts.iterrows():
                    color = source_cmap.get(str(row.get("val_source", "")), "#1f77b4")
                    folium.CircleMarker(
                        location=[float(row["sar_lat"]), float(row["sar_lon"])],
                        radius=4,
                        color=color,
                        fill=True,
                        fill_opacity=0.8,
                        tooltip=str(row.get("val_source", "")),
                    ).add_to(group)
            group.add_to(m)

        if bounds_list:
            all_lats = [b[0][0] for b in bounds_list] + [b[1][0] for b in bounds_list]
            all_lons = [b[0][1] for b in bounds_list] + [b[1][1] for b in bounds_list]
            m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]])

        folium.LayerControl().add_to(m)
        return m

    # --- Static matplotlib + cartopy ---
    try:
        import cartopy.crs as ccrs  # noqa: PLC0415
        import cartopy.feature as cfeature  # noqa: PLC0415
        HAS_CARTOPY = True
    except ImportError:
        HAS_CARTOPY = False
        warnings.warn(
            "cartopy is not installed — falling back to plain matplotlib axes.",
            UserWarning, stacklevel=2,
        )

    import matplotlib.pyplot as plt  # noqa: PLC0415
    import matplotlib.colors as mcolors  # noqa: PLC0415
    import matplotlib.cm as mcm  # noqa: PLC0415
    from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: PLC0415

    nrows = math.ceil(len(scene_names) / ncols)
    subplot_kw = {"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7 * ncols, 5 * nrows),
        subplot_kw=subplot_kw,
        squeeze=False,
    )

    # Collect global vmin/vmax across all scenes for a shared colourbar
    all_field_vals = []
    for scene_name in scene_names:
        scene_ds = sar_node[scene_name].to_dataset()
        arr = _sar_field(scene_ds, sar_var)
        if arr is not None:
            all_field_vals.append(arr[np.isfinite(arr)])

    if all_field_vals:
        all_vals_flat = np.concatenate(all_field_vals)
        vmin = float(np.nanpercentile(all_vals_flat, 2))
        vmax = float(np.nanpercentile(all_vals_flat, 98))
    else:
        vmin, vmax = 0.0, 1.0

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for idx, scene_name in enumerate(scene_names):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        scene_ds = sar_node[scene_name].to_dataset()
        if "lon" not in scene_ds.coords or "lat" not in scene_ds.coords:
            ax.set_visible(False)
            continue

        lon2d = scene_ds["lon"].values
        lat2d = scene_ds["lat"].values
        arr   = _sar_field(scene_ds, sar_var)

        if HAS_CARTOPY:
            ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=1)
            ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
            transform = ccrs.PlateCarree()
        else:
            transform = None

        if arr is not None:
            kw = {"transform": transform} if transform else {}
            ax.pcolormesh(lon2d, lat2d, arr, cmap=cmap, norm=norm,
                          shading="auto", zorder=2, **kw)

        # Overlay collocated points for this scene
        sub_coll = _filter_by_scene(collocation_ds, scene_name)
        if "collocation" in sub_coll.dims and sub_coll.sizes["collocation"] > 0:
            df_pts = sub_coll[["sar_lon", "sar_lat", "val_source"]].to_dataframe()
            for src, grp in df_pts.groupby("val_source"):
                color = source_cmap.get(str(src), "#ff0000")
                kw = {"transform": transform, "zorder": 5} if transform else {"zorder": 5}
                ax.scatter(grp["sar_lon"], grp["sar_lat"],
                           s=30, c=color, edgecolors="white", linewidths=0.5,
                           label=str(src), **kw)
            ax.legend(fontsize=6, loc="lower left", framealpha=0.7)

        ax.set_title(scene_name, fontsize=8)

    # Hide unused axes
    for idx in range(len(scene_names), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # Shared colourbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    fig.colorbar(sm, cax=cbar_ax, label=sar_var)

    fig.suptitle(f"SAR {sar_var} with collocated observations", fontsize=11, y=1.01)
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    return fig


# ---------------------------------------------------------------------------
# 3. Statistics bar chart
# ---------------------------------------------------------------------------

def plot_statistics(
    stats_ds,
    metrics: List[str] = None,
    *,
    interactive: bool = False,
):
    """
    Grouped bar chart of validation statistics per source.

    Parameters
    ----------
    stats_ds : xr.Dataset
        Output of :func:`~.statistics.compute_statistics`.
    metrics : list[str], optional
        Which metrics to plot.  Defaults to ``["bias", "rmse", "correlation"]``.
    interactive : bool
        Return a plotly Figure instead of matplotlib.

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    if metrics is None:
        metrics = ["bias", "rmse", "correlation"]

    available = [m for m in metrics if m in stats_ds]
    if not available:
        warnings.warn(f"None of {metrics} found in stats_ds. Available: {list(stats_ds.data_vars)}")
        return None

    sources = stats_ds["source"].values.tolist()
    sar_var = stats_ds.attrs.get("sar_var", "")
    val_var = stats_ds.attrs.get("val_var", "")
    title = f"Validation statistics: {sar_var} vs {val_var}"

    if interactive:
        _require("plotly")
        import plotly.graph_objects as go  # noqa: PLC0415
        from plotly.subplots import make_subplots  # noqa: PLC0415

        fig = make_subplots(rows=1, cols=len(available),
                            subplot_titles=available)
        for i, metric in enumerate(available, start=1):
            vals = stats_ds[metric].values.tolist()
            fig.add_bar(x=sources, y=vals, name=metric,
                        showlegend=False, row=1, col=i)
        fig.update_layout(title=title)
        return fig

    import matplotlib.pyplot as plt  # noqa: PLC0415

    ncols = len(available)
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4), squeeze=False)

    for i, metric in enumerate(available):
        ax = axes[0][i]
        vals = stats_ds[metric].values.astype(float)
        colors = [_SOURCE_COLORS[j % len(_SOURCE_COLORS)] for j in range(len(sources))]
        ax.bar(sources, vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Source")
        ax.set_xticklabels(sources, rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", linewidth=0.4)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Residuals histogram
# ---------------------------------------------------------------------------

def plot_residuals(
    collocation_ds,
    sar_var: str,
    val_var: str,
    *,
    by_source: bool = True,
    interactive: bool = False,
    ax=None,
):
    """
    Histogram of (SAR − validation) residuals.

    Parameters
    ----------
    collocation_ds : xr.Dataset
        Step-3 collocations.
    sar_var : str
        SAR variable name without ``sar_`` prefix.
    val_var : str
        Validation variable name without ``val_`` prefix.
    by_source : bool
        Overlay one histogram per ``val_source``.
    interactive : bool
        Return a plotly Figure instead of matplotlib.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into (static only).

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    sar_col = f"sar_{sar_var}"
    val_col = f"val_{val_var}"

    missing = [c for c in (sar_col, val_col) if c not in collocation_ds]
    if missing:
        warnings.warn(f"No valid data for {sar_col} vs {val_col}.")
        return None

    df = collocation_ds[[sar_col, val_col, "val_source"]].to_dataframe().dropna(
        subset=[sar_col, val_col]
    )
    if df.empty:
        warnings.warn(f"No valid data for {sar_col} vs {val_col}.")
        return None

    df["residual"] = df[sar_col] - df[val_col]
    title = f"Residuals: {sar_var} − {val_var}"

    if interactive:
        _require("plotly")
        import plotly.express as px  # noqa: PLC0415

        fig = px.histogram(
            df, x="residual",
            color="val_source" if by_source else None,
            barmode="overlay",
            opacity=0.6,
            nbins=40,
            labels={"residual": f"{sar_var} − {val_var}", "val_source": "Source"},
            title=title,
        )
        return fig

    import matplotlib.pyplot as plt  # noqa: PLC0415

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    if by_source:
        sources = sorted(df["val_source"].unique())
        cmap = _source_color_map(sources)
        for src in sources:
            sub = df[df["val_source"] == src]["residual"].dropna()
            ax.hist(sub, bins=30, alpha=0.5, color=cmap[src], label=src, density=True)
        ax.legend(fontsize=7)
    else:
        ax.hist(df["residual"].dropna(), bins=30, density=True, alpha=0.7, color="#1f77b4")

    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel(f"{sar_var} − {val_var}")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Validation report (convenience wrapper)
# ---------------------------------------------------------------------------

def validation_report(
    collocation_ds,
    datatree,
    recipe,
    stats_ds_map: Optional[Dict[str, "xr.Dataset"]] = None,
    out_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, list]:
    """
    Run all four plot functions for every (sar_var, val_var) pair inferred
    from *recipe* and save PNG files to ``<out_dir>/plots/``.

    Parameters
    ----------
    collocation_ds : xr.Dataset
        Step-3 collocations.
    datatree : xr.DataTree
        Step-2 DataTree.
    recipe : Recipe
        Recipe object (provides ``config.variable`` for pair inference).
    stats_ds_map : dict, optional
        Mapping ``"<sar_var>_vs_<val_var>"`` → statistics Dataset (from
        :func:`~.statistics.run_statistics`).  If provided,
        :func:`plot_statistics` is also called for each pair.
    out_dir : str or Path, optional
        Directory where PNG files are saved.  If None the figures are
        returned without saving.

    Returns
    -------
    dict[str, list[matplotlib.figure.Figure]]
        ``"<sar_var>_vs_<val_var>"`` → list of Figure objects for that pair.
    """
    from ._variable_map import infer_variable_pairs  # noqa: PLC0415

    if out_dir is not None:
        out_dir = Path(out_dir) / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)

    variable = recipe.config.variable
    try:
        pairs = infer_variable_pairs(variable)
    except KeyError as exc:
        logger.error("validation_report: %s", exc)
        return {}

    all_figures: Dict[str, list] = {}

    for sar_var, val_var in pairs:
        key = f"{sar_var}_vs_{val_var}"
        figs = []

        logger.info("Generating plots for %s vs %s …", sar_var, val_var)

        # Scatter
        fig_scatter = plot_scatter(collocation_ds, sar_var, val_var)
        if fig_scatter is not None:
            figs.append(fig_scatter)
            if out_dir:
                fig_scatter.savefig(out_dir / f"{key}_scatter.png", dpi=150, bbox_inches="tight")

        # Geographic (one subplot per scene)
        try:
            fig_geo = plot_geographic(datatree, collocation_ds, sar_var)
            if fig_geo is not None:
                figs.append(fig_geo)
                if out_dir:
                    fig_geo.savefig(out_dir / f"{key}_geographic.png", dpi=150, bbox_inches="tight")
        except Exception as exc:
            logger.warning("plot_geographic failed for %s: %s", sar_var, exc)

        # Statistics
        if stats_ds_map and key in stats_ds_map:
            fig_stats = plot_statistics(stats_ds_map[key])
            if fig_stats is not None:
                figs.append(fig_stats)
                if out_dir:
                    fig_stats.savefig(out_dir / f"{key}_statistics.png", dpi=150, bbox_inches="tight")

        # Residuals
        fig_res = plot_residuals(collocation_ds, sar_var, val_var)
        if fig_res is not None:
            figs.append(fig_res)
            if out_dir:
                fig_res.savefig(out_dir / f"{key}_residuals.png", dpi=150, bbox_inches="tight")

        all_figures[key] = figs

    if out_dir:
        logger.info("Plots saved to %s", out_dir)

    return all_figures
