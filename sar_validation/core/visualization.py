"""
Visualization — step 5 of the validation pipeline.

Four public plot functions:

* :func:`plot_scatter`      — SAR vs. validation variable scatter plot
* :func:`plot_geographic`   — SAR field + collocated points, one subplot per SAR scene
* :func:`plot_statistics`   — bar chart of bias / RMSE / correlation per source
* :func:`plot_residuals`    — histogram / KDE of (SAR − validation) residuals

Plus fallback and convenience wrappers:

* :func:`plot_sar_on_no_collocation` — SAR coverage map (fallback when collocation fails)
* :func:`validation_report` — runs all four plots, infers variable pairs from the recipe,
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
    "plot_sar_on_no_collocation",
    "plot_collocation_diagnostics",
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


def _deduplicate_obs(df, sar_col: str, val_col: str):
    """
    Collapse many-SAR-pixel-per-observation rows to one row per observation.

    ``collocate()`` matches **every** SAR pixel within the spatial tolerance to
    each in-situ observation, so one buoy reading produces N rows — one per
    matched pixel — all sharing the same validation-side values but with
    different SAR positions/values.  This helper groups by the observation
    identity (val_source + val_id + val_time + val_lat + val_lon, whichever
    columns are present) and aggregates:

    * SAR column → **mean** of matched-pixel values
    * validation column → **first** (all rows are identical for the same obs)
    """
    id_cols = [c for c in ("val_source", "val_id", "val_time", "val_lat", "val_lon")
               if c in df.columns]
    if not id_cols:
        return df
    agg = {sar_col: "mean", val_col: "first"}
    for c in df.columns:
        if c not in agg and c not in id_cols:
            agg[c] = "first"
    return df.groupby(id_cols, dropna=False, sort=False).agg(agg).reset_index()


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

    extra_cols = [c for c in ("val_id", "val_lat", "val_lon") if c in collocation_ds]
    base_cols = [sar_col, val_col, "val_source"] + extra_cols
    df_raw = collocation_ds[base_cols].to_dataframe()
    if "val_time" in collocation_ds.coords:
        df_raw["val_time"] = collocation_ds["val_time"].values
    df_raw = df_raw.dropna(subset=[sar_col, val_col])

    if df_raw.empty:
        warnings.warn(f"No valid data for {sar_col} vs {val_col}.")
        return None

    # Average many matched SAR pixels → one representative value per observation
    df = _deduplicate_obs(df_raw, sar_col, val_col)

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
    n_raw, n_obs = len(df_raw), len(df)
    if n_raw != n_obs:
        ax.set_title(f"{sar_var} vs {val_var}  (N={n_obs} obs, avg {n_raw // max(n_obs, 1)} px/obs)")
    else:
        ax.set_title(f"{sar_var} vs {val_var}  (N={n_obs})")
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
    val_var: str = None,
    *,
    ncols: int = 2,
    cmap: str = "viridis",
    val_cmap: str = "plasma",
    point_size: int = 40,
    split_by: str = "collocation_type",
    interactive: bool = False,
):
    """
    Geographic overview: SAR field as background + collocated points overlaid.

    One subplot per SAR scene.  Observations are **deduplicated** before
    plotting: when ``collocate()`` has matched multiple SAR pixels to one
    in-situ observation only one dot is shown, placed at the actual observation
    position (``val_lat`` / ``val_lon``).

    Parameters
    ----------
    datatree : xr.DataTree
        Step-2 DataTree (``datatree.nc``).
    collocation_ds : xr.Dataset
        Step-3 collocations (``collocation_results.nc``).
    sar_var : str
        SAR variable name (e.g. ``"owiWindSpeed"``).
    val_var : str, optional
        Validation variable name (e.g. ``"WSPD"``). Points are coloured by
        their measured value with a dedicated colorbar when provided.
    ncols : int
        Number of subplot columns.
    cmap : str
        Matplotlib colourmap for the SAR background field.
    val_cmap : str
        Matplotlib colourmap for the validation scatter points.
    point_size : int
        Scatter marker size in points² (matplotlib ``s`` argument).
    split_by : str or None
        Variable / coordinate to split collocations into separate figures.
        Default ``"collocation_type"`` creates one figure for in-situ
        (``point_vs_layer``) and one for scatterometer (``layer_vs_layer``).
        Pass ``None`` for a single combined figure.
    interactive : bool
        Return a folium Map instead of matplotlib.

    Returns
    -------
    dict[str, matplotlib.figure.Figure] or folium.Map
        When *split_by* is not None: a dict keyed by group value, one Figure
        per group.  When *split_by* is None: a single Figure.
    """
    sar_node = datatree.get("sar")
    if sar_node is None:
        raise ValueError("DataTree has no '/sar' group.")
    scene_names = list(sar_node.children.keys())
    if not scene_names:
        raise ValueError("No SAR scenes found in DataTree.")

    val_col = f"val_{val_var}" if val_var else None
    val_col_present = val_col is not None and val_col in collocation_ds

    val_sources = (
        collocation_ds["val_source"].values.tolist()
        if "val_source" in collocation_ds
        else []
    )
    source_cmap = _source_color_map(val_sources) if val_sources else {}

    # ── Determine group values for splitting ────────────────────────────────
    if split_by:
        if split_by in collocation_ds:
            group_values = sorted(set(str(v) for v in collocation_ds[split_by].values))
        elif split_by in collocation_ds.coords:
            group_values = sorted(set(str(v) for v in collocation_ds.coords[split_by].values))
        else:
            group_values = [None]
    else:
        group_values = [None]

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
            fg = folium.FeatureGroup(name=scene_name)
            sub_coll = _filter_by_scene(collocation_ds, scene_name)
            if "collocation" in sub_coll.dims and sub_coll.sizes["collocation"] > 0:
                cols_needed = ["val_lat", "val_lon", "val_source"]
                if val_col_present:
                    cols_needed.append(val_col)
                df_pts = sub_coll[cols_needed].to_dataframe()
                for _, row in df_pts.iterrows():
                    color = source_cmap.get(str(row.get("val_source", "")), "#1f77b4")
                    tooltip = (
                        f"{val_var}: {row[val_col]:.2f}" if val_col_present
                        else str(row.get("val_source", ""))
                    )
                    folium.CircleMarker(
                        location=[float(row["val_lat"]), float(row["val_lon"])],
                        radius=4, color=color, fill=True, fill_opacity=0.8,
                        tooltip=tooltip,
                    ).add_to(fg)
            fg.add_to(m)
        if bounds_list:
            all_lats = [b[0][0] for b in bounds_list] + [b[1][0] for b in bounds_list]
            all_lons = [b[0][1] for b in bounds_list] + [b[1][1] for b in bounds_list]
            m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]])
        folium.LayerControl().add_to(m)
        return m

    # ── Static matplotlib + cartopy ─────────────────────────────────────────
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
    import matplotlib.lines as mlines  # noqa: PLC0415

    # Global SAR colour limits (same across all figures/groups)
    all_field_vals = []
    for scene_name in scene_names:
        arr = _sar_field(sar_node[scene_name].to_dataset(), sar_var)
        if arr is not None:
            all_field_vals.append(arr[np.isfinite(arr)])
    if all_field_vals:
        flat = np.concatenate(all_field_vals)
        vmin = float(np.nanpercentile(flat, 2))
        vmax = float(np.nanpercentile(flat, 98))
    else:
        vmin, vmax = 0.0, 1.0
    sar_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sar_sm = mcm.ScalarMappable(cmap=cmap, norm=sar_norm)
    sar_sm.set_array([])

    # Global validation colour limits (from all collocations, all groups)
    val_norm = val_sm = None
    if val_col_present:
        finite_v = collocation_ds[val_col].values
        finite_v = finite_v[np.isfinite(finite_v)]
        if len(finite_v) > 0:
            val_norm = mcolors.Normalize(
                vmin=float(np.nanpercentile(finite_v, 2)),
                vmax=float(np.nanpercentile(finite_v, 98)),
            )
            val_sm = mcm.ScalarMappable(cmap=val_cmap, norm=val_norm)
            val_sm.set_array([])

    right_margin = 0.80 if val_sm is not None else 0.88
    subplot_kw = {"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}

    def _build_figure(group_coll_ds, group_label):
        """Build one Figure for a sub-set of collocations."""
        nrows = math.ceil(len(scene_names) / ncols)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(7 * ncols, 5 * nrows),
            subplot_kw=subplot_kw,
            squeeze=False,
        )

        for idx, scene_name in enumerate(scene_names):
            r, c = divmod(idx, ncols)
            ax = axes[r][c]
            scene_ds = sar_node[scene_name].to_dataset()

            if "lon" not in scene_ds.coords or "lat" not in scene_ds.coords:
                ax.set_visible(False)
                continue

            if HAS_CARTOPY:
                ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=1)
                gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
                gl.top_labels = False
                gl.right_labels = False
                transform = ccrs.PlateCarree()
            else:
                transform = None

            arr = _sar_field(scene_ds, sar_var)
            if arr is not None:
                kw = {"transform": transform} if transform else {}
                # Check if data is gridded (2D) or point-based (1D)
                if arr.ndim == 1:
                    # Point data (e.g., WV mode) — use scatter
                    ax.scatter(
                        scene_ds["lon"].values, scene_ds["lat"].values, c=arr,
                        cmap=cmap, norm=sar_norm, s=20, edgecolors="none",
                        zorder=3, **kw,
                    )
                else:
                    # Gridded data (e.g., IW/EW mode) — use pcolormesh
                    ax.pcolormesh(
                        scene_ds["lon"].values, scene_ds["lat"].values, arr,
                        cmap=cmap, norm=sar_norm, shading="auto", zorder=2, **kw,
                    )

            sub_coll = _filter_by_scene(group_coll_ds, scene_name)
            n_pts = sub_coll.sizes.get("collocation", 0)

            if n_pts > 0 and "val_lat" in sub_coll and "val_lon" in sub_coll:
                kw_sc = {"transform": transform, "zorder": 5} if transform else {"zorder": 5}

                # Build a dataframe with observation position + val value
                col_list = ["val_lat", "val_lon"]
                if val_col_present:
                    col_list.append(val_col)
                if "val_source" in sub_coll:
                    col_list.append("val_source")
                df_pts = sub_coll[col_list].to_dataframe()
                if "val_time" in sub_coll.coords:
                    df_pts["val_time"] = sub_coll.coords["val_time"].values
                if "val_id" in sub_coll.coords:
                    df_pts["val_id"] = sub_coll.coords["val_id"].values

                # Deduplicate: one dot per observation at its actual position
                if val_col_present and val_norm is not None:
                    # Use a proxy SAR column so _deduplicate_obs works
                    _proxy = "__sar_proxy__"
                    df_pts[_proxy] = np.nan
                    df_pts = _deduplicate_obs(df_pts, _proxy, val_col)
                    df_pts = df_pts.drop(columns=[_proxy], errors="ignore")
                else:
                    id_cols = [c for c in ("val_source", "val_id", "val_time",
                                           "val_lat", "val_lon") if c in df_pts.columns]
                    if id_cols:
                        df_pts = df_pts.drop_duplicates(subset=id_cols)

                if val_col_present and val_norm is not None:
                    ax.scatter(
                        df_pts["val_lon"], df_pts["val_lat"],
                        c=df_pts[val_col], cmap=val_cmap, norm=val_norm,
                        s=point_size, edgecolors="black", linewidths=0.4,
                        **kw_sc,
                    )
                    if "val_source" in df_pts.columns:
                        present = set(df_pts["val_source"].astype(str))
                        handles = [
                            mlines.Line2D([], [], marker="o", linestyle="None",
                                          markerfacecolor=clr, markeredgecolor="black",
                                          markersize=5, label=s)
                            for s, clr in source_cmap.items() if s in present
                        ]
                        if handles:
                            ax.legend(handles=handles, fontsize=6,
                                      loc="lower left", framealpha=0.7)
                elif "val_source" in df_pts.columns:
                    for src, grp in df_pts.groupby("val_source"):
                        color = source_cmap.get(str(src), "#ff0000")
                        ax.scatter(grp["val_lon"], grp["val_lat"],
                                   s=point_size, c=color,
                                   edgecolors="black", linewidths=0.4,
                                   label=str(src), **kw_sc)
                    ax.legend(fontsize=6, loc="lower left", framealpha=0.7)
                else:
                    ax.scatter(df_pts["val_lon"], df_pts["val_lat"],
                               s=point_size, c="#ff7f0e",
                               edgecolors="black", linewidths=0.4, **kw_sc)

            n_dedup = len(df_pts) if n_pts > 0 else 0
            ax.set_title(
                f"{scene_name.split('/')[-1]}  ({n_dedup} obs)", fontsize=8
            )

        # Hide unused axes
        for idx in range(len(scene_names), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.subplots_adjust(right=right_margin)
        cbar_ax = fig.add_axes([right_margin + 0.01, 0.15, 0.015, 0.70])
        fig.colorbar(sar_sm, cax=cbar_ax, label=f"SAR {sar_var}")
        if val_sm is not None:
            val_cbar_ax = fig.add_axes([right_margin + 0.055, 0.15, 0.015, 0.70])
            fig.colorbar(val_sm, cax=val_cbar_ax, label=f"In-situ {val_var}")

        title = f"SAR {sar_var}"
        if val_var:
            title += f" vs. {val_var}"
        if group_label is not None:
            title += f"  [{group_label}]"
        fig.suptitle(title + " — collocated observations", fontsize=11, y=1.01)
        return fig

    # ── Build figures ────────────────────────────────────────────────────────
    if group_values == [None]:
        return _build_figure(collocation_ds, None)

    figures: Dict[str, object] = {}
    for gv in group_values:
        if split_by in collocation_ds:
            mask = collocation_ds[split_by] == gv
        else:
            mask = collocation_ds.coords[split_by] == gv
        group_ds = collocation_ds.isel(collocation=mask)
        if group_ds.sizes.get("collocation", 0) == 0:
            continue
        figures[gv] = _build_figure(group_ds, gv)
    return figures


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
# 4b. SAR data plot (fallback for no-collocation cases)
# ---------------------------------------------------------------------------

def plot_sar_on_no_collocation(
    datatree,
    sar_var: str,
    recipe_name: str,
    output_dir: Union[str, Path],
    recipe=None,
    has_collocation: bool = False,
) -> Union[Path, None]:
    """
    Plot SAR data geographic coverage with optional validation data overlay.

    Creates map-view figures showing SAR measurements with overlaid validation data:
    - Subplot 1: SAR + in-situ data sources
    - Subplot 2 (if scatterometer data available): SAR + scatterometer data with time colormap

    Parameters
    ----------
    datatree : xr.DataTree
        Step-2 DataTree (``datatree.nc``).
    sar_var : str
        SAR variable name (e.g. ``"owiSignificantWaveHeight"``).
    recipe_name : str
        Name of the recipe (for filename/title), e.g., ``"waves_test"``.
    output_dir : str or Path
        Directory to save the PNG file.
    recipe : Recipe, optional
        Recipe object containing geographic bounds for filtering scatterometer data.
        If provided, enables scatterometer subplot with geographic filtering.
    has_collocation : bool, optional
        If True, names the plot ``sar_collocation_*.png``. If False (default),
        names it ``sar_no_collocation_*.png``. Default is False.

    Returns
    -------
    Path or None
        Path to the saved PNG file, or None if no SAR data could be plotted.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import matplotlib.colors as mcolors  # noqa: PLC0415
    import matplotlib.cm as mcm  # noqa: PLC0415
    from matplotlib.patches import Patch  # noqa: PLC0415

    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    sar_node = datatree.get("sar")
    if sar_node is None:
        logger.warning("plot_sar_on_no_collocation: DataTree has no '/sar' group.")
        return None

    scene_names = list(sar_node.children.keys())
    if not scene_names:
        logger.warning("plot_sar_on_no_collocation: No SAR scenes found in DataTree.")
        return None

    # ── Try to find the requested variable, or a suitable alternative ───
    actual_var = _find_available_sar_variable(sar_node, sar_var)
    if actual_var is None:
        logger.warning(
            "plot_sar_on_no_collocation: Variable '%s' not found and no "
            "suitable alternative available.", sar_var
        )
        return None

    if actual_var != sar_var:
        logger.info(
            "plot_sar_on_no_collocation: Requested variable '%s' not found; "
            "plotting alternative '%s' instead.", sar_var, actual_var
        )

    # ── Determine colour limits from all SAR data ────────────────────────
    all_field_vals = []
    for scene_name in scene_names:
        arr = _sar_field(sar_node[scene_name].to_dataset(), actual_var)
        if arr is not None:
            all_field_vals.append(arr[np.isfinite(arr)])

    if not all_field_vals:
        logger.warning(
            "plot_sar_on_no_collocation: No valid (non-NaN) data found for '%s'.", actual_var
        )
        return None

    flat = np.concatenate(all_field_vals)
    vmin = float(np.nanpercentile(flat, 2))
    vmax = float(np.nanpercentile(flat, 98))
    sar_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sar_sm = mcm.ScalarMappable(cmap="viridis", norm=sar_norm)
    sar_sm.set_array([])

    # ── Extract validation data ─────────────────────────────────────────
    val_data = _extract_validation_data_for_plot(datatree)
    has_validation = bool(val_data)

    # ── Separate in-situ and scatterometer data ────────────────────────
    insitu_data = {}
    scatterometer_data = {}
    if has_validation:
        # Convert to numpy arrays for efficient fancy indexing
        platform_types = np.array(val_data["platform_types"])
        insitu_mask = platform_types == "insitu"
        scatterometer_mask = platform_types == "scatterometer"
        insitu_idx = np.where(insitu_mask)[0]
        scatterometer_idx = np.where(scatterometer_mask)[0]
        
        if len(insitu_idx) > 0:
            insitu_data = {
                "lons": val_data["lons"][insitu_idx],
                "lats": val_data["lats"][insitu_idx],
                "platform_types": platform_types[insitu_idx].tolist(),
            }
        
        if len(scatterometer_idx) > 0:
            scatterometer_data = {
                "lons": val_data["lons"][scatterometer_idx],
                "lats": val_data["lats"][scatterometer_idx],
                "platform_types": platform_types[scatterometer_idx].tolist(),
                "times": val_data["times"][scatterometer_idx],
            }

    # ── Set up cartopy if available ─────────────────────────────────────
    try:
        import cartopy.crs as ccrs  # noqa: PLC0415
        import cartopy.feature as cfeature  # noqa: PLC0415
        HAS_CARTOPY = True
    except ImportError:
        HAS_CARTOPY = False
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "cartopy not installed — using plain matplotlib axes."
            )

    subplot_kw = {"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}

    # ── Determine number of subplots ────────────────────────────────────
    # Each row has: left column = SAR scene + in-situ, right column = scatterometer (row 0 only) or empty
    has_scatterometer = bool(scatterometer_data)
    nrows = len(scene_names)
    ncols = 2
    figsize = (14, 6 * nrows)
    
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        subplot_kw=subplot_kw,
        squeeze=False,
        dpi=100,  # Reduced DPI for faster rendering
    )
    axes_flat = axes.flatten()

    # ── Plot subplot: SAR + in-situ data (left column of each row) ────────
    for idx, scene_name in enumerate(scene_names):
        ax = axes_flat[2 * idx]  # Left column (every other subplot)
        scene_ds = sar_node[scene_name].to_dataset()

        if "lon" not in scene_ds.coords or "lat" not in scene_ds.coords:
            ax.set_visible(False)
            continue

        # ── Add map features ────────────────────────────────────────────
        if HAS_CARTOPY:
            ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0, alpha=0.3)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.3, zorder=1)
            gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.3)
            gl.top_labels = False
            gl.right_labels = False
            transform = ccrs.PlateCarree()
        else:
            transform = None

        # ── Plot SAR data (background) ──────────────────────────────────
        arr = _sar_field(scene_ds, actual_var)
        if arr is not None:
            kw = {"transform": transform} if transform else {}
            
            # Check if this is point-based data (WV mode) or grid data (IW/EW mode)
            is_point_based = len(arr.shape) == 1 or (len(arr.shape) == 2 and arr.shape[0] == 1)
            
            if is_point_based:
                # Point-based data (WV mode) — use scatter plot
                lon_1d = scene_ds["lon"].values.flatten()
                lat_1d = scene_ds["lat"].values.flatten()
                arr_1d = arr.flatten()
                
                # Filter out NaN values
                valid = np.isfinite(arr_1d)
                lon_valid = lon_1d[valid]
                lat_valid = lat_1d[valid]
                arr_valid = arr_1d[valid]
                
                if len(arr_valid) > 0:
                    ax.scatter(
                        lon_valid, lat_valid, c=arr_valid,
                        cmap="viridis", norm=sar_norm, s=20, zorder=2, 
                        rasterized=True, **kw,
                    )
            else:
                # Grid-based data (IW/EW mode) — use pcolormesh with rasterization
                ax.pcolormesh(
                    scene_ds["lon"].values, scene_ds["lat"].values, arr,
                    cmap="viridis", norm=sar_norm, shading="auto", zorder=2, 
                    rasterized=True, **kw,
                )

        # ── Overlay in-situ validation data ──────────────────────────────
        if insitu_data:
            source_color_map = _source_color_map(sorted(set(insitu_data["platform_types"])))
            for pt_type in sorted(set(insitu_data["platform_types"])):
                mask = np.array(insitu_data["platform_types"]) == pt_type
                pt_lons = np.array(insitu_data["lons"])[mask]
                pt_lats = np.array(insitu_data["lats"])[mask]
                kw = {"transform": transform} if transform else {}
                ax.scatter(
                    pt_lons, pt_lats, c=source_color_map[pt_type], s=100,
                    marker='D', edgecolors='black', linewidth=0.5, zorder=3, label=pt_type, **kw,
                )
            ax.legend(loc='best', fontsize=8, framealpha=0.8)

        ax.set_title(f"{scene_name.split('/')[-1]} — {actual_var} + in-situ", fontsize=10)

    # ── Plot subplot: SAR + scatterometer footprint outlines (right column, all rows) ──
    if has_scatterometer and scatterometer_data:
        # Pre-extract scatterometer overpasses once for all rows
        overpasses = None
        if recipe is not None:
            overpasses = _extract_scatterometer_overpasses(datatree, recipe)
        
        # Plot scatterometer on every row's right column
        for idx in range(len(scene_names)):
            ax_scat = axes_flat[2 * idx + 1]  # Right column of each row
            scene_ds = sar_node[scene_names[idx]].to_dataset()

            if "lon" not in scene_ds.coords or "lat" not in scene_ds.coords:
                ax_scat.set_visible(False)
                continue

            # ── Add map features (simplified for speed) ─────────────────
            if HAS_CARTOPY:
                ax_scat.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0, alpha=0.3)
                ax_scat.add_feature(cfeature.COASTLINE, linewidth=0.3, zorder=1)
                gl = ax_scat.gridlines(draw_labels=True, linewidth=0.2, alpha=0.3)
                gl.top_labels = False
                gl.right_labels = False
                transform = ccrs.PlateCarree()
            else:
                transform = None

            # ── Plot SAR data (background, rasterized for speed) ────────
            arr = _sar_field(scene_ds, actual_var)
            if arr is not None:
                kw = {"transform": transform} if transform else {}
                
                is_point_based = len(arr.shape) == 1 or (len(arr.shape) == 2 and arr.shape[0] == 1)
                
                if is_point_based:
                    lon_1d = scene_ds["lon"].values.flatten()
                    lat_1d = scene_ds["lat"].values.flatten()
                    arr_1d = arr.flatten()
                    valid = np.isfinite(arr_1d)
                    lon_valid = lon_1d[valid]
                    lat_valid = lat_1d[valid]
                    arr_valid = arr_1d[valid]
                    if len(arr_valid) > 0:
                        ax_scat.scatter(
                            lon_valid, lat_valid, c=arr_valid,
                            cmap="viridis", norm=sar_norm, s=10, zorder=2, 
                            rasterized=True, **kw,
                        )
                else:
                    ax_scat.pcolormesh(
                        scene_ds["lon"].values, scene_ds["lat"].values, arr,
                        cmap="viridis", norm=sar_norm, shading="auto", zorder=2, 
                        rasterized=True, **kw,
                    )

            # ── Extract and plot scatterometer footprints ──────────────
            if overpasses is not None:
                _plot_scatterometer_footprints(ax_scat, overpasses, fig, transform)
            else:
                ax_scat.text(0.5, 0.5, "No geographic bounds (recipe not provided)",
                            ha='center', va='center', transform=ax_scat.transAxes, fontsize=10)

            ax_scat.set_title(f"{actual_var} + scatterometer footprints (entry/exit times)", fontsize=10)

    # Hide unused subplots (right column for all rows if no scatterometer, plus any trailing subplots)
    num_used = nrows * ncols
    for idx in range(num_used, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    # Right column is not used if has_scatterometer is False
    if not has_scatterometer:
        for row in range(nrows):
            axes_flat[2 * row + 1].set_visible(False)

    # ── Add colourbar and title ─────────────────────────────────────────
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    fig.colorbar(sar_sm, cax=cbar_ax, label=actual_var)

    collocation_status = "collocation" if has_collocation else "no collocation"
    fig.suptitle(
        f"SAR {actual_var} — {collocation_status} (coverage overview)",
        fontsize=12, y=0.98,
    )

    # ── Save to PNG ─────────────────────────────────────────────────────
    collocation_prefix = "collocation" if has_collocation else "no_collocation"
    filename = f"sar_{collocation_prefix}_{actual_var}_{recipe_name}.png"
    filepath = plots_dir / filename
    try:
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        logger.info("Saved SAR coverage plot: %s", filepath)
        plt.close(fig)
        return filepath
    except Exception as exc:
        logger.error("Failed to save SAR coverage plot: %s", exc)
        plt.close(fig)
        return None

# ---------------------------------------------------------------------------
# 4b. Helper for scatterometer mesh visualization
# ---------------------------------------------------------------------------

def _plot_scatterometer_mesh(ax, lons, lats, times, color, label, transform, linewidth=0.8, alpha=0.6, time_tolerance_minutes=5):
    """
    Plot scatterometer data as a mesh (Delaunay triangulation edges) instead of individual points.
    
    Only connects points that are close in both space AND time to avoid connecting different
    orbital overpasses that may be spatially close but temporally distant.
    
    Parameters
    ----------
    ax : matplotlib axis
        Cartopy-enabled axis to plot on.
    lons, lats : np.ndarray
        Flattened arrays of scatterometer point coordinates.
    times : np.ndarray
        Times corresponding to each point (datetime64 or similar).
    color : str
        Color for mesh lines.
    label : str
        Label for the legend.
    transform : ccrs.CRS
        Cartopy coordinate reference system.
    linewidth : float, optional
        Width of mesh lines (default: 0.8).
    alpha : float, optional
        Transparency of mesh lines (default: 0.6).
    time_tolerance_minutes : float, optional
        Maximum time difference (in minutes) to connect two points (default: 5).
    
    Returns
    -------
    bool
        True if mesh was plotted successfully, False otherwise.
    """
    from scipy.spatial import Delaunay  # noqa: PLC0415
    from matplotlib.collections import LineCollection  # noqa: PLC0415
    
    if len(lons) < 4:  # Delaunay needs at least 4 points in 2D
        logger.debug(f"Skipping mesh for {label}: only {len(lons)} points (need ≥4)")
        return False
    
    try:
        # Convert times to numeric upfront (vectorized, more efficient)
        times_numeric = np.full(len(times), np.inf, dtype=np.float64)
        valid_times = []
        for i, t in enumerate(times):
            try:
                if t is not None and t is not np.datetime64('NaT'):
                    times_numeric[i] = np.datetime64(t).astype('datetime64[m]').astype(np.float64)
                    valid_times.append(i)
            except (TypeError, ValueError):
                pass
        
        if len(valid_times) == 0:
            logger.debug(f"No valid times for {label}, skipping temporal filtering")
            times_numeric = None
        
        # Stack lon/lat into (n, 2) array for Delaunay
        points = np.column_stack([lons, lats])
        
        # Create Delaunay triangulation
        tri = Delaunay(points)
        
        # Extract unique edges from simplices, filtering by time tolerance
        edges = set()
        time_tol_mins = time_tolerance_minutes
        
        for simplex in tri.simplices:
            for i in range(3):
                p1, p2 = simplex[i], simplex[(i + 1) % 3]
                
                # Check temporal proximity if times are available
                if times_numeric is not None:
                    t1, t2 = times_numeric[p1], times_numeric[p2]
                    # Skip if either time is NaT/None (represented as inf)
                    if t1 == np.inf or t2 == np.inf:
                        continue
                    time_diff_mins = abs(t1 - t2)
                    if time_diff_mins > time_tol_mins:
                        continue  # Skip this edge, points are too far apart in time
                
                edge = tuple(sorted([p1, p2]))
                edges.add(edge)
        
        if not edges:
            logger.debug(f"No temporally-valid edges for {label}")
            return False
        
        # Build line segments for all edges at once (vectorized)
        segments = np.array([[(lons[p1], lats[p1]), (lons[p2], lats[p2])] for p1, p2 in edges])
        
        if len(segments) == 0:
            return False
        
        # Create LineCollection and add to plot
        line_coll = LineCollection(
            segments,
            colors=color,
            linewidths=linewidth,
            alpha=alpha,
            transform=transform,
            zorder=5,
        )
        ax.add_collection(line_coll)
        
        # Add a dummy line for legend (LineCollection doesn't support labels directly)
        ax.plot([], [], color=color, linewidth=linewidth, alpha=alpha, label=label)
        
        logger.debug(f"Plotted scatterometer mesh: {len(edges)} edges in '{label}'")
        return True
        
    except Exception as e:
        logger.debug(f"Failed to create Delaunay triangulation for '{label}': {e}")
        return False

# ---------------------------------------------------------------------------
# 4c. Collocation diagnostics plot
# ---------------------------------------------------------------------------

def plot_collocation_diagnostics(
    datatree,
    collocation_ds,
    recipe,
    output_dir: Union[str, Path],
) -> Union[Path, None]:
    """
    Plot collocation diagnostics: SAR scene bounds, matched and unmatched validation points.

    Creates a geographic map showing:
    - SAR scene footprints (blue lines for each scene boundary)
    - Matched validation observations (colored dots)
    - Unmatched validation observations (red dots)
    - Statistics in title (total, matched, unmatched counts)

    Parameters
    ----------
    datatree : xr.DataTree
        Step-2 DataTree (``datatree.nc``).
    collocation_ds : xr.Dataset
        Step-3 collocation results (``collocation_results.nc``).
    recipe : Recipe
        Recipe object containing metadata.
    output_dir : str or Path
        Directory to save the PNG file (typically the base_dir).

    Returns
    -------
    Path or None
        Path to the saved PNG file, or None if plot could not be generated.
    """
    import xarray as xr  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.patches import Rectangle, Patch  # noqa: PLC0415
    import matplotlib.lines as mlines  # noqa: PLC0415

    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Set up cartopy if available
    try:
        import cartopy.crs as ccrs  # noqa: PLC0415
        import cartopy.feature as cfeature  # noqa: PLC0415
        HAS_CARTOPY = True
    except ImportError:
        HAS_CARTOPY = False
        logger.debug("cartopy not installed — collocation_diagnostics plot unavailable.")
        return None

    # ── Extract SAR scene bounds ────────────────────────────────────────
    sar_node = datatree.get("sar")
    if sar_node is None or not sar_node.children:
        logger.warning("plot_collocation_diagnostics: No SAR data found in DataTree.")
        return None

    scene_bounds = []
    scene_names = list(sar_node.children.keys())
    
    for scene_name in scene_names:
        scene_ds = sar_node[scene_name].to_dataset()
        if "lon" not in scene_ds.coords or "lat" not in scene_ds.coords:
            continue
        
        lons = scene_ds["lon"].values
        lats = scene_ds["lat"].values
        
        # Handle both 2D grids and 1D point arrays
        if len(lons.shape) > 1:
            lons_flat = lons.flatten()
            lats_flat = lats.flatten()
        else:
            lons_flat = lons
            lats_flat = lats
        
        valid_mask = np.isfinite(lons_flat) & np.isfinite(lats_flat)
        if valid_mask.any():
            lon_min = float(np.nanmin(lons_flat[valid_mask]))
            lon_max = float(np.nanmax(lons_flat[valid_mask]))
            lat_min = float(np.nanmin(lats_flat[valid_mask]))
            lat_max = float(np.nanmax(lats_flat[valid_mask]))
            scene_bounds.append({
                "name": scene_name,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "lat_min": lat_min,
                "lat_max": lat_max,
            })

    if not scene_bounds:
        logger.warning("plot_collocation_diagnostics: Could not extract SAR scene bounds.")
        return None

    # ── Extract all validation data ─────────────────────────────────────
    all_val_data = _extract_validation_data_for_plot(datatree)
    if not all_val_data:
        logger.warning("plot_collocation_diagnostics: No validation data found in DataTree.")
        return None

    all_val_lons = all_val_data["lons"]
    all_val_lats = all_val_data["lats"]
    all_val_times = all_val_data["times"]
    all_val_sources = np.array(all_val_data["sources"])
    all_val_platform_types = np.array(all_val_data["platform_types"])
    total_points = len(all_val_lons)

    # ── Extract matched validation points ───────────────────────────────
    matched_lons = []
    matched_lats = []
    matched_sources = []
    matched_platform_types = []
    
    if "val_lon" in collocation_ds and "val_lat" in collocation_ds:
        matched_lons = collocation_ds["val_lon"].values
        matched_lats = collocation_ds["val_lat"].values
        if "val_source" in collocation_ds:
            matched_sources = collocation_ds["val_source"].values
        else:
            matched_sources = np.full(len(matched_lons), "unknown")
        matched_platform_types = np.full(len(matched_lons), "unknown", dtype=object)
    else:
        matched_platform_types = np.array([], dtype=object)
    
    matched_points = len(matched_lons)
    unmatched_points = total_points - matched_points

    # ── Identify unmatched points ──────────────────────────────────────
    # Build a set of (lon, lat) tuples from matched points for fast lookup
    matched_set = set(zip(np.round(matched_lons, 6), np.round(matched_lats, 6)))
    
    unmatched_lons = []
    unmatched_lats = []
    unmatched_platform_types = []
    for i, (lon, lat) in enumerate(zip(all_val_lons, all_val_lats)):
        if (round(lon, 6), round(lat, 6)) not in matched_set:
            unmatched_lons.append(lon)
            unmatched_lats.append(lat)
            unmatched_platform_types.append(all_val_platform_types[i])
    
    unmatched_lons = np.array(unmatched_lons)
    unmatched_lats = np.array(unmatched_lats)
    unmatched_platform_types = np.array(unmatched_platform_types, dtype=object)
    
    # Extract unmatched times
    unmatched_times = []
    unmatched_indices = []
    for i, (lon, lat) in enumerate(zip(all_val_lons, all_val_lats)):
        if (round(lon, 6), round(lat, 6)) not in matched_set:
            unmatched_times.append(all_val_times[i])
            unmatched_indices.append(i)
    unmatched_times = np.array(unmatched_times, dtype=object)
    
    # Assign platform types to matched points by looking them up in the original data
    matched_times = []
    for i, (mlon, mlat) in enumerate(zip(matched_lons, matched_lats)):
        matched_idx = np.where(
            (np.round(all_val_lons, 6) == round(mlon, 6)) & 
            (np.round(all_val_lats, 6) == round(mlat, 6))
        )[0]
        if len(matched_idx) > 0:
            matched_platform_types[i] = all_val_platform_types[matched_idx[0]]
            matched_times.append(all_val_times[matched_idx[0]])
        else:
            matched_times.append(np.datetime64('NaT'))
    matched_times = np.array(matched_times, dtype=object)

    # ── Create geographic plot ──────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # ── Set geographic bounds from recipe (with +5° padding) ────────────
    try:
        geo_bounds = recipe.config.geographic_bounds
        if geo_bounds:
            padding = 5.0
            extent = [
                geo_bounds.min_lon - padding,
                geo_bounds.max_lon + padding,
                geo_bounds.min_lat - padding,
                geo_bounds.max_lat + padding,
            ]
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            logger.debug(f"Set plot extent to recipe bounds ± {padding}°: {extent}")
    except Exception as e:
        logger.debug(f"Could not set extent from recipe bounds: {e}")

    # Add coastlines and features
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    # ── Plot SAR scene bounds (blue lines) ──────────────────────────────
    transform = ccrs.PlateCarree()
    for i, bounds in enumerate(scene_bounds):
        lon_min, lon_max = bounds["lon_min"], bounds["lon_max"]
        lat_min, lat_max = bounds["lat_min"], bounds["lat_max"]
        
        # Draw rectangle outline for each scene
        lons_box = [lon_min, lon_max, lon_max, lon_min, lon_min]
        lats_box = [lat_min, lat_min, lat_max, lat_max, lat_min]
        ax.plot(lons_box, lats_box, color="blue", linewidth=1.5, 
                transform=transform, zorder=2, label="SAR scene bounds" if i == 0 else "")

    # ── Plot unmatched validation points (red dots for non-scatterometer) ──
    # Separate scatterometer from other in-situ data
    unmatch_scat_mask = unmatched_platform_types == "scatterometer"
    unmatch_other_lons = unmatched_lons[~unmatch_scat_mask]
    unmatch_other_lats = unmatched_lats[~unmatch_scat_mask]
    unmatch_scat_lons = unmatched_lons[unmatch_scat_mask]
    unmatch_scat_lats = unmatched_lats[unmatch_scat_mask]
    
    if len(unmatch_other_lons) > 0:
        ax.scatter(
            unmatch_other_lons, unmatch_other_lats,
            s=20, c="red", alpha=0.6, edgecolors="darkred", linewidths=0.3,
            transform=transform, zorder=3, label=f"Not matched ({len(unmatch_other_lons)})"
        )

    # ── Plot unmatched scatterometer mesh (red lines) ──────────────────
    if len(unmatch_scat_lons) > 0:
        unmatch_scat_times = unmatched_times[unmatch_scat_mask]
        _plot_scatterometer_mesh(
            ax, unmatch_scat_lons, unmatch_scat_lats, unmatch_scat_times,
            color="red", label="Unmatched scatterometer",
            transform=transform, linewidth=0.7, alpha=0.4,
            time_tolerance_minutes=1  # 5-minute tolerance for single overpass
        )

    # ── Plot matched validation points ─────────────────────────────────
    # Separate matched scatterometer from other in-situ data
    match_scat_mask = matched_platform_types == "scatterometer"
    match_other_lons = matched_lons[~match_scat_mask]
    match_other_lats = matched_lats[~match_scat_mask]
    match_other_sources = matched_sources[~match_scat_mask]
    match_scat_lons = matched_lons[match_scat_mask]
    match_scat_lats = matched_lats[match_scat_mask]

    # ── Plot matched non-scatterometer points (colored by source) ────────
    if len(match_other_lons) > 0:
        if len(np.unique(match_other_sources)) > 1:
            # Multiple sources: use color map
            source_map = _source_color_map(list(np.unique(match_other_sources)))
            for source in np.unique(match_other_sources):
                source_mask = match_other_sources == source
                source_lons = match_other_lons[source_mask]
                source_lats = match_other_lats[source_mask]
                color = source_map.get(str(source), "#ff7f0e")
                ax.scatter(
                    source_lons, source_lats,
                    s=25, c=color, alpha=0.7, edgecolors="black", linewidths=0.3,
                    transform=transform, zorder=4, label=f"Matched: {source}"
                )
        else:
            # Single non-scatterometer source: use green
            if len(match_other_sources) > 0:
                ax.scatter(
                    match_other_lons, match_other_lats,
                    s=25, c="green", alpha=0.7, edgecolors="black", linewidths=0.3,
                    transform=transform, zorder=4, label=f"Matched ({len(match_other_lons)})"
                )

    # ── Plot matched scatterometer mesh (green lines) ──────────────────
    if len(match_scat_lons) > 0:
        match_scat_times = np.array(matched_times)[match_scat_mask]
        _plot_scatterometer_mesh(
            ax, match_scat_lons, match_scat_lats, match_scat_times,
            color="green", label="Matched scatterometer",
            transform=transform, linewidth=0.9, alpha=0.7,
            time_tolerance_minutes=5  # 5-minute tolerance for single overpass
        )

    # ── Create title with statistics ────────────────────────────────────
    recipe_name = recipe.config.name or "unknown"
    title = (
        f"{recipe_name} Collocation Diagnostics\n"
        f"Total points: {total_points:,}, Matched: {matched_points}, "
        f"Not matched: {unmatched_points}"
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)

    # ── Add legend ──────────────────────────────────────────────────────
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    # ── Save figure ─────────────────────────────────────────────────────
    fig.tight_layout()
    output_file = plots_dir / f"collocation_diagnostics_{recipe_name}.png"
    fig.savefig(str(output_file), dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(
        "Collocation diagnostics plot saved: %s (%d matched, %d unmatched)",
        output_file, matched_points, unmatched_points
    )
    return output_file

# ---------------------------------------------------------------------------
# Helpers for no-collocation plot with validation data
# ---------------------------------------------------------------------------

def _extract_validation_data_for_plot(datatree):
    """
    Extract all validation observations and their metadata from DataTree.

    Includes in-situ data (mooring, buoy, altimeter, HF radar) and scatterometer
    data (osi_saf_winds, etc.). Handles nested structures (e.g., osi_saf_winds/swath_name/).

    Returns dict with structure:
    {
        'lons': array of all lon values,
        'lats': array of all lat values,
        'times': array of all time values,
        'sources': list of source names (e.g., 'mooring', 'osi_saf_winds'),
        'platform_types': list of platform types (e.g., 'buoy', 'scatterometer'),
        'source_to_data': {source_name: Dataset},
        'all_measurements': dict mapping variable_name -> array of values,
    }

    Returns empty dict if no validation data found.
    """
    val_node = datatree.get("validation")
    if val_node is None or not val_node.children:
        return {}

    all_lons = []
    all_lats = []
    all_times = []
    all_sources = []
    all_platform_types = []
    source_to_data = {}
    all_measurements = {}

    def process_node(node, source_name):
        """Recursively process a validation node."""
        ds = node.to_dataset()
        
        # If this node has lon/lat, process it
        if "lon" in ds.coords and "lat" in ds.coords:
            lons = ds["lon"].values
            lats = ds["lat"].values
            times = ds.coords.get("time", None)
            if times is not None:
                times = times.values
            else:
                times = np.full(len(lons), None, dtype=object)

            # Flatten if needed
            if len(lons.shape) > 1:
                lons = lons.flatten()
            if len(lats.shape) > 1:
                lats = lats.flatten()
            if len(times.shape) > 1:
                times = times.flatten()

            # Determine platform type from attributes and source name
            platform_type = ds.attrs.get("platform_type", None)
            if platform_type is None:
                data_type = ds.attrs.get("data_type", "")
                if data_type:
                    platform_type = data_type
                elif "osi_saf" in source_name:
                    platform_type = "scatterometer"
                else:
                    platform_type = source_name.replace("_", " ").title()

            # Accumulate observations using vectorized operations
            n = len(lons)
            all_lons.extend(lons)
            all_lats.extend(lats)
            all_times.extend(times)
            all_sources.extend([source_name] * n)
            all_platform_types.extend([platform_type] * n)

            # Collect measurement variables
            for var_name in ds.data_vars:
                if var_name not in all_measurements:
                    all_measurements[var_name] = []
                var_data = ds[var_name].values
                if len(var_data.shape) > 1:
                    var_data = var_data.flatten()
                # Use extend with array directly (more efficient than tolist())
                all_measurements[var_name].extend(var_data)

            source_to_data[source_name] = ds
        
        # If this node doesn't have data but has children, recurse
        elif node.children:
            for child_name, child_node in node.children.items():
                process_node(child_node, source_name)

    # Process all top-level validation sources
    for source_name, source_node in val_node.children.items():
        process_node(source_node, source_name)

    if not all_lons:
        return {}

    return {
        "lons": np.array(all_lons),
        "lats": np.array(all_lats),
        "times": np.array(all_times, dtype=object),
        "sources": all_sources,
        "platform_types": all_platform_types,
        "source_to_data": source_to_data,
        "all_measurements": all_measurements,
    }


def _find_available_sar_variable(sar_node, preferred_var: str) -> Optional[str]:
    """
    Find an available SAR variable, preferring *preferred_var* but falling
    back to related alternatives if not found.

    Parameters
    ----------
    sar_node : xr.DataTree
        The '/sar' node of the datatree.
    preferred_var : str
        The preferred SAR variable name (e.g., "owiSignificantWaveHeight").

    Returns
    -------
    str or None
        The name of the first available SAR variable, or None if no suitable
        variable is found in any scene.
    """
    if not sar_node.children:
        return None

    # Get variables from the first SAR scene
    first_scene_name = next(iter(sar_node.children.keys()))
    scene_ds = sar_node[first_scene_name].to_dataset()
    available_vars = list(scene_ds.data_vars)

    # 1. Try the preferred variable first
    for var in available_vars:
        if var == preferred_var:
            return preferred_var

    # 2. Try related variables based on the preferred variable's type
    # e.g., if preferred is owiSignificantWaveHeight, try owiWindSeaHs, owiWaveHs, etc.
    wave_related = [v for v in available_vars if "hs" in v.lower() or "wave" in v.lower()]
    if "wave" in preferred_var.lower() or "hs" in preferred_var.lower():
        if wave_related:
            return wave_related[0]

    wind_related = [v for v in available_vars if "wind" in v.lower()]
    if "wind" in preferred_var.lower():
        if wind_related:
            return wind_related[0]

    current_related = [v for v in available_vars if "current" in v.lower()]
    if "current" in preferred_var.lower():
        if current_related:
            return current_related[0]

    # 3. Fallback: use any non-quality/non-metadata variable
    non_metadata = [v for v in available_vars
                    if not any(x in v.lower() for x in
                              ("quality", "mask", "heading", "incidence", "elevation", "ecmwf"))]
    if non_metadata:
        return non_metadata[0]

    return None


def _extract_scatterometer_overpasses(datatree, recipe):
    """
    Extract scatterometer overpass footprints with entry/exit times.

    For each scatterometer file (= one satellite overpass), calculates the
    geographic boundary of points within the recipe's geographic bounds and
    extracts entry/exit times.

    Parameters
    ----------
    datatree : xr.DataTree
        Step-2 DataTree with validation data.
    recipe : Recipe
        Recipe containing geographic_bounds for filtering.

    Returns
    -------
    list of dict
        Each dict contains: {
            'file_name': str,
            'entry_time': datetime64 or None,
            'exit_time': datetime64 or None,
            'boundary_lons': array of lon values defining boundary,
            'boundary_lats': array of lat values defining boundary,
            'num_points_in_bounds': int,
        }
        Returns empty list if no scatterometer data or no in-bounds points.
    """
    from scipy.spatial import ConvexHull  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    val_node = datatree.get("validation")
    if val_node is None:
        return []

    osi_node = val_node.get("osi_saf_winds")
    if osi_node is None:
        return []

    overpasses = []
    bounds = recipe.config.geographic_bounds

    def process_overpass_node(node, file_name):
        """Process a single scatterometer file node."""
        ds = node.to_dataset()

        if "lon" not in ds.coords or "lat" not in ds.coords:
            return

        lons = ds["lon"].values.flatten()
        lats = ds["lat"].values.flatten()
        times = ds.coords.get("time", None)
        if times is not None:
            times = times.values.flatten()
        else:
            times = np.full(len(lons), None, dtype=object)

        # Filter by geographic bounds
        mask = (
            (lons >= bounds.min_lon) & (lons <= bounds.max_lon) &
            (lats >= bounds.min_lat) & (lats <= bounds.max_lat)
        )
        in_bounds_lons = lons[mask]
        in_bounds_lats = lats[mask]
        in_bounds_times = times[mask]

        if len(in_bounds_lons) == 0:
            return  # No points in bounds for this overpass

        # Calculate entry/exit times
        valid_times = in_bounds_times[in_bounds_times != None]
        if len(valid_times) > 0:
            entry_time = valid_times.min()
            exit_time = valid_times.max()
        else:
            entry_time = None
            exit_time = None

        # Calculate boundary via ConvexHull
        try:
            if len(in_bounds_lons) >= 3:
                points = np.column_stack((in_bounds_lons, in_bounds_lats))
                hull = ConvexHull(points)
                boundary_indices = hull.vertices
                boundary_lons = in_bounds_lons[boundary_indices]
                boundary_lats = in_bounds_lats[boundary_indices]
                # Close the polygon
                boundary_lons = np.append(boundary_lons, boundary_lons[0])
                boundary_lats = np.append(boundary_lats, boundary_lats[0])
            else:
                # Too few points for ConvexHull; just use the points as-is
                boundary_lons = in_bounds_lons
                boundary_lats = in_bounds_lats
        except Exception as e:
            logger.warning("Failed to calculate ConvexHull for overpass %s: %s", file_name, e)
            return

        overpasses.append({
            "file_name": file_name,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "boundary_lons": boundary_lons,
            "boundary_lats": boundary_lats,
            "num_points_in_bounds": len(in_bounds_lons),
        })

    # Recursively process all nodes under osi_saf_winds
    def process_node_tree(node, file_name):
        """Recursively traverse DataTree nodes."""
        ds = node.to_dataset()
        if "lon" in ds.coords and "lat" in ds.coords:
            # This node has data; process it as an overpass
            process_overpass_node(node, file_name)
        else:
            # No data in this node; recurse into children
            for child_name, child_node in node.children.items():
                process_node_tree(child_node, f"{file_name}/{child_name}")

    # Process all children of osi_saf_winds
    for stem_name, stem_node in osi_node.children.items():
        process_node_tree(stem_node, stem_name)

    logger.info("Extracted %d scatterometer overpasses with in-bounds coverage", len(overpasses))
    return overpasses


def _plot_scatterometer_footprints(ax, overpasses, fig, transform=None):
    """
    Plot scatterometer footprint outlines with entry/exit time annotations.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    overpasses : list of dict
        Output from _extract_scatterometer_overpasses().
    fig : matplotlib.figure.Figure
        Figure object for adding annotations.
    transform : cartopy.crs.CRS, optional
        Cartopy transform for plotting (ccrs.PlateCarree()).
    """
    import pandas as pd  # noqa: PLC0415

    if not overpasses:
        ax.text(0.5, 0.5, "No scatterometer coverage in geographic bounds",
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        return

    for idx, overpass in enumerate(overpasses):
        boundary_lons = overpass["boundary_lons"]
        boundary_lats = overpass["boundary_lats"]

        kw_plot = {"transform": transform} if transform else {}

        # Draw footprint boundary as red line
        ax.plot(boundary_lons, boundary_lats, color='red', linewidth=2, zorder=3, **kw_plot)

        # Optional: fill with semi-transparent red
        ax.fill(boundary_lons, boundary_lats, color='red', alpha=0.1, zorder=2, **kw_plot)

        # Calculate centroid for text placement
        centroid_lon = np.mean(boundary_lons)
        centroid_lat = np.mean(boundary_lats)

        # Format entry/exit times
        if overpass["entry_time"] is not None:
            entry_dt = pd.Timestamp(overpass["entry_time"]).strftime("%H:%M:%S")
            exit_dt = pd.Timestamp(overpass["exit_time"]).strftime("%H:%M:%S")
            time_text = f"In: {entry_dt}\nOut: {exit_dt}"
        else:
            time_text = f"Pass {idx + 1}"

        # Add annotation at centroid
        ax.text(centroid_lon, centroid_lat, time_text, fontsize=8, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7), **kw_plot)

        logger.info(
            "Plotted scatterometer overpass %d: %s points, entry=%s, exit=%s",
            idx + 1, overpass["num_points_in_bounds"],
            overpass["entry_time"], overpass["exit_time"]
        )


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
    from *recipe*, save individual PNG files to ``<out_dir>/plots/``, and
    write a combined ``validation_report.pdf`` to *out_dir* (alongside the
    ``validation_statistics_*.nc`` files).

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
        Base output directory.  PNGs go to ``<out_dir>/plots/``; the
        combined PDF is written to ``<out_dir>/validation_report.pdf``
        (alongside the ``validation_statistics_*.nc`` files).
        If None the figures are returned without saving.

    Returns
    -------
    dict[str, list[matplotlib.figure.Figure]]
        ``"<sar_var>_vs_<val_var>"`` → list of Figure objects for that pair.
    """
    from ._variable_map import infer_variable_pairs, filter_variable_pairs  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    base_dir: Optional[Path] = None
    plots_dir: Optional[Path] = None
    if out_dir is not None:
        base_dir = Path(out_dir)
        plots_dir = base_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

    variable = recipe.config.variable
    try:
        pairs = filter_variable_pairs(recipe, collocation_ds)
    except KeyError as exc:
        logger.error("validation_report: %s", exc)
        return {}

    all_figures: Dict[str, list] = {}
    pdf_pages: list = []   # (title, Figure) pairs fed into the PDF

    for sar_var, val_var in pairs:
        key = f"{sar_var}_vs_{val_var}"
        figs = []

        logger.info("Generating plots for %s vs %s …", sar_var, val_var)

        # Scatter
        fig_scatter = plot_scatter(collocation_ds, sar_var, val_var)
        if fig_scatter is not None:
            figs.append(fig_scatter)
            pdf_pages.append((f"{sar_var} vs {val_var} — scatter", fig_scatter))
            if plots_dir:
                fig_scatter.savefig(
                    plots_dir / f"{key}_scatter.png", dpi=150, bbox_inches="tight"
                )

        # Geographic — returns dict[collocation_type, Figure] by default
        try:
            geo_result = plot_geographic(datatree, collocation_ds, sar_var, val_var)
            if isinstance(geo_result, dict):
                for group, fig_geo in geo_result.items():
                    if fig_geo is not None:
                        figs.append(fig_geo)
                        pdf_pages.append(
                            (f"{sar_var} vs {val_var} — geographic [{group}]", fig_geo)
                        )
                        if plots_dir:
                            safe_group = str(group).replace("/", "-")
                            fig_geo.savefig(
                                plots_dir / f"{key}_geographic_{safe_group}.png",
                                dpi=150, bbox_inches="tight",
                            )
            elif geo_result is not None:
                figs.append(geo_result)
                pdf_pages.append((f"{sar_var} vs {val_var} — geographic", geo_result))
                if plots_dir:
                    geo_result.savefig(
                        plots_dir / f"{key}_geographic.png", dpi=150, bbox_inches="tight"
                    )
        except Exception as exc:
            logger.warning("plot_geographic failed for %s: %s", sar_var, exc)

        # Statistics
        if stats_ds_map and key in stats_ds_map:
            fig_stats = plot_statistics(stats_ds_map[key])
            if fig_stats is not None:
                figs.append(fig_stats)
                pdf_pages.append((f"{sar_var} vs {val_var} — statistics", fig_stats))
                if plots_dir:
                    fig_stats.savefig(
                        plots_dir / f"{key}_statistics.png", dpi=150, bbox_inches="tight"
                    )

        # Residuals
        fig_res = plot_residuals(collocation_ds, sar_var, val_var)
        if fig_res is not None:
            figs.append(fig_res)
            pdf_pages.append((f"{sar_var} vs {val_var} — residuals", fig_res))
            if plots_dir:
                fig_res.savefig(
                    plots_dir / f"{key}_residuals.png", dpi=150, bbox_inches="tight"
                )

        all_figures[key] = figs

    # Combined PDF — saved alongside the validation_statistics_*.nc files
    if base_dir is not None and pdf_pages:
        from matplotlib.backends.backend_pdf import PdfPages  # noqa: PLC0415
        import datetime as _dt  # noqa: PLC0415

        pdf_path = base_dir / "validation_report.pdf"
        with PdfPages(pdf_path) as pdf:
            # Cover page
            cover = plt.figure(figsize=(11, 8.5))
            cover.text(
                0.5, 0.60,
                f"SAR L2 Validation Report\n{recipe.config.name}",
                ha="center", va="center", fontsize=20, fontweight="bold",
            )
            cover.text(
                0.5, 0.44,
                f"Variable: {recipe.config.variable}\n"
                f"Generated: {_dt.date.today().isoformat()}",
                ha="center", va="center", fontsize=12,
            )
            pdf.savefig(cover, bbox_inches="tight")
            plt.close(cover)

            for _title, fig in pdf_pages:
                pdf.savefig(fig, bbox_inches="tight")

        logger.info("PDF report saved to %s", pdf_path)

    if plots_dir:
        logger.info("PNG plots saved to %s", plots_dir)

    return all_figures
