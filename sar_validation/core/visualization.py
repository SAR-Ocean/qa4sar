"""
Visualization — step 5b of the validation pipeline.

Five public plot functions:

* :func:`plot_scatter`      — SAR vs. validation variable scatter plot
* :func:`plot_geographic`   — SAR field + collocated points, one subplot per SAR scene
* :func:`plot_statistics`   — bar chart of bias / RMSE / correlation per source
* :func:`plot_residuals`    — histogram / KDE of (SAR − validation) residuals
* :func:`plot_temporal_offset` — |SAR − validation| residual magnitude vs.
  temporal collocation offset

Plus fallback and convenience wrappers:

* :func:`plot_collocation_diagnostics` — SAR scene bounds + matched/unmatched
  validation points (one category per validation source actually present),
  always generated as part of the collocation step (step 3), including when
  there are zero collocated pairs.
* :func:`validation_report` — runs all five plots, infers variable pairs from the recipe,
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
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

__all__ = [
    "plot_scatter",
    "plot_geographic",
    "plot_statistics",
    "plot_residuals",
    "plot_temporal_offset",
    "plot_collocation_diagnostics",
    "validation_report",
]

# Colour palette used for validation sources (cycles if more sources than
# colours). Deliberately avoids mid-gray shades (e.g. tab10's "#7f7f7f") —
# plot_collocation_diagnostics uses gray (#808080) to mean "unmatched", so a
# source landing on a near-gray palette entry would be visually
# indistinguishable from unmatched points of that same source.
_SOURCE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
    "#bcbd22",
]

# Marker shapes paired 1:1 with _SOURCE_COLORS by index, used wherever
# validation sources need to stay identifiable independently of color (e.g.
# when color is taken by a continuous value like wind speed or temporal
# offset instead of by source).
_SOURCE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _source_color_map(sources: List[str]) -> Dict[str, str]:
    return {s: _SOURCE_COLORS[i % len(_SOURCE_COLORS)] for i, s in enumerate(sorted(set(sources)))}


def _canonical_source_order() -> List[str]:
    """
    Fixed, alphabetically-sorted reference order for known validation
    source/platform types, built from the two canonical sets already
    maintained elsewhere in the codebase (avoids introducing a third list
    that could drift out of sync):

    * ``LAYER_DATA_TYPES`` (collocation.py) — scatterometer/altimeter/etc.
    * ``_INSITU_TYPES`` (orchestrator.py) — mooring/buoy/etc.
    """
    from .collocation import LAYER_DATA_TYPES  # noqa: PLC0415
    from .orchestrator import _INSITU_TYPES  # noqa: PLC0415

    return sorted(LAYER_DATA_TYPES | _INSITU_TYPES)


def _source_style_map(sources: List[str]) -> Dict[str, Tuple[str, str]]:
    """
    Map each source name in *sources* to a stable ``(color, marker)`` pair.

    The index used for each name is its position in the fixed canonical
    order (see :func:`_canonical_source_order`), not its position among
    whichever sources happen to be present in this particular call — so a
    known source (e.g. "altimeter") always gets the same color and marker
    everywhere in a report, and across separate report runs. Matching is
    case-insensitive (``plot_collocation_diagnostics`` title-cases layer
    source labels, e.g. "Altimeter", while other call sites use the raw
    lowercase source name — both must land on the same canonical slot).
    Names outside the canonical set are appended afterwards, in sorted order.
    """
    canonical = _canonical_source_order()
    present = sorted(set(sources))
    unknown = [s for s in present if s.lower() not in canonical]
    style: Dict[str, Tuple[str, str]] = {}
    for s in present:
        key = s.lower()
        idx = canonical.index(key) if key in canonical else len(canonical) + unknown.index(s)
        style[s] = (
            _SOURCE_COLORS[idx % len(_SOURCE_COLORS)],
            _SOURCE_MARKERS[idx % len(_SOURCE_MARKERS)],
        )
    return style


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
    if "sar_scene_name" not in collocation_ds:
        return collocation_ds   # old dataset without scene name — return all
    mask = collocation_ds["sar_scene_name"] == scene_name
    return collocation_ds.isel(collocation=mask)


def _drop_nondirectional_sources(coll_ds, val_var):
    """Drop validation sources with no finite ``val_<val_var>`` value.

    For circular variables (wind direction), non-directional instruments
    such as altimeter and radiometer carry all-NaN direction and would
    otherwise render as gray "No data" clutter on the direction maps.
    A source is kept iff it has at least one finite value for *val_var*.
    Returns the input unchanged if the needed columns are absent.
    """
    val_col = f"val_{val_var}"
    if val_col not in coll_ds or "val_source" not in coll_ds:
        return coll_ds
    finite = np.isfinite(np.asarray(coll_ds[val_col].values))
    sources = np.asarray(coll_ds["val_source"].values)
    keep = {s for s in np.unique(sources) if finite[sources == s].any()}
    mask = np.array([s in keep for s in sources])
    return coll_ds.isel(collocation=mask)


def _land_coastline_features(scale: str = "10m"):
    """Natural Earth land/coastline features at a finer resolution than
    cartopy's default 110m — the default is too coarse on complex
    coastlines (straits, bays) and visibly misaligns with SAR swath edges."""
    import cartopy.feature as cfeature  # noqa: PLC0415

    land = cfeature.NaturalEarthFeature("physical", "land", scale, facecolor=cfeature.COLORS["land"])
    coastline = cfeature.NaturalEarthFeature("physical", "coastline", scale, facecolor="none")
    return land, coastline


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
    color_by: str = "source",
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
        Whether per-source legend labels are shown (``color_by="source"``)
        or the per-source marker-shape legend is shown
        (``color_by="temporal_offset"``).
    color_by : str
        ``"source"`` (default) colours points by ``val_source``.
        ``"temporal_offset"`` colours points by ``temporal_distance_minutes``
        (continuous colormap + colorbar) instead, with marker shape still
        varying by source — falls back to ``"source"`` with a warning if
        ``temporal_distance_minutes`` is not present in *collocation_ds*.
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

    extra_cols = [c for c in ("val_id", "val_lat", "val_lon", "temporal_distance_minutes") if c in collocation_ds]
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
    import matplotlib.lines as mlines  # noqa: PLC0415

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    sources = df["val_source"].unique().tolist()
    style = _source_style_map(sources)

    color_by_offset = color_by == "temporal_offset"
    if color_by_offset and "temporal_distance_minutes" not in df.columns:
        warnings.warn(
            "color_by='temporal_offset' requested but collocation_ds has no "
            "'temporal_distance_minutes' column; falling back to color_by='source'."
        )
        color_by_offset = False

    offset_sm = None
    offset_vmin = offset_vmax = None
    if color_by_offset:
        offset_vmin = float(df["temporal_distance_minutes"].min())
        offset_vmax = float(df["temporal_distance_minutes"].max())
    for src in sorted(sources):
        sub = df[df["val_source"] == src]
        marker = style[src][1]
        if color_by_offset:
            offset_sm = ax.scatter(
                sub[val_col], sub[sar_col], s=18, alpha=0.7,
                c=sub["temporal_distance_minutes"], cmap="plasma",
                vmin=offset_vmin, vmax=offset_vmax,
                marker=marker, rasterized=True,
            )
        else:
            label = src if by_source else None
            ax.scatter(sub[val_col], sub[sar_col], s=18, alpha=0.6,
                       color=style[src][0], marker=marker, label=label, rasterized=True)

    all_vals = np.concatenate([df[val_col].values, df[sar_col].values])
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    line11 = ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1, label="1:1")[0]

    if color_by_offset and offset_sm is not None:
        fig.colorbar(offset_sm, ax=ax, label="Temporal offset (min)", shrink=0.8)

    # Annotate with N, bias, RMSE
    from ._variable_map import CIRCULAR_VAL_VARS, circular_diff_deg  # noqa: PLC0415

    if val_var in CIRCULAR_VAL_VARS:
        diff = circular_diff_deg(df[sar_col].values, df[val_col].values)
    else:
        diff = df[sar_col].values - df[val_col].values
    n = len(diff)
    bias = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    if n > 1 and np.std(df[val_col].values) > 0 and np.std(df[sar_col].values) > 0:
        corr = float(np.corrcoef(df[val_col].values, df[sar_col].values)[0, 1])
    else:
        corr = float("nan")
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

    if color_by_offset:
        if by_source:
            handles = [
                mlines.Line2D([], [], marker=style[src][1], linestyle="None",
                              markerfacecolor="lightgray", markeredgecolor="black",
                              markersize=6, label=src)
                for src in sorted(sources)
            ]
            handles.append(line11)
            ax.legend(handles=handles, fontsize=7, framealpha=0.7)
    elif by_source:
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
    val_var: Optional[str] = None,
    *,
    ncols: int = 2,
    cmap: str = "viridis",
    val_cmap: Optional[str] = None,
    point_size: int = 40,
    split_by: str = "collocation_type",
    scenes: Optional[Sequence[str]] = None,
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
    val_cmap : str, optional
        Matplotlib colourmap for the validation scatter points. Defaults to
        the same colourmap as *cmap* (and always shares its colour limits),
        so SAR and validation values are directly comparable by colour; the
        two layers stay visually distinguishable by shape (continuous field
        vs. black-edged markers) rather than by hue. Pass an explicit value
        to opt back into a separate palette.
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

    # Optional allowlist: keep only SAR scenes that matched validation points
    # (computed by validation_report as the union across all variable pairs).
    # An empty/None allowlist means "no filtering" — draw every scene.
    if scenes:
        allow = set(scenes)
        filtered = [s for s in scene_names if s in allow]
        if filtered:
            scene_names = filtered

    val_col = f"val_{val_var}" if val_var else None
    val_col_present = val_col is not None and val_col in collocation_ds

    val_sources = (
        collocation_ds["val_source"].values.tolist()
        if "val_source" in collocation_ds
        else []
    )
    source_style = _source_style_map(val_sources) if val_sources else {}

    # ── Determine group values for splitting ────────────────────────────────
    # None means "do not split" (handled by an early return below); otherwise
    # it is the list of string group keys.
    group_values: Optional[List[str]]
    if split_by:
        if split_by in collocation_ds:
            group_values = sorted(set(str(v) for v in collocation_ds[split_by].values))
        elif split_by in collocation_ds.coords:
            group_values = sorted(set(str(v) for v in collocation_ds.coords[split_by].values))
        else:
            group_values = None
    else:
        group_values = None

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
                    assert val_col is not None   # implied by val_col_present
                    cols_needed.append(val_col)
                df_pts = sub_coll[cols_needed].to_dataframe()
                for _, row in df_pts.iterrows():
                    color = source_style.get(str(row.get("val_source", "")), ("#1f77b4", "o"))[0]
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

    # Colour limits — pooled from the SAR field *and* the validation values
    # (when present) so both layers share one scale and are directly
    # comparable by colour (same across all figures/groups).
    all_field_vals = []
    for scene_name in scene_names:
        arr = _sar_field(sar_node[scene_name].to_dataset(), sar_var)
        if arr is not None:
            all_field_vals.append(arr[np.isfinite(arr)])
    flat = np.concatenate(all_field_vals) if all_field_vals else np.array([])

    finite_v = np.array([])
    if val_col_present:
        finite_v = collocation_ds[val_col].values
        finite_v = finite_v[np.isfinite(finite_v)]

    pooled = np.concatenate([flat, finite_v]) if len(flat) or len(finite_v) else np.array([0.0, 1.0])
    vmin = float(np.nanpercentile(pooled, 2))
    vmax = float(np.nanpercentile(pooled, 98))

    effective_val_cmap = val_cmap if val_cmap is not None else cmap
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sar_sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sar_sm.set_array([])
    sar_norm = norm

    val_norm = val_sm = None
    if len(finite_v) > 0:
        val_norm = norm
        val_sm = mcm.ScalarMappable(cmap=effective_val_cmap, norm=val_norm)
        val_sm.set_array([])

    # One shared colorbar when both layers use the same palette+scale
    # (the default); two only if the caller opted into a distinct val_cmap.
    single_colorbar = val_sm is not None and effective_val_cmap == cmap
    right_margin = 0.88 if (val_sm is None or single_colorbar) else 0.80
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
                land, coastline = _land_coastline_features()
                ax.add_feature(land, facecolor="lightgray", zorder=0, rasterized=True)
                ax.add_feature(coastline, linewidth=0.5, zorder=0, rasterized=True)
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
                        zorder=3, rasterized=True, **kw,
                    )
                else:
                    # Gridded data (e.g., IW/EW mode) — use pcolormesh
                    ax.pcolormesh(
                        scene_ds["lon"].values, scene_ds["lat"].values, arr,
                        cmap=cmap, norm=sar_norm, shading="auto", zorder=2,
                        rasterized=True, **kw,
                    )

            sub_coll = _filter_by_scene(group_coll_ds, scene_name)
            n_pts = sub_coll.sizes.get("collocation", 0)

            if n_pts > 0 and "val_lat" in sub_coll and "val_lon" in sub_coll:
                kw_sc = {"transform": transform, "zorder": 5} if transform else {"zorder": 5}

                # Build a dataframe with observation position + val value
                col_list = ["val_lat", "val_lon"]
                if val_col_present:
                    assert val_col is not None   # implied by val_col_present
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
                    assert val_col is not None   # implied by val_col_present
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
                    nan_mask = df_pts[val_col].isna()
                    valid_pts = df_pts[~nan_mask]
                    nan_pts = df_pts[nan_mask]

                    if len(valid_pts) and "val_source" in valid_pts.columns:
                        for src, grp in valid_pts.groupby("val_source"):
                            marker = source_style.get(str(src), ("#1f77b4", "o"))[1]
                            ax.scatter(
                                grp["val_lon"], grp["val_lat"],
                                c=grp[val_col], cmap=val_cmap, norm=val_norm,
                                marker=marker, s=point_size,
                                edgecolors="black", linewidths=0.4,
                                rasterized=True, **kw_sc,
                            )
                    elif len(valid_pts):
                        ax.scatter(
                            valid_pts["val_lon"], valid_pts["val_lat"],
                            c=valid_pts[val_col], cmap=val_cmap, norm=val_norm,
                            s=point_size, edgecolors="black", linewidths=0.4,
                            rasterized=True, **kw_sc,
                        )
                    if len(nan_pts):
                        # No retrieved value at this location/time — mark it
                        # clearly (gray + hatch) instead of leaving an
                        # invisible gap that looks like "no observation here".
                        ax.scatter(
                            nan_pts["val_lon"], nan_pts["val_lat"],
                            s=point_size, facecolor="lightgray", edgecolors="dimgray",
                            linewidths=0.6, hatch="////", rasterized=True, **kw_sc,
                        )

                    # Fill color varies continuously with the validation
                    # value here (shared with the SAR colorbar), so a solid
                    # legend swatch would misrepresent what's on the map —
                    # marker shape is the discriminator instead.
                    handles = []
                    if "val_source" in df_pts.columns:
                        present = set(df_pts["val_source"].astype(str))
                        handles += [
                            mlines.Line2D([], [], marker=mkr, linestyle="None",
                                          markerfacecolor="lightgray", markeredgecolor="black",
                                          markersize=5, label=s)
                            for s, (_, mkr) in source_style.items() if s in present
                        ]
                    if len(nan_pts):
                        handles.append(
                            mlines.Line2D([], [], marker="o", linestyle="None",
                                          markerfacecolor="lightgray", markeredgecolor="dimgray",
                                          markersize=5, label="No data (NaN)")
                        )
                    if handles:
                        ax.legend(handles=handles, fontsize=6,
                                  loc="lower left", framealpha=0.7)
                elif "val_source" in df_pts.columns:
                    for src, grp in df_pts.groupby("val_source"):
                        color, marker = source_style.get(str(src), ("#ff0000", "o"))
                        ax.scatter(grp["val_lon"], grp["val_lat"],
                                   s=point_size, c=color, marker=marker,
                                   edgecolors="black", linewidths=0.4,
                                   label=str(src), rasterized=True, **kw_sc)
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
        cbar_ax = fig.add_axes((right_margin + 0.01, 0.15, 0.015, 0.70))
        if single_colorbar:
            fig.colorbar(sar_sm, cax=cbar_ax, label=f"{sar_var} / {val_var}")
        else:
            fig.colorbar(sar_sm, cax=cbar_ax, label=f"SAR {sar_var}")
            if val_sm is not None:
                val_cbar_ax = fig.add_axes((right_margin + 0.055, 0.15, 0.015, 0.70))
                fig.colorbar(val_sm, cax=val_cbar_ax, label=f"In-situ {val_var}")

        title = f"SAR {sar_var}"
        if val_var:
            title += f" vs. {val_var}"
        if group_label is not None:
            title += f"  [{group_label}]"
        fig.suptitle(title + " — collocated observations", fontsize=11, y=1.01)
        return fig

    # ── Build figures ────────────────────────────────────────────────────────
    if group_values is None:
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
    metrics: Optional[List[str]] = None,
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

    x = np.arange(len(sources))
    for i, metric in enumerate(available):
        ax = axes[0][i]
        vals = stats_ds[metric].values.astype(float)
        colors = [_SOURCE_COLORS[j % len(_SOURCE_COLORS)] for j in range(len(sources))]
        ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Source")
        ax.set_xticks(x)
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

    from ._variable_map import CIRCULAR_VAL_VARS, circular_diff_deg  # noqa: PLC0415

    if val_var in CIRCULAR_VAL_VARS:
        df["residual"] = circular_diff_deg(df[sar_col].values, df[val_col].values)
        title = f"Residuals: {sar_var} − {val_var} (wrapped to ±180°)"
    else:
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
# 4a. Temporal offset vs. residual
# ---------------------------------------------------------------------------

def plot_temporal_offset(
    collocation_ds,
    sar_var: str,
    val_var: str,
    *,
    by_source: bool = True,
    interactive: bool = False,
    ax=None,
):
    """
    Scatter of |SAR - validation| residual magnitude vs. temporal collocation
    offset (minutes between the SAR acquisition and the validation
    observation) — pairs matched further apart in time are expected to agree
    less well, which helps explain a lower-than-expected correlation
    coefficient in the scatter plot.

    Parameters
    ----------
    collocation_ds : xr.Dataset
        Step-3 collocations (``collocation_results.nc``).
    sar_var : str
        SAR variable name without ``sar_`` prefix.
    val_var : str
        Validation variable name without ``val_`` prefix.
    by_source : bool
        Colour/marker points by ``val_source``.
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

    missing = [c for c in (sar_col, val_col, "temporal_distance_minutes") if c not in collocation_ds]
    if missing:
        warnings.warn(f"No valid data for {sar_col} vs {val_col} (missing {missing}).")
        return None

    extra_cols = [c for c in ("val_id", "val_lat", "val_lon") if c in collocation_ds]
    base_cols = [sar_col, val_col, "val_source", "temporal_distance_minutes"] + extra_cols
    df_raw = collocation_ds[base_cols].to_dataframe()
    if "val_time" in collocation_ds.coords:
        df_raw["val_time"] = collocation_ds["val_time"].values
    df_raw = df_raw.dropna(subset=[sar_col, val_col, "temporal_distance_minutes"])

    if df_raw.empty:
        warnings.warn(f"No valid data for {sar_col} vs {val_col}.")
        return None

    df = _deduplicate_obs(df_raw, sar_col, val_col)

    from ._variable_map import CIRCULAR_VAL_VARS, circular_diff_deg  # noqa: PLC0415

    if val_var in CIRCULAR_VAL_VARS:
        residual = circular_diff_deg(df[sar_col].values, df[val_col].values)
    else:
        residual = df[sar_col].values - df[val_col].values
    df["abs_residual"] = np.abs(residual)

    if interactive:
        _require("plotly")
        import plotly.express as px  # noqa: PLC0415

        fig = px.scatter(
            df, x="temporal_distance_minutes", y="abs_residual",
            color="val_source" if by_source else None,
            labels={
                "temporal_distance_minutes": "Temporal offset (min)",
                "abs_residual": f"|{sar_var} - {val_var}|",
                "val_source": "Source",
            },
            title=f"|{sar_var} - {val_var}| vs. temporal offset",
        )
        return fig

    import matplotlib.pyplot as plt  # noqa: PLC0415

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    if by_source:
        sources = sorted(df["val_source"].unique())
        style = _source_style_map(sources)
        for src in sources:
            sub = df[df["val_source"] == src]
            color, marker = style[src]
            ax.scatter(sub["temporal_distance_minutes"], sub["abs_residual"],
                       s=18, alpha=0.6, color=color, marker=marker, label=src,
                       rasterized=True)
        ax.legend(fontsize=7, framealpha=0.7)
    else:
        ax.scatter(df["temporal_distance_minutes"], df["abs_residual"],
                   s=18, alpha=0.6, color="#1f77b4", rasterized=True)

    n = len(df)
    if n > 1 and np.std(df["temporal_distance_minutes"].values) > 0 and np.std(df["abs_residual"].values) > 0:
        corr = float(np.corrcoef(df["temporal_distance_minutes"].values, df["abs_residual"].values)[0, 1])
    else:
        corr = float("nan")
    ax.text(0.04, 0.96, f"N={n}\nr={corr:.3f}", transform=ax.transAxes,
            va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    ax.set_xlabel("Temporal offset (min)")
    ax.set_ylabel(f"|{sar_var} − {val_var}|")
    ax.set_title(f"{sar_var} vs {val_var} — residual magnitude vs. temporal offset")
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4b. Collocation diagnostics plot
# ---------------------------------------------------------------------------

def plot_collocation_diagnostics(
    datatree,
    collocation_ds,
    recipe,
    output_dir: Union[str, Path],
    filename_suffix: str = "",
) -> Union[Path, None]:
    """
    Plot collocation diagnostics: SAR scene bounds, and matched/unmatched
    validation points for every validation source actually present (in-situ,
    plus one category per distinct layer_vs_layer source type — e.g.
    scatterometer, altimeter, hf_radar — found in the data; categories with
    zero points are omitted).

    Creates a geographic map showing:
    - SAR scene footprints (blue lines for each scene boundary)
    - Unmatched validation observations (red dots, drawn first) — restricted
      to points whose timestamp falls within a SAR scene's time range ±
      that source's collocation time tolerance, i.e. the same coarse
      temporal pre-filter the real matcher applies. Points that were never
      temporally eligible are omitted rather than cluttering the plot with
      "unmatched" observations that failed for a time reason, not a spatial
      one; the omitted count per category is reported in the log line.
    - Matched validation observations (colored dots, drawn last so a matched
      point from one category is never hidden underneath an unmatched point
      from another category); per-source matched counts appear in the legend
    - Plot extent set to the recipe's geographic bounds

    Always generated as part of the collocation step, including when there
    are zero collocated pairs — pass ``collocation_ds=None`` in that case and
    every validation point shows up as unmatched.

    Parameters
    ----------
    datatree : xr.DataTree
        Step-2 DataTree (``datatree.nc``).
    collocation_ds : xr.Dataset or None
        Step-3 collocation results (``collocation_results.nc``), or None if
        no collocated pairs were found.
    recipe : Recipe
        Recipe object containing metadata.
    output_dir : str or Path
        Directory to save the PNG file (typically the base_dir).
    filename_suffix : str
        Appended to the output filename stem, e.g. ``"_individual"``. Lets
        diagnostic plots for two collocation methods coexist without
        overwriting each other.

    Returns
    -------
    Path or None
        Path to the saved PNG file, or None if plot could not be generated.
    """
    from .collocation import LAYER_DATA_TYPES  # noqa: PLC0415
    from .recipe import DEFAULT_LAYER_TYPE_SPECS  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Set up cartopy if available
    try:
        import cartopy.crs as ccrs  # noqa: PLC0415
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

    scene_bounds = []            # bounding boxes for grid-mode (IW/EW) scenes
    footprint_points: list[tuple[float, float]] = []  # (lon, lat) per sparse WV-mode imagette
    # (time_min, time_max) per scene that has a usable time coordinate — used
    # to decide which validation points were even temporally eligible to
    # match any SAR acquisition, before we get to spatial matching at all.
    scene_time_windows: List[Tuple["pd.Timestamp", "pd.Timestamp"]] = []
    scene_names = list(sar_node.children.keys())
    # Footprint radius each sparse WV imagette actually matches within, so the
    # plot's circles line up with what the collocation step considered.
    footprint_radius_km = getattr(
        recipe.config.collocation, "sar_footprint_radius_km", 14.0
    )

    for scene_name in scene_names:
        scene_ds = sar_node[scene_name].to_dataset()
        if "lon" not in scene_ds.coords or "lat" not in scene_ds.coords:
            continue

        lons = scene_ds["lon"].values
        lats = scene_ds["lat"].values

        # WV/point-mode scenes are sparse imagettes, not a filled grid — draw
        # them as per-footprint circles instead of one misleading bounding box.
        is_wv_scene = (
            ("point" in scene_ds.dims and "y" not in scene_ds.dims)
            or str(scene_ds.attrs.get("swath_mode", "")).upper() == "WV"
        )

        # Handle both 2D grids and 1D point arrays
        if len(lons.shape) > 1:
            lons_flat = lons.flatten()
            lats_flat = lats.flatten()
        else:
            lons_flat = lons
            lats_flat = lats

        valid_mask = np.isfinite(lons_flat) & np.isfinite(lats_flat)
        if valid_mask.any():
            if is_wv_scene:
                footprint_points.extend(
                    zip(lons_flat[valid_mask].tolist(), lats_flat[valid_mask].tolist())
                )
            else:
                scene_bounds.append({
                    "lon_min": float(np.nanmin(lons_flat[valid_mask])),
                    "lon_max": float(np.nanmax(lons_flat[valid_mask])),
                    "lat_min": float(np.nanmin(lats_flat[valid_mask])),
                    "lat_max": float(np.nanmax(lats_flat[valid_mask])),
                })

        if "time" in scene_ds.coords:
            # Scalar for grid-mode (IW/EW) scenes, a (point,) array for
            # WV-mode scenes — atleast_1d handles both uniformly.
            scene_times = pd.to_datetime(np.atleast_1d(scene_ds["time"].values))
            scene_times = scene_times[~scene_times.isna()]
            if len(scene_times) > 0:
                scene_time_windows.append((scene_times.min(), scene_times.max()))

    if not scene_bounds and not footprint_points:
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
    platform_types_arr = np.array(all_val_data["platform_types"])

    # ── Resolve the time tolerance each point would actually be matched
    # with, mirroring run_collocation's own resolution (collocation.py) so
    # "temporally eligible" here means the same thing it means there. ─────
    coll_cfg = recipe.config.collocation
    pvl_default_tol = coll_cfg.point_vs_layer.time_tolerance_minutes
    source_type_tol: Dict[str, float] = {
        src.source_type: src.collocation_kwargs.get("time_tolerance_minutes", pvl_default_tol)
        for src in recipe.config.validation_sources
    }
    layer_vs_layer_specs = dict(DEFAULT_LAYER_TYPE_SPECS)
    if coll_cfg.layer_vs_layer is not None:
        layer_vs_layer_specs.update(coll_cfg.layer_vs_layer.layer_type_specs)
    # Collapse altimeter_1hz/altimeter_5hz down to one "altimeter" key (the
    # plot doesn't distinguish frequency), taking the max of the two so a
    # point isn't hidden just because it misses the stricter of the pair.
    layer_type_tol: Dict[str, float] = {}
    for key, spec in layer_vs_layer_specs.items():
        base = key.split("_")[0] if key.startswith("altimeter") else key
        tol = spec.get("time_tolerance_minutes", pvl_default_tol)
        layer_type_tol[base] = max(layer_type_tol.get(base, 0), tol)

    def _time_tolerance_minutes(label: str, ptype) -> float:
        if label == "In-situ":
            return source_type_tol.get(str(ptype), pvl_default_tol)
        return layer_type_tol.get(label.lower(), pvl_default_tol)

    def _time_eligible(point_time, tol_minutes: float) -> bool:
        # Points with no timestamp, or when no scene has usable time
        # metadata, can't be evaluated — never hide them on account of a
        # check we can't actually perform.
        if not scene_time_windows or point_time is None or pd.isnull(point_time):
            return True
        pt = pd.Timestamp(point_time)
        tol = pd.Timedelta(minutes=tol_minutes)
        return any(t_min - tol <= pt <= t_max + tol for t_min, t_max in scene_time_windows)

    # ── Split matched points (from collocation_ds) by their validation
    # source — "In-situ" for anything that isn't a layer type, plus one
    # bucket per distinct layer source (altimeter/scatterometer/hf_radar).
    # This mirrors the unmatched classification below and is independent of
    # collocation_type, so point_vs_point / point_vs_layer / layer_vs_layer
    # all bucket correctly. Tolerates collocation_ds=None (zero pairs) by
    # treating everything as unmatched. ──────────────────────────────────
    has_matches = collocation_ds is not None and "val_lon" in collocation_ds and "val_lat" in collocation_ds
    if has_matches:
        val_lon_all = np.asarray(collocation_ds["val_lon"].values)
        val_lat_all = np.asarray(collocation_ds["val_lat"].values)
        val_source_all = np.asarray(
            collocation_ds["val_source"].values if "val_source" in collocation_ds
            else np.full(len(val_lon_all), "unknown")
        )
    else:
        val_lon_all = np.array([])
        val_lat_all = np.array([])
        val_source_all = np.array([])

    matched_labels_all = np.array([
        str(src).title() if src in LAYER_DATA_TYPES else "In-situ"
        for src in val_source_all
    ])

    matched_by_category: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label in sorted(set(matched_labels_all.tolist())) or ["In-situ"]:
        sel = matched_labels_all == label
        matched_by_category[label] = (
            val_lon_all[sel], val_lat_all[sel], val_source_all[sel],
        )
    # Guarantee an In-situ bucket exists so downstream code can assume it.
    matched_by_category.setdefault("In-situ", (np.array([]), np.array([]), np.array([])))

    matched_lookup = {
        label: set(zip(np.round(lons, 6), np.round(lats, 6)))
        for label, (lons, lats, _) in matched_by_category.items()
    }

    # ── Classify every validation point into the same categories, using
    # LAYER_DATA_TYPES (the same constant collocation.py uses to route
    # sources to layer_vs_layer collocation) as the single source of truth
    # for "which platform_type values are layer types" — so this plot
    # automatically covers scatterometer, altimeter, hf_radar, or any future
    # addition, without hardcoding a fixed list here. ────────────────────
    unmatched_by_category: Dict[str, Tuple[List[float], List[float], List[str]]] = {
        label: ([], [], []) for label in matched_by_category
    }
    for lon, lat, ptype, ptime in zip(all_val_lons, all_val_lats, platform_types_arr, all_val_times):
        label = str(ptype).title() if ptype in LAYER_DATA_TYPES else "In-situ"
        if (round(lon, 6), round(lat, 6)) in matched_lookup.get(label, set()):
            continue
        tol_minutes = _time_tolerance_minutes(label, ptype)
        if not _time_eligible(ptime, tol_minutes):
            continue
        lon_lats_srcs = unmatched_by_category.setdefault(label, ([], [], []))
        lon_lats_srcs[0].append(lon)
        lon_lats_srcs[1].append(lat)
        lon_lats_srcs[2].append(str(ptype))

    # ── Assemble final per-category data, dropping empty categories ──────
    categories = []
    for label in sorted(set(matched_by_category) | set(unmatched_by_category)):
        m_lon, m_lat, m_src = matched_by_category.get(label, (np.array([]), np.array([]), np.array([])))
        u_lon, u_lat, u_src = unmatched_by_category.get(label, ([], [], []))
        u_lon = np.array(u_lon)
        u_lat = np.array(u_lat)
        u_src = np.array(u_src)
        if len(m_lon) + len(u_lon) == 0:
            continue
        categories.append({
            "label": label,
            "matched_lon": m_lon, "matched_lat": m_lat, "matched_source": m_src,
            "unmatched_lon": u_lon, "unmatched_lat": u_lat, "unmatched_source": u_src,
        })

    if not categories:
        logger.warning("plot_collocation_diagnostics: No classifiable validation points.")
        return None

    logger.debug(
        "Classification: %s",
        "; ".join(
            f"{c['label']}={len(c['matched_lon'])} matched/{len(c['unmatched_lon'])} unmatched"
            for c in categories
        ),
    )

    # ── Build one shared color map across every distinct source name, so
    # in-situ sub-sources (mooring/buoy/…) and layer-type categories
    # (altimeter/scatterometer/…) never collide on the same color ────────
    all_source_names: set[str] = set()
    for cat in categories:
        if cat["label"] == "In-situ" and len(cat["matched_source"]) > 0:
            all_source_names.update(str(s) for s in np.unique(np.asarray(cat["matched_source"])))
        else:
            all_source_names.add(str(cat["label"]))
    source_style_map = _source_style_map(sorted(all_source_names))

    bounds = recipe.config.geographic_bounds

    # ── Create geographic plot ──────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    transform = ccrs.PlateCarree()

    # Add coastlines and features
    land, coastline = _land_coastline_features()
    ax.add_feature(land, facecolor="lightgray", alpha=0.3, zorder=0)
    ax.add_feature(coastline, linewidth=0.5, zorder=0)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    # ── Set plot extent to the recipe's geographic bounds ────────────────
    ax.set_extent([bounds.min_lon, bounds.max_lon, bounds.min_lat, bounds.max_lat],
                  crs=ccrs.PlateCarree())

    # ── SAR coverage (zorder=1): Grid scenes → bounding box; sparse WV
    # imagettes → one footprint circle each (radius = the collocation footprint
    # radius), so it's visually clear that matches are only possible near each
    # imagette, not across the whole bounding rectangle. ──────────────────────
    for i, sb in enumerate(scene_bounds):
        lons_box = [sb["lon_min"], sb["lon_max"], sb["lon_max"], sb["lon_min"], sb["lon_min"]]
        lats_box = [sb["lat_min"], sb["lat_min"], sb["lat_max"], sb["lat_max"], sb["lat_min"]]
        ax.plot(lons_box, lats_box, color="blue", linewidth=1.5,
                transform=transform, zorder=1, label="SAR scene bounds" if i == 0 else "")

    if footprint_points:
        theta = np.linspace(0, 2 * np.pi, 60)
        r_lat_deg = footprint_radius_km / 111.0
        for j, (flon, flat) in enumerate(footprint_points):
            # Approximate circle in lon/lat (lon degrees shrink by cos(lat)).
            cos_lat = max(np.cos(np.radians(flat)), 1e-6)
            circ_lon = flon + (r_lat_deg / cos_lat) * np.cos(theta)
            circ_lat = flat + r_lat_deg * np.sin(theta)
            ax.plot(circ_lon, circ_lat, color="blue", linewidth=1.2,
                    transform=transform, zorder=1,
                    label=f"SAR footprint (±{footprint_radius_km:.0f} km)" if j == 0 else "")
            ax.scatter([flon], [flat], s=10, c="blue", marker="+",
                       transform=transform, zorder=1)

    # ── Tier 1 (zorder=2): unmatched layer data (non-in-situ categories) ────
    # Gray (#808080) with alpha=0.3, per-source markers, drawn first so matched
    # points (tiers 3-4) are never visually covered by an unmatched point from
    # a different category.
    for cat in categories:
        if cat["label"] != "In-situ" and len(cat["unmatched_lon"]) > 0:
            u_lon = np.asarray(cat["unmatched_lon"])
            u_lat = np.asarray(cat["unmatched_lat"])
            u_src = np.asarray(cat["unmatched_source"])
            if len(u_src) > 0:
                # Iterate through unique sources to apply per-source markers
                for source in np.unique(u_src):
                    mask = u_src == source
                    # Layer sources are stored lowercase but source_style_map has
                    # title-case keys (e.g., "altimeter" → "Altimeter")
                    color, marker = source_style_map.get(str(source).title(), ("#808080", "o"))
                    ax.scatter(
                        u_lon[mask], u_lat[mask],
                        s=18, c="#808080", marker=marker, alpha=0.3, edgecolors="none",
                        transform=transform, zorder=2,
                    )
            else:
                # Fallback if no source info
                ax.scatter(
                    u_lon, u_lat,
                    s=18, c="#808080", alpha=0.3, edgecolors="none",
                    transform=transform, zorder=2,
                )

    # ── Tier 2 (zorder=3): unmatched in-situ data ──────────────────────────
    # Gray (#808080) with alpha=0.3, per-source markers, drawn separately to
    # layer unmatched.
    for cat in categories:
        if cat["label"] == "In-situ" and len(cat["unmatched_lon"]) > 0:
            u_lon = np.asarray(cat["unmatched_lon"])
            u_lat = np.asarray(cat["unmatched_lat"])
            u_src = np.asarray(cat["unmatched_source"])
            if len(u_src) > 0:
                # Iterate through unique sources to apply per-source markers
                for source in np.unique(u_src):
                    mask = u_src == source
                    color, marker = source_style_map.get(str(source), ("#808080", "o"))
                    ax.scatter(
                        u_lon[mask], u_lat[mask],
                        s=18, c="#808080", marker=marker, alpha=0.3, edgecolors="none",
                        transform=transform, zorder=3,
                    )
            else:
                # Fallback if no source info
                ax.scatter(
                    u_lon, u_lat,
                    s=18, c="#808080", alpha=0.3, edgecolors="none",
                    transform=transform, zorder=3,
                )

    # ── Tier 3 (zorder=5): matched layer data ─────────────────────────────
    # Colored by source (from _source_style_map), alpha=1.0 (emphasized),
    # drawn before matched in-situ so in-situ markers are always on top. No
    # marker edge, matching the (visually edgeless) in-situ markers below.
    # Categories are drawn in descending order of matched-point count, so a
    # sparser source (e.g. altimeter) ends up layered on top of a denser one
    # (e.g. scatterometer) instead of being buried underneath it. ──────────
    layer_categories_by_count = sorted(
        (cat for cat in categories if cat["label"] != "In-situ"),
        key=lambda c: len(c["matched_lon"]),
        reverse=True,
    )
    for cat in layer_categories_by_count:
        m_lon = np.asarray(cat["matched_lon"])
        m_lat = np.asarray(cat["matched_lat"])
        if len(m_lon) == 0:
            continue
        color, marker = source_style_map.get(str(cat["label"]), ("#2ca02c", "o"))
        ax.scatter(
            m_lon, m_lat,
            s=25, c=color, marker=marker, alpha=1.0,
            edgecolors="none",
            transform=transform, zorder=5, label=f"{cat['label']} matched ({len(m_lon)})",
        )

    # ── Tier 4 (zorder=6): matched in-situ data ───────────────────────────
    # Colored by sub-source type (mooring/buoy/drifter/…), alpha=1.0 (fully
    # opaque, matching Tier 3), drawn last so in-situ markers are always on
    # top. Sub-sources are drawn in descending order of matched-point count
    # too, matching Tier 3's ordering so the sparsest in-situ instrument
    # stays visible on top. ─────────────────────────────────────────────────
    for cat in categories:
        if cat["label"] == "In-situ":
            m_lon = np.asarray(cat["matched_lon"])
            m_lat = np.asarray(cat["matched_lat"])
            m_src = np.asarray(cat["matched_source"])
            if len(m_lon) == 0:
                continue
            # In-situ keeps its per-source-type coloring (mooring/buoy/drifter/…),
            # each its own legend entry.
            if len(m_src) > 0:
                sources_by_count = sorted(
                    np.unique(m_src).tolist(),
                    key=lambda s: int(np.sum(m_src == s)),
                    reverse=True,
                )
                for source in sources_by_count:
                    mask = m_src == source
                    count = int(np.sum(mask))
                    color, marker = source_style_map.get(str(source), ("#ff7f0e", "o"))
                    ax.scatter(
                        m_lon[mask], m_lat[mask],
                        s=25, c=color, marker=marker, alpha=1.0,
                        edgecolors="none",
                        transform=transform, zorder=6,
                        label=f"In-situ matched: {source} ({count})",
                    )
            else:
                # Fallback if no source info available
                ax.scatter(
                    m_lon, m_lat,
                    s=25, c="#ff7f0e", marker="o", alpha=1.0,
                    edgecolors="none",
                    transform=transform, zorder=6,
                    label=f"In-situ matched ({len(m_lon)})",
                )

    # ── Title: recipe name only; per-category counts go to the log line
    # below and to the legend instead of cluttering the plot title. ───────
    recipe_name = recipe.config.name or "unknown"
    segments = [
        f"{cat['label']}: {len(cat['matched_lon']) + len(cat['unmatched_lon'])}, "
        f"Matched: {len(cat['matched_lon'])}, Unmatched: {len(cat['unmatched_lon'])}"
        for cat in categories
    ]
    title = f"{recipe_name} Collocation Diagnostics"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)

    # ── Add legend with an explanatory entry ──────────────────────────────
    # The explanation of matched-vs-unmatched rendering is folded into the
    # legend itself (rather than a separate floating text box) so it can
    # never visually collide with the legend — a floating annotation
    # overlapped the legend box when labels were long (e.g. "Radiometer
    # matched (1635)").
    import matplotlib.lines as mlines  # noqa: PLC0415
    handles, labels = ax.get_legend_handles_labels()
    explanation = mlines.Line2D(
        [], [], linestyle="None", marker="o", markerfacecolor="lightgray",
        markeredgecolor="black", markersize=6,
        label="Filled = matched, faint gray = unmatched",
    )
    handles.append(explanation)
    labels.append(explanation.get_label())
    ax.legend(handles=handles, labels=labels, loc="lower left", fontsize=8, framealpha=0.9)

    # ── Save figure ─────────────────────────────────────────────────────
    fig.tight_layout()
    output_file = plots_dir / f"collocation_diagnostics_{recipe_name}{filename_suffix}.png"
    fig.savefig(str(output_file), dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Collocation diagnostics plot saved: %s (%s)", output_file, "; ".join(segments))
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
    all_measurements: Dict[str, list] = {}

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

            n = len(lons)

            # Prefer a per-point platform_type coord when present — a single
            # in-situ CSV can mix multiple platform types (mooring/buoy/
            # drifter/tidal_gauge, see from_insitu_csv), and downstream
            # per-source time-tolerance lookups need that granularity, not
            # one dataset-wide label.
            if "platform_type" in ds.coords or "platform_type" in ds.data_vars:
                ptypes = ds["platform_type"].values
                if len(ptypes.shape) > 1:
                    ptypes = ptypes.flatten()
                platform_types_for_points = list(ptypes)
            else:
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
                platform_types_for_points = [platform_type] * n

            # Accumulate observations using vectorized operations
            all_lons.extend(lons)
            all_lats.extend(lats)
            all_times.extend(times)
            all_sources.extend([source_name] * n)
            all_platform_types.extend(platform_types_for_points)

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


# ---------------------------------------------------------------------------
# 5. Validation report (convenience wrapper)
# ---------------------------------------------------------------------------

def validation_report(
    collocation_ds,
    datatree,
    recipe,
    stats_ds_map: Optional[Dict[str, "xr.Dataset"]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    filename_suffix: str = "",
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
    filename_suffix : str
        Appended to the PNG/PDF filename stems, e.g. ``"_individual"``.
        Lets reports for two collocation methods coexist in the same
        ``plots/`` directory without overwriting each other.

    Returns
    -------
    dict[str, list[matplotlib.figure.Figure]]
        ``"<sar_var>_vs_<val_var>"`` → list of Figure objects for that pair.
    """
    from ._variable_map import infer_variable_pairs, filter_variable_pairs, CIRCULAR_VAL_VARS  # noqa: PLC0415
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

    # Union across all pairs of SAR scenes that matched at least one
    # validation point — used to drop scenes with no matches from the
    # geographic plots. collocation_ds holds only matched pairs, so every
    # scene present here has >= 1 match. None => don't filter.
    matched_scenes = (
        sorted(set(str(s) for s in collocation_ds["sar_scene_name"].values))
        if "sar_scene_name" in collocation_ds else None
    )

    for sar_var, val_var in pairs:
        key = f"{sar_var}_vs_{val_var}"
        figs = []

        # Direction-only sources for circular variables (WDIR): drop
        # non-directional instruments (altimeter/radiometer, all-NaN
        # direction) so they don't clutter the direction plots. Speed
        # pairs keep every source.
        pair_ds = (
            _drop_nondirectional_sources(collocation_ds, val_var)
            if val_var in CIRCULAR_VAL_VARS else collocation_ds
        )

        logger.info("Generating plots for %s vs %s …", sar_var, val_var)

        # Scatter
        fig_scatter = plot_scatter(pair_ds, sar_var, val_var)
        if fig_scatter is not None:
            figs.append(fig_scatter)
            pdf_pages.append((f"{sar_var} vs {val_var} — scatter", fig_scatter))
            if plots_dir:
                fig_scatter.savefig(
                    plots_dir / f"{key}{filename_suffix}_scatter.png", dpi=150, bbox_inches="tight"
                )

        # Geographic — returns dict[collocation_type, Figure] by default
        try:
            geo_result = plot_geographic(datatree, pair_ds, sar_var, val_var, scenes=matched_scenes)
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
                                plots_dir / f"{key}{filename_suffix}_geographic_{safe_group}.png",
                                dpi=150, bbox_inches="tight",
                            )
            elif geo_result is not None:
                figs.append(geo_result)
                pdf_pages.append((f"{sar_var} vs {val_var} — geographic", geo_result))
                if plots_dir:
                    geo_result.savefig(
                        plots_dir / f"{key}{filename_suffix}_geographic.png", dpi=150, bbox_inches="tight"
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
                        plots_dir / f"{key}{filename_suffix}_statistics.png", dpi=150, bbox_inches="tight"
                    )

        # Residuals
        fig_res = plot_residuals(pair_ds, sar_var, val_var)
        if fig_res is not None:
            figs.append(fig_res)
            pdf_pages.append((f"{sar_var} vs {val_var} — residuals", fig_res))
            if plots_dir:
                fig_res.savefig(
                    plots_dir / f"{key}{filename_suffix}_residuals.png", dpi=150, bbox_inches="tight"
                )

        # Scatter colored by temporal offset — same SAR-vs-validation
        # comparison as above, but colored by how far apart in time each
        # pair was matched, to help explain a lower-than-expected r.
        fig_scatter_offset = plot_scatter(pair_ds, sar_var, val_var, color_by="temporal_offset")
        if fig_scatter_offset is not None:
            figs.append(fig_scatter_offset)
            pdf_pages.append((f"{sar_var} vs {val_var} — scatter (colored by temporal offset)", fig_scatter_offset))
            if plots_dir:
                fig_scatter_offset.savefig(
                    plots_dir / f"{key}{filename_suffix}_scatter_by_offset.png", dpi=150, bbox_inches="tight"
                )

        # Temporal offset vs. residual magnitude
        fig_offset = plot_temporal_offset(pair_ds, sar_var, val_var)
        if fig_offset is not None:
            figs.append(fig_offset)
            pdf_pages.append((f"{sar_var} vs {val_var} — residual vs. temporal offset", fig_offset))
            if plots_dir:
                fig_offset.savefig(
                    plots_dir / f"{key}{filename_suffix}_temporal_offset.png", dpi=150, bbox_inches="tight"
                )

        all_figures[key] = figs

        # Each figure is already saved to PNG and queued in pdf_pages (which
        # holds a direct reference, still usable by PdfPages.savefig after
        # plt.close). Closing here just deregisters them from pyplot's
        # global figure manager so they don't accumulate across pairs and
        # across the two collocation-method passes triggered by
        # --layer-vs-layer-collocation-method both.
        if base_dir is not None:
            for fig in figs:
                plt.close(fig)

    # Collocation diagnostics plot — generated once per recipe
    fig_diag = None
    if base_dir is not None:
        try:
            diag_path = plot_collocation_diagnostics(
                datatree, collocation_ds, recipe, base_dir, filename_suffix
            )
            if diag_path is not None:
                logger.info("Collocation diagnostics plot saved to %s", diag_path)
                # Embed the saved PNG as a page in the combined PDF report —
                # plot_collocation_diagnostics() closes its own figure
                # internally (it's also called standalone from cli.py), so
                # the only way to include it in pdf_pages is to reload the
                # rendered image.
                diag_img = plt.imread(str(diag_path))
                img_h, img_w = diag_img.shape[0], diag_img.shape[1]
                fig_diag = plt.figure(figsize=(img_w / 150, img_h / 150), dpi=150)
                ax_diag = fig_diag.add_axes([0, 0, 1, 1])
                ax_diag.imshow(diag_img)
                ax_diag.axis("off")
                # Lead the report body with the diagnostics overview (the
                # cover page is written separately, so index 0 here becomes
                # the first page after the cover).
                pdf_pages.insert(0, (f"Collocation diagnostics — {recipe.config.name}", fig_diag))
        except Exception as exc:
            logger.warning("plot_collocation_diagnostics failed: %s", exc)

    # Combined PDF — saved alongside the validation_statistics_*.nc files
    if base_dir is not None and pdf_pages:
        from matplotlib.backends.backend_pdf import PdfPages  # noqa: PLC0415
        import datetime as _dt  # noqa: PLC0415

        pdf_path = base_dir / f"validation_report{filename_suffix}.pdf"
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
                pdf.savefig(fig, dpi=150, bbox_inches="tight")

        logger.info("PDF report saved to %s", pdf_path)

    # fig_diag isn't in the `figs` list closed earlier (it's created after
    # that loop), so it stays open until here — close it now that the PDF
    # write is done and it's no longer needed.
    if fig_diag is not None:
        plt.close(fig_diag)

    if plots_dir:
        logger.info("PNG plots saved to %s", plots_dir)

    return all_figures