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
  embeds plots in ``validation_report<suffix>.pdf``, and saves the collocation-diagnostics PNG to ``<out_dir>/plots/``

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
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr
from scipy import ndimage

logger = logging.getLogger(__name__)


def _pad_degenerate_range(vmin: float, vmax: float) -> Tuple[float, float]:
    """Pad an all-identical (vmin == vmax) value range to a non-degenerate
    one so set_xlim/set_ylim don't warn about singular limits."""
    if vmin == vmax:
        pad = max(0.5, abs(vmin) * 0.05)
        return vmin - pad, vmax + pad
    return vmin, vmax


def _source_marker_handles(items, *, markersize: float = 6, markeredgecolor: str = "black") -> list:
    """Build one legend Line2D per (label, marker) pair in *items*."""
    import matplotlib.lines as mlines  # noqa: PLC0415

    return [
        mlines.Line2D([], [], marker=marker, linestyle="None",
                      markerfacecolor="lightgray", markeredgecolor=markeredgecolor,
                      markersize=markersize, label=label)
        for label, marker in items
    ]


def _draw_colorbar(
    fig, right_margin: float, sar_sm, val_sm, single_colorbar: bool,
    collocation_ds, sar_var: str, val_var: Optional[str], sar_field_transform,
) -> None:
    """Compute SAR/validation axis labels and draw the shared or two-colorbar
    layout shared by plot_geographic's per-scene and grouped figure builders."""
    sar_label = (
        _labeled_var(collocation_ds, f"val_{val_var}", sar_var)
        if sar_field_transform is not None
        else _labeled_var(collocation_ds, f"sar_{sar_var}", sar_var)
    )
    val_label = _labeled_var(collocation_ds, f"val_{val_var}", val_var) if val_var else None

    fig.subplots_adjust(right=right_margin)
    cbar_ax = fig.add_axes((right_margin + 0.01, 0.15, 0.015, 0.70))
    if single_colorbar:
        fig.colorbar(sar_sm, cax=cbar_ax, label=f"{sar_label} / {val_label}")
    else:
        fig.colorbar(sar_sm, cax=cbar_ax, label=f"SAR {sar_label}")
        if val_sm is not None:
            val_cbar_ax = fig.add_axes((right_margin + 0.055, 0.15, 0.015, 0.70))
            fig.colorbar(val_sm, cax=val_cbar_ax, label=f"In-situ {val_label}")


if TYPE_CHECKING:
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    from .recipe import GeographicBounds

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
# colours). Deliberately avoids any grayscale/achromatic entry (mid-gray
# like tab10's "#7f7f7f", but also pure black/white) — plot_collocation_diagnostics
# uses gray (#808080) to mean "unmatched", so a source landing on a
# near-gray OR colorless palette entry reads as "grayish" regardless of
# exact lightness once alpha-blended (e.g. wind's reduced matched-layer
# alpha=0.65 turns solid black into a mid-gray blend, visually
# indistinguishable in hue from the faint unmatched gray even though the
# two differ in lightness) — this is what made scatterometer's matched
# points look gray once it landed on the (then-)black slot below. Must
# have at least as many entries as _canonical_source_order() returns
# (currently 12, after radiometer_ssm/scatterometer_ssm joined
# LAYER_DATA_TYPES for satellite soil moisture): a shorter palette wraps
# and silently reassigns two unrelated sources (e.g. tidal gauge landing
# back on altimeter's blue circle) — see the "matched tidal gauge and
# altimeter look identical" bug this comment documents. Because slots are
# assigned by *alphabetical position* in the combined canonical set, adding
# a new source name can shift every alphabetically-later source onto a
# different color than before -- this has now broken twice (once when
# radiometer_ssm/scatterometer_ssm were added, moving plain "scatterometer"
# off its original olive slot and onto the black one) - if a new
# LAYER_DATA_TYPES/​_INSITU_TYPES entry is ever added, re-check every
# existing slot for a newly-introduced grayscale collision, not just count.
_SOURCE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
    "#f032e6", "#e6194b", "#000080", "#ffff00",
]

# Marker shapes paired 1:1 with _SOURCE_COLORS by index, used wherever
# validation sources need to stay identifiable independently of color (e.g.
# when color is taken by a continuous value like wind speed or temporal
# offset instead of by source).
_SOURCE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p", "8", "<"]


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


def _set_lonlat_ticks(ax, gl):
    """Cheap plain-matplotlib degree-labeled ticks for a PlateCarree
    GeoAxes — replaces cartopy's gridliner label placement
    (``draw_labels=True``), whose curved-projection label-positioning
    logic is expensive to recompute across many subplots. Only valid for
    rectangular projections (PlateCarree/Mercator), which is all this
    module uses. Note: cartopy's GeoAxes ships with axis visibility disabled
    by default (ax.xaxis.get_visible() == False), so we must re-enable it
    for the formatted ticks to be rendered to canvas.

    ``gl`` is the Gridliner returned by the ``ax.gridlines(...)`` call that
    drew the (unlabeled) grid lines for this axes. Grid *lines* are placed by
    the gridliner's own locator, which is independent of matplotlib's default
    tick auto-locator — for most extents they happen to coincide, but they
    can diverge (different degree spacing), which would misalign the labels
    against the grid lines they're meant to describe. To guarantee labels and
    grid lines never drift apart, we read the gridliner's own locator-chosen
    tick values and apply them explicitly instead of trusting matplotlib's
    independent auto-locator. This is an eager computation — it depends on
    ``ax.get_xlim()``/``ax.get_ylim()`` already reflecting the final data
    extent — so callers must invoke this only after the axes' data (and thus
    autoscale/extent) is finalized.

    Both locators are first capped to a small ``nbins`` (see below) so
    narrow-extent subplots never get an unreadable pile of closely-spaced
    tick labels. ``nbins=2`` (not just a moderate cap like 4) is required
    because cartopy's PlateCarree GeoAxes always renders at equal aspect
    ratio: a WV-mode SAR scene's ground track is many degrees tall in
    latitude but only a few degrees wide in longitude, so the actual
    rendered map occupies a narrow vertical strip inside its subplot box
    regardless of the subplot's nominal (wide) figure width — even 3-4
    tick labels don't fit side by side in that strip without overlapping.

    ``tick_params(length=0)`` suppresses the small perpendicular tick marks
    that ``set_visible(True)`` would otherwise re-enable on the whole Axis
    artist; the original gridliner-only rendering never drew those, so this
    keeps pixel parity with the pre-fix appearance (labels only, no marks)."""
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter  # noqa: PLC0415

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    ax.tick_params(axis="both", which="both", length=0)

    # Cap the locator to a small number of ticks regardless of extent
    # width: the gridliner's default (nbins=8) picks tick counts oblivious
    # to how narrow the subplot's actual extent is. A sub-1°-wide WV-mode
    # SAR scene (narrow longitude, wide latitude) triggers many
    # closely-spaced, high-decimal-precision ticks that pile up into
    # unreadable overlapping labels; a wide overview map also reads more
    # cleanly with fewer ticks. nbins=2 (rather than a milder cap) is what
    # actually stops the overlap for tall/narrow WV-mode extents — see the
    # docstring note above on equal-aspect rendering.
    gl.xlocator.set_params(nbins=2)
    gl.ylocator.set_params(nbins=2)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xticks = [x for x in gl.xlocator.tick_values(*xlim) if xlim[0] <= x <= xlim[1]]
    yticks = [y for y in gl.ylocator.tick_values(*ylim) if ylim[0] <= y <= ylim[1]]
    ax.set_xticks(xticks, crs=ax.projection)
    ax.set_yticks(yticks, crs=ax.projection)


def _pad_extent_to_min_aspect(ax, min_aspect: float = 1.0, bounds=None) -> None:
    """Pad a geographic axes' latitude extent so height/width >= min_aspect.

    Keeps every scene panel in a report portrait-or-square: without this,
    a scene with a small latitude span (e.g. a handful of closely-spaced
    imagettes) renders as a short, wide strip next to otherwise-portrait
    satellite-track panels in the same figure.

    Parameters
    ----------
    bounds : GeographicBounds, optional
        If given, the padded extent is re-clamped to
        ``[bounds.min_lat, bounds.max_lat]`` after padding — otherwise a
        bbox much wider than tall (e.g. a 40x25-degree recipe bbox) gets
        padded past the very bounds it was already clamped to just before
        this function runs, silently showing data outside what was
        requested.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0 or height / width >= min_aspect:
        return
    pad = (width * min_aspect - height) / 2
    y0, y1 = max(y0 - pad, -90.0), min(y1 + pad, 90.0)
    if bounds is not None:
        y0 = max(y0, bounds.min_lat)
        y1 = min(y1, bounds.max_lat)
    if hasattr(ax, "set_extent"):
        ax.set_extent([x0, x1, y0, y1], crs=ax.projection)
    else:
        ax.set_ylim(y0, y1)


def _fill_nan_nearest(a: np.ndarray) -> np.ndarray:
    """
    Fill NaN cells in a 2D array with the value of their nearest finite cell.

    Used to repair geolocation (lon/lat) grids before ``pcolormesh``, which
    rejects non-finite coordinates outright; S1 OCN products commonly carry
    NaN lon/lat at swath-edge/invalid-retrieval cells.
    """
    invalid = ~np.isfinite(a)
    if not invalid.any():
        return a
    idx = ndimage.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def _downsample_grid(arr: np.ndarray, lon2d: np.ndarray, lat2d: np.ndarray, max_dim: int):
    """Stride-decimate a gridded SAR field (and its lon/lat) for display.

    cartopy's ``pcolormesh`` runs every cell through a non-affine CRS
    transform plus antimeridian-wrap interpolation; on a full-resolution
    scene (e.g. CLMS SSM's 4144x6832 = 28M cells) this dominates report
    generation time (~15s per panel measured, vs ~0.2s at ~440K cells) for
    resolution far beyond what a ~150dpi printed page can show. Purely a
    rendering-time decimation -- statistics and every other output still
    use the full-resolution ``collocation_results.nc``/``datatree.nc``.
    """
    ny, nx = arr.shape
    stride = max(1, math.ceil(max(ny, nx) / max_dim))
    if stride == 1:
        return arr, lon2d, lat2d
    return arr[::stride, ::stride], lon2d[::stride, ::stride], lat2d[::stride, ::stride]


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


def _val_units_for_source(collocation_ds, val_source=None):
    """
    Resolve the correct 'units' text for a val_* column, source-aware.

    Returns None if collocation_ds has no `val_units` companion variable
    (the common case: every present source already shares one unit, so
    the caller should fall back to the column's own `attrs['units']`).
    With `val_source` given, returns that source's specific unit text.
    With no `val_source`, returns a single answer only if every present
    row's val_units value is identical -- otherwise None, so the caller
    can show a neutral "varies by source" label instead of picking one
    source's unit arbitrarily.
    """
    if "val_units" not in collocation_ds:
        return None
    if val_source is not None:
        mask = collocation_ds["val_source"].values == val_source
        vals = collocation_ds["val_units"].values[mask]
        return str(vals[0]) if len(vals) else None
    uniq = set(str(v) for v in collocation_ds["val_units"].values)
    return uniq.pop() if len(uniq) == 1 else None


def _labeled_var(collocation_ds, col_name: str, var_code: str, val_source=None) -> str:
    """
    Return ``"<var_code> (<units>)"`` using the CF ``units`` attribute on
    ``col_name`` in *collocation_ds*, or just ``var_code`` if that column
    is absent or carries no ``units`` attribute.

    Units are stamped onto every ``sar_<var>``/``val_<var>`` collocation
    column by ``annotate_collocation_ds`` (see ``_cf_metadata.py``), so
    this works for any variable pair without per-variable special-casing —
    e.g. ``"WSPD"`` → ``"WSPD (m s-1)"``, ``"sarSSM"`` → ``"sarSSM (%)"``.

    For ``val_<var>`` columns whose sources genuinely have different
    native units (see annotate_collocation_ds / val_units companion
    variable), pass ``val_source`` to get that specific source's unit
    text; with no ``val_source``, a neutral "(units vary by source)"
    label is used instead of guessing. Absent a val_units companion
    (every non-soil-moisture recipe), behavior is unchanged from before.
    """
    if col_name.startswith("val_"):
        per_source_units = _val_units_for_source(collocation_ds, val_source)
        if per_source_units is not None:
            return f"{var_code} ({per_source_units})"
        if "val_units" in collocation_ds:
            return f"{var_code} (units vary by source)"

    units = None
    if col_name in collocation_ds:
        units = collocation_ds[col_name].attrs.get("units")
    return f"{var_code} ({units})" if units else var_code


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
    split_when_imbalanced: bool = True,
    force_split: bool = False,
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
    split_when_imbalanced : bool
        When True (default) and one val_source's share of points exceeds
        70% of the total *and* there are at least 2 distinct sources,
        render one small-multiples subplot per source (matching
        plot_residuals' by_source layout) instead of a single shared
        axes -- otherwise a dominant source (e.g. ASCAT's thousands of
        points vs. SMOS's dozens) visually buries every other source.
        Ignored when interactive=True or ax is explicitly provided.
    force_split : bool
        When True, always render the per-source small multiples (as
        split_when_imbalanced would for a >70% dominant source),
        regardless of dominant_share -- e.g. soil moisture forces this
        once a source has been CDF-matched into a different reference
        domain, since piling every source into one shared axes at that
        point is too visually busy even when no single source dominates
        by point count. Ignored when interactive=True or ax is explicitly
        provided, same as split_when_imbalanced.

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

    sources_for_split = df["val_source"].unique().tolist() if "val_source" in df.columns else []
    dominant_share = 0.0
    if len(sources_for_split) >= 2:
        counts = df["val_source"].value_counts()
        dominant_share = float(counts.max()) / float(counts.sum())

    # Resolved here (rather than only below, in the single-shared-axes
    # path) so the split branch immediately below can forward the caller's
    # actual color_by intent into small multiples instead of silently
    # dropping it -- previously, whenever imbalance/force_split triggered
    # a split, color_by="temporal_offset" was ignored entirely and
    # _plot_scatter_small_multiples always rendered the plain by-source
    # view, producing a page that duplicated the main scatter instead of
    # showing temporal-offset coloring. Guarded by `not interactive`
    # because the interactive/plotly branch below ignores color_by
    # entirely already and must not emit this warning for it.
    color_by_offset = color_by == "temporal_offset"
    if not interactive and color_by_offset and "temporal_distance_minutes" not in df.columns:
        warnings.warn(
            "color_by='temporal_offset' requested but collocation_ds has no "
            "'temporal_distance_minutes' column; falling back to color_by='source'."
        )
        color_by_offset = False

    if (
        not interactive and ax is None and split_when_imbalanced
        and len(sources_for_split) >= 2 and (dominant_share > 0.7 or force_split)
    ):
        return _plot_scatter_small_multiples(
            df, sar_col, val_col, sar_var, val_var, collocation_ds,
            color_by="temporal_offset" if color_by_offset else "source",
        )

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
    style = _source_style_map(sources)

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
    vmin, vmax = _pad_degenerate_range(float(np.nanmin(all_vals)), float(np.nanmax(all_vals)))
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

    ax.set_xlabel(_labeled_var(collocation_ds, val_col, val_var))
    ax.set_ylabel(_labeled_var(collocation_ds, sar_col, sar_var))
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
            handles = _source_marker_handles(
                ((src, style[src][1]) for src in sorted(sources)), markersize=6,
            )
            handles.append(line11)
            ax.legend(handles=handles, fontsize=7, framealpha=0.7)
    elif by_source:
        ax.legend(fontsize=7, framealpha=0.7)

    fig.tight_layout()
    return fig


def _plot_scatter_small_multiples(
    df, sar_col, val_col, sar_var, val_var, collocation_ds, *, color_by: str = "source",
):
    """One scatter subplot per val_source -- used by plot_scatter when one
    source's point count dominates enough to visually bury the others in
    a single shared axes (see split_when_imbalanced), or when force_split
    requests a split regardless of imbalance.

    color_by mirrors plot_scatter's own parameter: ``"source"`` (default)
    colors each subplot with that source's stable color; ``"temporal_offset"``
    colors by temporal_distance_minutes instead, sharing one color scale
    (and one colorbar) across every subplot so offsets stay comparable
    source to source -- previously this parameter was silently dropped
    whenever a split was triggered, producing a duplicate of the plain
    by-source view under a "colored by temporal offset" title."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sources = sorted(df["val_source"].unique())
    color_map = _source_color_map(sources)
    ncols = 2
    nrows = math.ceil(len(sources) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False)

    color_by_offset = color_by == "temporal_offset" and "temporal_distance_minutes" in df.columns
    offset_vmin = offset_vmax = None
    last_offset_sm = None
    if color_by_offset:
        offset_vmin = float(df["temporal_distance_minutes"].min())
        offset_vmax = float(df["temporal_distance_minutes"].max())

    for idx, src in enumerate(sources):
        r, c = divmod(idx, ncols)
        sub_ax = axes[r][c]
        sub = df[df["val_source"] == src]
        if color_by_offset:
            last_offset_sm = sub_ax.scatter(
                sub[val_col], sub[sar_col], s=18, alpha=0.7,
                c=sub["temporal_distance_minutes"], cmap="plasma",
                vmin=offset_vmin, vmax=offset_vmax, rasterized=True,
            )
        else:
            sub_ax.scatter(sub[val_col], sub[sar_col], s=18, alpha=0.6,
                            color=color_map[src], rasterized=True)
        sub_ax.set_xlabel(_labeled_var(collocation_ds, val_col, val_var, val_source=src))
        sub_ax.set_ylabel(_labeled_var(collocation_ds, sar_col, sar_var))
        sub_ax.set_title(f"{src} (N={len(sub)})", fontsize=9)
        sub_ax.grid(True, linewidth=0.4)

        sub_vals = np.concatenate([sub[val_col].values, sub[sar_col].values])
        sub_vmin, sub_vmax = _pad_degenerate_range(float(np.nanmin(sub_vals)), float(np.nanmax(sub_vals)))
        sub_ax.plot([sub_vmin, sub_vmax], [sub_vmin, sub_vmax], "k--", linewidth=1, label="1:1")

    visible_axes = [axes[idx // ncols][idx % ncols] for idx in range(len(sources))]
    for idx in range(len(sources), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"{sar_var} vs {val_var}")
    # tight_layout before the colorbar -- fig.colorbar(ax=<list>) creates a
    # standalone Axes outside the subplot grid, which tight_layout can't
    # account for (and warns about) if it already exists.
    fig.tight_layout()
    if color_by_offset and last_offset_sm is not None:
        fig.colorbar(last_offset_sm, ax=visible_axes, label="Temporal offset (min)", shrink=0.6)
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
    point_size: Union[int, Dict[str, int]] = 40,
    max_points_per_panel: int = 4000,
    max_raster_dim: int = 1200,
    split_by: str = "collocation_type",
    scenes: Optional[Sequence[str]] = None,
    interactive: bool = False,
    geographic_bounds: Optional["GeographicBounds"] = None,
    two_column_by_type: bool = False,
    skip_domain_harmonization: bool = False,
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
    point_size : int or dict[str, int]
        Scatter marker size in points² (matplotlib ``s`` argument). Pass a
        dict keyed by ``collocation_type`` (e.g.
        ``{"point_vs_layer": 15, "layer_vs_layer": 5}``) to size each type
        independently -- a pair combining sparse in-situ points with dense
        scatterometer/radiometer coverage (e.g. soil moisture's ISMN vs.
        ASCAT/SMAP/SMOS) needs different sizes per type; one shared size
        computed from the pooled point count is dominated by whichever
        type has more points. A ``collocation_type`` value missing from
        the dict falls back to 40.
    max_points_per_panel : int
        If a scene's deduplicated point count exceeds this, points are
        randomly subsampled (fixed seed, for reproducible figures) for
        *this plot only* — statistics and every other output still use
        the full dataset. Keeps individual markers distinguishable in
        very dense scenes (soil moisture's satellite sources can produce
        thousands of matched points per scene) instead of a solid blob.
    max_raster_dim : int
        Gridded SAR background fields (e.g. CLMS SSM's ~4144x6832 CEURO
        tiles) are stride-decimated to at most this many pixels per axis
        before ``pcolormesh`` — full resolution is invisible at print size
        and cartopy's non-affine transform is ~O(cells), so an un-decimated
        multi-million-cell scene can dominate report generation time
        (measured ~15s/panel at full res vs ~0.2s decimated). Statistics
        and every other output still use the full-resolution data.
    split_by : str or None
        Variable / coordinate to split collocations into separate figures.
        Default ``"collocation_type"`` creates one figure for in-situ
        (``point_vs_layer``) and one for scatterometer (``layer_vs_layer``).
        Pass ``None`` for a single combined figure.
    interactive : bool
        Return a folium Map instead of matplotlib.
    geographic_bounds : GeographicBounds, optional
        Clamp each static (non-interactive) subplot's extent to the
        recipe's requested bounding box instead of the SAR field's full
        native extent — e.g. CLMS Surface Soil Moisture's grid covers all
        of mainland Europe regardless of what a recipe actually requested,
        so without this every scene panel shows far more than was asked
        for. Ignored when the bounding box itself crosses the antimeridian
        (``min_lon > max_lon``) and the scene's own projection does not,
        since a plain (non-recentred) axes can't cleanly represent that
        span — the scene keeps its native extent in that case.
    two_column_by_type : bool
        When True and split_by == "collocation_type": instead of one
        Figure per collocation_type (each containing a grid of every
        scene), build one Figure *per scene*, with point_vs_layer drawn
        in the left column and layer_vs_layer in the right (falling back
        to a single column if a scene only has one of the two types).
        Returns dict[scene_name, Figure] in this mode instead of
        dict[collocation_type, Figure]. Ignored (no-op) unless split_by
        == "collocation_type".
    skip_domain_harmonization : bool
        When True, force ``domains_differ`` to False unconditionally,
        skipping the whole SAR-vs-validation units-mismatch detection and
        any associated field-transform fitting / point-level
        harmonization. Intended for callers (e.g. validation_report's
        native-units section) that have already row-filtered
        *collocation_ds* down to val_source groups that share SAR's own
        units family — in that case, *every* source actually present is
        already domain-compatible with SAR by construction, even though
        the val_<var> column's own ``units`` attrs may still carry a
        stale "mixed — see val_units" sentinel inherited from the
        original, unfiltered dataset (row-filtering doesn't recompute
        column-level attrs). Relying on that stale string would
        incorrectly trigger harmonization/two-colorbar fallback for a
        case that needs neither.

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
        HAS_CARTOPY = True
    except ImportError:
        HAS_CARTOPY = False
        warnings.warn(
            "cartopy is not installed — falling back to plain matplotlib axes.",
            UserWarning, stacklevel=2,
        )

    import matplotlib.cm as mcm  # noqa: PLC0415
    import matplotlib.colors as mcolors  # noqa: PLC0415
    import matplotlib.lines as mlines  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    finite_v = np.array([])
    if val_col_present:
        finite_v = collocation_ds[val_col].values
        finite_v = finite_v[np.isfinite(finite_v)]

    sar_units = None
    if sar_var in sar_node[scene_names[0]].to_dataset().variables if scene_names else False:
        sar_units = sar_node[scene_names[0]].to_dataset()[sar_var].attrs.get("units")
    val_units = collocation_ds[val_col].attrs.get("units") if val_col_present else None
    domains_differ = (
        not skip_domain_harmonization
        and val_col_present and sar_units is not None and val_units is not None
        and sar_units != val_units
    )

    # Point-level rendering dataset. Defaults to the raw, un-harmonized
    # collocation_ds (unchanged behavior); replaced below with a
    # harmonized copy when domains_differ, so that point markers agree
    # with what the field/statistics sections show — e.g. omitting
    # entirely a val_source that _harmonize_percent_domain_sources
    # couldn't harmonize (not enough reference-source data), instead of
    # plotting its still-raw values under a colour scale calibrated for
    # the harmonized domain. Deliberately a *separate* variable from
    # collocation_ds itself (never reassigned) -- fit_sar_to_val_transform
    # below still needs the true raw collocation_ds to pair a raw sar_col
    # with the harmonized val_col for its own internal harmonize call;
    # handing it an already-harmonized collocation_ds would harmonize a
    # second time (by val_source label, not by value) and corrupt
    # already-converted sources.
    point_collocation_ds = collocation_ds

    # When the SAR field and validation series live in different physical
    # domains (e.g. soil_moisture: SAR's relative saturation index in "%"
    # vs. ISMN's volumetric fraction in "1"), convert the SAR *field*
    # itself into the validation domain before plotting, so the whole map
    # (background + points) shares one meaningful colour scale — rather
    # than showing two separate colorbars, which is confusing to read at
    # a glance. This uses a single CDF-matching transform fit once from
    # every collocated pair (pooled across groups, for display purposes
    # only — statistics still use add_rescaled_sar_column's per-group
    # rescaling) and applied to every scene's full grid, not just
    # collocated pixels. Falls back to two separate colorbars (one per
    # layer's own percentile range) if there isn't enough collocated data
    # to fit a transform.
    sar_field_transform = None
    if domains_differ:
        assert val_var is not None   # implied by domains_differ requiring val_col_present
        from .statistics import (  # noqa: PLC0415
            _harmonize_percent_domain_sources,
            fit_sar_to_val_transform,
        )
        sar_field_transform = fit_sar_to_val_transform(collocation_ds, sar_var, val_var)

        # Harmonize the *points*, independently of whether the field
        # transform above succeeded: a val_source that couldn't be
        # harmonized (e.g. ISMN too sparse to fit a reference transform)
        # is identified via dropped_sources and removed entirely below --
        # at real-world scale a dropped source can be the majority of all
        # collocated points (e.g. ASCAT ~56% of a run), so treating it as
        # per-point "No data" hatching would flood the map and bury the
        # sources that DID harmonize successfully. Dropped sources remain
        # fully visible in the separate native-units section.
        point_collocation_ds, _converted, dropped_sources = _harmonize_percent_domain_sources(
            collocation_ds, sar_var, val_var,
        )
        if dropped_sources and "val_source" in point_collocation_ds:
            point_collocation_ds = point_collocation_ds.where(
                ~point_collocation_ds["val_source"].isin(list(dropped_sources)), drop=True,
            )
        finite_v = point_collocation_ds[val_col].values
        finite_v = finite_v[np.isfinite(finite_v)]

        if sar_field_transform is None:
            logger.warning(
                "plot_geographic: could not fit a SAR-to-validation-domain "
                "transform (not enough collocated pairs) — falling back to "
                "two separate colorbars for %s vs %s.", sar_var, val_var,
            )

    # Loaded once per scene here and reused below (colour-limit pooling,
    # antimeridian detection, and every _draw_scene_panel call) instead of
    # re-running ``.to_dataset()`` — a netCDF read — for each: with
    # two_column_by_type each scene is drawn twice (point_vs_layer +
    # layer_vs_layer columns), and the repeated reads were a measured,
    # avoidable contributor to report generation time on multi-scene,
    # multi-source (soil moisture) recipes.
    scene_ds_cache: Dict[str, "xr.Dataset"] = {
        scene_name: sar_node[scene_name].to_dataset() for scene_name in scene_names
    }

    # Colour limits — pooled from the SAR field *and* the validation values
    # so both layers share one scale and are directly comparable by
    # colour (same across all figures/groups): the default behaviour, and
    # also correct once the field above has been converted into the
    # validation domain.
    all_field_vals = []
    for scene_name in scene_names:
        arr = _sar_field(scene_ds_cache[scene_name], sar_var)
        if arr is not None:
            if sar_field_transform is not None:
                arr = sar_field_transform(arr)
            all_field_vals.append(arr[np.isfinite(arr)])
    flat = np.concatenate(all_field_vals) if all_field_vals else np.array([])

    from ._variable_map import CIRCULAR_VAL_VARS  # noqa: PLC0415
    is_circular = val_var in CIRCULAR_VAL_VARS

    # Only genuinely un-convertible domain mismatches still need separate
    # ranges/colorbars below.
    needs_separate_scale = domains_differ and sar_field_transform is None

    # Circular variables (e.g. WDIR) skip percentile pooling: 0-360 is a
    # fixed, physically meaningful range, and percentile-clamping a value
    # that wraps at the 0/360 seam would be actively wrong.
    if is_circular:
        cmap = "twilight"
        vmin, vmax = 0.0, 360.0
    elif needs_separate_scale:
        vmin = float(np.nanpercentile(flat, 2)) if len(flat) else 0.0
        vmax = float(np.nanpercentile(flat, 98)) if len(flat) else 1.0
    else:
        pooled = np.concatenate([flat, finite_v]) if len(flat) or len(finite_v) else np.array([0.0, 1.0])
        vmin = float(np.nanpercentile(pooled, 2))
        vmax = float(np.nanpercentile(pooled, 98))

    effective_val_cmap = "twilight" if is_circular else (val_cmap if val_cmap is not None else cmap)
    val_cmap = effective_val_cmap
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sar_sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sar_sm.set_array([])
    sar_norm = norm

    val_norm = val_sm = None
    if len(finite_v) > 0:
        if needs_separate_scale:
            val_vmin = float(np.nanpercentile(finite_v, 2))
            val_vmax = float(np.nanpercentile(finite_v, 98))
            if val_vmin == val_vmax:
                val_vmin -= 0.5
                val_vmax += 0.5
            val_norm = mcolors.Normalize(vmin=val_vmin, vmax=val_vmax)
        else:
            val_norm = norm
        val_sm = mcm.ScalarMappable(cmap=effective_val_cmap, norm=val_norm)
        val_sm.set_array([])

    # One shared colorbar when both layers use the same palette+scale
    # (the default, and also true once a units mismatch was resolved by
    # converting the field above); two if the caller opted into a
    # distinct val_cmap, or if a units mismatch couldn't be converted.
    single_colorbar = val_sm is not None and effective_val_cmap == cmap and not needs_separate_scale
    right_margin = 0.88 if (val_sm is None or single_colorbar) else 0.80

    # Per-scene antimeridian detection: a scene "crosses" when its raw
    # (unshifted, [-180, 180]) lon coordinate spans more than 180 degrees —
    # e.g. an imagette with points at 179E and 179W. Such a scene must get
    # its own central_longitude=180 axes, otherwise a *shared*
    # central_longitude=0 projection autoscales it to a full [-180, 180]
    # world map with the swath split across both edges (this is a distinct
    # bug from the one already fixed in plot_collocation_diagnostics — that
    # function draws one map for the whole recipe bbox, this one draws one
    # subplot per scene). Scenes without lon/lat coords count as
    # non-crossing (the "no coords" branch below hides the axes anyway).
    scene_crosses_dateline: Dict[str, bool] = {}
    for scene_name in scene_names:
        crosses = False
        scene_ds_for_check = scene_ds_cache[scene_name]
        if "lon" in scene_ds_for_check.coords:
            lon_vals = np.asarray(scene_ds_for_check["lon"].values)
            finite_lon = lon_vals[np.isfinite(lon_vals)]
            if finite_lon.size:
                crosses = bool(finite_lon.max() - finite_lon.min() > 180)
        scene_crosses_dateline[scene_name] = crosses

    def _scene_projection(scene_name):
        if not HAS_CARTOPY:
            return None
        if scene_crosses_dateline.get(scene_name):
            return ccrs.PlateCarree(central_longitude=180)
        return ccrs.PlateCarree()

    def _resolve_point_size(group_label) -> int:
        """point_size may be a plain int (uniform) or a dict keyed by
        collocation_type (per-type sizing, see plot_geographic's
        docstring) -- resolve it to the scalar this call's group actually
        uses. group_label is the collocation_type value for a group
        already restricted to one type (the only case a dict is
        meaningful for); falls back to 40 for an unlisted type."""
        if isinstance(point_size, dict):
            return point_size.get(group_label, 40)
        return point_size

    def _draw_scene_panel(ax, scene_name, group_coll_ds, pt_size):
        """Draw one SAR scene's field + collocated validation points into
        *ax*. Extracted from _build_figure's per-scene loop (behavior
        unchanged) so both the by-group figure layout (_build_figure) and
        the two-column by-collocation-type layout
        (_build_scene_pair_figure, added for soil_moisture) can share it."""
        scene_ds = scene_ds_cache[scene_name]

        if "lon" not in scene_ds.coords or "lat" not in scene_ds.coords:
            ax.set_visible(False)
            return

        if HAS_CARTOPY:
            land, coastline = _land_coastline_features()
            ax.add_feature(land, facecolor="lightgray", zorder=0, rasterized=True)
            ax.add_feature(coastline, linewidth=0.5, zorder=0, rasterized=True)
            gl = ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5)
            transform = ccrs.PlateCarree()
        else:
            transform = None

        arr = _sar_field(scene_ds, sar_var)
        if arr is not None:
            if sar_field_transform is not None:
                arr = sar_field_transform(arr)
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
                # Gridded data (e.g., IW/EW mode) — use pcolormesh.
                # pcolormesh rejects non-finite x/y, so repair NaN
                # geolocation cells (common at swath edges) via
                # nearest-neighbour fill and mask the corresponding data.
                lon2d = scene_ds["lon"].values
                lat2d = scene_ds["lat"].values
                invalid_xy = ~(np.isfinite(lon2d) & np.isfinite(lat2d))
                if invalid_xy.any():
                    arr = np.where(invalid_xy, np.nan, arr)
                    lon2d = _fill_nan_nearest(lon2d)
                    lat2d = _fill_nan_nearest(lat2d)
                arr, lon2d, lat2d = _downsample_grid(arr, lon2d, lat2d, max_raster_dim)
                ax.pcolormesh(
                    lon2d, lat2d, np.ma.masked_invalid(arr),
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

            # Subsample if point count exceeds max_points_per_panel
            if len(df_pts) > max_points_per_panel:
                df_pts = df_pts.sample(n=max_points_per_panel, random_state=0)

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
                            marker=marker, s=pt_size,
                            edgecolors="black", linewidths=0.4,
                            rasterized=True, **kw_sc,
                        )
                elif len(valid_pts):
                    ax.scatter(
                        valid_pts["val_lon"], valid_pts["val_lat"],
                        c=valid_pts[val_col], cmap=val_cmap, norm=val_norm,
                        s=pt_size, edgecolors="black", linewidths=0.4,
                        rasterized=True, **kw_sc,
                    )
                if len(nan_pts):
                    # No retrieved value at this location/time — mark it
                    # clearly (gray + hatch) instead of leaving an
                    # invisible gap that looks like "no observation here".
                    ax.scatter(
                        nan_pts["val_lon"], nan_pts["val_lat"],
                        s=pt_size, facecolor="lightgray", edgecolors="dimgray",
                        linewidths=0.6, hatch="////", rasterized=True, **kw_sc,
                    )

                # Fill color varies continuously with the validation
                # value here (shared with the SAR colorbar), so a solid
                # legend swatch would misrepresent what's on the map —
                # marker shape is the discriminator instead.
                handles = []
                if "val_source" in valid_pts.columns:
                    # Built from valid_pts (rows actually drawn with their
                    # real colored marker), not df_pts (every row,
                    # including ones with a genuine per-point NaN, drawn
                    # instead as the generic gray hatched "no data" marker
                    # below) -- otherwise a source with zero actually-
                    # colored points (e.g. every one of its retrievals
                    # happening to be missing at these particular
                    # locations/times) would still show its normal marker
                    # in the legend even though nothing of that source
                    # exists on the map. Note: a source dropped *entirely*
                    # by _harmonize_percent_domain_sources (e.g. ASCAT when
                    # ISMN is too sparse to harmonize against) never
                    # reaches this function at all -- point_collocation_ds
                    # above already excludes it, rows and all.
                    present = set(valid_pts["val_source"].astype(str))
                    handles += _source_marker_handles(
                        ((s, mkr) for s, (_, mkr) in source_style.items() if s in present),
                        markersize=5,
                    )
                if len(nan_pts):
                    handles.append(
                        mlines.Line2D([], [], marker="o", linestyle="None",
                                      markerfacecolor="lightgray", markeredgecolor="dimgray",
                                      markersize=5, label="No data (NaN)")
                    )
                # loc="best" on a GeoAxes is catastrophically slow: matplotlib's
                # placement search runs every plotted artist's vertices
                # (including the whole-Europe SAR background field) through
                # cartopy's non-affine projection transform once per candidate
                # position — observed to hang for minutes with multi-GB memory
                # growth on a real recipe. A fixed corner is effectively free.
                if handles:
                    ax.legend(handles=handles, fontsize=6,
                              loc="upper right", framealpha=0.7)
            elif "val_source" in df_pts.columns:
                for src, grp in df_pts.groupby("val_source"):
                    color, marker = source_style.get(str(src), ("#ff0000", "o"))
                    ax.scatter(grp["val_lon"], grp["val_lat"],
                               s=pt_size, c=color, marker=marker,
                               edgecolors="black", linewidths=0.4,
                               label=str(src), rasterized=True, **kw_sc)
                ax.legend(fontsize=6, loc="upper right", framealpha=0.7)
            else:
                ax.scatter(df_pts["val_lon"], df_pts["val_lat"],
                           s=pt_size, c="#ff7f0e",
                           edgecolors="black", linewidths=0.4, **kw_sc)

        n_dedup = len(df_pts) if n_pts > 0 else 0
        ax.set_title(
            f"{scene_name.split('/')[-1]}  ({n_dedup} obs)", fontsize=8
        )
        # Clamp to the recipe's requested bbox instead of the SAR
        # field's full native extent (e.g. CLMS SSM's grid covers all
        # of mainland Europe regardless of what was actually
        # requested). Applied before _pad_extent_to_min_aspect so the
        # aspect padding below operates on the clamped box, not the
        # unclamped one.
        bounds_applied = (
            geographic_bounds is not None
            and geographic_bounds.min_lon <= geographic_bounds.max_lon
            and not scene_crosses_dateline.get(scene_name)
        )
        if bounds_applied:
            assert geographic_bounds is not None  # implied by bounds_applied
            if HAS_CARTOPY:
                ax.set_extent(
                    [geographic_bounds.min_lon, geographic_bounds.max_lon,
                     geographic_bounds.min_lat, geographic_bounds.max_lat],
                    crs=transform,
                )
            else:
                ax.set_xlim(geographic_bounds.min_lon, geographic_bounds.max_lon)
                ax.set_ylim(geographic_bounds.min_lat, geographic_bounds.max_lat)
        # Also deferred until now, for the same reason as _set_lonlat_ticks
        # below: it needs the finalized autoscaled extent from
        # ax.get_xlim()/get_ylim(). Applies to both the HAS_CARTOPY
        # (GeoAxes) and plain-matplotlib fallback path — the helper itself
        # branches on hasattr(ax, "set_extent").
        _pad_extent_to_min_aspect(ax, bounds=geographic_bounds if bounds_applied else None)
        if HAS_CARTOPY:
            # Deferred until now (rather than right after ax.gridlines above):
            # this reads the finalized data extent via ax.get_xlim()/get_ylim(),
            # which only reflects this scene's plotted data after the
            # pcolormesh/scatter calls above have run their autoscale.
            _set_lonlat_ticks(ax, gl)

    def _build_scene_pair_figure(scene_name):
        """Build one two-column Figure for *scene_name*: point_vs_layer
        collocations on the left, layer_vs_layer on the right (or a single
        column if only one type has data for this scene)."""
        type_datasets = []
        for ctype in ("point_vs_layer", "layer_vs_layer"):
            if "collocation_type" not in point_collocation_ds:
                continue
            type_mask = (
                (point_collocation_ds["collocation_type"] == ctype)
                & (point_collocation_ds["sar_scene_name"] == scene_name)
            )
            type_ds = point_collocation_ds.isel(collocation=type_mask)
            if type_ds.sizes.get("collocation", 0) > 0:
                type_datasets.append((ctype, type_ds))

        if not type_datasets:
            return None

        n_cols = len(type_datasets)
        fig = plt.figure(figsize=(9 * n_cols, 7))
        for i, (ctype, type_ds) in enumerate(type_datasets):
            ax = fig.add_subplot(
                1, n_cols, i + 1,
                **({"projection": _scene_projection(scene_name)} if HAS_CARTOPY else {}),
            )
            _draw_scene_panel(ax, scene_name, type_ds, _resolve_point_size(ctype))
            ax.set_title(f"{ax.get_title()}  [{ctype}]", fontsize=8)

        fig.suptitle(
            f"SAR {sar_var} vs. {val_var}  [{scene_name.split('/')[-1]}]"
            " — collocated observations", fontsize=11, y=1.02,
        )
        fig.tight_layout()

        # _build_scene_pair_figure historically never drew a colorbar at
        # all -- structurally absent, not intermittent. Add it now, reusing
        # the exact same shared/two-colorbar objects _build_figure already
        # computes once for the whole plot_geographic call.
        _draw_colorbar(
            fig, right_margin, sar_sm, val_sm, single_colorbar,
            collocation_ds, sar_var, val_var, sar_field_transform,
        )

        return fig

    def _build_figure(group_coll_ds, group_label):
        """Build one Figure for a sub-set of collocations."""
        nrows = math.ceil(len(scene_names) / ncols)
        fig = plt.figure(figsize=(9 * ncols, 7 * nrows))
        axes = [
            [
                fig.add_subplot(
                    nrows, ncols, r * ncols + c + 1,
                    **({"projection": _scene_projection(scene_names[r * ncols + c])}
                       if HAS_CARTOPY and (r * ncols + c) < len(scene_names) else {}),
                )
                for c in range(ncols)
            ]
            for r in range(nrows)
        ]

        pt_size_for_group = _resolve_point_size(group_label)
        for idx, scene_name in enumerate(scene_names):
            r, c = divmod(idx, ncols)
            _draw_scene_panel(axes[r][c], scene_name, group_coll_ds, pt_size_for_group)

        # Hide unused axes
        for idx in range(len(scene_names), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        # Once the field has been converted into the validation domain,
        # label it with the validation column's units, not the SAR
        # column's original ones — matching what's actually displayed.
        _draw_colorbar(
            fig, right_margin, sar_sm, val_sm, single_colorbar,
            collocation_ds, sar_var, val_var, sar_field_transform,
        )

        title = f"SAR {sar_var}"
        if val_var:
            title += f" vs. {val_var}"
        if group_label is not None:
            title += f"  [{group_label}]"
        fig.suptitle(title + " — collocated observations", fontsize=11, y=1.01)
        return fig

    # ── Build figures ────────────────────────────────────────────────────────
    if two_column_by_type and split_by == "collocation_type":
        scene_figures: Dict[str, object] = {}
        for scene_name in scene_names:
            fig = _build_scene_pair_figure(scene_name)
            if fig is not None:
                scene_figures[scene_name] = fig
        return scene_figures

    if group_values is None:
        return _build_figure(point_collocation_ds, None)

    figures: Dict[str, object] = {}
    for gv in group_values:
        if split_by in point_collocation_ds:
            mask = point_collocation_ds[split_by] == gv
        else:
            mask = point_collocation_ds.coords[split_by] == gv
        group_ds = point_collocation_ds.isel(collocation=mask)
        if group_ds.sizes.get("collocation", 0) == 0:
            continue
        figures[gv] = _build_figure(group_ds, gv)
    return figures


# ---------------------------------------------------------------------------
# 3. Statistics bar chart
# ---------------------------------------------------------------------------

def plot_summary_table(
    stats_ds,
    metrics: Optional[List[str]] = None,
):
    """
    Table of bias/RMSE/correlation (one row per validation source), meant
    to precede plot_statistics' bar charts in the report so a reader gets
    the exact numbers before the visual comparison.

    Parameters
    ----------
    stats_ds : xr.Dataset
        Output of :func:`~.statistics.compute_statistics` -- the same
        Dataset plot_statistics already consumes.
    metrics : list[str], optional
        Which metrics become columns. Defaults to
        ``["bias", "rmse", "correlation"]``, matching plot_statistics'
        own default.

    Returns
    -------
    matplotlib.figure.Figure or None
        None if none of *metrics* are present in *stats_ds*.
    """
    if metrics is None:
        metrics = ["bias", "rmse", "correlation"]

    available = [m for m in metrics if m in stats_ds]
    if not available:
        warnings.warn(f"None of {metrics} found in stats_ds. Available: {list(stats_ds.data_vars)}")
        return None

    import matplotlib.pyplot as plt  # noqa: PLC0415

    sources = stats_ds["source"].values.tolist()
    if not sources:
        warnings.warn("stats_ds has no source groups (empty 'source' coordinate); nothing to tabulate.")
        return None

    sar_var = stats_ds.attrs.get("sar_var", "")
    val_var = stats_ds.attrs.get("val_var", "")

    cell_text = [
        [f"{float(stats_ds[m].values[i]):.4f}" for m in available]
        for i in range(len(sources))
    ]

    fig, ax = plt.subplots(figsize=(2.2 * len(available) + 2, 0.6 * len(sources) + 1.5))
    ax.axis("off")
    ax.set_title(f"Validation summary: {sar_var} vs {val_var}", fontsize=11, fontweight="bold")
    table = ax.table(
        cellText=cell_text,
        rowLabels=sources,
        colLabels=[m.replace("_", " ").title() for m in available],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    fig.tight_layout()
    return fig


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
    hist_range: Optional[Union[Tuple[float, float], Dict[str, Tuple[float, float]]]] = None,
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
        Draw one subplot per ``val_source`` ("small multiples"), each with
        its own y-axis but a shared x-range — so a source with a very
        narrow residual spread (e.g. two tightly-clustered points) can't
        produce a density spike that dwarfs every other source's bars, the
        way it would sharing one axes. When False, draw a single combined
        histogram instead (``ax`` honored in this case only).
    interactive : bool
        Return a plotly Figure instead of matplotlib.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into (static, ``by_source=False`` only — the
        small-multiples grid always creates its own figure).
    hist_range : tuple[float, float] or dict[str, tuple[float, float]], optional
        Fixed (min, max) passed to the underlying histogram's ``range=``
        and to ``set_xlim``, overriding the default data-driven range.
        For CDF-matched pairs (e.g. soil moisture), a rare CDF-matching
        edge artifact can produce one residual far outside the real
        distribution's spread, ballooning the data-driven range until
        the real distribution collapses into a single bar -- passing a
        fixed range excludes such outliers from the density calculation
        (not just the visible window), since ``range=`` drops
        out-of-range values before binning.

        A plain tuple applies uniformly to every source (``by_source=True``)
        or to the single combined histogram (``by_source=False``). A
        ``dict`` (``by_source=True`` only) maps ``val_source`` to its own
        override range -- sources absent from the dict fall back to
        their OWN per-source data-driven range, not one pooled across
        every source. This is useful whenever a caller's ``pair_ds`` mixes
        validation sources whose residuals genuinely live on different
        scales (e.g. different unit families, or different instrument
        noise floors) -- a single shared range across all of them would
        only be meaningful within one such group; giving each source its
        own range keeps a tight source's panel from inheriting a wider
        source's spread just because they share one pair. (For the
        soil-moisture CDF-matched pairs this module builds, every source
        is harmonized into one common domain before ``plot_residuals``
        is called -- see ``_volumetric_hist_range_overrides`` -- so in
        that specific case the dict ends up mapping every source to the
        same fixed range; the dict form itself remains general-purpose
        for other callers with genuinely inconsistent per-source domains.)

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

    # A residual's units are whatever domain both sides share (val's, for
    # a soil-moisture pair after add_rescaled_sar_column; the same domain
    # by construction for every other existing variable pair).
    def _residual_label(val_source=None):
        units = _val_units_for_source(collocation_ds, val_source)
        if units is not None:
            return f"{sar_var} − {val_var} ({units})"
        if "val_units" in collocation_ds:
            return f"{sar_var} − {val_var} (units vary by source)"
        units = collocation_ds[val_col].attrs.get("units")
        return f"{sar_var} − {val_var}" + (f" ({units})" if units else "")

    residual_label = _residual_label()

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
            labels={"residual": residual_label, "val_source": "Source"},
            title=title,
        )
        return fig

    import matplotlib.pyplot as plt  # noqa: PLC0415

    if not by_source:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = ax.get_figure()
        ax.hist(df["residual"].dropna(), bins=30, range=hist_range, density=True, alpha=0.7, color="#1f77b4")
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel(residual_label)
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.grid(True, linewidth=0.4)
        if hist_range is not None:
            ax.set_xlim(*hist_range)
        fig.tight_layout()
        return fig

    # by_source=True: one subplot per source, sharing a common x-range but
    # each with its own y-axis (see docstring for why a shared axes breaks
    # under density=True when spreads differ wildly across sources).
    sources = sorted(df["val_source"].unique())
    color_map = _source_color_map(sources)
    if hist_range is None:
        residual_min = float(df["residual"].min())
        residual_max = float(df["residual"].max())
        if residual_min == residual_max:
            residual_min -= 0.5
            residual_max += 0.5
        shared_range = (residual_min, residual_max)
    elif not isinstance(hist_range, dict):
        shared_range = hist_range
    else:
        shared_range = None  # resolved per-source below

    ncols = 2
    nrows = math.ceil(len(sources) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for idx, src in enumerate(sources):
        r, c = divmod(idx, ncols)
        sub_ax = axes[r][c]
        sub = df.loc[df["val_source"] == src, "residual"].dropna()

        if isinstance(hist_range, dict):
            if src in hist_range:
                src_range = hist_range[src]
            else:
                # No override for this source -- its OWN data-driven
                # range, never the other sources' pooled data (that's
                # exactly the bug this dict form exists to avoid).
                src_min, src_max = float(sub.min()), float(sub.max())
                if src_min == src_max:
                    src_min -= 0.5
                    src_max += 0.5
                src_range = (src_min, src_max)
        else:
            assert shared_range is not None  # only unset when hist_range is a dict
            src_range = shared_range

        sub_ax.hist(sub, bins=30, range=src_range, density=True, alpha=0.7, color=color_map[src])
        if hist_range is not None:
            sub_ax.set_xlim(*src_range)
        sub_ax.axvline(0, color="black", linewidth=1, linestyle="--")
        sub_ax.set_xlabel(_residual_label(val_source=src))
        sub_ax.set_ylabel("Density")
        sub_ax.set_title(f"{src} (N={len(sub)})", fontsize=9)
        sub_ax.grid(True, linewidth=0.4)

    for idx in range(len(sources), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(title)
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

    _units_for_label = _val_units_for_source(collocation_ds)
    if _units_for_label is not None:
        residual_label = f"|{sar_var} − {val_var}| ({_units_for_label})"
    elif "val_units" in collocation_ds:
        residual_label = f"|{sar_var} − {val_var}| (units vary by source)"
    else:
        val_units = collocation_ds[val_col].attrs.get("units")
        residual_label = f"|{sar_var} − {val_var}|" + (f" ({val_units})" if val_units else "")

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
                "abs_residual": residual_label,
                "val_source": "Source",
            },
            title=f"{residual_label} vs. temporal offset",
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
    ax.set_ylabel(residual_label)
    ax.set_title(f"{sar_var} vs {val_var} — residual magnitude vs. temporal offset")
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4b. Collocation diagnostics plot
# ---------------------------------------------------------------------------

def _downsampled_valid_pixel_coords(
    valid_mask: np.ndarray, lons: np.ndarray, lats: np.ndarray, target_count: int = 3000,
) -> list:
    """
    Return (lon, lat) centers of a strided subsample of *valid_mask*'s True
    cells, capped at roughly *target_count* points.

    Cheap enough to scatter-plot even for a multi-million-cell grid (e.g.
    CLMS Surface Soil Moisture's ~28M-cell continental raster), while still
    tracing the actual overpass-swath shape instead of a bounding rectangle
    over the grid's full nominal extent.
    """
    total = int(valid_mask.sum())
    if total == 0:
        return []
    stride = max(1, int(np.sqrt(total / target_count)))
    strided_mask = valid_mask[::stride, ::stride]
    ys, xs = np.where(strided_mask)
    lon_sub = lons[::stride, ::stride][ys, xs]
    lat_sub = lats[::stride, ::stride][ys, xs]
    return list(zip(lon_sub.tolist(), lat_sub.tolist()))


def _subsample_matched_points(
    lon: np.ndarray, lat: np.ndarray, max_points: Optional[int] = 1000, seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly subsample a matched-point tier to at most *max_points* points
    (fixed seed, reproducible across runs).

    A dense matched-layer source (e.g. soil moisture's ASCAT/SMAP/SMOS
    footprints, routinely 1000+ points per recipe) saturates into a solid
    color blob at any fixed marker size/alpha once points overlap heavily —
    thinning the plotted points (while callers keep using the *original*
    array length for legend counts) keeps individual markers distinguishable
    without misrepresenting how many matches actually occurred. Returns
    *lon*/*lat* unchanged when already at or below *max_points*, or when
    *max_points* is None (subsampling disabled for this variable).
    """
    if max_points is None:
        return lon, lat
    n = len(lon)
    if n <= max_points:
        return lon, lat
    idx = np.random.RandomState(seed).choice(n, size=max_points, replace=False)
    return lon[idx], lat[idx]


def _matched_point_alpha(
    base_alpha: float, n_points: int, max_points: Optional[int] = 1000,
) -> float:
    """Cap alpha lower for a matched-point tier once it's dense enough to be
    subsampled (see :func:`_subsample_matched_points`) — the overlapping
    points that remain after thinning still need enough transparency to
    stay distinguishable from each other. No-op when *max_points* is None."""
    if max_points is None:
        return base_alpha
    return min(base_alpha, 0.35) if n_points > max_points else base_alpha


def _diagnostics_zoom_extent(
    scene_bounds: List[Dict[str, float]],
    footprint_points: List[Tuple[float, float]],
    coverage_points: List[Tuple[float, float]],
    categories: List[dict],
) -> Optional[Tuple[float, float, float, float]]:
    """Bounding box ``(lon_min, lon_max, lat_min, lat_max)`` of everything
    :func:`_plot_collocation_diagnostics_impl` actually draws — SAR scene
    boxes/footprints/coverage plus every validation point shown, matched or
    unmatched — or ``None`` if there's nothing to bound.

    Used so the plot can zoom to where the data actually is instead of
    always the recipe's full requested bbox: a big win for a source whose
    individual scenes are small relative to that bbox (e.g. NISAR SME2's
    per-orbit granules against a continent-scale recipe), and a no-op for a
    source whose scenes already span close to the full bbox (e.g.
    Sentinel-1 CLMS SSM's daily Europe-wide mosaics), since the caller
    clamps the padded result back to the recipe's own bounds.
    """
    lons: List[float] = []
    lats: List[float] = []
    for b in scene_bounds:
        lons.extend((b["lon_min"], b["lon_max"]))
        lats.extend((b["lat_min"], b["lat_max"]))
    for lon, lat in footprint_points:
        lons.append(lon)
        lats.append(lat)
    for lon, lat in coverage_points:
        lons.append(lon)
        lats.append(lat)
    for cat in categories:
        lons.extend(np.asarray(cat["matched_lon"]).tolist())
        lons.extend(np.asarray(cat["unmatched_lon"]).tolist())
        lats.extend(np.asarray(cat["matched_lat"]).tolist())
        lats.extend(np.asarray(cat["unmatched_lat"]).tolist())
    if not lons:
        return None
    return min(lons), max(lons), min(lats), max(lats)


def plot_collocation_diagnostics(
    datatree,
    collocation_ds,
    recipe,
    output_dir: Union[str, Path],
    filename_suffix: str = "",
    layer_vs_layer_collocation_method: str = "cell-averaging",
) -> Union[Path, List[Path], None]:
    """
    Plot collocation diagnostics, dispatching to one plot per SAR file for
    soil_moisture recipes whose SAR source wants it (see
    ``sar_sources.SARSourceSpec.diagnostics_split_by_scene``) and have
    multiple scenes.

    Sentinel-1 CLMS SSM's SAR "scenes" are daily, mutually-overlapping,
    continent-wide overpass mosaics -- overlaying every day's coverage and
    matches onto a single map makes individual passes visually
    indistinguishable, so each scene gets its own diagnostics PNG instead
    (returned as a list of Path). Every other source's scenes (e.g. NISAR
    SME2's small, non-overlapping per-orbit granules) don't have that
    problem and keep the original single combined-map behaviour (returned
    as one Path, or None) -- as does any soil_moisture recipe with only one
    scene, and every non-soil_moisture variable.

    See :func:`_plot_collocation_diagnostics_impl` for the actual plotting
    logic and full parameter documentation.
    """
    from .sar_sources import SAR_SOURCES

    sar_node = datatree.get("sar")
    scene_names = list(sar_node.children.keys()) if sar_node is not None else []

    sar_spec = SAR_SOURCES.get(recipe.config.sar_data.source)
    split_by_scene = sar_spec is not None and sar_spec.diagnostics_split_by_scene

    if recipe.config.variable == "soil_moisture" and split_by_scene and len(scene_names) > 1:
        paths: List[Path] = []
        for scene_name in scene_names:
            scene_tree = datatree.copy()
            for other in scene_names:
                if other != scene_name:
                    del scene_tree["sar"][other]
            scene_coll_ds = (
                _filter_by_scene(collocation_ds, scene_name) if collocation_ds is not None else None
            )
            path = _plot_collocation_diagnostics_impl(
                scene_tree, scene_coll_ds, recipe, output_dir,
                filename_suffix=f"{filename_suffix}_{scene_name}",
                layer_vs_layer_collocation_method=layer_vs_layer_collocation_method,
                scene_label=scene_name,
            )
            if path is not None:
                paths.append(path)
        return paths

    return _plot_collocation_diagnostics_impl(
        datatree, collocation_ds, recipe, output_dir,
        filename_suffix=filename_suffix,
        layer_vs_layer_collocation_method=layer_vs_layer_collocation_method,
    )


def _plot_collocation_diagnostics_impl(
    datatree,
    collocation_ds,
    recipe,
    output_dir: Union[str, Path],
    filename_suffix: str = "",
    layer_vs_layer_collocation_method: str = "cell-averaging",
    scene_label: Optional[str] = None,
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
    layer_vs_layer_collocation_method : str
        Which layer-vs-layer collocation method produced *collocation_ds*
        ("cell-averaging" or "individual"). Individual-mode matches are far
        denser than cell-averaging (one point per matched SAR pixel vs. one
        per validation-instrument location), so this controls matched-layer
        point transparency — see the alpha computation below.
    scene_label : str, optional
        SAR scene/file name to append to the plot title, e.g.
        ``"<recipe> Collocation Diagnostics — <scene_label>"``. Set by
        :func:`plot_collocation_diagnostics` when it splits a soil_moisture
        recipe into one plot per SAR file, so each PNG's own title (not
        just its filename) identifies which day/overpass it shows.

    Returns
    -------
    Path or None
        Path to the saved PNG file, or None if plot could not be generated.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    from .recipe import DEFAULT_LAYER_TYPE_SPECS  # noqa: PLC0415

    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Set up cartopy if available
    try:
        import cartopy.crs as ccrs  # noqa: PLC0415
    except ImportError:
        logger.debug("cartopy not installed — collocation_diagnostics plot unavailable.")
        return None

    # Dense matched-point subsampling (see _subsample_matched_points) only
    # applies to soil_moisture's typically saturated ASCAT/SMAP/SMOS point
    # clouds; every other variable plots every matched point it has.
    density_subsample_max = 1000 if recipe.config.variable == "soil_moisture" else None

    # ── Extract SAR scene bounds ────────────────────────────────────────
    sar_node = datatree.get("sar")
    if sar_node is None or not sar_node.children:
        logger.warning("plot_collocation_diagnostics: No SAR data found in DataTree.")
        return None

    scene_bounds = []            # bounding boxes for grid-mode (IW/EW) scenes
    footprint_points: list[tuple[float, float]] = []  # (lon, lat) per sparse WV-mode imagette
    # (lon, lat) of actual valid-data pixels for overpass-mosaic products
    # (soil_moisture) — see is_overpass_mosaic below.
    coverage_points: list[tuple[float, float]] = []
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
    # CLMS Surface Soil Moisture's grid has valid lon/lat everywhere across
    # the continent, but the actual retrieved value is NaN except along
    # that day's satellite overpass swaths — a min/max lon/lat bounding
    # rectangle (which the else-branch below uses for every other grid
    # product, where it's accurate) would therefore claim coverage across
    # mostly-empty regions. Scope this to soil_moisture specifically:
    # other grid products (wind/currents/waves) are single-swath and their
    # bounding rectangle already matches their real footprint.
    is_overpass_mosaic = recipe.config.variable == "soil_moisture"

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
            data_vars = list(scene_ds.data_vars)
            if is_wv_scene:
                footprint_points.extend(
                    zip(lons_flat[valid_mask].tolist(), lats_flat[valid_mask].tolist())
                )
            elif is_overpass_mosaic and len(lons.shape) > 1 and data_vars:
                data_valid = np.isfinite(scene_ds[data_vars[0]].values)
                if data_valid.shape == lons.shape:
                    coverage_points.extend(
                        _downsampled_valid_pixel_coords(data_valid, lons, lats)
                    )
                else:
                    scene_bounds.append({
                        "lon_min": float(np.nanmin(lons_flat[valid_mask])),
                        "lon_max": float(np.nanmax(lons_flat[valid_mask])),
                        "lat_min": float(np.nanmin(lats_flat[valid_mask])),
                        "lat_max": float(np.nanmax(lats_flat[valid_mask])),
                    })
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

    if not scene_bounds and not footprint_points and not coverage_points:
        logger.warning("plot_collocation_diagnostics: Could not extract SAR scene bounds.")
        return None

    # ── Extract all validation data ─────────────────────────────────────
    # No 'validation' node at all (e.g. a validation source collected zero
    # files -- an unconfigured/awaiting ISMN archive, say) is a stricter
    # case than "zero collocated pairs": there isn't even a validation
    # point to mark unmatched. Still plot the SAR coverage rather than
    # bailing out, so a recipe with real SAR data but no validation data
    # yet still gets the diagnostic (and not a silently-missing plot).
    all_val_data = _extract_validation_data_for_plot(datatree)
    if not all_val_data:
        logger.warning(
            "plot_collocation_diagnostics: No validation data found in "
            "DataTree -- plotting SAR coverage only."
        )
        all_val_lons = np.array([])
        all_val_lats = np.array([])
        all_val_times = np.array([])
        platform_types_arr = np.array([])
    else:
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
        return layer_type_tol.get(str(ptype).lower(), pvl_default_tol)

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

    matched_labels_all = np.array([_diagnostics_category(src) for src in val_source_all])

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
        label = _diagnostics_category(ptype)
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
        u_lon_list, u_lat_list, u_src_list = unmatched_by_category.get(label, ([], [], []))
        u_lon = np.array(u_lon_list)
        u_lat = np.array(u_lat_list)
        u_src = np.array(u_src_list)
        if len(m_lon) + len(u_lon) == 0:
            continue
        categories.append({
            "label": label,
            "matched_lon": m_lon, "matched_lat": m_lat, "matched_source": m_src,
            "unmatched_lon": u_lon, "unmatched_lat": u_lat, "unmatched_source": u_src,
        })

    if not categories:
        logger.debug("plot_collocation_diagnostics: No classifiable validation points -- SAR coverage only.")

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
    variable = recipe.config.variable

    # Matched-point styling depends on the recipe's variable type and, for
    # layer-source matches, the collocation method that produced them:
    # - individual-method matches are far denser than cell-averaging (one
    #   point per matched SAR pixel vs. one per validation-instrument
    #   location — routinely 100-500x more points for the same recipe), so
    #   the same alpha used for cell-averaging would saturate into a solid
    #   blob; a lower fixed alpha keeps density visible instead.
    # - wind (cell-averaging): layer-source matches (Tier 3) get a moderate
    #   alpha instead of full opacity, since a dense swath (e.g.
    #   scatterometer) would otherwise fully occlude a sparser layer source
    #   (e.g. radiometer) plotted underneath it in the same tier.
    # - waves: all matched points (Tier 3 + Tier 4) get a larger marker and
    #   a black edge, making individual matches easier to pick out.
    if layer_vs_layer_collocation_method == "individual":
        matched_layer_alpha = 0.15
    else:
        # wind and soil_moisture both have visually dense, overlapping
        # matched-layer sources (scatterometer swaths; ASCAT/SMAP/SMOS
        # soil-moisture footprints) that bury the SAR field and each other
        # at full opacity.
        matched_layer_alpha = 0.65 if variable in ("wind", "soil_moisture") else 1.0
    if variable == "waves":
        matched_marker_size = 45
        matched_edgecolors = "black"
        matched_linewidths = 0.5
    else:
        matched_marker_size = 25
        matched_edgecolors = "none"
        matched_linewidths = 0.0

    # ── Create geographic plot ──────────────────────────────────────────
    # A crossing bbox (min_lon > max_lon, see GeographicBounds' antimeridian
    # convention) is centered on 180 deg instead of Greenwich, so the map
    # itself doesn't get cut at the dateline.
    crosses_dateline = bounds.min_lon > bounds.max_lon
    proj = ccrs.PlateCarree(central_longitude=180) if crosses_dateline else ccrs.PlateCarree()
    fig = plt.figure(figsize=(14, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    transform = ccrs.PlateCarree()

    # Add coastlines and features
    land, coastline = _land_coastline_features()
    ax.add_feature(land, facecolor="lightgray", alpha=0.3, zorder=0)
    ax.add_feature(coastline, linewidth=0.5, zorder=0)
    gl = ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5)

    # ── Set plot extent, zoomed to the actual data when that's tighter
    # than the recipe's full requested bbox ──────────────────────────────
    # In the central_longitude=180 axes frame, true longitude L maps to
    # (L % 360) - 180, which turns the wrapped [min_lon, 180] +
    # [-180, max_lon] range into one contiguous span with no wraparound.
    def _shift(lon: float) -> float:
        return (lon % 360) - 180

    if crosses_dateline:
        # Combining the dateline's own longitude-shifting with a
        # data-driven zoom needs more care than this fix's scope covers —
        # keep the always-full-bounds behaviour here.
        ax.set_extent(
            [_shift(bounds.min_lon), _shift(bounds.max_lon), bounds.min_lat, bounds.max_lat],
            crs=proj,
        )
    else:
        lon0, lon1, lat0, lat1 = bounds.min_lon, bounds.max_lon, bounds.min_lat, bounds.max_lat
        data_extent = _diagnostics_zoom_extent(scene_bounds, footprint_points, coverage_points, categories)
        if data_extent is not None:
            d_lon_min, d_lon_max, d_lat_min, d_lat_max = data_extent
            lon_pad = max((d_lon_max - d_lon_min) * 0.2, 0.2)
            lat_pad = max((d_lat_max - d_lat_min) * 0.2, 0.2)
            # Never zoom OUT past what was actually requested — only ever
            # in, closer than the full recipe bbox.
            lon0 = max(d_lon_min - lon_pad, bounds.min_lon)
            lon1 = min(d_lon_max + lon_pad, bounds.max_lon)
            lat0 = max(d_lat_min - lat_pad, bounds.min_lat)
            lat1 = min(d_lat_max + lat_pad, bounds.max_lat)
        ax.set_extent([lon0, lon1, lat0, lat1], crs=transform)
    _set_lonlat_ticks(ax, gl)

    # Scene-box/footprint longitudes above are pre-shifted into the axes'
    # own central_longitude=180 frame when crossing the dateline (so they
    # form one contiguous span with no wraparound). They must therefore be
    # plotted with a transform matching that same frame (``proj``) rather
    # than the raw-lon ``transform`` — otherwise cartopy reprojects the
    # already-shifted values a second time and they land far outside the
    # visible extent. Non-crossing maps are unaffected: box_transform ==
    # transform there.
    box_transform = proj if crosses_dateline else transform

    # ── SAR coverage (zorder=1): Grid scenes → bounding box; sparse WV
    # imagettes → one footprint circle each (radius = the collocation footprint
    # radius), so it's visually clear that matches are only possible near each
    # imagette, not across the whole bounding rectangle. ──────────────────────
    for i, sb in enumerate(scene_bounds):
        lons_box = [sb["lon_min"], sb["lon_max"], sb["lon_max"], sb["lon_min"], sb["lon_min"]]
        lats_box = [sb["lat_min"], sb["lat_min"], sb["lat_max"], sb["lat_max"], sb["lat_min"]]
        if crosses_dateline:
            lons_box = [_shift(lon) for lon in lons_box]
        ax.plot(lons_box, lats_box, color="blue", linewidth=1.5,
                transform=box_transform, zorder=1, label="SAR scene bounds" if i == 0 else "")

    if footprint_points:
        theta = np.linspace(0, 2 * np.pi, 60)
        r_lat_deg = footprint_radius_km / 111.0
        for j, (flon, flat) in enumerate(footprint_points):
            # Approximate circle in lon/lat (lon degrees shrink by cos(lat)).
            cos_lat = max(np.cos(np.radians(flat)), 1e-6)
            center_lon = _shift(flon) if crosses_dateline else flon
            circ_lon = center_lon + (r_lat_deg / cos_lat) * np.cos(theta)
            circ_lat = flat + r_lat_deg * np.sin(theta)
            ax.plot(circ_lon, circ_lat, color="blue", linewidth=1.2,
                    transform=box_transform, zorder=1,
                    label=f"SAR footprint (±{footprint_radius_km:.0f} km)" if j == 0 else "")
            ax.scatter([center_lon], [flat], s=10, c="blue", marker="+",
                       transform=box_transform, zorder=1)

    if coverage_points:
        cov_lons = [
            (_shift(lon) if crosses_dateline else lon) for lon, _ in coverage_points
        ]
        cov_lats = [lat for _, lat in coverage_points]
        ax.scatter(
            cov_lons, cov_lats, s=3, c="blue", alpha=0.4, marker=".",
            transform=box_transform, zorder=1, label="SAR coverage (overpasses)",
        )

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
                    # source_style_map is keyed by display category (the same
                    # consolidated labels _diagnostics_category produces), not
                    # by a naive title-cased raw data_type/platform_type token.
                    color, marker = source_style_map.get(_diagnostics_category(str(source)), ("#808080", "o"))
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
        n_matched = len(m_lon)
        if n_matched == 0:
            continue
        color, marker = source_style_map.get(str(cat["label"]), ("#2ca02c", "o"))
        plot_lon, plot_lat = _subsample_matched_points(m_lon, m_lat, max_points=density_subsample_max)
        ax.scatter(
            plot_lon, plot_lat,
            s=matched_marker_size, c=color, marker=marker,
            alpha=_matched_point_alpha(matched_layer_alpha, n_matched, max_points=density_subsample_max),
            edgecolors=matched_edgecolors, linewidths=matched_linewidths,
            transform=transform, zorder=5, label=f"{cat['label']} matched ({n_matched})",
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
                    plot_lon, plot_lat = _subsample_matched_points(
                        m_lon[mask], m_lat[mask], max_points=density_subsample_max
                    )
                    ax.scatter(
                        plot_lon, plot_lat,
                        s=matched_marker_size, c=color, marker=marker,
                        alpha=_matched_point_alpha(1.0, count, max_points=density_subsample_max),
                        edgecolors=matched_edgecolors, linewidths=matched_linewidths,
                        transform=transform, zorder=6,
                        label=f"In-situ matched: {source} ({count})",
                    )
            else:
                # Fallback if no source info available
                n_matched = len(m_lon)
                plot_lon, plot_lat = _subsample_matched_points(m_lon, m_lat, max_points=density_subsample_max)
                ax.scatter(
                    plot_lon, plot_lat,
                    s=matched_marker_size, c="#ff7f0e", marker="o",
                    alpha=_matched_point_alpha(1.0, n_matched, max_points=density_subsample_max),
                    edgecolors=matched_edgecolors, linewidths=matched_linewidths,
                    transform=transform, zorder=6,
                    label=f"In-situ matched ({n_matched})",
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
    if scene_label is not None:
        title += f" — {scene_label}"
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
    # loc="best" is prohibitively expensive on a GeoAxes covering the full
    # recipe bbox with thousands of collocated points — see the identical
    # fix (and its rationale) in _draw_scene_panel above.
    ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=8, framealpha=0.9)

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


def _image_page_figure(img, dpi: int = 150):
    """Build a throwaway Figure that exactly fills its canvas with *img* —
    used to embed an already-rendered PNG as a PDF page without drawing
    the original (often much more expensive, e.g. cartopy) figure a
    second time."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    img_h, img_w = img.shape[0], img.shape[1]
    fig = plt.figure(figsize=(img_w / dpi, img_h / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(img)
    ax.axis("off")
    return fig


def _finalize_figure_for_report(fig, png_path: Optional[Path], dpi: int = 150):
    """Render *fig* to PNG exactly once, optionally save it to *png_path*,
    close *fig*, and return a lightweight image-only Figure for embedding
    as a PDF page. Avoids drawing the same (often expensive) figure a
    second time via ``PdfPages.savefig``."""
    import io

    import matplotlib.pyplot as plt  # noqa: PLC0415

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    if png_path is not None:
        png_path.write_bytes(buf.getvalue())
    buf.seek(0)
    return _image_page_figure(plt.imread(buf, format="png"), dpi=dpi)


def _stamp_banner(fig: "Figure", text: str, color: str) -> "Figure":
    """Draw a small, visually distinct italic banner across the top of
    *fig*. Adds an overlay text rather than touching the figure's own
    title/suptitle (set by plot_scatter/plot_residuals/plot_statistics
    themselves), so those plotting functions stay unchanged. Called
    before the figure is finalized/rasterized for the report so the
    banner survives into the saved PDF/PNG."""
    fig.text(
        0.5, 1.04, text,
        ha="center", va="bottom", fontsize=9, style="italic", color=color,
    )
    return fig


def _mark_native_units(fig: "Figure") -> "Figure":
    """Stamp a "— native units —" banner across the top of *fig* so
    native-units report pages can never be mistaken for the CDF-matched
    section's otherwise-identical-looking plots."""
    return _stamp_banner(fig, "— native units —", "darkred")


def _mark_cdf_matched(fig: "Figure") -> "Figure":
    """Stamp a "(CDF-matched)" banner across the top of *fig* so
    soil-moisture's main-section scatter/geographic/residuals pages
    (which plot the CDF-matched/rescaled SAR series, not raw values)
    can't be mistaken for a units bug — mirrors ``_mark_native_units``
    but with a visually distinct color so the two banner types remain
    distinguishable."""
    return _stamp_banner(fig, "(CDF-matched)", "navy")


def _cdf_matched_suffix(variable: str) -> str:
    """Return ``" (CDF-matched)"`` for soil_moisture recipes, else ``""``.

    Soil moisture's main-section SAR series is CDF-matched/rescaled (via
    ``add_rescaled_sar_column``) before being plotted, not raw — unlike
    every other variable's main section. Appended to the main section's
    scatter/geographic/residuals page titles so that isn't mistaken for a
    units bug during manual testing (the native-units section, which does
    use raw values, is already labelled via ``_mark_native_units``).

    Note: this suffix only affects the (currently unused-for-rendering)
    ``pdf_pages`` title strings. The actual visible page annotation is
    ``_mark_cdf_matched``, stamped directly onto the figure."""
    return " (CDF-matched)" if variable == "soil_moisture" else ""


#: val_source (matched path, literal per-satellite name) and data_type
#: (unmatched path, generic category token, set by datatree_converter.py
#: via DEFAULT_LAYER_TYPE_SPECS) -> display category, for soil-moisture
#: satellite sources specifically. LAYER_DATA_TYPES (collocation.py)
#: alone isn't enough here because plot_collocation_diagnostics' matched
#: and unmatched code paths carry different string forms for the same
#: physical source -- see _diagnostics_category's two call sites below.
_SOIL_MOISTURE_SOURCE_CATEGORY: Dict[str, str] = {
    "ascat_ssm": "Scatterometer",
    "amsr_ssm": "Radiometer",
    "smap_ssm": "Radiometer",
    "smos_ssm": "Radiometer",
    "scatterometer_ssm": "Scatterometer",
    "radiometer_ssm": "Radiometer",
}


def _diagnostics_category(value: str) -> str:
    """
    Display category for a val_source (matched path) or platform_type/
    data_type (unmatched path) value in plot_collocation_diagnostics.

    Checks the soil-moisture literal/generic-name mapping first (the
    matched and unmatched paths disagree on which string form they
    carry for these sources), then falls back to the pre-existing
    LAYER_DATA_TYPES-based generic-category check (unchanged behavior
    for every other variable's sources: wind's scatterometer/
    radiometer/altimeter, currents' hf_radar, ...), then "In-situ".
    """
    s = str(value)
    if s in _SOIL_MOISTURE_SOURCE_CATEGORY:
        return _SOIL_MOISTURE_SOURCE_CATEGORY[s]
    from .collocation import LAYER_DATA_TYPES  # noqa: PLC0415
    return s.title() if s in LAYER_DATA_TYPES else "In-situ"


def _volumetric_hist_range_overrides(pair_ds) -> Dict[str, Tuple[float, float]]:
    """Map every ``val_source`` present in *pair_ds* to a fixed ``(-1, 1)``
    residuals x-axis range.

    Used as ``plot_residuals``'s per-source ``hist_range`` dict for
    CDF-matched soil-moisture pairs. By the time this function runs,
    ``add_rescaled_sar_column`` has already harmonized every source in
    *pair_ds* into one consistent volumetric (~0-1) domain via
    ``_harmonize_percent_domain_sources`` -- a source that can't be
    harmonized (e.g. ``ascat_ssm`` with no ``ismn`` reference present to
    fit a transform against) has its rows dropped to NaN rather than left
    in its native percent-scale domain, so there is no longer any source
    left in *pair_ds* whose residuals could genuinely span far beyond
    +-1. That means a single shared ``(-1, 1)`` range is safe for every
    source uniformly -- no per-source unit-family lookup is needed
    anymore (this function no longer imports or checks
    ``_VAL_SOURCE_UNITS_FAMILY``). A source with all-NaN residuals
    (dropped by the harmonize fallback) is harmless to still range at
    ``(-1, 1)`` since there is no data to plot for it either way.

    Returns ``{}`` when ``val_source`` is absent or empty, which makes
    ``plot_residuals`` fall back to its own per-source data-driven range
    for every source (safe default, never a crash or a wrong guess).
    """
    if "val_source" not in pair_ds or pair_ds.sizes.get("collocation", 0) == 0:
        return {}

    sources = set(str(s) for s in pair_ds["val_source"].values)
    return {s: (-1.0, 1.0) for s in sources}


def plot_rvl_land_qa(datatree) -> Optional["Figure"]:
    """
    Build a table figure listing, for every SAR RVL scene with at least one
    land-flagged cell, the land pixel count/fraction and the pre-mask mean
    ``rvlRadVel`` over those land cells (expected ~0 m/s — a meaningfully
    non-zero value signals land contamination worth investigating).

    Reads the ``rvl_land_pixel_count`` / ``rvl_land_pixel_fraction`` /
    ``rvl_land_mean_radvel`` attrs stamped on each SAR node by
    ``DataTreeConverter._extract_rvl_grid_data``.

    Returns
    -------
    matplotlib.figure.Figure or None
        None if *datatree* has no "sar" node, or no scene in it has any
        land-flagged cells — callers should skip adding a page in that case.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if "sar" not in datatree.children:
        return None

    rows = []
    for name, node in datatree["sar"].children.items():
        attrs = node.to_dataset().attrs
        land_count = attrs.get("rvl_land_pixel_count", 0)
        if not land_count:
            continue
        rows.append((
            name,
            int(land_count),
            100 * attrs.get("rvl_land_pixel_fraction", float("nan")),
            attrs.get("rvl_land_mean_radvel", float("nan")),
        ))

    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(11, 0.6 * len(rows) + 2))
    ax.axis("off")
    ax.set_title(
        "RVL land-contamination QA — cells masked out of rvlRadVel/rvlRadVelStd",
        fontsize=12, fontweight="bold",
    )
    table = ax.table(
        cellText=[
            [
                scene,
                str(count),
                "n/a" if np.isnan(frac) else f"{frac:.1f}%",
                "n/a" if np.isnan(mean) else f"{mean:.4f}",
            ]
            for scene, count, frac, mean in rows
        ],
        colLabels=["Scene", "Land pixels", "Land %", "Mean rvlRadVel over land (m/s)"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    fig.tight_layout()
    return fig


def _collect_sensing_depths(datatree, recipe) -> list[str]:
    """
    Collect distinct "<source> ~<depth>cm (<band>-band)" strings from every
    validation node's sensing_depth_cm/band attrs, plus ISMN's recipe-level
    min_depth/max_depth (which predates the structured attrs). Used only by
    the soil_moisture cover-page note (validation_report).
    """
    seen: dict[tuple[str, str], list[str]] = {}
    for node in datatree.subtree:
        depth = node.attrs.get("sensing_depth_cm")
        band = node.attrs.get("band")
        source = node.attrs.get("platform_type") or node.attrs.get("source")
        if depth and band and source:
            seen.setdefault((depth, band), []).append(str(source))

    lines = [
        f"{'/'.join(sorted(set(sources)))} ~{depth}cm ({band}-band)"
        for (depth, band), sources in seen.items()
    ]

    for src in recipe.config.validation_sources:
        if src.source_type == "ismn":
            lines.append(
                f"ISMN {src.resolved_min_depth}-{src.resolved_max_depth}m depth window"
            )
    return lines


def validation_report(
    collocation_ds,
    datatree,
    recipe,
    stats_ds_map: Optional[Dict[str, "xr.Dataset"]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    filename_suffix: str = "",
    download_warnings: Optional[list[str]] = None,
    layer_vs_layer_collocation_method: str = "cell-averaging",
    native_units_stats_ds_map: Optional[Dict[str, "xr.Dataset"]] = None,
) -> Dict[str, list]:
    """
    Run all four plot functions for every (sar_var, val_var) pair inferred
    from *recipe*, embed all plots in a combined ``validation_report.pdf``,
    and save the collocation-diagnostics PNG to ``<out_dir>/plots/`` (alongside
    the ``validation_statistics_*.nc`` files).

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
        Base output directory.  The combined PDF is written to
        ``<out_dir>/validation_report<suffix>.pdf``; the collocation-diagnostics
        PNG is saved to ``<out_dir>/plots/collocation_diagnostics_<recipe_name><suffix>.png``
        (alongside the ``validation_statistics_*.nc`` files).
        If None the figures are returned without saving.
    filename_suffix : str
        Appended to PDF and collocation-diagnostics PNG filenames,
        e.g. ``"_individual"``. Lets reports from two collocation methods
        coexist without overwriting each other.
    download_warnings : list[str], optional
        Download-step error messages (from ``download_metadata.json``'s
        ``errors`` list) to surface on the PDF cover page. None or an empty
        list adds no warning text.
    layer_vs_layer_collocation_method : str
        Which layer-vs-layer collocation method produced *collocation_ds*
        ("cell-averaging" or "individual"). Passed through to
        :func:`plot_collocation_diagnostics` for method-aware matched-point
        transparency.
    native_units_stats_ds_map : dict, optional
        Output of ``run_statistics_native_units`` (or ``None``). When
        provided and ``recipe.config.variable == 'soil_moisture'``, a
        second, non-CDF-matched scatter/residual/statistics section is
        added per pair, restricted to the ``val_source`` groups present in
        each pair's native-units stats Dataset, titled with a
        " — native units" suffix.

    Returns
    -------
    dict[str, list[matplotlib.figure.Figure]]
        ``"<sar_var>_vs_<val_var>"`` → list of Figure objects for that pair.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from ._variable_map import CIRCULAR_VAL_VARS, filter_variable_pairs  # noqa: PLC0415

    base_dir: Optional[Path] = None
    if out_dir is not None:
        base_dir = Path(out_dir)

    variable = recipe.config.variable
    # Whether each soil_moisture geographic scene panel should clamp its
    # extent to the recipe's full requested bbox (needed for
    # sentinel1_clms_ssm, whose raw grid covers all of mainland Europe
    # regardless of what was requested) or auto-zoom to its own actual
    # data (correct for sources like nisar_sme2, whose native grid is
    # already tight around real data) -- see
    # SARSourceSpec.geographic_plot_clamp_to_bounds.
    from .sar_sources import SAR_SOURCES

    _geo_sar_spec = SAR_SOURCES.get(recipe.config.sar_data.source)
    geo_clamp_bounds = (
        variable == "soil_moisture"
        and _geo_sar_spec is not None
        and _geo_sar_spec.geographic_plot_clamp_to_bounds
    )
    try:
        pairs = filter_variable_pairs(recipe, collocation_ds)
    except KeyError as exc:
        logger.error("validation_report: %s", exc)
        return {}

    all_figures: Dict[str, list] = {}

    # Union across all pairs of SAR scenes that matched at least one
    # validation point — used to drop scenes with no matches from the
    # geographic plots. collocation_ds holds only matched pairs, so every
    # scene present here has >= 1 match. None => don't filter.
    matched_scenes = (
        sorted(set(str(s) for s in collocation_ds["sar_scene_name"].values))
        if "sar_scene_name" in collocation_ds else None
    )

    # PDF writer: opens lazily on the first page actually written (so, as
    # before, no file is created at all if nothing ever gets added) and
    # writes+closes each page's Figure immediately. Previously every
    # lightweight page Figure (built by _finalize_figure_for_report /
    # _image_page_figure) was accumulated in a list for the whole report
    # and only closed in one final loop -- with several (sar_var, val_var)
    # pairs that left 20+ figures open simultaneously, which crashed
    # VSCode during real-data testing.
    pdf_path = base_dir / f"validation_report{filename_suffix}.pdf" if base_dir is not None else None
    pdf_cm: Optional["PdfPages"] = None
    pdf_writer: Optional["PdfPages"] = None

    def _open_pdf():
        nonlocal pdf_cm, pdf_writer
        import datetime as _dt  # noqa: PLC0415

        from matplotlib.backends.backend_pdf import PdfPages  # noqa: PLC0415

        pdf_cm = PdfPages(pdf_path)
        pdf_writer = pdf_cm.__enter__()

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
        if variable == "soil_moisture":
            depth_lines = _collect_sensing_depths(datatree, recipe)
            if depth_lines:
                cover.text(
                    0.5, 0.34,
                    "Sensing depths: " + " · ".join(depth_lines),
                    ha="center", va="center", fontsize=8, wrap=True,
                )
        if download_warnings:
            cover.text(
                0.5, 0.26,
                "⚠ " + "; ".join(download_warnings),
                ha="center", va="center", fontsize=9, color="firebrick", wrap=True,
            )
        pdf_writer.savefig(cover, bbox_inches="tight")
        plt.close(cover)
        return pdf_writer

    def _write_page(_title, fig):
        writer = pdf_writer if pdf_writer is not None else _open_pdf()
        writer.savefig(fig, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Collocation diagnostics plot — generated once per recipe, written
    # first (right after the cover page) so a reader sees the spatial/
    # matching overview before the per-pair detail sections below.
    if base_dir is not None:
        try:
            diag_result = plot_collocation_diagnostics(
                datatree, collocation_ds, recipe, base_dir,
                filename_suffix=filename_suffix,
                layer_vs_layer_collocation_method=layer_vs_layer_collocation_method,
            )
            # soil_moisture recipes with multiple SAR files return one Path
            # per file (see plot_collocation_diagnostics); every other case
            # returns a single Path or None.
            diag_paths = (
                diag_result if isinstance(diag_result, list)
                else ([diag_result] if diag_result is not None else [])
            )
            for diag_path in diag_paths:
                logger.info("Collocation diagnostics plot saved to %s", diag_path)
                diag_img = plt.imread(str(diag_path))
                title = (
                    f"Collocation diagnostics — {recipe.config.name}"
                    if len(diag_paths) == 1
                    else f"Collocation diagnostics — {recipe.config.name} [{diag_path.stem}]"
                )
                # plot_collocation_diagnostics() closes its own figure(s)
                # internally (it's also called standalone from cli.py), so
                # the only way to embed them here is to reload the
                # rendered images.
                _write_page(title, _image_page_figure(diag_img))
        except Exception as exc:
            logger.warning("plot_collocation_diagnostics failed: %s", exc)

    # RVL land-contamination QA page — currents recipes only, and only when
    # at least one scene actually has land-flagged cells (plot_rvl_land_qa
    # returns None otherwise, so no empty page is added). Written right
    # after the diagnostics page(s) rather than appended at the end, per
    # the design spec.
    if base_dir is not None and variable == "currents":
        try:
            fig_land_qa = plot_rvl_land_qa(datatree)
            if fig_land_qa is not None:
                _write_page("RVL land-contamination QA", _finalize_figure_for_report(fig_land_qa, None))
        except Exception as exc:
            logger.warning("plot_rvl_land_qa failed: %s", exc)

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

        # Soil moisture: the raw SAR series lives in a different physical
        # domain than the validation series (e.g. a relative SAR
        # saturation index vs. ISMN's absolute volumetric content) —
        # compute_statistics_soil_moisture already CDF-matches them before
        # computing bias/RMSE, but that rescaled series was previously
        # discarded, so every point-based plot below was comparing raw,
        # non-comparable values. Substitute the rescaled series here once,
        # so scatter/residuals/temporal-offset all compare like with like.
        #
        # plot_geographic is the one exception: it keeps the pre-rescale
        # `pair_ds` (as `geo_pair_ds` below) deliberately, and fits its own
        # whole-field transform internally (fit_sar_to_val_transform) from
        # the *raw* sar_<var>/val_<var> pairs. If it were handed the
        # already-rescaled column instead, it would fit a transform whose
        # training "source" is already in the validation domain (~0.05-0.5)
        # and then apply that transform to the real, raw SAR field
        # (~0-100) — wildly out of the fitted range, producing nonsense
        # (confirmed against real data: predicted values above 300 for a
        # variable that should span roughly 0-1).
        geo_pair_ds = pair_ds
        cdf_matched_suffix = _cdf_matched_suffix(variable)
        harmonized_sources: set = set()
        if variable == "soil_moisture":
            from .statistics import (  # noqa: PLC0415
                _harmonize_percent_domain_sources,
                add_rescaled_sar_column,
            )
            _, harmonized_sources, _ = _harmonize_percent_domain_sources(pair_ds, sar_var, val_var)
            pair_ds = add_rescaled_sar_column(pair_ds, sar_var, val_var)

        logger.info("Generating plots for %s vs %s …", sar_var, val_var)

        # Geographic — returns dict[collocation_type, Figure] by default.
        # Rendered before the scatter section below, so a reader sees the
        # spatial context (where the matches are, how dense) before the
        # more abstract point-cloud comparison -- applies to every recipe
        # type, not just soil moisture, since this loop is shared.
        try:
            # HF radar (currents' only layer-type source) forms a
            # near-continuous coverage grid that at the default marker size
            # tiles edge-to-edge and completely hides the SAR field
            # underneath it — use a smaller marker for currents recipes so
            # the SAR scene stays visible through the validation points.
            # Scatterometer/radiometer (wind/waves/soil_moisture) similarly
            # occludes the SAR field; use adaptive sizing: if >~300 points
            # per scene, use smaller markers (5), else 15. soil_moisture
            # mixes sparse in-situ ISMN points with dense scatterometer/
            # radiometer coverage in the same panel, so it needs the same
            # per-scene density check as wind rather than one fixed size
            # (10 made ISMN points too small and ASCAT/SMAP/SMOS too big).
            # Currents always 15, other variables default to 40.
            geo_point_size: Union[int, Dict[str, int]]
            if variable == "currents":
                geo_point_size = 15
            elif variable in ("wind", "soil_moisture"):
                # len(pair_ds) counts an xr.Dataset's *data variables*, not
                # its collocation rows -- .sizes["collocation"] is the
                # actual per-scene point count this check needs. Computed
                # separately per collocation_type (point_vs_layer vs.
                # layer_vs_layer, see plot_geographic's dict point_size) --
                # pooling both into one average let a dense scatterometer/
                # radiometer type dictate the size for a sparse in-situ
                # type sharing the same pair (e.g. soil moisture's dense
                # ASCAT/SMAP/SMOS vs. sparse ISMN), making ISMN's points
                # too small.
                n_scenes = max(1, len(matched_scenes or []))
                if "collocation_type" in geo_pair_ds:
                    geo_point_size = {}
                    ctypes_present = sorted(set(str(v) for v in geo_pair_ds["collocation_type"].values))
                    for ctype in ctypes_present:
                        n_pts = int((geo_pair_ds["collocation_type"] == ctype).sum())
                        if variable == "soil_moisture" and ctype == "point_vs_layer":
                            # ISMN stations (point_vs_layer) are sparse
                            # in-situ points, not a dense occlusion-prone
                            # swath like the satellite layer_vs_layer
                            # sources sharing this dict -- size them
                            # generously (25) so individual stations stay
                            # legible, rather than the same
                            # density-adaptive 5/15 the dense types use.
                            geo_point_size[ctype] = 25
                        else:
                            geo_point_size[ctype] = 5 if (n_pts / n_scenes) > 300 else 15
                else:
                    n_points = geo_pair_ds.sizes.get("collocation", len(geo_pair_ds))
                    geo_point_size = 5 if (n_points / n_scenes) > 300 else 15
            else:
                geo_point_size = 40
            geo_result = plot_geographic(
                datatree, geo_pair_ds, sar_var, val_var, scenes=matched_scenes,
                point_size=geo_point_size,
                geographic_bounds=(
                    recipe.config.geographic_bounds if geo_clamp_bounds else None
                ),
                two_column_by_type=(variable == "soil_moisture"),
            )
            if isinstance(geo_result, dict):
                for group, fig_geo in geo_result.items():
                    if fig_geo is not None:
                        if cdf_matched_suffix:
                            fig_geo = _mark_cdf_matched(fig_geo)
                        figs.append(fig_geo)
                        title = f"{sar_var} vs {val_var} — geographic [{group}]{cdf_matched_suffix}"
                        if base_dir is not None:
                            _write_page(title, _finalize_figure_for_report(fig_geo, None))
            elif geo_result is not None:
                if cdf_matched_suffix:
                    geo_result = _mark_cdf_matched(geo_result)
                figs.append(geo_result)
                title = f"{sar_var} vs {val_var} — geographic{cdf_matched_suffix}"
                if base_dir is not None:
                    _write_page(title, _finalize_figure_for_report(geo_result, None))
        except Exception as exc:
            logger.warning("plot_geographic failed for %s: %s", sar_var, exc, exc_info=True)

        # Scatter — split into per-source small multiples not only when one
        # source dominates by point count (see plot_scatter's
        # split_when_imbalanced), but also whenever harmonization actually
        # converted a source (e.g. ASCAT) into the reference domain: piling
        # every source into one shared axes at that point is too visually
        # busy even when no single source dominates by point count
        # (confirmed against real data, soil_moisture_satellite_example).
        force_split = bool(harmonized_sources)
        fig_scatter = plot_scatter(pair_ds, sar_var, val_var, force_split=force_split)
        if fig_scatter is not None:
            if cdf_matched_suffix:
                fig_scatter = _mark_cdf_matched(fig_scatter)
            figs.append(fig_scatter)
            title = f"{sar_var} vs {val_var} — scatter{cdf_matched_suffix}"
            if base_dir is not None:
                _write_page(title, _finalize_figure_for_report(fig_scatter, None))

        # Summary table (immediately before the statistics bar charts, so
        # a reader has the exact numbers before the visual comparison —
        # applies to every recipe type, not just soil moisture, since this
        # loop is shared).
        if stats_ds_map and key in stats_ds_map:
            fig_table = plot_summary_table(stats_ds_map[key])
            if fig_table is not None:
                figs.append(fig_table)
                title = f"{sar_var} vs {val_var} — summary table"
                if base_dir is not None:
                    _write_page(title, _finalize_figure_for_report(fig_table, None))

        # Statistics
        if stats_ds_map and key in stats_ds_map:
            fig_stats = plot_statistics(stats_ds_map[key])
            if fig_stats is not None:
                figs.append(fig_stats)
                title = f"{sar_var} vs {val_var} — statistics"
                if base_dir is not None:
                    _write_page(title, _finalize_figure_for_report(fig_stats, None))

        # Residuals
        fig_res = plot_residuals(
            pair_ds, sar_var, val_var,
            hist_range=(
                _volumetric_hist_range_overrides(pair_ds) or None
                if cdf_matched_suffix else None
            ),
        )
        if fig_res is not None:
            if cdf_matched_suffix:
                fig_res = _mark_cdf_matched(fig_res)
            figs.append(fig_res)
            title = f"{sar_var} vs {val_var} — residuals{cdf_matched_suffix}"
            if base_dir is not None:
                _write_page(title, _finalize_figure_for_report(fig_res, None))

        # Scatter colored by temporal offset, and temporal offset vs.
        # residual magnitude — both meaningless for soil_moisture, whose
        # ISMN/satellite values are pre-averaged over a ±12h window around
        # each SAR overpass (validation_temporal_averaging_minutes /
        # §8.6-§8.7 in design-choices.md) rather than matched to a single
        # raw timestamp, so per-pair temporal offset no longer explains
        # anything about the residual.
        if variable != "soil_moisture":
            # Scatter colored by temporal offset — same SAR-vs-validation
            # comparison as above, but colored by how far apart in time each
            # pair was matched, to help explain a lower-than-expected r.
            fig_scatter_offset = plot_scatter(
                pair_ds, sar_var, val_var, color_by="temporal_offset", force_split=force_split,
            )
            if fig_scatter_offset is not None:
                figs.append(fig_scatter_offset)
                title = f"{sar_var} vs {val_var} — scatter (colored by temporal offset)"
                if base_dir is not None:
                    _write_page(title, _finalize_figure_for_report(fig_scatter_offset, None))

            # Temporal offset vs. residual magnitude
            fig_offset = plot_temporal_offset(pair_ds, sar_var, val_var)
            if fig_offset is not None:
                figs.append(fig_offset)
                title = f"{sar_var} vs {val_var} — residual vs. temporal offset"
                if base_dir is not None:
                    _write_page(title, _finalize_figure_for_report(fig_offset, None))

        # Native-units section: same plot functions, applied to the raw
        # (non-rescaled) geo_pair_ds restricted to val_source groups whose
        # units already match SAR's — see run_statistics_native_units. Only
        # ever populated for soil_moisture; a no-op for every other variable.
        if variable == "soil_moisture" and native_units_stats_ds_map and key in native_units_stats_ds_map:
            nu_stats = native_units_stats_ds_map[key]
            matching_sources = [str(s) for s in nu_stats["source"].values]
            nu_mask = geo_pair_ds["val_source"].isin(matching_sources)
            nu_pair_ds = geo_pair_ds.where(nu_mask, drop=True)

            # Geographic first, matching the CDF-matched section's order
            # above (§9.3) -- spatial context before the point-cloud
            # comparison, for the same reason in both sections.
            try:
                fig_nu_geo_result = plot_geographic(
                    datatree, nu_pair_ds, sar_var, val_var, scenes=matched_scenes,
                    point_size=geo_point_size,
                    geographic_bounds=(
                        recipe.config.geographic_bounds if geo_clamp_bounds else None
                    ),
                    # nu_pair_ds is already row-filtered to val_source
                    # groups sharing SAR's own units family (see
                    # matching_sources above) -- every source present is
                    # domain-compatible with SAR by construction, even
                    # though val_<var>'s column-level units attr may still
                    # read the stale "mixed — see val_units" sentinel
                    # inherited from the pre-filter geo_pair_ds (row
                    # filtering via .where(..., drop=True) doesn't
                    # recompute column attrs). Without this, plot_geographic
                    # would misread that stale attrs string as a genuine
                    # units mismatch and wrongly fall back to two separate
                    # colorbars.
                    skip_domain_harmonization=True,
                )
                if isinstance(fig_nu_geo_result, dict):
                    for group, fig_nu_geo in fig_nu_geo_result.items():
                        if fig_nu_geo is not None:
                            fig_nu_geo = _mark_native_units(fig_nu_geo)
                            figs.append(fig_nu_geo)
                            title = f"{sar_var} vs {val_var} — native units — geographic [{group}]"
                            if base_dir is not None:
                                _write_page(title, _finalize_figure_for_report(fig_nu_geo, None))
                elif fig_nu_geo_result is not None:
                    fig_nu_geo_result = _mark_native_units(fig_nu_geo_result)
                    figs.append(fig_nu_geo_result)
                    title = f"{sar_var} vs {val_var} — native units — geographic"
                    if base_dir is not None:
                        _write_page(title, _finalize_figure_for_report(fig_nu_geo_result, None))
            except Exception as exc:
                logger.warning("plot_geographic failed for native-units %s: %s", sar_var, exc)

            fig_nu_scatter = plot_scatter(nu_pair_ds, sar_var, val_var)
            if fig_nu_scatter is not None:
                fig_nu_scatter = _mark_native_units(fig_nu_scatter)
                figs.append(fig_nu_scatter)
                title = f"{sar_var} vs {val_var} — native units — scatter"
                if base_dir is not None:
                    _write_page(title, _finalize_figure_for_report(fig_nu_scatter, None))

            fig_nu_residuals = plot_residuals(nu_pair_ds, sar_var, val_var)
            if fig_nu_residuals is not None:
                fig_nu_residuals = _mark_native_units(fig_nu_residuals)
                figs.append(fig_nu_residuals)
                title = f"{sar_var} vs {val_var} — native units — residuals"
                if base_dir is not None:
                    _write_page(title, _finalize_figure_for_report(fig_nu_residuals, None))

            fig_nu_stats = plot_statistics(nu_stats)
            if fig_nu_stats is not None:
                fig_nu_stats = _mark_native_units(fig_nu_stats)
                figs.append(fig_nu_stats)
                title = f"{sar_var} vs {val_var} — native units — statistics"
                if base_dir is not None:
                    _write_page(title, _finalize_figure_for_report(fig_nu_stats, None))

        all_figures[key] = figs

        # When base_dir is not None, each fig here was already closed inside
        # _finalize_figure_for_report. _write_page's page Figures are separate
        # lightweight image-page objects, not these same objects — so this
        # loop is a safe, idempotent no-op double-close. Closing here
        # deregisters them from pyplot's global figure manager to avoid
        # accumulating figures across pairs and across the two
        # collocation-method passes triggered by
        # --layer-vs-layer-collocation-method both.
        if base_dir is not None:
            for fig in figs:
                plt.close(fig)

    # Combined PDF — saved alongside the validation_statistics_*.nc files.
    # Only opened (via _open_pdf, above) if at least one page was actually
    # written, matching the previous "no file unless there's content" behavior.
    if pdf_cm is not None:
        pdf_cm.__exit__(None, None, None)
        logger.info("PDF report saved to %s", pdf_path)

    return all_figures