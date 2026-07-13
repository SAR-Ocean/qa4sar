# Validation-Plot Source Markers & Temporal Offset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `validation_report.pdf` legible when multiple validation sources are present, and surface the temporal collocation offset so a lower-than-expected correlation coefficient can be explained.

**Architecture:** All changes live in `sar_validation/core/visualization.py` (one new stable-style helper, marker shapes threaded through three existing plot functions, one new plot function, two new pages wired into `validation_report()`). No changes to `collocation.py` or `datatree_converter.py` — `temporal_distance_minutes` is already computed and written to `collocation_ds` today.

**Tech Stack:** Python, xarray, matplotlib (+ cartopy for the geographic/diagnostics plots), pytest.

## Global Constraints

- Canonical source order = `sorted(LAYER_DATA_TYPES | _INSITU_TYPES)`, where `LAYER_DATA_TYPES` comes from `sar_validation/core/collocation.py` and `_INSITU_TYPES` from `sar_validation/core/orchestrator.py` — reuse these constants, do not redefine the list.
- Style-map lookups are case-insensitive (`plot_collocation_diagnostics` uses title-cased labels like `"Altimeter"` while other call sites use lowercase `"altimeter"` — both must resolve to the same canonical index).
- Marker set: `["o", "s", "^", "D", "v", "P", "X", "*", "h"]` (9 entries). Color palette extended from 8 to 9 entries by adding `"#bcbd22"`, so colors and markers pair 1:1 by index.
- `plot_residuals` and `plot_statistics` are explicitly out of scope — no changes.
- `plot_temporal_offset`'s y-axis is `|SAR − validation|` (absolute residual), no binned/trend overlay — raw scatter only.
- No new third-party dependencies.

---

### Task 1: `_source_style_map()` — stable per-source (color, marker) mapping

**Files:**
- Modify: `sar_validation/core/visualization.py:49-60`
- Test: `tests/test_visualization.py`

**Interfaces:**
- Produces: `_source_style_map(sources: List[str]) -> Dict[str, Tuple[str, str]]` — maps each name in `sources` to a `(color, marker)` pair. Order comes from a fixed canonical list, not from what else is in `sources`, and matching is case-insensitive. Consumed by Tasks 2, 3, 4, 5, 6.
- Produces: `_SOURCE_MARKERS: List[str]` (module-level constant, 9 entries).
- Leaves `_source_color_map()` untouched (still used by `plot_residuals`, unchanged per scope).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_visualization.py`, after the existing imports (keep all existing fixtures/tests as-is):

```python
class TestSourceStyleMap:
    def test_stable_regardless_of_other_sources_present(self):
        from sar_validation.core.visualization import _source_style_map
        style_alone = _source_style_map(["altimeter"])
        style_with_others = _source_style_map(["altimeter", "radiometer", "mooring"])
        assert style_alone["altimeter"] == style_with_others["altimeter"]

    def test_distinct_known_sources_get_distinct_styles(self):
        from sar_validation.core.visualization import _source_style_map
        style = _source_style_map(["altimeter", "radiometer", "mooring", "buoy"])
        colors = [c for c, _ in style.values()]
        markers = [m for _, m in style.values()]
        assert len(set(colors)) == 4
        assert len(set(markers)) == 4

    def test_unknown_source_does_not_crash(self):
        # NOTE: with exactly 9 canonical sources and a 9-entry palette, an
        # unknown source's index (9) wraps to the same palette slot as
        # canonical index 0 ("altimeter") — this is the same "cycles if
        # more sources than colours" behavior _SOURCE_COLORS already
        # documents, not a bug, so this test only asserts "present, no
        # crash," not "visually distinct from every canonical source."
        from sar_validation.core.visualization import _source_style_map
        style = _source_style_map(["altimeter", "some_future_sensor"])
        assert "some_future_sensor" in style
        assert style["altimeter"] == ("#1f77b4", "o")

    def test_case_insensitive_matches_canonical_entry(self):
        from sar_validation.core.visualization import _source_style_map
        style_lower = _source_style_map(["altimeter"])
        style_title = _source_style_map(["Altimeter"])
        assert style_lower["altimeter"] == style_title["Altimeter"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_visualization.py::TestSourceStyleMap -v`
Expected: FAIL with `ImportError: cannot import name '_source_style_map'`

- [ ] **Step 3: Implement `_source_style_map`**

In `sar_validation/core/visualization.py`, replace lines 49-60:

```python
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
```

with:

```python
# Colour palette used for validation sources (cycles if more sources than colours)
_SOURCE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_visualization.py::TestSourceStyleMap -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add sar_validation/core/visualization.py tests/test_visualization.py
git commit -m "feat: add stable per-source (color, marker) style map for validation plots"
```

---

### Task 2: `plot_scatter` — marker per source

**Files:**
- Modify: `sar_validation/core/visualization.py:213-220`
- Test: `tests/test_visualization.py`

**Interfaces:**
- Consumes: `_source_style_map()` from Task 1.
- No change to `plot_scatter`'s public signature yet (that's Task 5).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_visualization.py`, inside `class TestPlotScatter:`:

```python
    def test_distinct_sources_get_distinct_markers(self, collocation_ds, monkeypatch):
        import matplotlib.pyplot as plt
        import matplotlib.axes

        recorded_markers = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded_markers.append(kwargs.get("marker"))
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        fig = plot_scatter(collocation_ds, "owiWindSpeed", "WSPD")
        plt.close(fig)

        assert len(recorded_markers) == 2
        assert len(set(recorded_markers)) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_visualization.py::TestPlotScatter::test_distinct_sources_get_distinct_markers -v`
Expected: FAIL (`recorded_markers` is `[None, None]` — no `marker` kwarg passed yet, so `len(set(...))` is 1, not 2)

- [ ] **Step 3: Add markers to `plot_scatter`**

In `sar_validation/core/visualization.py`, replace lines 213-220:

```python
    sources = df["val_source"].unique().tolist()
    cmap = _source_color_map(sources)

    for src in sorted(sources):
        sub = df[df["val_source"] == src]
        label = src if by_source else None
        ax.scatter(sub[val_col], sub[sar_col], s=18, alpha=0.6,
                   color=cmap[src], label=label, rasterized=True)
```

with:

```python
    sources = df["val_source"].unique().tolist()
    style = _source_style_map(sources)

    for src in sorted(sources):
        sub = df[df["val_source"] == src]
        label = src if by_source else None
        color, marker = style[src]
        ax.scatter(sub[val_col], sub[sar_col], s=18, alpha=0.6,
                   color=color, marker=marker, label=label, rasterized=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_visualization.py::TestPlotScatter -v`
Expected: all `TestPlotScatter` tests pass (including the new one)

- [ ] **Step 5: Commit**

```bash
git add sar_validation/core/visualization.py tests/test_visualization.py
git commit -m "feat: plot_scatter uses a distinct marker shape per validation source"
```

---

### Task 3: `plot_geographic` — per-source markers in the value-colormap branch + honest legend

This is the fix for the originally-reported bug: legend swatches that don't match the colormap-filled dots on the map.

**Files:**
- Modify: `sar_validation/core/visualization.py:341` and `:536-587` (see exact old/new blocks below)
- Test: `tests/test_visualization.py`

**Interfaces:**
- Consumes: `_source_style_map()` from Task 1.
- Produces test fixture `geo_datatree_and_collocation` (module-level, reused by Tasks 4 and 7).

- [ ] **Step 1: Add the shared fixture and write the failing test**

Add to `tests/test_visualization.py`, near the top (add `import pandas as pd` to the existing imports if not already present), after the existing `collocation_ds` fixture:

```python
@pytest.fixture
def geo_datatree_and_collocation():
    """Synthetic DataTree + collocation_ds with two known validation
    sources (mooring, altimeter) within one SAR scene's bounds — used to
    test plot_geographic's and plot_collocation_diagnostics' per-source
    marker handling, and validation_report's temporal-offset pages."""
    from sar_validation.core.datatree_converter import DataTreeConverter

    y, x = 4, 5
    lon2d, lat2d = np.meshgrid(np.linspace(-10.0, -8.0, x), np.linspace(50.0, 52.0, y))
    wind = np.linspace(5.0, 12.0, y * x).reshape(y, x)
    sar_ds = xr.Dataset(
        {"owiWindSpeed": (("y", "x"), wind)},
        coords={
            "lon": (("y", "x"), lon2d),
            "lat": (("y", "x"), lat2d),
            "time": pd.Timestamp("2026-07-10T19:00:00"),
        },
    )

    n = 4
    mooring_ds = xr.Dataset(
        {"WSPD": ("point", np.array([6.0, 6.5, 7.0, 7.5]))},
        coords={
            "lon": ("point", np.array([-9.8, -9.6, -9.4, -9.2])),
            "lat": ("point", np.array([50.2, 50.4, 50.6, 50.8])),
            "time": ("point", pd.date_range("2026-07-10T19:05", periods=n, freq="5min")),
        },
        attrs={"platform_type": "mooring"},
    )
    altimeter_ds = xr.Dataset(
        {"WSPD": ("point", np.array([8.0, 8.5, 9.0, 9.5]))},
        coords={
            "lon": ("point", np.array([-9.0, -8.8, -8.6, -8.4])),
            "lat": ("point", np.array([51.0, 51.2, 51.4, 51.6])),
            "time": ("point", pd.date_range("2026-07-10T19:10", periods=n, freq="5min")),
        },
        attrs={"platform_type": "altimeter"},
    )

    datatree = DataTreeConverter.to_datatree({
        "sar/sceneA": sar_ds,
        "validation/mooring": mooring_ds,
        "validation/altimeter": altimeter_ds,
    })

    collocation_ds = xr.Dataset({
        "sar_owiWindSpeed":            ("collocation", np.array([6.1, 6.9, 8.2, 9.3])),
        "val_WSPD":                    ("collocation", np.array([6.0, 7.0, 8.0, 9.5])),
        "val_source":                  ("collocation", ["mooring", "mooring", "altimeter", "altimeter"]),
        "sar_scene_name":              ("collocation", ["sceneA"] * n),
        "val_lon":                     ("collocation", np.array([-9.8, -9.6, -9.0, -8.8])),
        "val_lat":                     ("collocation", np.array([50.2, 50.4, 51.0, 51.2])),
        "val_id":                      ("collocation", ["mo0", "mo1", "al0", "al1"]),
        "temporal_distance_minutes":   ("collocation", np.array([10.0, 45.0, 90.0, 150.0])),
    })
    collocation_ds = collocation_ds.assign_coords(
        val_time=("collocation", pd.date_range("2026-07-10T19:05", periods=n, freq="5min")),
    )
    return datatree, collocation_ds


class TestPlotGeographic:
    def test_distinct_sources_get_distinct_markers(self, geo_datatree_and_collocation, monkeypatch):
        import matplotlib.pyplot as plt
        import matplotlib.axes
        from sar_validation.core.visualization import plot_geographic

        datatree, collocation_ds = geo_datatree_and_collocation

        recorded_markers = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded_markers.append(kwargs.get("marker"))
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        fig = plot_geographic(
            datatree, collocation_ds, "owiWindSpeed", "WSPD", split_by=None,
        )
        plt.close("all")

        assert fig is not None
        source_markers = [m for m in recorded_markers if m is not None]
        assert len(set(source_markers)) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_visualization.py::TestPlotGeographic -v`
Expected: FAIL (`source_markers` is empty or all `None` — the value-colormap branch currently issues one un-marked `ax.scatter()` call for all valid points, no `marker=` kwarg at all, so `set(source_markers)` has 0 or 1 entries, not 2)

- [ ] **Step 3: Split the value-colormap scatter by source and fix the legend**

In `sar_validation/core/visualization.py`, line 341, replace:

```python
    source_cmap = _source_color_map(val_sources) if val_sources else {}
```

with:

```python
    source_style = _source_style_map(val_sources) if val_sources else {}
```

Then, in the same function's folium/interactive branch, replace:

```python
                    color = source_cmap.get(str(row.get("val_source", "")), "#1f77b4")
```

with:

```python
                    color = source_style.get(str(row.get("val_source", "")), ("#1f77b4", "o"))[0]
```

Then replace the whole block from lines 536-587 (the value-colormap / non-colormap scatter + legend block):

```python
                if val_col_present and val_norm is not None:
                    nan_mask = df_pts[val_col].isna()
                    valid_pts = df_pts[~nan_mask]
                    nan_pts = df_pts[nan_mask]

                    if len(valid_pts):
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

                    handles = []
                    if "val_source" in df_pts.columns:
                        present = set(df_pts["val_source"].astype(str))
                        handles += [
                            mlines.Line2D([], [], marker="o", linestyle="None",
                                          markerfacecolor=clr, markeredgecolor="black",
                                          markersize=5, label=s)
                            for s, clr in source_cmap.items() if s in present
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
                        color = source_cmap.get(str(src), "#ff0000")
                        ax.scatter(grp["val_lon"], grp["val_lat"],
                                   s=point_size, c=color,
                                   edgecolors="black", linewidths=0.4,
                                   label=str(src), rasterized=True, **kw_sc)
                    ax.legend(fontsize=6, loc="lower left", framealpha=0.7)
                else:
                    ax.scatter(df_pts["val_lon"], df_pts["val_lat"],
                               s=point_size, c="#ff7f0e",
                               edgecolors="black", linewidths=0.4, **kw_sc)
```

with:

```python
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
                            mlines.Line2D([], [], marker=marker, linestyle="None",
                                          markerfacecolor="lightgray", markeredgecolor="black",
                                          markersize=5, label=s)
                            for s, (_, marker) in source_style.items() if s in present
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_visualization.py::TestPlotGeographic -v`
Expected: 1 passed

- [ ] **Step 5: Visually verify the legend fix on a rendered PNG**

Run this from the repo root to render the fixture data and inspect it:

```bash
python -c "
import matplotlib
matplotlib.use('Agg')
import numpy as np, pandas as pd, xarray as xr
from sar_validation.core.datatree_converter import DataTreeConverter
from sar_validation.core.visualization import plot_geographic

y, x = 4, 5
lon2d, lat2d = np.meshgrid(np.linspace(-10.0, -8.0, x), np.linspace(50.0, 52.0, y))
wind = np.linspace(5.0, 12.0, y * x).reshape(y, x)
sar_ds = xr.Dataset(
    {'owiWindSpeed': (('y', 'x'), wind)},
    coords={'lon': (('y', 'x'), lon2d), 'lat': (('y', 'x'), lat2d), 'time': pd.Timestamp('2026-07-10T19:00:00')},
)
n = 4
mooring_ds = xr.Dataset(
    {'WSPD': ('point', np.array([6.0, 6.5, 7.0, 7.5]))},
    coords={'lon': ('point', np.array([-9.8, -9.6, -9.4, -9.2])),
            'lat': ('point', np.array([50.2, 50.4, 50.6, 50.8])),
            'time': ('point', pd.date_range('2026-07-10T19:05', periods=n, freq='5min'))},
    attrs={'platform_type': 'mooring'},
)
altimeter_ds = xr.Dataset(
    {'WSPD': ('point', np.array([8.0, 8.5, 9.0, 9.5]))},
    coords={'lon': ('point', np.array([-9.0, -8.8, -8.6, -8.4])),
            'lat': ('point', np.array([51.0, 51.2, 51.4, 51.6])),
            'time': ('point', pd.date_range('2026-07-10T19:10', periods=n, freq='5min'))},
    attrs={'platform_type': 'altimeter'},
)
datatree = DataTreeConverter.to_datatree({'sar/sceneA': sar_ds, 'validation/mooring': mooring_ds, 'validation/altimeter': altimeter_ds})
collocation_ds = xr.Dataset({
    'sar_owiWindSpeed': ('collocation', np.array([6.1, 6.9, 8.2, 9.3])),
    'val_WSPD':         ('collocation', np.array([6.0, 7.0, 8.0, 9.5])),
    'val_source':       ('collocation', ['mooring', 'mooring', 'altimeter', 'altimeter']),
    'sar_scene_name':   ('collocation', ['sceneA'] * n),
    'val_lon':          ('collocation', np.array([-9.8, -9.6, -9.0, -8.8])),
    'val_lat':          ('collocation', np.array([50.2, 50.4, 51.0, 51.2])),
})
fig = plot_geographic(datatree, collocation_ds, 'owiWindSpeed', 'WSPD', split_by=None)
fig.savefig('/tmp/claude-449455/-home-chvan0015-git-sar-l2-validation-toolbox/60f82032-61a5-4d1a-9045-1d8d4f1be126/scratchpad/task3_geo_check.png', dpi=120, bbox_inches='tight')
print('saved')
"
```

Then use the Read tool on `/tmp/claude-449455/-home-chvan0015-git-sar-l2-validation-toolbox/60f82032-61a5-4d1a-9045-1d8d4f1be126/scratchpad/task3_geo_check.png` and confirm: mooring and altimeter points use visibly different marker shapes, and the legend shows those same shapes (not a solid-color swatch that doesn't match the dots).

- [ ] **Step 6: Commit**

```bash
git add sar_validation/core/visualization.py tests/test_visualization.py
git commit -m "fix: plot_geographic uses per-source markers instead of a mismatched legend color"
```

---

### Task 4: `plot_collocation_diagnostics` — marker per source

**Files:**
- Modify: `sar_validation/core/visualization.py:1095` and the two scatter calls that follow it (~lines 1164-1178)
- Test: `tests/test_visualization.py`

**Interfaces:**
- Consumes: `_source_style_map()` from Task 1, `geo_datatree_and_collocation` fixture from Task 3.
- Produces test fixture `diagnostics_recipe` (module-level, reusable by any future diagnostics test).

- [ ] **Step 1: Add the recipe fixture and write the failing test**

Add to `tests/test_visualization.py`:

```python
@pytest.fixture
def diagnostics_recipe():
    from sar_validation.core.recipe import (
        GeographicBounds, Recipe, RecipeConfig, ValidationDataSource,
        CollocationType, PointVsLayerCollocation,
    )
    config = RecipeConfig(
        name="test_recipe",
        variable="wind",
        geographic_bounds=GeographicBounds(min_lon=-11.0, max_lon=-7.0, min_lat=49.0, max_lat=53.0),
        validation_sources=[
            ValidationDataSource(source_type="mooring"),
            ValidationDataSource(source_type="altimeter"),
        ],
        collocation=CollocationType(point_vs_layer=PointVsLayerCollocation(time_tolerance_minutes=30)),
    )
    return Recipe(config=config)


class TestPlotCollocationDiagnostics:
    def test_distinct_sources_get_distinct_markers(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path, monkeypatch
    ):
        import matplotlib.pyplot as plt
        import matplotlib.axes
        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, collocation_ds = geo_datatree_and_collocation

        recorded_markers = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded_markers.append(kwargs.get("marker"))
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        out_path = plot_collocation_diagnostics(
            datatree, collocation_ds, diagnostics_recipe, tmp_path,
        )
        plt.close("all")

        assert out_path is not None
        matched_markers = [m for m in recorded_markers if m is not None]
        assert len(set(matched_markers)) >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_visualization.py::TestPlotCollocationDiagnostics -v`
Expected: FAIL (matched-point scatter calls don't pass `marker=` yet, so `set(matched_markers)` has 0 or 1 entries)

- [ ] **Step 3: Add markers to the matched-point scatter calls**

In `sar_validation/core/visualization.py`, line 1095, replace:

```python
    source_color_map = _source_color_map(sorted(all_source_names))
```

with:

```python
    source_style_map = _source_style_map(sorted(all_source_names))
```

Then replace the in-situ per-source loop:

```python
            for source in np.unique(m_src):
                mask = m_src == source
                color = source_color_map.get(str(source), "#ff7f0e")
                ax.scatter(
                    m_lon[mask], m_lat[mask],
                    s=25, c=color, alpha=0.7, edgecolors="black", linewidths=0.3,
                    transform=transform, zorder=4, label=f"In-situ matched: {source}",
                )
```

with:

```python
            for source in np.unique(m_src):
                mask = m_src == source
                color, marker = source_style_map.get(str(source), ("#ff7f0e", "o"))
                ax.scatter(
                    m_lon[mask], m_lat[mask],
                    s=25, c=color, marker=marker, alpha=0.7,
                    edgecolors="black", linewidths=0.3,
                    transform=transform, zorder=4, label=f"In-situ matched: {source}",
                )
```

Then replace the non-in-situ (layer category) scatter call:

```python
        else:
            color = source_color_map.get(str(cat["label"]), "#2ca02c")
            ax.scatter(
                m_lon, m_lat,
                s=20, c=color, alpha=0.6, edgecolors="none",
                transform=transform, zorder=4, label=f"{cat['label']} matched ({len(m_lon)})",
            )
```

with:

```python
        else:
            color, marker = source_style_map.get(str(cat["label"]), ("#2ca02c", "o"))
            ax.scatter(
                m_lon, m_lat,
                s=20, c=color, marker=marker, alpha=0.6, edgecolors="none",
                transform=transform, zorder=4, label=f"{cat['label']} matched ({len(m_lon)})",
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_visualization.py::TestPlotCollocationDiagnostics -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add sar_validation/core/visualization.py tests/test_visualization.py
git commit -m "feat: plot_collocation_diagnostics uses per-source markers for matched points"
```

---

### Task 5: `plot_scatter` — `color_by="temporal_offset"` parameter

**Files:**
- Modify: `sar_validation/core/visualization.py` (function signature at line 135, docstring, and the matplotlib body from Task 2)
- Modify: `tests/test_visualization.py` (add `temporal_distance_minutes` to the shared `collocation_ds` fixture)
- Test: `tests/test_visualization.py`

**Interfaces:**
- Consumes: `_source_style_map()` from Task 1.
- Produces: `plot_scatter(..., color_by: str = "source")` — new keyword-only parameter. `by_source` keeps its existing meaning (legend/label visibility); `color_by="temporal_offset"` swaps point color from per-source to a continuous `temporal_distance_minutes` colormap, with marker shape still per-source.

- [ ] **Step 1: Extend the shared fixture and write the failing tests**

In `tests/test_visualization.py`, add a `temporal_distance_minutes` entry to the existing `collocation_ds` fixture (do not remove any existing keys):

```python
@pytest.fixture
def collocation_ds():
    """Minimal synthetic collocation dataset."""
    n = 30
    rng = np.random.default_rng(0)
    sar_vals = rng.uniform(3, 12, n)
    val_vals = sar_vals + rng.normal(0, 0.4, n)
    sources = ["mooring"] * 15 + ["scatterometer"] * 15
    scene_names = ["sceneA"] * 15 + ["sceneB"] * 15

    return xr.Dataset({
        "sar_owiWindSpeed": ("collocation", sar_vals),
        "val_WSPD":         ("collocation", val_vals),
        "val_source":       ("collocation", sources),
        "sar_scene_name":   ("collocation", scene_names),
        "sar_y_idx":        ("collocation", rng.integers(0, 10, n)),
        "sar_x_idx":        ("collocation", rng.integers(0, 10, n)),
        "sar_lon":          ("collocation", rng.uniform(-10, 5, n)),
        "sar_lat":          ("collocation", rng.uniform(50, 65, n)),
        "val_lon":          ("collocation", rng.uniform(-10, 5, n)),
        "val_lat":          ("collocation", rng.uniform(50, 65, n)),
        "temporal_distance_minutes": ("collocation", rng.uniform(0, 180, n)),
    })
```

Add a new test class:

```python
class TestPlotScatterColorByTemporalOffset:
    def test_returns_figure_with_colorbar(self, collocation_ds):
        import matplotlib.pyplot as plt
        fig = plot_scatter(collocation_ds, "owiWindSpeed", "WSPD", color_by="temporal_offset")
        assert fig is not None
        assert len(fig.axes) >= 2  # main axes + colorbar axes
        plt.close(fig)

    def test_uses_distinct_markers_per_source(self, collocation_ds, monkeypatch):
        import matplotlib.pyplot as plt
        import matplotlib.axes

        recorded_markers = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded_markers.append(kwargs.get("marker"))
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        fig = plot_scatter(collocation_ds, "owiWindSpeed", "WSPD", color_by="temporal_offset")
        plt.close(fig)

        assert len(set(recorded_markers)) == 2

    def test_missing_temporal_column_falls_back_to_source(self):
        n = 5
        ds = xr.Dataset({
            "sar_owiWindSpeed": ("collocation", [8.0, 7.0, 6.0, 9.0, 10.0]),
            "val_WSPD":         ("collocation", [7.5, 7.2, 6.1, 8.9, 9.8]),
            "val_source":       ("collocation", ["buoy"] * n),
        })
        import matplotlib.pyplot as plt
        with pytest.warns(UserWarning, match="temporal_distance_minutes"):
            fig = plot_scatter(ds, "owiWindSpeed", "WSPD", color_by="temporal_offset")
        assert fig is not None
        plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_visualization.py::TestPlotScatterColorByTemporalOffset -v`
Expected: FAIL with `TypeError: plot_scatter() got an unexpected keyword argument 'color_by'`

- [ ] **Step 3: Implement `color_by` in `plot_scatter`**

Replace the entire `plot_scatter` function body in `sar_validation/core/visualization.py` (from `def plot_scatter(` through its final `return fig`, i.e. the function as it stands after Task 2) with:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_visualization.py::TestPlotScatterColorByTemporalOffset tests/test_visualization.py::TestPlotScatter -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add sar_validation/core/visualization.py tests/test_visualization.py
git commit -m "feat: plot_scatter can color points by temporal collocation offset"
```

---

### Task 6: New `plot_temporal_offset()` function

**Files:**
- Modify: `sar_validation/core/visualization.py` (add function after `plot_residuals`, i.e. after its closing `return fig` and before the `# 4b. Collocation diagnostics plot` section header; add `"plot_temporal_offset"` to `__all__` at line 39; add one line to the module docstring's function list)
- Test: `tests/test_visualization.py`

**Interfaces:**
- Consumes: `_deduplicate_obs()`, `_source_style_map()`, `_require()`, `CIRCULAR_VAL_VARS`/`circular_diff_deg` from `_variable_map` (all already used elsewhere in this file).
- Produces: `plot_temporal_offset(collocation_ds, sar_var, val_var, *, by_source=True, interactive=False, ax=None) -> Figure | None`, consumed by Task 7.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_visualization.py`:

```python
class TestPlotTemporalOffset:
    def test_returns_figure(self, collocation_ds):
        import matplotlib.pyplot as plt
        from sar_validation.core.visualization import plot_temporal_offset
        fig = plot_temporal_offset(collocation_ds, "owiWindSpeed", "WSPD")
        assert fig is not None
        plt.close(fig)

    def test_missing_temporal_column_returns_none(self):
        import warnings
        from sar_validation.core.visualization import plot_temporal_offset
        n = 5
        ds = xr.Dataset({
            "sar_owiWindSpeed": ("collocation", [8.0, 7.0, 6.0, 9.0, 10.0]),
            "val_WSPD":         ("collocation", [7.5, 7.2, 6.1, 8.9, 9.8]),
            "val_source":       ("collocation", ["buoy"] * n),
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = plot_temporal_offset(ds, "owiWindSpeed", "WSPD")
        assert result is None

    def test_distinct_sources_get_distinct_markers(self, collocation_ds, monkeypatch):
        import matplotlib.pyplot as plt
        import matplotlib.axes
        from sar_validation.core.visualization import plot_temporal_offset

        recorded_markers = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded_markers.append(kwargs.get("marker"))
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        fig = plot_temporal_offset(collocation_ds, "owiWindSpeed", "WSPD")
        plt.close(fig)

        assert len(set(recorded_markers)) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_visualization.py::TestPlotTemporalOffset -v`
Expected: FAIL with `ImportError: cannot import name 'plot_temporal_offset'`

- [ ] **Step 3: Implement `plot_temporal_offset`**

In `sar_validation/core/visualization.py`, add `"plot_temporal_offset"` to `__all__` (line 39):

```python
__all__ = [
    "plot_scatter",
    "plot_geographic",
    "plot_statistics",
    "plot_residuals",
    "plot_temporal_offset",
    "plot_collocation_diagnostics",
    "validation_report",
]
```

Then insert this new function immediately after `plot_residuals`'s closing `return fig` (i.e. right before the `# 4b. Collocation diagnostics plot` section comment block that precedes `def plot_collocation_diagnostics`):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_visualization.py::TestPlotTemporalOffset -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add sar_validation/core/visualization.py tests/test_visualization.py
git commit -m "feat: add plot_temporal_offset (|residual| vs. temporal collocation offset)"
```

---

### Task 7: Wire both temporal-offset additions into `validation_report()` + regenerate the real example report

**Files:**
- Modify: `sar_validation/core/visualization.py:1451` (insert two new blocks right after the existing "Residuals" block, before `all_figures[key] = figs` at line 1460)
- Test: `tests/test_visualization.py`

**Interfaces:**
- Consumes: `plot_scatter(..., color_by=...)` (Task 5), `plot_temporal_offset()` (Task 6), `geo_datatree_and_collocation` fixture (Task 3).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_visualization.py`:

```python
class TestValidationReport:
    def test_includes_temporal_offset_plots(self, geo_datatree_and_collocation, tmp_path):
        import matplotlib.pyplot as plt
        from sar_validation.core.visualization import validation_report
        from sar_validation.core.recipe import Recipe, RecipeConfig

        datatree, collocation_ds = geo_datatree_and_collocation
        recipe = Recipe(config=RecipeConfig(name="test", variable="wind"))

        figures = validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)

        key = "owiWindSpeed_vs_WSPD"
        assert key in figures
        assert (tmp_path / "plots" / f"{key}_scatter_by_offset.png").exists()
        assert (tmp_path / "plots" / f"{key}_temporal_offset.png").exists()
        assert (tmp_path / "validation_report.pdf").exists()
        plt.close("all")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_visualization.py::TestValidationReport -v`
Expected: FAIL (`{key}_scatter_by_offset.png` and `{key}_temporal_offset.png` don't exist yet — `validation_report()` doesn't generate them)

- [ ] **Step 3: Wire the two new plots into `validation_report()`**

In `sar_validation/core/visualization.py`, immediately after the existing "Residuals" block (ending at line 1458 with the closing `)` of `fig_res.savefig(...)`) and before line 1460 (`all_figures[key] = figs`), insert:

```python
        # Scatter colored by temporal offset — same SAR-vs-validation
        # comparison as above, but colored by how far apart in time each
        # pair was matched, to help explain a lower-than-expected r.
        fig_scatter_offset = plot_scatter(collocation_ds, sar_var, val_var, color_by="temporal_offset")
        if fig_scatter_offset is not None:
            figs.append(fig_scatter_offset)
            pdf_pages.append((f"{sar_var} vs {val_var} — scatter (by temporal offset)", fig_scatter_offset))
            if plots_dir:
                fig_scatter_offset.savefig(
                    plots_dir / f"{key}{filename_suffix}_scatter_by_offset.png", dpi=150, bbox_inches="tight"
                )

        # Temporal offset vs. residual magnitude
        fig_offset = plot_temporal_offset(collocation_ds, sar_var, val_var)
        if fig_offset is not None:
            figs.append(fig_offset)
            pdf_pages.append((f"{sar_var} vs {val_var} — residual vs. temporal offset", fig_offset))
            if plots_dir:
                fig_offset.savefig(
                    plots_dir / f"{key}{filename_suffix}_temporal_offset.png", dpi=150, bbox_inches="tight"
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_visualization.py -v`
Expected: all tests in the file pass (full regression check across every task in this plan)

- [ ] **Step 5: Regenerate the real example report and visually confirm**

The example report at `data/2026-07-10-190000-2026-07-10-200000_-25.00_-5.00_30.00_60.00/` was built from `recipes/radiometer_test.yaml`. Its `download_metadata.json`, `datatree.nc`, and `collocation_results.nc` already exist, so re-running with `--plot` will skip steps 1-3 and only regenerate statistics + plots/PDF in place:

```bash
python -m sar_validation.cli --recipe recipes/radiometer_test.yaml --plot
```

Then inspect the newly generated PNGs with the Read tool:
- `data/2026-07-10-190000-2026-07-10-200000_-25.00_-5.00_30.00_60.00/plots/owiWindSpeed_vs_WSPD_geographic_layer_vs_layer.png` — confirm altimeter/radiometer points now use different marker shapes and the legend shapes match what's plotted (the originally reported bug).
- `data/2026-07-10-190000-2026-07-10-200000_-25.00_-5.00_30.00_60.00/plots/owiWindSpeed_vs_WSPD_scatter_by_offset.png` — confirm a colorbar labeled "Temporal offset (min)" is present and markers are distinct per source.
- `data/2026-07-10-190000-2026-07-10-200000_-25.00_-5.00_30.00_60.00/plots/owiWindSpeed_vs_WSPD_temporal_offset.png` — confirm a sensible N/r annotation and a readable residual-vs-offset scatter.

- [ ] **Step 6: Commit**

```bash
git add sar_validation/core/visualization.py tests/test_visualization.py
git commit -m "feat: add temporal-offset scatter and residual-vs-offset pages to validation_report"
```
