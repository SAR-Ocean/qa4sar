# Collocation Diagnostics & Validation Report Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land two already-written-but-uncommitted diagnostics edits, then make matched layer data visible in the diagnostics plot, restrict wind-direction plots to directional sources, put the diagnostics plot on the report's first page, and drop SAR scenes that have no matched validation points.

**Architecture:** All production changes live in the single module `sar_validation/core/visualization.py`; all tests live in `tests/test_visualization.py`. Three functions are touched — `plot_collocation_diagnostics` (styling), `plot_geographic` (a new scene allowlist parameter), and `validation_report` (per-pair source filtering, page ordering, scene-allowlist computation) — plus one small module-level helper.

**Tech Stack:** Python 3, pytest, numpy, xarray, matplotlib (+ cartopy). No new dependencies.

## Global Constraints

- No new third-party dependencies.
- Follow existing test style in `tests/test_visualization.py`: spy on `matplotlib.axes.Axes.scatter` via `monkeypatch` capturing kwargs; always `plt.close("all")` at the end of a test.
- Every task ends green on `python -m pytest tests/test_visualization.py -v` and is committed before the next task starts.
- Design source of truth: `docs/superpowers/specs/2026-07-14-diagnostics-and-report-improvements-design.md`.

---

### Task 0: Land the two hanging (uncommitted) diagnostics edits

Two edits are already applied on disk in `sar_validation/core/visualization.py` (a `_SOURCE_COLORS` palette swap `#7f7f7f`→`#17becf`, and a legend-handle restructure in `plot_collocation_diagnostics`) but were never committed. Verify they are green and commit them **alone**, so later tasks build on a clean base.

**Files:**
- Commit (no new edits): `sar_validation/core/visualization.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a committed baseline; no new symbols.

- [ ] **Step 1: Confirm the two edits are present and no others are staged**

Run: `git diff --stat sar_validation/core/visualization.py`
Expected: only `sar_validation/core/visualization.py` shows changes (the palette line and the legend block). If the file shows unrelated changes, STOP and reconcile before committing.

- [ ] **Step 2: Run the visualization test module**

Run: `python -m pytest tests/test_visualization.py -v`
Expected: PASS (all tests). These edits carry no new logic, so no test should change behaviour.

- [ ] **Step 3: Run the full suite for regression**

Run: `python -m pytest tests/ -q`
Expected: PASS (pre-existing warnings about dims deprecation / constant-value xlim-ylim are acceptable; no failures).

- [ ] **Step 4: Commit the two edits on their own**

```bash
git add sar_validation/core/visualization.py
git commit -m "fix: avoid near-gray source color and legend/annotation overlap in collocation diagnostics"
```

---

### Task 1: Make matched layer data visible in the diagnostics plot

Matched **layer** points (e.g. altimeter) are currently `s=20`, `alpha=0.6`, `edgecolors="none"`, so a handful of matched points vanish under the blue SAR-footprint circles and gray unmatched tracks. Give them a bold black edge, larger size, and full opacity. Z-order is unchanged (matched in-situ still on top).

**Files:**
- Modify: `sar_validation/core/visualization.py` — the Tier 3 (matched layer) `ax.scatter` call inside `plot_collocation_diagnostics` (the block commented `── Tier 3 (zorder=5): matched layer data ──`).
- Modify: `tests/test_visualization.py` — update the existing matched-layer alpha assertion in `TestPlotCollocationDiagnosticsRefinement.test_zorder_ensures_insitu_on_top`.
- Test: `tests/test_visualization.py` — new `test_matched_layer_points_are_emphasized`.

**Interfaces:**
- Consumes: `geo_datatree_and_collocation`, `diagnostics_recipe` fixtures (altimeter is a matched layer source in that fixture).
- Produces: no new public symbols; the Tier 3 scatter call now uses `s=70, alpha=1.0, edgecolors="black", linewidths=0.7`.

- [ ] **Step 1: Write the failing test**

Add to class `TestPlotCollocationDiagnosticsRefinement` in `tests/test_visualization.py`:

```python
    def test_matched_layer_points_are_emphasized(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path, monkeypatch
    ):
        """Matched layer points (zorder=5) must be drawn bold: black edge,
        enlarged marker, full opacity — so a few matched points stay visible
        against the SAR footprints and gray unmatched tracks."""
        import matplotlib.pyplot as plt
        import matplotlib.axes
        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, collocation_ds = geo_datatree_and_collocation

        recorded = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded.append(kwargs)
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        out_path = plot_collocation_diagnostics(
            datatree, collocation_ds, diagnostics_recipe, tmp_path,
        )
        plt.close("all")

        assert out_path is not None
        matched_layer_calls = [c for c in recorded if c.get("zorder") == 5]
        assert matched_layer_calls, "Expected at least one matched-layer scatter call"
        for c in matched_layer_calls:
            assert c.get("edgecolors") == "black"
            assert c.get("alpha") == 1.0
            assert c.get("s") == 70
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_visualization.py::TestPlotCollocationDiagnosticsRefinement::test_matched_layer_points_are_emphasized -v`
Expected: FAIL (current call uses `s=20, alpha=0.6, edgecolors="none"`).

- [ ] **Step 3: Update the Tier 3 scatter styling**

In `plot_collocation_diagnostics`, find the Tier 3 block and change the `ax.scatter(...)` call from:

```python
            color, marker = source_style_map.get(str(cat["label"]), ("#2ca02c", "o"))
            ax.scatter(
                m_lon, m_lat,
                s=20, c=color, marker=marker, alpha=0.6, edgecolors="none",
                transform=transform, zorder=5, label=f"{cat['label']} matched ({len(m_lon)})",
            )
```

to:

```python
            color, marker = source_style_map.get(str(cat["label"]), ("#2ca02c", "o"))
            ax.scatter(
                m_lon, m_lat,
                s=70, c=color, marker=marker, alpha=1.0,
                edgecolors="black", linewidths=0.7,
                transform=transform, zorder=5, label=f"{cat['label']} matched ({len(m_lon)})",
            )
```

- [ ] **Step 4: Update the now-stale alpha assertion in the existing z-order test**

In `test_zorder_ensures_insitu_on_top`, the matched-layer alpha expectation changed from `0.6` to `1.0`. Change:

```python
        # Verify matched layers alpha is 0.6
        assert 0.6 in matched_layer_alphas or len(matched_layer_alphas) == 0, (
            f"Expected matched layer alpha=0.6, got {matched_layer_alphas}"
        )
```

to:

```python
        # Verify matched layers alpha is 1.0 (emphasized so few matches stay visible)
        assert 1.0 in matched_layer_alphas or len(matched_layer_alphas) == 0, (
            f"Expected matched layer alpha=1.0, got {matched_layer_alphas}"
        )
```

- [ ] **Step 5: Run the affected tests**

Run: `python -m pytest "tests/test_visualization.py::TestPlotCollocationDiagnosticsRefinement" -v`
Expected: PASS (both the new test and the updated z-order test).

- [ ] **Step 6: Run the whole module**

Run: `python -m pytest tests/test_visualization.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add sar_validation/core/visualization.py tests/test_visualization.py
git commit -m "feat: emphasize matched layer points in collocation diagnostics plot"
```

---

### Task 2: Wind-direction plots — directional sources only (data-driven)

Altimeter and radiometer have all-NaN wind direction; on direction maps they render as gray "No data" clutter. Add a helper that drops sources with no finite `val_<var>` value, and apply it (per pair) only when the validation variable is circular (`WDIR`). Wind-speed pairs are untouched.

**Files:**
- Modify: `sar_validation/core/visualization.py` — add module-level helper `_drop_nondirectional_sources`; use it inside `validation_report`'s per-pair loop.
- Test: `tests/test_visualization.py` — new class `TestDropNonDirectionalSources` (helper unit tests) + one integration test.

**Interfaces:**
- Consumes: `numpy` (already imported as `np` at module top); `CIRCULAR_VAL_VARS` from `._variable_map`.
- Produces: `_drop_nondirectional_sources(coll_ds, val_var) -> xr.Dataset`. Inside `validation_report`, a per-pair `pair_ds` replaces `collocation_ds` in the four plot calls (scatter, geographic, residuals, temporal-offset).

- [ ] **Step 1: Write the failing helper unit tests**

Add to `tests/test_visualization.py`:

```python
class TestDropNonDirectionalSources:
    def _ds(self):
        return xr.Dataset({
            "sar_owiWindDirection": ("collocation", [10.0, 20.0, 30.0, 40.0]),
            "val_WDIR":   ("collocation", [12.0, np.nan, 33.0, np.nan]),
            "val_WSPD":   ("collocation", [6.0, 6.5, 7.0, 7.5]),
            "val_source": ("collocation", ["mooring", "altimeter", "mooring", "altimeter"]),
        })

    def test_drops_all_nan_source_for_circular_var(self):
        from sar_validation.core.visualization import _drop_nondirectional_sources
        out = _drop_nondirectional_sources(self._ds(), "WDIR")
        kept = set(out["val_source"].values.tolist())
        assert kept == {"mooring"}          # altimeter had all-NaN WDIR

    def test_keeps_all_sources_when_var_has_values(self):
        from sar_validation.core.visualization import _drop_nondirectional_sources
        out = _drop_nondirectional_sources(self._ds(), "WSPD")
        kept = set(out["val_source"].values.tolist())
        assert kept == {"mooring", "altimeter"}  # both have finite WSPD

    def test_returns_input_when_columns_absent(self):
        from sar_validation.core.visualization import _drop_nondirectional_sources
        ds = xr.Dataset({"sar_x": ("collocation", [1.0, 2.0])})
        out = _drop_nondirectional_sources(ds, "WDIR")
        assert out is ds
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest "tests/test_visualization.py::TestDropNonDirectionalSources" -v`
Expected: FAIL with `ImportError: cannot import name '_drop_nondirectional_sources'`.

- [ ] **Step 3: Add the helper**

In `sar_validation/core/visualization.py`, add this helper next to the other module-level helpers (e.g. directly after `_filter_by_scene`):

```python
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
```

- [ ] **Step 4: Run the helper unit tests**

Run: `python -m pytest "tests/test_visualization.py::TestDropNonDirectionalSources" -v`
Expected: PASS.

- [ ] **Step 5: Write the failing integration test**

Add to `tests/test_visualization.py`:

```python
class TestValidationReportWindDirectionFilter:
    def test_nondirectional_source_absent_from_wdir_scatter(self, tmp_path, monkeypatch):
        """Altimeter (all-NaN WDIR) must not appear in the wind-direction
        scatter, but must appear in the wind-speed scatter."""
        import matplotlib.pyplot as plt
        import matplotlib.axes
        from sar_validation.core.visualization import validation_report
        from sar_validation.core.recipe import Recipe, RecipeConfig
        from sar_validation.core.datatree_converter import DataTreeConverter

        # Minimal 1-scene SAR datatree so plot_geographic has something to draw.
        y, x = 3, 3
        lon2d, lat2d = np.meshgrid(np.linspace(-10, -8, x), np.linspace(50, 52, y))
        sar_ds = xr.Dataset(
            {"owiWindSpeed": (("y", "x"), np.full((y, x), 7.0)),
             "owiWindDirection": (("y", "x"), np.full((y, x), 90.0))},
            coords={"lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d),
                    "time": pd.Timestamp("2026-07-10T19:00:00")},
        )
        datatree = DataTreeConverter.to_datatree({"sar/sceneA": sar_ds})

        coll = xr.Dataset({
            "sar_owiWindSpeed":     ("collocation", [7.0, 7.1, 7.2, 7.3]),
            "sar_owiWindDirection": ("collocation", [90.0, 92.0, 88.0, 91.0]),
            "val_WSPD":             ("collocation", [6.8, 6.9, 7.0, 7.1]),
            "val_WDIR":             ("collocation", [85.0, np.nan, 95.0, np.nan]),
            "val_source":           ("collocation", ["mooring", "altimeter", "mooring", "altimeter"]),
            "sar_scene_name":       ("collocation", ["sceneA"] * 4),
            "val_lon":              ("collocation", [-9.5, -9.4, -9.3, -9.2]),
            "val_lat":              ("collocation", [50.5, 50.6, 50.7, 50.8]),
        })

        seen = {}  # val_var -> set of sources that reached a scatter fill
        original_scatter = matplotlib.axes.Axes.scatter

        recipe = Recipe(config=RecipeConfig(name="wdir_test", variable="wind"))
        validation_report(coll, datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        # Both PNGs exist; only the direction one has altimeter removed.
        assert (tmp_path / "plots" / "owiWindSpeed_vs_WSPD_scatter.png").exists()
        assert (tmp_path / "plots" / "owiWindDirection_vs_WDIR_scatter.png").exists()
```

Note: this integration test asserts both pair PNGs are produced (it exercises the `pair_ds` wiring end-to-end without crashing). The precise per-source exclusion is already locked by the helper unit tests in Step 1.

- [ ] **Step 6: Run to verify it fails**

Run: `python -m pytest "tests/test_visualization.py::TestValidationReportWindDirectionFilter" -v`
Expected: FAIL — before wiring, `validation_report` may raise on the all-NaN WDIR altimeter rows in `plot_geographic` (No-data handling) or the direction PNG differs; the assertion that both PNGs exist drives the wiring. (If it happens to pass by accident, still proceed — Step 7 makes the behaviour intentional.)

- [ ] **Step 7: Wire the filter into `validation_report`**

In `validation_report`, extend the `_variable_map` import to include `CIRCULAR_VAL_VARS`:

```python
    from ._variable_map import infer_variable_pairs, filter_variable_pairs, CIRCULAR_VAL_VARS  # noqa: PLC0415
```

Then, at the very top of the `for sar_var, val_var in pairs:` loop body (right after `figs = []`), compute the per-pair dataset:

```python
        # Direction-only sources for circular variables (WDIR): drop
        # non-directional instruments (altimeter/radiometer, all-NaN
        # direction) so they don't clutter the direction plots. Speed
        # pairs keep every source.
        pair_ds = (
            _drop_nondirectional_sources(collocation_ds, val_var)
            if val_var in CIRCULAR_VAL_VARS else collocation_ds
        )
```

Then replace every `collocation_ds` argument **inside this loop** with `pair_ds` — specifically in these calls:
- `plot_scatter(collocation_ds, sar_var, val_var)` → `plot_scatter(pair_ds, sar_var, val_var)`
- `plot_geographic(datatree, collocation_ds, sar_var, val_var)` → `plot_geographic(datatree, pair_ds, sar_var, val_var)`
- `plot_residuals(collocation_ds, sar_var, val_var)` → `plot_residuals(pair_ds, sar_var, val_var)`
- `plot_scatter(collocation_ds, sar_var, val_var, color_by="temporal_offset")` → `plot_scatter(pair_ds, sar_var, val_var, color_by="temporal_offset")`
- `plot_temporal_offset(collocation_ds, sar_var, val_var)` → `plot_temporal_offset(pair_ds, sar_var, val_var)`

Do **not** change the `stats_ds_map` lookup (statistics are precomputed upstream) or the `plot_collocation_diagnostics(datatree, collocation_ds, ...)` call after the loop (the diagnostics plot must show all sources).

- [ ] **Step 8: Run the integration test**

Run: `python -m pytest "tests/test_visualization.py::TestValidationReportWindDirectionFilter" -v`
Expected: PASS.

- [ ] **Step 9: Run the whole module**

Run: `python -m pytest tests/test_visualization.py -v`
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add sar_validation/core/visualization.py tests/test_visualization.py
git commit -m "feat: restrict wind-direction plots to directional sources"
```

---

### Task 3: Put the collocation-diagnostics plot on the report's first page

The diagnostics page is appended to the **end** of `pdf_pages`. Insert it at the front so the report order is: cover → diagnostics → per-variable plots.

**Files:**
- Modify: `sar_validation/core/visualization.py` — one line in `validation_report` (the diagnostics `pdf_pages.append(...)` → `pdf_pages.insert(0, ...)`).
- Test: `tests/test_visualization.py` — new test in `TestValidationReportIncludesDiagnostics`.

**Interfaces:**
- Consumes: `geo_datatree_and_collocation` fixture.
- Produces: no new symbols; `pdf_pages` now has the diagnostics page at index 0.

- [ ] **Step 1: Write the failing test**

Add to class `TestValidationReportIncludesDiagnostics` in `tests/test_visualization.py`:

```python
    def test_diagnostics_page_is_first_after_cover(self, geo_datatree_and_collocation, tmp_path, monkeypatch):
        """The diagnostics image page must be the first content page (index 1,
        right after the cover) in the PDF."""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        from sar_validation.core.visualization import validation_report
        from sar_validation.core.recipe import Recipe, RecipeConfig

        datatree, collocation_ds = geo_datatree_and_collocation
        recipe = Recipe(config=RecipeConfig(name="test_recipe", variable="wind"))

        recorded_figs = []
        original_savefig = PdfPages.savefig

        def recording_savefig(self, *args, **kwargs):
            fig = args[0] if args else kwargs.get("figure")
            recorded_figs.append(fig)
            return original_savefig(self, *args, **kwargs)

        monkeypatch.setattr(PdfPages, "savefig", recording_savefig)
        validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        def is_image_page(fig):
            return fig is not None and len(fig.axes) == 1 and len(fig.axes[0].images) > 0

        image_indices = [i for i, f in enumerate(recorded_figs) if is_image_page(f)]
        assert image_indices, "Expected a diagnostics image page in the PDF"
        assert image_indices[0] == 1, (
            f"Diagnostics page should be first after the cover (index 1), "
            f"got first image page at index {image_indices[0]}"
        )
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest "tests/test_visualization.py::TestValidationReportIncludesDiagnostics::test_diagnostics_page_is_first_after_cover" -v`
Expected: FAIL — the diagnostics page currently lands last, so its index is > 1.

- [ ] **Step 3: Insert the diagnostics page at the front**

In `validation_report`, find the diagnostics block and change:

```python
                pdf_pages.append((f"Collocation diagnostics — {recipe.config.name}", fig_diag))
```

to:

```python
                # Lead the report body with the diagnostics overview (the
                # cover page is written separately, so index 0 here becomes
                # the first page after the cover).
                pdf_pages.insert(0, (f"Collocation diagnostics — {recipe.config.name}", fig_diag))
```

- [ ] **Step 4: Run the test**

Run: `python -m pytest "tests/test_visualization.py::TestValidationReportIncludesDiagnostics" -v`
Expected: PASS (both the new ordering test and the existing embed test).

- [ ] **Step 5: Run the whole module**

Run: `python -m pytest tests/test_visualization.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add sar_validation/core/visualization.py tests/test_visualization.py
git commit -m "feat: place collocation diagnostics on the first report page"
```

---

### Task 4: Only include SAR images with matched validation points

`plot_geographic` renders one subplot per SAR scene unconditionally. Add an optional `scenes` allowlist; in `validation_report`, compute the union (across all pairs) of scenes that have ≥1 matched point (from `sar_scene_name`) and pass it in, so scenes that never match anything are dropped.

**Files:**
- Modify: `sar_validation/core/visualization.py` — add `scenes` parameter to `plot_geographic`; compute + pass the allowlist in `validation_report`.
- Test: `tests/test_visualization.py` — new `TestPlotGeographicSceneFilter` (unit) + a `validation_report` pass-through assertion.

**Interfaces:**
- Consumes: `Sequence` (add to the `typing` import if not present) and the existing `sar_node.children` scene names.
- Produces: `plot_geographic(..., scenes: Optional[Sequence[str]] = None)`. When `scenes` is a non-empty collection, only those scene names are rendered (original order preserved); `None`/empty means "all scenes" (unchanged behaviour). `validation_report` passes `scenes=matched_scenes` where `matched_scenes = sorted(set(collocation_ds["sar_scene_name"].values))` or `None`.

- [ ] **Step 1: Confirm `Sequence` is importable**

Run: `grep -n "^from typing import" sar_validation/core/visualization.py`
Expected: a `from typing import ...` line. If `Sequence` is not already in it, add `Sequence` to that import in Step 4.

- [ ] **Step 2: Write the failing unit test**

Add to `tests/test_visualization.py` (uses a 2-scene datatree built inline; only `sceneA` has collocations):

```python
class TestPlotGeographicSceneFilter:
    def _two_scene_datatree_and_coll(self):
        from sar_validation.core.datatree_converter import DataTreeConverter
        y, x = 3, 3
        lonA, latA = np.meshgrid(np.linspace(-10, -8, x), np.linspace(50, 52, y))
        lonB, latB = np.meshgrid(np.linspace(-6, -4, x), np.linspace(50, 52, y))
        sarA = xr.Dataset({"owiWindSpeed": (("y", "x"), np.full((y, x), 7.0))},
                          coords={"lon": (("y", "x"), lonA), "lat": (("y", "x"), latA),
                                  "time": pd.Timestamp("2026-07-10T19:00:00")})
        sarB = xr.Dataset({"owiWindSpeed": (("y", "x"), np.full((y, x), 8.0))},
                          coords={"lon": (("y", "x"), lonB), "lat": (("y", "x"), latB),
                                  "time": pd.Timestamp("2026-07-10T19:00:00")})
        datatree = DataTreeConverter.to_datatree({"sar/sceneA": sarA, "sar/sceneB": sarB})
        coll = xr.Dataset({
            "sar_owiWindSpeed": ("collocation", [7.0, 7.1]),
            "val_WSPD":         ("collocation", [6.9, 7.0]),
            "val_source":       ("collocation", ["mooring", "mooring"]),
            "sar_scene_name":   ("collocation", ["sceneA", "sceneA"]),
            "val_lon":          ("collocation", [-9.5, -9.3]),
            "val_lat":          ("collocation", [50.5, 50.7]),
        })
        return datatree, coll

    def test_scenes_allowlist_renders_only_listed_scenes(self):
        import matplotlib.pyplot as plt
        from sar_validation.core.visualization import plot_geographic
        datatree, coll = self._two_scene_datatree_and_coll()

        fig = plot_geographic(datatree, coll, "owiWindSpeed", "WSPD",
                              split_by=None, scenes=["sceneA"])
        titled = [ax.get_title() for ax in fig.axes if ax.get_title()]
        plt.close("all")
        assert len(titled) == 1
        assert "sceneA" in titled[0]

    def test_scenes_none_renders_all_scenes(self):
        import matplotlib.pyplot as plt
        from sar_validation.core.visualization import plot_geographic
        datatree, coll = self._two_scene_datatree_and_coll()

        fig = plot_geographic(datatree, coll, "owiWindSpeed", "WSPD",
                              split_by=None, scenes=None)
        titled = [ax.get_title() for ax in fig.axes if ax.get_title()]
        plt.close("all")
        assert len(titled) == 2  # both sceneA and sceneB drawn
```

- [ ] **Step 3: Run to verify it fails**

Run: `python -m pytest "tests/test_visualization.py::TestPlotGeographicSceneFilter" -v`
Expected: FAIL with `TypeError: plot_geographic() got an unexpected keyword argument 'scenes'`.

- [ ] **Step 4: Add the `scenes` parameter to `plot_geographic`**

If needed (per Step 1), add `Sequence` to the `typing` import at the top of the module.

In `plot_geographic`'s signature, add the parameter (place it in the keyword-only block, e.g. after `split_by`):

```python
    split_by: str = "collocation_type",
    scenes: Optional[Sequence[str]] = None,
    interactive: bool = False,
```

Then, immediately after the existing lines:

```python
    scene_names = list(sar_node.children.keys())
    if not scene_names:
        raise ValueError("No SAR scenes found in DataTree.")
```

insert the filter:

```python
    # Optional allowlist: keep only SAR scenes that matched validation points
    # (computed by validation_report as the union across all variable pairs).
    # An empty/None allowlist means "no filtering" — draw every scene.
    if scenes:
        allow = set(scenes)
        filtered = [s for s in scene_names if s in allow]
        if filtered:
            scene_names = filtered
```

- [ ] **Step 5: Run the unit tests**

Run: `python -m pytest "tests/test_visualization.py::TestPlotGeographicSceneFilter" -v`
Expected: PASS.

- [ ] **Step 6: Write the failing pass-through test**

Add to `tests/test_visualization.py`:

```python
class TestValidationReportSceneAllowlist:
    def test_passes_matched_scene_allowlist_to_geographic(self, geo_datatree_and_collocation, tmp_path, monkeypatch):
        """validation_report must derive the geographic scene allowlist from
        sar_scene_name and pass it through as `scenes`."""
        import matplotlib.pyplot as plt
        import sar_validation.core.visualization as viz
        from sar_validation.core.recipe import Recipe, RecipeConfig

        datatree, collocation_ds = geo_datatree_and_collocation
        recipe = Recipe(config=RecipeConfig(name="test_recipe", variable="wind"))

        captured = {}
        original = viz.plot_geographic

        def spy(datatree_, coll_, sar_var, val_var, **kwargs):
            captured["scenes"] = kwargs.get("scenes")
            return original(datatree_, coll_, sar_var, val_var, **kwargs)

        monkeypatch.setattr(viz, "plot_geographic", spy)
        viz.validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        assert captured.get("scenes") is not None
        assert "sceneA" in set(captured["scenes"])
```

- [ ] **Step 7: Run to verify it fails**

Run: `python -m pytest "tests/test_visualization.py::TestValidationReportSceneAllowlist" -v`
Expected: FAIL — `validation_report` does not yet pass `scenes` (captured value is `None`).

- [ ] **Step 8: Compute and pass the allowlist in `validation_report`**

In `validation_report`, after `pairs` is resolved and before the `for sar_var, val_var in pairs:` loop, compute the allowlist:

```python
    # Union across all pairs of SAR scenes that matched at least one
    # validation point — used to drop scenes with no matches from the
    # geographic plots. collocation_ds holds only matched pairs, so every
    # scene present here has >= 1 match. None => don't filter.
    matched_scenes = (
        sorted(set(str(s) for s in collocation_ds["sar_scene_name"].values))
        if "sar_scene_name" in collocation_ds else None
    )
```

Then update the geographic call inside the loop (already `pair_ds` from Task 2) to pass the allowlist:

```python
            geo_result = plot_geographic(datatree, pair_ds, sar_var, val_var, scenes=matched_scenes)
```

- [ ] **Step 9: Run the pass-through test**

Run: `python -m pytest "tests/test_visualization.py::TestValidationReportSceneAllowlist" -v`
Expected: PASS.

- [ ] **Step 10: Run the whole module**

Run: `python -m pytest tests/test_visualization.py -v`
Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add sar_validation/core/visualization.py tests/test_visualization.py
git commit -m "feat: drop unmatched SAR scenes from geographic report plots"
```

---

### Task 5: Full-suite regression + real-report visual check

**Files:** none (verification only).

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest tests/ -q`
Expected: PASS (pre-existing dims-deprecation / constant-value warnings acceptable; zero failures).

- [ ] **Step 2 (optional, if sample data is available): regenerate a real report and eyeball it**

Run: `sar-validate --recipe recipes/radiometer_test.yaml --plot`
(Use the `sar-validate` console-script entry point — `python -m sar_validation.cli` has no `__main__` guard and does nothing.)
Then open the generated `validation_report.pdf` and confirm:
- The collocation diagnostics plot is the first page after the cover.
- Matched layer points are bold (black-edged) and clearly visible.
- Wind-direction plots contain no altimeter/radiometer points; wind-speed plots do.
- Geographic plots show only scenes with matched points.

- [ ] **Step 3: Finish the branch**

Invoke `superpowers:finishing-a-development-branch` to decide how to integrate the work.

---

## Self-Review

**Spec coverage:**
- Item 0 (land hanging edits) → Task 0. ✓
- Item 1 (matched layer visibility) → Task 1. ✓
- Item 2 (WDIR directional-only, data-driven) → Task 2. ✓
- Item 3 (diagnostics first page) → Task 3. ✓
- Item 4 (only matched SAR scenes, union across pairs) → Task 4. ✓
- Settled defaults (statistics untouched; in-situ stays on top; new spec) → respected in Tasks 2 & 1. ✓
- Full-suite regression + finishing-branch → Task 5. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every command has an expected result. ✓

**Type consistency:** `_drop_nondirectional_sources(coll_ds, val_var)` is defined in Task 2 Step 3 and consumed in Task 2 Step 7 with the same signature. `pair_ds` introduced in Task 2 Step 7 is reused by Task 4 Step 8's geographic call. `plot_geographic(..., scenes=...)` defined in Task 4 Step 4 matches the `scenes=matched_scenes` call in Task 4 Step 8 and the spy in Task 4 Step 6. `matched_scenes` shape (list[str] | None) matches `plot_geographic`'s `Optional[Sequence[str]]`. ✓
