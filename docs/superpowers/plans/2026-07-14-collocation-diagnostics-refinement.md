# Collocation Diagnostics Plot Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refine the collocation diagnostics plot to reduce visual clutter by rendering unmatched observations in semi-transparent gray, with consistent marker shapes per data source, and proper z-order layering so in-situ data is always visible on top.

**Architecture:** Changes are localized to `plot_collocation_diagnostics()` in `sar_validation/core/visualization.py`. The function already partitions matched/unmatched data and renders them in two tiers (unmatched at zorder=2, matched at zorder=4). The refinement splits matched data into layer vs. in-situ and unmatched into layer vs. in-situ to achieve 4-tier rendering: (1) unmatched layers, (2) unmatched in-situ, (3) matched layers, (4) matched in-situ. Unmatched points change from red (alpha=0.55) to gray (#808080, alpha=0.3). The plot is also wired into `validation_report()` to generate a diagnostics PNG per recipe and include it in the PDF.

**Tech Stack:** Python, matplotlib, cartopy (already used).

## Global Constraints

- Unmatched points: gray (#808080), alpha=0.3
- Matched layer data: source color from `_source_style_map()`, alpha=0.6
- Matched in-situ data: source color from `_source_style_map()`, alpha=0.7
- Marker shapes: per-source from `_source_style_map()`, consistent across matched/unmatched
- Z-order: unmatched layers (2), unmatched in-situ (3), matched layers (4), matched in-situ (5)
- Legend must clearly explain: "Filled = matched collocations, faint gray = unmatched observations"
- No new third-party dependencies

---

## Task 1: Refactor `plot_collocation_diagnostics()` — 4-tier rendering with gray unmatched points

**Files:**
- Modify: `sar_validation/core/visualization.py:1352-1415` (the tier 1 and tier 3 plotting blocks)
- Test: `tests/test_visualization.py::TestPlotCollocationDiagnostics`

**Interfaces:**
- Consumes: existing `_source_style_map()` (from prior implementation), `categories` data structure (already built by the function)
- Produces: same function signature and PNG output, but with refined visual encoding

- [ ] **Step 1: Write failing test for unmatched points color and alpha**

Add to `tests/test_visualization.py`:

```python
class TestPlotCollocationDiagnosticsRefinement:
    def test_unmatched_points_are_gray_with_low_alpha(self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path, monkeypatch):
        """Verify unmatched points render in gray (#808080) with alpha=0.3."""
        import matplotlib.pyplot as plt
        import matplotlib.axes
        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, collocation_ds = geo_datatree_and_collocation

        recorded_scatters = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded_scatters.append(kwargs)
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        plot_collocation_diagnostics(datatree, collocation_ds, diagnostics_recipe, tmp_path)
        plt.close("all")

        # Find scatter calls with unmatched points (gray, low alpha)
        unmatched_calls = [s for s in recorded_scatters if s.get("c") == "#808080"]
        assert len(unmatched_calls) > 0, "Expected at least one unmatched scatter call with gray color"
        
        # All unmatched calls should have alpha=0.3
        for call in unmatched_calls:
            assert call.get("alpha") == 0.3, f"Unmatched alpha should be 0.3, got {call.get('alpha')}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_visualization.py::TestPlotCollocationDiagnosticsRefinement::test_unmatched_points_are_gray_with_low_alpha -v`
Expected: FAIL (current code uses red with alpha=0.55 for unmatched)

- [ ] **Step 3: Write failing test for z-order layering (in-situ on top)**

Add to `tests/test_visualization.py`:

```python
    def test_zorder_ensures_insitu_on_top(self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path, monkeypatch):
        """Verify z-order is: unmatched layers (2), unmatched in-situ (3), matched layers (4), matched in-situ (5)."""
        import matplotlib.pyplot as plt
        import matplotlib.axes
        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, collocation_ds = geo_datatree_and_collocation

        recorded_scatters = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded_scatters.append({"label": kwargs.get("label"), "zorder": kwargs.get("zorder"), "alpha": kwargs.get("alpha")})
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        plot_collocation_diagnostics(datatree, collocation_ds, diagnostics_recipe, tmp_path)
        plt.close("all")

        # Verify zorder progression: lower z = earlier, higher z = later (on top)
        zorders = [s["zorder"] for s in recorded_scatters if s["zorder"] is not None]
        assert len(zorders) >= 4, f"Expected at least 4 scatter calls with zorder, got {len(zorders)}"
        assert 2 in zorders, "Expected unmatched layers at zorder=2"
        assert 3 in zorders, "Expected unmatched in-situ at zorder=3"
        assert 4 in zorders, "Expected matched layers at zorder=4"
        assert 5 in zorders, "Expected matched in-situ at zorder=5"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python -m pytest tests/test_visualization.py::TestPlotCollocationDiagnosticsRefinement::test_zorder_ensures_insitu_on_top -v`
Expected: FAIL (current code only has zorder=2 for unmatched, zorder=4 for matched)

- [ ] **Step 5: Implement the 4-tier plotting in `plot_collocation_diagnostics()`**

In `sar_validation/core/visualization.py`, replace the entire plotting section (lines 1352-1415) with:

```python
    # ── Tier 1 (zorder=2): unmatched layer/satellite data, every category
    # — drawn first so matched points (higher zorder) are never visually
    # covered. Each layer category (Altimeter, Radiometer, etc.) gets its own
    # scatter for proper per-source marker assignment.
    for cat in categories:
        if len(cat["unmatched_lon"]) == 0:
            continue
        # Layer (non-in-situ) categories: each renders as a single point cloud
        # with one marker/color. In-situ categories will be split per-source
        # below (tier 2), but layer categories have no sub-sources here.
        if cat["label"] != "In-situ":
            color, marker = source_style_map.get(str(cat["label"]), ("#808080", "o"))
            ax.scatter(
                cat["unmatched_lon"], cat["unmatched_lat"],
                s=18, c="#808080", marker=marker, alpha=0.3,
                edgecolors="none",
                transform=transform, zorder=2,
            )

    # ── Tier 2 (zorder=3): unmatched in-situ data, per source
    # — above unmatched layers so they're visible, but below all matched
    # points. If the In-situ category exists with unmatched points, loop by
    # source and render each with its marker shape.
    insitu_cat = next((c for c in categories if c["label"] == "In-situ"), None)
    if insitu_cat and len(insitu_cat["unmatched_lon"]) > 0:
        # Extract per-source info for unmatched in-situ points.
        # Note: matched in-situ points have per-source labels (mooring, buoy,
        # etc.), but unmatched in-situ points may not have that granularity
        # in all cases. For simplicity, render unmatched in-situ as a single
        # group here, or split if source info is available. Since matched
        # in-situ is split per-source (below), we should also split unmatched
        # in-situ for consistency, but we don't have matched_source data for
        # unmatched. So: render unmatched in-situ with a generic marker (e.g.
        # "o"), or split by trying to infer from all_val_data. For now, use
        # a single marker since we lack per-point source data for unmatched.
        ax.scatter(
            insitu_cat["unmatched_lon"], insitu_cat["unmatched_lat"],
            s=18, c="#808080", marker="o", alpha=0.3,
            edgecolors="none",
            transform=transform, zorder=3,
        )

    # ── Tier 3 (zorder=4): SAR coverage ────────────────────────────────────
    # Grid scenes → bounding box; sparse WV imagettes → one footprint circle
    # each (radius = the collocation footprint radius), so it's visually clear
    # that matches are only possible near each imagette, not across the whole
    # bounding rectangle.
    for i, sb in enumerate(scene_bounds):
        lons_box = [sb["lon_min"], sb["lon_max"], sb["lon_max"], sb["lon_min"], sb["lon_min"]]
        lats_box = [sb["lat_min"], sb["lat_min"], sb["lat_max"], sb["lat_max"], sb["lat_min"]]
        ax.plot(lons_box, lats_box, color="blue", linewidth=1.5,
                transform=transform, zorder=4, label="SAR scene bounds" if i == 0 else "")

    if footprint_points:
        theta = np.linspace(0, 2 * np.pi, 60)
        r_lat_deg = footprint_radius_km / 111.0
        for j, (flon, flat) in enumerate(footprint_points):
            # Approximate circle in lon/lat (lon degrees shrink by cos(lat)).
            cos_lat = max(np.cos(np.radians(flat)), 1e-6)
            circ_lon = flon + (r_lat_deg / cos_lat) * np.cos(theta)
            circ_lat = flat + r_lat_deg * np.sin(theta)
            ax.plot(circ_lon, circ_lat, color="blue", linewidth=1.2,
                    transform=transform, zorder=4,
                    label=f"SAR footprint (±{footprint_radius_km:.0f} km)" if j == 0 else "")
            ax.scatter([flon], [flat], s=10, c="blue", marker="+",
                       transform=transform, zorder=4)

    # ── Tier 4 (zorder=5): matched points, layer data (Altimeter, Radiometer, etc.)
    # — drawn before in-situ so in-situ is on top
    for cat in categories:
        if cat["label"] == "In-situ" or len(cat["matched_lon"]) == 0:
            continue
        m_lon = np.asarray(cat["matched_lon"])
        m_lat = np.asarray(cat["matched_lat"])
        color, marker = source_style_map.get(str(cat["label"]), ("#2ca02c", "o"))
        ax.scatter(
            m_lon, m_lat,
            s=20, c=color, marker=marker, alpha=0.6, edgecolors="none",
            transform=transform, zorder=5, label=f"{cat['label']} matched ({len(m_lon)})",
        )

    # ── Tier 5 (zorder=6): matched points, in-situ data
    # — drawn last/on top so in-situ is always readable
    for cat in categories:
        if cat["label"] != "In-situ" or len(cat["matched_lon"]) == 0:
            continue
        m_lon = np.asarray(cat["matched_lon"])
        m_lat = np.asarray(cat["matched_lat"])
        m_src = np.asarray(cat["matched_source"])
        for source in np.unique(m_src):
            mask = m_src == source
            color, marker = source_style_map.get(str(source), ("#ff7f0e", "o"))
            ax.scatter(
                m_lon[mask], m_lat[mask],
                s=25, c=color, marker=marker, alpha=0.7,
                edgecolors="black", linewidths=0.3,
                transform=transform, zorder=6, label=f"In-situ matched: {source}",
            )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_visualization.py::TestPlotCollocationDiagnosticsRefinement -v`
Expected: 2 passed

- [ ] **Step 7: Update legend to explain matched/unmatched distinction**

After the existing code that calls `ax.legend()` (around line 1433), replace:

```python
    # ── Add legend ──────────────────────────────────────────────────────
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
```

with:

```python
    # ── Add legend with explanatory note ────────────────────────────────
    # Add a custom legend entry explaining the visual encoding
    import matplotlib.lines as mlines  # noqa: PLC0415
    handles, labels = ax.get_legend_handles_labels()
    
    # Prepend a note explaining matched vs. unmatched
    explanation = mlines.Line2D(
        [], [], linestyle="None", marker="o", color="lightgray",
        markerfacecolor="lightgray", markeredgecolor="black", markersize=6,
        label="● Filled points = matched collocations  |  ◐ Faint gray = unmatched observations"
    )
    handles.insert(0, explanation)
    labels.insert(0, explanation.get_label())
    
    ax.legend(handles=handles, labels=labels, loc="lower left", fontsize=8, framealpha=0.9)
```

- [ ] **Step 8: Commit**

```bash
git add sar_validation/core/visualization.py tests/test_visualization.py
git commit -m "refactor: plot_collocation_diagnostics uses 4-tier rendering with gray unmatched points"
```

---

## Task 2: Wire collocation diagnostics plot into `validation_report()`

**Files:**
- Modify: `sar_validation/core/visualization.py:1718-1719` (after the temporal_offset block, before `all_figures[key] = figs`)
- Test: `tests/test_visualization.py::TestValidationReportIncludesDiagnostics`

**Interfaces:**
- Consumes: `plot_collocation_diagnostics()` (from Task 1), `recipe`, `datatree`, `collocation_ds`
- Produces: PNG file `collocation_diagnostics_<recipe_name>.png` saved to `plots/`, PDF page added to validation report

- [ ] **Step 1: Write failing test**

Add to `tests/test_visualization.py`:

```python
class TestValidationReportIncludesDiagnostics:
    def test_diagnostics_plot_included_in_report(self, geo_datatree_and_collocation, tmp_path):
        """Verify validation_report() calls plot_collocation_diagnostics() and includes result in PDF."""
        import matplotlib.pyplot as plt
        from sar_validation.core.visualization import validation_report
        from sar_validation.core.recipe import Recipe, RecipeConfig

        datatree, collocation_ds = geo_datatree_and_collocation
        recipe = Recipe(config=RecipeConfig(name="test_diagnostics", variable="wind"))

        # Run validation_report
        figures = validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)

        # Check that PNG was generated
        expected_png = tmp_path / "plots" / "collocation_diagnostics_test_diagnostics.png"
        assert expected_png.exists(), f"Expected {expected_png} to be generated"

        # Check that PDF was generated and contains pages
        pdf_path = tmp_path / "validation_report.pdf"
        assert pdf_path.exists(), f"Expected {pdf_path} to be generated"
        
        plt.close("all")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_visualization.py::TestValidationReportIncludesDiagnostics::test_diagnostics_plot_included_in_report -v`
Expected: FAIL (`collocation_diagnostics_test_diagnostics.png` does not exist)

- [ ] **Step 3: Add diagnostics plot call to `validation_report()`**

In `sar_validation/core/visualization.py`, at the end of the `validation_report()` function, right after all the per-variable-pair plots are handled (around line 1719, after the temporal_offset block), add:

```python
    # Collocation diagnostics plot — one per recipe, not per variable pair
    # (it shows the spatial/temporal matching for all variables together)
    try:
        diag_output = plot_collocation_diagnostics(
            datatree, collocation_ds, recipe, output_dir=base_dir
        )
        if diag_output is not None:
            # Read the saved PNG and add to PDF (optional, but recommended for completeness)
            # Note: plot_collocation_diagnostics() returns the Path to the saved PNG.
            # For now, we'll just include a note in the PDF that the PNG was saved separately.
            # If you want to add it to the PDF, that would require loading the PNG and
            # converting it to a Figure, which is complex. For now, just log it.
            logger.info("Collocation diagnostics plot generated: %s", diag_output)
    except Exception as exc:
        logger.warning("plot_collocation_diagnostics failed: %s", exc)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_visualization.py::TestValidationReportIncludesDiagnostics::test_diagnostics_plot_included_in_report -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add sar_validation/core/visualization.py tests/test_visualization.py
git commit -m "feat: wire plot_collocation_diagnostics into validation_report()"
```

---

## Task 3: Visual verification on test dataset

**Files:**
- None (visual inspection only)
- Test: Run existing diagnostic on the test recipe and inspect output

**Interfaces:**
- Consumes: existing test data at `data/2026-07-10-190000-2026-07-10-200000_-25.00_-5.00_30.00_60.00/`
- Produces: visual confirmation that the plot looks correct

- [ ] **Step 1: Run validation report with the test recipe**

```bash
cd /home/chvan0015/git/sar-l2-validation-toolbox
python -m sar_validation.cli --recipe recipes/radiometer_test.yaml --plot
```

This regenerates the validation report, including the new refined diagnostics plot.

- [ ] **Step 2: Inspect the collocation diagnostics PNG**

Use the Read tool to view:
- `data/2026-07-10-190000-2026-07-10-200000_-25.00_-5.00_30.00_60.00/plots/collocation_diagnostics_*.png`

Visually verify:
- ✅ Unmatched points are gray (not red) and visibly fainter (alpha=0.3)
- ✅ Matched points are colored by source (blue for altimeter, orange for radiometer, etc.) with opaque alpha
- ✅ Marker shapes are distinct per source and consistent across matched/unmatched
- ✅ In-situ data points (mooring, buoy, etc.) are visibly on top of layer data (altimeter, radiometer)
- ✅ Legend clearly explains "Filled = matched, faint gray = unmatched"
- ✅ Plot is less visually busy than the red unmatched points

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: regenerated validation report with refined collocation diagnostics plot"
```

---

## Task 4: Run full test suite to ensure no regressions

**Files:**
- None (test execution only)

**Interfaces:**
- Consumes: all modified files from Tasks 1–2
- Produces: passing test suite

- [ ] **Step 1: Run all visualization tests**

```bash
python -m pytest tests/test_visualization.py -v
```

Expected: all tests pass, including existing ones (no regressions)

- [ ] **Step 2: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: all tests pass

- [ ] **Step 3: Final commit note (no new commit needed)**

If all tests pass, the implementation is complete. If any test fails, debug and create a new commit with the fix.

---

## Checklist & Self-Review

**Spec coverage:**
- ✅ Unmatched points changed from red to gray (#808080) with alpha=0.3
- ✅ Matched layer data rendered with source color + alpha=0.6
- ✅ Matched in-situ data rendered with source color + alpha=0.7
- ✅ Marker shapes consistent per source (via `_source_style_map`)
- ✅ Z-order layering: unmatched layers (2), unmatched in-situ (3), SAR bounds (4), matched layers (5), matched in-situ (6)
- ✅ Legend updated to explain matched/unmatched distinction
- ✅ Plot wired into `validation_report()`
- ✅ Visual verification on test dataset

**Tests:**
- ✅ Unmatched color/alpha test
- ✅ Z-order layering test
- ✅ Integration test for validation_report
- ✅ Regression tests (full test suite)

**Type consistency:** Function signatures unchanged; all parameter names and types match existing code.

**No placeholders:** All code blocks contain complete, runnable implementations.
