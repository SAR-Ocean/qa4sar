# Cartopy / Report-Rendering Performance Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut `validation_report()` generation time (with `--plot`) on the
`recipes/wind_example.yaml` profiling case from 42.3s to under 20s, and the
full `--plot` routine to under 30s total, without changing the 10m coastline
resolution or the visual content of the report.

**Architecture:** Two independent fixes in
`sar_validation/core/visualization.py`:
1. Replace cartopy's expensive gridliner label placement
   (`ax.gridlines(draw_labels=True, ...)`) with plain matplotlib tick labels
   formatted via `cartopy.mpl.ticker.LongitudeFormatter`/`LatitudeFormatter`
   — valid because every plot here uses the rectangular `ccrs.PlateCarree()`
   projection.
2. Stop rendering every report figure twice (once for its standalone PNG,
   once again for the combined PDF page) by rendering each figure to a PNG
   buffer exactly once and reusing that raster for both outputs — the same
   trick `plot_collocation_diagnostics` already uses, generalized to the
   other five figure types built in `validation_report`.

**Tech Stack:** Python, matplotlib, cartopy 0.25, pytest.

## Global Constraints

- Do not change the 10m Natural Earth coastline resolution
  (`_land_coastline_features`) — it's an intentional prior tradeoff.
- No visible change to rendered report content: gridlines, degree-labeled
  axes, and PDF page content must look the same as before this fix.
- PNG/PDF output filenames and `validation_report`'s return type
  (`Dict[str, list[Figure]]`) are unchanged — no public API changes.
- dpi=150 is used consistently for every PNG/PDF page in this module; keep
  that convention.

---

### Task 1: Replace gridliner labels with plain ticks in `plot_geographic`

**Files:**
- Modify: `sar_validation/core/visualization.py:164-172` (add helper after
  `_land_coastline_features`), `sar_validation/core/visualization.py:614`
  (swap the gridlines call inside `plot_geographic`'s `_build_figure`)
- Test: `tests/test_visualization.py`

**Interfaces:**
- Produces: `_set_lonlat_ticks(ax) -> None` — module-level function in
  `sar_validation/core/visualization.py`. Applies
  `LongitudeFormatter`/`LatitudeFormatter` to `ax.xaxis`/`ax.yaxis`. Safe to
  call on any `ccrs.PlateCarree()` GeoAxes, regardless of whether data has
  been plotted on it yet (tick positions are resolved lazily at draw time).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_visualization.py`, immediately after the `TestPlotGeographic`
class (currently lines 358-539, ending right before `class TestPlotCollocationDiagnostics:`
at line 540):

```python
class TestPlotGeographicTicks:
    def test_subplots_get_degree_formatted_ticks(self, geo_datatree_and_collocation):
        """Regression test for the gridliner -> plain-tick swap: subplots
        must still show degree-labeled lon/lat ticks, just without
        gridliner's expensive label-placement machinery."""
        import matplotlib.pyplot as plt
        from sar_validation.core.visualization import plot_geographic

        datatree, collocation_ds = geo_datatree_and_collocation
        # This fixture has no "collocation_type" field, so split_by=None
        # (matching TestPlotGeographic's existing convention) makes
        # plot_geographic return a single Figure rather than a dict.
        fig = plot_geographic(
            datatree, collocation_ds, "owiWindSpeed", "WSPD", split_by=None,
        )
        fig.canvas.draw()
        ax = fig.axes[0]

        xlabels = [t.get_text() for t in ax.get_xticklabels()]
        ylabels = [t.get_text() for t in ax.get_yticklabels()]
        assert any("°" in lbl for lbl in xlabels), xlabels
        assert any("°" in lbl for lbl in ylabels), ylabels
        plt.close("all")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_visualization.py::TestPlotGeographicTicks -v`
Expected: FAIL — with the current gridliner-based code, `ax.get_xticklabels()`
/`ax.get_yticklabels()` already return the axes' default plain-numeric tick
labels (e.g. `'-10.5'`, `'50.0'`) — gridliner draws its degree-formatted
labels as separate text artists outside the normal tick machinery, so the
*default* tick labels underneath are un-formatted numbers with no `°`
character, and the `any("°" in lbl ...)` assertions fail.

- [ ] **Step 3: Add the `_set_lonlat_ticks` helper**

In `sar_validation/core/visualization.py`, right after `_land_coastline_features`
(currently ending at line 172), add:

```python
def _set_lonlat_ticks(ax):
    """Cheap plain-matplotlib degree-labeled ticks for a PlateCarree
    GeoAxes — replaces cartopy's gridliner label placement
    (``draw_labels=True``), whose curved-projection label-positioning
    logic is expensive to recompute across many subplots. Only valid for
    rectangular projections (PlateCarree/Mercator), which is all this
    module uses."""
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # noqa: PLC0415

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
```

- [ ] **Step 4: Swap the gridlines call in `plot_geographic`**

In `sar_validation/core/visualization.py`, inside `plot_geographic`'s
`_build_figure` (around line 610-617), change:

```python
            if HAS_CARTOPY:
                land, coastline = _land_coastline_features()
                ax.add_feature(land, facecolor="lightgray", zorder=0, rasterized=True)
                ax.add_feature(coastline, linewidth=0.5, zorder=0, rasterized=True)
                gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
                gl.top_labels = False
                gl.right_labels = False
                transform = ccrs.PlateCarree()
```

to:

```python
            if HAS_CARTOPY:
                land, coastline = _land_coastline_features()
                ax.add_feature(land, facecolor="lightgray", zorder=0, rasterized=True)
                ax.add_feature(coastline, linewidth=0.5, zorder=0, rasterized=True)
                ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5)
                _set_lonlat_ticks(ax)
                transform = ccrs.PlateCarree()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_visualization.py::TestPlotGeographicTicks -v`
Expected: PASS

- [ ] **Step 6: Run the full existing visualization test suite to check for regressions**

Run: `python -m pytest tests/test_visualization.py -v`
Expected: all PASS (no test asserts on gridliner internals like `gl.top_labels`
— confirmed by inspection before writing this plan).

- [ ] **Step 7: Commit**

```bash
git add sar_validation/core/visualization.py tests/test_visualization.py
git commit -m "perf: replace gridliner labels with plain ticks in plot_geographic"
```

---

### Task 2: Apply the same tick fix to `plot_collocation_diagnostics`

**Files:**
- Modify: `sar_validation/core/visualization.py:1381-1387`
- Test: `tests/test_visualization.py`

**Interfaces:**
- Consumes: `_set_lonlat_ticks(ax)` from Task 1.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_visualization.py`, immediately after the
`TestPlotCollocationDiagnosticsRefinement` class (currently lines 568-811,
ending right before `class TestValidationReport:` at line 812):

```python
class TestPlotCollocationDiagnosticsTicks:
    def test_overview_plot_gets_degree_formatted_ticks(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path
    ):
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, collocation_ds = geo_datatree_and_collocation
        out_path = plot_collocation_diagnostics(
            datatree, collocation_ds, diagnostics_recipe, tmp_path,
        )
        plt.close("all")

        assert out_path is not None
        # plot_collocation_diagnostics saves its own PNG and closes its
        # figure internally, so we can only verify the rendered image
        # exists and is non-trivial in size (a proxy for "axes labels were
        # drawn"), not inspect live tick-label Text objects.
        img = mpimg.imread(str(out_path))
        assert img.shape[0] > 100 and img.shape[1] > 100
```

Note: `plot_collocation_diagnostics` closes its figure before returning (it
only returns a path), so this test can't directly inspect tick label text
like Task 1's test does. To actually verify the formatter is applied, add a
second, more direct test using a monkeypatch spy on `_set_lonlat_ticks`
itself:

```python
    def test_overview_plot_calls_set_lonlat_ticks(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path, monkeypatch
    ):
        import matplotlib.pyplot as plt
        import sar_validation.core.visualization as viz

        datatree, collocation_ds = geo_datatree_and_collocation

        calls = []
        original = viz._set_lonlat_ticks

        def spy(ax):
            calls.append(ax)
            return original(ax)

        monkeypatch.setattr(viz, "_set_lonlat_ticks", spy)
        viz.plot_collocation_diagnostics(
            datatree, collocation_ds, diagnostics_recipe, tmp_path,
        )
        plt.close("all")

        assert len(calls) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_visualization.py::TestPlotCollocationDiagnosticsTicks -v`
Expected: `test_overview_plot_calls_set_lonlat_ticks` FAILs with
`AttributeError` or `assert 0 == 1` (function doesn't exist yet / isn't
called yet at this call site). `test_overview_plot_gets_degree_formatted_ticks`
may already pass since it only checks image dimensions — that's fine, it's
a smoke test, not the regression guard.

- [ ] **Step 3: Swap the gridlines call in `plot_collocation_diagnostics`**

In `sar_validation/core/visualization.py`, around lines 1377-1387, change:

```python
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
```

to:

```python
    # Add coastlines and features
    land, coastline = _land_coastline_features()
    ax.add_feature(land, facecolor="lightgray", alpha=0.3, zorder=0)
    ax.add_feature(coastline, linewidth=0.5, zorder=0)
    ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5)

    # ── Set plot extent to the recipe's geographic bounds ────────────────
    ax.set_extent([bounds.min_lon, bounds.max_lon, bounds.min_lat, bounds.max_lat],
                  crs=ccrs.PlateCarree())
    _set_lonlat_ticks(ax)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_visualization.py::TestPlotCollocationDiagnosticsTicks -v`
Expected: PASS

- [ ] **Step 5: Run the full existing visualization test suite**

Run: `python -m pytest tests/test_visualization.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add sar_validation/core/visualization.py tests/test_visualization.py
git commit -m "perf: replace gridliner labels with plain ticks in plot_collocation_diagnostics"
```

---

### Task 3: Add render-once helpers (`_image_page_figure`, `_finalize_figure_for_report`)

**Files:**
- Modify: `sar_validation/core/visualization.py` (insert helpers right after
  the `# 5. Validation report (convenience wrapper)` section header, around
  line 1696-1699, before `def validation_report(`)
- Test: `tests/test_visualization.py`

**Interfaces:**
- Produces:
  - `_image_page_figure(img: numpy.ndarray, dpi: int = 150) -> matplotlib.figure.Figure`
    — builds a throwaway Figure sized to match `img`'s pixel dimensions at
    `dpi`, with a single borderless `Axes` filled with `imshow(img)`.
  - `_finalize_figure_for_report(fig: matplotlib.figure.Figure, png_path: Optional[pathlib.Path], dpi: int = 150) -> matplotlib.figure.Figure`
    — renders `fig` to an in-memory PNG buffer exactly once, closes `fig`,
    optionally writes the buffer's bytes to `png_path`, and returns
    `_image_page_figure(...)` built from that same buffer.

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/test_visualization.py` (currently 1115 lines):

```python
class TestImagePageFigure:
    def test_figure_size_matches_image_pixel_dimensions(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from sar_validation.core.visualization import _image_page_figure

        img = np.zeros((300, 450, 3), dtype=np.uint8)
        fig = _image_page_figure(img, dpi=150)

        w_in, h_in = fig.get_size_inches()
        assert w_in == pytest.approx(450 / 150)
        assert h_in == pytest.approx(300 / 150)
        plt.close(fig)


class TestFinalizeFigureForReport:
    def test_writes_png_closes_original_returns_image_page(self, tmp_path):
        import matplotlib.pyplot as plt
        import matplotlib._pylab_helpers as pylab_helpers
        from sar_validation.core.visualization import _finalize_figure_for_report

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        png_path = tmp_path / "out.png"

        page_fig = _finalize_figure_for_report(fig, png_path)

        assert png_path.exists()
        assert png_path.stat().st_size > 0
        # Check object identity against pyplot's live figure registry, not
        # `fig.number` — matplotlib recycles figure numbers after `close()`,
        # so a freshly created page figure can legitimately reuse the
        # original's number; that would make an `fignum_exists()`-based
        # check pass even if `fig` were never actually closed.
        open_figs = [m.canvas.figure for m in pylab_helpers.Gcf.figs.values()]
        assert fig not in open_figs, "original figure should be closed"
        assert page_fig is not fig
        plt.close(page_fig)

    def test_none_png_path_skips_disk_write(self, tmp_path):
        import matplotlib.pyplot as plt
        from sar_validation.core.visualization import _finalize_figure_for_report

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        page_fig = _finalize_figure_for_report(fig, None)

        assert page_fig is not None
        assert list(tmp_path.iterdir()) == []
        plt.close(page_fig)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_visualization.py::TestImagePageFigure tests/test_visualization.py::TestFinalizeFigureForReport -v`
Expected: FAIL with `ImportError: cannot import name '_image_page_figure'`
(and same for `_finalize_figure_for_report`) — neither exists yet.

- [ ] **Step 3: Implement the helpers**

In `sar_validation/core/visualization.py`, right after the
`# 5. Validation report (convenience wrapper)` section header comment
(currently immediately before `def validation_report(`), add:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_visualization.py::TestImagePageFigure tests/test_visualization.py::TestFinalizeFigureForReport -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sar_validation/core/visualization.py tests/test_visualization.py
git commit -m "perf: add render-once helpers for report PDF pages"
```

---

### Task 4: Rewire `validation_report` to render each figure only once

**Files:**
- Modify: `sar_validation/core/visualization.py:1786-1936`
  (the six `pdf_pages.append(...)` call sites, the
  `plot_collocation_diagnostics` PDF-embedding block, and the final
  PDF-writing loop)
- Test: `tests/test_visualization.py`

**Interfaces:**
- Consumes: `_finalize_figure_for_report`, `_image_page_figure` from Task 3.
- `validation_report`'s public signature and return type
  (`Dict[str, list[Figure]]`) are unchanged.

This task's TDD cycle is ordered differently from Tasks 1-3: the rewiring
in Steps 1-2 below *introduces* a figure leak (new lightweight page figures
that nothing closes yet), and the regression test in Step 3 is what proves
that leak exists before Step 4 fixes it. Do the steps in this exact order.

- [ ] **Step 1: Rewire the six pdf_pages call sites**

In `sar_validation/core/visualization.py`, replace lines 1786-1863
(everything from the `# Scatter` comment through the temporal-offset
`if plots_dir:` block) with:

```python
        # Scatter
        fig_scatter = plot_scatter(pair_ds, sar_var, val_var)
        if fig_scatter is not None:
            figs.append(fig_scatter)
            title = f"{sar_var} vs {val_var} — scatter"
            if plots_dir:
                png_path = plots_dir / f"{key}{filename_suffix}_scatter.png"
                pdf_pages.append((title, _finalize_figure_for_report(fig_scatter, png_path)))
            else:
                pdf_pages.append((title, fig_scatter))

        # Geographic — returns dict[collocation_type, Figure] by default
        try:
            geo_result = plot_geographic(datatree, pair_ds, sar_var, val_var, scenes=matched_scenes)
            if isinstance(geo_result, dict):
                for group, fig_geo in geo_result.items():
                    if fig_geo is not None:
                        figs.append(fig_geo)
                        title = f"{sar_var} vs {val_var} — geographic [{group}]"
                        if plots_dir:
                            safe_group = str(group).replace("/", "-")
                            png_path = plots_dir / f"{key}{filename_suffix}_geographic_{safe_group}.png"
                            pdf_pages.append((title, _finalize_figure_for_report(fig_geo, png_path)))
                        else:
                            pdf_pages.append((title, fig_geo))
            elif geo_result is not None:
                figs.append(geo_result)
                title = f"{sar_var} vs {val_var} — geographic"
                if plots_dir:
                    png_path = plots_dir / f"{key}{filename_suffix}_geographic.png"
                    pdf_pages.append((title, _finalize_figure_for_report(geo_result, png_path)))
                else:
                    pdf_pages.append((title, geo_result))
        except Exception as exc:
            logger.warning("plot_geographic failed for %s: %s", sar_var, exc)

        # Statistics
        if stats_ds_map and key in stats_ds_map:
            fig_stats = plot_statistics(stats_ds_map[key])
            if fig_stats is not None:
                figs.append(fig_stats)
                title = f"{sar_var} vs {val_var} — statistics"
                if plots_dir:
                    png_path = plots_dir / f"{key}{filename_suffix}_statistics.png"
                    pdf_pages.append((title, _finalize_figure_for_report(fig_stats, png_path)))
                else:
                    pdf_pages.append((title, fig_stats))

        # Residuals
        fig_res = plot_residuals(pair_ds, sar_var, val_var)
        if fig_res is not None:
            figs.append(fig_res)
            title = f"{sar_var} vs {val_var} — residuals"
            if plots_dir:
                png_path = plots_dir / f"{key}{filename_suffix}_residuals.png"
                pdf_pages.append((title, _finalize_figure_for_report(fig_res, png_path)))
            else:
                pdf_pages.append((title, fig_res))

        # Scatter colored by temporal offset — same SAR-vs-validation
        # comparison as above, but colored by how far apart in time each
        # pair was matched, to help explain a lower-than-expected r.
        fig_scatter_offset = plot_scatter(pair_ds, sar_var, val_var, color_by="temporal_offset")
        if fig_scatter_offset is not None:
            figs.append(fig_scatter_offset)
            title = f"{sar_var} vs {val_var} — scatter (colored by temporal offset)"
            if plots_dir:
                png_path = plots_dir / f"{key}{filename_suffix}_scatter_by_offset.png"
                pdf_pages.append((title, _finalize_figure_for_report(fig_scatter_offset, png_path)))
            else:
                pdf_pages.append((title, fig_scatter_offset))

        # Temporal offset vs. residual magnitude
        fig_offset = plot_temporal_offset(pair_ds, sar_var, val_var)
        if fig_offset is not None:
            figs.append(fig_offset)
            title = f"{sar_var} vs {val_var} — residual vs. temporal offset"
            if plots_dir:
                png_path = plots_dir / f"{key}{filename_suffix}_temporal_offset.png"
                pdf_pages.append((title, _finalize_figure_for_report(fig_offset, png_path)))
            else:
                pdf_pages.append((title, fig_offset))
```

- [ ] **Step 2: Simplify the `plot_collocation_diagnostics` embedding block**

In `sar_validation/core/visualization.py`, around lines 1877-1902, change:

```python
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
```

to:

```python
    # Collocation diagnostics plot — generated once per recipe
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
                # Lead the report body with the diagnostics overview (the
                # cover page is written separately, so index 0 here becomes
                # the first page after the cover).
                pdf_pages.insert(
                    0,
                    (f"Collocation diagnostics — {recipe.config.name}", _image_page_figure(diag_img)),
                )
        except Exception as exc:
            logger.warning("plot_collocation_diagnostics failed: %s", exc)
```

- [ ] **Step 3: Write the regression test and confirm it currently fails**

Add to `tests/test_visualization.py`, immediately after the
`TestValidationReportIncludesDiagnostics` class (currently lines 831-919,
ending right before `class TestDropNonDirectionalSources:` at line 920):

```python
class TestValidationReportClosesPageFigures:
    def test_no_figures_left_open_after_report(self, geo_datatree_and_collocation, tmp_path):
        """Regression guard for the render-once refactor: the new
        lightweight PDF-page figures (built by _finalize_figure_for_report /
        _image_page_figure) must be closed once written, not leaked —
        unlike the original heavy figures, they aren't tracked in
        `all_figures` / `figs`, so nothing else closes them."""
        import matplotlib.pyplot as plt
        from sar_validation.core.visualization import validation_report
        from sar_validation.core.recipe import Recipe, RecipeConfig

        datatree, collocation_ds = geo_datatree_and_collocation
        recipe = Recipe(config=RecipeConfig(name="test", variable="wind"))

        plt.close("all")
        validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)

        assert plt.get_fignums() == []
```

Run: `python -m pytest tests/test_visualization.py::TestValidationReportClosesPageFigures -v`
Expected: FAIL with a non-empty `plt.get_fignums()` list — after Steps 1-2,
every `pdf_pages` entry saved via `plots_dir` is a freshly created
lightweight page figure (from `_finalize_figure_for_report` /
`_image_page_figure`) that isn't in `figs`, so nothing closes it yet; the
still-unmodified final PDF loop only calls `pdf.savefig(fig, ...)`, never
`plt.close(fig)`.

- [ ] **Step 4: Close page figures after they're written to the PDF, and drop the now-redundant `fig_diag` close block**

In `sar_validation/core/visualization.py`, change:

```python
            for _title, fig in pdf_pages:
                pdf.savefig(fig, dpi=150, bbox_inches="tight")

        logger.info("PDF report saved to %s", pdf_path)

    # fig_diag isn't in the `figs` list closed earlier (it's created after
    # that loop), so it stays open until here — close it now that the PDF
    # write is done and it's no longer needed.
    if fig_diag is not None:
        plt.close(fig_diag)
```

to:

```python
            for _title, fig in pdf_pages:
                pdf.savefig(fig, dpi=150, bbox_inches="tight")
                plt.close(fig)

        logger.info("PDF report saved to %s", pdf_path)
```

`pdf_pages` entries are, in every case reaching this loop, either the cover
page (closed separately, unaffected) or lightweight image-only figures
produced by `_finalize_figure_for_report` / `_image_page_figure` — never the
original heavy figures already tracked (and closed) via `figs`, since
`base_dir is not None` (required for this PDF block to run) implies
`plots_dir` is also set (see lines 1745-1750), which is exactly when
`_finalize_figure_for_report` is used instead of appending the raw figure.

- [ ] **Step 5: Run the new and existing tests to verify they pass**

Run: `python -m pytest tests/test_visualization.py -v`
Expected: all PASS, including
`TestValidationReportClosesPageFigures::test_no_figures_left_open_after_report`
(now `plt.get_fignums() == []`) and every pre-existing `TestValidationReport*`
test (PNG/PDF file existence checks are unaffected — filenames and content
are unchanged).

- [ ] **Step 6: Commit**

```bash
git add sar_validation/core/visualization.py tests/test_visualization.py
git commit -m "perf: render each validation_report figure once, reuse raster for PDF page"
```

---

### Task 5: Verify the performance target (manual, not a pytest task)

**Files:** none modified — this is a measurement checkpoint.

- [ ] **Step 1: Run the profiling command used in the original investigation**

`sar-validate` is the installed console script (`pyproject.toml:41`,
`sar_validation.cli:main`). Run:

```bash
python -m cProfile -o /tmp/prof_after.out "$(which sar-validate)" \
  --recipe recipes/wind_example.yaml --collocation --collocation-log --stats --plot
```

- [ ] **Step 2: Inspect the profile**

```bash
python -c "
import pstats
p = pstats.Stats('/tmp/prof_after.out')
p.sort_stats('cumulative').print_stats(20)
"
```

Expected: `_draw_gridliner` no longer appears near the top (or its total
time has dropped sharply — it should now only be called with
`draw_labels=False`, which skips the expensive label-placement path), and
the overall `validation_report` cumulative time (grep the stats output, or
wrap the call in `time.perf_counter()` in a throwaway script) is under 20s.

- [ ] **Step 3: Confirm against the stated targets**

If `validation_report` is still at or above 20s, or the full `--plot`
routine is at or above 30s, profile again to find the next-largest
remaining cost center (candidates already identified and intentionally not
addressed by this plan: the ~4.9s of shapely-intersects clipping, and
whatever remains of the ~6.3s of Agg canvas initializations) before deciding
on further work — do not guess at further optimizations without re-profiling
first.

- [ ] **Step 4: Visual spot-check**

Open the generated `validation_report.pdf` and one `*_geographic_*.png`
from `<out_dir>/plots/` and confirm: gridlines are present, axis tick labels
show degree-formatted lon/lat values, and the PDF pages look the same as
before this change (same content, same layout).
