# Plot Step / Validation Report Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix five concrete problems in the `--plot` step's validation report, identified from real generated reports: redundant individual PNGs cluttering `plots/`, a density-normalization bug that makes the residual histogram unreadable, a linear colormap misapplied to circular wind-direction data plus a dense scatterometer swath hiding sparser sources underneath it, overlapping longitude tick labels on narrow SAR-scene subplots (plus making wave matches easier to spot), and oversized HF radar markers blanketing the SAR field on currents geographic plots.

**Architecture:** All changes are confined to `sar_validation/core/visualization.py` (five targeted edits: `validation_report`, `plot_residuals`, `plot_geographic`, `plot_collocation_diagnostics`, `_set_lonlat_ticks`) plus a one-line message update in `sar_validation/cli.py` and a documentation fix in `docs/cli-statistics-and-plots.md`. No new files, no new dependencies.

**Tech Stack:** Python, matplotlib, cartopy, xarray, pytest.

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-07-16-plot-step-report-improvements-design.md` — every task below implements one numbered item from it.
- No new individual-pair PNGs may be written to `<out_dir>/plots/`; only `collocation_diagnostics_<recipe_name><suffix>.png` may land there (from `plot_collocation_diagnostics`, unaffected by this plan).
- The wind-direction colormap fix (`cmap="twilight"`, `vmin=0, vmax=360`) applies only when `val_var in CIRCULAR_VAL_VARS` (currently just `"WDIR"`); every other variable is unaffected.
- The Tier-3 matched-layer alpha reduction (0.65) in `plot_collocation_diagnostics` applies only when `recipe.config.variable == "wind"`.
- The larger/outlined matched-marker style (`s=45, edgecolors="black"`) in `plot_collocation_diagnostics` applies only when `recipe.config.variable == "waves"`.
- The longitude/latitude tick-count cap (`nbins=4`) in `_set_lonlat_ticks` applies unconditionally (all recipes, all callers of that helper).
- The reduced HF radar marker size (`point_size=15`) passed from `validation_report` to `plot_geographic` applies only when `recipe.config.variable == "currents"`; every other variable keeps the existing default (`point_size=40`).
- Run the full test suite (`python -m pytest tests/ -q`) after every task; all tests must pass before moving to the next task.

---

### Task 1: Stop writing individual per-pair PNGs to `plots/`

**Files:**
- Modify: `sar_validation/core/visualization.py:1807-2044` (`validation_report`)
- Modify: `sar_validation/cli.py:704-708` (`_generate_plots`)
- Modify: `docs/cli-statistics-and-plots.md:56-66,158` (output-tree example + code comment)
- Test: `tests/test_visualization.py`

**Interfaces:**
- Consumes: `_finalize_figure_for_report(fig, png_path)` (visualization.py:1790) — already supports `png_path=None` to skip the disk write (verified by the existing `TestFinalizeFigureForReport.test_none_png_path_skips_disk_write` test); no change needed to this helper.
- Produces: `validation_report(...)` no longer creates or writes to a `plots_dir` variable of its own; `plot_collocation_diagnostics` (unchanged) continues to create `<out_dir>/plots/` and write `collocation_diagnostics_<recipe_name><suffix>.png` there, both at the collocate step (`cli.py:624`) and when re-embedded from inside `validation_report` (visualization.py:1991-2010).

- [ ] **Step 1: Update/add tests to lock in the new behavior**

In `tests/test_visualization.py`, remove the two stale PNG-existence assertions from `TestValidationReport.test_includes_temporal_offset_plots` (lines 969-984):

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
        assert (tmp_path / "validation_report.pdf").exists()
        plt.close("all")
```

Remove the two stale PNG-existence assertions from
`TestValidationReportWindDirectionFilter.test_nondirectional_source_absent_from_wdir_scatter`
(currently lines 1198-1200):

```python
        recipe = Recipe(config=RecipeConfig(name="wdir_test", variable="wind"))
        viz.validation_report(coll, datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        # The pair_ds wiring actually reached plot_scatter/plot_geographic
        # with the filtered dataset: altimeter (all-NaN WDIR) must never be
        # present in any dataset handed to a plot call made for the WDIR
        # pair, but must still be present for the WSPD pair, where every
        # source is kept untouched.
        wdir_sources = set().union(*captured.get("WDIR", [set()]))
        wspd_sources = set().union(*captured.get("WSPD", [set()]))
```

(i.e. delete the `# Both PNGs exist; only the direction one has altimeter removed.` comment and
the two `assert (tmp_path / "plots" / "...").exists()` lines that preceded it — everything else in
that test is unchanged.)

Add a new test class at the end of `tests/test_visualization.py`:

```python
class TestValidationReportOnlyDiagnosticsPngSaved:
    def test_plots_dir_contains_only_diagnostics_png(self, geo_datatree_and_collocation, tmp_path):
        """Every plot is already embedded in validation_report.pdf, so
        plots/ must contain only the collocation-diagnostics PNG — no
        individual scatter/geographic/statistics/residuals/temporal-offset
        PNGs."""
        import matplotlib.pyplot as plt
        from sar_validation.core.visualization import validation_report
        from sar_validation.core.recipe import Recipe, RecipeConfig

        datatree, collocation_ds = geo_datatree_and_collocation
        recipe = Recipe(config=RecipeConfig(name="test_recipe", variable="wind"))

        validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        plots_dir = tmp_path / "plots"
        png_files = sorted(p.name for p in plots_dir.glob("*.png"))
        assert png_files == ["collocation_diagnostics_test_recipe.png"]
```

- [ ] **Step 2: Run tests to verify the new/changed assertions fail**

Run: `python -m pytest tests/test_visualization.py -k "OnlyDiagnosticsPngSaved or test_includes_temporal_offset_plots or test_nondirectional_source_absent_from_wdir_scatter" -v`
Expected: `test_plots_dir_contains_only_diagnostics_png` FAILS (plots/ currently contains 17 PNGs, not 1); the other two still PASS at this point since nothing in the source has changed yet (they were only trimmed, not tightened).

- [ ] **Step 3: Remove `plots_dir` creation and individual PNG writes from `validation_report`**

In `sar_validation/core/visualization.py`, change the top of `validation_report` (lines 1851-1856):

```python
    base_dir: Optional[Path] = None
    plots_dir: Optional[Path] = None
    if out_dir is not None:
        base_dir = Path(out_dir)
        plots_dir = base_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
```

to:

```python
    base_dir: Optional[Path] = None
    if out_dir is not None:
        base_dir = Path(out_dir)
```

Change the scatter block (lines 1892-1901):

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
```

to:

```python
        # Scatter
        fig_scatter = plot_scatter(pair_ds, sar_var, val_var)
        if fig_scatter is not None:
            figs.append(fig_scatter)
            title = f"{sar_var} vs {val_var} — scatter"
            if base_dir is not None:
                pdf_pages.append((title, _finalize_figure_for_report(fig_scatter, None)))
            else:
                pdf_pages.append((title, fig_scatter))
```

Change the geographic block (lines 1903-1926):

```python
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
```

to:

```python
        # Geographic — returns dict[collocation_type, Figure] by default
        try:
            geo_result = plot_geographic(datatree, pair_ds, sar_var, val_var, scenes=matched_scenes)
            if isinstance(geo_result, dict):
                for group, fig_geo in geo_result.items():
                    if fig_geo is not None:
                        figs.append(fig_geo)
                        title = f"{sar_var} vs {val_var} — geographic [{group}]"
                        if base_dir is not None:
                            pdf_pages.append((title, _finalize_figure_for_report(fig_geo, None)))
                        else:
                            pdf_pages.append((title, fig_geo))
            elif geo_result is not None:
                figs.append(geo_result)
                title = f"{sar_var} vs {val_var} — geographic"
                if base_dir is not None:
                    pdf_pages.append((title, _finalize_figure_for_report(geo_result, None)))
                else:
                    pdf_pages.append((title, geo_result))
        except Exception as exc:
            logger.warning("plot_geographic failed for %s: %s", sar_var, exc)
```

Change the statistics block (lines 1928-1938):

```python
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
```

to:

```python
        # Statistics
        if stats_ds_map and key in stats_ds_map:
            fig_stats = plot_statistics(stats_ds_map[key])
            if fig_stats is not None:
                figs.append(fig_stats)
                title = f"{sar_var} vs {val_var} — statistics"
                if base_dir is not None:
                    pdf_pages.append((title, _finalize_figure_for_report(fig_stats, None)))
                else:
                    pdf_pages.append((title, fig_stats))
```

Change the residuals block (lines 1940-1949):

```python
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
```

to:

```python
        # Residuals
        fig_res = plot_residuals(pair_ds, sar_var, val_var)
        if fig_res is not None:
            figs.append(fig_res)
            title = f"{sar_var} vs {val_var} — residuals"
            if base_dir is not None:
                pdf_pages.append((title, _finalize_figure_for_report(fig_res, None)))
            else:
                pdf_pages.append((title, fig_res))
```

Change the scatter-by-offset block (lines 1951-1962):

```python
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
```

to:

```python
        # Scatter colored by temporal offset — same SAR-vs-validation
        # comparison as above, but colored by how far apart in time each
        # pair was matched, to help explain a lower-than-expected r.
        fig_scatter_offset = plot_scatter(pair_ds, sar_var, val_var, color_by="temporal_offset")
        if fig_scatter_offset is not None:
            figs.append(fig_scatter_offset)
            title = f"{sar_var} vs {val_var} — scatter (colored by temporal offset)"
            if base_dir is not None:
                pdf_pages.append((title, _finalize_figure_for_report(fig_scatter_offset, None)))
            else:
                pdf_pages.append((title, fig_scatter_offset))
```

Change the temporal-offset block (lines 1964-1973):

```python
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

to:

```python
        # Temporal offset vs. residual magnitude
        fig_offset = plot_temporal_offset(pair_ds, sar_var, val_var)
        if fig_offset is not None:
            figs.append(fig_offset)
            title = f"{sar_var} vs {val_var} — residual vs. temporal offset"
            if base_dir is not None:
                pdf_pages.append((title, _finalize_figure_for_report(fig_offset, None)))
            else:
                pdf_pages.append((title, fig_offset))
```

Finally, remove the trailing PNG-directory log line (lines 2041-2043):

```python
    if plots_dir:
        logger.info("PNG plots saved to %s", plots_dir)

    return all_figures
```

to:

```python
    return all_figures
```

- [ ] **Step 4: Update the CLI's post-plot message**

In `sar_validation/cli.py`, change (lines 704-708):

```python
    plots_dir = base_dir / "plots"
    pdf_path = base_dir / f"validation_report{filename_suffix}.pdf"
    print(f"  PNG plots saved to {plots_dir}")
    if pdf_path.exists():
        print(f"  PDF report saved to {pdf_path}")
```

to:

```python
    pdf_path = base_dir / f"validation_report{filename_suffix}.pdf"
    if pdf_path.exists():
        print(f"  PDF report saved to {pdf_path}")
    print(f"  Collocation diagnostics PNG saved to {base_dir / 'plots'}")
```

- [ ] **Step 5: Update the user-facing docs' output-tree example**

In `docs/cli-statistics-and-plots.md`, change (lines 56-66):

```
│  ── Step 5b ──
├── validation_report.pdf          ← combined PDF (all plots, one file)
└── plots/
    ├── owiWindSpeed_vs_WSPD_scatter.png
    ├── owiWindSpeed_vs_WSPD_geographic_point_vs_layer.png
    ├── owiWindSpeed_vs_WSPD_geographic_layer_vs_layer.png
    ├── owiWindSpeed_vs_WSPD_statistics.png
    ├── owiWindSpeed_vs_WSPD_residuals.png
    ├── owiWindDirection_vs_WDIR_scatter.png
    └── ...
```

to:

```
│  ── Step 5b ──
├── validation_report.pdf          ← combined PDF (every plot, one file)
└── plots/
    └── collocation_diagnostics_<recipe_name>.png   ← also written at Step 3
```

Change line 158:

```
# Step 5b — PNGs saved to DATA_DIR/plots/, PDF to DATA_DIR/validation_report.pdf
```

to:

```
# Step 5b — PDF report saved to DATA_DIR/validation_report.pdf (collocation
# diagnostics PNG in DATA_DIR/plots/; every other plot is PDF-only)
```

- [ ] **Step 6: Run the full test suite**

Run: `python -m pytest tests/test_visualization.py -v`
Expected: PASS (all tests, including the new `test_plots_dir_contains_only_diagnostics_png`).

Run: `python -m pytest tests/ -q`
Expected: PASS (no regressions elsewhere).

- [ ] **Step 7: Commit**

```bash
git add sar_validation/core/visualization.py sar_validation/cli.py docs/cli-statistics-and-plots.md tests/test_visualization.py
git commit -m "fix: stop writing individual per-pair PNGs, keep only collocation diagnostics in plots/"
```

---

### Task 2: Fix the residual histogram density spike with small multiples

**Files:**
- Modify: `sar_validation/core/visualization.py:937-1030` (`plot_residuals`)
- Test: `tests/test_visualization.py`

**Interfaces:**
- Consumes: `_source_color_map(sources: List[str]) -> Dict[str, str]` (visualization.py:78), `CIRCULAR_VAL_VARS` / `circular_diff_deg` from `._variable_map` (unchanged usage).
- Produces: `plot_residuals(collocation_ds, sar_var, val_var, *, by_source=True, interactive=False, ax=None)` — same signature and return type (`matplotlib.figure.Figure` or `None`), but when `by_source=True` (the default) the returned figure now contains one subplot per distinct `val_source` on a shared x-range instead of one shared axes with overlaid histograms. `ax=None` behavior for `by_source=False` is unchanged (draws into a single axes, or the passed-in `ax`). When `by_source=True` and a caller passes `ax=...`, `ax` is now ignored (a fresh multi-subplot figure is always created) — no existing caller passes `ax`, confirmed via `grep -rn "plot_residuals(" sar_validation/ tests/`.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_visualization.py`, inside `class TestPlotResiduals` (after `test_missing_var_returns_none`, currently ending at line 239):

```python
    def test_by_source_creates_one_subplot_per_source(self, collocation_ds):
        import matplotlib.pyplot as plt
        # collocation_ds fixture has 2 distinct sources: mooring, scatterometer
        fig = plot_residuals(collocation_ds, "owiWindSpeed", "WSPD")
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_by_source_false_returns_single_axes(self, collocation_ds):
        import matplotlib.pyplot as plt
        fig = plot_residuals(collocation_ds, "owiWindSpeed", "WSPD", by_source=False)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_shares_bin_range_across_sources(self, monkeypatch):
        """Regression test for the buoy-spike bug: plot_residuals used to
        call ax.hist(..., bins=30, density=True) once per source without a
        shared range, so a source with a very narrow residual spread (e.g.
        2 tightly-clustered buoy points) got a tiny bin width and a density
        spike that dwarfed every other source sharing the axes. Each source
        now gets its own subplot, but all subplots must still share one
        bin range so bars stay position-comparable."""
        import matplotlib.pyplot as plt
        import matplotlib.axes

        n_wide = 20
        rng = np.random.default_rng(1)
        sar_wide = rng.uniform(3, 12, n_wide)
        val_wide = sar_wide + rng.normal(0, 1.5, n_wide)
        sar_narrow = np.array([7.0, 7.02])
        val_narrow = np.array([6.0, 6.0])

        ds = xr.Dataset({
            "sar_owiWindSpeed": ("collocation", np.concatenate([sar_wide, sar_narrow])),
            "val_WSPD":         ("collocation", np.concatenate([val_wide, val_narrow])),
            "val_source":       ("collocation", ["mooring"] * n_wide + ["buoy"] * 2),
        })

        recorded_ranges = []
        original_hist = matplotlib.axes.Axes.hist

        def recording_hist(self, *args, **kwargs):
            recorded_ranges.append(kwargs.get("range"))
            return original_hist(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "hist", recording_hist)
        fig = plot_residuals(ds, "owiWindSpeed", "WSPD")
        plt.close(fig)

        assert len(recorded_ranges) == 2
        assert all(r is not None for r in recorded_ranges)
        assert recorded_ranges[0] == recorded_ranges[1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_visualization.py -k TestPlotResiduals -v`
Expected: `test_by_source_creates_one_subplot_per_source` FAILS (`len(fig.axes) == 1`, not 2 — current implementation draws everything into one shared axes); `test_shares_bin_range_across_sources` FAILS (`recorded_ranges` entries are `None` — current implementation never passes `range=`); `test_by_source_false_returns_single_axes` PASSES already (unaffected by the bug).

- [ ] **Step 3: Rewrite `plot_residuals` with a small-multiples layout**

In `sar_validation/core/visualization.py`, replace the entire function body from `def plot_residuals(` (line 937) through its closing `return fig` (line 1030) with:

```python
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

    if not by_source:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = ax.get_figure()
        ax.hist(df["residual"].dropna(), bins=30, density=True, alpha=0.7, color="#1f77b4")
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel(f"{sar_var} − {val_var}")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.grid(True, linewidth=0.4)
        fig.tight_layout()
        return fig

    # by_source=True: one subplot per source, sharing a common x-range but
    # each with its own y-axis (see docstring for why a shared axes breaks
    # under density=True when spreads differ wildly across sources).
    sources = sorted(df["val_source"].unique())
    color_map = _source_color_map(sources)
    residual_min = float(df["residual"].min())
    residual_max = float(df["residual"].max())
    if residual_min == residual_max:
        residual_min -= 0.5
        residual_max += 0.5
    shared_range = (residual_min, residual_max)

    ncols = 2
    nrows = math.ceil(len(sources) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for idx, src in enumerate(sources):
        r, c = divmod(idx, ncols)
        sub_ax = axes[r][c]
        sub = df.loc[df["val_source"] == src, "residual"].dropna()
        sub_ax.hist(sub, bins=30, range=shared_range, density=True, alpha=0.7, color=color_map[src])
        sub_ax.axvline(0, color="black", linewidth=1, linestyle="--")
        sub_ax.set_xlabel(f"{sar_var} − {val_var}")
        sub_ax.set_ylabel("Density")
        sub_ax.set_title(f"{src} (N={len(sub)})", fontsize=9)
        sub_ax.grid(True, linewidth=0.4)

    for idx in range(len(sources), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_visualization.py -k TestPlotResiduals -v`
Expected: PASS (all 5 tests in the class).

Run: `python -m pytest tests/ -q`
Expected: PASS (no regressions — `plot_residuals` is called by `validation_report`, covered by `TestValidationReport*` classes, which don't assert axes count).

- [ ] **Step 5: Commit**

```bash
git add sar_validation/core/visualization.py tests/test_visualization.py
git commit -m "fix: residual histogram uses small multiples so a tightly-clustered source can't spike the shared axes"
```

---

### Task 3: Cyclic colormap for wind direction in `plot_geographic`

**Files:**
- Modify: `sar_validation/core/visualization.py:454-772` (`plot_geographic`)
- Test: `tests/test_visualization.py`

**Interfaces:**
- Consumes: `CIRCULAR_VAL_VARS` from `._variable_map` (already imported elsewhere in this module).
- Produces: `plot_geographic(...)` — same signature and return type. When `val_var in CIRCULAR_VAL_VARS` (i.e. `"WDIR"`), both the SAR-field render (`pcolormesh`/`scatter`, via the `cmap` local variable) and the validation-point scatter render (via the `val_cmap` local variable) now use `cmap="twilight"` with fixed `vmin=0, vmax=360`, instead of `"viridis"` with 2nd/98th-percentile-derived limits. This also fixes a latent bug where validation-point scatter calls (lines 751, 759) referenced the raw `val_cmap` parameter (`None` by default) instead of the resolved `effective_val_cmap`, so a caller-supplied `val_cmap` was silently ignored by the actual rendering (only affected the colorbar) — confirmed via a standalone repro before writing this task. Fixing this is required for the circular-colormap branch to actually reach the validation points, not just the colorbar.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_visualization.py`, as a new class placed after `class TestPlotGeographicTicks` (which currently ends at line 491, right before the `geo_datatree_and_collocation_with_unmatched` fixture):

```python
class TestPlotGeographicCircularColormap:
    def test_wdir_uses_twilight_cmap_and_0_360_range(self, geo_datatree_and_collocation, monkeypatch):
        """owiWindDirection/WDIR is a circular (0-360°) variable: both the
        SAR field (pcolormesh) and the validation points (scatter) must
        render with a shared cyclic colormap and fixed 0-360 color limits,
        not viridis + percentile-derived limits (meaningless for data that
        wraps at the 0/360 seam)."""
        import matplotlib.pyplot as plt
        import matplotlib.axes
        from sar_validation.core.visualization import plot_geographic
        from sar_validation.core.datatree_converter import DataTreeConverter

        datatree, collocation_ds = geo_datatree_and_collocation
        scene_ds = datatree["sar/sceneA"].to_dataset()
        scene_ds = scene_ds.assign(owiWindDirection=scene_ds["owiWindSpeed"])
        datatree = DataTreeConverter.to_datatree({
            "sar/sceneA": scene_ds,
            "validation/mooring": datatree["validation/mooring"].to_dataset(),
            "validation/altimeter": datatree["validation/altimeter"].to_dataset(),
        })
        coll = collocation_ds.rename({"val_WSPD": "val_WDIR"}).assign(
            sar_owiWindDirection=collocation_ds["sar_owiWindSpeed"]
        )

        mesh_calls = []
        scatter_calls = []
        original_scatter = matplotlib.axes.Axes.scatter
        original_pcolormesh = matplotlib.axes.Axes.pcolormesh

        def recording_scatter(self, *args, **kwargs):
            if "c" in kwargs:
                scatter_calls.append((kwargs.get("cmap"), kwargs.get("norm")))
            return original_scatter(self, *args, **kwargs)

        def recording_pcolormesh(self, *args, **kwargs):
            mesh_calls.append((kwargs.get("cmap"), kwargs.get("norm")))
            return original_pcolormesh(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        monkeypatch.setattr(matplotlib.axes.Axes, "pcolormesh", recording_pcolormesh)
        fig = plot_geographic(datatree, coll, "owiWindDirection", "WDIR", split_by=None)
        plt.close("all")

        assert fig is not None
        assert mesh_calls, "expected the SAR field to be drawn with pcolormesh"
        assert scatter_calls, "expected validation points to be drawn with scatter"

        def cmap_name(c):
            return getattr(c, "name", c)

        for cmap, norm in mesh_calls + scatter_calls:
            assert cmap_name(cmap) == "twilight"
            assert norm.vmin == 0.0
            assert norm.vmax == 360.0

    def test_non_circular_var_keeps_viridis_and_percentile_limits(self, geo_datatree_and_collocation, monkeypatch):
        import matplotlib.pyplot as plt
        import matplotlib.axes
        from sar_validation.core.visualization import plot_geographic

        datatree, collocation_ds = geo_datatree_and_collocation

        mesh_calls = []
        original_pcolormesh = matplotlib.axes.Axes.pcolormesh

        def recording_pcolormesh(self, *args, **kwargs):
            mesh_calls.append((kwargs.get("cmap"), kwargs.get("norm")))
            return original_pcolormesh(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "pcolormesh", recording_pcolormesh)
        fig = plot_geographic(datatree, collocation_ds, "owiWindSpeed", "WSPD", split_by=None)
        plt.close("all")

        assert fig is not None
        assert mesh_calls

        def cmap_name(c):
            return getattr(c, "name", c)

        for cmap, norm in mesh_calls:
            assert cmap_name(cmap) == "viridis"
            assert not (norm.vmin == 0.0 and norm.vmax == 360.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_visualization.py -k TestPlotGeographicCircularColormap -v`
Expected: `test_wdir_uses_twilight_cmap_and_0_360_range` FAILS (current cmap is `"viridis"`/percentile-derived, and the validation-point scatter calls currently pass `cmap=None`); `test_non_circular_var_keeps_viridis_and_percentile_limits` PASSES already (describes current, unaffected behavior).

- [ ] **Step 3: Add the circular-variable branch**

In `sar_validation/core/visualization.py`, change (lines 629-643):

```python
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
```

to:

```python
    from ._variable_map import CIRCULAR_VAL_VARS  # noqa: PLC0415
    is_circular = val_var in CIRCULAR_VAL_VARS

    # Circular variables (e.g. WDIR) skip percentile pooling: 0-360 is a
    # fixed, physically meaningful range, and percentile-clamping a value
    # that wraps at the 0/360 seam would be actively wrong.
    if is_circular:
        cmap = "twilight"
        vmin, vmax = 0.0, 360.0
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
        val_norm = norm
        val_sm = mcm.ScalarMappable(cmap=effective_val_cmap, norm=val_norm)
        val_sm.set_array([])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_visualization.py -k TestPlotGeographicCircularColormap -v`
Expected: PASS (both tests).

Run: `python -m pytest tests/ -q`
Expected: PASS (no regressions — `val_cmap` is now always rebound to `effective_val_cmap`, which for non-circular vars with no caller-supplied `val_cmap` resolves to the same value as `cmap`, i.e. unchanged for every existing caller).

- [ ] **Step 5: Commit**

```bash
git add sar_validation/core/visualization.py tests/test_visualization.py
git commit -m "fix: wind direction uses a cyclic colormap and fixed 0-360 range in plot_geographic"
```

---

### Task 4: Recipe-variable-gated matched-point styling in `plot_collocation_diagnostics`

**Files:**
- Modify: `sar_validation/core/visualization.py:1159-1643` (`plot_collocation_diagnostics`)
- Test: `tests/test_visualization.py`

**Interfaces:**
- Consumes: `recipe.config.variable` (str: `"wind"` | `"waves"` | `"currents"`), `recipe.config.geographic_bounds` (already read at line 1444).
- Produces: `plot_collocation_diagnostics(...)` — same signature and return type. Tier 3 (matched layer, zorder=5) scatter calls now use `alpha=0.65` instead of `1.0` when `recipe.config.variable == "wind"` (else unchanged at `1.0`). Tier 3 and Tier 4 (matched in-situ, zorder=6) scatter calls now use `s=45, edgecolors="black", linewidths=0.5` instead of `s=25, edgecolors="none"` when `recipe.config.variable == "waves"` (else unchanged at `s=25, edgecolors="none"`).

- [ ] **Step 1: Add waves/currents recipe fixtures**

In `tests/test_visualization.py`, add two new fixtures immediately after the existing `diagnostics_recipe` fixture (currently lines 631-647):

```python
@pytest.fixture
def diagnostics_recipe_waves():
    from sar_validation.core.recipe import (
        GeographicBounds, Recipe, RecipeConfig, ValidationDataSource,
        CollocationType, PointVsLayerCollocation,
    )
    config = RecipeConfig(
        name="test_recipe",
        variable="waves",
        geographic_bounds=GeographicBounds(min_lon=-11.0, max_lon=-7.0, min_lat=49.0, max_lat=53.0),
        validation_sources=[
            ValidationDataSource(source_type="mooring"),
            ValidationDataSource(source_type="altimeter"),
        ],
        collocation=CollocationType(point_vs_layer=PointVsLayerCollocation(time_tolerance_minutes=30)),
    )
    return Recipe(config=config)


@pytest.fixture
def diagnostics_recipe_currents():
    from sar_validation.core.recipe import (
        GeographicBounds, Recipe, RecipeConfig, ValidationDataSource,
        CollocationType, PointVsLayerCollocation,
    )
    config = RecipeConfig(
        name="test_recipe",
        variable="currents",
        geographic_bounds=GeographicBounds(min_lon=-11.0, max_lon=-7.0, min_lat=49.0, max_lat=53.0),
        validation_sources=[
            ValidationDataSource(source_type="mooring"),
            ValidationDataSource(source_type="altimeter"),
        ],
        collocation=CollocationType(point_vs_layer=PointVsLayerCollocation(time_tolerance_minutes=30)),
    )
    return Recipe(config=config)
```

- [ ] **Step 2: Update the two existing tests that hardcode Tier-3 `alpha=1.0` for the wind fixture**

In `tests/test_visualization.py`, in `TestPlotCollocationDiagnosticsRefinement.test_zorder_ensures_insitu_on_top`, change (lines 786-789):

```python
        # Verify matched layers alpha is 1.0 (emphasized so few matches stay visible)
        assert 1.0 in matched_layer_alphas or len(matched_layer_alphas) == 0, (
            f"Expected matched layer alpha=1.0, got {matched_layer_alphas}"
        )
```

to:

```python
        # Verify matched layers alpha is 0.65 for wind recipes (a dense
        # source like scatterometer would otherwise fully occlude a
        # sparser layer source, e.g. radiometer, drawn underneath it)
        assert 0.65 in matched_layer_alphas or len(matched_layer_alphas) == 0, (
            f"Expected matched layer alpha=0.65 for a wind recipe, got {matched_layer_alphas}"
        )
```

In `TestPlotCollocationDiagnosticsRefinement.test_matched_layer_points_are_emphasized`, change the docstring and assertion (lines 803-835):

```python
    def test_matched_layer_points_are_emphasized(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path, monkeypatch
    ):
        """Matched layer points (zorder=5) must be drawn bold: full opacity,
        no marker edge, same marker size as matched in-situ points (s=25)
        — so a few matched points stay visible against the SAR footprints
        and gray unmatched tracks, without visually outsizing in-situ."""
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
            assert c.get("edgecolors") == "none"
            assert c.get("alpha") == 1.0
            assert c.get("s") == 25
```

to:

```python
    def test_matched_layer_points_are_emphasized(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path, monkeypatch
    ):
        """Matched layer points (zorder=5) must be drawn bold: no marker
        edge, same marker size as matched in-situ points (s=25), and for a
        wind recipe alpha=0.65 (not full opacity — a dense source like
        scatterometer would otherwise fully occlude a sparser one, e.g.
        radiometer, drawn underneath it in the same tier)."""
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
            assert c.get("edgecolors") == "none"
            assert c.get("alpha") == 0.65
            assert c.get("s") == 25
```

- [ ] **Step 3: Add new tests for the recipe-variable-gated styling**

Add to `tests/test_visualization.py`, as a new class placed after `class TestPlotCollocationDiagnosticsRefinement` (currently ends at line 919, right before `class TestPlotCollocationDiagnosticsTicks`):

```python
class TestPlotCollocationDiagnosticsRecipeVariableStyling:
    def test_wind_matched_layer_alpha_is_reduced(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path, monkeypatch
    ):
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
        plot_collocation_diagnostics(datatree, collocation_ds, diagnostics_recipe, tmp_path)
        plt.close("all")

        layer_calls = [c for c in recorded if c.get("zorder") == 5]
        assert layer_calls
        for c in layer_calls:
            assert c.get("alpha") == 0.65

    def test_waves_matched_layer_alpha_stays_opaque(
        self, geo_datatree_and_collocation, diagnostics_recipe_waves, tmp_path, monkeypatch
    ):
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
        plot_collocation_diagnostics(datatree, collocation_ds, diagnostics_recipe_waves, tmp_path)
        plt.close("all")

        layer_calls = [c for c in recorded if c.get("zorder") == 5]
        assert layer_calls
        for c in layer_calls:
            assert c.get("alpha") == 1.0

    def test_currents_matched_layer_alpha_stays_opaque(
        self, geo_datatree_and_collocation, diagnostics_recipe_currents, tmp_path, monkeypatch
    ):
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
        plot_collocation_diagnostics(datatree, collocation_ds, diagnostics_recipe_currents, tmp_path)
        plt.close("all")

        layer_calls = [c for c in recorded if c.get("zorder") == 5]
        assert layer_calls
        for c in layer_calls:
            assert c.get("alpha") == 1.0

    def test_waves_matched_points_are_larger_with_black_edge(
        self, geo_datatree_and_collocation, diagnostics_recipe_waves, tmp_path, monkeypatch
    ):
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
        plot_collocation_diagnostics(datatree, collocation_ds, diagnostics_recipe_waves, tmp_path)
        plt.close("all")

        matched_calls = [c for c in recorded if c.get("zorder") in (5, 6)]
        assert matched_calls
        for c in matched_calls:
            assert c.get("s") == 45
            assert c.get("edgecolors") == "black"

    def test_wind_matched_points_keep_default_size_and_no_edge(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path, monkeypatch
    ):
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
        plot_collocation_diagnostics(datatree, collocation_ds, diagnostics_recipe, tmp_path)
        plt.close("all")

        matched_calls = [c for c in recorded if c.get("zorder") in (5, 6)]
        assert matched_calls
        for c in matched_calls:
            assert c.get("s") == 25
            assert c.get("edgecolors") == "none"
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `python -m pytest tests/test_visualization.py -k "TestPlotCollocationDiagnosticsRefinement or TestPlotCollocationDiagnosticsRecipeVariableStyling" -v`
Expected: `test_zorder_ensures_insitu_on_top` and `test_matched_layer_points_are_emphasized` FAIL (current alpha is 1.0 for wind, not 0.65); `test_wind_matched_layer_alpha_is_reduced` and `test_waves_matched_points_are_larger_with_black_edge` FAIL (current code has no recipe-variable branching at all); `test_waves_matched_layer_alpha_stays_opaque`, `test_currents_matched_layer_alpha_stays_opaque`, `test_wind_matched_points_keep_default_size_and_no_edge` PASS already (describe unchanged current behavior).

- [ ] **Step 5: Add the recipe-variable-gated styling**

In `sar_validation/core/visualization.py`, change (line 1444):

```python
    bounds = recipe.config.geographic_bounds
```

to:

```python
    bounds = recipe.config.geographic_bounds
    variable = recipe.config.variable

    # Matched-point styling depends on the recipe's variable type:
    # - wind: layer-source matches (Tier 3) get a moderate alpha instead of
    #   full opacity, since a dense swath (e.g. scatterometer) would
    #   otherwise fully occlude a sparser layer source (e.g. radiometer)
    #   plotted underneath it in the same tier.
    # - waves: all matched points (Tier 3 + Tier 4) get a larger marker and
    #   a black edge, making individual matches easier to pick out.
    matched_layer_alpha = 0.65 if variable == "wind" else 1.0
    if variable == "waves":
        matched_marker_size = 45
        matched_edgecolors = "black"
        matched_linewidths = 0.5
    else:
        matched_marker_size = 25
        matched_edgecolors = "none"
        matched_linewidths = 0.0
```

Change the Tier 3 scatter call (lines 1558-1564):

```python
        color, marker = source_style_map.get(str(cat["label"]), ("#2ca02c", "o"))
        ax.scatter(
            m_lon, m_lat,
            s=25, c=color, marker=marker, alpha=1.0,
            edgecolors="none",
            transform=transform, zorder=5, label=f"{cat['label']} matched ({len(m_lon)})",
        )
```

to:

```python
        color, marker = source_style_map.get(str(cat["label"]), ("#2ca02c", "o"))
        ax.scatter(
            m_lon, m_lat,
            s=matched_marker_size, c=color, marker=marker, alpha=matched_layer_alpha,
            edgecolors=matched_edgecolors, linewidths=matched_linewidths,
            transform=transform, zorder=5, label=f"{cat['label']} matched ({len(m_lon)})",
        )
```

Change the first Tier 4 scatter call (lines 1587-1597):

```python
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
```

to:

```python
                for source in sources_by_count:
                    mask = m_src == source
                    count = int(np.sum(mask))
                    color, marker = source_style_map.get(str(source), ("#ff7f0e", "o"))
                    ax.scatter(
                        m_lon[mask], m_lat[mask],
                        s=matched_marker_size, c=color, marker=marker, alpha=1.0,
                        edgecolors=matched_edgecolors, linewidths=matched_linewidths,
                        transform=transform, zorder=6,
                        label=f"In-situ matched: {source} ({count})",
                    )
```

Change the fallback Tier 4 scatter call (lines 1599-1606):

```python
            else:
                # Fallback if no source info available
                ax.scatter(
                    m_lon, m_lat,
                    s=25, c="#ff7f0e", marker="o", alpha=1.0,
                    edgecolors="none",
                    transform=transform, zorder=6,
                    label=f"In-situ matched ({len(m_lon)})",
                )
```

to:

```python
            else:
                # Fallback if no source info available
                ax.scatter(
                    m_lon, m_lat,
                    s=matched_marker_size, c="#ff7f0e", marker="o", alpha=1.0,
                    edgecolors=matched_edgecolors, linewidths=matched_linewidths,
                    transform=transform, zorder=6,
                    label=f"In-situ matched ({len(m_lon)})",
                )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_visualization.py -k "TestPlotCollocationDiagnostics" -v`
Expected: PASS (every test in every `TestPlotCollocationDiagnostics*` class).

Run: `python -m pytest tests/ -q`
Expected: PASS (no regressions).

- [ ] **Step 7: Commit**

```bash
git add sar_validation/core/visualization.py tests/test_visualization.py
git commit -m "fix: gate matched-point styling in plot_collocation_diagnostics on recipe variable (wind alpha, waves marker size/edge)"
```

---

### Task 5: Cap longitude/latitude tick count in `_set_lonlat_ticks`

**Files:**
- Modify: `sar_validation/core/visualization.py:176-217` (`_set_lonlat_ticks`)
- Test: `tests/test_visualization.py`

**Interfaces:**
- Consumes: `gl.xlocator` / `gl.ylocator` — `cartopy.mpl.ticker.LongitudeLocator` / `LatitudeLocator` instances (both subclass `matplotlib.ticker.MaxNLocator`, confirmed via `cartopy.mpl.ticker` source), which support `.set_params(nbins=...)` to change their maximum tick count in place.
- Produces: `_set_lonlat_ticks(ax, gl)` — same signature, `None` return. Behavior change: both `gl.xlocator` and `gl.ylocator` are reconfigured to `nbins=4` (from cartopy's default `nbins=8`) before their tick positions are read, so narrow-extent subplots (e.g. sub-1°-wide WV-mode SAR scenes) no longer produce many closely-spaced, high-decimal-precision, overlapping tick labels. This is a mutation of the locator objects the caller already owns (`gl` is passed in), not a new return value.

- [ ] **Step 1: Write a failing regression test**

Add to `tests/test_visualization.py`, inside `class TestPlotGeographicTicks` (after `test_set_lonlat_ticks_aligns_with_gridliner_locator`, currently ending at line 491):

```python
    def test_narrow_extent_caps_tick_count(self):
        """Regression test: narrow WV-mode SAR scenes (sub-1° longitude
        span) must not get more than a handful of ticks — the gridliner's
        default locator (nbins=8) produces many closely-spaced,
        high-decimal-precision ticks that overlap into unreadable labels
        otherwise."""
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        from sar_validation.core.visualization import _set_lonlat_ticks

        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        lon = np.linspace(-49.55, -49.50, 5)   # ~0.05° wide, like a WV imagette
        lat = np.linspace(20.0, 43.0, 5)       # wide latitude range
        lon2d, lat2d = np.meshgrid(lon, lat)
        data = np.linspace(5.0, 12.0, 25).reshape(5, 5)
        ax.pcolormesh(lon2d, lat2d, data, transform=ccrs.PlateCarree())

        gl = ax.gridlines(draw_labels=False, alpha=0.3)
        _set_lonlat_ticks(ax, gl)

        assert len(ax.get_xticks()) <= 4
        plt.close(fig)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_visualization.py -k test_narrow_extent_caps_tick_count -v`
Expected: FAIL (`len(ax.get_xticks())` is 6 with the current uncapped `nbins=8` locator — confirmed via a standalone repro before writing this task).

- [ ] **Step 3: Cap the locator's `nbins` in `_set_lonlat_ticks`**

In `sar_validation/core/visualization.py`, change (lines 210-217):

```python
    ax.tick_params(axis="both", which="both", length=0)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xticks = [x for x in gl.xlocator.tick_values(*xlim) if xlim[0] <= x <= xlim[1]]
    yticks = [y for y in gl.ylocator.tick_values(*ylim) if ylim[0] <= y <= ylim[1]]
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
```

to:

```python
    ax.tick_params(axis="both", which="both", length=0)

    # Cap the locator to a small number of ticks regardless of extent
    # width: the gridliner's default (nbins=8) picks tick counts oblivious
    # to how narrow the subplot's actual extent is. A sub-1°-wide WV-mode
    # SAR scene (narrow longitude, wide latitude) triggers many
    # closely-spaced, high-decimal-precision ticks that pile up into
    # unreadable overlapping labels; a wide overview map also reads more
    # cleanly with fewer ticks.
    gl.xlocator.set_params(nbins=4)
    gl.ylocator.set_params(nbins=4)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xticks = [x for x in gl.xlocator.tick_values(*xlim) if xlim[0] <= x <= xlim[1]]
    yticks = [y for y in gl.ylocator.tick_values(*ylim) if ylim[0] <= y <= ylim[1]]
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
```

Also update the function's docstring (lines 176-202) to mention the cap — insert this sentence right after the existing paragraph that starts with "``gl`` is the Gridliner returned by...":

```
    Both locators are first capped to a small ``nbins`` (see below) so
    narrow-extent subplots never get an unreadable pile of closely-spaced
    tick labels.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_visualization.py -k "TestPlotGeographicTicks or TestPlotCollocationDiagnosticsTicks" -v`
Expected: PASS (all tests — including `test_set_lonlat_ticks_aligns_with_gridliner_locator`, which recomputes its "expected" ticks from `gl.xlocator`/`gl.ylocator` *after* calling `_set_lonlat_ticks`, so it observes the same already-capped locator state and still matches; confirmed via a standalone repro before writing this task).

Run: `python -m pytest tests/ -q`
Expected: PASS (no regressions).

- [ ] **Step 5: Commit**

```bash
git add sar_validation/core/visualization.py tests/test_visualization.py
git commit -m "fix: cap lon/lat tick count in _set_lonlat_ticks to stop narrow SAR-scene subplots from overlapping labels"
```

---

### Task 6: Shrink HF radar markers on currents geographic plots

**Files:**
- Modify: `sar_validation/core/visualization.py:1903-1926` (`validation_report`'s `plot_geographic` call site)
- Test: `tests/test_visualization.py`

**Interfaces:**
- Consumes: `plot_geographic(..., point_size: int = 40, ...)` (visualization.py:454-463) — existing parameter, no signature change.
- Produces: `validation_report` now passes `point_size=15` to `plot_geographic` when `recipe.config.variable == "currents"` (else `point_size=40`, the existing default) — currents' only layer-type source, HF radar, forms a near-continuous coverage grid that at the default marker size tiles edge-to-edge and completely hides the SAR field underneath it (confirmed in `data/2026-06-01-000000-2026-06-01-235959_-130.00_-115.00_33.00_48.00/validation_report.pdf`).

- [ ] **Step 1: Write failing tests**

Add to `tests/test_visualization.py`, as a new class placed after `class TestValidationReportSceneAllowlist` (currently ends at line 1291, right before `class TestImagePageFigure`):

```python
class TestValidationReportCurrentsPointSize:
    def test_currents_recipe_passes_reduced_point_size_to_geographic(self, tmp_path, monkeypatch):
        """HF radar (layer_vs_layer) currents validation points are dense
        enough at the default marker size (s=40) to fully blanket the SAR
        field underneath — validation_report must request a smaller marker
        for currents recipes so the SAR scene stays visible through them."""
        import matplotlib.pyplot as plt
        import sar_validation.core.visualization as viz
        from sar_validation.core.recipe import Recipe, RecipeConfig
        from sar_validation.core.datatree_converter import DataTreeConverter

        y, x = 3, 3
        lon2d, lat2d = np.meshgrid(np.linspace(-10, -8, x), np.linspace(50, 52, y))
        sar_ds = xr.Dataset(
            {"rvlRadVel": (("y", "x"), np.full((y, x), 0.3))},
            coords={"lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d),
                    "time": pd.Timestamp("2026-07-10T19:00:00")},
        )
        datatree = DataTreeConverter.to_datatree({"sar/sceneA": sar_ds})

        coll = xr.Dataset({
            "sar_rvlRadVel":            ("collocation", [0.3, 0.31, 0.29, 0.32]),
            "val_rvlRadVel_projection": ("collocation", [0.28, 0.30, 0.27, 0.31]),
            "val_source":               ("collocation", ["hf_radar"] * 4),
            "sar_scene_name":           ("collocation", ["sceneA"] * 4),
            "val_lon":                  ("collocation", [-9.5, -9.4, -9.3, -9.2]),
            "val_lat":                  ("collocation", [50.5, 50.6, 50.7, 50.8]),
        })

        captured = {}
        original = viz.plot_geographic

        def spy(datatree_, coll_, sar_var, val_var, **kwargs):
            captured["point_size"] = kwargs.get("point_size")
            return original(datatree_, coll_, sar_var, val_var, **kwargs)

        monkeypatch.setattr(viz, "plot_geographic", spy)
        recipe = Recipe(config=RecipeConfig(name="currents_test", variable="currents"))
        viz.validation_report(coll, datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        assert captured.get("point_size") == 15

    def test_wind_recipe_keeps_default_point_size(self, geo_datatree_and_collocation, tmp_path, monkeypatch):
        import matplotlib.pyplot as plt
        import sar_validation.core.visualization as viz
        from sar_validation.core.recipe import Recipe, RecipeConfig

        datatree, collocation_ds = geo_datatree_and_collocation
        captured = {}
        original = viz.plot_geographic

        def spy(datatree_, coll_, sar_var, val_var, **kwargs):
            captured["point_size"] = kwargs.get("point_size")
            return original(datatree_, coll_, sar_var, val_var, **kwargs)

        monkeypatch.setattr(viz, "plot_geographic", spy)
        recipe = Recipe(config=RecipeConfig(name="test_recipe", variable="wind"))
        viz.validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        assert captured.get("point_size") == 40
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_visualization.py -k TestValidationReportCurrentsPointSize -v`
Expected: `test_currents_recipe_passes_reduced_point_size_to_geographic` FAILS (`captured["point_size"]` is `None` — `plot_geographic` is currently called without a `point_size` kwarg at all, so it silently uses its own default); `test_wind_recipe_keeps_default_point_size` FAILS for the same reason (`None != 40`).

- [ ] **Step 3: Pass a recipe-variable-gated `point_size` to `plot_geographic`**

In `sar_validation/core/visualization.py`, change the geographic block (the version already updated by Task 1, lines 1903-1926):

```python
        # Geographic — returns dict[collocation_type, Figure] by default
        try:
            geo_result = plot_geographic(datatree, pair_ds, sar_var, val_var, scenes=matched_scenes)
            if isinstance(geo_result, dict):
```

to:

```python
        # Geographic — returns dict[collocation_type, Figure] by default
        try:
            # HF radar (currents' only layer-type source) forms a
            # near-continuous coverage grid that at the default marker size
            # tiles edge-to-edge and completely hides the SAR field
            # underneath it — use a smaller marker for currents recipes so
            # the SAR scene stays visible through the validation points.
            geo_point_size = 15 if variable == "currents" else 40
            geo_result = plot_geographic(
                datatree, pair_ds, sar_var, val_var, scenes=matched_scenes,
                point_size=geo_point_size,
            )
            if isinstance(geo_result, dict):
```

(the rest of the block — the `if isinstance(geo_result, dict): ... elif geo_result is not None: ... except Exception as exc: ...` — is unchanged from Task 1's version.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_visualization.py -k TestValidationReportCurrentsPointSize -v`
Expected: PASS (both tests).

Run: `python -m pytest tests/ -q`
Expected: PASS (no regressions).

- [ ] **Step 5: Commit**

```bash
git add sar_validation/core/visualization.py tests/test_visualization.py
git commit -m "fix: shrink HF radar markers on currents geographic plots so the SAR field stays visible underneath"
```

---

### Task 7: Final full-suite verification

**Files:**
- None (verification only).

**Interfaces:**
- Consumes: everything produced by Tasks 1-6.
- Produces: confirmation that the full test suite passes and that a real report regenerates cleanly end-to-end.

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest tests/ -q`
Expected: PASS, 0 failures.

- [ ] **Step 2: Regenerate a real report and spot-check the four fixes visually**

Using one of the example data directories referenced in the design spec (adjust the path if it no longer exists locally):

```bash
sar-validate --recipe recipes/wind_europe_example.yaml --plot
```

(or the recipe that produced
`data/2026-07-05-180000-2026-07-05-220000_-15.00_0.00_35.00_60.00/`)

Then inspect the regenerated `plots/` directory and `validation_report.pdf`:
- `plots/` contains only `collocation_diagnostics_*.png` (item 1).
- The residuals page in the PDF shows one subplot per source, no single source dominating (item 2).
- The wind-direction geographic page uses the twilight colormap; the collocation-diagnostics page's scatterometer swath no longer fully hides radiometer stars underneath it (item 3).

For a waves recipe (e.g. the one that produced
`data/2026-07-01-000000-2026-07-03-000000_-70.00_-20.00_0.00_40.00/`):

```bash
sar-validate --recipe <waves-recipe>.yaml --plot
```

- The per-scene geographic subplots no longer show overlapping longitude labels (item 4B).
- Matched points in the collocation-diagnostics page are visibly larger with a black edge (item 4A).

For a currents recipe (e.g. the one that produced
`data/2026-06-01-000000-2026-06-01-235959_-130.00_-115.00_33.00_48.00/`):

```bash
sar-validate --recipe <currents-recipe>.yaml --plot
```

- The geographic pages show HF radar points as small, sparse markers — the SAR `rvlRadVel` field is visible underneath them again, not fully blanketed (item 5).

- [ ] **Step 3: Report status**

No commit for this task (verification only) — if Step 2 surfaces a visual issue, open a follow-up task rather than amending prior commits.
