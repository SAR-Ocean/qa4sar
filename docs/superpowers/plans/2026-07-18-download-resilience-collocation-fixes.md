# Download Resilience, Individual-Collocation Caching, and Diagnostics Transparency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the CLI continue past a partial download failure and warn about it in the PDF, add real per-file download resume across all 7 downloaders (replacing a broken `copernicusmarine` kwarg discovered along the way), fix `--layer-vs-layer-collocation-method individual` silently reading/writing the wrong collocation file, and give individual-mode's diagnostics plot enough transparency to show point density instead of a solid blob.

**Architecture:** Four independent bug fixes touching `cli.py`, `orchestrator.py`, `visualization.py`, and all 7 downloader modules under `sar_validation/downloaders/`. Bugs B (download resilience), C (filename bug), and D (plot transparency) are each a handful of small, focused edits to `cli.py`/`visualization.py` (Tasks 1-4). Bug B's per-file resume is the largest piece: one shared helper (Task 5) plus one task per downloader (Tasks 6-12) each adding a `force_download: bool` constructor parameter and a skip-if-exists check, followed by a final integration task wiring `DataOrchestrator` and the CLI's `--force-download` flag through to all of them (Task 13).

**Tech Stack:** Python >=3.10, copernicusmarine 2.4 (`skip_existing`/`overwrite`, not `force_download`), eumdac, matplotlib, pytest.

## Global Constraints

- `copernicusmarine.subset()`/`.get()` do not accept a `force_download` keyword in the installed version (2.4.1) — the real, mutually-exclusive options are `skip_existing` and `overwrite`. No task should pass `force_download=...` to either function.
- The toolbox's own `force_download: bool` (thread from the CLI's `--force-download` flag) means "don't skip existing files" — default `False` skips a file/product that already exists on disk; `True` re-downloads it regardless.
- Non-behavior-changing default: every downloader's new `force_download` constructor parameter defaults to `False`. Existing callers that don't pass it get the new skip-existing behavior, which is additive (previously-missing per-file resume), not a behavior regression for any currently-passing test.
- `--layer-vs-layer-collocation-method individual` (alone, not `"both"`) must resolve to filename suffix `"_individual"` — the same suffix `"both"` mode already uses for that method — so `collocation_results_individual.nc` is read/written consistently across collocate/stats/plot.
- `plot_collocation_diagnostics()`'s `matched_layer_alpha` for individual-mode layer-source matches is a fixed `0.15`, independent of `recipe.config.variable`. Cell-averaging mode's existing per-variable alpha (`0.65` wind / `1.0` else) is unchanged.

---

### Task 1: Continue past partial download failure (Bug B1)

**Files:**
- Modify: `sar_validation/cli.py:511-516`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `_execute_recipe(...)` no longer calls `sys.exit(1)` when `orchestrator.download_all()` returns `False` — later tasks (and manual runs) can rely on convert/collocate/stats/plot still executing after a partial download failure.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`, as a new class at the end of the file:

```python
class TestExecuteRecipeContinuesPastDownloadFailure:
    def test_does_not_exit_when_download_all_returns_false(self, tmp_path, capsys):
        from unittest.mock import patch
        from sar_validation.core.recipe import Recipe, RecipeConfig

        recipe_path = tmp_path / "recipe.yaml"
        Recipe(RecipeConfig(
            name="test-partial-failure",
            variable="wind",
            output_dir=str(tmp_path / "run"),
        )).to_yaml(recipe_path)

        with patch(
            "sar_validation.core.orchestrator.DataOrchestrator.download_all",
            return_value=False,
        ):
            # Must not raise SystemExit.
            cli._execute_recipe(str(recipe_path), force_download=True)

        out = capsys.readouterr().out
        assert "continuing with available data" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py::TestExecuteRecipeContinuesPastDownloadFailure -v`
Expected: FAIL with `SystemExit: 1` (today's `sys.exit(1)` still fires).

- [ ] **Step 3: Implement the fix**

In `sar_validation/cli.py`, replace lines 511-516:

```python
        elif not success:
            print("\nOne or more downloads failed — continuing with available data.")
            print("Check download_metadata.json for details.")
            print(f"Data directory: {orchestrator.base_dir}")
        else:
            print("\nAll downloads completed.")
            print(f"Data directory: {orchestrator.base_dir}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py::TestExecuteRecipeContinuesPastDownloadFailure -v`
Expected: PASS

- [ ] **Step 5: Run the full test_cli.py file to check for regressions**

Run: `pytest tests/test_cli.py -v`
Expected: PASS (all tests, including the pre-existing `TestLoadPrecomputedStats` class)

- [ ] **Step 6: Commit**

```bash
git add sar_validation/cli.py tests/test_cli.py
git commit -m "fix: continue pipeline past a partial download failure instead of exiting"
```

---

### Task 2: Surface download failures on the PDF cover page (Bug B2)

**Files:**
- Modify: `sar_validation/core/visualization.py:1970-2006` (`validation_report` signature/docstring), `:2201-2214` (cover page)
- Modify: `sar_validation/cli.py:683-711` (`_generate_plots`)
- Test: `tests/test_visualization.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `download_metadata.json`'s `"errors"` list (written by `DataOrchestrator._save_metadata()`, unchanged).
- Produces: `validation_report(..., download_warnings: Optional[list[str]] = None)` — a new keyword parameter later tasks don't need. `cli._load_download_warnings(base_dir: Path) -> Optional[list[str]]` — new function, returns `None` when there's no metadata file or its `errors` list is empty, else the `errors` list itself.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli.py`, as a new class at the end of the file:

```python
class TestLoadDownloadWarnings:
    def test_no_metadata_file_returns_none(self, tmp_path):
        assert cli._load_download_warnings(tmp_path) is None

    def test_empty_errors_returns_none(self, tmp_path):
        import json

        (tmp_path / "download_metadata.json").write_text(json.dumps({"errors": []}))
        assert cli._load_download_warnings(tmp_path) is None

    def test_returns_errors_list(self, tmp_path):
        import json

        (tmp_path / "download_metadata.json").write_text(
            json.dumps({"errors": ["altimeter download failed: timeout"]})
        )
        assert cli._load_download_warnings(tmp_path) == ["altimeter download failed: timeout"]
```

Add to `tests/test_visualization.py`, as a new class at the end of the file:

```python
class TestValidationReportDownloadWarnings:
    def test_download_warning_appears_on_cover_page(
        self, geo_datatree_and_collocation, tmp_path, monkeypatch
    ):
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
        validation_report(
            collocation_ds, datatree, recipe, out_dir=tmp_path,
            download_warnings=["altimeter download failed: timeout"],
        )
        plt.close("all")

        cover = recorded_figs[0]
        cover_text = " ".join(t.get_text() for t in cover.texts)
        assert "altimeter download failed: timeout" in cover_text

    def test_no_warning_text_when_download_warnings_omitted(
        self, geo_datatree_and_collocation, tmp_path, monkeypatch
    ):
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

        cover = recorded_figs[0]
        # Exactly the same two text() calls as before this change: title + variable/date.
        assert len(cover.texts) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py::TestLoadDownloadWarnings tests/test_visualization.py::TestValidationReportDownloadWarnings -v`
Expected: `TestLoadDownloadWarnings` FAILs with `AttributeError: module 'sar_validation.cli' has no attribute '_load_download_warnings'`. `TestValidationReportDownloadWarnings::test_download_warning_appears_on_cover_page` FAILs with `TypeError: validation_report() got an unexpected keyword argument 'download_warnings'`. `test_no_warning_text_when_download_warnings_omitted` already PASSes (today's cover page already has exactly 2 `text()` calls) — confirm it passes both before and after this task.

- [ ] **Step 3: Implement `_load_download_warnings` and thread it through `_generate_plots`**

In `sar_validation/cli.py`, replace `_generate_plots` (lines 683-711) and add a new function immediately after it:

```python
def _generate_plots(recipe, base_dir: Path, filename_suffix: str = "") -> None:
    """Run step 5b: generate validation plots and save PDF to <base_dir>/, PNG to <base_dir>/plots/."""
    import xarray as xr
    from .core.visualization import validation_report

    coll_path = base_dir / f"collocation_results{filename_suffix}.nc"
    datatree_path = base_dir / "datatree.nc"

    if not coll_path.exists():
        print(f"  collocation_results{filename_suffix}.nc not found — plotting skipped.")
        return
    if not datatree_path.exists():
        print("  datatree.nc not found — plotting skipped.")
        return

    print("\nStep 5b: Generating validation plots…")
    collocation_ds = xr.open_dataset(str(coll_path))
    datatree = xr.open_datatree(str(datatree_path), engine="netcdf4")

    stats_ds_map = _load_precomputed_stats(recipe, collocation_ds, base_dir, filename_suffix)
    download_warnings = _load_download_warnings(base_dir)

    validation_report(collocation_ds, datatree, recipe,
                      stats_ds_map=stats_ds_map or None,
                      out_dir=base_dir,
                      filename_suffix=filename_suffix,
                      download_warnings=download_warnings)
    pdf_path = base_dir / f"validation_report{filename_suffix}.pdf"
    if pdf_path.exists():
        print(f"  PDF report saved to {pdf_path}")
    print(f"  Collocation diagnostics PNG saved to {base_dir / 'plots'}")


def _load_download_warnings(base_dir: Path) -> Optional[list[str]]:
    """Read download_metadata.json's ``errors`` list, if present, for
    surfacing on the PDF cover page. Returns None if there's no metadata
    file, it can't be parsed, or it has no errors."""
    import json as _json

    meta_path = base_dir / "download_metadata.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            meta = _json.load(f)
    except Exception:
        return None
    errors = meta.get("errors") or []
    return errors or None
```

- [ ] **Step 4: Add the `download_warnings` parameter and cover-page text to `validation_report`**

In `sar_validation/core/visualization.py`, replace the `validation_report` signature and docstring (lines 1970-2006):

```python
def validation_report(
    collocation_ds,
    datatree,
    recipe,
    stats_ds_map: Optional[Dict[str, "xr.Dataset"]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    filename_suffix: str = "",
    download_warnings: Optional[list[str]] = None,
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

    Returns
    -------
    dict[str, list[matplotlib.figure.Figure]]
        ``"<sar_var>_vs_<val_var>"`` → list of Figure objects for that pair.
    """
    from ._variable_map import infer_variable_pairs, filter_variable_pairs, CIRCULAR_VAL_VARS  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415
```

Then replace the cover-page block (lines 2201-2214):

```python
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
            if download_warnings:
                cover.text(
                    0.5, 0.34,
                    "⚠ " + "; ".join(download_warnings),
                    ha="center", va="center", fontsize=9, color="firebrick", wrap=True,
                )
            pdf.savefig(cover, bbox_inches="tight")
            plt.close(cover)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_cli.py::TestLoadDownloadWarnings tests/test_visualization.py::TestValidationReportDownloadWarnings -v`
Expected: PASS (5 tests total across both classes)

- [ ] **Step 6: Run both full test files to check for regressions**

Run: `pytest tests/test_cli.py tests/test_visualization.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add sar_validation/cli.py sar_validation/core/visualization.py tests/test_cli.py tests/test_visualization.py
git commit -m "feat: surface download-step failures on the PDF report cover page"
```

---

### Task 3: Fix `collocation_results_individual.nc` filename bug (Bug C)

**Files:**
- Modify: `sar_validation/cli.py:525-530` (add module-level constant right before `_execute_recipe`, update `method_runs` logic)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `cli._METHOD_SUFFIX: dict[str, str]` — module-level constant, `{"cell-averaging": "", "individual": "_individual"}`. Not consumed by any later task, but available if a future change needs the same mapping.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`, as a new class at the end of the file:

```python
class TestMethodRunsSuffixMapping:
    def _write_recipe_with_skippable_download(self, tmp_path):
        import json
        from sar_validation.core.recipe import Recipe, RecipeConfig

        recipe_path = tmp_path / "recipe.yaml"
        Recipe(RecipeConfig(
            name="test", variable="wind", output_dir=str(tmp_path / "run"),
        )).to_yaml(recipe_path)
        (tmp_path / "run").mkdir()
        (tmp_path / "run" / "download_metadata.json").write_text(json.dumps({"errors": []}))
        return recipe_path

    def test_individual_alone_maps_to_individual_suffix(self, tmp_path):
        from unittest.mock import patch

        recipe_path = self._write_recipe_with_skippable_download(tmp_path)

        with patch("sar_validation.cli._collocate_data") as mock_collocate:
            cli._execute_recipe(
                str(recipe_path), collocate=True,
                layer_vs_layer_collocation_method="individual",
            )

        _, kwargs = mock_collocate.call_args
        assert kwargs["filename_suffix"] == "_individual"
        assert kwargs["layer_vs_layer_collocation_method"] == "individual"

    def test_cell_averaging_alone_still_maps_to_empty_suffix(self, tmp_path):
        from unittest.mock import patch

        recipe_path = self._write_recipe_with_skippable_download(tmp_path)

        with patch("sar_validation.cli._collocate_data") as mock_collocate:
            cli._execute_recipe(
                str(recipe_path), collocate=True,
                layer_vs_layer_collocation_method="cell-averaging",
            )

        _, kwargs = mock_collocate.call_args
        assert kwargs["filename_suffix"] == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py::TestMethodRunsSuffixMapping -v`
Expected: `test_individual_alone_maps_to_individual_suffix` FAILs — `kwargs["filename_suffix"] == ""` today (the bug), not `"_individual"`. `test_cell_averaging_alone_still_maps_to_empty_suffix` already PASSes — confirm it passes both before and after this task.

- [ ] **Step 3: Implement the fix**

In `sar_validation/cli.py`, add a module-level constant immediately before `def _execute_recipe(` (currently line 466):

```python
# Filename suffix per layer-vs-layer collocation method. "both" mode writes
# both suffixes directly (see method_runs below); a single method must map
# to the SAME suffix "both" mode would use for it, so that e.g.
# --layer-vs-layer-collocation-method individual alone reads/writes
# collocation_results_individual.nc consistently across collocate/stats/plot
# instead of the unsuffixed collocation_results.nc cell-averaging uses.
_METHOD_SUFFIX = {"cell-averaging": "", "individual": "_individual"}


def _execute_recipe(
```

Then replace lines 525-530:

```python
    if layer_vs_layer_collocation_method == "both":
        # Run the full pipeline once per method, writing distinctly-suffixed
        # outputs so neither run overwrites the other.
        method_runs = [("cell-averaging", ""), ("individual", "_individual")]
    else:
        method_runs = [
            (layer_vs_layer_collocation_method, _METHOD_SUFFIX[layer_vs_layer_collocation_method])
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py::TestMethodRunsSuffixMapping -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full test_cli.py file to check for regressions**

Run: `pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add sar_validation/cli.py tests/test_cli.py
git commit -m "fix: individual-only collocation method now maps to collocation_results_individual.nc"
```

---

### Task 4: Density-aware transparency for individual-mode matched points (Bug D)

**Files:**
- Modify: `sar_validation/core/visualization.py:1223-1229` (`plot_collocation_diagnostics` signature), `:1511-1518` (`matched_layer_alpha`), `:1970-1977` + the new `download_warnings` line (`validation_report` signature, as left by Task 2), `:2154-2156` (`validation_report`'s internal `plot_collocation_diagnostics` call)
- Modify: `sar_validation/cli.py:584-590` + `:628` (`_collocate_data`'s call), `:683` + `:704-707` (`_generate_plots`'s signature and `validation_report` call, as left by Task 2), `:549-550` (`_execute_recipe`'s call to `_generate_plots`)
- Test: `tests/test_visualization.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `plot_collocation_diagnostics(..., layer_vs_layer_collocation_method: str = "cell-averaging")` and `validation_report(..., layer_vs_layer_collocation_method: str = "cell-averaging")` — new keyword parameters. `_generate_plots(recipe, base_dir, filename_suffix="", layer_vs_layer_collocation_method="cell-averaging")` — new keyword parameter.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_visualization.py`, as a new class at the end of the file:

```python
class TestPlotCollocationDiagnosticsIndividualMethodAlpha:
    def test_individual_method_uses_low_fixed_alpha(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path, monkeypatch
    ):
        import sar_validation.core.visualization as viz

        datatree, collocation_ds = geo_datatree_and_collocation
        captured_alphas = []
        original_scatter = viz.plt.Axes.scatter

        def spy_scatter(self, *args, **kwargs):
            if "alpha" in kwargs:
                captured_alphas.append(kwargs["alpha"])
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(viz.plt.Axes, "scatter", spy_scatter)
        viz.plot_collocation_diagnostics(
            datatree, collocation_ds, diagnostics_recipe, tmp_path,
            layer_vs_layer_collocation_method="individual",
        )

        assert 0.15 in captured_alphas

    def test_cell_averaging_default_keeps_todays_alpha(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path, monkeypatch
    ):
        """Non-regression: omitting layer_vs_layer_collocation_method (or
        passing 'cell-averaging' explicitly) must reproduce today's exact
        variable-dependent alpha (0.65 for wind, matching diagnostics_recipe)."""
        import sar_validation.core.visualization as viz

        datatree, collocation_ds = geo_datatree_and_collocation
        captured_alphas = []
        original_scatter = viz.plt.Axes.scatter

        def spy_scatter(self, *args, **kwargs):
            if "alpha" in kwargs:
                captured_alphas.append(kwargs["alpha"])
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(viz.plt.Axes, "scatter", spy_scatter)
        viz.plot_collocation_diagnostics(datatree, collocation_ds, diagnostics_recipe, tmp_path)

        assert 0.15 not in captured_alphas
        assert 0.65 in captured_alphas
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_visualization.py::TestPlotCollocationDiagnosticsIndividualMethodAlpha -v`
Expected: `test_individual_method_uses_low_fixed_alpha` FAILs with `TypeError: plot_collocation_diagnostics() got an unexpected keyword argument 'layer_vs_layer_collocation_method'`. `test_cell_averaging_default_keeps_todays_alpha` already PASSes — confirm it passes both before and after this task.

- [ ] **Step 3: Add the parameter and method-aware alpha to `plot_collocation_diagnostics`**

In `sar_validation/core/visualization.py`, replace the `plot_collocation_diagnostics` signature (lines 1223-1229):

```python
def plot_collocation_diagnostics(
    datatree,
    collocation_ds,
    recipe,
    output_dir: Union[str, Path],
    filename_suffix: str = "",
    layer_vs_layer_collocation_method: str = "cell-averaging",
) -> Union[Path, None]:
```

Add a parameter doc line to the docstring's `Parameters` section (immediately after the existing `filename_suffix` entry, before the blank line and `Returns` section):

```python
    layer_vs_layer_collocation_method : str
        Which layer-vs-layer collocation method produced *collocation_ds*
        ("cell-averaging" or "individual"). Individual-mode matches are far
        denser than cell-averaging (one point per matched SAR pixel vs. one
        per validation-instrument location), so this controls matched-layer
        point transparency — see the alpha computation below.
```

Replace the `matched_layer_alpha` block (lines 1511-1518):

```python
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
        matched_layer_alpha = 0.65 if variable == "wind" else 1.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_visualization.py::TestPlotCollocationDiagnosticsIndividualMethodAlpha -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Thread the parameter through `validation_report` and both `cli.py` call sites**

In `sar_validation/core/visualization.py`, replace the `validation_report` signature (as left by Task 2 — lines 1970-1978):

```python
def validation_report(
    collocation_ds,
    datatree,
    recipe,
    stats_ds_map: Optional[Dict[str, "xr.Dataset"]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    filename_suffix: str = "",
    download_warnings: Optional[list[str]] = None,
    layer_vs_layer_collocation_method: str = "cell-averaging",
) -> Dict[str, list]:
```

Add a matching docstring `Parameters` entry (after the `download_warnings` entry Task 2 added):

```python
    layer_vs_layer_collocation_method : str
        Which layer-vs-layer collocation method produced *collocation_ds*
        ("cell-averaging" or "individual"). Passed through to
        :func:`plot_collocation_diagnostics` for method-aware matched-point
        transparency.
```

Replace the internal `plot_collocation_diagnostics` call (lines 2154-2156):

```python
            diag_path = plot_collocation_diagnostics(
                datatree, collocation_ds, recipe, base_dir,
                filename_suffix=filename_suffix,
                layer_vs_layer_collocation_method=layer_vs_layer_collocation_method,
            )
```

In `sar_validation/cli.py`, replace `_collocate_data`'s `plot_collocation_diagnostics` call (line 628):

```python
        diag_path = plot_collocation_diagnostics(
            tree, collocation_ds, recipe, base_dir,
            filename_suffix=filename_suffix,
            layer_vs_layer_collocation_method=layer_vs_layer_collocation_method,
        )
```

Replace `_generate_plots`'s signature and its `validation_report` call (as left by Task 2 — lines 683 and 704-707):

```python
def _generate_plots(
    recipe, base_dir: Path, filename_suffix: str = "",
    layer_vs_layer_collocation_method: str = "cell-averaging",
) -> None:
```

```python
    validation_report(collocation_ds, datatree, recipe,
                      stats_ds_map=stats_ds_map or None,
                      out_dir=base_dir,
                      filename_suffix=filename_suffix,
                      download_warnings=download_warnings,
                      layer_vs_layer_collocation_method=layer_vs_layer_collocation_method)
```

Replace `_execute_recipe`'s call to `_generate_plots` (lines 549-550):

```python
        if plot:
            _generate_plots(
                recipe, orchestrator.base_dir, filename_suffix=suffix,
                layer_vs_layer_collocation_method=method,
            )
```

- [ ] **Step 6: Run the full visualization and cli test files to check for regressions**

Run: `pytest tests/test_visualization.py tests/test_cli.py -v`
Expected: PASS — in particular, every other `TestPlotCollocationDiagnostics*` and `TestValidationReport*` class, since `layer_vs_layer_collocation_method` defaults to `"cell-averaging"` everywhere and reproduces today's exact alpha values.

- [ ] **Step 7: Commit**

```bash
git add sar_validation/core/visualization.py sar_validation/cli.py tests/test_visualization.py
git commit -m "fix: lower matched-layer-point alpha for individual-method collocation diagnostics"
```

---

### Task 5: Shared `copernicus_marine_download_kwargs` helper

**Files:**
- Modify: `sar_validation/downloaders/base.py`
- Test: `tests/test_downloaders.py`

**Interfaces:**
- Produces: `copernicus_marine_download_kwargs(force_download: bool) -> dict`, importable as `from .base import copernicus_marine_download_kwargs`. Returns `{"skip_existing": not force_download, "overwrite": force_download}`. Tasks 6 and 7 import and use this.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_downloaders.py`, directly after the `class TestSplitAntimeridianBbox:` block (before the SAR antimeridian tests):

```python
# ---------------------------------------------------------------------------
# Tests for copernicus_marine_download_kwargs()
# ---------------------------------------------------------------------------

class TestCopernicusMarineDownloadKwargs:
    def test_default_skips_existing_files(self):
        assert copernicus_marine_download_kwargs(force_download=False) == {
            "skip_existing": True, "overwrite": False,
        }

    def test_force_download_overwrites(self):
        assert copernicus_marine_download_kwargs(force_download=True) == {
            "skip_existing": False, "overwrite": True,
        }
```

Update the existing import line at the top of `tests/test_downloaders.py`:

```python
from sar_validation.downloaders.base import (
    normalize_datetime, is_date_recent, split_antimeridian_bbox,
    copernicus_marine_download_kwargs,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_downloaders.py::TestCopernicusMarineDownloadKwargs -v`
Expected: FAIL with `ImportError: cannot import name 'copernicus_marine_download_kwargs'`

- [ ] **Step 3: Implement the helper**

In `sar_validation/downloaders/base.py`, add `"copernicus_marine_download_kwargs"` to the `__all__` list, then add the function after `split_antimeridian_bbox` (end of file):

```python
def copernicus_marine_download_kwargs(force_download: bool) -> dict:
    """
    Return the ``skip_existing``/``overwrite`` kwargs for a
    ``copernicusmarine.subset()``/``.get()`` call, matching this toolbox's
    ``--force-download`` semantics.

    ``copernicusmarine`` has no ``force_download`` parameter (the two real,
    mutually-exclusive options are ``overwrite`` and ``skip_existing``) —
    this is the single place that translates the toolbox's boolean flag into
    the real API, so no downloader has to reason about the mapping itself.
    """
    return {"skip_existing": not force_download, "overwrite": force_download}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_downloaders.py::TestCopernicusMarineDownloadKwargs -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add sar_validation/downloaders/base.py tests/test_downloaders.py
git commit -m "feat: add copernicus_marine_download_kwargs helper for real skip_existing/overwrite semantics"
```

---

### Task 6: Altimeter downloader — real skip_existing/overwrite

**Files:**
- Modify: `sar_validation/downloaders/altimeter_downloader.py:34` (import), `:97-103` (constructor), `:210-225` (`subset()` call)
- Test: `tests/test_downloaders.py`

**Interfaces:**
- Consumes: `copernicus_marine_download_kwargs` (Task 5).
- Produces: `AltimeterDownloader.__init__(..., force_download: bool = False)` — new constructor parameter, `self.force_download`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_downloaders.py`, directly after `class TestAltimeterDownloaderAntimeridian:` ends (before `TestInSituDownloaderAntimeridian`):

```python
class TestAltimeterDownloaderForceDownload:
    def _patch_subset(self):
        from unittest.mock import MagicMock

        fake_module = MagicMock()

        def fake_subset(**kwargs):
            Path(kwargs["output_directory"], kwargs["output_filename"]).write_bytes(b"")

        fake_module.subset.side_effect = fake_subset
        return fake_module

    def test_default_skips_existing(self, tmp_path):
        from unittest.mock import patch
        from sar_validation.downloaders.altimeter_downloader import AltimeterDownloader

        dl = AltimeterDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = self._patch_subset()

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-06-01", end="2026-06-02",
                frequencies=["1hz"], satellites=["al"],
            )

        kwargs = fake_module.subset.call_args.kwargs
        assert kwargs["skip_existing"] is True
        assert kwargs["overwrite"] is False

    def test_force_download_overwrites(self, tmp_path):
        from unittest.mock import patch
        from sar_validation.downloaders.altimeter_downloader import AltimeterDownloader

        dl = AltimeterDownloader(output_dir=tmp_path, dry_run=False, force_download=True)
        fake_module = self._patch_subset()

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-06-01", end="2026-06-02",
                frequencies=["1hz"], satellites=["al"],
            )

        kwargs = fake_module.subset.call_args.kwargs
        assert kwargs["skip_existing"] is False
        assert kwargs["overwrite"] is True

    def test_force_download_kwarg_never_passed_to_subset(self, tmp_path):
        """Regression: copernicusmarine.subset() has no force_download
        parameter in the installed version (verified via
        inspect.signature) — passing it raises TypeError in real
        (non-mocked) usage."""
        from unittest.mock import patch
        from sar_validation.downloaders.altimeter_downloader import AltimeterDownloader

        dl = AltimeterDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = self._patch_subset()

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-06-01", end="2026-06-02",
                frequencies=["1hz"], satellites=["al"],
            )

        assert "force_download" not in fake_module.subset.call_args.kwargs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_downloaders.py::TestAltimeterDownloaderForceDownload -v`
Expected: FAIL — `AltimeterDownloader(..., force_download=True)` raises `TypeError: unexpected keyword argument 'force_download'`; the other two tests see `kwargs["skip_existing"]`/`kwargs["force_download"]` absent/wrong since today's code passes `force_download=False` unconditionally.

- [ ] **Step 3: Implement**

In `sar_validation/downloaders/altimeter_downloader.py`, update the import (line 34):

```python
from .base import (
    normalize_datetime, build_output_dir, split_antimeridian_bbox,
    copernicus_marine_download_kwargs,
)
```

Replace the constructor (lines 97-103):

```python
    def __init__(
        self,
        output_dir: Path,
        dry_run: bool = False,
        force_download: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.force_download = force_download
```

Replace the `subset()` call (lines 210-225):

```python
                    try:
                        copernicusmarine.subset(
                            dataset_id=dataset_id,
                            variables=variables,
                            minimum_longitude=win_min_lon,
                            maximum_longitude=win_max_lon,
                            minimum_latitude=min_lat,
                            maximum_latitude=max_lat,
                            start_datetime=eff_start_dt,
                            end_datetime=end_dt,
                            minimum_depth=0,
                            maximum_depth=0,
                            output_directory=self.output_dir,
                            output_filename=filename,
                            **copernicus_marine_download_kwargs(self.force_download),
                        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_downloaders.py::TestAltimeterDownloaderForceDownload -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full downloader test file to check for regressions**

Run: `pytest tests/test_downloaders.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add sar_validation/downloaders/altimeter_downloader.py tests/test_downloaders.py
git commit -m "fix(altimeter): use real copernicusmarine skip_existing/overwrite instead of broken force_download kwarg"
```

---

### Task 7: HF radar (Copernicus grid) downloader — real skip_existing/overwrite

**Files:**
- Modify: `sar_validation/downloaders/hf_radar_downloader.py:39` (import), `:72-80` (constructor), `:192-214` (`_subset_with_part`)
- Test: `tests/test_downloaders.py`

**Interfaces:**
- Consumes: `copernicus_marine_download_kwargs` (Task 5).
- Produces: `HFRadarDownloader.__init__(..., force_download: bool = False)` — new constructor parameter, `self.force_download`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_downloaders.py`, directly after `class TestHFRadarDownloaderGridAntimeridian:` ends (before the `# Tests for HFRadarHistoricalDownloader` comment block):

```python
class TestHFRadarDownloaderGridForceDownload:
    def _patch_subset(self):
        from unittest.mock import MagicMock

        fake_module = MagicMock()

        def fake_subset(**kwargs):
            Path(kwargs["output_directory"], kwargs["output_filename"]).write_bytes(b"")

        fake_module.subset.side_effect = fake_subset
        return fake_module

    def test_default_skips_existing(self, tmp_path):
        from unittest.mock import patch
        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = self._patch_subset()

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(-90.0, -60.0, 30.0, 40.0, "2026-01-01", "2026-01-02")

        kwargs = fake_module.subset.call_args.kwargs
        assert kwargs["skip_existing"] is True
        assert kwargs["overwrite"] is False

    def test_force_download_overwrites(self, tmp_path):
        from unittest.mock import patch
        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=False, force_download=True)
        fake_module = self._patch_subset()

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(-90.0, -60.0, 30.0, 40.0, "2026-01-01", "2026-01-02")

        kwargs = fake_module.subset.call_args.kwargs
        assert kwargs["skip_existing"] is False
        assert kwargs["overwrite"] is True

    def test_force_download_kwarg_never_passed_to_subset(self, tmp_path):
        from unittest.mock import patch
        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = self._patch_subset()

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(-90.0, -60.0, 30.0, 40.0, "2026-01-01", "2026-01-02")

        assert "force_download" not in fake_module.subset.call_args.kwargs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_downloaders.py::TestHFRadarDownloaderGridForceDownload -v`
Expected: FAIL — `HFRadarDownloader(..., force_download=True)` raises `TypeError`; the other two see today's unconditional `force_download=True` instead of the expected `skip_existing`/`overwrite` kwargs.

- [ ] **Step 3: Implement**

In `sar_validation/downloaders/hf_radar_downloader.py`, update the import (line 39):

```python
from .base import (
    normalize_datetime, is_date_recent, build_output_dir, split_antimeridian_bbox,
    copernicus_marine_download_kwargs,
)
```

Replace the constructor (lines 72-80):

```python
    def __init__(
        self,
        output_dir: Path,
        dry_run: bool = False,
        min_depth: float = -2.0,
        max_depth: float = 2.0,
        force_download: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.force_download = force_download
```

Replace `_subset_with_part` (lines 192-214):

```python
    def _subset_with_part(
        self, copernicusmarine, dataset_part,
        min_lon, max_lon, min_lat, max_lat,
        start_dt, end_dt, dest_path,
    ) -> None:
        # No `variables=` filter: omitting it makes copernicusmarine return
        # every variable in the dataset_part (verified live — 14 vars for
        # *-radar-total--<Region>, including EWCS/NSCS standard deviations
        # and all *_QC/QCflag fields), so the converter (Task 3) always has
        # the full ancillary set to pick from on disk.
        copernicusmarine.subset(
            dataset_id=DATASET_ID,
            dataset_part=dataset_part,
            minimum_longitude=min_lon,
            maximum_longitude=max_lon,
            minimum_latitude=min_lat,
            maximum_latitude=max_lat,
            start_datetime=start_dt,
            end_datetime=end_dt,
            output_directory=str(dest_path.parent),
            output_filename=dest_path.name,
            **copernicus_marine_download_kwargs(self.force_download),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_downloaders.py::TestHFRadarDownloaderGridForceDownload -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full downloader test file to check for regressions**

Run: `pytest tests/test_downloaders.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add sar_validation/downloaders/hf_radar_downloader.py tests/test_downloaders.py
git commit -m "fix(hf_radar): use real copernicusmarine skip_existing/overwrite instead of broken force_download kwarg"
```

---

### Task 8: In-situ downloader — pre-download existence check

**Files:**
- Modify: `sar_validation/downloaders/insitu_downloader.py:134-144` (constructor), `:196-234` (`_download_window`), `:295-320` (`_download_with_part`)
- Test: `tests/test_downloaders.py`

**Interfaces:**
- Consumes: nothing new (doesn't use `copernicus_marine_download_kwargs` — `subset()` here doesn't take an explicit `output_filename`, so `skip_existing`/`overwrite` don't target the right destination; the fix is a `dest_path.exists()` pre-check instead).
- Produces: `InSituDownloader.__init__(..., force_download: bool = False)` — new constructor parameter, `self.force_download`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_downloaders.py`, directly after `class TestInSituDownloaderAntimeridian:` ends (before `TestScatterometerDownloaderAntimeridian`):

```python
class TestInSituDownloaderForceDownload:
    def test_skips_download_when_file_already_exists(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from sar_validation.downloaders.insitu_downloader import (
            InSituDownloader, _build_csv_filename,
        )

        dl = InSituDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = MagicMock()

        start_dt, end_dt = "2026-01-01T00:00:00", "2026-01-02T00:00:00"
        fname = _build_csv_filename(-20.0, 0.0, 35.0, 60.0, start_dt, end_dt, -20.0, 20.0)
        (tmp_path / fname).write_text("platform_type\n")

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            paths = dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-01-01", end="2026-01-02",
            )

        fake_module.subset.assert_not_called()
        assert len(paths) == 1
        assert paths[0].name == fname

    def test_force_download_redownloads_existing_file(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from sar_validation.downloaders.insitu_downloader import (
            InSituDownloader, _build_csv_filename,
        )

        dl = InSituDownloader(output_dir=tmp_path, dry_run=False, force_download=True)
        fake_module = MagicMock()
        fake_module.subset.side_effect = lambda **kwargs: None

        start_dt, end_dt = "2026-01-01T00:00:00", "2026-01-02T00:00:00"
        fname = _build_csv_filename(-20.0, 0.0, 35.0, 60.0, start_dt, end_dt, -20.0, 20.0)
        (tmp_path / fname).write_text("platform_type\n")

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-01-01", end="2026-01-02",
            )

        fake_module.subset.assert_called_once()

    def test_force_download_kwarg_never_passed_to_subset(self, tmp_path):
        """Regression: copernicusmarine.subset() has no force_download
        parameter in the installed version."""
        from unittest.mock import patch, MagicMock
        from sar_validation.downloaders.insitu_downloader import InSituDownloader

        dl = InSituDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = MagicMock()
        fake_module.subset.side_effect = lambda **kwargs: None

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-01-05", end="2026-01-06",
            )

        assert "force_download" not in fake_module.subset.call_args.kwargs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_downloaders.py::TestInSituDownloaderForceDownload -v`
Expected: `test_skips_download_when_file_already_exists` FAILs — `subset()` is called today even though `dest_path` already exists. `test_force_download_redownloads_existing_file` already PASSes coincidentally (today's code always calls `subset()`), but `InSituDownloader(..., force_download=True)` raises `TypeError` first — confirm the actual failure is the `TypeError` from the unexpected constructor kwarg. `test_force_download_kwarg_never_passed_to_subset` FAILs — today's `_download_with_part` passes `force_download=False`.

- [ ] **Step 3: Implement**

In `sar_validation/downloaders/insitu_downloader.py`, replace the constructor (lines 134-144):

```python
    def __init__(
        self,
        output_dir: Path,
        dry_run: bool = False,
        min_depth: float = -20.0,
        max_depth: float = 20.0,
        force_download: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.force_download = force_download
```

Replace `_download_window` (lines 196-234) — add the pre-check right after the `dry_run` branch:

```python
    def _download_window(
        self,
        copernicusmarine,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start_dt: str,
        end_dt: str,
        source_types: Optional[list[str]],
        dataset_part: Optional[str],
    ) -> Optional[Path]:
        """Download and save one CSV for a single (non-crossing) window."""
        expected_filename = _build_csv_filename(
            min_lon, max_lon, min_lat, max_lat,
            start_dt, end_dt, self.min_depth, self.max_depth,
        )
        dest_path = self.output_dir / expected_filename

        if self.dry_run:
            print(
                f"[DRY RUN] Would download in-situ data to:\n  {dest_path}"
            )
            return None

        if not self.force_download and dest_path.exists():
            print(f"  Already downloaded: {dest_path}")
            return dest_path

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect dataset_part if not provided
        resolved_part = dataset_part
        if resolved_part is None:
            resolved_part = "latest" if is_date_recent(end_dt) else "monthly"

        # Run the copernicusmarine subset call (downloads to CWD by default)
        print(f"Downloading in-situ data …")
        print(f"  Region: lon [{min_lon}, {max_lon}] lat [{min_lat}, {max_lat}]")
        print(f"  Time:   {start_dt} → {end_dt}")
        print(f"  Depth:  {self.min_depth} to {self.max_depth} m")
        print(f"  Dataset: {resolved_part}")

        # Try initial dataset_part, with fallback if data not available
        try:
            self._download_with_part(
                copernicusmarine,
                resolved_part,
                min_lon, max_lon, min_lat, max_lat,
                start_dt, end_dt,
            )
        except Exception as e:
            error_msg = str(e)
            # Check if error is about data exceeding coordinates (date outside available range)
            if "exceed the dataset coordinates" in error_msg or "out of bounds" in error_msg.lower():
                # Try the opposite dataset_part
                alt_dataset_part = "monthly" if resolved_part == "latest" else "latest"
                print(f"  Retrying with dataset_part='{alt_dataset_part}' due to: {error_msg[:100]}…")
                try:
                    self._download_with_part(
                        copernicusmarine,
                        alt_dataset_part,
                        min_lon, max_lon, min_lat, max_lat,
                        start_dt, end_dt,
                    )
                    resolved_part = alt_dataset_part
                except Exception as e2:
                    # Both failed, raise the second error
                    raise e2
            else:
                # Not a data availability error, re-raise original
                raise

        # Move the file (copernicusmarine writes it to CWD) to our output_dir
        if Path(expected_filename).exists():
            shutil.move(str(expected_filename), str(dest_path))
            print(f"  Saved to {dest_path}")
        elif dest_path.exists():
            print(f"  Already at {dest_path}")
        else:
            # Try to find a recently-created CSV in CWD
            candidates = sorted(Path(".").glob(f"{DATASET_ID}*.csv"), key=os.path.getmtime, reverse=True)
            if candidates:
                shutil.move(str(candidates[0]), str(dest_path))
                print(f"  Saved to {dest_path}")
            else:
                raise FileNotFoundError(
                    f"copernicusmarine download completed but output CSV not found.\n"
                    f"Expected: {expected_filename}"
                )

        # Apply platform-type filter
        if source_types:
            platform_codes = _resolve_platform_codes(source_types)
            if platform_codes:
                df = pd.read_csv(dest_path)
                if "platform_type" in df.columns:
                    df = df[df["platform_type"].isin(platform_codes)]
                    df.to_csv(dest_path, index=False)
                    print(f"  Filtered to {len(df)} rows ({', '.join(source_types)})")

        return dest_path
```

Replace `_download_with_part` (lines 295-320) — drop the invalid `force_download` kwarg (no explicit `output_filename`/`output_directory` here, so `skip_existing`/`overwrite` have no meaningful target; the pre-check above already handles resume):

```python
    def _download_with_part(
        self,
        copernicusmarine,
        dataset_part: str,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start_dt: str,
        end_dt: str,
    ) -> None:
        """Internal helper to run copernicusmarine.subset() with a specific dataset_part."""
        copernicusmarine.subset(
            dataset_id=DATASET_ID,
            dataset_part=dataset_part,
            variables=ALL_VARIABLES,
            minimum_longitude=min_lon,
            maximum_longitude=max_lon,
            minimum_latitude=min_lat,
            maximum_latitude=max_lat,
            start_datetime=start_dt,
            end_datetime=end_dt,
            minimum_depth=self.min_depth,
            maximum_depth=self.max_depth,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_downloaders.py::TestInSituDownloaderForceDownload -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full downloader test file to check for regressions**

Run: `pytest tests/test_downloaders.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add sar_validation/downloaders/insitu_downloader.py tests/test_downloaders.py
git commit -m "fix(insitu): skip re-download when output CSV already exists, drop broken force_download kwarg"
```

---

### Task 9: HF radar historical downloader — pre-download existence check

**Files:**
- Modify: `sar_validation/downloaders/hf_radar_historical_downloader.py:119-121` (constructor), `:160-183` (`_download_region_window`)
- Test: `tests/test_downloaders.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `HFRadarHistoricalDownloader.__init__(..., force_download: bool = False)` — new constructor parameter, `self.force_download`.

**Note:** the raw-archive fetch already uses `copernicusmarine.get(..., skip_existing=True)` correctly (unchanged by this task, since a region's archive file covers many years and is meant to be cached across runs regardless of `--force-download`). This task only adds a `dest_path.exists()` pre-check to skip the *local* re-subset/re-write of the final output file — a minor efficiency improvement, not a correctness fix (unlike Tasks 6-8).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_downloaders.py`, directly after `class TestHFRadarHistoricalDownloaderAntimeridian:` ends (before `class TestOrchestratorHFRadarHistoricalWiring:`):

```python
class TestHFRadarHistoricalDownloaderForceDownload:
    def test_skips_when_output_already_exists(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader, DATASET_ID,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=False)
        dest_path = tmp_path / f"{DATASET_ID}_US-WestCoast_2019-01-01.nc"
        dest_path.write_bytes(b"")

        fake_module = MagicMock()
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            out = dl.download(-121.0, -120.0, 33.0, 34.0, "2019-01-01", "2019-01-01")

        fake_module.get.assert_not_called()
        assert out == [dest_path]

    def test_force_download_refetches_existing_output(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader, DATASET_ID,
        )
        import xarray as xr
        import numpy as np
        import pandas as pd

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=False, force_download=True)
        dest_path = tmp_path / f"{DATASET_ID}_US-WestCoast_2019-01-01.nc"
        dest_path.write_bytes(b"stale")

        raw_dir = tmp_path / "_raw"
        raw_dir.mkdir()
        raw_path = raw_dir / "GL_TV_HF_HFR-US-WestCoast_Total.nc"
        times = pd.date_range("2019-01-01", periods=5, freq="1h")
        shape = (5, 1, 2, 2)
        ds = xr.Dataset(
            {
                "EWCT": (("TIME", "DEPTH", "LATITUDE", "LONGITUDE"), np.random.rand(*shape)),
                "NSCT": (("TIME", "DEPTH", "LATITUDE", "LONGITUDE"), np.random.rand(*shape)),
            },
            coords={
                "TIME": times, "DEPTH": [0.0],
                "LATITUDE": [33.0, 34.0], "LONGITUDE": [-121.0, -120.0],
            },
        )
        ds.to_netcdf(raw_path)

        fake_module = MagicMock()

        def fake_get(**kwargs):
            return FileGetResult(files=[type("F", (), {"file_path": raw_path})()])

        fake_module.get.side_effect = fake_get
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(-121.0, -120.0, 33.0, 34.0, "2019-01-01", "2019-01-01T04:00:00")

        fake_module.get.assert_called_once()
```

`FileGetResult` is already defined at module scope earlier in `tests/test_downloaders.py` (used by the existing `TestHFRadarHistoricalDownloader` class) — no new import needed.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_downloaders.py::TestHFRadarHistoricalDownloaderForceDownload -v`
Expected: `test_skips_when_output_already_exists` FAILs — today's code calls `copernicusmarine.get` regardless. `test_force_download_refetches_existing_output` FAILs with `TypeError` — `force_download` isn't yet a constructor parameter.

- [ ] **Step 3: Implement**

In `sar_validation/downloaders/hf_radar_historical_downloader.py`, replace the constructor (lines 119-121):

```python
    def __init__(self, output_dir: Path, dry_run: bool = False, force_download: bool = False) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.force_download = force_download
```

In `_download_region_window`, add the pre-check right after the `dry_run` branch (lines 178-183, i.e. immediately before `try: import copernicusmarine`):

```python
        if self.dry_run:
            print(
                f"[DRY RUN] Would fetch Copernicus HF-radar historical archive "
                f"'{remote_filename}' for region '{region}' and subset to:\n  {dest_path}"
            )
            return None

        if not self.force_download and dest_path.exists():
            print(f"  Already downloaded: {dest_path}")
            return dest_path

        try:
            import copernicusmarine
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_downloaders.py::TestHFRadarHistoricalDownloaderForceDownload -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full downloader test file to check for regressions**

Run: `pytest tests/test_downloaders.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add sar_validation/downloaders/hf_radar_historical_downloader.py tests/test_downloaders.py
git commit -m "feat(hf_radar_historical): skip re-subsetting when the final output already exists"
```

---

### Task 10: SAR downloader — per-product existence check

**Files:**
- Modify: `sar_validation/downloaders/sar_downloader.py:70-81` (constructor), `:189-203` (download loop)
- Test: `tests/test_downloaders.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `SARDownloader.__init__(..., force_download: bool = False)` — new constructor parameter, `self.force_download`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_downloaders.py`, directly after `class TestSARDownloaderAntimeridian:` ends (before `class TestAltimeterDownloaderAntimeridian:`):

```python
class TestSARDownloaderForceDownload:
    def _fake_record(self):
        return {
            "Id": "abc", "Name": "S1A_IW_OCN__2SDV_20260702T000000",
            "ContentDate_Start": "2026-07-02T00:00:00Z",
            "ContentDate_End": "2026-07-02T00:00:10Z",
            "ContentLength_GB": 1.0, "Online": True,
        }

    def test_skips_product_whose_directory_already_exists(self, tmp_path, capsys):
        from unittest.mock import MagicMock
        from sar_validation.downloaders.sar_downloader import SARDownloader

        dl = SARDownloader(output_dir=tmp_path, dry_run=False)
        fake_client = MagicMock()
        fake_client.query_products.return_value = [self._fake_record()]
        dl._client = fake_client
        (tmp_path / "S1A_IW_OCN__2SDV_20260702T000000").mkdir()

        dl.download(
            min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
            start="2026-07-02", end="2026-07-03",
        )

        fake_client.download_product.assert_not_called()
        assert "Already downloaded" in capsys.readouterr().out

    def test_force_download_redownloads_existing_product(self, tmp_path):
        from unittest.mock import MagicMock
        from sar_validation.downloaders.sar_downloader import SARDownloader

        dl = SARDownloader(output_dir=tmp_path, dry_run=False, force_download=True)
        fake_client = MagicMock()
        fake_client.query_products.return_value = [self._fake_record()]
        dl._client = fake_client
        fake_client.download_product.return_value = (
            tmp_path / "S1A_IW_OCN__2SDV_20260702T000000.SAFE"
        )
        (tmp_path / "S1A_IW_OCN__2SDV_20260702T000000").mkdir()

        dl.download(
            min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
            start="2026-07-02", end="2026-07-03",
        )

        fake_client.download_product.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_downloaders.py::TestSARDownloaderForceDownload -v`
Expected: `test_skips_product_whose_directory_already_exists` FAILs — today's code always calls `download_product`. `test_force_download_redownloads_existing_product` FAILs with `TypeError` — `force_download` isn't yet a constructor parameter.

- [ ] **Step 3: Implement**

In `sar_validation/downloaders/sar_downloader.py`, replace the constructor (lines 70-81):

```python
    def __init__(
        self,
        output_dir: Path,
        dry_run: bool = False,
        username: Optional[str] = None,
        password: Optional[str] = None,
        force_download: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self._username = username
        self._password = password
        self._client: Optional[CopernicusODataClient] = None
        self.force_download = force_download
```

Replace the download loop (lines 189-203):

```python
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            product_name = row["Name"]
            if not self.force_download and (self.output_dir / product_name).exists():
                print(f"[{i}/{len(df)}] Already downloaded: {product_name}")
                continue
            print(f"[{i}/{len(df)}] Downloading {product_name} …")
            try:
                path = client.download_product(row["Id"], self.output_dir, product_name)
                # Unzip if needed
                if path.suffix == ".zip":
                    with zipfile.ZipFile(path, "r") as zf:
                        zf.extractall(self.output_dir)
                    path.unlink()
                    print(f"  Unzipped to {self.output_dir}")
                else:
                    downloaded.append(path)
                    print(f"  Saved to {path}")
            except Exception as exc:
                print(f"  ERROR: {exc}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_downloaders.py::TestSARDownloaderForceDownload -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full downloader test file to check for regressions**

Run: `pytest tests/test_downloaders.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add sar_validation/downloaders/sar_downloader.py tests/test_downloaders.py
git commit -m "feat(sar): skip re-download when a product's output directory already exists"
```

---

### Task 11: Scatterometer downloader — per-product existence check

**Files:**
- Modify: `sar_validation/downloaders/scatterometer_downloader.py:73-84` (constructor), `:151-179` (download loop)
- Test: `tests/test_downloaders.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `ScatterometerDownloader.__init__(..., force_download: bool = False)` — new constructor parameter, `self.force_download`.

**Note:** EUMDAC doesn't expose a product's final filename before opening its stream, so the pre-check matches on the product ID string appearing as a substring of an existing output file's name (the naming scheme EUMDAC's own files already use, e.g. `OASWC12_20260705_183300_71590_M01.nc` contains the product's numeric ID).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_downloaders.py`, directly after `class TestScatterometerDownloaderAntimeridian:` ends (before `class TestInsituPlatformCodeMapping:`):

```python
class TestScatterometerDownloaderForceDownload:
    def test_skips_product_whose_output_file_already_exists(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from sar_validation.downloaders.scatterometer_downloader import ScatterometerDownloader

        dl = ScatterometerDownloader(output_dir=tmp_path, dry_run=False)
        dl._token = "fake-token"
        (tmp_path / "OASWC12_20260705_183300_71590_metopb.nc").write_bytes(b"")

        fake_eumdac = MagicMock()
        fake_collection = MagicMock()
        fake_collection.search.return_value = ["71590_metopb"]
        fake_datastore = MagicMock()
        fake_datastore.get_collection.return_value = fake_collection
        fake_eumdac.DataStore.return_value = fake_datastore

        with patch.dict("sys.modules", {"eumdac": fake_eumdac}):
            dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-07-02", end="2026-07-03",
            )

        fake_datastore.get_product.assert_not_called()

    def test_force_download_redownloads_existing_product(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from sar_validation.downloaders.scatterometer_downloader import ScatterometerDownloader

        dl = ScatterometerDownloader(output_dir=tmp_path, dry_run=False, force_download=True)
        dl._token = "fake-token"
        (tmp_path / "OASWC12_20260705_183300_71590_metopb.nc").write_bytes(b"")

        fake_eumdac = MagicMock()
        fake_collection = MagicMock()
        fake_collection.search.return_value = ["71590_metopb"]
        fake_datastore = MagicMock()
        fake_datastore.get_collection.return_value = fake_collection
        fake_eumdac.DataStore.return_value = fake_datastore

        fake_file = MagicMock()
        fake_file.name = "OASWC12_20260705_183300_71590_metopb.nc"
        fake_file.read.side_effect = [b"data", b""]
        fake_product = MagicMock()
        fake_product.open.return_value.__enter__.return_value = fake_file
        fake_product.open.return_value.__exit__.return_value = False
        fake_datastore.get_product.return_value = fake_product

        with patch.dict("sys.modules", {"eumdac": fake_eumdac}):
            dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-07-02", end="2026-07-03",
            )

        fake_datastore.get_product.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_downloaders.py::TestScatterometerDownloaderForceDownload -v`
Expected: `test_skips_product_whose_output_file_already_exists` FAILs — today's code always calls `get_product`. `test_force_download_redownloads_existing_product` FAILs with `TypeError` — `force_download` isn't yet a constructor parameter.

- [ ] **Step 3: Implement**

In `sar_validation/downloaders/scatterometer_downloader.py`, replace the constructor (lines 73-84):

```python
    def __init__(
        self,
        output_dir: Path,
        dry_run: bool = False,
        username: Optional[str] = None,
        password: Optional[str] = None,
        force_download: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self._username = username
        self._password = password
        self._token = None
        self.force_download = force_download
```

Replace the download loop (lines 151-179):

```python
        for product_id in products:
            product_str = str(product_id)
            if not any(sat in product_str.lower() for sat in SATELLITES):
                continue

            if not self.force_download and any(
                product_str in f.name for f in self.output_dir.glob("*") if f.is_file()
            ):
                print(f"  Already downloaded: {product_id}")
                continue

            try:
                product = datastore.get_product(
                    # eumdac's search yields product objects that get_product
                    # accepts here; its stub types product_id as str.
                    product_id=product_id,  # type: ignore[arg-type]
                    collection_id=COLLECTION_ID,
                )
                with product.open() as fsrc:
                    out_path = self.output_dir / fsrc.name
                    print(f"  Downloading {fsrc.name} …")
                    with open(out_path, "wb") as fdst:
                        shutil.copyfileobj(fsrc, fdst)

                if out_path.suffix == ".zip":
                    with zipfile.ZipFile(out_path, "r") as zf:
                        zf.extractall(self.output_dir)
                    out_path.unlink()
                    print(f"  Unzipped to {self.output_dir}")
                else:
                    downloaded.append(out_path)
                    print(f"  Saved to {out_path}")

            except Exception as exc:
                print(f"  ERROR downloading {product_id}: {exc}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_downloaders.py::TestScatterometerDownloaderForceDownload -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full downloader test file to check for regressions**

Run: `pytest tests/test_downloaders.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add sar_validation/downloaders/scatterometer_downloader.py tests/test_downloaders.py
git commit -m "feat(scatterometer): skip re-download when a product's output file already exists"
```

---

### Task 12: NOAA HF radar downloader — pre-download existence check

**Files:**
- Modify: `sar_validation/downloaders/noaa_hfradar_downloader.py:171-175` (constructor), `:202-226` (`_download_window`)
- Test: `tests/test_downloaders.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `NOAAHFRadarDownloader.__init__(..., force_download: bool = False)` — new constructor parameter, `self.force_download`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_downloaders.py`, directly after `class TestNOAAHFRadarDownloaderAntimeridian:` ends (before `class TestOrchestratorHFRadarNOAAWiring:`):

```python
class TestNOAAHFRadarDownloaderForceDownload:
    def test_skips_when_output_already_exists(self, tmp_path):
        from unittest.mock import patch
        from sar_validation.downloaders.noaa_hfradar_downloader import (
            NOAAHFRadarDownloader, select_erddap_dataset,
        )

        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
        dataset_id = select_erddap_dataset(-125, -119, 33, 38, 6)
        out_path = tmp_path / f"{dataset_id}_6km_{_RECENT_START}.nc"
        out_path.write_bytes(b"")

        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlretrieve"
        ) as m:
            out = dl.download(-125, -119, 33, 38, _RECENT_START, _RECENT_END)

        m.assert_not_called()
        assert out == [out_path]

    def test_force_download_refetches_existing_output(self, tmp_path):
        from unittest.mock import patch
        from sar_validation.downloaders.noaa_hfradar_downloader import (
            NOAAHFRadarDownloader, select_erddap_dataset,
        )

        dl = NOAAHFRadarDownloader(
            output_dir=tmp_path, dry_run=False, resolution_km=6, force_download=True,
        )
        dataset_id = select_erddap_dataset(-125, -119, 33, 38, 6)
        out_path = tmp_path / f"{dataset_id}_6km_{_RECENT_START}.nc"
        out_path.write_bytes(b"stale")

        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlretrieve"
        ) as m:
            dl.download(-125, -119, 33, 38, _RECENT_START, _RECENT_END)

        m.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_downloaders.py::TestNOAAHFRadarDownloaderForceDownload -v`
Expected: `test_skips_when_output_already_exists` FAILs — today's code always calls `urlretrieve`. `test_force_download_refetches_existing_output` FAILs with `TypeError` — `force_download` isn't yet a constructor parameter.

- [ ] **Step 3: Implement**

In `sar_validation/downloaders/noaa_hfradar_downloader.py`, replace the constructor (lines 171-175):

```python
    def __init__(self, output_dir: Path, dry_run: bool = False,
                 resolution_km: int = DEFAULT_RESOLUTION_KM,
                 force_download: bool = False) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.resolution_km = resolution_km
        self.force_download = force_download
```

Replace `_download_window` (lines 202-226) — compute `out_path` before the `dry_run` branch (so both the dry-run message and the new pre-check can reference it), and add the pre-check after that branch:

```python
    def _download_window(
        self, min_lon, max_lon, min_lat, max_lat, start: str, end: str, filename_suffix: str,
    ) -> Optional[Path]:
        backend = select_backend(end)  # raises if archive (Phase 3b) needed
        dataset_id = select_erddap_dataset(
            min_lon, max_lon, min_lat, max_lat, self.resolution_km
        )
        min_lon, max_lon, min_lat, max_lat = clamp_to_region_bbox(
            min_lon, max_lon, min_lat, max_lat
        )
        start_d = normalize_datetime(start).split("T")[0]
        end_d = normalize_datetime(end).split("T")[0]
        date_str = start_d if start_d == end_d else f"{start_d}_{end_d}"
        out_path = self.output_dir / f"{dataset_id}_{self.resolution_km}km_{date_str}{filename_suffix}.nc"
        url = build_erddap_subset_url(
            dataset_id, min_lon, max_lon, min_lat, max_lat, start, end
        )

        if self.dry_run:
            print(f"[dry-run] NOAA HF-radar ({backend}) would download:\n  {url}")
            return None

        if not self.force_download and out_path.exists():
            print(f"  Already downloaded: {out_path}")
            return out_path

        self.output_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(out_path))
        return out_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_downloaders.py::TestNOAAHFRadarDownloaderForceDownload -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full downloader test file to check for regressions**

Run: `pytest tests/test_downloaders.py -v`
Expected: PASS — in particular the existing `TestNOAAHFRadarDownload` class (`test_dry_run_returns_empty_list_and_no_fetch`, `test_download_fetches_url_to_expected_path`, the two clamp tests), since `out_path` is now computed earlier but its value and the dry-run message are unchanged.

- [ ] **Step 6: Commit**

```bash
git add sar_validation/downloaders/noaa_hfradar_downloader.py tests/test_downloaders.py
git commit -m "feat(noaa_hfradar): skip re-download when the expected output file already exists"
```

---

### Task 13: Wire `--force-download` through `DataOrchestrator` to every downloader

**Files:**
- Modify: `sar_validation/core/orchestrator.py:44-56` (`__init__`), `:139-142` (`_download_sar`), `:177-182` (`_download_insitu`), `:229` (`_download_scatterometer`), `:257-262` (`_download_hf_radar`), `:294-298` (`_download_noaa_hfradar`), `:324` (`_download_hf_radar_historical`), `:358` (`_download_altimeter`)
- Modify: `sar_validation/cli.py:494` (`_execute_recipe`'s `DataOrchestrator(...)` call)
- Test: `tests/test_downloaders.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: every downloader's `force_download: bool = False` constructor parameter (Tasks 6-12); `AltimeterDownloader`/`ScatterometerDownloader`/`RadiometerDownloader` already accepted arbitrary kwargs correctly — this task is the only one that touches `orchestrator.py`.
- Produces: `DataOrchestrator.__init__(..., force_download: bool = False)` — new constructor parameter, `self.force_download`, passed to every downloader constructor `download_all()` instantiates (except `RadiometerDownloader`, out of scope — see the design spec's "Out of scope" section).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_downloaders.py`, at the end of the file:

```python
# ---------------------------------------------------------------------------
# DataOrchestrator force_download wiring
# ---------------------------------------------------------------------------

class TestOrchestratorForceDownloadWiring:
    def _make_orchestrator(self, tmp_path, force_download):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig

        recipe = Recipe(RecipeConfig(
            name="test-force-download",
            variable="wind",
            output_dir=str(tmp_path),
        ))
        return DataOrchestrator(recipe, dry_run=True, force_download=force_download)

    def test_sar_receives_force_download(self, tmp_path):
        from unittest.mock import patch

        orchestrator = self._make_orchestrator(tmp_path, force_download=True)
        with patch("sar_validation.downloaders.sar_downloader.SARDownloader") as mock_cls:
            mock_cls.return_value.download.return_value = []
            orchestrator._download_sar()
        assert mock_cls.call_args.kwargs["force_download"] is True

    def test_insitu_receives_force_download(self, tmp_path):
        from unittest.mock import patch

        orchestrator = self._make_orchestrator(tmp_path, force_download=True)
        with patch("sar_validation.downloaders.insitu_downloader.InSituDownloader") as mock_cls:
            mock_cls.return_value.download.return_value = []
            orchestrator._download_insitu(["mooring"], -20.0, 20.0)
        assert mock_cls.call_args.kwargs["force_download"] is True

    def test_scatterometer_receives_force_download(self, tmp_path):
        from unittest.mock import patch
        from sar_validation.core.recipe import ValidationDataSource

        orchestrator = self._make_orchestrator(tmp_path, force_download=True)
        with patch(
            "sar_validation.downloaders.scatterometer_downloader.ScatterometerDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = []
            orchestrator._download_scatterometer(ValidationDataSource(source_type="scatterometer"))
        assert mock_cls.call_args.kwargs["force_download"] is True

    def test_hf_radar_receives_force_download(self, tmp_path):
        from unittest.mock import patch
        from sar_validation.core.recipe import ValidationDataSource

        orchestrator = self._make_orchestrator(tmp_path, force_download=True)
        with patch("sar_validation.downloaders.hf_radar_downloader.HFRadarDownloader") as mock_cls:
            mock_cls.return_value.download.return_value = []
            orchestrator._download_hf_radar(ValidationDataSource(source_type="hf_radar"))
        assert mock_cls.call_args.kwargs["force_download"] is True

    def test_noaa_hfradar_receives_force_download(self, tmp_path):
        from unittest.mock import patch
        from sar_validation.core.recipe import ValidationDataSource

        orchestrator = self._make_orchestrator(tmp_path, force_download=True)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.NOAAHFRadarDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = []
            orchestrator._download_noaa_hfradar(ValidationDataSource(source_type="hf_radar_noaa"))
        assert mock_cls.call_args.kwargs["force_download"] is True

    def test_hf_radar_historical_receives_force_download(self, tmp_path):
        from unittest.mock import patch
        from sar_validation.core.recipe import ValidationDataSource

        orchestrator = self._make_orchestrator(tmp_path, force_download=True)
        with patch(
            "sar_validation.downloaders.hf_radar_historical_downloader.HFRadarHistoricalDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = []
            orchestrator._download_hf_radar_historical(
                ValidationDataSource(source_type="hf_radar_historical")
            )
        assert mock_cls.call_args.kwargs["force_download"] is True

    def test_altimeter_receives_force_download(self, tmp_path):
        from unittest.mock import patch
        from sar_validation.core.recipe import ValidationDataSource

        orchestrator = self._make_orchestrator(tmp_path, force_download=True)
        with patch("sar_validation.downloaders.altimeter_downloader.AltimeterDownloader") as mock_cls:
            mock_cls.return_value.download.return_value = []
            orchestrator._download_altimeter(ValidationDataSource(source_type="altimeter"))
        assert mock_cls.call_args.kwargs["force_download"] is True

    def test_default_force_download_is_false(self, tmp_path):
        from unittest.mock import patch

        orchestrator = self._make_orchestrator(tmp_path, force_download=False)
        with patch("sar_validation.downloaders.sar_downloader.SARDownloader") as mock_cls:
            mock_cls.return_value.download.return_value = []
            orchestrator._download_sar()
        assert mock_cls.call_args.kwargs["force_download"] is False
```

Add to `tests/test_cli.py`, as a new class at the end of the file:

```python
class TestExecuteRecipePassesForceDownloadToOrchestrator:
    def test_force_download_flag_reaches_orchestrator_constructor(self, tmp_path):
        from unittest.mock import patch
        from sar_validation.core.recipe import Recipe, RecipeConfig

        recipe_path = tmp_path / "recipe.yaml"
        Recipe(RecipeConfig(
            name="test", variable="wind", output_dir=str(tmp_path / "run"),
        )).to_yaml(recipe_path)

        with patch("sar_validation.core.orchestrator.DataOrchestrator") as mock_cls:
            mock_cls.return_value.download_all.return_value = True
            mock_cls.return_value.base_dir = tmp_path / "run"
            cli._execute_recipe(str(recipe_path), force_download=True)

        assert mock_cls.call_args.kwargs["force_download"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_downloaders.py::TestOrchestratorForceDownloadWiring tests/test_cli.py::TestExecuteRecipePassesForceDownloadToOrchestrator -v`
Expected: All FAIL — `DataOrchestrator(recipe, dry_run=True, force_download=True)` raises `TypeError` today (no such constructor parameter); the CLI test similarly fails since `_execute_recipe` doesn't yet pass `force_download` to `DataOrchestrator(...)`.

- [ ] **Step 3: Implement**

In `sar_validation/core/orchestrator.py`, replace `__init__` (lines 44-56):

```python
    def __init__(self, recipe: Recipe, dry_run: bool = False, force_download: bool = False) -> None:
        self.recipe   = recipe
        self.dry_run  = dry_run
        self.force_download = force_download
        self.base_dir = self._setup_base_dir()
        self.metadata: Dict[str, Any] = {
            "recipe_name": recipe.config.name,
            "variable":    recipe.config.variable,
            "created":     datetime.now().isoformat(),
            "geographic_bounds": recipe.config.geographic_bounds.to_dict(),
            "temporal_bounds":   recipe.config.temporal_bounds.to_dict(),
            "downloads": {},
            "errors":    [],
        }
```

Replace the `SARDownloader(...)` construction in `_download_sar` (lines 139-142):

```python
            dl = SARDownloader(
                output_dir=out_dir,
                dry_run=self.dry_run,
                force_download=self.force_download,
            )
```

Replace the `InSituDownloader(...)` construction in `_download_insitu` (lines 177-182):

```python
            dl = InSituDownloader(
                output_dir=out_dir,
                dry_run=self.dry_run,
                min_depth=min_depth,
                max_depth=max_depth,
                force_download=self.force_download,
            )
```

Replace the `ScatterometerDownloader(...)` construction in `_download_scatterometer` (line 229):

```python
            dl = ScatterometerDownloader(
                output_dir=out_dir, dry_run=self.dry_run, force_download=self.force_download,
            )
```

Replace the `HFRadarDownloader(...)` construction in `_download_hf_radar` (lines 257-262):

```python
            dl = HFRadarDownloader(
                output_dir=out_dir,
                dry_run=self.dry_run,
                min_depth=source.resolved_min_depth,
                max_depth=source.resolved_max_depth,
                force_download=self.force_download,
            )
```

Replace the `NOAAHFRadarDownloader(...)` construction in `_download_noaa_hfradar` (lines 294-298):

```python
            dl = NOAAHFRadarDownloader(
                output_dir=out_dir,
                dry_run=self.dry_run,
                resolution_km=resolution_km,
                force_download=self.force_download,
            )
```

Replace the `HFRadarHistoricalDownloader(...)` construction in `_download_hf_radar_historical` (line 324):

```python
            dl = HFRadarHistoricalDownloader(
                output_dir=out_dir, dry_run=self.dry_run, force_download=self.force_download,
            )
```

Replace the `AltimeterDownloader(...)` construction in `_download_altimeter` (line 358):

```python
            dl = AltimeterDownloader(
                output_dir=out_dir, dry_run=self.dry_run, force_download=self.force_download,
            )
```

`_download_radiometer` is unchanged — `RadiometerDownloader` is out of scope (it downloads whole global grids unconditionally and crops locally; per-file resume there isn't part of this plan).

In `sar_validation/cli.py`, replace the `DataOrchestrator(...)` construction in `_execute_recipe` (line 494):

```python
    orchestrator = DataOrchestrator(recipe, dry_run=dry_run, force_download=force_download)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_downloaders.py::TestOrchestratorForceDownloadWiring tests/test_cli.py::TestExecuteRecipePassesForceDownloadToOrchestrator -v`
Expected: PASS (9 tests total across both classes)

- [ ] **Step 5: Run the entire test suite**

Run: `pytest -v`
Expected: PASS (no regressions anywhere in the suite)

- [ ] **Step 6: Manual smoke test with the real CLI**

Run these two commands in sequence against any example recipe, e.g. `recipes/wind_validation.yaml` (or any recipe with a `data/` directory already downloaded from a prior run):

```bash
python -m sar_validation --recipe recipes/wind_validation.yaml --dry-run
python -m sar_validation --recipe recipes/wind_validation.yaml --force-download --dry-run
```

Expected: both commands complete without error (note: `python -m sar_validation.cli` is a no-op in this repo — `cli.py` has no `__main__` guard; use `python -m sar_validation` or the installed `sar-validate` script, per the antimeridian-crossing plan's Task 11 finding). This is a `--dry-run`, so no network calls or per-file skip/overwrite decisions are actually exercised (every downloader's `dry_run` branch short-circuits before the new checks) — this step only confirms the wiring itself doesn't crash the CLI end-to-end. A full non-dry-run confirmation of skip/overwrite behavior is already covered by Tasks 6-12's unit tests.

- [ ] **Step 7: Commit**

```bash
git add sar_validation/core/orchestrator.py sar_validation/cli.py tests/test_downloaders.py tests/test_cli.py
git commit -m "feat: wire --force-download through DataOrchestrator to every downloader"
```

---

## Post-plan verification checklist

- [ ] `pytest -v` passes in full (all tasks' tests plus the pre-existing suite).
- [ ] `python -m sar_validation --recipe <any recipe> --dry-run` and the same with `--force-download` both run clean (Task 13, Step 6).
- [ ] Re-read `docs/superpowers/specs/2026-07-18-download-resilience-collocation-fixes-design.md` and confirm every section (B1, B2, B3, Bug C, Bug D, Testing) has a corresponding task above.
