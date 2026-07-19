# Lint Cleanup and Ruff Adoption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all dead code package-wide, fix the six pytest warnings, adopt ruff (rules E/F/I, 120-char lines), and enforce both via GitHub Actions CI.

**Architecture:** Pure cleanup — no new modules. Six tasks: (1) the two VS Code warnings in statistics.py, (2) the six pytest warnings, (3) ruff config + auto-fixes, (4) remaining manual fixes in package code, (5) remaining manual fixes in tests, (6) CI workflow + final verification. Each task leaves the full test suite green.

**Tech Stack:** Python ≥3.10, ruff (lint only, no formatter), pytest, mypy, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-07-18-lint-cleanup-ruff-design.md`

## Global Constraints

- Ruff rules: `select = ["E", "F", "I"]`, `line-length = 120`, `target-version = "py310"`. No formatter adoption.
- `tests/*` gets a per-file ignore for `E402` only (test files deliberately import section-by-section).
- No behavior change except the axis-limit padding in `plot_scatter` for constant-value data (spec §4).
- No typing modernization (`Optional[X]` stays `Optional[X]`), no refactoring beyond what a finding requires.
- Test count must stay 414 passing; the six pytest warnings from spec §4 must be gone.
- `ruff` is not installed in the project env at start. Use `uvx ruff` for all ruff commands, or `pip install ruff` once and call `ruff` directly — either is fine, be consistent.
- Before deleting any import/variable not explicitly listed in this plan, grep the repo for usage first (spec §6).

---

### Task 1: statistics.py dead code + docstring (the original VS Code warnings)

**Files:**
- Modify: `sar_validation/core/statistics.py:17` (import), `:151-155` (long line), `:252-254` (docstring), `:267-268` (dead assignment)
- Test: `tests/test_statistics.py` (existing suite, no new tests)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: nothing other tasks rely on; `statistics.py` public API unchanged.

- [ ] **Step 1: Capture the mypy baseline (used by Task 6's "no worse" check)**

Run:
```bash
python -m mypy sar_validation/ 2>&1 | tail -1 | tee /tmp/mypy_baseline.txt
```
If mypy is not installed: `pip install mypy`, then rerun. Expected: a summary line like `Found N errors in M files` or `Success: no issues found`. Whatever it says is the baseline.

- [ ] **Step 2: Remove the unused import name**

In `sar_validation/core/statistics.py` line 17, change:

```python
from ._variable_map import CIRCULAR_VAL_VARS, circular_diff_deg, infer_variable_pairs, filter_variable_pairs
```

to:

```python
from ._variable_map import CIRCULAR_VAL_VARS, circular_diff_deg, filter_variable_pairs
```

- [ ] **Step 3: Remove the dead local variable**

In `run_statistics` (around line 267), change:

```python
    base_dir = Path(base_dir)
    variable = recipe.config.variable

    try:
```

to:

```python
    base_dir = Path(base_dir)

    try:
```

- [ ] **Step 4: Fix the stale docstring**

In the `run_statistics` docstring (around line 252), change:

```python
    recipe : Recipe
        Recipe object; its ``config.variable`` field is used to infer the
        (sar_var, val_var) pairs via :func:`~._variable_map.infer_variable_pairs`.
```

to:

```python
    recipe : Recipe
        Recipe object; its ``config.variable`` field determines the
        (sar_var, val_var) pairs via :func:`~._variable_map.filter_variable_pairs`.
```

- [ ] **Step 5: Wrap the one >120-char line (E501)**

Around line 153 in `compute_statistics`, change:

```python
                diff_rad = np.radians(diff)
                resultant_length = np.hypot(np.mean(np.cos(diff_rad)), np.mean(np.sin(diff_rad)))
                std = float(np.degrees(np.sqrt(-2.0 * np.log(resultant_length)))) if resultant_length > 0 else float("nan")
```

to:

```python
                diff_rad = np.radians(diff)
                resultant_length = np.hypot(np.mean(np.cos(diff_rad)), np.mean(np.sin(diff_rad)))
                if resultant_length > 0:
                    std = float(np.degrees(np.sqrt(-2.0 * np.log(resultant_length))))
                else:
                    std = float("nan")
```

- [ ] **Step 6: Run the statistics tests**

Run: `pytest tests/test_statistics.py -q`
Expected: all pass, 0 failures.

- [ ] **Step 7: Confirm the two findings are gone**

Run: `uvx ruff check sar_validation/core/statistics.py --select F401,F841,E501 --line-length 120`
Expected: `All checks passed!`

- [ ] **Step 8: Commit**

```bash
git add sar_validation/core/statistics.py
git commit -m "fix: remove dead code in statistics.py left by filter_variable_pairs refactor"
```

---

### Task 2: Fix the six pytest warnings

**Files:**
- Modify: `tests/test_datatree_converter.py:204,448` (FutureWarning ×2)
- Modify: `sar_validation/core/visualization.py:418-419` area (`plot_scatter`, UserWarning ×2)
- Modify: `tests/test_visualization.py:133-147` (strengthen existing test), `:1739-1746` (fixture, UserWarning ×2)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: nothing other tasks rely on. `plot_scatter` signature unchanged; only its axis-limit behavior for constant-value input changes.

- [ ] **Step 1: Verify the FutureWarnings fire today**

Run: `pytest tests/test_datatree_converter.py -W error::FutureWarning -q`
Expected: 2 FAILURES (`TestFromInsituCsv::test_point_dimension`, `TestFromCollocations::test_basic`), each with `FutureWarning: The return type of Dataset.dims will be changed…`.

- [ ] **Step 2: Switch `ds.dims[...]` to `ds.sizes[...]`**

`tests/test_datatree_converter.py` line 204:

```python
        assert ds.dims["point"] == 10
```
→
```python
        assert ds.sizes["point"] == 10
```

Line 448:

```python
        assert ds.dims["collocation"] == 3
```
→
```python
        assert ds.sizes["collocation"] == 3
```

- [ ] **Step 3: Verify the FutureWarnings are gone**

Run: `pytest tests/test_datatree_converter.py -W error::FutureWarning -q`
Expected: all pass.

- [ ] **Step 4: Strengthen the constant-values test so it also fails on UserWarning (TDD)**

In `tests/test_visualization.py`, `test_constant_values_no_runtime_warning` (line ~133), change:

```python
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            fig = plot_scatter(ds, "owiWindSpeed", "WSPD")
```

to:

```python
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            warnings.simplefilter("error", UserWarning)
            fig = plot_scatter(ds, "owiWindSpeed", "WSPD")
```

- [ ] **Step 5: Run it to verify it fails**

Run: `pytest tests/test_visualization.py -k constant_values -q`
Expected: FAIL with `UserWarning: Attempting to set identical low and high xlims…`.

- [ ] **Step 6: Pad singular axis limits in `plot_scatter`**

In `sar_validation/core/visualization.py` (around line 417, the static matplotlib branch), change:

```python
    all_vals = np.concatenate([df[val_col].values, df[sar_col].values])
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    line11 = ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1, label="1:1")[0]
```

to:

```python
    all_vals = np.concatenate([df[val_col].values, df[sar_col].values])
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    if vmin == vmax:
        # All values identical — pad to a non-degenerate range so
        # set_xlim/set_ylim don't warn about singular limits.
        pad = max(0.5, abs(vmin) * 0.05)
        vmin -= pad
        vmax += pad
    line11 = ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1, label="1:1")[0]
```

(Do NOT touch the similar `vmin, vmax` at line ~372 — that's the plotly branch, which doesn't warn.)

- [ ] **Step 7: Run the test to verify it passes**

Run: `pytest tests/test_visualization.py -k constant_values -q`
Expected: PASS.

- [ ] **Step 8: Verify the fallback UserWarnings fire today**

Run:
```bash
pytest "tests/test_visualization.py::TestValidationReportCurrentsPointSize::test_currents_recipe_passes_reduced_point_size_to_geographic" -W error::UserWarning -q
```
Expected: FAIL with `UserWarning: color_by='temporal_offset' requested but collocation_ds has no 'temporal_distance_minutes' column…`. (If an *unrelated* UserWarning fails first — e.g. a cartopy data download notice — note it and still proceed; the assertion in step 10 is the real gate.)

- [ ] **Step 9: Add the missing column to the test fixture**

In `tests/test_visualization.py`, `test_currents_recipe_passes_reduced_point_size_to_geographic` (line ~1739), change:

```python
        coll = xr.Dataset({
            "sar_rvlRadVel":            ("collocation", [0.3, 0.31, 0.29, 0.32]),
            "val_rvlRadVel_projection": ("collocation", [0.28, 0.30, 0.27, 0.31]),
            "val_source":               ("collocation", ["hf_radar"] * 4),
            "sar_scene_name":           ("collocation", ["sceneA"] * 4),
            "val_lon":                  ("collocation", [-9.5, -9.4, -9.3, -9.2]),
            "val_lat":                  ("collocation", [50.5, 50.6, 50.7, 50.8]),
        })
```

to:

```python
        coll = xr.Dataset({
            "sar_rvlRadVel":             ("collocation", [0.3, 0.31, 0.29, 0.32]),
            "val_rvlRadVel_projection":  ("collocation", [0.28, 0.30, 0.27, 0.31]),
            "val_source":                ("collocation", ["hf_radar"] * 4),
            "sar_scene_name":            ("collocation", ["sceneA"] * 4),
            "val_lon":                   ("collocation", [-9.5, -9.4, -9.3, -9.2]),
            "val_lat":                   ("collocation", [50.5, 50.6, 50.7, 50.8]),
            "temporal_distance_minutes": ("collocation", [10.0, 12.0, 8.0, 15.0]),
        })
```

- [ ] **Step 10: Verify both fallback warnings are gone and the test still passes**

Run:
```bash
pytest "tests/test_visualization.py::TestValidationReportCurrentsPointSize::test_currents_recipe_passes_reduced_point_size_to_geographic" -q 2>&1 | tail -15
```
Expected: PASS, and the output contains NO `color_by='temporal_offset'` warning and NO `No valid data for sar_rvlRadVel` warning.

- [ ] **Step 11: Run both affected test files fully**

Run: `pytest tests/test_datatree_converter.py tests/test_visualization.py -q`
Expected: all pass, and the warnings summary no longer lists any of the six spec §4 warnings.

- [ ] **Step 12: Commit**

```bash
git add tests/test_datatree_converter.py tests/test_visualization.py sar_validation/core/visualization.py
git commit -m "fix: eliminate all six pytest warnings (Dataset.sizes, singular axis limits, fixture column)"
```

---

### Task 3: Ruff configuration + automatic fixes

**Files:**
- Modify: `pyproject.toml` (ruff config + dev dependency)
- Modify: many files via `ruff --fix` (import sorting, duplicate imports, most unused imports)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: the `[tool.ruff]` config that Tasks 4–6 run against. Tasks 4+ assume `ruff check .` reports only the manual findings listed there.

- [ ] **Step 1: Add ruff config to `pyproject.toml`**

Insert between `[tool.pytest.ini_options]` and `[tool.mypy]`:

```toml
[tool.ruff]
line-length = 120
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I"]

[tool.ruff.lint.per-file-ignores]
# Test modules deliberately import section-by-section (each downloader /
# component gets its own section with its own imports), so module-level
# imports below the top of the file are fine there.
"tests/*" = ["E402"]
```

- [ ] **Step 2: Add ruff to the dev extra**

In `pyproject.toml` `[project.optional-dependencies]`, change:

```toml
dev = [
    "pytest",
    "pytest-cov",
    "mypy",
```

to:

```toml
dev = [
    "pytest",
    "pytest-cov",
    "mypy",
    "ruff",
```

- [ ] **Step 3: Record the pre-fix finding count**

Run: `uvx ruff check . --statistics`
Expected roughly: ~150 I001, ~25 F811, ~20 F401, single-digit F841/E501/E731, 1 F821, 1 F541. (E402 should no longer appear — the per-file ignore covers all 11.)

- [ ] **Step 4: Apply automatic fixes**

Run: `uvx ruff check . --fix`
Expected: ~200 findings fixed; remaining ~20 (the manual ones for Tasks 4–5).

- [ ] **Step 5: Verify the MagicMock interplay resolved correctly**

`tests/test_downloaders.py` had `MagicMock` unused at top (F401) *because* 25 later sections re-imported it locally (F811). The F811 fix removes the local re-imports, which makes the top-level import used again. Verify:

Run: `grep -n "MagicMock" tests/test_downloaders.py | head -5`
Expected: line 6 still contains `from unittest.mock import ... MagicMock ...`, and the former local `from unittest.mock import ... MagicMock` re-imports inside the file are gone (any that remain must import only names not already at top). If ruff removed the top-level `MagicMock` too, re-add it to line 6 — the tests below will catch this as `NameError` otherwise.

- [ ] **Step 6: Review the diff**

Run: `git diff --stat`
Expected: changes concentrated in import blocks. Skim `git diff` for anything that is not an import re-ordering or a removed unused import; anything else needs manual inspection before proceeding.

Note on `sar_validation/core/orchestrator.py`: the autofix removes its unused `RecipeConfig`, `GeographicBounds`, `TemporalBounds` imports. This is safe — already verified: both `sar_validation/__init__.py` and `sar_validation/core/__init__.py` re-export those names from `.recipe` directly, and `cli.py` imports them from the recipe module, so nothing imports them via orchestrator.

- [ ] **Step 7: Run the full test suite**

Run: `pytest -q 2>&1 | tail -3`
Expected: `414 passed` (plus possible skips), 0 failures.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: adopt ruff (E/F/I, 120 cols) and apply automatic fixes"
```

---

### Task 4: Manual lint fixes — package code

**Files:**
- Modify: `sar_validation/core/visualization.py:34` (typing import), `:~620` (cfeature), `:~1295` (HAS_CARTOPY), `:1921` (annotation)
- Modify: `sar_validation/downloaders/insitu_downloader.py:105-106` (lambdas)
- Modify: `sar_validation/cli.py:55-59` (long help lines)
- Modify: `sar_validation/core/_cf_metadata.py:174,178` (long dict lines)
- Modify: `sar_validation/core/collocation.py:289,445` (long lines)
- Modify: `sar_validation/core/datatree_converter.py:~658` (long set literal)

**Interfaces:**
- Consumes: ruff config from Task 3.
- Produces: `ruff check sar_validation/` passes clean; Task 5 only has `tests/` findings left.

Line numbers below are pre-Task-3 positions; import sorting may have shifted them slightly. Locate each site by its code content.

- [ ] **Step 1: Remove the unused `cartopy.feature` probe import (F401)**

In `sar_validation/core/visualization.py` (~line 620, inside `plot_geographic`'s static branch), change:

```python
    try:
        import cartopy.crs as ccrs  # noqa: PLC0415
        import cartopy.feature as cfeature  # noqa: PLC0415
        HAS_CARTOPY = True
```

to:

```python
    try:
        import cartopy.crs as ccrs  # noqa: PLC0415
        HAS_CARTOPY = True
```

The spec suggested `importlib.util.find_spec`, but that's unnecessary here: the `ccrs` import in the same `try` is genuinely used, so it already serves as the availability probe. This `HAS_CARTOPY` *is* used later in the function — leave it.

- [ ] **Step 2: Remove the dead `HAS_CARTOPY` in `plot_collocation_diagnostics` (F841)**

In `sar_validation/core/visualization.py` (~line 1293), change:

```python
    # Set up cartopy if available
    try:
        import cartopy.crs as ccrs  # noqa: PLC0415
        HAS_CARTOPY = True
    except ImportError:
        HAS_CARTOPY = False
        logger.debug("cartopy not installed — collocation_diagnostics plot unavailable.")
        return None
```

to:

```python
    # Set up cartopy if available
    try:
        import cartopy.crs as ccrs  # noqa: PLC0415
    except ImportError:
        logger.debug("cartopy not installed — collocation_diagnostics plot unavailable.")
        return None
```

(Unlike step 1's block, this `HAS_CARTOPY` is never read — the except branch returns immediately.)

- [ ] **Step 3: Fix the `plt.Figure` annotation (F821)**

In `sar_validation/core/visualization.py` line 34, change:

```python
from typing import Dict, List, Optional, Sequence, Tuple, Union
```

to:

```python
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
```

After the top-level imports (right before `logger = logging.getLogger(__name__)`), add:

```python
if TYPE_CHECKING:
    from matplotlib.figure import Figure
```

At line ~1921, change:

```python
def plot_rvl_land_qa(datatree) -> Optional["plt.Figure"]:
```

to:

```python
def plot_rvl_land_qa(datatree) -> Optional["Figure"]:
```

(`plt` is only imported inside function bodies, so the old string annotation named something that doesn't exist at module scope. Matplotlib is an optional dependency — hence `TYPE_CHECKING`, not a plain import.)

- [ ] **Step 4: Replace the two lambda assignments (E731)**

In `sar_validation/downloaders/insitu_downloader.py` (~line 105), change:

```python
    lon_sfx = lambda v: "W" if v < 0 else "E"
    lat_sfx = lambda v: "S" if v < 0 else "N"
```

to:

```python
    def lon_sfx(v: float) -> str:
        return "W" if v < 0 else "E"

    def lat_sfx(v: float) -> str:
        return "S" if v < 0 else "N"
```

- [ ] **Step 5: Wrap the five long help-epilog lines in `cli.py` (E501)**

At lines 55–59, change:

```
  sar-validate --create-recipe wind --min-lon -10 --max-lon 5 --min-lat 50 --max-lat 65 --start 2026-03-01 --end 2026-03-31
  sar-validate --create-recipe wind --min-lon -10 --max-lon 5 --min-lat 50 --max-lat 65 --start 2026-03-01 --end 2026-03-31 --recipe-name north_sea_march_2026
  sar-validate --recipe recipes/wind_validation.yaml --dry-run #no data will be downloaded, just show what would be downloaded
  sar-validate --recipe recipes/wind_validation.yaml # for downloading the data if there is not already a download_metadata.json file in the data folder
  sar-validate --recipe recipes/wind_validation.yaml --force-download # overrides the download_metadata.json and redownloads the data
```

to:

```
  sar-validate --create-recipe wind --min-lon -10 --max-lon 5 --min-lat 50 --max-lat 65 \
      --start 2026-03-01 --end 2026-03-31
  sar-validate --create-recipe wind --min-lon -10 --max-lon 5 --min-lat 50 --max-lat 65 \
      --start 2026-03-01 --end 2026-03-31 --recipe-name north_sea_march_2026
  # Dry run: no data downloaded, just show what would be downloaded
  sar-validate --recipe recipes/wind_validation.yaml --dry-run
  # Download the data (skipped when download_metadata.json already exists in the data folder)
  sar-validate --recipe recipes/wind_validation.yaml
  # Ignore download_metadata.json and redownload the data
  sar-validate --recipe recipes/wind_validation.yaml --force-download
```

This is inside the argparse `epilog` string — keep the two-space indent so the examples line up in `--help` output.

- [ ] **Step 6: Wrap the two long CF-metadata dict lines (E501)**

In `sar_validation/core/_cf_metadata.py` line 174, change:

```python
    "spatial_distance_km": {"long_name": "great-circle distance between SAR cell and validation observation", "units": "km"},
```

to:

```python
    "spatial_distance_km": {
        "long_name": "great-circle distance between SAR cell and validation observation",
        "units": "km",
    },
```

Line 178, change:

```python
    "temporal_distance_minutes": {"long_name": "absolute time offset between SAR acquisition and validation observation, in minutes"},
```

to:

```python
    "temporal_distance_minutes": {
        "long_name": "absolute time offset between SAR acquisition and validation observation, in minutes",
    },
```

(Keep the existing comment above the `temporal_distance_minutes` entry about deliberately having no `units` attr.)

- [ ] **Step 7: Fix the two long lines in `collocation.py` (E501)**

Line ~289, change:

```python
        # Pre-filters: eliminate validation rows that cannot match
        # Use spatial_tolerance_km for initial bounding box
        deg_buf = self.spatial_tolerance_km / 100.0   # Use spatial_tolerance_km for pre-filter; 100 converts from km to degrees (conservatively)
```

to:

```python
        # Pre-filters: eliminate validation rows that cannot match
        # Use spatial_tolerance_km for the initial bounding box; dividing
        # by 100 converts km to degrees (conservatively).
        deg_buf = self.spatial_tolerance_km / 100.0
```

Line ~445, change:

```python
                if val:
                    try:
                        t0 = np.datetime64(pd.Timestamp(val).tz_convert(None) if hasattr(pd.Timestamp(val), 'tz') and pd.Timestamp(val).tzinfo else pd.Timestamp(val), "ns")
                        time_arr = np.full(n_points, t0)
                        break
```

to:

```python
                if val:
                    try:
                        ts = pd.Timestamp(val)
                        if ts.tzinfo is not None:
                            ts = ts.tz_convert(None)
                        t0 = np.datetime64(ts, "ns")
                        time_arr = np.full(n_points, t0)
                        break
```

(Behavior-identical: `hasattr(Timestamp, "tz")` is always true, so the old condition reduced to `tzinfo is not None`.)

- [ ] **Step 8: Wrap the long set literal in `datatree_converter.py` (E501)**

At ~line 658, change:

```python
LAYER_SOURCE_PATHS = {"osi_saf_winds", "scatterometer", "altimeter", "hf_radar", "hf_radar_grid", "hfr_noaa", "radiometer"}
```

to:

```python
LAYER_SOURCE_PATHS = {
    "osi_saf_winds", "scatterometer", "altimeter", "hf_radar",
    "hf_radar_grid", "hfr_noaa", "radiometer",
}
```

- [ ] **Step 9: Verify package code is fully clean**

Run: `uvx ruff check sar_validation/`
Expected: `All checks passed!` If any E501 stragglers remain (line numbers may have shifted since the scan), wrap them the same way as above — no `# noqa: E501`.

- [ ] **Step 10: Run the full test suite**

Run: `pytest -q 2>&1 | tail -3`
Expected: `414 passed`, 0 failures.

- [ ] **Step 11: Commit**

```bash
git add sar_validation/
git commit -m "chore: fix remaining lint findings in package code"
```

---

### Task 5: Manual lint fixes — tests

**Files:**
- Modify: `tests/test_downloaders.py:120,131,142,153,171` (unused date locals)
- Modify: `tests/test_visualization.py:~1425` (unused `figures`)

**Interfaces:**
- Consumes: ruff config from Task 3; package code clean from Task 4.
- Produces: `ruff check .` passes clean repo-wide (gate for Task 6's CI).

- [ ] **Step 1: Delete the five unused date locals in `tests/test_downloaders.py` (F841)**

These are pure documentation locals — the ISO date string passed to `is_date_recent(...)` already carries the meaning, and the `today` locals (which ARE used via `mock_datetime.now.return_value`) must stay. Delete exactly these five lines:

- line ~120: `yesterday = datetime(2026, 7, 1)`
- line ~131: `thirty_days_ago = datetime(2026, 6, 2)`
- line ~142: `thirty_one_days_ago = datetime(2026, 6, 1)`
- line ~153: `old_date = datetime(2026, 3, 15)`
- line ~171: `sixty_days_ago = datetime(2026, 5, 3)`

Example (`test_yesterday_is_recent`) — change:

```python
        today = datetime(2026, 7, 2)
        yesterday = datetime(2026, 7, 1)
        mock_datetime.now.return_value = today
```

to:

```python
        today = datetime(2026, 7, 2)
        mock_datetime.now.return_value = today
```

Apply the same pattern to the other four tests.

- [ ] **Step 2: Drop the unused `figures` binding in `tests/test_visualization.py` (F841)**

At ~line 1425, change:

```python
        figures = validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)
```

to:

```python
        validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)
```

- [ ] **Step 3: Verify the whole repo is clean**

Run: `uvx ruff check .`
Expected: `All checks passed!` If anything remains, fix it now using the same patterns (delete verified-dead code, wrap long lines); do not add `# noqa` outside the cases already specified.

- [ ] **Step 4: Run the full test suite**

Run: `pytest -q 2>&1 | tail -3`
Expected: `414 passed`, 0 failures.

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "chore: fix remaining lint findings in tests"
```

---

### Task 6: CI workflow + final verification

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: clean `ruff check .` from Task 5; `ruff` in the `dev` extra from Task 3.
- Produces: CI gate on every push to master and every PR.

- [ ] **Step 1: Create the workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [master]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - name: Install package with dev extras
        run: pip install -e .[dev]
      - name: Lint
        run: ruff check .
      - name: Test
        run: pytest
```

(Push trigger limited to master so PR branches don't run every job twice; PRs are covered by the `pull_request` trigger.)

- [ ] **Step 2: Validate the workflow YAML parses**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Run the exact CI commands locally**

Run:
```bash
uvx ruff check .
pytest -q 2>&1 | tail -3
```
Expected: `All checks passed!` and `414 passed`, 0 failures.

- [ ] **Step 4: Confirm the six spec §4 warnings are gone from the full run**

Run: `pytest 2>&1 | grep -E "FutureWarning|identical low and high|temporal_offset|No valid data" || echo "NO SPEC WARNINGS"`
Expected: `NO SPEC WARNINGS`

- [ ] **Step 5: Confirm mypy is no worse than the Task 1 baseline**

Run:
```bash
python -m mypy sar_validation/ 2>&1 | tail -1
cat /tmp/mypy_baseline.txt
```
Expected: error count ≤ baseline. (If `/tmp/mypy_baseline.txt` is gone — e.g. different machine/session — run mypy on the pre-change commit via `git stash` / `git worktree` to re-derive it, or accept the current output if it reports no errors.)

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: run ruff and pytest on push and pull requests"
```
