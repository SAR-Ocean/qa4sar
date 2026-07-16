# Wave SAR-Variable Fallback Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `filter_variable_pairs()` so wave statistics/plots are produced from the actual SAR columns present in the collocation dataset, instead of columns implied by the recipe's requested (but possibly unfulfilled) `swath_mode`.

**Architecture:** Replace the `swath_mode`-based `is_wv_only` branch in `sar_validation/core/_variable_map.py` with two independent selection rules: a single-winner fallback between `oswTotalHs` and `oswHs` (by column existence in `collocation_ds`), plus an *additive* inclusion of `owiSignificantWaveHeight` whenever that column exists and has at least one non-NaN value. Up to two SAR variables are then each crossed against the (unchanged) validation-variable candidate list.

**Tech Stack:** Python, xarray, pytest.

## Global Constraints

- The `oswTotalHs` → `oswHs` fallback triggers only on column *absence*, never on all-NaN values (per spec decision — "Column missing only").
- `owiSignificantWaveHeight` is *additive*, not a fallback: it is included alongside the primary variable only when its column exists **and** has at least one non-NaN value (per spec decision — run stats on both when it's not empty).
- The `oswTotalHs`/`oswHs` choice is still single-winner — never emit pairs for both in the same call.
- Up to two SAR variables total may be selected per call (primary + `owiSignificantWaveHeight`), each crossed against every available validation variable.
- The validation-variable candidate list (`VHM0, VAVH, VGHS, VAVH_UNFILTERED`) and its existence-based filtering are unchanged.
- New tests live in `tests/test_statistics.py` (per spec decision).

---

### Task 1: Fix SAR-variable fallback in `filter_variable_pairs` and add regression tests

**Files:**
- Modify: `sar_validation/core/_variable_map.py:117-153` (`filter_variable_pairs`)
- Test: `tests/test_statistics.py`

**Interfaces:**
- Consumes: `sar_validation.core._variable_map.filter_variable_pairs(recipe, collocation_ds) -> List[Tuple[str, str]]` (existing signature, unchanged).
- Produces: no new public names — the behavior change is internal to `filter_variable_pairs`.

- [ ] **Step 1: Write the failing/locking tests**

Add this class to `tests/test_statistics.py`. It needs `SARDataSpec` in addition to the `Recipe`/`RecipeConfig` already imported at the top of the file — update the import line:

```python
from sar_validation.core.recipe import Recipe, RecipeConfig, SARDataSpec
```

Then append the new test class at the end of the file:

```python
# ---------------------------------------------------------------------------
# filter_variable_pairs
# ---------------------------------------------------------------------------

def _waves_recipe(swath_mode):
    return Recipe(RecipeConfig(
        name="waves_test",
        variable="waves",
        sar_data=SARDataSpec(swath_mode=swath_mode),
    ))


class TestFilterVariablePairs:
    def test_mixed_mode_uses_oswTotalHs_when_present(self):
        """Regression test: recipe requests [WV, SM] but only WV scenes were
        actually downloaded, so the dataset only has sar_oswTotalHs. This is
        the exact scenario from recipes/waves_example.yaml that produced zero
        statistics before the fix."""
        recipe = _waves_recipe(["WV", "SM"])
        ds = xr.Dataset({
            "sar_oswTotalHs": ("collocation", [1.4, 1.5]),
            "val_VAVH":       ("collocation", [1.42, 1.48]),
            "val_source":     ("collocation", ["altimeter", "altimeter"]),
        })
        pairs = filter_variable_pairs(recipe, ds)
        assert pairs == [("oswTotalHs", "VAVH")]

    def test_falls_back_to_oswHs_when_oswTotalHs_absent(self):
        recipe = _waves_recipe(["WV", "SM"])
        ds = xr.Dataset({
            "sar_oswHs":  ("collocation", [1.4, 1.5]),
            "val_VAVH":   ("collocation", [1.42, 1.48]),
            "val_source": ("collocation", ["altimeter", "altimeter"]),
        })
        pairs = filter_variable_pairs(recipe, ds)
        assert pairs == [("oswHs", "VAVH")]

    def test_owiSignificantWaveHeight_excluded_when_all_nan(self):
        """owiSignificantWaveHeight must NOT be selected when its column is
        entirely NaN — this matches every real product observed so far."""
        recipe = _waves_recipe(["WV", "SM"])
        ds = xr.Dataset({
            "sar_oswTotalHs":               ("collocation", [1.4, 1.5]),
            "sar_owiSignificantWaveHeight": ("collocation", [np.nan, np.nan]),
            "val_VAVH":                     ("collocation", [1.42, 1.48]),
            "val_source":                   ("collocation", ["altimeter", "altimeter"]),
        })
        pairs = filter_variable_pairs(recipe, ds)
        assert pairs == [("oswTotalHs", "VAVH")]

    def test_owiSignificantWaveHeight_additive_when_it_has_data(self):
        """When owiSignificantWaveHeight has at least one real value, stats
        must be produced for BOTH it and the primary variable (oswTotalHs) —
        not just one or the other."""
        recipe = _waves_recipe(["WV", "SM"])
        ds = xr.Dataset({
            "sar_oswTotalHs":               ("collocation", [1.4, 1.5]),
            "sar_owiSignificantWaveHeight": ("collocation", [1.35, np.nan]),
            "val_VAVH":                     ("collocation", [1.42, 1.48]),
            "val_source":                   ("collocation", ["altimeter", "altimeter"]),
        })
        pairs = filter_variable_pairs(recipe, ds)
        assert set(pairs) == {("oswTotalHs", "VAVH"), ("owiSignificantWaveHeight", "VAVH")}

    def test_does_not_double_count_oswTotalHs_and_oswHs(self):
        """oswTotalHs must win outright over oswHs — oswHs must not also
        appear even though its column exists in the dataset."""
        recipe = _waves_recipe(["WV"])
        ds = xr.Dataset({
            "sar_oswTotalHs": ("collocation", [1.4, 1.5]),
            "sar_oswHs":      ("collocation", [1.3, 1.6]),
            "val_VAVH":       ("collocation", [1.42, 1.48]),
            "val_source":     ("collocation", ["altimeter", "altimeter"]),
        })
        pairs = filter_variable_pairs(recipe, ds)
        assert pairs == [("oswTotalHs", "VAVH")]

    def test_multiple_val_vars_cross_single_sar_winner(self):
        """Validation-side candidates are unaffected: every val_var that
        exists still produces its own pair against the one winning sar_var."""
        recipe = _waves_recipe(["WV", "SM"])
        ds = xr.Dataset({
            "sar_oswTotalHs": ("collocation", [1.4, 1.5]),
            "val_VAVH":       ("collocation", [1.42, 1.48]),
            "val_VHM0":       ("collocation", [1.40, 1.50]),
            "val_source":     ("collocation", ["altimeter", "buoy"]),
        })
        pairs = filter_variable_pairs(recipe, ds)
        assert set(pairs) == {("oswTotalHs", "VAVH"), ("oswTotalHs", "VHM0")}
```

Also add `filter_variable_pairs` to the existing import line near the top of
the file (it currently imports only `infer_variable_pairs`):

```python
from sar_validation.core._variable_map import infer_variable_pairs, filter_variable_pairs
```

- [ ] **Step 2: Run tests to verify current (pre-fix) pass/fail state**

Run: `pytest tests/test_statistics.py::TestFilterVariablePairs -v`

Expected, against the *current* (unfixed) code:
- `test_mixed_mode_uses_oswTotalHs_when_present` — **FAIL** (`pairs == []`, not `[("oswTotalHs", "VAVH")]`) — this is the reported bug.
- `test_falls_back_to_oswHs_when_oswTotalHs_absent` — PASS (mixed-mode branch already tries `oswHs`).
- `test_owiSignificantWaveHeight_excluded_when_all_nan` — **FAIL** (old code has no NaN check at all for `owiSignificantWaveHeight` and never tries `oswTotalHs` in mixed mode, so it wrongly returns `[("owiSignificantWaveHeight", "VAVH")]` instead of `[("oswTotalHs", "VAVH")]`).
- `test_owiSignificantWaveHeight_additive_when_it_has_data` — **FAIL** (old code returns only `[("owiSignificantWaveHeight", "VAVH")]`, missing the `oswTotalHs` pair entirely).
- `test_does_not_double_count_oswTotalHs_and_oswHs` — **FAIL** (`swath_mode=["WV"]` hits the WV-only branch, which tries both `oswTotalHs` and `oswHs` unconditionally, returning 2 pairs instead of 1).
- `test_multiple_val_vars_cross_single_sar_winner` — **FAIL** (same root cause as the first test: mixed mode never tries `oswTotalHs`).

Confirm the failures match this list before proceeding — if a different test fails for a different reason, stop and re-diagnose rather than continuing to Step 3.

- [ ] **Step 3: Implement the fallback chain**

In `sar_validation/core/_variable_map.py`, replace lines 117-141 (the body of
`filter_variable_pairs` from `variable = recipe.config.variable` through the
`pairs = base_pairs.copy()` else-branch) with:

```python
    variable = recipe.config.variable
    base_pairs = infer_variable_pairs(variable)

    # For waves: expand to all available wave validation parameters
    if variable == "waves":
        # Wave validation parameter candidates (in preferred order)
        wave_val_params = ["VHM0", "VAVH", "VGHS", "VAVH_UNFILTERED"]

        # Primary SAR wave-height variable: single-winner fallback driven by
        # which sar_<name> column actually exists in collocation_ds — NOT by
        # recipe.config.sar_data.swath_mode, since a recipe can request
        # multiple modes (e.g. [WV, SM]) while the downloader only ends up
        # returning scenes for one of them. Using the requested mode to pick
        # candidates caused real WV-only results to be silently dropped when
        # a mixed mode was requested (see
        # docs/superpowers/specs/2026-07-16-wave-sar-variable-fallback-design.md).
        primary_candidates = ["oswTotalHs", "oswHs"]
        primary_var = next(
            (v for v in primary_candidates if f"sar_{v}" in collocation_ds),
            None,
        )

        sar_vars = [primary_var] if primary_var is not None else []

        # owiSignificantWaveHeight is additive, not a fallback: it's a
        # genuinely different measurement (IW/EW grid product), so when it
        # actually carries data it gets its own statistics alongside the
        # primary variable rather than replacing it. In every real product
        # seen so far this column is either absent or entirely NaN, in which
        # case it must NOT be selected.
        owi_col = "sar_owiSignificantWaveHeight"
        if owi_col in collocation_ds and bool(collocation_ds[owi_col].notnull().any()):
            sar_vars.append("owiSignificantWaveHeight")

        # Generate all combinations of the selected SAR variable(s) and
        # available validation pairs
        pairs = []
        for sv in sar_vars:
            for val_param in wave_val_params:
                pairs.append((sv, val_param))
    else:
        pairs = base_pairs.copy()
```

Leave the rest of the function (the "Filter to only pairs where both
variables actually exist" block, lines 145-153 in the original) unchanged —
it still does the final existence check on the *validation* column, which is
still needed since the SAR-side selection above only checked the SAR
column(s).

Also delete the now-unused `swath_modes` and `is_wv_only` lines (originally
lines 119-120) and the `is_wv_only` docstring reference in bullet 2 of the
function's docstring (originally lines 96-99) — update it to:

```python
    For "waves" variable type, this function:
    1. Detects which wave validation parameters are available (VHM0, VAVH, VGHS, etc.)
    2. Picks the primary SAR wave variable by fallback (oswTotalHs, else
       oswHs, based on which column actually exists in collocation_ds), and
       additionally includes owiSignificantWaveHeight whenever that column
       exists and has at least one non-NaN value
    3. Filters to only pairs where both SAR and validation variables exist
```

- [ ] **Step 4: Run tests to verify they now pass**

Run: `pytest tests/test_statistics.py::TestFilterVariablePairs -v`
Expected: all 6 tests **PASS**.

- [ ] **Step 5: Run the full statistics and variable-map test files to check for regressions**

Run: `pytest tests/test_statistics.py -v`
Expected: all tests PASS (including the pre-existing `TestComputeStatistics` class, unaffected by this change).

- [ ] **Step 6: Commit**

```bash
git add sar_validation/core/_variable_map.py tests/test_statistics.py
git commit -m "$(cat <<'EOF'
fix: select SAR wave variable by dataset presence, not requested swath_mode

filter_variable_pairs picked oswHs/owiSignificantWaveHeight whenever a
recipe requested a non-WV-only swath_mode, even when every scene actually
downloaded was WV and only sar_oswTotalHs existed in the collocation
dataset. This silently dropped all wave statistics/plots for mixed-mode
recipes like recipes/waves_example.yaml. Now the primary SAR variable
(oswTotalHs, falling back to oswHs) is chosen by which sar_<name> column
is actually present, and owiSignificantWaveHeight is additionally
included whenever it carries real (non-NaN) data.
EOF
)"
```

---

### Task 2: Verify the fix against the real reported scenario

**Files:**
- None modified — this task re-runs the actual recipe that reported the bug to confirm statistics are now produced.

**Interfaces:**
- Consumes: `sar_validation.core.statistics.run_statistics` (existing, unchanged signature), the fixed `filter_variable_pairs` from Task 1.
- Produces: nothing new — verification only.

- [ ] **Step 1: Re-run statistics on the existing collocation output that reproduced the bug**

This project already has a saved `collocation_results.nc` from the exact run
that reported the bug — no need to re-download or re-collocate. Run:

```bash
python3 -c "
import xarray as xr
from sar_validation.core.recipe import Recipe
from sar_validation.core.statistics import run_statistics

recipe = Recipe.from_yaml('recipes/waves_example.yaml')
ds = xr.open_dataset('data/2026-06-11-000000-2026-06-14-000000_-80.00_-40.00_35.00_60.00/collocation_results.nc')
results = run_statistics(ds, recipe, base_dir='/tmp/wave_stats_verify')
print('Produced stats keys:', list(results.keys()))
assert results, 'Expected non-empty statistics results'
print('OK')
"
```

Expected output: `Produced stats keys: ['oswTotalHs_vs_VAVH', 'oswTotalHs_vs_VAVH_UNFILTERED']` (order may vary) followed by `OK`, with **no** `WARNING ... no statistics produced` log line.

- [ ] **Step 2: Clean up the verification output**

```bash
rm -rf /tmp/wave_stats_verify
```

No commit for this task — it's a verification-only step with no file changes.
