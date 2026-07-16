# Wave SAR-variable fallback fix — design

## Problem

Running `recipes/waves_example.yaml` finds collocated points (e.g. 8 in a
recent test run) but produces zero statistics:

```
Step 5a: Computing validation statistics…
WARNING sar_validation.core.statistics — run_statistics: no statistics produced (check variable names).
```

### Root cause

`filter_variable_pairs()` in
[`sar_validation/core/_variable_map.py`](../../../sar_validation/core/_variable_map.py)
selects candidate SAR wave-height variable names based on the recipe's
*requested* `sar_data.swath_mode`, not on what actually got downloaded:

```python
is_wv_only = set(swath_modes) == {"WV"}
if is_wv_only:
    sar_vars = ["oswTotalHs", "oswHs"]
else:
    sar_vars = ["oswHs", "owiSignificantWaveHeight"]
```

`waves_example.yaml` requests `swath_mode: [WV, SM]`. Every scene the
downloader actually returned for the query window was a WV product
(`S1A_WV_OCN__…`, `S1D_WV_OCN__…`), so the collocation dataset only contains
`sar_oswTotalHs`. Because the recipe's requested mode is `[WV, SM]` (not
`{WV}` exactly), `is_wv_only` is `False`, so the mixed-mode branch is taken —
which never tries `oswTotalHs`. Every `(sar_var, val_var)` combination then
fails the "column exists in `collocation_ds`" check, `filter_variable_pairs`
returns an empty list, and no statistics are computed.

The validation-side candidate list (`VHM0, VAVH, VGHS, VAVH_UNFILTERED`,
`_variable_map.py:125`) was already correct and is not the cause — this was
verified against the actual collocation output, which has `val_VAVH` and
`val_VAVH_UNFILTERED` (from the altimeter source) but no `VHM0`/`VGHS` (no
source in that run provided them).

`filter_variable_pairs` is the single source of truth for wave variable
pairs and is used by both
[`statistics.py:271`](../../../sar_validation/core/statistics.py#L271) and
[`visualization.py:1860`](../../../sar_validation/core/visualization.py#L1860),
so this bug can also suppress wave plots for the same scenario.

## Fix

Replace the `swath_mode`-based branch with a fixed, existence-driven
fallback chain that no longer depends on the recipe's requested mode:

```python
sar_var_candidates = ["oswTotalHs", "oswHs", "owiSignificantWaveHeight"]
```

Selection rules:

- Try `oswTotalHs` first. Only fall back to `oswHs` if `sar_oswTotalHs` is
  **not present as a column** in `collocation_ds` (existence check only —
  not an all-NaN check). This mirrors `datatree_converter`'s own internal
  fallback from `oswTotalHs` to `oswHs` partitions when building the
  dataset, and matches the existing existence-based filtering style used
  everywhere else in this function.
- If `oswHs` is also absent, fall back to `owiSignificantWaveHeight`.
- This is a **single-winner fallback**, not "try every candidate that
  exists": once a candidate is found to exist, later candidates in the
  chain are not also tried. This differs from the current WV-only branch,
  which tries both `oswTotalHs` and `oswHs` unconditionally when both
  happen to be present. The new behavior is what was explicitly requested:
  prefer `oswTotalHs`, only use `oswHs` when `oswTotalHs` is unavailable.
- `owiSignificantWaveHeight` is kept in the chain as a last-resort fallback
  even though it is reportedly always NaN in current products — it costs
  nothing when absent/all-NaN (existing existence/dropna filters already
  discard it) and avoids silently losing coverage if it's populated by a
  future product version.
- `recipe.config.sar_data.swath_mode` is no longer read for this purpose;
  the `is_wv_only` variable is deleted from `filter_variable_pairs`.

The validation-variable candidate list (`VHM0, VAVH, VGHS,
VAVH_UNFILTERED`) is unchanged. Each one that exists in `collocation_ds` is
still crossed against the single winning `sar_var` and produces its own
statistics entry (e.g. `oswTotalHs_vs_VAVH`, `oswTotalHs_vs_VHM0` if both
are present).

## Testing

No tests currently exist for `filter_variable_pairs` (verified — none found
under `tests/`). Add a `TestFilterVariablePairs` class to
`tests/test_statistics.py`, using synthetic `Recipe`/`collocation_ds`
fixtures consistent with the file's existing style, covering:

1. **Regression case (the reported bug):** recipe `swath_mode=[WV, SM]`,
   dataset has only `sar_oswTotalHs` and `val_VAVH` → pair
   `("oswTotalHs", "VAVH")` is returned. This must fail before the fix and
   pass after.
2. **Fallback to oswHs:** `sar_oswTotalHs` column absent, `sar_oswHs`
   present → `oswHs` is selected.
3. **Fallback to owiSignificantWaveHeight:** neither `oswTotalHs` nor
   `oswHs` present, only `sar_owiSignificantWaveHeight` → that is selected.
4. **No double-counting:** when `sar_oswTotalHs` exists, `oswHs` is *not*
   also tried, even if a `sar_oswHs` column happens to also be present in
   the dataset.

## Out of scope

- All-NaN-triggered fallback (only column-existence triggers fallback, per
  explicit decision).
- Changes to the validation-variable (`VHM0`/`VAVH`/`VGHS`/
  `VAVH_UNFILTERED`) candidate list — already correct.
- Removing `owiSignificantWaveHeight` from the candidate chain.
