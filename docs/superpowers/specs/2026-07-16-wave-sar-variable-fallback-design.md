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

Replace the `swath_mode`-based branch with a rule that no longer depends on
the recipe's requested mode, and that treats `owiSignificantWaveHeight` as
an *additive* second SAR variable rather than a last-resort fallback:

1. **Primary variable** — a single-winner fallback between the two
   `oswHs`-family variables:
   - Try `oswTotalHs` first. Only fall back to `oswHs` if `sar_oswTotalHs`
     is **not present as a column** in `collocation_ds` (existence check
     only — not an all-NaN check). This mirrors `datatree_converter`'s own
     internal fallback from `oswTotalHs` to `oswHs` partitions when
     building the dataset.
   - This part is still single-winner: if both `sar_oswTotalHs` and
     `sar_oswHs` exist, only `oswTotalHs` is used, never both.
2. **Additive variable** — `owiSignificantWaveHeight` is included
   *alongside* the primary variable, but only if it actually carries data:
   `sar_owiSignificantWaveHeight` exists as a column **and** has at least
   one non-NaN value in `collocation_ds`. If the column is absent, or
   present but entirely NaN (the situation observed in every product seen
   so far), it is left out and only the primary variable is used.

So up to **two** SAR variables can be selected per run: the primary
(`oswTotalHs` or `oswHs`) always, plus `owiSignificantWaveHeight` whenever
it has real data. This is a deliberate change from a strict single-winner
model — `owiSignificantWaveHeight` is a genuinely different measurement
(IW/EW grid product) rather than a redundant alternative to `oswTotalHs`,
so when both carry data they should both get their own statistics/plots.

- `recipe.config.sar_data.swath_mode` is no longer read for this purpose;
  the `is_wv_only` variable is deleted from `filter_variable_pairs`.

The validation-variable candidate list (`VHM0, VAVH, VGHS,
VAVH_UNFILTERED`) is unchanged. Each one that exists in `collocation_ds` is
still crossed against every selected `sar_var` (one or two) and produces
its own statistics entry (e.g. `oswTotalHs_vs_VAVH`,
`owiSignificantWaveHeight_vs_VAVH` if both SAR variables have data and
`VAVH` is present).

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
   present → `oswHs` is selected as the primary variable.
3. **owiSignificantWaveHeight all-NaN is excluded:** `sar_oswTotalHs`
   present, `sar_owiSignificantWaveHeight` present but entirely NaN → only
   `oswTotalHs` is selected (matches every real product observed so far).
4. **owiSignificantWaveHeight with real data is additive:**
   `sar_oswTotalHs` present, `sar_owiSignificantWaveHeight` present with at
   least one non-NaN value → both `oswTotalHs` and
   `owiSignificantWaveHeight` are selected, each crossed against the
   available validation variables.
5. **No double-counting between oswTotalHs and oswHs:** when
   `sar_oswTotalHs` exists, `oswHs` is *not* also tried, even if a
   `sar_oswHs` column happens to also be present in the dataset.

## Out of scope

- All-NaN-triggered fallback between `oswTotalHs` and `oswHs` (only
  column-existence triggers that fallback, per explicit decision). The
  all-NaN check applies only to deciding whether
  `owiSignificantWaveHeight` is additive.
- Changes to the validation-variable (`VHM0`/`VAVH`/`VGHS`/
  `VAVH_UNFILTERED`) candidate list — already correct.
