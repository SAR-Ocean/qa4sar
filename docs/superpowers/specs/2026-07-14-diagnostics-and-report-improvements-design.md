# Collocation Diagnostics & Validation Report Improvements — Design

**Date:** 2026-07-14
**Status:** Approved design
**Module:** `sar_validation/core/visualization.py` (single file)

---

## Overview

Six improvements to the collocation diagnostics plot and the combined
validation report. All work is confined to
`sar_validation/core/visualization.py`. The first item finishes previously
started, still-uncommitted work; the remaining five are new behaviour.

| # | Item | Function affected |
|---|------|-------------------|
| 0 | Land the two hanging (uncommitted) edits | `plot_collocation_diagnostics` (already on disk) |
| 1 | Make matched layer data (altimeter) visible | `plot_collocation_diagnostics` — Tier 3 |
| 2 | Wind-direction plots: directional sources only | `validation_report` |
| 3 | Diagnostics plot on the first report page | `validation_report` |
| 4 | Only include SAR images with matched points | `plot_geographic` (+ caller) |

---

## Item 0 — Land the hanging changes (prerequisite)

Two edits are already applied on disk (confirmed via `git diff`) but were never
committed because a tool outage blocked test verification in the prior session
(see `.superpowers/sdd/progress.md`):

1. `_SOURCE_COLORS`: replaced the near-gray `#7f7f7f` with `#17becf` so no
   source color collides with the unmatched-point gray `#808080`.
2. `plot_collocation_diagnostics` legend: folded the matched/unmatched
   explanation into a legend entry instead of a floating text box that could
   overlap the legend when labels were long.

**Plan for this item:**
1. Run `python -m pytest tests/test_visualization.py -v` and
   `python -m pytest tests/ -q` — expect all pass (both edits are
   mechanical: a palette hex swap and a legend-handle restructure).
2. Regenerate a real report and visually confirm (legend no longer overlaps;
   radiometer matched points are teal `#17becf`, distinct from unmatched gray).
3. Commit these two edits **on their own**, before any new work, so the new
   items stack on a clean, verified base.

No new tests are required for item 0 (the edits carry no new logic); the new
items below each add their own coverage.

---

## Item 1 — Make matched layer data visible in the diagnostics plot

### Problem

In the waves example, only 7 of 17,115 altimeter observations matched. Those
7 matched points are rendered by Tier 3 with `s=20`, `alpha=0.6`,
`edgecolors="none"`, and a fill color (`#1f77b4`, blue) that collides with the
blue SAR-footprint circles. They are visually lost among thousands of gray
unmatched-track points and the footprint circles.

### Fix

Change **only** the Tier 3 (matched layer) `ax.scatter` styling in
`plot_collocation_diagnostics`:

| Property | Before | After |
|----------|--------|-------|
| `s` (size) | 20 | 70 |
| `alpha` | 0.6 | 1.0 |
| `edgecolors` | `"none"` | `"black"` |
| `linewidths` | (unset) | 0.7 |
| `zorder` | 5 | 5 (unchanged) |

Rationale: a bold black edge + larger size + full opacity makes a handful of
matched points pop regardless of fill color, without needing to recolor the SAR
footprints. The z-order is unchanged, so matched in-situ points (Tier 4) still
draw on top and unmatched gray points (Tiers 1–2) still draw underneath.

Tiers 1, 2, and 4 are untouched.

### Test

Extend `tests/test_visualization.py`: assert that the matched-layer scatter call
is invoked with `edgecolors="black"` and an enlarged marker size (e.g. via a
scatter spy / captured kwargs), so the visibility contract is locked in.

---

## Item 2 — Wind-direction plots: directional sources only (data-driven)

### Problem

Altimeter and radiometer measure wind **speed** but not **direction**. Their
`val_WDIR` values are NaN. `plot_scatter` already drops NaN pairs, but
`plot_geographic` renders NaN validation values as gray hatched "No data (NaN)"
markers, so non-directional sources still clutter the wind-direction maps.

### Fix

In `validation_report`, inside the per-pair loop, when the validation variable
is circular (`val_var in CIRCULAR_VAL_VARS`, i.e. `WDIR`), pass a **filtered**
collocation dataset to that pair's plots. The filter drops every source whose
`val_<val_var>` column is entirely NaN across the dataset:

```python
def _drop_nondirectional_sources(coll_ds, val_var):
    """For a circular val_var, drop sources with no finite val_<var> value.
    Non-directional instruments (altimeter, radiometer) have all-NaN
    direction and would otherwise render as 'No data' clutter."""
    val_col = f"val_{val_var}"
    if val_col not in coll_ds or "val_source" not in coll_ds:
        return coll_ds
    finite = np.isfinite(coll_ds[val_col].values)
    sources = np.asarray(coll_ds["val_source"].values)
    keep_sources = {s for s in np.unique(sources)
                    if finite[sources == s].any()}
    mask = np.array([s in keep_sources for s in sources])
    return coll_ds.isel(collocation=mask)
```

Apply this once per pair (only when `val_var in CIRCULAR_VAL_VARS`) and use the
result as the `collocation_ds` argument for all four plot calls (scatter,
geographic, residuals, temporal-offset) for that pair. `WSPD` pairs are
untouched, so altimeter/radiometer remain in the wind-speed comparison.

Self-maintaining: any future source that lacks direction is excluded
automatically; no hardcoded instrument list.

### Test

Add a `validation_report` test with a synthetic collocation dataset containing a
`WSPD` + `WDIR` mix where one source (e.g. `altimeter`) has all-NaN `val_WDIR`.
Assert the source is absent from the WDIR-pair figures but present in the
WSPD-pair figures.

---

## Item 3 — Collocation-diagnostics plot on the first report page

### Problem

The diagnostics page is currently appended to the **end** of `pdf_pages`, so it
lands after every per-variable plot.

### Fix

Build the diagnostics figure and insert it at the **front** of `pdf_pages`
(index 0), so it becomes the first content page after the cover. The simplest
implementation: keep generating the figure where it is convenient, then
`pdf_pages.insert(0, (title, fig_diag))` rather than `pdf_pages.append(...)`;
alternatively generate it before the per-pair loop. Either way:

- The cover page is still written first (it is added separately when the
  `PdfPages` context opens).
- Order becomes: cover → **diagnostics** → per-variable plots.
- The PNG output path and standalone-call behaviour are unchanged.
- `fig_diag` close/cleanup handling is preserved.

### Test

Assert the first non-cover entry of `pdf_pages` is the diagnostics page (title
starts with "Collocation diagnostics").

---

## Item 4 — Only include SAR images with matched points (union across pairs)

### Problem

`plot_geographic` draws one subplot per SAR scene unconditionally
(`for scene_name in scene_names`), including scenes with zero matched points —
inflating the report with empty maps.

### Fix

Restrict the scene grid to the **union across all variable pairs** of scenes
that have ≥1 matched point:

1. In `validation_report`, before generating geographic figures, compute the set
   of scene names that appear (via `sar_scene_name`) in the collocation dataset
   with ≥1 matched collocation, across all pairs. In practice this is
   `set(collocation_ds["sar_scene_name"].values)` restricted to scenes actually
   present in the collocations (a scene with no collocations never appears).
2. Pass this allowlist into `plot_geographic` (new optional parameter, e.g.
   `scenes: Optional[Sequence[str]] = None`; when provided, `scene_names` is
   filtered to it, preserving order).
3. Every geographic figure uses the same scene set, so the grid is consistent
   across figures. A scene that matched only in a different pair may still show
   an empty subplot in a given figure, but scenes that never match anything are
   dropped entirely.

Edge case: if the allowlist is empty (no scene has any match), fall back to the
current behaviour (all scenes) rather than producing an empty figure.

### Test

Add a `plot_geographic` test with two scenes where only one has collocations;
assert that passing the computed allowlist renders a single subplot for the
matched scene. Add a `validation_report`-level assertion that the allowlist is
derived from `sar_scene_name`.

---

## Settled defaults (non-decisions)

- **Statistics / `N`**: unchanged. `run_statistics` already excludes NaN pairs
  upstream, so the WDIR statistics page naturally reflects directional-only
  counts; no change needed here.
- **Tier order in item 1**: matched in-situ points stay on top; only the matched
  **layer** tier is made bolder.
- **New spec**: this document is separate from the earlier completed
  `2026-07-14-collocation-diagnostics-refinement-design.md`.

---

## Affected code locations

- `plot_collocation_diagnostics` — Tier 3 scatter styling (item 1);
  already-applied palette + legend edits (item 0).
- `validation_report` — per-pair circular-source filter (item 2); diagnostics
  page ordering (item 3); scene-allowlist computation + pass-through (item 4).
- `plot_geographic` — new optional `scenes` allowlist parameter (item 4).
- `tests/test_visualization.py` — new coverage for items 1, 2, 3, 4.

## Out of scope

- Recoloring SAR-footprint circles or the unmatched-gray tiers.
- Changes to `run_statistics` or the statistics page.
- The deferred Phase 3a Task 9 Step 4 integration test (separate plan).
