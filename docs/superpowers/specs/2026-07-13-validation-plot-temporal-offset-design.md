# Surfacing temporal collocation offset in validation_report.pdf — design

## Background

Collocation tolerances allow up to `time_tolerance_minutes` (e.g. 180 min /
±3h for some layer-vs-layer sources — see `LayerLayerCollocation` in
`collocation.py`) between a SAR acquisition and a matched validation
observation. A pair matched a few minutes apart should agree much better
than one matched close to the tolerance limit, but nothing in
`validation_report.pdf` currently shows this — so a lower-than-expected
correlation coefficient is hard to explain from the report alone.

No pipeline change is needed: every `CollocatedPoint` already carries
`temporal_distance_minutes` (absolute minutes between SAR and validation
timestamps, computed in `collocation.py`, e.g. line 589:
`temporal_dist = abs((v_time - s_time).total_seconds() / 60.0)`), and it is
already written straight through into `collocation_ds` by
`DataTreeConverter.from_collocations()` (`datatree_converter.py:500`). This is
a pure `sar_validation/core/visualization.py` addition.

## Decisions

1. **Two additions to the report**, per the "both" scope decision:
   - Recolor a variant of the existing SAR-vs-validation scatter by temporal
     offset instead of by source.
   - A new dedicated plot: `|residual|` vs. temporal offset.
2. **New plot's y-axis**: absolute residual `|SAR − validation|` (not signed)
   — directly reads as "does the error magnitude grow with the time gap,"
   matching the motivating question, at the cost of not showing bias
   direction (already covered by the existing residuals histogram).
3. **No binned trend line** — raw per-observation scatter only, consistent
   with the existing `plot_scatter` / `plot_residuals` style (no binning
   logic anywhere else in this module). Can be revisited later if the raw
   scatter turns out too noisy to read.
4. **Source stays visible via marker shape** in both additions, reusing the
   `_source_style_map()` (color, marker) mapping from the companion
   marker-shapes design
   (`2026-07-13-validation-plot-source-markers-design.md`) — different
   validation sources can have very different tolerances (e.g. a mooring's
   30 min vs. a scatterometer's 180 min), so keeping sources visually
   distinguishable prevents misreading a source-driven pattern as a
   pure time-offset effect.

## Change 1 — `plot_scatter`: new `color_by` parameter

```python
def plot_scatter(
    collocation_ds, sar_var, val_var, *,
    by_source: bool = True,
    color_by: str = "source",   # "source" | "temporal_offset"
    interactive: bool = False,
    ax=None,
):
```

- `color_by="source"` (default): unchanged behavior — color and marker both
  vary by source (marker addition comes from the companion design).
- `color_by="temporal_offset"`: points colored by `temporal_distance_minutes`
  (continuous colormap + colorbar labeled "Temporal offset (min)"); marker
  still varies by source (from the style map) so sources stay identifiable.
  Legend swatches for source use a neutral gray face + black edge (shape is
  the only discriminator), mirroring the fix already designed for
  `plot_geographic`'s value-colormap branch — same reasoning: color is taken
  by a continuous quantity, so a solid per-source swatch would misrepresent
  what's on the plot.
- `by_source=False` still means "no source distinction at all" (single
  marker/style), independent of `color_by`.

`validation_report()` calls `plot_scatter` twice per `(sar_var, val_var)`
pair: once as today, and once with `color_by="temporal_offset"`, saved as
`{key}_scatter_by_offset.png` and added as its own PDF page titled
`"{sar_var} vs {val_var} — scatter (colored by temporal offset)"`.

## Change 2 — new `plot_temporal_offset()` function

```python
def plot_temporal_offset(
    collocation_ds, sar_var, val_var, *,
    by_source: bool = True,
    interactive: bool = False,
    ax=None,
):
    """Scatter of |SAR - validation| residual vs. temporal collocation offset."""
```

- Mirrors `plot_residuals`' structure: same missing-column check, same
  `circular_diff_deg` handling for circular validation variables (wind
  direction etc.), same `_deduplicate_obs()` call before plotting (one point
  per observation, not per matched SAR pixel).
- x-axis: `temporal_distance_minutes` (minutes). y-axis: `|residual|`.
- Colored and marker-shaped by source (from the shared style map) when
  `by_source=True`, single color otherwise — same as `plot_residuals` today.
- Annotation box (same visual style as `plot_scatter`'s N/Bias/RMSE/r box):
  `N=<count>` and `r=<Pearson correlation between temporal_distance_minutes
  and |residual|>` — directly quantifies "does the error grow with time
  offset" instead of leaving it to visual inspection alone.
- Added to `validation_report()` as one more page per pair:
  `{key}_temporal_offset.png`, PDF title
  `"{sar_var} vs {val_var} — residual vs. temporal offset"`.

## Out of scope

- No vertical reference line at the configured `time_tolerance_minutes` —
  that would require threading `recipe` into a function that (like
  `plot_scatter`/`plot_residuals`) currently only takes `collocation_ds` +
  variable names. Can be added later as an optional parameter if useful.
- No binning/trend line (see decision 3).
- No changes to `collocation.py` or `datatree_converter.py` — the data is
  already there.

## Testing / verification

- Regenerate the example report and visually confirm: the offset-colored
  scatter shows a colorbar and legible source markers, and the new
  residual-vs-offset plot renders with a sensible N/r annotation.
- Add a smoke test in `tests/test_visualization.py` for
  `plot_temporal_offset` (returns a Figure) and for
  `plot_scatter(..., color_by="temporal_offset")` (returns a Figure, has a
  colorbar), following the existing smoke-test style in that file.
