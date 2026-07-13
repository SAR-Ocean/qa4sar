# Distinguishing validation sources by marker shape — design

## Background

`validation_report.pdf` is hard to read when it contains multiple validation
sources (e.g. altimeter + radiometer). The problem is concentrated in
`plot_geographic()` (`sar_validation/core/visualization.py`): when a
validation variable is supplied, points are colored by their **measured
value** using the same colormap (e.g. viridis) as the SAR background field —
this lets SAR and validation values be compared directly by color. But the
legend for that plot draws its swatches with a *separate*, unrelated
qualitative color per source (built from `_source_color_map()`), so the
legend color never matches what's actually drawn on the map. The only way to
tell sources apart is to read the legend text.

`plot_scatter()` and `plot_collocation_diagnostics()` don't have this
problem — they already color points directly by source (no continuous
colormap involved) — but per the "consistent everywhere" decision below, they
get matching marker shapes too, for a single visual convention across the
whole report.

`plot_residuals()` (histogram) and `plot_statistics()` (bar chart) have no
per-point markers and are out of scope.

## Decisions

1. **Scope**: apply the same shape-per-source convention to every plot that
   draws per-observation markers — `plot_scatter`, `plot_geographic`,
   `plot_collocation_diagnostics`. `plot_residuals` and `plot_statistics` are
   unaffected (no markers to vary).
2. **Mechanism**: marker **shape** per source, not just a colored outline.
   Shape is meaningful in both plot styles used in this codebase — where
   color already encodes source (scatter, diagnostics) and where color is
   taken by a continuous value (geographic) — and remains distinguishable in
   grayscale / for colorblind readers, unlike color alone.
3. **Stable assignment**: source → (color, marker) is derived from a **fixed,
   alphabetically-sorted canonical list** of known source/platform types,
   not from whichever sources happen to be present in one particular plot
   call. This makes "altimeter" always the same shape *and* color everywhere
   in a report (and across separate report runs), and incidentally fixes a
   latent inconsistency where the same source could already get different
   colors in different plots depending on which other sources shared that
   call.

## Canonical source list

Reuse the two existing constants rather than introducing a new list:

- `LAYER_DATA_TYPES = {"scatterometer", "altimeter", "hf_radar", "radiometer"}`
  (`sar_validation/core/collocation.py`)
- `_INSITU_TYPES = {"mooring", "buoy", "drifter", "ferrybox", "tidal_gauge"}`
  (`sar_validation/core/orchestrator.py`)

Combined and alphabetically sorted → 9 known sources. Any source name
encountered that isn't in this set is appended deterministically (sorted,
after the known ones) so unrecognized/future source types still get a
usable, stable slot instead of erroring.

## New helper

Replace the color-only `_source_color_map()` with a combined style map, e.g.:

```python
_SOURCE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]  # 9, matches canonical list length
# _SOURCE_COLORS extended from 8 to 9 entries to pair 1:1 with the markers

def _source_style_map(sources) -> dict[str, tuple[str, str]]:
    """Map each source name to a stable (color, marker) pair.

    Order is fixed (canonical known types, alphabetically, then any extra
    names appended in sorted order) rather than depending on which sources
    are present in *this* call, so the same source always renders the same
    way across every plot and every report run.
    """
```

Call sites that need only color (none, after this change — all three
functions need marker too) are updated to unpack `(color, marker)`.

## Per-function changes

### `plot_scatter`
Existing per-source loop (`for src in sorted(sources): ax.scatter(...)`) gains
`marker=style[src][1]` alongside the existing `color=style[src][0]`.

### `plot_geographic`
- **Value-colormap branch** (points colored by measured value): currently one
  `ax.scatter()` call for all valid (non-NaN) points. Split into one
  `ax.scatter()` per `val_source` subgroup, each with its own marker, all
  sharing the same `cmap`/`norm`/`vmin`/`vmax` so the colorbar stays unified
  across sources.
- **NaN branch** (no retrieved value): stays exactly as-is — one gray,
  hatched circle group. "No data" isn't a source, so it keeps its own fixed,
  distinct symbol rather than participating in the per-source shape scheme.
- **Legend handles**: since the actual on-map fill color varies continuously
  with value in this branch, a solid-color legend swatch would still be a
  fiction. Switch legend markers to a neutral light-gray face + black edge,
  with **shape** as the sole discriminator and the label naming the source.
  This directly removes the reported color mismatch instead of trying to
  patch it.
- **Non-colormap branch** (`val_var` not supplied — points already colored
  directly by source): add the matching marker; color stays as today.

### `plot_collocation_diagnostics`
The per-source in-situ loop and the one-scatter-per-layer-category call both
gain `marker=style[...][1]`. Legends here are built automatically by
matplotlib from labeled artists and already show the correct fill color, so
no manual legend rework is needed — adding `marker=` to the `ax.scatter()`
calls is sufficient.

### `plot_residuals`, `plot_statistics`
No change.

## Testing / verification

- Regenerate the example report at
  `data/2026-07-10-190000-2026-07-10-200000_-25.00_-5.00_30.00_60.00/` and
  visually confirm: the geographic-plot legend is legible (no color
  mismatch), shapes are distinguishable at the plotted marker size, and the
  scatter/diagnostics plots still read correctly with shapes added.
- Existing smoke tests in `tests/test_visualization.py` only assert
  `Figure is not None` / return type — expected to pass unchanged.
- Add one small test asserting two different `val_source` values in
  `plot_scatter` produce scatter artists with different `marker` values.
