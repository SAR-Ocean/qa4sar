# `--plot` Step / Validation Report Improvements — Design

**Date:** 2026-07-16
**Status:** Approved design
**Module:** `sar_validation/core/visualization.py` (single file); `sar_validation/cli.py` gets a small message-text update.

---

## Overview

Four improvements to the `--plot` step, driven by concrete problems observed
in real generated reports (`data/2026-07-05-180000-2026-07-05-220000_-15.00_0.00_35.00_60.00/`
for wind, `data/2026-07-01-000000-2026-07-03-000000_-70.00_-20.00_0.00_40.00/`
for waves).

| # | Item | Scope | Function(s) affected |
|---|------|-------|-----------------------|
| 1 | Stop writing individual per-pair PNGs to `plots/` | All recipes | `validation_report` |
| 2 | Fix the buoy-spike residual histogram | All recipes | `plot_residuals` |
| 3 | Cyclic colormap for wind direction + layer-point transparency | Wind only | `plot_geographic`, `plot_collocation_diagnostics` |
| 4 | Larger/outlined matched markers + longitude tick overlap fix | Waves only (markers); all recipes (tick cap) | `plot_collocation_diagnostics`, `_set_lonlat_ticks` |

---

## Item 1 — Individual PNGs → PDF only

### Problem

`validation_report` (visualization.py:1807) calls `_finalize_figure_for_report`
for every plot (scatter, geographic, statistics, residuals, scatter-by-offset,
temporal-offset), which both writes a standalone PNG under `<out_dir>/plots/`
*and* embeds the same rendered image as a page in `validation_report.pdf`.
Every plot is already in the PDF, so the individual PNGs are redundant and
clutter `plots/` (17 files for a single wind recipe, e.g.
`owiWindSpeed_vs_WSPD_scatter.png`, `..._geographic_point_vs_layer.png`, etc.).

### Fix

In `validation_report`, stop passing a real `png_path` to
`_finalize_figure_for_report` for these six plot kinds — pass `None` instead.
The figure is still rendered exactly once and embedded as a PDF page; only the
disk write is skipped. Concretely:

- Remove the `if plots_dir: png_path = plots_dir / "..."` branching for
  scatter / geographic / statistics / residuals / scatter-by-offset /
  temporal-offset. Always call `_finalize_figure_for_report(fig, None)` when
  `base_dir is not None`, and use the raw `fig` directly (as today) when
  `out_dir is None` (the existing no-save / programmatic-use path used by
  some tests).
- The top-level `plots_dir = base_dir / "plots"; plots_dir.mkdir(...)` block
  at the top of `validation_report` is no longer needed by this function —
  `plot_collocation_diagnostics` creates and writes to `plots/` itself
  (visualization.py:1218-1219) when called at the end of `validation_report`,
  so `plots/collocation_diagnostics_<recipe>.png` continues to exist exactly
  as today. This is also written independently at the collocate step
  (`cli.py:624`), which is unaffected by this change.
- Drop the trailing `if plots_dir: logger.info("PNG plots saved to %s", ...)`
  line in `validation_report`, and update the corresponding `print(...)` in
  `cli.py` (`_generate_plots`, ~line 706) so it no longer implies every plot
  is saved as a PNG.

### Test

Update `tests/test_visualization.py` assertions that currently check
`(tmp_path / "plots" / f"{key}_scatter_by_offset.png").exists()` etc.
(lines ~981-982, ~1199-1200) to assert those files **do not** exist, while
`plots/collocation_diagnostics_<name>.png` still does. Verify the PDF still
contains one page per plot (page count unchanged).

---

## Item 2 — Residual histogram: small multiples instead of overlay

### Problem

`plot_residuals` (visualization.py:937) draws one shared axes with
`ax.hist(sub, bins=30, density=True)` called once per source, overlaid with
`alpha=0.5`. Because `bins=30` is an integer, matplotlib derives bin edges
independently **per source** from that source's own min/max range. A source
with very few points clustered in a narrow residual range (e.g. `buoy`, 2
points) gets a tiny bin width; `density=True` normalizes area-under-curve to
1 for that source alone, so its bar height explodes relative to sources with
a wider spread. Confirmed in
`data/2026-07-05-180000-2026-07-05-220000_-15.00_0.00_35.00_60.00/validation_report.pdf`:
buoy peaks at density≈150 while five other sources are flattened near 0 on
the same axes.

### Fix

Replace the single overlaid axes with a grid of subplots, one per
`val_source`, following the same `ncols` / `math.ceil(len(sources) / ncols)`
layout pattern already used in `plot_geographic`:

- Compute the pooled residual min/max across **all** sources up front and use
  it as a shared `range=` for every subplot's `ax.hist(...)`, so bars stay
  positionally comparable across subplots even though each has its own
  y-axis scale.
- Each subplot keeps `density=True`, its source's color (from
  `_source_color_map`), the vertical `axvline(0, ...)` zero-reference line,
  and a title with the source name and its `N` (e.g. `"buoy (N=2)"`).
- The overall figure keeps the existing top-level title
  (`f"Residuals: {sar_var} − {val_var} ..."`, including the "(wrapped to
  ±180°)" suffix for circular variables) as a `fig.suptitle(...)`.
- Unused grid cells (when source count doesn't evenly fill `nrows*ncols`) are
  hidden, matching `plot_geographic`'s pattern.
- `by_source=False` and `interactive=True` code paths are unchanged (the
  bug only affects the by-source static overlay).

### Test

Update `tests/test_visualization.py` residual tests to assert one subplot
(`fig.axes`) per distinct `val_source`, and that no single subplot's bar
height is inflated by a different source's narrow range (e.g. construct a
synthetic buoy-like tightly-clustered source alongside a wide-spread source
and assert both subplots' y-limits are computed independently).

---

## Item 3 — Wind: cyclic colormap + layer-point transparency

### Problem A — linear colormap on a circular variable

`plot_geographic` (visualization.py:454) defaults `cmap="viridis"` for both
the SAR background field and the validation scatter overlay, with color
limits from `np.nanpercentile(pooled, [2, 98])`. For `owiWindDirection` /
`WDIR` (already flagged as circular via `CIRCULAR_VAL_VARS`), this is wrong
twice over: `viridis` is not perceptually cyclic (0° and 359° render as
maximally different colors despite being adjacent headings), and
percentile-based color limits are meaningless for data that wraps at the
0°/360° seam.

### Fix A

In `plot_geographic`, when `val_var in CIRCULAR_VAL_VARS`:
- Use `cmap="twilight"` for **both** `cmap` and `effective_val_cmap` (the SAR
  direction field and the validation direction points share one cyclic
  palette, consistent with the existing "share a scale" design intent).
- Use a fixed `vmin=0, vmax=360` instead of the percentile-derived `vmin`/
  `vmax`, skipping the `pooled` percentile computation for this case.
- Non-circular variables (`WSPD`, `VHM0`, `rvlRadVel`, ...) are completely
  unaffected — this is a conditional branch keyed on `CIRCULAR_VAL_VARS`,
  not a change to the function's defaults.

### Problem B — dense scatterometer swath occludes sparser layer sources

`plot_collocation_diagnostics` (visualization.py:1541-1564, Tier 3 "matched
layer data") draws all layer-source categories (scatterometer, altimeter,
radiometer, hf_radar) at `alpha=1.0`, sorted so sparser sources draw on top
of denser ones by **total point count**. This ordering heuristic breaks down
when a source is geometrically dense along a narrow swath even with a lower
total count: confirmed in
`data/2026-07-05-180000-2026-07-05-220000_-15.00_0.00_35.00_60.00/plots/collocation_diagnostics_wind_europe_example_individual.png`,
where the scatterometer swath (164,749 matched points, drawn after/above
radiometer per the count sort) forms a solid opaque band that completely
hides the radiometer stars (515,691 points) directly underneath it.

### Fix B

Gated on `recipe.config.variable == "wind"`: change Tier 3's
`ax.scatter(..., alpha=1.0, ...)` to `alpha=0.65` for all layer-source
categories in that loop. Waves and currents recipes (which also have layer
sources — altimeter, hf_radar) keep `alpha=1.0`, unchanged. Tier 4 (in-situ)
is untouched in all cases; it doesn't exhibit this problem since in-situ
points are always sparse.

### Test

- `plot_geographic`: assert `ScalarMappable.cmap.name == "twilight"` and
  `norm.vmin == 0, norm.vmax == 360` when called with `val_var="WDIR"`;
  assert unchanged (`viridis`, percentile-derived) behavior for `val_var="WSPD"`.
- `plot_collocation_diagnostics`: assert Tier 3 scatter calls receive
  `alpha=0.65` when `recipe.config.variable == "wind"`, and `alpha=1.0` for a
  `"waves"`/`"currents"` recipe fixture.

---

## Item 4 — Waves: bigger outlined matched markers + longitude tick cap

### Problem A — matched points hard to distinguish from unmatched

For waves recipes, `plot_collocation_diagnostics` Tier 3/4 matched points use
`s=25, edgecolors="none"` — the same as every other recipe type — making
matches hard to pick out at a glance in dense wave validation maps.

### Fix A

Gated on `recipe.config.variable == "waves"`: change Tier 3 and Tier 4
matched-point `ax.scatter()` calls to `s=45, edgecolors="black",
linewidths=0.5` (vs. the current `s=25, edgecolors="none"`). Unmatched
(gray, faint) points and non-waves recipes are untouched.

### Problem B — overlapping longitude labels on narrow SAR-scene subplots

`_set_lonlat_ticks` (visualization.py:176) reads tick positions from the
gridliner's own locator (`gl.xlocator.tick_values(*xlim)`), which has no
awareness of subplot width or how narrow the underlying extent is. WV-mode
scenes are sparse imagettes with a very narrow longitude footprint (often
<1°) but a wide latitude range; the locator responds by picking many
closely-spaced, high-decimal-precision tick values, which pile up into an
unreadable jumble at the bottom of the subplot. Confirmed by rendering
`data/2026-07-01-000000-2026-07-03-000000_-70.00_-20.00_0.00_40.00/validation_report.pdf`
page 4: labels like `"49.5486V54\\535°W"` — actually 6+ overlapping
individual tick labels — appear under each narrow WV-scene subplot in
`plot_geographic`.

### Fix B

In `_set_lonlat_ticks`, cap both the longitude and latitude tick counts to a
small fixed maximum (`nbins=4`) instead of trusting the gridliner locator's
unbounded output — e.g. filter `gl.xlocator.tick_values(...)` /
`gl.ylocator.tick_values(...)` down via `matplotlib.ticker.MaxNLocator(nbins=4)`
applied to the already-computed tick candidates (or reconfigure the
locators themselves before reading `tick_values`, whichever keeps grid
lines and labels in sync per the existing docstring contract). This is a
change to the shared helper, so it applies to:
- `plot_geographic`'s per-SAR-scene subplots (visualization.py:817) — fixes
  the reported bug directly.
- `plot_collocation_diagnostics`'s single overview map
  (visualization.py:1460) — a wide-extent map (e.g. -70° to -20°) is also
  fine with ~4 ticks, so this is a mild decluttering side effect, not a
  regression.

Not waves-specific: any recipe with narrow-extent SAR scenes (not just WV
wave mode) could hit the same locator behavior, so the fix applies
universally via the shared helper.

### Test

- `plot_collocation_diagnostics`: assert Tier 3/4 matched scatter calls use
  `s=45, edgecolors="black"` for a `"waves"` recipe fixture, and the current
  `s=25, edgecolors="none"` for `"wind"`/`"currents"` fixtures.
- `_set_lonlat_ticks`: unit test with a narrow synthetic extent (e.g.
  0.05° wide) asserting `len(ax.get_xticks()) <= 4`; existing wide-extent
  tests continue to pass with a small, non-overlapping tick count.

---

## Settled defaults (non-decisions)

- **Item 1**: `plot_collocation_diagnostics`'s own PNG (written both at the
  collocate step and re-embedded during `--plot`) is the *only* file that
  continues to land in `plots/`. No other exceptions.
- **Item 2**: grid layout (not a single tall column) so the report page stays
  a reasonable height regardless of source count (up to 6 sources seen in
  practice: altimeter, buoy, mooring, radiometer, scatterometer,
  tidal_gauge).
- **Item 3**: the cyclic colormap and 0-360 color limits apply to the SAR
  field *and* the validation points together (one shared scale, per the
  function's existing "share a scale" design), not just one layer.
- **Item 3**: layer-point transparency (alpha=0.65) is applied to **all**
  layer-source categories when the recipe is wind, not scoped to
  scatterometer alone, so any future dense wind layer source is covered
  automatically.
- **Item 4**: the tick-count cap is a global change to the shared
  `_set_lonlat_ticks` helper, not conditioned on recipe variable — it fixes
  a locator behavior bug that happens to be most visible in wave WV-mode
  scenes today but isn't specific to them.

---

## Affected code locations

- `validation_report` — drop individual PNG writes (item 1).
- `plot_residuals` — small-multiples subplot grid (item 2).
- `plot_geographic` — cyclic colormap + fixed 0-360 limits for circular
  `val_var` (item 3A).
- `plot_collocation_diagnostics` — Tier 3 alpha for wind (item 3B); Tier 3/4
  marker size+edge for waves (item 4A).
- `_set_lonlat_ticks` — tick-count cap (item 4B).
- `cli.py` — update the "PNG plots saved to ..." message (item 1).
- `tests/test_visualization.py` — updated/new coverage for all four items.

## Out of scope

- Changes to `run_statistics` or the statistics page.
- Recoloring the SAR-footprint circles or the unmatched-gray tiers.
- Any change to `plot_scatter`, `plot_temporal_offset`, or `plot_statistics`
  beyond no longer being saved as standalone PNGs (item 1).
- The deferred Phase 3a Task 9 Step 4 integration test (separate, unrelated
  plan).

---

## Addendum (2026-07-16) — Item 5: currents HF radar point size in `plot_geographic`

### Problem

Confirmed in
`data/2026-06-01-000000-2026-06-01-235959_-130.00_-115.00_33.00_48.00/validation_report.pdf`
(geographic page): `plot_geographic`'s validation-point scatter uses a fixed
`point_size=40` (visualization.py:463) regardless of source density. HF
radar — currents' only layer-type (`layer_vs_layer`) source — is dense
enough (a near-continuous coverage grid) that at `s=40` the matched points
tile edge-to-edge and completely blanket the SAR `rvlRadVel` field
underneath, making the SAR scene invisible wherever HF radar coverage
exists.

### Fix

`validation_report` already computes `variable = recipe.config.variable`
(visualization.py:1858) before its per-pair plotting loop. Pass a smaller
`point_size` through to `plot_geographic` when `variable == "currents"`
(`15` vs. the existing default of `40`) — no change to `plot_geographic`
itself, which already accepts `point_size` as a parameter. Not scoped to
HF radar specifically (`plot_geographic` doesn't distinguish sources when
choosing `point_size`, only `validation_report` decides the value to pass),
which is an acceptable simplification since HF radar is currents' only
dense layer source today.

### Test

`validation_report`-level spy on `plot_geographic`, asserting
`point_size=15` is passed for a `"currents"` recipe and `point_size=40`
(the existing default) for other variable types.
