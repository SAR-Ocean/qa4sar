# Cartopy / report-rendering performance fix — Design

**Date:** 2026-07-15
**Status:** Approved (pending spec review)
**Type:** Performance fix

## 1. Problem

Generating a `validation_report()` with `--plot` has gotten slow enough to be
painful. Profiling from a prior session (`python -m cProfile` on
`sar-validate --recipe recipes/wind_example.yaml --collocation --collocation-log --stats --plot`,
a broad North Atlantic/Europe recipe, 8 validation sources, IW+EW mode, 5-hour
window) showed a total run of 74.5s, with `validation_report` alone taking
42.3s. None of that time is application logic — it's entirely matplotlib /
cartopy rendering internals, broken down as:

- `cartopy.mpl.gridliner._draw_gridliner`: **20.3s across 872 calls** — the
  single largest cost center, more than the coastline rendering itself.
- ~1.2 million shapely `intersects` calls (4.9s) from cartopy clipping
  coastline geometry to each subplot's extent.
- 453 matplotlib Agg canvas initializations (6.3s).
- `_land_coastline_features(scale="10m")` (`visualization.py:164`) called 73
  times — once per geographic subplot.

**Re-investigated for this design:** the earlier note speculated that the
10m-resolution Natural Earth shapefile was being re-parsed on every one of
those 73 calls. That's not the case — cartopy 0.25 (the version installed
here) already memoizes parsed shapefile geometries in a module-level
`_NATURAL_EARTH_GEOM_CACHE` dict inside `NaturalEarthFeature.geometries()`,
keyed by `(name, category, scale)`. So repeated calls to
`_land_coastline_features()` are not re-reading/re-parsing the shapefile —
there is no redundant-caching problem to fix there.

Two distinct, addressable cost centers remain:

**(A) The gridliner.** Triggered by `ax.gridlines(draw_labels=True, ...)`,
called once per geographic subplot in `plot_geographic`
(`visualization.py:614`) and once in `plot_collocation_diagnostics`'s single
overview plot (`visualization.py:1381`). `draw_labels=True` makes cartopy
compute label placement along the (possibly curved) projected axes boundary —
expensive machinery that isn't needed here because every plot in this
codebase uses `ccrs.PlateCarree()`, a rectangular projection where plain
axis-aligned tick labels are geometrically correct and much cheaper to
compute.

**(B) Every figure destined for the combined PDF is fully rendered twice.**
In `validation_report` (`visualization.py:1786-1928`), each figure (scatter,
geographic, statistics, residuals, offset) is saved with
`fig.savefig(png_path, dpi=150, bbox_inches="tight")` for its standalone PNG,
and the *same* Figure object is later drawn again with
`pdf.savefig(fig, dpi=150, bbox_inches="tight")` when assembling the combined
PDF. `bbox_inches="tight"` itself forces an internal extra draw pass (to
measure the tight bbox before the real render), so each figure is effectively
drawn ~4 times total (2 draws × 2 savefig calls). For the geographic figures —
the ones with expensive coastline/scatter/gridliner rendering — this
quadruples their true cost inside the 42.3s number, and plausibly explains
why the gridliner call count (872) is far higher than the raw subplot count
(~73). The codebase already has the fix pattern in place for one plot:
`plot_collocation_diagnostics` renders once to PNG, then reloads that PNG as
a raster to embed as a PDF page (see the comment at
`visualization.py:1888-1890`) specifically to avoid a second vector redraw —
this design generalizes that trick to the other five figure types.

**Not addressed by this fix, and why:**
- The shapely-intersects cost (4.9s) is inherent per-subplot extent clipping;
  each subplot has a different extent, so this work can't be shared across
  subplots without changing what's drawn.
- The 10m coastline resolution is an intentional prior tradeoff (visible
  misalignment with SAR swath edges at the default 110m) and is left
  unchanged.

## 2. Goal

Bring `validation_report` generation time on the profiled recipe from 42.3s
to **under 20s**, and the entire `--plot` routine to **under 30s** total,
via two complementary fixes (below), without changing the 10m coastline
resolution and without visually changing the report (grid lines,
degree-labeled axes, and PDF page content should look the same as today).

## 3. Fix

### Phase 1 — Replace gridliner labels with plain axis ticks

In both call sites, replace the combined "draw lines + compute labels" call
with two cheaper, decoupled steps:

1. `ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5)` — draws the
   visual grid lines only; skips gridliner's label-placement logic entirely.
2. Plain matplotlib tick labels via `cartopy.mpl.ticker.LongitudeFormatter`
   and `LatitudeFormatter`, applied through `ax.xaxis.set_major_formatter` /
   `ax.yaxis.set_major_formatter` (with `ax.set_xticks`/`set_yticks` at the
   gridline positions). This produces the same degree-formatted axis labels
   as before, without gridliner's curved-projection label placement.

Applied at:
- `plot_geographic`, per-scene subplot loop (`visualization.py:614`, inside
  `_build_figure`).
- `plot_collocation_diagnostics`, single overview plot
  (`visualization.py:1381`).

No changes to `_land_coastline_features`, the 10m scale, subplot count, or
any other rendering behavior.

### Phase 2 — Render each figure once, reuse the raster for the PDF page

Add one small helper (in `visualization.py`, near where `pdf_pages` is
assembled) that, given a Figure and a target PNG path (optional):

1. Renders the figure exactly once into an in-memory PNG buffer
   (`fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")`).
2. If a PNG path was given, writes the buffer's bytes to disk directly (no
   redraw).
3. Reads the buffer back as an image array and returns a lightweight
   throwaway Figure (`imshow` + `axis("off")`, matching the existing
   `plot_collocation_diagnostics`-embedding pattern) for use as the PDF page.
4. Closes the original (heavy) Figure immediately after step 1, rather than
   waiting until the end of the loop — freeing cartopy/Agg memory sooner for
   large multi-scene figures.

Replace the current repeated pattern at each of the five call sites in
`validation_report` (scatter, geographic, statistics, residuals,
scatter-by-offset, temporal-offset — `visualization.py:1787-1863`):

```python
if plots_dir:
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
figs.append(fig)
pdf_pages.append((title, fig))
```

with a call to the new helper, which returns the image-page Figure to append
to `pdf_pages` instead of the original heavy Figure. `figs.append(fig)` (the
list backing `all_figures`, `validation_report`'s return value to callers)
is untouched — callers still get the original, fully-populated Figure
objects exactly as before, since that assembly happens before the render
step regardless.

The existing `plot_collocation_diagnostics` embedding code
(`visualization.py:1888-1893`) is switched to use the same helper for
consistency, replacing its own manual `plt.imread(path)` +
`plt.figure()` + `imshow` block.

## 4. Testing / verification

- Re-run the same profiling command
  (`sar-validate --recipe recipes/wind_example.yaml --collocation --collocation-log --stats --plot`
  under `cProfile`, sorted by cumulative time) before/after each phase and
  confirm: `_draw_gridliner` cost drops sharply after Phase 1, total draw
  count / Agg canvas inits drop after Phase 2, and total
  `validation_report` time is under 20s (under 30s for the full `--plot`
  routine).
- Existing `tests/test_visualization.py` coverage of `plot_geographic`,
  `plot_collocation_diagnostics`, and `TestValidationReport` /
  `TestValidationReportIncludesDiagnostics` should pass unchanged: those
  tests check for PNG/PDF file existence and figure dict keys, not gridliner
  internals or Figure identity, so no test changes are expected.
- Visual spot-check: generate a report and confirm gridlines + axis degree
  labels still render correctly on both plot types, and that the PDF pages
  are visually identical to today's (same raster, same dpi).

## 5. Risks / rollback

Low risk — both changes are localized to rendering code, with no data or API
surface changes.

- Phase 1: if the tick-label approach produces incorrectly formatted or
  misplaced labels on some edge case (e.g. an extent crossing the
  antimeridian), it can be reverted to `draw_labels=True` with no other side
  effects.
- Phase 2: PDF pages become raster images (at dpi=150, matching today's PNG
  output) instead of re-rendered vector figures. This means the PDF is no
  longer vector-zoomable for those pages — an acceptable trade-off given the
  PNGs at the same dpi were already the primary output, but worth calling
  out explicitly since it's a small, real behavior change (not just a
  performance change).
