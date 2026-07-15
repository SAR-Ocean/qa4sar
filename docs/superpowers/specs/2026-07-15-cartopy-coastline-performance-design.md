# Cartopy gridliner performance fix — Design

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

That leaves the gridliner as the dominant, addressable cost. It's triggered
by `ax.gridlines(draw_labels=True, ...)`, called once per geographic subplot
in `plot_geographic` (`visualization.py:614`) and once in
`plot_collocation_diagnostics`'s single overview plot
(`visualization.py:1381`). `draw_labels=True` makes cartopy compute label
placement along the (possibly curved) projected axes boundary — expensive
machinery that isn't needed here because every plot in this codebase uses
`ccrs.PlateCarree()`, a rectangular projection where plain axis-aligned tick
labels are geometrically correct and much cheaper to compute.

The shapely-intersects cost (4.9s) is inherent per-subplot extent clipping
and not addressed by this fix (each subplot has a different extent, so this
work can't be shared across subplots without changing what's drawn). The
10m coastline resolution is an intentional prior tradeoff (visible
misalignment with SAR swath edges at the default 110m) and is left
unchanged.

## 2. Goal

Bring `validation_report` generation time on the profiled recipe from 42.3s
to **under 30s**, primarily by removing the gridliner label-placement cost,
without changing the 10m coastline resolution and without visually changing
the report (grid lines and degree-labeled axes should look the same).

## 3. Fix

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

## 4. Testing / verification

- Re-run the same profiling command
  (`sar-validate --recipe recipes/wind_example.yaml --collocation --collocation-log --stats --plot`
  under `cProfile`, sorted by cumulative time) before/after the change and
  confirm: `_draw_gridliner` cost drops sharply, and total
  `validation_report` time is under 30s.
- Existing `tests/test_visualization.py` coverage of `plot_geographic` and
  `plot_collocation_diagnostics` should pass unchanged — none of those tests
  assert on gridliner internals (`gl.top_labels`, `gl.right_labels`, etc.),
  only on figure/axes structure and drawn artists.
- Visual spot-check: generate a report and confirm gridlines + axis degree
  labels still render correctly on both plot types.

## 5. Risks / rollback

Low risk — this is a localized rendering change in two functions, with no
data or API surface changes. If the tick-label approach produces
incorrectly formatted or misplaced labels on some edge case (e.g. an extent
crossing the antimeridian), the change can be reverted to
`draw_labels=True` with no other side effects.
