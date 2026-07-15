# Collocation Diagnostics Plot Refinement Design

**Date:** 2026-07-14  
**Status:** Design phase  
**Context:** Refinement of `plot_collocation_diagnostics()` to improve visual clarity and reduce clutter during an existing implementation.

---

## Overview

The collocation diagnostics plot currently marks matched vs. unmatched data using different visual properties. This refinement ensures:
1. **Consistent marker shapes per source** — matched and unmatched points for the same data source use identical marker shapes (only color/alpha changes)
2. **Reduced visual clutter** — unmatched observations rendered semi-transparent and gray to de-emphasize them
3. **Clear layering** — in-situ data always visible on top of layer/satellite data
4. **Explicit matched/unmatched distinction** — conveyed via color (source-colored = matched, gray = unmatched) and transparency (opaque = matched, faint = unmatched)

---

## Visual Design

### Color & Transparency Encoding

| Category | Color | Alpha | Purpose |
|----------|-------|-------|---------|
| Matched layer data (e.g. altimeter) | Source color (e.g. #1f77b4) | 0.6 | Primary finding, fully opaque |
| Matched in-situ data (e.g. mooring) | Source color | 0.7 | Primary finding, fully opaque |
| Unmatched layer data | Gray (#808080) | 0.3 | Background noise, de-emphasized |
| Unmatched in-situ data | Gray (#808080) | 0.3 | Background noise, de-emphasized |

### Marker Shapes

Per-source markers (from `_source_style_map`):
- Altimeter: circle (`o`)
- Radiometer: square (`s`)
- Scatterometer: triangle (`^`)
- Mooring: diamond (`D`)
- Buoy: inverted triangle (`v`)
- etc.

**Key principle:** A given data source always uses the same marker shape, regardless of matched/unmatched status. Only the color and alpha change to indicate match status.

### Z-Order (Rendering Sequence)

Points are drawn in this order (first drawn = behind):

1. **Unmatched layer data** (e.g. altimeter points with no collocated validation)
   - Gray, alpha=0.3, z-order=2
   
2. **Unmatched in-situ data** (e.g. mooring points with no collocated SAR)
   - Gray, alpha=0.3, z-order=3
   
3. **Matched layer data** (e.g. altimeter collocated with validation)
   - Source color, alpha=0.6, z-order=4
   
4. **Matched in-situ data** (e.g. mooring collocated with SAR)
   - Source color, alpha=0.7, z-order=5

This ensures:
- Matched points are never hidden
- In-situ observations are always readable on top of layer data
- Unmatched points provide context without overwhelming the plot

---

## Implementation Strategy

### Changes to `plot_collocation_diagnostics()` in `sar_validation/core/visualization.py`

The function already loops over data categories (layer types and in-situ groups). The refinement requires:

1. **Split matched/unmatched before plotting**
   - For each data category (e.g. "Altimeter", "Mooring"), partition into matched and unmatched subsets
   - Unmatched = points in the category that are not in `matched_indices` / `matched_mask`

2. **Plot in z-order sequence**
   - Loop order: unmatched layers → unmatched in-situ → matched layers → matched in-situ
   - Use `zorder=` parameter on each `ax.scatter()` call to enforce explicit layering

3. **Apply color/alpha styling**
   - Unmatched: always gray (#808080), alpha=0.3
   - Matched: source color from `source_style_map`, alpha=0.6 (layer) or 0.7 (in-situ)
   - Marker: from `source_style_map`, same for both matched and unmatched

4. **Update legend**
   - Add a single explanatory entry: "Filled = matched, faint gray = unmatched"
   - Keep per-source labels (e.g., "In-situ matched: Mooring", "Altimeter matched")
   - Remove any old "unmatched" labels if they exist with distinct colors

### Integration into `validation_report()`

If `plot_collocation_diagnostics()` is not already wired into `validation_report()` (Task 7 of the original plan), add it:
- Call the diagnostics plot once per recipe
- Save the output PNG to `plots/collocation_diagnostics.png`
- Add a PDF page to the validation report

Current status: Task 7 of the original plan includes wiring per-variable plots (scatter, offset) but **not the geography/diagnostics plots**. The diagnostics plot should be added as a separate section if not already present.

---

## Affected Code Locations

- **Main function:** `sar_validation/core/visualization.py::plot_collocation_diagnostics()`
- **Style mapping:** Uses existing `_source_style_map()` from Task 1
- **Tests:** `tests/test_visualization.py::TestPlotCollocationDiagnostics`
- **Integration:** `sar_validation/core/visualization.py::validation_report()`

---

## Success Criteria

1. ✅ Unmatched points render in gray with alpha=0.3
2. ✅ Matched points render in source color with alpha=0.6–0.7
3. ✅ Same marker shape per source for both matched and unmatched
4. ✅ In-situ data visibly on top of layer data (no overlap hiding)
5. ✅ Legend clearly explains matched vs. unmatched
6. ✅ Collocation diagnostics plot included in `validation_report.pdf`
7. ✅ Visual inspection confirms plot is less busy and easier to read

---

## Dependencies

- Existing `_source_style_map()` from Task 1 (already implemented)
- Existing `plot_collocation_diagnostics()` function (already exists, needs refinement)
- No new third-party dependencies

---

## Notes

- The gray color (#808080) is neutral and works in both light and dark themes.
- Alpha values (0.3 for unmatched, 0.6–0.7 for matched) can be tuned during visual testing if needed.
- The z-order values (2–5) leave room for additional layers in future refinements.
