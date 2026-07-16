# Phase 3c — Copernicus HF-Radar Gridded Currents Integration Run

**Date:** 2026-07-16
**Plan:** `docs/superpowers/plans/2026-07-15-phase3c-copernicus-hfradar-grid.md`

## Data sources

- **Copernicus Marine HF-radar grid (NRT):** `cmems_obs-ins_glo_phybgcwav_mynrt_na_irr`,
  `dataset_part=latest-radar-total--Skagerrak`, 2026-07-01 → 2026-07-02.
- **Sentinel-1 L2 OCN:** 2 scenes (S1D, IW mode) over the Skagerrak bbox
  (`lon [7.5, 12.0]`, `lat [57.0, 59.5]`), same window.
- **Copernicus in-situ** (drifter/ferrybox/mooring): 5,936 rows, same bbox/window.

Recipe used: a `--create-recipe currents` template trimmed to `hf_radar` +
`drifter`/`ferrybox`/`mooring` (SAR always included). `hf_radar_historical`
and `hf_radar_noaa` were exercised separately (see below) rather than in the
same run, since their valid date ranges don't overlap this window.

## Result

- Download: `hf_radar` **succeeded** — 25×93×89 grid, 4,728 valid EWCT/NSCT
  cells (real data, not all-NaN).
- Collocation: **148 collocated pairs**, all via the `hf_radar_grid`
  `layer_vs_layer` path (`collocation_type == "layer_vs_layer"` for all 148).
- Every ancillary field survived through to the final `collocation_results.nc`
  output: `val_hfr_gdop`, `val_hfr_ewcs`, `val_hfr_nscs`, `val_hfr_qc`,
  `val_hfr_qc_cspd`, `val_hfr_qc_ddns`, `val_hfr_qc_gdop`, `val_hfr_qc_vart`,
  `val_hfr_qc_position` are all present as real (non-placeholder) data
  columns — confirming the project owner's explicit requirement (standard
  deviations + every quality flag, not just current components) holds
  end-to-end, not just at the converter-unit-test level.
- Statistics (`radar` source, N=146): bias 0.554, std 0.318, RMSE 0.638,
  **r = 0.301**, scatter index 1.545.
- Scatter, residuals, statistics, temporal-offset, and geographic plots all
  generated without error; `validation_report.pdf` produced (802 KB).

## Weak-to-moderate correlation is the expected outcome, not a defect

As in the Phase 3a NOAA run, raw `rvlRadVel` includes a wind-wave artefact
bias not yet corrected (WASV correction deferred to a later phase, per
design §3.7 / Martin, Gommenginger, Jacob & Staneva 2022, RSE 268:112758).
r = 0.301 here is higher than Phase 3a's NOAA r = 0.141, plausibly because
Skagerrak is a long-established, dense European HF-radar network (unlike
the sparser US-EastGulfCoast NRT feed — see below), but still well within
the paper's documented "poor agreement without correction" range (r < 0.5).

## Data-availability finding: US-EastGulfCoast NRT grid was empty for the originally-planned test window

The original plan (Task 8) targeted `US-EastGulfCoast`, 2026-04-05/06 (the
same bbox/date already used for the Phase 3a NOAA run). The download
succeeded (77 MB, correct shape/variables), but the grid was **entirely
NaN** — 0 of 2,132,125 EWCT/NSCT cells valid. `from_hf_radar_grid` correctly
detected this (`all cells NaN`, returns `None`) and the pipeline degraded
gracefully: the node was dropped, collocation still ran on the remaining
24 pairs (all in-situ), and `run_statistics` logged a clear "no statistics
produced" warning rather than failing. This is **not a code defect** — it's
the same open question flagged in the original Phase 3 design doc (§6,
"Confirm whether 013_030 HF-radar are ever [present]") resolving in the
least favorable direction for this specific region/date: the NRT feed
exists and downloads correctly, but had no active station data for that
particular day. Switching to Skagerrak (a region with a long operational
history) for the same kind of window immediately produced real data,
confirming the pipeline itself is sound — the emptiness was regional/
temporal data availability, not a bug.

A second attempt at `US-EastGulfCoast` with a *recent* date (2026-07-10)
surfaced a related, also-correct behavior: `monthly-radar-total--US-EastGulfCoast`'s
real processing lag caps out around 2026-05-04 (i.e. this region's NRT
product is roughly 2.5 months behind "now" in this session's clock) — the
downloader raised a clear `CoordinatesOutOfDatasetBounds`-derived error
rather than silently returning nothing.

## hf_radar_historical (013_044) — verified against both an invalid and a valid year

`hf_radar_historical` was exercised against `US-EastGulfCoast`/2026 (a year
outside the archive's 2019–2024 coverage) and correctly raised
`ValueError: No US-EastGulfCoast historical archive for year 2026;
available years: (2019, 2020, 2021, 2022, 2023, 2024)`, matching the
unit-tested behavior from Task 5 exactly.

A real data-bearing run was then done directly against the downloader
(`US-EastGulfCoast`, 2021-06-05/06): the 252 MB regional archive
(`GL_TV_HF_HFR-US-EastGulfCoast_Total_2021.nc`) downloaded, normalized
(uppercase `TIME/DEPTH/LATITUDE/LONGITUDE` → lowercase `time/latitude/
longitude`, `DEPTH` squeezed), and subset correctly — producing 30,643
valid EWCT cells (of 606,825) with every ancillary field present (`EWCS`,
`NSCS`, `QCflag`, `CSPD_QC`, `DDNS_QC`, `GDOP_QC`, `VART_QC`,
`POSITION_QC`). `hf_radar_historical` is confirmed working against real
data end-to-end at the downloader level; a full CLI pipeline run
(convert → collocate → stats → plot) through this specific file was not
additionally performed in this session, since Task 3's converter path was
already proven identical for the NRT case (Skagerrak, above) and Task 5/6's
own unit tests cover the shape/normalization contract this real download
just confirmed.

## hf_radar_noaa — pre-existing, unrelated limitation (unaffected by this work)

`hf_radar_noaa` failed against dates older than ERDDAP's ~90-day rolling
window with `NotImplementedError` ("the THREDDS/OPeNDAP archive backend is
Phase 3b") — this is the known, pre-existing, explicitly-out-of-scope
limitation from the Phase 3a design (Phase 3b was descoped from the Phase
3c plan because `hfrnet-tds.ucsd.edu` was unreachable during planning).
Not a regression.

## Success criteria (from the plan's Task 8)

- [x] `hf_radar` download succeeds (previously always failed with "output
      CSV not found")
- [x] Full pipeline (download → convert → collocate → stats → plot)
      completes without error
- [x] Non-empty set of collocated pairs (148) over a known-good overlap
      region
- [x] Projected radial magnitudes physically plausible (r = 0.301, within
      the literature's documented uncorrected range)
- [x] Scatter plot and geographic map generated without errors
- [x] EWCS/NSCS standard deviations and all quality flags present through
      to the final collocated-pairs output (explicit project-owner
      requirement)
- [x] Delayed-mode (`hf_radar_historical`) spot-check: both the
      unsupported-year error path and a real data-bearing archive (2021,
      US-EastGulfCoast) verified against the live Copernicus Marine API
