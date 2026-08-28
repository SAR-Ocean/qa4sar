# QC flag implementation status

This document surveys every data source in the toolbox for quality-control
(QC) flag handling: does the raw product carry a QC/quality flag, and if so,
is the code actually reading and applying it? It was compiled on 2026-08-28
by inspecting `src/sar_validation/core/datatree_converter.py` (where nearly
all flag logic lives, not the downloaders) and, for sources whose format
wasn't already documented in code, by inspecting real cached sample files
under `data/` and, where no sample existed, official product documentation.

Sources are grouped SAR products first, then wind/wave/currents validation
sources, then soil moisture — matching validation priority in this toolbox.

## At-a-glance summary

| Source | Flag in raw data? | Flag name(s) | Current handling | Status |
|---|---|---|---|---|
| Sentinel-1 OWI wind | Yes | `owiWindQuality`, `owiInversionQuality`, `owiMask` (land bit) | Land bit filtered; `owiWindQuality`/`owiInversionQuality` reject cells scoring 2 or 3, or both NaN | Done (on `qc-flag`, not yet merged) |
| Sentinel-1 RVL currents | Yes (land only) | `rvlLandFlag` | Land-contaminated cells NaN'd | Done |
| Sentinel-1 OSW waves | Yes | `oswQualityFlag`, `oswQualityFlagPartition`, `oswLandFlag`, `oswIconf`, `oswSnr`, `oswAmbiFac` | None read or applied at all (WV mode only; IW/EW/SM grid mode falls back to OWI/RVL, OSW extraction not implemented there) | Not implemented |
| RADARSAT-2 wind | Yes | `pixel_level_quality_flags` (+ `mask`/`icemask` fallback) | Filtered to flag==5 (valid wind, valid water) | Done |
| NISAR L3 SME2 | Yes, verified redundant | `retrievalQualityFlag` | Deliberately unused — confirmed to flag the same cells as the fill value | Done (by design) |
| Copernicus Marine HF-radar totals | Yes | Overall `QCflag`, per-parameter `*_QC` | Overall flag excludes only code 4 on `main`; widened to CMEMS's full valid-code set on `qc-flag`; per-parameter flags unused | Done (on `qc-flag`, not yet merged) |
| NOAA HF-radar | No | — | N/A — NOAA filters upstream before publishing | N/A |
| Copernicus Marine in-situ (buoy/mooring/drifter/ferrybox/tide-gauge, incl. ADCP/Argo/glider) | Yes | `value_qc` (CMEMS 0-9 scale) | Not read on `main`; filtered by valid QC code on `qc-flag` | Done (on `qc-flag`, not yet merged) |
| ISMN (soil moisture ground stations) | Yes, at source portal | ISMN's own "Good"/etc. scheme | Enforced only via unenforced manual portal instructions; no code-level filtering | Not implemented |
| Altimeter (CMEMS L3 SWH) | No | — | N/A — filtered upstream (`VAVH` vs `VAVH_UNFILTERED` naming implies pre-filtering) | N/A |
| ERA5 (reference model) | Land mask, not a QC flag | `land_sea_mask`/`lsm` | Applied at collocation time to exclude land points from wind comparisons | Done |
| HYCOM (reference model) | No | — | N/A — fixed-grid reanalysis, no per-cell retrieval QC concept | N/A |
| Scatterometer (MetOp ASCAT winds, HY-2, Oceansat-3) | Yes | `wvc_quality_flag` bitmask | Cells with any reject bit dropped entirely | Done |
| CLMS SSM (Sentinel-1 L3 GeoTIFF) | Yes | GDAL `flag_values` (241/242/251/252/253) | Flagged codes masked to NaN | Done |
| SMAP SSM (SPL2SMP_E) | Yes, confirmed | `retrieval_qual_flag` (+ `surface_flag`, `tb_qual_flag_*`) | Not read at all; only fill-value/NaN filtering | Not implemented |
| SMOS SSM (SM_OPER_MIR_SMUDP2) | No discrete flag | Only continuous `RFI_probability` | Not read; only fill-value/NaN filtering | N/A (nothing to filter on) |
| CDS SSM (C3S active/passive/combined) | Yes, confirmed | `flag` bitmask (snow, vegetation, no-convergence, out-of-range, low-weight, barren-ground, unreliable) | Not read at all | Not implemented |
| H-SAF ASCAT SSM (H29) | Yes, per product docs | Surface State Flag (SSF, freeze/thaw) + correction flags | Not read; format itself unconfirmed against a real file | Not implemented |
| ASCAT SSM (SOMO12, `ascat` package/EUMDAC) | Yes, confirmed | `agg_flag`, `proc_flag`, `corr_flag`, `snow_prob`, `frozen_prob`, `f_land` | Not read at all; only NaN filtering | Not implemented |
| AMSR-E/AMSR2 SSM — AU_Land (L2B) | Yes, named but unused | `RetrievalQualityFlagNPD`/`RetrievalQualityFlagSCA` | Fields named in code docstring but never opened | Not implemented |
| AMSR-E/AMSR2 SSM — G-Portal (L3SGSMC) | No separate flag | Sentinel codes (-32768/-32767) | Sentinel-code validity filter applied (functionally equivalent) | Done for available signal |
| AMSR-E/AMSR2 SSM — NSIDC-0451 | Unconfirmed | — | Fill-value filtering only; format itself unconfirmed against a real file | Not implemented / unconfirmed format |
| Radiometer wind (RSS AMSR2/GMI/SSMIS/WindSat) | Yes, as encoded fill codes | Bytemap codes ≥251 (land/ice/coast/rain/no-obs/bad) | Decoded to NaN at decode time; NetCDF path relies on RSS's own upstream NaN'ing | Done |

## Per-source detail

### SAR products

**Sentinel-1 OWI wind** (`_extract_owi_grid_data`, `datatree_converter.py:3427-3606`,
on worktree `qc-flag`)
Three flags exist in the product: `owiWindQuality` (0=good, 1=medium,
2=low, 3=poor), `owiInversionQuality` (0=good, 1=medium, 2=poor), and
`owiMask` (a CF bitmask whose bit 0 is land). The land bit is applied first
— cells are NaN'd where `owiMask` marks land. `owiWindQuality` and
`owiInversionQuality` are then read and used to reject cells: a cell is
rejected, and `owiWindSpeed`/`owiWindDirection` NaN'd, when either flag is
present in the product and scores 2 or 3 for that cell, or when both flags
are present and both are NaN for that cell. A flag missing entirely from
the product never contributes to rejection, and a flag that is NaN for a
cell while the other flag is present and 0/1 does not reject that cell
either — the present flag governs. Both flags pass through the output
dataset unmodified, even at rejected cells, for downstream inspection.
Land-masking and quality-masking pixel counts are tracked independently
via `owi_land_pixel_count`/`fraction` and
`owi_quality_masked_pixel_count`/`fraction`. Not yet merged to `main`.

**Sentinel-1 RVL currents** (`_extract_rvl_grid_data`, `datatree_converter.py:3107-3230`
grid path; `:3358-3370` WV-mode vignette path)
`rvlLandFlag` is set to 1 where a cell's land coverage exceeds 10%.
`rvlRadVel`/`rvlRadVelStd` are NaN'd where the flag is set; the pre-mask
mean is retained as a QA statistic. `rvlHeading`/`rvlIncidenceAngle` are
left unmasked since they're geometry, not a measurement. There is no
separate QC-code flag for RVL beyond the land flag — this source is
considered complete.

**Sentinel-1 OSW waves** (`from_sar_l2_ocn_wv_safe`, `datatree_converter.py:2960-3045`)
Only implemented for WV mode (sparse vignette points); IW/EW/SM grid mode
falls back to OWI/RVL instead, with a `logger.debug("OSW extraction not
yet implemented...")` at `_from_sar_l2_ocn_iw_safe` (`:3722-3725`). Even in
the WV-mode path, only `oswLon`/`oswLat` (`:3020-3021`) and
`oswTotalHs`/`oswHs` (`:3032-3045`) are read, filtered with `np.isfinite`
and, in the `oswHs`-partition fallback only, a `-1` fill-code drop
(`:3039`). Confirmed by inspecting a real cached WV OCN file
(`S1D_WV_OCN__.../measurement/s1d-wv1-ocn-vv-...-005.nc`) that the product
also carries `oswQualityFlag`, `oswQualityFlagPartition`, `oswLandFlag`, an
`oswIconf` confidence indicator, `oswSnr`, and `oswAmbiFac` — none of these
are read anywhere in the repo (zero grep hits). This is a stricter gap
than OWI: OWI at least reads its land flag and applies it; OSW reads no
flag at all, including its own land flag. `docs/design-choices.md`
discusses the `oswTotalHs` vs `oswHs`-partition choice (§5.5, lines
397-415) but never mentions `oswQualityFlag`/`oswLandFlag`.

**RADARSAT-2 wind** (`from_radarsat2_wind`, `datatree_converter.py:548-620`)
New-era files carry `pixel_level_quality_flags`; a cell is valid when the
flag equals 5 ("valid wind in valid water region"), cross-validated against
`quality_information.total_number_of_valid_water_pixels`. Old-era files
lacking that flag fall back to `mask==-1 & icemask==1`. `sar_wind` is NaN'd
elsewhere. Complete.

**NISAR L3 SME2** (`datatree_converter.py:456-505`)
`retrievalQualityFlag` exists alongside `soilMoisture` but was checked and
found to flag exactly the same cells the fill value already excludes — a
deliberate, verified decision not to apply it separately, not an oversight.

### Wind / wave / currents validation sources

**Copernicus Marine HF-radar totals** (`from_hf_radar_grid`,
`datatree_converter.py:2349-2517`)
An overall `QCflag` (1=good … 4=bad) excludes only code 4 on `main`. On
worktree `qc-flag`, the check is widened to keep only cells whose code is
one of CMEMS's documented valid QC codes (1, 2, 5, 7, or 8). Per-parameter
flags (`CSPD_QC`, `DDNS_QC`, `GDOP_QC`, `VART_QC`, `POSITION_QC`) are
extracted into `hfr_qc_<param>` output fields but not used to filter
anything, on either branch. Not yet merged to `main`.

**NOAA HF-radar** — confirmed via an explicit code comment that NOAA's
product carries no equivalent flag; NOAA filters upstream before
publishing. The `QCflag` code path is shared with Copernicus HF-radar but
is a documented no-op for NOAA files.

**Copernicus Marine in-situ** (buoys, moorings, drifters, ferrybox,
tide-gauges, plus ADCP/Argo/glider/drifter historical currents — all flow
through `from_insitu_csv`, `datatree_converter.py:721-734` on `main`)
The raw CSV carries a per-variable `value_qc` column (CMEMS 0-9 scale) that
`from_insitu_csv` drops entirely during the pivot on `main`. On worktree
`qc-flag`, the pivot keeps `value_qc`, flattens it to `<CODE>_QC` companion
columns, and nulls both the value and its `_QC` column when the code is
not one of CMEMS's valid codes (1, 2, 5, 7, or 8) — see
`tests/test_insitu_qc_filtering.py` and the extensions to
`tests/test_datatree_converter_insitu.py`/`tests/test_cf_metadata.py` in
that worktree, and `docs/design-choices.md` §3.7. `EWCT`/`NSCT` derived
from `HCSP`/`HCDT` inherit NaN from a bad-QC `HCSP`/`HCDT` through
`sin`/`cos`/multiplication rather than through an explicit gate. Not yet
merged to `main`.

**ISMN** (`ismn_downloader.py:146-148`, `:513-538`)
ISMN's own quality-flag scheme exists at the source portal, but
`ISMNDownloader` reads `ts["soil_moisture"]` with no flag column, and the
only enforcement mechanism is instructions printed to the user to manually
select "Quality flags: Good only" on the ISMN web portal before
downloading. This step is not enforced in code, and the resulting CSVs
carry no `value_qc` column — so even after the in-situ CSV filtering above
ships, ISMN falls into its "no `value_qc` column → skipped" branch and
receives no in-code QC filtering.

**Altimeter (CMEMS L3 SWH)** (`from_altimeter`, `datatree_converter.py:830-940`)
Confirmed by inspecting a real cached file (`SARAL-Altika.nc`): no
per-observation flag variable is delivered. The product ships both `VAVH`
and `VAVH_UNFILTERED`, implying quality filtering already happens upstream
before publication (the same pattern as NOAA HF-radar). Nothing to
implement.

**ERA5** (`model_collocation.py:290-307` point mode, `:613-661`
cell-averaging mode)
Not a QC flag in the retrieval sense — `land_sea_mask`/`lsm > 0.5` is
applied at collocation time (not conversion time) to exclude land
cells/points from wind comparisons. Included here as a bonus finding since
it's flag-like handling of a reference dataset, at a different pipeline
stage than everything else in this document.

**HYCOM** — confirmed the downloader only ever requests `water_u`/`water_v`.
As a fixed-grid ocean reanalysis rather than a satellite retrieval, there is
no per-cell QC-flag concept to apply; land is encoded via the model's own
bathymetry/grid mask, not a flag field.

**Scatterometer** (MetOp ASCAT winds via EUMDAC, HY-2/Oceansat-3 via FTP;
`from_scatterometer_nc`, `datatree_converter.py:34-46`, `:1163-1202`)
`wvc_quality_flag` is a CF bitmask; `_ASCAT_REJECT_FLAGS` covers
land-contaminated, ice-contaminated, failed wind inversion, insufficient
good sigma0, and distance-to-GMF-too-large. Cells carrying any reject bit
are dropped entirely (not NaN'd — removed) before collocation. Complete.

**Radiometer wind** (RSS AMSR2/GMI/SSMIS/WindSat;
`downloaders/_rss_bytemap.py:18-19,42-43,157`;
`datatree_converter.py:2683-2686`)
Bytemap-format files encode land/ice/coast/rain-flag/no-observation/bad as
special byte codes ≥251, decoded to NaN for every variable. The NetCDF path
keeps only finite `wspd` cells, relying on RSS having already NaN'd
land/ice/rain upstream. Functionally equivalent to explicit flag masking.
Complete.

### Soil moisture sources

**CLMS SSM (Sentinel-1 L3 GeoTIFF)** (`from_sar_l3_ssm_geotiff`,
`datatree_converter.py:407-417`)
GDAL-embedded `flag_values` = `{241, 242, 251, 252, 253}` correspond to
ExceedingMin/ExceedingMax/WaterMask/SensitivityMask/SlopeMask. These are
masked to NaN before the scale/offset is applied. Complete.

**SMAP SSM (SPL2SMP_E)** (`from_smap_ssm`, `datatree_converter.py:1727-1837`)
Confirmed by inspecting a real cached file
(`SMAP_L2_SM_P_E_55649_A_...h5`): the product ships
`Soil_Moisture_Retrieval_Data/retrieval_qual_flag` (plus
`retrieval_qual_flag_option1`/`option2`, `surface_flag`, and four
`tb_qual_flag_*` variables) as a `uint16` bitmask, present in both the
standard and polar grids. None of these are read; only the `-9999.0` fill
value and NaN are used for validity. This is the clearest unaddressed gap
in the soil moisture group — a well-documented, standard bitmask sitting
entirely unused.

**SMOS SSM (SM_OPER_MIR_SMUDP2)** (`from_smos_ssm`,
`datatree_converter.py:1839-1935`)
Confirmed by inspecting a real cached file: the NRT L2SM product delivered
to this toolbox carries only `longitude`, `latitude`, `soil_moisture`,
`soil_moisture_uncertainty`, `RFI_probability`, and time fields — no
discrete Science/Confidence/Processing flag variables (those exist in
SMOS's fuller L2SM product but aren't in this simplified NRT delivery).
`RFI_probability` is continuous (not a discrete flag) and currently unused.
There is no discrete flag to filter on here; this is not a gap in the same
sense as the others.

**CDS SSM (C3S active/passive/combined)** (`from_c3s_ssm`,
`datatree_converter.py:1937-2036`)
Confirmed by inspecting a real cached file (`c3s_ssm_active_20250702.nc`):
the product carries an explicit `flag` bitmask with documented
`flag_meanings`: `snow_coverage_or_temperature_below_zero`,
`dense_vegetation`, `no_convergence_in_the_model`,
`soil_moisture_value_exceeds_physical_boundary`,
`weight_of_measurement_below_threshold`, `all_datasets_deemed_unreliable`,
`barren_ground_advisory_flag`. None of this is read; only
`_FillValue`/`scale_factor`/`add_offset` handling is applied to `sm`. A
clear, well-documented gap.

**H-SAF ASCAT SSM (H29)** (`from_hsaf_ssm`, `datatree_converter.py:1396-1461`)
No cached sample exists in this repo, and the code's own docstring notes
the field layout "has not been confirmed against a real downloaded file."
Official H-SAF product documentation confirms H29 carries a Surface State
Flag (SSF, derived from ECMWF forecast data, flagging freeze/thaw
conditions) plus correction flags for extreme events and no-retrieval
reasons — the same flag family found in the related SOMO12 product below.
Neither the format nor the flag handling is verified in code; implementing
this requires downloading and inspecting a real H29 file first. (Sources:
[H29 product page](https://hsaf.meteoam.it/Products/Detail?prod=h29),
[ESSD paper on next-gen ASCAT SSM](https://essd.copernicus.org/articles/18/4393/2026/))

**ASCAT SSM (SOMO12, `ascat` package/EUMDAC)** (`from_ascat_ssm`,
`datatree_converter.py:1317-1393`)
Confirmed by reading a real cached `.nat` file
(`ASCA_SMR_02_M01_...nat`) through the `ascat.eumetsat.level2.AscatL2File`
reader: the struct carries `f_land` (land fraction), `agg_flag`,
`proc_flag`, `corr_flag`, `snow_prob`, `frozen_prob`, `wetland`, and `topo`
fields, none of which are read. Only NaN filtering is applied. A clear gap,
and one where the reader package needed to open the flags is already a
dependency.

**AMSR-E/AMSR2 SSM — AU_Land (L2B)**
(`_from_amsr_ssm_au_land_points`, `datatree_converter.py:1596-1621`)
`RetrievalQualityFlagNPD`/`RetrievalQualityFlagSCA` (per-algorithm quality
flags) are named in the code's own docstring as existing in the product but
are never opened; only `(sm != -9999.0) & ~np.isnan(sm)` is applied.

**AMSR-E/AMSR2 SSM — G-Portal (L3SGSMC)**
(`_from_amsr_ssm_gportal_l3_grid`, `datatree_converter.py:1664-1684`)
No separate QC variable exists in this product; validity is expressed via
documented sentinel codes (-32768 = no-retrieval, -32767 = missing), which
are filtered. Functionally complete — there's no distinct flag field to
add.

**AMSR-E/AMSR2 SSM — NSIDC-0451**
(`datatree_converter.py:1499-1509`)
Format itself is unconfirmed against a real downloaded file (per the
code's own docstring). Only fill-value filtering (`-9999.0`) is applied.
Whether a QC flag exists here is unknown until the format is verified.

## Prioritized gap list

Ranked by group (SAR → wind/wave/currents → soil moisture), then within
each group by risk of silently bad data reaching validation results —
sources where a real, documented flag already exists in the data and is
simply unread rank above sources whose format or flag existence is still
unconfirmed.

**SAR products**
1. **Sentinel-1 OSW wave quality/land masking** — `oswQualityFlag`,
   `oswQualityFlagPartition`, and `oswLandFlag` are real, present, and
   completely unused, in the only mode (WV) where OSW extraction exists at
   all; this is a stricter gap than OWI since not even a land flag is
   applied. No spec exists yet for this one.

**Wind / wave / currents**
2. **ISMN in-code QC enforcement** — currently relies entirely on an
   unenforced manual portal step; converting that into a code-level
   guarantee (e.g. verifying selected quality flags at download/read time,
   or reading whatever flag the `ismn` package's reader exposes) would
   close a real trust gap in a heavily-used soil-moisture reference source.
   (Grouped here rather than under soil moisture because it's an in-situ
   validation source, matching this toolbox's own category boundaries.)

**Soil moisture**
3. **CDS SSM (C3S) `flag` bitmask** — explicit, well-documented bitmask
   confirmed present and entirely unused; likely the most contained
   soil-moisture fix (single variable, already-known meanings).
4. **SMAP `retrieval_qual_flag`** — confirmed present bitmask, entirely
   unused; NASA's official flag documentation would need a short lookup to
   pick the right bits to reject on, but the variable itself is already
   confirmed to exist and load cleanly.
5. **ASCAT SSM (SOMO12) flag fields** — `agg_flag`/`proc_flag`/`corr_flag`
   confirmed present and unused; the reader package is already a
   dependency, so no new parsing code is needed, only flag-based filtering
   logic.
6. **AMSR AU_Land `RetrievalQualityFlagNPD`/`SCA`** — named in code but
   never opened; smaller in scope (one source variant) than the above.
7. **H-SAF ASCAT SSM (H29) flag support** — real flag confirmed to exist
   via product docs, but the file format itself is unconfirmed against a
   real download in this codebase; implementing this requires first
   obtaining and inspecting a genuine H29 file, so it carries more
   up-front research risk than the others above.
8. **AMSR-E/AMSR2 NSIDC-0451 format confirmation** — lowest priority: even
   the base file format (not just its QC flag) is unconfirmed against a
   real download.

**Not gaps** (confirmed no code action needed): NOAA HF-radar, Altimeter
(CMEMS L3 SWH), HYCOM, SMOS (no discrete flag exists in the delivered NRT
product), AMSR G-Portal L3SGSMC (sentinel-code filtering already
functionally equivalent), NISAR SME2 (verified redundant with fill-value
masking), RVL, RADARSAT-2, CLMS SSM, scatterometer, radiometer wind — all
already complete or genuinely not applicable.

**Resolved on worktree `qc-flag`, not yet merged to `main`**: Copernicus
Marine in-situ `value_qc` filtering, Copernicus HF-radar `QCflag` widening
to CMEMS's valid-code set, and Sentinel-1 OWI wind quality masking
(`owiWindQuality`/`owiInversionQuality`).

## In-flight work status

- **`.claude/worktrees/qc-flag`** (branch `worktree-qc-flag`, local only,
  not pushed): implements `value_qc` filtering for Copernicus Marine
  in-situ data, widens the Copernicus HF-radar `QCflag` check to CMEMS's
  full valid-code set, and adds Sentinel-1 OWI wind quality masking via
  `owiWindQuality`/`owiInversionQuality`. Covers both halves of the
  2026-08-25 in-situ/HF-radar spec plus the full 2026-08-26 OWI spec,
  including the `docs/design-choices.md` §3.7 write-up and the OWI
  land-pixel-filtering entry.
