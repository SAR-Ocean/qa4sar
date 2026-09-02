# QC flag implementation status

This document surveys every data source in the toolbox for quality-control
(QC) flag handling: does the raw product carry a QC/quality flag, and if so,
is the code actually reading and applying it? It was compiled on 2026-08-28
by inspecting `src/sar_validation/core/datatree_converter.py` (where nearly
all flag logic lives, not the downloaders) and, for sources whose format
wasn't already documented in code, by inspecting real cached sample files
under `data/` and, where no sample existed, official product documentation.
Last updated 2026-09-02 to reflect SM/IW/EW OSW grid extraction closing
the mode gap noted in the previous update.

Sources are grouped SAR products first, then wind/wave/currents validation
sources, then soil moisture — matching validation priority in this toolbox.

## At-a-glance summary

| Source | Flag in raw data? | Flag name(s) | Current handling | Status |
|---|---|---|---|---|
| Sentinel-1 OWI wind | Yes | `owiWindQuality`, `owiInversionQuality`, `owiMask` (land bit) | Land bit filtered; `owiWindQuality`/`owiInversionQuality` reject cells scoring 2 or 3, or both NaN | Done |
| Sentinel-1 RVL currents | Yes (land only) | `rvlLandFlag` | Land-contaminated cells NaN'd | Done |
| Sentinel-1 OSW waves | Yes | `oswLandFlag`, `oswQualityFlag` (`oswQualityFlagPartition`, `oswIconf`, `oswSnr`, `oswAmbiFac` deliberately not used) | `oswLandFlag==1` and `oswQualityFlag>=2` each independently reject a point (WV) or grid cell (SM/IW/EW), sharing the same reject thresholds; both modes fully implemented | Done |
| RADARSAT-2 wind | Yes | `pixel_level_quality_flags` (+ `mask`/`icemask` fallback) | Filtered to flag==5 (valid wind, valid water) | Done |
| NISAR L3 SME2 | Yes, verified redundant | `retrievalQualityFlag` | Deliberately unused — confirmed to flag the same cells as the fill value | Done (by design) |
| Copernicus Marine HF-radar totals | Yes | Overall `QCflag`, per-parameter `*_QC` | Overall flag filtered to CMEMS's valid-code set (1, 2, 5, 7, 8); per-parameter flags unused | Done |
| NOAA HF-radar | No | — | N/A — NOAA filters upstream before publishing | N/A |
| Copernicus Marine in-situ (buoy/mooring/drifter/ferrybox/tide-gauge, incl. ADCP/Argo/glider) | Yes | `value_qc` (CMEMS 0-9 scale) | Filtered to CMEMS's valid-code set (1, 2, 5, 7, 8); value and QC column nulled together otherwise | Done |
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

**Sentinel-1 OWI wind** (`_extract_owi_grid_data`, `datatree_converter.py:3485-3707`)
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
`owi_quality_masked_pixel_count`/`fraction`.

**Sentinel-1 RVL currents** (`_extract_rvl_grid_data`, `datatree_converter.py:3107-3230`
grid path; `:3358-3370` WV-mode vignette path)
`rvlLandFlag` is set to 1 where a cell's land coverage exceeds 10%.
`rvlRadVel`/`rvlRadVelStd` are NaN'd where the flag is set; the pre-mask
mean is retained as a QA statistic. `rvlHeading`/`rvlIncidenceAngle` are
left unmasked since they're geometry, not a measurement. There is no
separate QC-code flag for RVL beyond the land flag — this source is
considered complete.

**Sentinel-1 OSW waves** (`from_sar_l2_ocn_wv_safe`, `datatree_converter.py:2899-3103`)
Implemented for both WV mode (sparse vignette points, `from_sar_l2_ocn_wv_safe`)
and SM/IW/EW grid mode (`_extract_osw_grid_data`), the latter extracting
the product's native `oswAzSize x oswRaSize` OSW grid rather than falling
back to OWI/RVL. Both paths gate `oswTotalHs`/`oswHs` on two independent checks,
straight from ESA's own product flags: `oswLandFlag==1` rejects a point
(land coverage exceeds 10% of the vignette), and `oswQualityFlag>=2`
rejects a point (the product's own total-quality flag, 0=good to 3=poor).
Both checks require the flag to be present and finite for that point — a
missing or fill-value flag is a no-op, never a rejection — and neither is
deduplicated against the other, so a point failing both is counted by
each. `oswQualityFlag` is never populated by ESA's processor today (always its
fill value), so that check is currently a documented no-op, kept in place
so it activates automatically if a future processor version starts
populating it — the same precedent as NOAA HF-radar's `QCflag` handling. `oswTotalHsStdev`,
`oswQualityFlagPartition`, and `oswSnr` were deliberately not used: none
is an ESA-defined rejection criterion for `oswTotalHs`, and using one
would mean inventing this toolbox's own OSW quality-control rule rather
than applying what ESA already publishes. Land-masking and
quality-masking pixel counts are tracked independently via
`osw_land_pixel_count`/`fraction` and
`osw_quality_masked_pixel_count`/`fraction` in both paths, sharing the
same reject thresholds (`_OSW_LAND_FLAG_REJECT_VALUE`,
`_OSW_QUALITY_FLAG_REJECT_THRESHOLD`). Both flags are also surfaced as
output data variables for downstream inspection. On the WV point path,
`_collocate_wv_points` (`collocation.py:1294`) excludes both from its
"does this vignette have usable data" check (`_WV_AUXILIARY_FLAG_VARS`,
`collocation.py:1297`) so a masked `oswTotalHs` does not produce a
phantom collocation match just because the always-finite flags are
present. The SM/IW/EW grid path does not need an equivalent guard: grid
collocation (`_compute_aggregated_sar_value`) aggregates each variable
independently and simply omits a variable from a match when all its
nearby cells are NaN, rather than gating a whole match on "any variable
finite" — the same reason `_extract_owi_grid_data`'s own
`owiMask`/`owiWindQuality`/`owiInversionQuality` passthrough needed no
such guard. `docs/design-choices.md` §5.5 documents this under "WV wave
quality-flag masking" and "SM/IW/EW: the native OSW grid".

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
`datatree_converter.py:2288-2493`)
An overall `QCflag` (1=good … 4=bad) is filtered to keep only cells whose
code is one of CMEMS's documented valid QC codes (1, 2, 5, 7, or 8).
Per-parameter flags (`CSPD_QC`, `DDNS_QC`, `GDOP_QC`, `VART_QC`,
`POSITION_QC`) are extracted into `hfr_qc_<param>` output fields but not
used to filter anything.

**NOAA HF-radar** — confirmed via an explicit code comment that NOAA's
product carries no equivalent flag; NOAA filters upstream before
publishing. The `QCflag` code path is shared with Copernicus HF-radar but
is a documented no-op for NOAA files.

**Copernicus Marine in-situ** (buoys, moorings, drifters, ferrybox,
tide-gauges, plus ADCP/Argo/glider/drifter historical currents — all flow
through `from_insitu_csv`, `datatree_converter.py:645-847`)
The raw CSV carries a per-variable `value_qc` column (CMEMS 0-9 scale).
The pivot keeps `value_qc`, flattens it to `<CODE>_QC` companion columns,
and nulls both the value and its `_QC` column when the code is not one of
CMEMS's valid codes (1, 2, 5, 7, or 8) (`:713-756`) — see
`tests/test_insitu_qc_filtering.py`, the extensions to
`tests/test_datatree_converter_insitu.py`/`tests/test_cf_metadata.py`, and
`docs/design-choices.md` §3.7. `EWCT`/`NSCT` derived from `HCSP`/`HCDT`
inherit NaN from a bad-QC `HCSP`/`HCDT` through `sin`/`cos`/multiplication
rather than through an explicit gate.

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

Ranked by group (wind/wave/currents → soil moisture), then within each
group by risk of silently bad data reaching validation results — sources
where a real, documented flag already exists in the data and is simply
unread rank above sources whose format or flag existence is still
unconfirmed. No SAR-product gaps remain open.

**Wind / wave / currents**
1. **ISMN in-code QC enforcement** — currently relies entirely on an
   unenforced manual portal step; converting that into a code-level
   guarantee (e.g. verifying selected quality flags at download/read time,
   or reading whatever flag the `ismn` package's reader exposes) would
   close a real trust gap in a heavily-used soil-moisture reference source.
   (Grouped here rather than under soil moisture because it's an in-situ
   validation source, matching this toolbox's own category boundaries.)

**Soil moisture**
2. **CDS SSM (C3S) `flag` bitmask** — explicit, well-documented bitmask
   confirmed present and entirely unused; likely the most contained
   soil-moisture fix (single variable, already-known meanings).
3. **SMAP `retrieval_qual_flag`** — confirmed present bitmask, entirely
   unused; NASA's official flag documentation would need a short lookup to
   pick the right bits to reject on, but the variable itself is already
   confirmed to exist and load cleanly.
4. **ASCAT SSM (SOMO12) flag fields** — `agg_flag`/`proc_flag`/`corr_flag`
   confirmed present and unused; the reader package is already a
   dependency, so no new parsing code is needed, only flag-based filtering
   logic.
5. **AMSR AU_Land `RetrievalQualityFlagNPD`/`SCA`** — named in code but
   never opened; smaller in scope (one source variant) than the above.
6. **H-SAF ASCAT SSM (H29) flag support** — real flag confirmed to exist
   via product docs, but the file format itself is unconfirmed against a
   real download in this codebase; implementing this requires first
   obtaining and inspecting a genuine H29 file, so it carries more
   up-front research risk than the others above.
7. **AMSR-E/AMSR2 NSIDC-0451 format confirmation** — lowest priority: even
   the base file format (not just its QC flag) is unconfirmed against a
   real download.

**Not gaps** (confirmed no code action needed): NOAA HF-radar, Altimeter
(CMEMS L3 SWH), HYCOM, SMOS (no discrete flag exists in the delivered NRT
product), AMSR G-Portal L3SGSMC (sentinel-code filtering already
functionally equivalent), NISAR SME2 (verified redundant with fill-value
masking), RVL, RADARSAT-2, CLMS SSM, scatterometer, radiometer wind — all
already complete or genuinely not applicable.
