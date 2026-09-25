# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- Fixed altimeter validation observations being collocated more than once when a recipe is re-run against an output directory that already holds an earlier, overlapping-window download of the same product — this previously inflated the reported N and skewed bias/RMSE/correlation for affected altimeter sources.
- Fixed `--layer-vs-layer-collocation-method` not applying to ERA5/HYCOM ("model") validation sources — the flag previously affected only scatterometer/radiometer/altimeter. A recipe's own per-source `collocation_kwargs` override still takes precedence over the flag.
- Fixed the in-situ download batch recording no metadata entry in download_metadata.json when every source in it is skipped (covered by a historical source, or predicted not to collocate), which caused every subsequent run to redownload the batch from scratch even with nothing missing.
- Scaled the shared Copernicus Marine in-situ index cache's lifetime by how historic a recipe's query window is (1 day for recent windows, up to 30 days for windows more than ~2 months old), reducing repeated large downloads when validating older time periods.
- Added a gridded geographic difference plot (SAR-minus-validation) using pcolormesh: cell size follows each source's own SAR aggregation footprint for cell-averaged collocations, or its native SAR pixel resolution for individual-method collocations.
- Renamed the Copernicus Marine in-situ `buoy`/`ferrybox` source types to `buoy_cmems`/`ship_cmems_family` for clarity alongside the new GTS sources. Added `buoy_cmems_family`, a convenience source_type combining `mooring` + `buoy_cmems` + `drifter` (moored buoys, drifting buoys, and autonomous drifters) into one entry, matching GTS's `buoy_gts` scope so a recipe can switch feeds with a one-line change. `tidal_gauges` is now suppressed by default when creating a wind recipe to not mix gts and cmems sources.
- Added GTS (WMO Global Telecommunication System, via ECMWF MARS) buoy and ship data sources (`buoy_gts`, `ship_gts`), including deduplication against overlapping Copernicus Marine in-situ stations (`buoy_waterfall`) and `--dry-collocation` prediction support.
- Added a reprocessed (multi-year) altimeter product, automatically split from the near-real-time product at the appropriate cutover date.
- Renamed `--dry-run` to `--dry-download` for clarity, now that `--dry-collocation` also exists.
- Added QC flags for Sentinel-1 OWI and OSW and Copernicus Marine in-situ data.
- Added `--dry-collocation` CLI mode: predicts which validation sources would actually collocate with a recipe's SAR data, without downloading anything, and writes a report.
- Real runs now default to only downloading validation-source data predicted to collocate with the recipe's SAR data; the opt-out flag `--download-all-in-bbox` restores the previous behavior of downloading everything within the recipe's bounding box.
- Fixed `SARDownloader.download()` silently losing its Sentinel-1 OCN zip-extraction file list, which also re-activates SAR-scene-aware temporal window narrowing for Sentinel-1 OCN recipes' validation-source downloads (previously silently inert since `orchestrator.py`'s scene-time computation always fell back to `None`). This changes what gets downloaded for every Sentinel-1 OCN recipe, independent of `--dry-collocation`/`--download-all-in-bbox`.
- Apply a spatial filter to satellite data downloaders (OSI-SAF (HY-2B/2C & Oceansat-3), G-Portal (AMSR), and SMOS) using satellite orbits to reduce the amount of downloaded data.
- added --altimeter-freq choice to CLI for waves-recipes. Default is "1hz", other options are "5hz" or "both"
- Removed temporal offset plots from the validation report for clarity.
- Initial reorganization to `src/` layout and added repository guidelines.
