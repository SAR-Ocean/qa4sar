# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- Added `--dry-collocation` CLI mode: predicts which validation sources would actually collocate with a recipe's SAR data, without downloading anything, and writes a report.
- Real runs now default to only downloading validation-source data predicted to collocate with the recipe's SAR data; the opt-out flag `--download-all-in-bbox` restores the previous behavior of downloading everything within the recipe's bounding box.
- Apply a spatial filter to satellite data downloaders (OSI-SAF (HY-2B/2C & Oceansat-3), G-Portal (AMSR), and SMOS) using satellite orbits to reduce the amount of downloaded data.
- added --altimeter-freq choice to CLI for waves-recipes. Default is "1hz", other options are "5hz" or "both"
- Removed temporal offset plots from the validation report for clarity.
- Initial reorganization to `src/` layout and added repository guidelines.
