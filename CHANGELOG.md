# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- Removed temporal offset plots from the validation report for clarity.
- Initial reorganization to `src/` layout and added repository guidelines.
- Apply a spatial filter to satellite data downloaders (OSI-SAF (HY-2B/2C & Oceansat-3), G-Portal (AMSR), and SMOS) using satellite orbits to reduce the amount of downloaded data.
- added --altimeter-freq choice to CLI for waves-recipes. Default is "1hz", other options are "5hz" or "both"
