"""
Convert validation data to xarray.DataTree format.

Step 2 of the validation pipeline.

Provides converters for:
  - SAR L2_OCN data              → standardised Dataset
  - In-situ CSV (Copernicus)     → point-geometry Dataset (point dim)
  - Scatterometer NetCDF (ASCAT) → flattened point-geometry Dataset
  - Altimeter netCDF             → Dataset
  - Collocated results           → Dataset  (step 4 output)

And a ``to_datatree()`` helper to assemble multiple Datasets into one
hierarchical DataTree.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import logging

logger = logging.getLogger(__name__)

__all__ = ["DataTreeConverter"]


class DataTreeConverter:
    """Convert various data formats to standardised xarray objects."""

    # ------------------------------------------------------------------
    # Step 2 converters
    # ------------------------------------------------------------------

    @staticmethod
    def from_sar_l2_ocn(
        sar_data: Union[xr.Dataset, Dict],
    ) -> xr.Dataset:
        """
        Wrap SAR L2_OCN data in a standardised Dataset.

        Parameters
        ----------
        sar_data : xr.Dataset or dict
            SAR data as an xarray Dataset or a plain dict of arrays.

        Returns
        -------
        xr.Dataset
            Dataset with ``data_type`` attribute set to ``"sar_l2_ocn"``.
        """
        if isinstance(sar_data, xr.Dataset):
            ds = sar_data.copy()
        else:
            ds = xr.Dataset(sar_data)

        ds.attrs.setdefault("data_type", "sar_l2_ocn")
        ds.attrs.setdefault("source", "Sentinel-1")
        return ds

    @staticmethod
    def from_insitu_csv(
        csv_path: Union[str, Path],
        source_type: str = "mooring",
    ) -> Optional[xr.Dataset]:
        """
        Convert a Copernicus Marine in-situ CSV to a point-geometry Dataset.

        Each row becomes one observation indexed by a flat ``point`` dimension.
        Coordinates are ``lon``, ``lat``, ``time``, and optionally
        ``platform_id``.

        Parameters
        ----------
        csv_path : str or Path
            Path to CSV file (output from the in-situ or HF radar downloader).
        source_type : str
            Platform category; stored as a Dataset attribute.

        Returns
        -------
        xr.Dataset or None
            None if the file does not exist.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            logger.warning("CSV not found: %s", csv_path)
            return None

        df = pd.read_csv(csv_path)

        # Normalise column names
        rename = {"longitude": "lon", "latitude": "lat"}
        df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

        for col in ("lon", "lat", "time"):
            if col not in df.columns:
                raise ValueError(
                    f"Required column '{col}' not found in {csv_path}. "
                    f"Available columns: {list(df.columns)}"
                )

        df["time"] = pd.to_datetime(df["time"])

        # ------------------------------------------------------------------
        # Copernicus Marine in-situ files use a long format where each row
        # holds one variable code ("WSPD", "EWCT", …) in a ``variable``
        # column and the measurement in a ``value`` column.  Pivot to wide
        # format so that each variable code becomes its own column — this is
        # what the collocation and statistics modules expect.
        # ------------------------------------------------------------------
        if "variable" in df.columns and "value" in df.columns:
            pivot_id_cols = [
                c for c in ("platform_id", "time", "lon", "lat", "depth")
                if c in df.columns
            ]
            df = (
                df.pivot_table(
                    index=pivot_id_cols,
                    columns="variable",
                    values="value",
                    aggfunc="first",
                )
                .reset_index()
            )
            df.columns.name = None  # remove the "variable" MultiIndex label
            logger.debug(
                "Pivoted in-situ CSV to wide format; variable columns: %s",
                [c for c in df.columns if c not in pivot_id_cols],
            )

        coord_cols = {"lon", "lat", "time", "platform_id"}
        data_cols  = [c for c in df.columns if c not in coord_cols]

        platform_id = (
            df["platform_id"].values
            if "platform_id" in df.columns
            else np.array(["unknown"] * len(df))
        )

        def _to_numpy(arr):
            """Convert pandas extension arrays (e.g. StringDtype) to numpy object."""
            if hasattr(arr, "dtype") and not isinstance(arr.dtype, np.dtype):
                return arr.astype(object)
            return arr

        ds = xr.Dataset(
            {col: ("point", _to_numpy(df[col].values)) for col in data_cols},
            coords={
                "lon":         ("point", df["lon"].values),
                "lat":         ("point", df["lat"].values),
                "time":        ("point", df["time"].values),
                "platform_id": ("point", _to_numpy(platform_id)),
            },
        )
        ds.attrs["data_type"]     = "insitu_observations"
        ds.attrs["platform_type"] = source_type
        ds.attrs["source"]        = "Copernicus Marine"
        return ds

    @staticmethod
    def from_altimeter(
        nc_path: Union[str, Path],
    ) -> Optional[xr.Dataset]:
        """
        Open an altimeter netCDF file as a Dataset.

        Parameters
        ----------
        nc_path : str or Path
            Path to a netCDF file.

        Returns
        -------
        xr.Dataset or None
            None if the file does not exist.
        """
        nc_path = Path(nc_path)
        if not nc_path.exists():
            logger.warning("NetCDF not found: %s", nc_path)
            return None

        try:
            ds = xr.open_dataset(nc_path)
        except Exception as exc:
            logger.warning("Could not open %s: %s", nc_path, exc)
            return None

        ds.attrs.setdefault("data_type", "altimeter")
        ds.attrs.setdefault("source",    "Copernicus")
        return ds

    @staticmethod
    def from_collocations(
        collocations: list,
    ) -> Optional[xr.Dataset]:
        """
        Convert a list of CollocatedPoint objects (step 3 output) to a Dataset.

        Each collocation becomes one record along the ``collocation`` dimension.
        SAR variables are prefixed with ``sar_`` and validation variables with
        ``val_``.

        Parameters
        ----------
        collocations : list[CollocatedPoint]
            Output from any of the collocation classes.

        Returns
        -------
        xr.Dataset or None
            None if the list is empty.
        """
        if not collocations:
            logger.warning("No collocations to convert.")
            return None

        # Collect all unique variable names across collocations
        sar_vars: set = set()
        val_vars: set = set()
        for c in collocations:
            sar_vars.update(c.sar_data.keys())
            val_vars.update(c.val_data.keys())

        data: Dict[str, tuple] = {}

        # Geometry / offset variables
        for key, getter in (
            ("sar_lon",                    lambda c: c.sar_lon),
            ("sar_lat",                    lambda c: c.sar_lat),
            ("val_lon",                    lambda c: c.val_lon),
            ("val_lat",                    lambda c: c.val_lat),
            ("spatial_distance_km",        lambda c: c.spatial_distance_km),
            ("temporal_distance_minutes",  lambda c: c.temporal_distance_minutes),
            ("val_source",                 lambda c: c.val_source),
        ):
            data[key] = ("collocation", [getter(c) for c in collocations])

        # SAR data variables
        for var in sorted(sar_vars):
            data[f"sar_{var}"] = (
                "collocation",
                [c.sar_data.get(var, np.nan) for c in collocations],
            )

        # Validation data variables
        for var in sorted(val_vars):
            data[f"val_{var}"] = (
                "collocation",
                [c.val_data.get(var, np.nan) for c in collocations],
            )

        # Collocation type provenance
        data["collocation_type"] = (
            "collocation",
            [c.collocation_type for c in collocations],
        )

        # Pixel indices — needed by patch_extractor to slice SAR arrays
        data["sar_y_idx"] = ("collocation", [c.sar_y_idx for c in collocations])
        data["sar_x_idx"] = ("collocation", [c.sar_x_idx for c in collocations])
        data["sar_scene_name"] = ("collocation", [c.sar_scene_name for c in collocations])

        coords = {
            "time":     ("collocation", [c.sar_time  for c in collocations]),
            "val_time": ("collocation", [c.val_time  for c in collocations]),
            "val_id":   ("collocation", [c.val_id or "unknown" for c in collocations]),
        }

        ds = xr.Dataset(data, coords=coords)
        ds.attrs["data_type"]        = "collocations"
        ds.attrs["num_collocations"] = len(collocations)
        return ds

    # ------------------------------------------------------------------
    # DataTree assembly
    # ------------------------------------------------------------------

    @staticmethod
    def from_scatterometer_nc(
        nc_path: Union[str, Path],
    ) -> Optional[xr.Dataset]:
        """
        Open an OSI-SAF / ASCAT scatterometer NetCDF and return a standardised
        Dataset with a flat ``point`` dimension.

        The converter handles both 1-D (along-track) and 2-D (swath) layouts
        by flattening all spatial dimensions.  Acquisition time is read from
        (in order of preference):

        1. A ``time`` variable in the file.
        2. Global attributes ``time_coverage_start`` or ``start_date``.
        3. The filename timestamp pattern ``_YYYYMMDD_HHMMSS_``.

        Only float / int data variables (excluding coordinate-like arrays
        and quality flags) are kept.

        Parameters
        ----------
        nc_path : str or Path
            Path to the scatterometer NetCDF file.

        Returns
        -------
        xr.Dataset or None
            Dataset with ``data_type="scatterometer"``, or None on failure.
        """
        nc_path = Path(nc_path)
        if not nc_path.exists():
            logger.warning("NetCDF not found: %s", nc_path)
            return None

        try:
            raw = xr.open_dataset(nc_path)
        except Exception as exc:
            logger.warning("Could not open %s: %s", nc_path, exc)
            return None

        # ------------------------------------------------------------------
        # Locate lat / lon arrays
        # ------------------------------------------------------------------
        lat_arr: Optional[np.ndarray] = None
        lon_arr: Optional[np.ndarray] = None
        for name_lat, name_lon in (
            ("lat", "lon"),
            ("latitude", "longitude"),
            ("Latitude", "Longitude"),
            ("LAT", "LON"),
        ):
            if name_lat in raw and name_lon in raw:
                lat_arr = raw[name_lat].values.ravel()
                lon_arr = raw[name_lon].values.ravel()
                break
            if name_lat in raw.coords and name_lon in raw.coords:
                lat_arr = raw[name_lat].values.ravel()
                lon_arr = raw[name_lon].values.ravel()
                break

        if lat_arr is None or lon_arr is None:
            logger.warning(
                "Could not find lat/lon in %s (available: %s)",
                nc_path.name,
                list(raw.coords) + list(raw.data_vars),
            )
            raw.close()
            return None

        n_points = len(lat_arr)

        # ------------------------------------------------------------------
        # Extract acquisition time
        # ------------------------------------------------------------------
        time_arr: Optional[np.ndarray] = None

        # 1. Per-observation time variable (1-D or 2-D, same spatial dims)
        for tname in ("time", "Time", "TIME"):
            if tname in raw:
                try:
                    t_vals = pd.to_datetime(raw[tname].values.ravel())
                    if len(t_vals) == n_points:
                        time_arr = t_vals.values
                    elif len(t_vals) == 1:
                        time_arr = np.full(n_points, t_vals[0])
                    break
                except Exception:
                    pass

        # 2. Global attribute
        if time_arr is None:
            for attr in ("time_coverage_start", "start_date", "firstMeasurementTime"):
                val = raw.attrs.get(attr)
                if val:
                    try:
                        t0 = np.datetime64(pd.Timestamp(val).tz_convert(None) if hasattr(pd.Timestamp(val), 'tz') and pd.Timestamp(val).tzinfo else pd.Timestamp(val), "ns")
                        time_arr = np.full(n_points, t0)
                        break
                    except Exception:
                        pass

        # 3. Filename pattern  OASWC12_YYYYMMDD_HHMMSS_
        if time_arr is None:
            m = re.search(r"_(\d{8})_(\d{6})_", nc_path.stem)
            if m:
                try:
                    t0 = np.datetime64(
                        pd.Timestamp(m.group(1) + "T" + m.group(2), format="%Y%m%dT%H%M%S"), "ns"
                    )
                    time_arr = np.full(n_points, t0)
                except Exception:
                    pass

        if time_arr is None:
            logger.warning(
                "Could not determine acquisition time for %s; using NaT.",
                nc_path.name,
            )
            time_arr = np.full(n_points, np.datetime64("NaT", "ns"))

        # ------------------------------------------------------------------
        # Collect numeric data variables (skip lat/lon/time/flag arrays)
        # ------------------------------------------------------------------
        _skip = {
            "lat", "latitude", "Latitude", "LAT",
            "lon", "longitude", "Longitude", "LON",
            "time", "Time", "TIME",
        }
        data_vars: Dict[str, tuple] = {}
        for vname, da in raw.data_vars.items():
            if vname in _skip:
                continue
            if da.dtype.kind not in ("f", "i", "u"):   # floats and integers only
                continue
            flat = da.values.ravel()
            if len(flat) != n_points:
                continue   # different spatial grid — skip
            data_vars[vname] = ("point", flat.astype(float))

        if not data_vars:
            logger.warning(
                "No usable data variables found in %s.", nc_path.name
            )
            raw.close()
            return None

        ds = xr.Dataset(
            data_vars,
            coords={
                "lon":  ("point", lon_arr),
                "lat":  ("point", lat_arr),
                "time": ("point", time_arr),
            },
        )
        ds.attrs["data_type"] = "scatterometer"
        ds.attrs["source"]    = "OSI-SAF ASCAT"
        ds.attrs["filename"]  = nc_path.name

        raw.close()
        return ds

    @staticmethod
    def to_datatree(
        datasets: Dict[str, Optional[xr.Dataset]],
    ) -> xr.DataTree:
        """
        Combine multiple Datasets into an xarray DataTree.

        Parameters
        ----------
        datasets : dict
            Mapping of node names to Datasets.
            e.g. {"sar": ds_sar, "insitu/buoy": ds_buoy, "collocations": ds_coll}
            Keys may use "/" to create nested nodes.

        Returns
        -------
        xr.DataTree
            Hierarchical tree with one child node per entry.
        """
        clean = {
            f"/{k.lstrip('/')}": ds
            for k, ds in datasets.items()
            if ds is not None
        }
        return xr.DataTree.from_dict(clean)

    # ------------------------------------------------------------------
    # Export helpers (step 4)
    # ------------------------------------------------------------------

    @staticmethod
    def to_dataframe(collocations: list) -> pd.DataFrame:
        """
        Convert a list of CollocatedPoint objects to a flat DataFrame.

        Convenient for CSV export and matplotlib-based visualisation (step 5).

        Parameters
        ----------
        collocations : list[CollocatedPoint]

        Returns
        -------
        pd.DataFrame
        """
        if not collocations:
            return pd.DataFrame()

        rows = []
        for c in collocations:
            row = {
                "sar_lon":                   c.sar_lon,
                "sar_lat":                   c.sar_lat,
                "sar_time":                  c.sar_time,
                "val_lon":                   c.val_lon,
                "val_lat":                   c.val_lat,
                "val_time":                  c.val_time,
                "spatial_distance_km":       c.spatial_distance_km,
                "temporal_distance_minutes": c.temporal_distance_minutes,
                "val_source":                c.val_source,
                "val_id":                    c.val_id,
            }
            for var, value in c.sar_data.items():
                row[f"sar_{var}"] = value
            for var, value in c.val_data.items():
                row[f"val_{var}"] = value
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Step 2 high-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def from_sar_l2_ocn_safe(
        safe_dir: Union[str, Path],
    ) -> Optional[xr.Dataset]:
        """
        Open SAR L2 OCN data from one Sentinel-1 SAFE directory and return a
        standardised Dataset.

        Automatically detects mode by inspecting the SAFE directory name:
        - If directory name contains "WV": reads WV mode oswHs point measurements
        - Otherwise: attempts to extract RVL data (keeping 2D grid structure);
          returns None if RVL not available

        Parameters
        ----------
        safe_dir : str or Path
            Path to a ``*.SAFE`` directory.

        Returns
        -------
        xr.Dataset or None
            Dataset with oswHs points (WV) or RVL grids (IW/EW/SM), or None if
            no suitable data found.
        """
        safe_dir = Path(safe_dir)
        safe_name = safe_dir.name.upper()

        # Detect mode from SAFE directory name
        if "WV" in safe_name:
            return DataTreeConverter.from_sar_l2_ocn_wv_safe(safe_dir)
        else:
            return DataTreeConverter._from_sar_l2_ocn_iw_safe(safe_dir)

    @staticmethod
    def from_sar_l2_ocn_wv_safe(
        safe_dir: Union[str, Path],
    ) -> Optional[xr.Dataset]:
        """
        Open Wave Mode (WV) data from one Sentinel-1 SAFE directory and extract
        oswHs (Ocean Surface Wave Height) point measurements.

        The WV mode produces multiple measurement files (~16 per SAFE product),
        each containing a 1×1 oswHs point measurement. This method extracts
        oswHs from all .nc files in the measurement folder and creates a
        point-geometry Dataset with dimension ``point``.

        Each point's time is extracted from the filename timestamp; coordinates
        are oswLat/oswLon (promoting them to named dimensions).

        Parameters
        ----------
        safe_dir : str or Path
            Path to a WV mode ``*.SAFE`` directory.

        Returns
        -------
        xr.Dataset or None
            None if no measurement files are found or opening fails.
        """
        safe_dir = Path(safe_dir)
        measurement_dir = safe_dir / "measurement"
        if not measurement_dir.exists():
            logger.warning("No measurement/ directory in %s", safe_dir)
            return None

        # List all WV measurement files (pattern: s1a-wv*-ocn-*.nc or s1b-wv*-ocn-*.nc)
        wv_files = sorted(
            f for f in measurement_dir.glob("*.nc")
            if ("-wv" in f.name.lower() or "-wv" in safe_dir.name.lower())
            and "-ocn-" in f.name
        )
        if not wv_files:
            logger.warning("No WV measurement files found in %s", measurement_dir)
            return None

        # Extract oswHs point measurements from all files
        point_lons = []
        point_lats = []
        point_hs = []
        point_times = []
        file_names = []

        for nc_path in wv_files:
            try:
                ds_raw = xr.open_dataset(nc_path)
            except Exception as exc:
                logger.warning("Could not open %s: %s", nc_path, exc)
                continue

            try:
                # Extract coordinates as scalars (1×1 grid)
                lon = float(ds_raw["oswLon"].values.item())
                lat = float(ds_raw["oswLat"].values.item())

                # Extract oswHs from first partition
                hs_val = ds_raw["oswHs"].isel(oswPartitions=0).values.item()
                hs = float(hs_val) if np.isfinite(hs_val) else np.nan

                # Acquisition time from filename (format: YYYYMMDDtHHMMSS)
                m = re.search(r"(\d{8}t\d{6})", nc_path.stem, re.IGNORECASE)
                if m:
                    acq_time = pd.to_datetime(m.group(1), format="%Y%m%dT%H%M%S")
                else:
                    # Fallback to global attribute
                    time_str = ds_raw.attrs.get("firstMeasurementTime")
                    acq_time = pd.to_datetime(time_str) if time_str else None

                if acq_time is not None:
                    point_lons.append(lon)
                    point_lats.append(lat)
                    point_hs.append(hs)
                    point_times.append(np.datetime64(
                        acq_time.tz_convert(None) if acq_time.tzinfo else acq_time, "ns"
                    ))
                    file_names.append(nc_path.name)

            except Exception as exc:
                logger.debug("Could not extract oswHs from %s: %s", nc_path.name, exc)
            finally:
                ds_raw.close()

        if not point_hs:
            logger.warning("No valid oswHs data extracted from %s", measurement_dir)
            return None

        # Create Dataset with point dimension
        data_vars = {
            "oswHs": (["point"], point_hs),
        }

        coords = {
            "lon": (["point"], point_lons),
            "lat": (["point"], point_lats),
            "time": (["point"], point_times),
            "filename": (["point"], file_names),
        }

        ds = xr.Dataset(data_vars, coords=coords)
        ds.attrs["data_type"] = "sar_l2_ocn"
        ds.attrs["source"] = "Sentinel-1"
        ds.attrs["safe_dir"] = safe_dir.name
        ds.attrs["swath_mode"] = "WV"
        ds.attrs["measurement_type"] = "oswHs"
        ds.attrs["num_points"] = len(point_hs)

        logger.info(
            "Extracted %d oswHs points from WV product %s",
            len(point_hs), safe_dir.name
        )
        return ds

    @staticmethod
    def _extract_rvl_grid_data(
        measurement_dir: Path,
        safe_dir: Union[str, Path],
        flatten_to_points: bool = False,
    ) -> Optional[xr.Dataset]:
        """
        Extract RVL (Radial Velocity Linesight) data from measurement directory.

        RVL is a 13×13 grid measurement in SAR products. By default, this function
        keeps the grid structure (rvlLat × rvlLon dimensions). If flatten_to_points=True,
        the grid is flattened to points (169 points per file) for collocation.

        Parameters
        ----------
        measurement_dir : Path
            Path to the measurement/ directory within a SAFE archive.
        safe_dir : str or Path
            Path to the SAFE directory (used for logging and attributes).
        flatten_to_points : bool, optional
            If True, flatten 13×13 grids to 169 points with point dimension (default: False).
            If False, keep 2D grid structure with (rvlLat, rvlLon) dimensions.

        Returns
        -------
        xr.Dataset or None
            Dataset with RVL variables and coordinates, or None if no RVL data found.
        """
        safe_dir = Path(safe_dir)

        # Discover measurement files containing RVL data
        # Pattern: files with "-ocn-" and optionally "-wv" (works for all modes)
        rvl_files = sorted(
            f for f in measurement_dir.glob("*.nc")
            if "-ocn-" in f.name
        )
        if not rvl_files:
            return None

        if not flatten_to_points:
            # Extract RVL data in 2D grid form (for IW/EW/SM modes)
            try:
                ds_raw = xr.open_dataset(rvl_files[0])
            except Exception as exc:
                logger.debug("Could not open %s: %s", rvl_files[0], exc)
                return None

            try:
                # Check if RVL data exists
                if "rvlRadVel" not in ds_raw:
                    ds_raw.close()
                    return None

                # Extract RVL grid arrays (keep as 2D)
                rvl_radvel_full = ds_raw["rvlRadVel"].values  # May be (rvlLat, rvlLon) or (rvlLat, rvlLon, swath)
                rvl_lats_full = ds_raw["rvlLat"].values
                rvl_lons_full = ds_raw["rvlLon"].values

                # If 3D data (with swaths), use first swath only for collocation compatibility
                if rvl_radvel_full.ndim == 3:
                    rvl_radvel = rvl_radvel_full[:, :, 0]
                    rvl_lats = rvl_lats_full[:, :, 0]
                    rvl_lons = rvl_lons_full[:, :, 0]
                else:
                    rvl_radvel = rvl_radvel_full
                    rvl_lats = rvl_lats_full
                    rvl_lons = rvl_lons_full

                rvl_heading_full = (
                    ds_raw["rvlHeading"].values
                    if "rvlHeading" in ds_raw
                    else None
                )
                if rvl_heading_full is not None:
                    rvl_heading = (
                        rvl_heading_full[:, :, 0] if rvl_heading_full.ndim == 3
                        else rvl_heading_full
                    )
                else:
                    rvl_heading = np.full_like(rvl_radvel, np.nan)

                rvl_incidence_full = (
                    ds_raw["rvlIncidenceAngle"].values
                    if "rvlIncidenceAngle" in ds_raw
                    else None
                )
                if rvl_incidence_full is not None:
                    rvl_incidence = (
                        rvl_incidence_full[:, :, 0] if rvl_incidence_full.ndim == 3
                        else rvl_incidence_full
                    )
                else:
                    rvl_incidence = np.full_like(rvl_radvel, np.nan)

                # Get acquisition time (scalar for grid)
                time_str = ds_raw.attrs.get("firstMeasurementTime")
                if time_str:
                    acq_time = pd.to_datetime(time_str)
                    acq_time_ns = np.datetime64(
                        acq_time.tz_convert(None) if acq_time.tzinfo else acq_time, "ns"
                    )
                else:
                    m = re.search(r"(\d{8}t\d{6})", rvl_files[0].stem, re.IGNORECASE)
                    if m:
                        acq_time = pd.to_datetime(m.group(1), format="%Y%m%dT%H%M%S")
                        acq_time_ns = np.datetime64(
                            acq_time.tz_convert(None) if acq_time.tzinfo else acq_time, "ns"
                        )
                    else:
                        acq_time_ns = np.datetime64("NaT", "ns")

                # Infer dimension names from rvlLat shape
                dims = ds_raw["rvlLat"].dims if hasattr(ds_raw["rvlLat"], "dims") else ("rvlLat", "rvlLon")

                # Create Dataset with 2D grid structure
                data_vars = {
                    "rvlRadVel": (dims, rvl_radvel),
                    "rvlHeading": (dims, rvl_heading),
                    "rvlIncidenceAngle": (dims, rvl_incidence),
                }

                coords = {
                    "lon": (dims, rvl_lons),
                    "lat": (dims, rvl_lats),
                    "time": acq_time_ns,
                }

                ds = xr.Dataset(data_vars, coords=coords)
                ds.attrs["data_type"] = "sar_l2_ocn"
                ds.attrs["source"] = "Sentinel-1"
                ds.attrs["safe_dir"] = safe_dir.name
                ds.attrs["measurement_type"] = "rvl"
                ds.attrs["grid_shape"] = rvl_radvel.shape

                logger.info(
                    "Extracted RVL grid %s from product %s",
                    rvl_radvel.shape, safe_dir.name
                )
                return ds

            except Exception as exc:
                logger.debug("Could not extract RVL grid from %s: %s", rvl_files[0].name, exc)
                return None
            finally:
                ds_raw.close()

        else:
            # Flatten RVL grids to points (for WV mode backward compat)
            point_lons = []
            point_lats = []
            point_radvel = []
            point_heading = []
            point_incidence = []
            point_times = []
            file_names = []

            for nc_path in rvl_files:
                try:
                    ds_raw = xr.open_dataset(nc_path)
                except Exception as exc:
                    logger.debug("Could not open %s: %s", nc_path, exc)
                    continue

                try:
                    # Check if RVL data exists in this file
                    if "rvlRadVel" not in ds_raw:
                        continue

                    # Extract RVL grid arrays and flatten to 1D
                    rvl_radvel = ds_raw["rvlRadVel"].values.ravel()
                    rvl_lats = ds_raw["rvlLat"].values.ravel()
                    rvl_lons = ds_raw["rvlLon"].values.ravel()

                    rvl_heading = (
                        ds_raw["rvlHeading"].values.ravel()
                        if "rvlHeading" in ds_raw
                        else np.full_like(rvl_radvel, np.nan)
                    )
                    rvl_incidence = (
                        ds_raw["rvlIncidenceAngle"].values.ravel()
                        if "rvlIncidenceAngle" in ds_raw
                        else np.full_like(rvl_radvel, np.nan)
                    )

                    # Get acquisition time
                    m = re.search(r"(\d{8}t\d{6})", nc_path.stem, re.IGNORECASE)
                    if m:
                        acq_time = pd.to_datetime(m.group(1), format="%Y%m%dT%H%M%S")
                        acq_time_ns = np.datetime64(
                            acq_time.tz_convert(None) if acq_time.tzinfo else acq_time, "ns"
                        )
                    else:
                        acq_time_ns = np.datetime64("NaT", "ns")

                    # Add all RVL points from this file
                    n_points = len(rvl_lons)
                    point_lons.extend(rvl_lons)
                    point_lats.extend(rvl_lats)
                    point_radvel.extend(rvl_radvel)
                    point_heading.extend(rvl_heading)
                    point_incidence.extend(rvl_incidence)
                    point_times.extend([acq_time_ns] * n_points)
                    file_names.extend([nc_path.name] * n_points)

                except Exception as exc:
                    logger.debug("Could not extract RVL from %s: %s", nc_path.name, exc)
                finally:
                    ds_raw.close()

            if not point_radvel:
                logger.debug("No RVL data found in %s", measurement_dir)
                return None

            # Create Dataset with point dimension (flattened RVL grids)
            data_vars = {
                "rvlRadVel": (["point"], point_radvel),
                "rvlHeading": (["point"], point_heading),
                "rvlIncidenceAngle": (["point"], point_incidence),
            }

            coords = {
                "lon": (["point"], point_lons),
                "lat": (["point"], point_lats),
                "time": (["point"], point_times),
                "filename": (["point"], file_names),
            }

            ds = xr.Dataset(data_vars, coords=coords)
            ds.attrs["data_type"] = "sar_l2_ocn"
            ds.attrs["source"] = "Sentinel-1"
            ds.attrs["safe_dir"] = safe_dir.name
            ds.attrs["measurement_type"] = "rvl"
            ds.attrs["num_points"] = len(point_radvel)

            logger.info(
                "Extracted %d RVL points from product %s",
                len(point_radvel), safe_dir.name
            )
            return ds

    @staticmethod
    def _extract_rvl_from_wv_safe(
        safe_dir: Union[str, Path],
    ) -> Optional[xr.Dataset]:
        """
        Extract RVL (Radial Velocity Linesight) data from WV mode SAFE directory.

        RVL is a 13×13 grid measurement in WV mode products. This function flattens
        the grid to points (169 points per file) with coordinates lon/lat and variables
        rvlRadVel, rvlHeading, rvlIncidenceAngle, allowing it to be processed by the
        point-based collocation algorithm.

        Parameters
        ----------
        safe_dir : str or Path
            Path to a WV mode ``*.SAFE`` directory.

        Returns
        -------
        xr.Dataset or None
            None if no RVL data is found or opening fails.
        """
        safe_dir = Path(safe_dir)
        measurement_dir = safe_dir / "measurement"
        if not measurement_dir.exists():
            return None

        # Use helper to extract RVL data, flattened to points for backward compatibility
        ds = DataTreeConverter._extract_rvl_grid_data(
            measurement_dir, safe_dir, flatten_to_points=True
        )
        
        if ds is not None:
            ds.attrs["swath_mode"] = "WV"
        
        return ds


    @staticmethod
    def _from_sar_l2_ocn_iw_safe(
        safe_dir: Union[str, Path],
    ) -> Optional[xr.Dataset]:
        """
        Extract RVL (Radial Velocity Linesight) or OWI data from one Sentinel-1 IW/EW/SM
        mode SAFE directory and return a standardised Dataset.

        For currents validation, this function prioritizes RVL data (Radial Velocity
        Linesight), which is a 13×13 grid measurement available in some IW/EW/SM products.
        RVL is returned as a 2D grid Dataset (rvlLat × rvlLon dimensions).

        If RVL data is not available, returns None (no OWI fallback).

        Parameters
        ----------
        safe_dir : str or Path
            Path to an IW/EW/SM mode ``*.SAFE`` directory.

        Returns
        -------
        xr.Dataset or None
            RVL Dataset if RVL data is found; None otherwise.
        """
        safe_dir = Path(safe_dir)
        measurement_dir = safe_dir / "measurement"
        if not measurement_dir.exists():
            logger.debug("No measurement/ directory in %s", safe_dir)
            return None

        # Attempt to extract RVL data (keeping 2D grid structure)
        ds_rvl = DataTreeConverter._extract_rvl_grid_data(
            measurement_dir, safe_dir, flatten_to_points=False
        )

        if ds_rvl is not None:
            ds_rvl.attrs["swath_mode"] = "IW/EW/SM"
            logger.info("Extracted RVL data from IW/EW/SM product %s", safe_dir.name)
            return ds_rvl

        # No RVL data found; return None (no OWI fallback)
        logger.debug(
            "No RVL data found in IW/EW/SM product %s (OWI fallback disabled)",
            safe_dir.name,
        )
        return None


    @staticmethod
    def convert_downloaded_data(
        base_dir: Union[str, Path],
    ) -> Optional[xr.DataTree]:
        """
        Auto-discover all downloaded files inside *base_dir* and convert them
        to a hierarchical DataTree, which is saved to
        ``<base_dir>/datatree.nc``.

        Discovery rules
        ---------------
        - ``S1_L2_OCN/*.SAFE``        → ``sar/<SAFE-name>`` nodes
        - ``copernicus_insitu/*.csv``  → ``validation/<stem>`` nodes
        - ``osi_saf_winds/*.nc``       → ``validation/osi_saf_winds/<stem>`` nodes
        - ``scatterometer/*.nc``       → ``validation/scatterometer/<stem>`` nodes
        - ``altimeter/*.nc``           → ``validation/altimeter/<stem>`` nodes

        Parameters
        ----------
        base_dir : str or Path
            Root directory produced by the download step (contains
            ``download_metadata.json``).

        Returns
        -------
        xr.DataTree or None
            None if no data files were found.
        """
        base_dir = Path(base_dir)
        datasets: Dict[str, xr.Dataset] = {}

        # SAR L2_OCN SAFE directories
        sar_dir = base_dir / "S1_L2_OCN"
        if sar_dir.exists():
            for safe_dir in sorted(d for d in sar_dir.iterdir()
                                   if d.is_dir() and d.suffix == ".SAFE"):
                ds = DataTreeConverter.from_sar_l2_ocn_safe(safe_dir)
                if ds is not None:
                    datasets[f"sar/{safe_dir.name}"] = ds
                    logger.info("Converted SAR SAFE: %s", safe_dir.name)

        # In-situ CSV (Copernicus Marine)
        insitu_dir = base_dir / "copernicus_insitu"
        if insitu_dir.exists():
            for csv_path in sorted(insitu_dir.glob("*.csv")):
                ds = DataTreeConverter.from_insitu_csv(csv_path, source_type="insitu")
                if ds is not None:
                    datasets[f"validation/{csv_path.stem}"] = ds
                    logger.info("Converted in-situ CSV: %s", csv_path.name)

        # Scatterometer / OSI-SAF winds (standardised to point dimension)
        for subdir_name in ("osi_saf_winds", "scatterometer"):
            subdir = base_dir / subdir_name
            if subdir.exists():
                for nc_path in sorted(subdir.glob("*.nc")):
                    ds = DataTreeConverter.from_scatterometer_nc(nc_path)
                    if ds is not None:
                        datasets[f"validation/{subdir_name}/{nc_path.stem}"] = ds
                        logger.info("Converted %s (scatterometer): %s", subdir_name, nc_path.name)

        # Altimeter NetCDF products (kept as raw dataset)
        subdir = base_dir / "altimeter"
        if subdir.exists():
            for nc_path in sorted(subdir.glob("*.nc")):
                ds = DataTreeConverter.from_altimeter(nc_path)
                if ds is not None:
                    datasets[f"validation/altimeter/{nc_path.stem}"] = ds
                    logger.info("Converted altimeter: %s", nc_path.name)

        if not datasets:
            logger.warning("No convertible data found in %s", base_dir)
            return None

        tree = DataTreeConverter.to_datatree(datasets)

        out_path = base_dir / "datatree.nc"
        tree.to_netcdf(out_path)
        logger.info("DataTree saved to %s (%d nodes)", out_path, len(datasets))

        return tree
