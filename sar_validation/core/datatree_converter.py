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
        Open the OWI (Ocean Wind field Inversion) data from one Sentinel-1
        SAFE directory and return a standardised Dataset.

        The merged OCN file (``*-ocn-*``, not OSW subswath files) is used.
        Longitude and latitude are promoted to named coordinates ``lon`` and
        ``lat``; acquisition time is read from the ``firstMeasurementTime``
        global attribute (falling back to the filename timestamp).

        Parameters
        ----------
        safe_dir : str or Path
            Path to a ``*.SAFE`` directory.

        Returns
        -------
        xr.Dataset or None
            None if no suitable measurement file is found or opening fails.
        """
        safe_dir = Path(safe_dir)
        measurement_dir = safe_dir / "measurement"
        if not measurement_dir.exists():
            logger.warning("No measurement/ directory in %s", safe_dir)
            return None

        # Pick the merged OCI file (contains "-ocn-"; OSW subswaths have "-osw-")
        ocn_files = sorted(f for f in measurement_dir.glob("*.nc") if "-ocn-" in f.name)
        if not ocn_files:
            logger.warning("No OCN measurement file found in %s", measurement_dir)
            return None

        nc_path = ocn_files[0]
        try:
            ds_raw = xr.open_dataset(nc_path)
        except Exception as exc:
            logger.warning("Could not open %s: %s", nc_path, exc)
            return None

        # Acquisition time
        time_str = ds_raw.attrs.get("firstMeasurementTime")
        if time_str:
            acq_time = pd.to_datetime(time_str)
        else:
            m = re.search(r"(\d{8}t\d{6})", nc_path.stem, re.IGNORECASE)
            acq_time = pd.to_datetime(m.group(1), format="%Y%m%dT%H%M%S") if m else None

        # Collect all OWI variables on the (owiAzSize, owiRaSize) grid
        owi_dims = ("owiAzSize", "owiRaSize")
        data_vars = {
            k: (["y", "x"], ds_raw[k].values)
            for k in ds_raw.data_vars
            if ds_raw[k].dims == owi_dims and k not in ("owiLon", "owiLat")
        }

        if not data_vars:
            logger.warning("No OWI grid variables found in %s", nc_path.name)
            return None

        coords: Dict = {
            "lon":  (["y", "x"], ds_raw["owiLon"].values),
            "lat":  (["y", "x"], ds_raw["owiLat"].values),
        }
        if acq_time is not None:
            coords["time"] = np.datetime64(acq_time.tz_convert(None) if acq_time.tzinfo else acq_time, "ns")

        ds = xr.Dataset(data_vars, coords=coords)
        ds.attrs["data_type"] = "sar_l2_ocn"
        ds.attrs["source"] = "Sentinel-1"
        ds.attrs["safe_dir"] = safe_dir.name
        if time_str:
            ds.attrs["firstMeasurementTime"] = time_str

        ds_raw.close()
        return ds

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
