"""
Convert validation data to xarray.DataTree format.

Step 2 of the validation pipeline.

Provides converters for:
  - SAR L2_OCN data          → standardised Dataset
  - In-situ CSV (Copernicus) → point-geometry Dataset
  - Altimeter netCDF         → Dataset
  - Collocated results       → Dataset  (step 4 output)

And a ``to_datatree()`` helper to assemble multiple Datasets into one
hierarchical DataTree.
"""

from __future__ import annotations

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

        ds = xr.Dataset(
            {col: ("point", df[col].values) for col in data_cols},
            coords={
                "lon":         ("point", df["lon"].values),
                "lat":         ("point", df["lat"].values),
                "time":        ("point", df["time"].values),
                "platform_id": ("point", platform_id),
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

        ds = xr.open_dataset(nc_path)
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

        coords = {
            "time":     ("collocation", [c.sar_time  for c in collocations]),
            "val_time": ("collocation", [c.val_time  for c in collocations]),
            "val_id":   ("collocation", [c.val_id or "unknown" for c in collocations]),
        }

        ds = xr.Dataset(data, coords=coords)
        ds.attrs["data_type"]       = "collocations"
        ds.attrs["num_collocations"] = len(collocations)
        return ds

    # ------------------------------------------------------------------
    # DataTree assembly
    # ------------------------------------------------------------------

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
