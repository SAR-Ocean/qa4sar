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
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import logging

from ._cf_metadata import apply_cf_metadata

logger = logging.getLogger(__name__)

__all__ = ["DataTreeConverter"]

# OSI-SAF/ASCAT `wvc_quality_flag` bit meanings (see the file's own
# flag_masks/flag_meanings attrs) that indicate the wind retrieval itself is
# invalid or contaminated. Wind-vector cells carrying any of these bits are
# dropped in ``DataTreeConverter.from_scatterometer_nc`` before they ever
# reach collocation — otherwise they show up as spurious points over land/ice
# with no (or a bogus) retrieved value.
_ASCAT_REJECT_FLAGS = {
    "some_portion_of_wvc_is_over_land",
    "some_portion_of_wvc_is_over_ice",
    "wind_inversion_not_successful",
    "not_enough_good_sigma0_for_wind_retrieval",
    "distance_to_gmf_too_large",
}


def _subset_point_ds(
    ds: xr.Dataset,
    *,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    t_start,
    t_end,
    buffer_km: float,
    time_tolerance_minutes: float,
) -> Optional[xr.Dataset]:
    """
    Subset a point-geometry Dataset to a recipe's domain plus tolerances.

    Keeps only points inside the geographic bounding box expanded by
    ``buffer_km`` (converted to degrees with the same ~55 km/deg convention
    used by the collocation pre-filters) and inside the temporal window
    expanded by ``time_tolerance_minutes`` on both sides. Points with a NaT
    timestamp are kept — they cannot be proven out-of-window.

    Datasets without a ``point`` dimension or without ``lon``/``lat``
    coordinates are returned unchanged.

    Returns
    -------
    xr.Dataset or None
        The subset Dataset, or None if no points survive the filter.
    """
    if "point" not in ds.dims or "lon" not in ds.coords or "lat" not in ds.coords:
        return ds

    deg_buf = buffer_km / 55.0
    lon = ds["lon"].values
    lat = ds["lat"].values
    mask = (
        (lon >= min_lon - deg_buf) & (lon <= max_lon + deg_buf)
        & (lat >= min_lat - deg_buf) & (lat <= max_lat + deg_buf)
    )

    if "time" in ds.coords:
        tol = pd.Timedelta(minutes=time_tolerance_minutes)
        t0 = (pd.Timestamp(t_start) - tol).to_datetime64()
        t1 = (pd.Timestamp(t_end) + tol).to_datetime64()
        times = pd.to_datetime(ds["time"].values)
        if times.tz is not None:
            times = times.tz_localize(None)
        times = times.values
        in_window = (times >= t0) & (times <= t1)
        mask &= in_window | pd.isna(times)

    n_total = ds.sizes["point"]
    n_kept = int(mask.sum())
    if n_kept == 0:
        return None
    if n_kept == n_total:
        return ds
    logger.info(
        "Domain filter kept %d/%d points (%.1f%%)",
        n_kept, n_total, 100.0 * n_kept / n_total,
    )
    return ds.isel(point=np.flatnonzero(mask))


def _build_subset_kwargs(recipe) -> Dict[str, Any]:
    """
    Derive the widest filter envelope for :func:`_subset_point_ds` from a
    recipe: its geographic/temporal bounds plus the most permissive spatial
    and temporal collocation tolerance any source could use — so the filter
    can never discard a point that some collocation pass would have matched.
    """
    from .recipe import DEFAULT_LAYER_TYPE_SPECS

    cfg = recipe.config
    coll = cfg.collocation
    pvl = coll.point_vs_layer

    layer_specs = dict(DEFAULT_LAYER_TYPE_SPECS)
    if coll.layer_vs_layer is not None:
        for key, spec in coll.layer_vs_layer.layer_type_specs.items():
            layer_specs[key] = {**layer_specs.get(key, {}), **spec}
    # Per-source collocation overrides can also raise the tolerances
    override_specs = [
        src.collocation_kwargs
        for src in cfg.validation_sources
        if src.collocation_kwargs
    ]
    all_specs = list(layer_specs.values()) + override_specs

    buffer_km = max(
        pvl.aggregation_window_km,
        pvl.spatial_tolerance_km,
        coll.sar_footprint_radius_km,
        *(
            float(spec["aggregation_window_km"])
            for spec in all_specs
            if "aggregation_window_km" in spec
        ),
        *(
            float(spec["spatial_tolerance_km"])
            for spec in all_specs
            if "spatial_tolerance_km" in spec
        ),
    )
    time_tolerance_minutes = max(
        pvl.time_tolerance_minutes,
        *(
            float(spec["time_tolerance_minutes"])
            for spec in all_specs
            if "time_tolerance_minutes" in spec
        ),
    )

    b = cfg.geographic_bounds
    t = cfg.temporal_bounds
    return {
        "min_lon": b.min_lon, "max_lon": b.max_lon,
        "min_lat": b.min_lat, "max_lat": b.max_lat,
        "t_start": t.start, "t_end": t.end,
        "buffer_km": buffer_km,
        "time_tolerance_minutes": time_tolerance_minutes,
    }


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

        # Provenance columns present in Copernicus Marine in-situ exports;
        # captured before the wide-format pivot drops them.
        doi = None
        insitu_institution = None
        for col, target in (("product_doi", "doi"), ("doi", "doi"), ("institution", "institution")):
            if col in df.columns:
                vals = df[col].dropna()
                if not vals.empty:
                    if target == "doi" and doi is None:
                        doi = str(vals.iloc[0])
                    elif target == "institution" and insitu_institution is None:
                        insitu_institution = str(vals.iloc[0])

        # ------------------------------------------------------------------
        # Copernicus Marine in-situ files use a long format where each row
        # holds one variable code ("WSPD", "EWCT", …) in a ``variable``
        # column and the measurement in a ``value`` column.  Pivot to wide
        # format so that each variable code becomes its own column — this is
        # what the collocation and statistics modules expect.
        # ------------------------------------------------------------------
        if "variable" in df.columns and "value" in df.columns:
            pivot_id_cols = [
                c for c in ("platform_id", "platform_type", "time", "lon", "lat", "depth")
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

        coord_cols = {"lon", "lat", "time", "platform_id", "platform_type"}
        data_cols  = [c for c in df.columns if c not in coord_cols]

        platform_id = (
            df["platform_id"].values
            if "platform_id" in df.columns
            else np.array(["unknown"] * len(df))
        )

        # Per-point platform type, e.g. "mooring"/"buoy"/"drifter"/etc. — a
        # single Copernicus in-situ CSV can mix multiple platform types (see
        # its ``platform_type`` column, raw codes like "MO"/"DB"/"AD"), so
        # this is tracked per point rather than as a single Dataset-level
        # value. Falls back to *source_type* when the raw code is missing or
        # unrecognised (e.g. hand-built CSVs without a platform_type column).
        from ..downloaders.insitu_downloader import PLATFORM_CODE_TO_SOURCE_TYPE

        if "platform_type" in df.columns:
            platform_type = (
                df["platform_type"]
                .map(PLATFORM_CODE_TO_SOURCE_TYPE)
                .fillna(source_type)
                .values
            )
        else:
            platform_type = np.array([source_type] * len(df))

        def _to_numpy(arr):
            """Convert pandas extension arrays (e.g. StringDtype) to numpy object."""
            if hasattr(arr, "dtype") and not isinstance(arr.dtype, np.dtype):
                return arr.astype(object)
            return arr

        ds = xr.Dataset(
            {col: ("point", _to_numpy(df[col].values)) for col in data_cols},
            coords={
                "lon":           ("point", df["lon"].values),
                "lat":           ("point", df["lat"].values),
                "time":          ("point", df["time"].values),
                "platform_id":   ("point", _to_numpy(platform_id)),
                "platform_type": ("point", _to_numpy(platform_type)),
            },
        )
        if doi:
            ds.attrs["references"] = doi
        if insitu_institution:
            ds.attrs["institution"] = insitu_institution
        apply_cf_metadata(ds, "insitu", {
            "platform_id":   {"long_name": "platform identifier (Copernicus Marine)"},
            "platform_type": {"long_name": "platform category (mooring, buoy, drifter, ...)"},
        })

        ds.attrs["data_type"]     = "insitu_observations"
        ds.attrs["platform_type"] = source_type
        ds.attrs["source"]        = "Copernicus Marine"
        return ds

    @staticmethod
    def from_altimeter(
        nc_path: Union[str, Path],
    ) -> Optional[xr.Dataset]:
        """
        Open a Copernicus Marine along-track altimeter NetCDF (L3 SWH product)
        and return a standardised Dataset with a flat ``point`` dimension,
        matching the layout produced by ``from_scatterometer_nc``.

        These files carry a single ``time`` dimension, with ``latitude``,
        ``longitude`` and every variable (``VAVH``, ``VAVH_UNFILTERED``, and
        either ``WIND_SPEED`` (1 Hz) or ``VAVH_UNCERTAINTY`` (5 Hz)) all
        indexed purely by ``time`` — i.e. a flat along-track point series.

        The raw ``WIND_SPEED`` variable is renamed to the canonical ``WSPD``
        code (matching the in-situ and scatterometer converters) so all
        wind-speed sources are compared in the same validation section.

        Parameters
        ----------
        nc_path : str or Path
            Path to the altimeter NetCDF file.

        Returns
        -------
        xr.Dataset or None
            Dataset with ``data_type="altimeter"``, or None on failure.
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

        if "latitude" not in raw or "longitude" not in raw:
            logger.warning(
                "Could not find latitude/longitude in %s (available: %s)",
                nc_path.name, list(raw.coords) + list(raw.data_vars),
            )
            raw.close()
            return None

        lat_arr = raw["latitude"].values.ravel().astype(float)
        lon_arr = raw["longitude"].values.ravel().astype(float)
        # Files use 0-360 degrees_east; normalize to -180..180.
        lon_arr = ((lon_arr + 180) % 360) - 180

        if "time" in raw:
            time_arr = pd.to_datetime(raw["time"].values.ravel()).values
        else:
            logger.warning("Could not find time in %s; using NaT.", nc_path.name)
            time_arr = np.full(len(lat_arr), np.datetime64("NaT", "ns"))

        n_points = len(lat_arr)

        # 1 Hz files carry WIND_SPEED; 5 Hz files carry VAVH_UNCERTAINTY
        # instead — this also doubles as the frequency signal used by the
        # collocation step to pick a resolution-appropriate aggregation
        # window (7km at 1 Hz vs 1.4km at 5 Hz).
        frequency = "5hz" if "VAVH_UNCERTAINTY" in raw else "1hz"

        _skip = {"latitude", "longitude", "time"}
        # CMEMS altimeter wind speed → the canonical WSPD code used by the
        # in-situ and scatterometer converters, so all wind-speed sources
        # land in the same comparison (statistics section / report plots).
        _rename = {"WIND_SPEED": "WSPD"}
        data_vars: Dict[str, tuple] = {}
        var_attrs: Dict[str, Dict] = {}
        for vname, da in raw.data_vars.items():
            if vname in _skip:
                continue
            if da.dtype.kind not in ("f", "i", "u"):
                continue
            flat = da.values.ravel()
            if len(flat) != n_points:
                continue
            out_name = _rename.get(vname, vname)
            data_vars[out_name] = ("point", flat.astype(float))
            var_attrs[out_name] = dict(da.attrs)

        if not data_vars:
            logger.warning("No usable data variables found in %s.", nc_path.name)
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
        if raw.attrs.get("doi"):
            ds.attrs["references"] = str(raw.attrs["doi"])
        apply_cf_metadata(ds, "altimeter", var_attrs)

        ds.attrs["data_type"]     = "altimeter"
        ds.attrs["platform_type"] = "altimeter"
        ds.attrs["satellite"]     = raw.attrs.get("platform", "")
        ds.attrs["frequency"]     = frequency
        ds.attrs["source"]        = "Copernicus Marine altimeter L3"
        ds.attrs["filename"]      = nc_path.name

        raw.close()
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

        # Pixel indices — the SAR pixel and scene this observation was matched to
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

        The raw ``wind_speed``/``wind_dir`` variables are renamed to the
        canonical ``WSPD``/``WDIR`` codes. ``wind_dir`` is additionally
        rotated 180° during this rename, converting ASCAT's oceanographic
        direction convention ("blowing towards") to the meteorological
        convention ("blowing from") used by Sentinel-1 OWI
        (``owiWindDirection``) and the Copernicus Marine in-situ ``WDIR``
        code — so all three direction sources are directly comparable
        downstream, in collocation and validation statistics.

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

        # Normalize longitude to -180 to +180 range (handle both 0-360 and -180-180 conventions)
        lon_arr = ((lon_arr + 180) % 360) - 180
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
        # Drop wind-vector cells whose quality flag marks the retrieval as
        # invalid (over land/ice, failed inversion, etc.) — see
        # ``_ASCAT_REJECT_FLAGS``. Without this, such cells still end up as
        # collocated validation points with a bogus or missing value, often
        # sitting on land.
        # ------------------------------------------------------------------
        n_points_raw = n_points
        keep_mask = np.ones(n_points_raw, dtype=bool)
        if "wvc_quality_flag" in raw:
            qf_attrs = raw["wvc_quality_flag"].attrs
            flag_masks = qf_attrs.get("flag_masks")
            flag_meanings = qf_attrs.get("flag_meanings")
            if flag_masks is not None and flag_meanings:
                qf = raw["wvc_quality_flag"].values.ravel()
                if len(qf) == n_points_raw:
                    reject_bits = 0
                    for mask_val, meaning in zip(flag_masks, flag_meanings.split()):
                        if meaning in _ASCAT_REJECT_FLAGS:
                            reject_bits |= int(mask_val)
                    if reject_bits:
                        keep_mask = (qf.astype(np.int64) & reject_bits) == 0

        n_dropped = n_points_raw - int(keep_mask.sum())
        if n_dropped:
            logger.info(
                "from_scatterometer_nc: dropped %d/%d cells in %s (core QC reject flags)",
                n_dropped, n_points_raw, nc_path.name,
            )
            lat_arr = lat_arr[keep_mask]
            lon_arr = lon_arr[keep_mask]
            time_arr = time_arr[keep_mask]
            n_points = int(keep_mask.sum())

        if n_points == 0:
            logger.warning(
                "from_scatterometer_nc: all cells in %s were rejected by QC flags.",
                nc_path.name,
            )
            raw.close()
            return None

        # ------------------------------------------------------------------
        # Collect numeric data variables (skip lat/lon/time/flag arrays)
        # ------------------------------------------------------------------
        _skip = {
            "lat", "latitude", "Latitude", "LAT",
            "lon", "longitude", "Longitude", "LON",
            "time", "Time", "TIME",
            "wvc_quality_flag", "wvc_index",
        }
        # OSI-SAF/ASCAT variable names → canonical validation codes, so that
        # scatterometer wind speed/direction line up with the WSPD/WDIR codes
        # produced by the in-situ (Copernicus Marine) converter and expected
        # by ``_variable_map.VARIABLE_PAIRS``.
        _rename = {"wind_speed": "WSPD", "wind_dir": "WDIR"}
        data_vars: Dict[str, tuple] = {}
        var_attrs: Dict[str, Dict] = {}
        for vname, da in raw.data_vars.items():
            if vname in _skip:
                continue
            if da.dtype.kind not in ("f", "i", "u"):   # floats and integers only
                continue
            flat = da.values.ravel()
            if len(flat) != n_points_raw:
                continue   # different spatial grid — skip
            flat = flat[keep_mask].astype(float)
            out_name = _rename.get(vname, vname)
            attrs = dict(da.attrs)
            if vname == "wind_dir":
                # ASCAT/OSI-SAF direction is oceanographic convention (the
                # direction the wind is blowing TOWARDS), while Sentinel-1
                # OWI owiWindDirection and the Copernicus Marine in-situ WDIR
                # code both use meteorological convention (the direction the
                # wind is blowing FROM). Rotate by 180° here, once, so every
                # downstream consumer (collocation, statistics, plots) sees a
                # single consistent convention.
                flat = (flat + 180.0) % 360.0
                # The raw attrs describe the pre-rotation convention
                # (standard_name "wind_to_direction") — fix them to match
                # the rotated values.
                attrs["standard_name"] = "wind_from_direction"
                attrs["long_name"] = "wind direction at 10 m (meteorological convention)"
                attrs["comment"] = (
                    "Rotated 180 degrees from the OSI-SAF oceanographic "
                    "convention (wind_to_direction) by sar-l2-validation-toolbox."
                )
            data_vars[out_name] = ("point", flat)
            var_attrs[out_name] = attrs

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
        # Carry over the raw product's descriptive globals before stamping
        # the CF conventions/references/history.
        for gattr in ("title", "institution"):
            if raw.attrs.get(gattr):
                ds.attrs[gattr] = str(raw.attrs[gattr])
        apply_cf_metadata(ds, "scatterometer", var_attrs)

        ds.attrs["data_type"]     = "scatterometer"
        ds.attrs["platform_type"] = "scatterometer"
        ds.attrs["source"]        = "OSI-SAF ASCAT"
        ds.attrs["filename"]      = nc_path.name

        raw.close()
        return ds

    #: Radiometer wind-speed variables in order of preference. RSS AMSR2 L3
    #: carries Low-Frequency (LF, 10.7 GHz), Medium-Frequency (MF, 18.7 GHz)
    #: and All-Weather (AW) winds. LF is the standard all-purpose 10 m wind and
    #: is set to fill (NaN) under rain, so keeping LF also drops rain-flagged
    #: cells for free; MF/AW are fallbacks if a product lacks LF.
    _RADIOMETER_WSPD_VARS = ("wind_speed_LF", "wind_speed_MF", "wind_speed_AW", "wind_speed")

    @staticmethod
    def from_radiometer_nc(
        nc_path: Union[str, Path],
    ) -> Optional[xr.Dataset]:
        """
        Open an RSS radiometer daily gridded NetCDF (e.g. AMSR2 L3) and return
        a standardised Dataset with a flat ``point`` dimension, matching the
        layout produced by :meth:`from_scatterometer_nc`.

        RSS distributes these products **already resampled to a common 0.25°
        global grid** (dims ``pass`` × ``lat`` × ``lon``), with two passes
        (ascending / descending) per file and a per-cell measurement ``time``.
        This method flattens every (pass, lat, lon) cell to a point, keeps the
        cells with a valid wind retrieval, and renames the chosen wind-speed
        variable (see :data:`_RADIOMETER_WSPD_VARS`) to the canonical ``WSPD``
        code so radiometer wind lands in the same wind comparison as the
        in-situ, scatterometer and altimeter sources. **No change to
        ``_variable_map.py`` is needed.**

        The node is tagged with ``sensor`` (e.g. ``"amsr2"``) so collocation
        can look up a per-sensor spec (``radiometer_<sensor>``).

        Parameters
        ----------
        nc_path : str or Path
            Path to the radiometer NetCDF file.

        Returns
        -------
        xr.Dataset or None
            Dataset with ``data_type="radiometer"``, or None on failure.
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

        if "lon" not in raw.coords or "lat" not in raw.coords:
            logger.warning(
                "from_radiometer_nc: no lon/lat coords in %s (have %s)",
                nc_path.name, list(raw.coords),
            )
            raw.close()
            return None

        # Pick the wind-speed variable (LF preferred).
        wspd_name = next((v for v in DataTreeConverter._RADIOMETER_WSPD_VARS if v in raw), None)
        if wspd_name is None:
            logger.warning(
                "from_radiometer_nc: no wind-speed variable in %s (tried %s)",
                nc_path.name, DataTreeConverter._RADIOMETER_WSPD_VARS,
            )
            raw.close()
            return None

        wind = raw[wspd_name]  # dims (pass, lat, lon) — or (lat, lon) if single-pass
        lat1d = raw["lat"].values
        lon1d = raw["lon"].values

        # Broadcast the 1-D grid coords to the full data shape, then flatten.
        # np.meshgrid gives (lat, lon); broadcasting adds the leading pass axis.
        lat_grid, lon_grid = np.meshgrid(lat1d, lon1d, indexing="ij")
        lat_full = np.broadcast_to(lat_grid, wind.shape).ravel().astype(float)
        lon_full = np.broadcast_to(lon_grid, wind.shape).ravel().astype(float)
        wspd_full = wind.values.ravel().astype(float)

        # Per-cell acquisition time (same dims as wind); fall back to the file
        # date if absent.
        if "time" in raw and raw["time"].shape == wind.shape:
            time_full = pd.to_datetime(raw["time"].values.ravel()).values
        else:
            m = re.search(r"(\d{4})-(\d{2})-(\d{2})", nc_path.stem)
            t0 = (
                np.datetime64(f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "ns")
                if m else np.datetime64("NaT", "ns")
            )
            time_full = np.full(wspd_full.shape, t0)

        # Normalize longitude to -180..180 (RSS grids are 0..360).
        lon_full = ((lon_full + 180) % 360) - 180

        # Keep only cells with a valid wind retrieval and a valid time. Wind is
        # already NaN over land/ice/rain, so this also removes those.
        keep = np.isfinite(wspd_full) & ~np.isnat(time_full)
        n_keep = int(keep.sum())
        if n_keep == 0:
            logger.info("from_radiometer_nc: no valid wind cells in %s.", nc_path.name)
            raw.close()
            return None

        data_vars: Dict[str, tuple] = {"WSPD": ("point", wspd_full[keep])}
        var_attrs: Dict[str, Dict] = {"WSPD": dict(wind.attrs)}

        # Optional wind direction (polarimetric sensors like WindSat).
        for dname in ("wind_direction", "wind_dir"):
            if dname in raw and raw[dname].shape == wind.shape:
                wdir = raw[dname].values.ravel().astype(float)[keep]
                data_vars["WDIR"] = ("point", wdir)
                var_attrs["WDIR"] = dict(raw[dname].attrs)
                break

        ds = xr.Dataset(
            data_vars,
            coords={
                "lon":  ("point", lon_full[keep]),
                "lat":  ("point", lat_full[keep]),
                "time": ("point", time_full[keep]),
            },
        )

        # Sensor tag (e.g. "amsr2") for the per-sensor collocation spec.
        sensor = str(raw.attrs.get("sensor", "")).strip().lower()
        if not sensor:
            m = re.search(r"RSS_([A-Za-z0-9]+)_", nc_path.name)
            sensor = m.group(1).lower() if m else "unknown"

        for gattr in ("title", "institution"):
            if raw.attrs.get(gattr):
                ds.attrs[gattr] = str(raw.attrs[gattr])
        apply_cf_metadata(ds, "radiometer", var_attrs)

        ds.attrs["data_type"]     = "radiometer"
        ds.attrs["platform_type"] = "radiometer"
        ds.attrs["sensor"]        = sensor
        ds.attrs["source"]        = f"RSS radiometer ({sensor.upper()})"
        ds.attrs["filename"]      = nc_path.name

        logger.info(
            "from_radiometer_nc: %s → %d valid wind points (sensor=%s, wspd_var=%s)",
            nc_path.name, n_keep, sensor, wspd_name,
        )
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
        product_type: str = "wind",
    ) -> Optional[xr.Dataset]:
        """
        Open SAR L2 OCN data from one Sentinel-1 SAFE directory and return a
        standardised Dataset.

        Automatically detects mode by inspecting the SAFE directory name:
        - If directory name contains "WV": reads WV mode oswHs point measurements
        - Otherwise: dispatches to extraction based on product_type:
          - "wind" (default): extracts OWI (Ocean Wind Index) grid data
          - "waves": extracts OSW (Ocean Surface Waves) grid data
          - "currents": extracts RVL (Radial Velocity Linesight) grid data
          Returns None if no suitable data found.

        Parameters
        ----------
        safe_dir : str or Path
            Path to a ``*.SAFE`` directory.
        product_type : str, optional
            Type of product to extract: "wind" (OWI), "waves" (OSW), or "currents" (RVL).
            Only used for IW/EW/SM modes; WV mode always extracts oswHs.
            Default is "wind".

        Returns
        -------
        xr.Dataset or None
            Dataset with oswHs points (WV) or product-specific grids (IW/EW/SM),
            or None if no suitable data found.
        """
        safe_dir = Path(safe_dir)
        safe_name = safe_dir.name.upper()

        # Detect mode from SAFE directory name
        if "WV" in safe_name:
            return DataTreeConverter.from_sar_l2_ocn_wv_safe(safe_dir)
        else:
            return DataTreeConverter._from_sar_l2_ocn_iw_safe(safe_dir, product_type=product_type)

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
        osw_attrs: Dict[str, Dict] = {}

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
                if not osw_attrs:
                    osw_attrs = {"oswHs": dict(ds_raw["oswHs"].attrs)}

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
        apply_cf_metadata(ds, "sar", osw_attrs)
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
                apply_cf_metadata(ds, "sar", {
                    var: dict(ds_raw[var].attrs)
                    for var in data_vars
                    if var in ds_raw
                })
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
            rvl_attrs: Dict[str, Dict] = {}

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

                    if not rvl_attrs:
                        rvl_attrs = {
                            v: dict(ds_raw[v].attrs)
                            for v in ("rvlRadVel", "rvlHeading", "rvlIncidenceAngle")
                            if v in ds_raw
                        }

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
            apply_cf_metadata(ds, "sar", rvl_attrs)
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
    def _extract_owi_grid_data(
        measurement_dir: Path,
        safe_dir: Union[str, Path],
    ) -> Optional[xr.Dataset]:
        """
        Extract OWI (Ocean Wind Index) data from measurement directory.

        OWI is a 2D wind inversion grid measurement in SAR products. This function
        keeps the grid structure (owiAzSize × owiRaSize dimensions, renamed to y × x)
        for direct use in collocation.

        Parameters
        ----------
        measurement_dir : Path
            Path to the measurement/ directory within a SAFE archive.
        safe_dir : str or Path
            Path to the SAFE directory (used for logging and attributes).

        Returns
        -------
        xr.Dataset or None
            Dataset with OWI variables and coordinates, or None if no OWI data found.
        """
        safe_dir = Path(safe_dir)

        # Discover measurement files containing OWI data
        # Pattern: files with "-ocn-" (all L2 OCN products have OWI)
        owi_files = sorted(
            f for f in measurement_dir.glob("*.nc")
            if "-ocn-" in f.name
        )
        if not owi_files:
            return None

        # Try to extract OWI data from the first file
        try:
            ds_raw = xr.open_dataset(owi_files[0])
        except Exception as exc:
            logger.debug("Could not open %s: %s", owi_files[0], exc)
            return None

        try:
            # Check if OWI data exists
            if "owiWindSpeed" not in ds_raw:
                ds_raw.close()
                logger.debug("No owiWindSpeed found in %s", owi_files[0])
                return None

            # Extract OWI grid arrays (2D: owiAzSize × owiRaSize)
            owi_windspeed = ds_raw["owiWindSpeed"].values  # (owiAzSize, owiRaSize)
            owi_lats = ds_raw["owiLat"].values
            owi_lons = ds_raw["owiLon"].values

            # Extract other OWI variables if available
            owi_winddir = (
                ds_raw["owiWindDirection"].values
                if "owiWindDirection" in ds_raw
                else np.full_like(owi_windspeed, np.nan)
            )

            owi_nrcs = (
                ds_raw["owiNrcs"].values[:, :, 0]  # Use first polarisation if 3D
                if "owiNrcs" in ds_raw
                else np.full_like(owi_windspeed, np.nan)
            )
            if owi_nrcs.ndim == 3:
                owi_nrcs = owi_nrcs[:, :, 0]

            owi_incidence = (
                ds_raw["owiIncidenceAngle"].values
                if "owiIncidenceAngle" in ds_raw
                else np.full_like(owi_windspeed, np.nan)
            )

            owi_heading = (
                ds_raw["owiHeading"].values
                if "owiHeading" in ds_raw
                else np.full_like(owi_windspeed, np.nan)
            )

            owi_windquality = (
                ds_raw["owiWindQuality"].values
                if "owiWindQuality" in ds_raw
                else np.full_like(owi_windspeed, np.nan)
            )

            owi_mask = (
                ds_raw["owiMask"].values
                if "owiMask" in ds_raw
                else np.ones_like(owi_windspeed, dtype=np.int8)
            )

            # Get acquisition time (scalar for grid)
            time_str = ds_raw.attrs.get("firstMeasurementTime")
            if time_str:
                acq_time = pd.to_datetime(time_str)
                acq_time_ns = np.datetime64(
                    acq_time.tz_convert(None) if acq_time.tzinfo else acq_time, "ns"
                )
            else:
                m = re.search(r"(\d{8}t\d{6})", owi_files[0].stem, re.IGNORECASE)
                if m:
                    acq_time = pd.to_datetime(m.group(1), format="%Y%m%dT%H%M%S")
                    acq_time_ns = np.datetime64(
                        acq_time.tz_convert(None) if acq_time.tzinfo else acq_time, "ns"
                    )
                else:
                    acq_time_ns = np.datetime64("NaT", "ns")

            # Infer dimension names from owiLat shape (typically owiAzSize, owiRaSize)
            dims = ds_raw["owiLat"].dims if hasattr(ds_raw["owiLat"], "dims") else ("y", "x")
            dims = tuple("y" if d.startswith("owi") or d == "owiAzSize" else "x" if d in ("owiRaSize",) else d for d in dims)
            # Simplify to standard (y, x) naming
            dims = ("y", "x")

            # Create Dataset with 2D grid structure
            data_vars = {
                "owiWindSpeed": (dims, owi_windspeed),
                "owiWindDirection": (dims, owi_winddir),
                "owiNrcs": (dims, owi_nrcs),
                "owiIncidenceAngle": (dims, owi_incidence),
                "owiHeading": (dims, owi_heading),
                "owiWindQuality": (dims, owi_windquality),
                "owiMask": (dims, owi_mask),
            }

            coords = {
                "lon": (dims, owi_lons),
                "lat": (dims, owi_lats),
                "time": acq_time_ns,
            }

            ds = xr.Dataset(data_vars, coords=coords)
            apply_cf_metadata(ds, "sar", {
                var: dict(ds_raw[var].attrs)
                for var in data_vars
                if var in ds_raw
            })
            ds.attrs["data_type"] = "sar_l2_ocn"
            ds.attrs["source"] = "Sentinel-1"
            ds.attrs["safe_dir"] = safe_dir.name
            ds.attrs["measurement_type"] = "owi"
            ds.attrs["swath_mode"] = "IW/EW/SM"

            logger.info(
                "Extracted OWI data from product %s (grid shape: %s)",
                safe_dir.name, owi_windspeed.shape
            )
            return ds

        except Exception as exc:
            logger.debug("Could not extract OWI from %s: %s", owi_files[0], exc)
            return None
        finally:
            ds_raw.close()

    @staticmethod
    def _from_sar_l2_ocn_iw_safe(
        safe_dir: Union[str, Path],
        product_type: str = "wind",
    ) -> Optional[xr.Dataset]:
        """
        Extract product-specific data from one Sentinel-1 IW/EW/SM mode SAFE directory.

        Dispatches to the appropriate extraction function based on product_type:
        - "wind": Extracts OWI (Ocean Wind Index) 2D grid data
        - "waves": Extracts OSW (Ocean Surface Waves) grid data (currently not implemented; tries OWI as fallback)
        - "currents": Extracts RVL (Radial Velocity Linesight) 2D grid data

        All returned data maintains 2D grid structure (y, x) for collocation compatibility.

        Parameters
        ----------
        safe_dir : str or Path
            Path to an IW/EW/SM mode ``*.SAFE`` directory.
        product_type : str, optional
            Type of product to extract: "wind" (OWI), "waves" (OSW), or "currents" (RVL).
            Default is "wind".

        Returns
        -------
        xr.Dataset or None
            Dataset with product-specific 2D grid data, or None if no suitable data found.
        """
        safe_dir = Path(safe_dir)
        measurement_dir = safe_dir / "measurement"
        if not measurement_dir.exists():
            logger.debug("No measurement/ directory in %s", safe_dir)
            return None

        # Dispatch based on product type
        if product_type.lower() == "wind":
            # Try OWI extraction for wind products
            ds = DataTreeConverter._extract_owi_grid_data(measurement_dir, safe_dir)
            if ds is not None:
                ds.attrs["swath_mode"] = "IW/EW/SM"
                logger.info("Extracted OWI data from IW/EW/SM product %s", safe_dir.name)
                return ds
            # Fall back to RVL if OWI not available
            logger.debug("OWI data not found in %s; trying RVL fallback", safe_dir.name)
            ds_rvl = DataTreeConverter._extract_rvl_grid_data(
                measurement_dir, safe_dir, flatten_to_points=False
            )
            if ds_rvl is not None:
                ds_rvl.attrs["swath_mode"] = "IW/EW/SM"
                logger.info("Extracted RVL data (fallback) from IW/EW/SM product %s", safe_dir.name)
                return ds_rvl

        elif product_type.lower() == "waves":
            # Try OSW extraction for wave products (future implementation)
            # For now, try OWI as fallback
            logger.debug("OSW extraction not yet implemented for %s; trying OWI fallback", safe_dir.name)
            ds = DataTreeConverter._extract_owi_grid_data(measurement_dir, safe_dir)
            if ds is not None:
                ds.attrs["swath_mode"] = "IW/EW/SM"
                return ds
            # Try RVL as second fallback
            ds_rvl = DataTreeConverter._extract_rvl_grid_data(
                measurement_dir, safe_dir, flatten_to_points=False
            )
            if ds_rvl is not None:
                ds_rvl.attrs["swath_mode"] = "IW/EW/SM"
                return ds_rvl

        elif product_type.lower() == "currents":
            # Try RVL extraction for currents products
            ds_rvl = DataTreeConverter._extract_rvl_grid_data(
                measurement_dir, safe_dir, flatten_to_points=False
            )
            if ds_rvl is not None:
                ds_rvl.attrs["swath_mode"] = "IW/EW/SM"
                logger.info("Extracted RVL data from IW/EW/SM product %s", safe_dir.name)
                return ds_rvl
            # Fall back to OWI if RVL not available
            logger.debug("RVL data not found in %s; trying OWI fallback", safe_dir.name)
            ds = DataTreeConverter._extract_owi_grid_data(measurement_dir, safe_dir)
            if ds is not None:
                ds.attrs["swath_mode"] = "IW/EW/SM"
                logger.info("Extracted OWI data (fallback) from IW/EW/SM product %s", safe_dir.name)
                return ds

        else:
            logger.warning("Unknown product_type: %s; trying wind (OWI) extraction", product_type)
            ds = DataTreeConverter._extract_owi_grid_data(measurement_dir, safe_dir)
            if ds is not None:
                ds.attrs["swath_mode"] = "IW/EW/SM"
                return ds

        # No data found for any extraction method
        logger.debug(
            "No %s data found in IW/EW/SM product %s",
            product_type, safe_dir.name,
        )
        return None


    @staticmethod
    def convert_downloaded_data(
        base_dir: Union[str, Path],
        product_type: str = "wind",
        recipe=None,
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
        product_type : str, optional
            Type of SAR product to extract: "wind" (OWI), "waves" (OSW), or "currents" (RVL).
            Default is "wind".
        recipe : Recipe, optional
            When given, every validation node is subset to the recipe's
            geographic/temporal bounds expanded by the largest collocation
            tolerance in play (see :func:`_subset_point_ds`) before it is
            stored, so ``datatree.nc`` only carries points that can actually
            collocate. Full-orbit scatterometer files shrink by >95% this
            way. SAR nodes are never cropped. ``None`` keeps everything.

        Returns
        -------
        xr.DataTree or None
            None if no data files were found.
        """
        base_dir = Path(base_dir)
        datasets: Dict[str, xr.Dataset] = {}

        # Domain-filter envelope derived from the recipe (None → no filtering)
        subset_kwargs: Optional[Dict[str, Any]] = None
        if recipe is not None:
            subset_kwargs = _build_subset_kwargs(recipe)

        def _filtered(ds: Optional[xr.Dataset], label: str) -> Optional[xr.Dataset]:
            """Apply the recipe domain filter to a validation Dataset."""
            if ds is None or subset_kwargs is None:
                return ds
            out = _subset_point_ds(ds, **subset_kwargs)
            if out is None:
                logger.info(
                    "Dropped %s — no points within recipe bounds + tolerances.",
                    label,
                )
            return out

        # SAR L2_OCN SAFE directories
        sar_dir = base_dir / "S1_L2_OCN"
        if sar_dir.exists():
            for safe_dir in sorted(d for d in sar_dir.iterdir()
                                   if d.is_dir() and d.suffix == ".SAFE"):
                ds = DataTreeConverter.from_sar_l2_ocn_safe(safe_dir, product_type=product_type)
                if ds is not None:
                    datasets[f"sar/{safe_dir.name}"] = ds
                    logger.info("Converted SAR SAFE: %s", safe_dir.name)

        # In-situ CSV (Copernicus Marine)
        insitu_dir = base_dir / "copernicus_insitu"
        if insitu_dir.exists():
            for csv_path in sorted(insitu_dir.glob("*.csv")):
                ds = _filtered(
                    DataTreeConverter.from_insitu_csv(csv_path, source_type="insitu"),
                    csv_path.name,
                )
                if ds is not None:
                    datasets[f"validation/{csv_path.stem}"] = ds
                    logger.info("Converted in-situ CSV: %s", csv_path.name)

        # Scatterometer / OSI-SAF winds (standardised to point dimension)
        for subdir_name in ("osi_saf_winds", "scatterometer"):
            subdir = base_dir / subdir_name
            if subdir.exists():
                for nc_path in sorted(subdir.glob("*.nc")):
                    ds = _filtered(
                        DataTreeConverter.from_scatterometer_nc(nc_path),
                        nc_path.name,
                    )
                    if ds is not None:
                        datasets[f"validation/{subdir_name}/{nc_path.stem}"] = ds
                        logger.info("Converted %s (scatterometer): %s", subdir_name, nc_path.name)

        # Altimeter NetCDF products (kept as raw dataset). copernicusmarine
        # sometimes can't merge a satellite/frequency request into a single
        # flat file and instead writes a directory (named after the
        # requested output filename) containing one .nc per platform, so
        # discovery must recurse. The node key is built from the path
        # relative to `altimeter/` (not just the file stem) because
        # different frequency subdirectories can contain identically-named
        # platform files (e.g. "Cryosat-2.nc" under both a 1 Hz and 5 Hz
        # dataset folder), which would otherwise collide in `datasets`.
        subdir = base_dir / "altimeter"
        if subdir.exists():
            for nc_path in sorted(subdir.rglob("*.nc")):
                ds = _filtered(DataTreeConverter.from_altimeter(nc_path), nc_path.name)
                if ds is not None:
                    rel = nc_path.relative_to(subdir).with_suffix("")
                    key = "_".join(rel.parts)
                    datasets[f"validation/altimeter/{key}"] = ds
                    logger.info("Converted altimeter: %s", nc_path.relative_to(subdir))

        # Radiometer daily gridded NetCDF products (RSS AMSR2 etc.). Each file
        # is a global 0.25° grid; from_radiometer_nc flattens it to points and
        # the domain filter crops to the recipe bbox (>95% reduction, like the
        # scatterometer).
        subdir = base_dir / "radiometer"
        if subdir.exists():
            for nc_path in sorted(subdir.glob("*.nc")):
                ds = _filtered(DataTreeConverter.from_radiometer_nc(nc_path), nc_path.name)
                if ds is not None:
                    datasets[f"validation/radiometer/{nc_path.stem}"] = ds
                    logger.info("Converted radiometer: %s", nc_path.name)

        if not datasets:
            logger.warning("No convertible data found in %s", base_dir)
            return None

        tree = DataTreeConverter.to_datatree(datasets)

        # Compress every numeric variable — SAR grids are float64 with large
        # NaN land masks and deflate extremely well.
        encoding = {
            f"/{name.lstrip('/')}": {
                var: {"zlib": True, "complevel": 4}
                for var, da in ds.variables.items()
                if da.dtype.kind in ("f", "i", "u")
            }
            for name, ds in datasets.items()
        }

        out_path = base_dir / "datatree.nc"
        tree.to_netcdf(out_path, encoding=encoding)
        logger.info("DataTree saved to %s (%d nodes)", out_path, len(datasets))

        return tree
