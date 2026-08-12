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

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr

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

#: ERA5 variable metadata per recipe variable -- raw CDS/NetCDF short
#: names, the data_type tag stamped on the result, and CF-ish attrs. Kept
#: gridded (never flattened to `point`, unlike every other validation
#: source) since ModelLayerCollocation interpolates it directly onto SAR
#: pixel locations at collocation time.
_ERA5_VARS: dict[str, dict] = {
    "wind": {
        "raw": ["u10", "v10"],
        "data_type": "era5_wind",
        "cf": {
            "u10": {"units": "m s-1", "standard_name": "eastward_wind", "long_name": "ERA5 10m u-component of wind"},
            "v10": {"units": "m s-1", "standard_name": "northward_wind", "long_name": "ERA5 10m v-component of wind"},
        },
    },
    "waves": {
        "raw": ["swh"],
        "data_type": "era5_waves",
        "cf": {
            "swh": {
                "units": "m",
                "standard_name": "sea_surface_wave_significant_height",
                "long_name": "ERA5 significant height of combined wind waves and swell",
            },
        },
    },
    "soil_moisture": {
        "raw": ["swvl1"],
        "data_type": "era5_soil_moisture",
        "cf": {
            "swvl1": {
                "units": "m3 m-3",
                "standard_name": "volume_fraction_of_water_in_soil_layer",
                "long_name": "ERA5-Land volumetric soil water layer 1 (0-7cm, H-TESSEL)",
            },
        },
    },
}

#: Matches a window-suffixed ERA5 daily filename stem, e.g.
#: "era5_wind_20260712_w0" -> stem="era5_wind_20260712", idx=0. Produced by
#: ERA5Downloader.download() when a recipe bbox crosses the antimeridian.
_ERA5_WINDOW_SUFFIX_RE = re.compile(r"^(?P<stem>.+)_w(?P<idx>\d+)$")


def _normalize_era5_grib_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Normalize an ERA5 daily NetCDF's coordinate names/extra coords to the
    ``time``/``lat``/``lon`` convention the rest of this converter (and
    ``sar_validation.core.model_collocation``) expects.

    Live-verified 2026-08-07: the CDS API's ``"data_format": "netcdf"``
    facet for ``reanalysis-era5-single-levels`` (and ``-land``) is actually
    produced by converting the underlying GRIB message via ``cfgrib``,
    which names the time dimension ``valid_time`` (not ``time``) and adds
    two GRIB-bookkeeping coordinates that carry no useful information for a
    deterministic reanalysis request: ``number`` (ensemble member, always
    0) and ``expver`` (experiment version, e.g. preliminary ERA5T vs final
    ERA5). Applied per-file (before any concatenation) so both the
    single-file and antimeridian-stitched paths through :meth:`from_era5`
    end up with a consistent ``time`` dim to concatenate/index on.
    """
    rename = {}
    if "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename["longitude"] = "lon"
    if "valid_time" in ds.coords and "time" not in ds.coords:
        rename["valid_time"] = "time"
    if rename:
        ds = ds.rename(rename)
    drop = [c for c in ("number", "expver") if c in ds.coords]
    if drop:
        ds = ds.drop_vars(drop)
    return ds


def _group_era5_paths_by_day(paths: List[Path]) -> Dict[str, List[Path]]:
    """
    Group ERA5 file paths by their day-stem, so a day's antimeridian-split
    window pair (``..._w0.nc`` / ``..._w1.nc``) is grouped together for
    stitching before concatenation across days, while an ordinary
    non-split day's single file is its own group of one.
    """
    groups: Dict[str, List[Path]] = {}
    for p in paths:
        m = _ERA5_WINDOW_SUFFIX_RE.match(p.stem)
        key = m.group("stem") if m else p.stem
        groups.setdefault(key, []).append(p)
    return groups


def _stitch_antimeridian_window_files(paths: List[Path]) -> Optional[xr.Dataset]:
    """
    Combine the 2 antimeridian-split window files for one day (see
    ``ERA5Downloader.download`` / ``split_antimeridian_bbox``) into a
    single contiguous grid: window 1 (west, originally ``-180..max_lon``)
    has its longitude axis shifted by +360 degrees so it becomes
    numerically continuous with window 0 (east, ``min_lon..180``), then
    the two are concatenated along the longitude dimension.

    The combined lon axis may extend past 180 (e.g. up to 190) -- this is
    fine, it's a coordinate array, not required to stay within +/-180; SAR
    query longitudes are remapped to match at collocation time (see
    ``model_collocation._normalize_query_lon``).

    Both windows are requested inclusive of the shared antimeridian
    boundary (``split_antimeridian_bbox`` returns ``[min_lon, 180]`` and
    ``[-180, max_lon]``), so the east window's ``180.0`` and the west
    window's shifted ``-180.0 -> 180.0`` commonly land on the exact same
    grid cell -- the duplicate is dropped after concatenation so the
    combined lon axis stays strictly increasing.

    Returns ``None`` (closing any opened files first) if *paths* doesn't
    contain exactly window indices ``{0, 1}``.
    """
    by_idx: Dict[int, xr.Dataset] = {}
    for p in paths:
        m = _ERA5_WINDOW_SUFFIX_RE.match(p.stem)
        if not m:
            logger.warning("Expected a window-suffixed ERA5 file, got %s", p.name)
            for d in by_idx.values():
                d.close()
            return None
        by_idx[int(m.group("idx"))] = xr.open_dataset(p)

    if set(by_idx) != {0, 1}:
        logger.warning("Expected exactly window indices {0, 1}, got %s", sorted(by_idx))
        for d in by_idx.values():
            d.close()
        return None

    east = by_idx[0]
    west = by_idx[1]
    lon_name = "longitude" if "longitude" in west.coords else "lon"
    west = west.assign_coords({lon_name: west[lon_name] + 360.0})
    combined = xr.concat([east, west], dim=lon_name)
    combined = combined.drop_duplicates(lon_name, keep="first")
    east.close()
    west.close()
    return combined


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
    if min_lon <= max_lon:
        lon_mask = (lon >= min_lon - deg_buf) & (lon <= max_lon + deg_buf)
    else:
        # Antimeridian-crossing bbox (GeographicBounds.min_lon > max_lon):
        # valid longitudes are the union of the two wrap-around windows,
        # not their (empty) intersection.
        lon_mask = (lon >= min_lon - deg_buf) | (lon <= max_lon + deg_buf)
    mask = (
        lon_mask
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


def _parse_ssm_timestamp(filename: str) -> np.datetime64:
    """
    Extract the acquisition timestamp from a CLMS SSM filename.

    CLMS filenames embed a ``YYYYMMDDHHMM`` (or ``YYYYMMDD``) token, e.g.
    ``c_gls_SSM1km_202601010000_CEURO_S1CSAR_V1.1.1.tif``. This parser
    accepts either width.
    """
    match = re.search(r"(\d{12}|\d{8})", filename)
    if not match:
        raise ValueError(f"Could not find a date token in SSM filename: {filename}")
    token = match.group(1)
    if len(token) == 8:
        token += "0000"
    return np.datetime64(
        f"{token[0:4]}-{token[4:6]}-{token[6:8]}T{token[8:10]}:{token[10:12]}:00"
    )


class DataTreeConverter:
    """Convert various data formats to standardised xarray objects."""

    # ------------------------------------------------------------------
    # Step 2 converters
    # ------------------------------------------------------------------

    @staticmethod
    def from_sar_l3_ssm_geotiff(tif_path: Union[str, Path]) -> Optional[xr.Dataset]:
        """
        Open a Sentinel-1 CLMS Surface Soil Moisture GeoTIFF and return a
        standardised Dataset with a native ``(y, x)`` grid — the same
        grid-shape-role SAR L2_OCN products use, so it reuses the existing
        grid-collocation path unchanged.

        Decodes the GeoTIFF's own embedded GDAL tags (no-data value,
        ``scale_factor``/``add_offset``, and named QC ``flag_values``) into
        a physical ``sarSSM`` percent-saturation grid, and derives 2-D
        ``lon``/``lat`` coordinates from the raster's affine transform/CRS.

        Parameters
        ----------
        tif_path : str or Path
            Path to a CLMS SSM GeoTIFF (as downloaded by
            ``SoilMoistureDownloader``).

        Returns
        -------
        xr.Dataset or None
            None if the file does not exist.
        """
        import rioxarray  # noqa: F401 — lazy import; registers the .rio accessor

        tif_path = Path(tif_path)
        if not tif_path.exists():
            logger.warning("GeoTIFF not found: %s", tif_path)
            return None

        # masked=True replaces the file's own no-data value (missing_value
        # GDAL tag, 255 for this product) with NaN; it does not apply
        # scale_factor/add_offset, which are handled explicitly below.
        raw = rioxarray.open_rasterio(tif_path, masked=True)
        assert isinstance(raw, xr.DataArray), (
            f"Expected a single-band GeoTIFF DataArray, got {type(raw)}: {tif_path}"
        )
        raw = raw.squeeze("band", drop=True)

        # CLMS SSM decoding — confirmed against a real downloaded product's
        # embedded GDAL tags: scale_factor=0.5, add_offset=0.0, units="%",
        # flag_values={241,242,251,252,253} (ExceedingMin/ExceedingMax/
        # WaterMask/SensitivityMask/SlopeMask QC flags, masked to NaN like
        # real no-data — these are specific reserved DNs, not a range).
        scale = float(raw.attrs.get("scale_factor", 1.0))
        offset = float(raw.attrs.get("add_offset", 0.0))
        flag_values = raw.attrs.get("flag_values", [])
        if len(flag_values):
            raw = raw.where(~raw.isin(flag_values))
        valid = raw * scale + offset

        lon2d, lat2d = np.meshgrid(raw["x"].values, raw["y"].values)

        ds = xr.Dataset(
            {"sarSSM": (("y", "x"), valid.values)},
            coords={
                "lon":  (("y", "x"), lon2d),
                "lat":  (("y", "x"), lat2d),
                "time": _parse_ssm_timestamp(tif_path.name),
            },
        )
        apply_cf_metadata(ds, "sar", {
            "sarSSM": {"long_name": "Sentinel-1 CLMS surface soil moisture (percent saturation)", "units": "%"},
        })
        ds.attrs["data_type"] = "sar_l3_ssm"
        ds.attrs["source"]    = "Sentinel-1 CLMS SSM"
        return ds

    @staticmethod
    def from_nisar_sme2(h5_path: Union[str, Path]) -> Optional[xr.Dataset]:
        """
        Open a NISAR SME2 (beta) soil-moisture HDF5 granule and return a
        standardised Dataset with a native ``(y, x)`` grid -- the same
        grid-shape-role Sentinel-1 CLMS SSM (``from_sar_l3_ssm_geotiff``)
        and SAR L2_OCN products use, so it reuses the existing
        grid-collocation path unchanged.

        Confirmed 2026-07-31 against a real downloaded granule
        (``NISAR_L3_PR_SME2_003_005_A_014_..._001.h5``): ``soilMoisture``
        (float32, meter^3/meter^3, CF-1.7 dataset attrs including its own
        ``_FillValue``) lives directly under ``science/LSAR/SME2/grids`` --
        *not* a ``frequencyA`` subgroup. (A ``grids/radarData/frequencyA``
        subgroup does exist, but only holds backscatter/sigma0 fields, no
        soil moisture.) ``latitude``/``longitude`` are 1-D EASE-grid axis
        arrays, not a 2-D meshgrid -- meshed into ``(y, x)`` coords below,
        the same way ``from_sar_l3_ssm_geotiff`` meshes its GeoTIFF axes.
        The acquisition time lives at ``science/LSAR/identification/
        zeroDopplerStartTime`` as a scalar string *dataset*, not a root
        file attribute. ``retrievalQualityFlag`` (sibling to
        ``soilMoisture``) was checked and found to flag exactly the same
        cells ``soilMoisture``'s own fill value already does, so no
        separate quality-flag masking is applied here.

        Parameters
        ----------
        h5_path : str or Path
            Path to a downloaded ``NISAR_L3_PR_SME2_*.h5`` granule.

        Returns
        -------
        xr.Dataset or None
            Dataset with ``data_type="sar_l3_ssm"``, or None on failure.
        """
        import h5py

        h5_path = Path(h5_path)
        if not h5_path.exists():
            logger.warning("NISAR SME2 file not found: %s", h5_path)
            return None

        group_path = "science/LSAR/SME2/grids"
        time_path = "science/LSAR/identification/zeroDopplerStartTime"
        try:
            with h5py.File(h5_path, "r") as f:
                if group_path not in f:
                    logger.warning(
                        "Missing %s group in %s.", group_path, h5_path.name,
                    )
                    return None
                grp = f[group_path]
                required = ("soilMoisture", "longitude", "latitude")
                if not all(field in grp for field in required):
                    logger.warning(
                        "Missing soilMoisture/longitude/latitude field(s) in %s (available: %s).",
                        h5_path.name, list(grp.keys()),
                    )
                    return None
                sm_dset = grp["soilMoisture"]
                sm = np.asarray(sm_dset[:], dtype=float)
                lon_1d = np.asarray(grp["longitude"][:], dtype=float)
                lat_1d = np.asarray(grp["latitude"][:], dtype=float)
                fill_value = float(sm_dset.attrs.get("_FillValue", -9999.0))
                time_raw = f[time_path][()] if time_path in f else None
        except Exception as exc:
            logger.warning("Could not read NISAR SME2 file %s: %s", h5_path, exc)
            return None

        sm = np.where(np.isclose(sm, fill_value) | np.isnan(sm), np.nan, sm)
        lon, lat = np.meshgrid(lon_1d, lat_1d)

        if time_raw is None:
            time_val = pd.NaT
        else:
            time_str = time_raw.decode() if isinstance(time_raw, bytes) else str(time_raw)
            time_val = pd.to_datetime(time_str, errors="coerce")

        ds = xr.Dataset(
            {"sarSSM": (("y", "x"), sm)},
            coords={
                "lon":  (("y", "x"), lon),
                "lat":  (("y", "x"), lat),
                "time": time_val,
            },
        )
        apply_cf_metadata(ds, "sar", {
            "sarSSM": {
                "long_name": "NISAR SME2 (beta) surface soil moisture",
                "units": "m3 m-3",
            },
        })
        ds.attrs["data_type"] = "sar_l3_ssm"
        ds.attrs["source"]    = "NISAR SME2 (beta)"
        return ds

    @staticmethod
    def from_radarsat2_wind(
        nc_path: Union[str, Path],
        product_type: str = "wind",
    ) -> Optional[xr.Dataset]:
        """
        Open a RADARSAT-2 SAR wind granule (NOAA NCEI THREDDS archive) and
        return a standardised Dataset with a native (y, x) grid, matching
        Sentinel-1 OWI's grid shape so it reuses the existing
        grid-collocation path unchanged.

        Wind SPEED only (``owiWindSpeed``, from ``sar_wind``) -- this
        product's ``input_dir`` field is the NWP model direction fed into
        the CMOD wind-inversion, not an independent SAR retrieval, so no
        ``owiWindDirection`` is produced (see design-choices.md Sec 10).

        Land/ice/quality masking confirmed 2026-08-05 against a real
        downloaded granule (SAR-Wind-HH-64N-174E_v3r0_rsat2_...): the
        file's own ``pixel_level_quality_flags`` == 5 ("valid wind in
        valid water region") is the strict, authoritative validity
        criterion (matches the file's own
        ``quality_information.total_number_of_valid_water_pixels``
        exactly, 63,810 pixels). ``mask``/``icemask`` alone are **not**
        a substitute for it when it's available: ``mask == -1`` (water)
        AND ``icemask == 1`` (water) alone keeps 115,267 pixels on that
        same file -- 51,457 more than flag 5, including flag-0 cells
        where ``sar_wind`` is a fill-like 0.0 and flag-4 "valid in
        buffer region" lower-confidence retrievals. ``pixel_level_quality_flags``
        does not exist in the old filename era (confirmed live against a
        2019 granule), so this method uses it directly when present, and
        falls back to ``mask == -1 AND icemask == 1`` only when it's
        absent -- a documented, era-specific approximation, not claimed
        equivalent to the new era's precision.

        Parameters
        ----------
        nc_path : str or Path
            Path to a downloaded ``*_wind_level2_norcs.nc`` (old era) or
            ``SAR-Wind-*.nc`` (new era) granule.
        product_type : str, optional
            Ignored -- accepted only for signature compatibility with
            every other ``SARSourceSpec.convert`` callback. This source
            only ever produces wind.

        Returns
        -------
        xr.Dataset or None
            Dataset with ``data_type="sar_l2_ocn"``, or None on failure.
        """
        nc_path = Path(nc_path)
        if not nc_path.exists():
            logger.warning("RADARSAT-2 wind file not found: %s", nc_path)
            return None

        try:
            ds_raw = xr.open_dataset(nc_path)
        except Exception as exc:
            logger.warning("Could not open %s: %s", nc_path, exc)
            return None

        try:
            required = ("sar_wind", "mask", "icemask", "longitude", "latitude")
            if not all(var in ds_raw for var in required):
                logger.warning(
                    "Missing required variable(s) in %s (available: %s).",
                    nc_path.name, list(ds_raw.data_vars) + list(ds_raw.coords),
                )
                return None

            if "pixel_level_quality_flags" in ds_raw:
                # New filename era only. Flag 5 = "valid wind in valid
                # water region" -- the strict, authoritative criterion
                # (matches quality_information.total_number_of_valid_water_pixels
                # exactly). mask/icemask alone are NOT a substitute: they
                # let through ~51k extra pixels this flag correctly
                # excludes (fill-like flag-0 cells, lower-confidence
                # flag-4 "buffer region" retrievals) -- confirmed live,
                # see design-choices.md Sec 10.
                valid = ds_raw["pixel_level_quality_flags"].values == 5
            else:
                # Old filename era: pixel_level_quality_flags doesn't
                # exist. mask/icemask are the best available signal -- a
                # documented approximation, slightly more permissive than
                # the new era's flag-based criterion.
                mask = ds_raw["mask"].values
                icemask = ds_raw["icemask"].values
                valid = (mask == -1) & (icemask == 1)

            speed = np.where(valid, ds_raw["sar_wind"].values, np.nan).astype(float)
            lon = ds_raw["longitude"].values
            lat = ds_raw["latitude"].values

            time_str = ds_raw.attrs.get("time_coverage_start")
            if time_str:
                raw_time = pd.to_datetime(time_str)
                if raw_time.tzinfo is not None:
                    raw_time = raw_time.tz_convert(None)
                acq_time_ns = np.datetime64(raw_time, "ns")
            else:
                acq_time_ns = np.datetime64("NaT", "ns")

            ds = xr.Dataset(
                {"owiWindSpeed": (("y", "x"), speed)},
                coords={
                    "lon": (("y", "x"), lon),
                    "lat": (("y", "x"), lat),
                    "time": acq_time_ns,
                },
            )
            apply_cf_metadata(ds, "sar", {
                "owiWindSpeed": {
                    "long_name": "RADARSAT-2 SAR-derived wind speed at 10-m height neutral stability",
                    "units": "m s-1",
                },
            })
            ds.attrs["data_type"] = "sar_l2_ocn"
            ds.attrs["source"] = "RADARSAT-2"
            ds.attrs["filename"] = nc_path.name
            return ds
        except Exception as exc:
            logger.warning("Could not extract wind data from %s: %s", nc_path.name, exc)
            return None
        finally:
            ds_raw.close()

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

        # Derive eastward/northward current components from speed + direction
        # when the direct components are absent or all-NaN.
        if "HCSP" in df.columns and "HCDT" in df.columns:
            hcdt_rad = np.radians(df["HCDT"].to_numpy(dtype=float, na_value=np.nan))
            hcsp = df["HCSP"].to_numpy(dtype=float, na_value=np.nan)
            for col, trig in (("EWCT", np.sin), ("NSCT", np.cos)):
                if col in df.columns and not df[col].isna().all():
                    continue  # column has real data — leave it untouched
                df[col] = hcsp * trig(hcdt_rad)
                logger.debug("Derived %s from HCSP+HCDT", col)

        # A single in-situ platform can report more than one significant
        # wave height estimate in the same row -- VHM0 (spectral Hm0) and
        # VAVH (time-domain H1/3) are independently-computed, non-identical
        # quantities (confirmed live 2026-08-10: mooring 6200442 reported
        # VAVH=1.0 and VHM0=1.1 for the same reading), and CMEMS's
        # long-format export pivots each into its own column above. Left
        # as-is, that one physical observation would land in BOTH the
        # VAVH-paired (altimeter) and VHM0-paired (ERA5) report sections --
        # double-counting a single match across two comparisons. Keep only
        # the highest-precedence column per row (VHM0 > VAVH > VGHS,
        # matching _variable_map.py's own wave_val_params fallback order)
        # and null the rest, so each observation contributes to exactly one
        # comparison; a row reporting only one of them is untouched.
        wave_height_cols = [c for c in ("VHM0", "VAVH", "VGHS") if c in df.columns]
        if len(wave_height_cols) > 1:
            claimed = pd.Series(False, index=df.index)
            for col in wave_height_cols:
                has_val = df[col].notna()
                df.loc[claimed & has_val, col] = np.nan
                claimed = claimed | has_val

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
            out_name = _rename.get(str(vname), str(vname))
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
                        ts = pd.Timestamp(val)
                        if ts.tzinfo is not None:
                            ts = ts.tz_convert(None)
                        t0 = np.datetime64(ts, "ns")
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
            out_name = _rename.get(str(vname), str(vname))
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

    @staticmethod
    def _build_ssm_point_dataset(
        sm: np.ndarray, lon: np.ndarray, lat: np.ndarray, time_vals: np.ndarray,
        *, data_type: str, var_attrs: dict, platform_type: str, source: str,
        sensing_depth_cm: str, band: str, filename: str,
        sensor: Optional[str] = None, native_grid_deg: Optional[float] = None,
    ) -> xr.Dataset:
        """Build the flat-point SOIL_MOISTURE Dataset + attrs shared by every
        ``from_*_ssm`` parser. Caller is responsible for pre-filtering
        *sm*/*lon*/*lat*/*time_vals* down to valid cells."""
        ds = xr.Dataset(
            {"SOIL_MOISTURE": ("point", sm)},
            coords={
                "lon":  ("point", lon),
                "lat":  ("point", lat),
                "time": ("point", time_vals),
            },
        )
        apply_cf_metadata(ds, data_type, var_attrs)

        ds.attrs["data_type"]        = data_type
        ds.attrs["platform_type"]    = platform_type
        if sensor is not None:
            ds.attrs["sensor"] = sensor
        ds.attrs["source"]           = source
        ds.attrs["sensing_depth_cm"] = sensing_depth_cm
        ds.attrs["band"]             = band
        ds.attrs["filename"]         = filename
        if native_grid_deg is not None:
            ds.attrs["native_grid_deg"] = native_grid_deg
        return ds

    @staticmethod
    def from_ascat_ssm(path: Union[str, Path]) -> Optional[xr.Dataset]:
        """
        Open an ASCAT Soil Moisture (SOMO12) product via the TU Wien
        ``ascat`` package's format-agnostic reader (transparently handles
        .nc/.bfr/.nat — see ``ascat.eumetsat.level2.AscatL2File``) and
        return a standardised Dataset with a flat ``point`` dimension.

        CONFIRMED against a real downloaded product: a genuine
        ``EO:EUM:DAT:METOP:SOMO12`` ``.nat`` file (METOP-B, 2024-01-02)
        fetched via EUMDAC was read successfully by ``AscatL2File(...).read(
        generic=True, to_xarray=True)``, yielding generic ``sm``/``lon``/
        ``lat``/``time`` fields as assumed below; the resulting
        ``SOIL_MOISTURE`` values fell in the expected 0-100 (percent
        saturation) range across 26789 points.

        Parameters
        ----------
        path : str or Path
            Path to the downloaded ASCAT SSM product file.

        Returns
        -------
        xr.Dataset or None
            Dataset with ``data_type="scatterometer_ssm"``, or None on failure.
        """
        path = Path(path)
        if not path.exists():
            logger.warning("ASCAT SSM file not found: %s", path)
            return None

        try:
            from ascat.eumetsat.level2 import AscatL2File

            reader = AscatL2File(str(path))
            data, _metadata = reader.read(generic=True, to_xarray=True)
        except Exception as exc:
            logger.warning("Could not read ASCAT SSM file %s: %s", path, exc)
            return None

        if not isinstance(data, xr.Dataset) or "sm" not in data:
            logger.warning("No usable 'sm' field in %s.", path.name)
            return None

        if not ("lon" in data and "lat" in data and "time" in data):
            logger.warning(
                "Missing lon/lat/time field(s) in %s (available: %s).",
                path.name,
                list(data.coords) + list(data.data_vars),
            )
            return None

        lon = ((data["lon"].values.ravel() + 180) % 360) - 180
        lat = data["lat"].values.ravel()
        time_vals = pd.to_datetime(data["time"].values.ravel())
        sm = data["sm"].values.ravel().astype(float)

        valid = ~np.isnan(sm)
        if not valid.any():
            logger.warning("from_ascat_ssm: all cells NaN in %s.", path.name)
            return None

        var_attrs = {
            "SOIL_MOISTURE": {
                "units": "%",
                "standard_name": "soil_moisture_saturation",
                "long_name": "ASCAT surface soil moisture (~0-5cm, C-band)",
            }
        }
        return DataTreeConverter._build_ssm_point_dataset(
            sm[valid], lon[valid], lat[valid], time_vals.values[valid],
            data_type="scatterometer_ssm", var_attrs=var_attrs,
            platform_type="ascat_ssm",
            source="ASCAT Soil Moisture 12.5km Swath Grid (EO:EUM:DAT:METOP:SOMO12)",
            sensing_depth_cm="0-5", band="C", filename=path.name,
        )

    @staticmethod
    def from_hsaf_ssm(path: Union[str, Path]) -> Optional[xr.Dataset]:
        """
        Open an H-SAF ASCAT Surface Soil Moisture NRT (H29) product.

        Unlike the EUMDAC/SOMO12 path (from_ascat_ssm, via the ``ascat``
        package's WARP5-grid-aware reader), H29 files ship a flat
        ``(obs,)`` array with ``latitude``/``longitude``/``time`` already
        resolved to real per-observation coordinates -- CONFIRMED against
        a real downloaded file (see hsaf_downloader.py's module
        docstring): no grid-point-ID lookup is needed, so this reads the
        file directly with xarray rather than routing through ``ascat``.

        Parameters
        ----------
        path : str or Path
            Path to the downloaded H-SAF H29 product file.

        Returns
        -------
        xr.Dataset or None
            Dataset with ``data_type="scatterometer_ssm"``, or None on failure.
        """
        path = Path(path)
        if not path.exists():
            logger.warning("H-SAF SSM file not found: %s", path)
            return None

        try:
            data = xr.open_dataset(path)
        except Exception as exc:
            logger.warning("Could not read H-SAF SSM file %s: %s", path, exc)
            return None

        if "surface_soil_moisture" not in data:
            logger.warning("No 'surface_soil_moisture' field in %s.", path.name)
            return None
        if not ("longitude" in data and "latitude" in data and "time" in data):
            logger.warning(
                "Missing longitude/latitude/time field(s) in %s (available: %s).",
                path.name,
                list(data.coords) + list(data.data_vars),
            )
            return None

        lon = ((data["longitude"].values.ravel() + 180) % 360) - 180
        lat = data["latitude"].values.ravel()
        time_vals = pd.to_datetime(data["time"].values.ravel())
        sm = data["surface_soil_moisture"].values.ravel().astype(float)

        valid = ~np.isnan(sm)
        if not valid.any():
            logger.warning("from_hsaf_ssm: all cells NaN in %s.", path.name)
            return None

        var_attrs = {
            "SOIL_MOISTURE": {
                "units": "%",
                "standard_name": "soil_moisture_saturation",
                "long_name": "ASCAT surface soil moisture (~0-5cm, C-band)",
            }
        }
        return DataTreeConverter._build_ssm_point_dataset(
            sm[valid], lon[valid], lat[valid], time_vals.values[valid],
            data_type="scatterometer_ssm", var_attrs=var_attrs,
            platform_type="ascat_ssm",
            source="H-SAF ASCAT Surface Soil Moisture NRT 12.5km (H29)",
            sensing_depth_cm="0-5", band="C", filename=path.name,
        )

    @staticmethod
    def from_amsr_ssm(path: Union[str, Path]) -> Optional[xr.Dataset]:
        """
        Open an AMSR-E/AMSR2 soil-moisture product and return a
        standardised Dataset with a flat ``point`` dimension. Three on-disk
        formats are supported, auto-detected from the file's own group
        layout:

        - **JAXA G-Portal L3SGSMC** (L3 daily 0.1-degree global grid;
          what :class:`~sar_validation.downloaders.gportal_downloader.
          GPortalAMSR2Downloader` actually delivers -- ``GW1AM2_
          YYYYMMDD_01D_EQM[AD]_L3SGSMCHF*.h5``) -- detected via root-level
          ``Geophysical Data``/``Time Information`` datasets and delegated
          to :meth:`_from_amsr_ssm_gportal_l3_grid`. **Confirmed against a
          real downloaded granule** (GW1AM2_20250701_01D_EQMA_
          L3SGSMCHF3300300.h5) -- this is the format actually seen in
          practice; the NSIDC-0451 branch below has not been.
        - **NSIDC-0451** (L3 daily global grid; used for dates on or
          before 2023-12-31, per the orchestrator's ``_NSIDC_0451_CUTOFF``)
          -- handled below.
        - **AU_Land_NRT_R02**/**AU_Land** (L2B half-orbit granule,
          HDF-EOS5 POINTS layout -- not SWATHS, despite the product's own
          "half-orbit swath" description; the historical-coverage-
          extension replacement for NSIDC-0451, used for dates after that
          cutoff) -- detected via the presence of an ``HDFEOS/POINTS``
          group and delegated to :meth:`_from_amsr_ssm_au_land_points`.
          **Confirmed against a real downloaded granule**
          (AMSR_U2_L2_Land_B02_202312312326_D.he5) -- the field layout
          guessed before that (a ``SWATHS`` group with separate named
          datasets) does not match any real granule and always fell
          through to the "Missing vsm/longitude/latitude field(s)"
          warning below.

        Field names for the NSIDC-0451 branch are assumed from the
        NSIDC-0451 v3.1 technical readme: ``vsm`` (surface, <=2cm,
        volumetric soil moisture, X-band 10.7 GHz), ``longitude``/
        ``latitude`` as flat root-level datasets, plus a
        ``time_coverage_start`` file attribute. This has not been confirmed
        against a real downloaded file (see the plan's open items) -- if
        the real product nests these under a group (e.g.
        ``/Data Fields/vsm``), the ``f["vsm"]``/``f["longitude"]``/
        ``f["latitude"]`` lookups below will need updating.
        NSIDC-0451 uses -9999.0 as its fill value (not NaN) -- cells equal
        to this value are dropped.

        Parameters
        ----------
        path : str or Path
            Path to the downloaded HDF5 file.

        Returns
        -------
        xr.Dataset or None
            Dataset with ``data_type="radiometer_ssm"``, ``sensor="amsr"``,
            or None on failure.
        """
        import h5py

        path = Path(path)
        if not path.exists():
            logger.warning("AMSR SSM file not found: %s", path)
            return None

        try:
            with h5py.File(path, "r") as f:
                if "HDFEOS/POINTS" in f:
                    return DataTreeConverter._from_amsr_ssm_au_land_points(f, path)
                if "Geophysical Data" in f and "Time Information" in f:
                    return DataTreeConverter._from_amsr_ssm_gportal_l3_grid(f, path)
                # ... existing NSIDC-0451 L3 daily-grid parsing continues below,
                # unchanged ...
                if not ("vsm" in f and "longitude" in f and "latitude" in f):
                    logger.warning(
                        "Missing vsm/longitude/latitude field(s) in %s (available: %s).",
                        path.name, list(f.keys()),
                    )
                    return None
                sm = np.asarray(f["vsm"][:], dtype=float).ravel()
                lon = np.asarray(f["longitude"][:], dtype=float).ravel()
                lat = np.asarray(f["latitude"][:], dtype=float).ravel()
                time_attr = f.attrs.get("time_coverage_start")
        except Exception as exc:
            logger.warning("Could not read AMSR SSM file %s: %s", path, exc)
            return None

        lon = ((lon + 180) % 360) - 180

        valid = (sm != -9999.0) & ~np.isnan(sm)
        if not valid.any():
            logger.warning("from_amsr_ssm: all cells invalid in %s.", path.name)
            return None

        if time_attr:
            time_val = pd.Timestamp(time_attr.decode() if isinstance(time_attr, bytes) else time_attr)
        else:
            logger.warning("No time_coverage_start in %s; using NaT.", path.name)
            time_val = pd.NaT
        n_valid = int(valid.sum())
        time_arr = np.full(n_valid, np.datetime64(time_val, "ns"))

        var_attrs = {
            "SOIL_MOISTURE": {
                "units": "m3 m-3",
                "standard_name": "volume_fraction_of_water_in_soil",
                "long_name": "AMSR-E/2 surface soil moisture (~0-1cm, X/Ka-band)",
            }
        }
        return DataTreeConverter._build_ssm_point_dataset(
            sm[valid], lon[valid], lat[valid], time_arr,
            data_type="radiometer_ssm", var_attrs=var_attrs,
            platform_type="amsr_ssm", sensor="amsr",
            source="AMSR-E/AMSR2 Daily Global Land Parameters (NSIDC-0451)",
            sensing_depth_cm="0-1", band="X/Ka", filename=path.name,
        )

    @staticmethod
    def _from_amsr_ssm_au_land_points(f: Any, path: Path) -> Optional[xr.Dataset]:
        """
        Parse an ``AU_Land``/``AU_Land_NRT_R02`` L2B half-orbit granule
        (HDF-EOS5 format) -- the historical-coverage-extension replacement
        for the fully-discontinued NSIDC-0451 L3 daily grid.

        **Confirmed against a real downloaded granule**
        (AMSR_U2_L2_Land_B02_202312312326_D.he5, fetched live 2026-08-07):
        despite the product's own "half-orbit swath" description, its
        real on-disk layout is HDF-EOS5's POINTS structure, not SWATHS --
        a single compound (structured) dataset at
        ``HDFEOS/POINTS/AMSR-2 Level 2 Land Data/Data/Combined NPD and SCA
        Output Fields``, whose named fields include ``Time``,
        ``Latitude``, ``Longitude``, and -- per NSIDC's collection
        abstract -- two independent, co-equal soil-moisture retrievals
        with no stated "primary" one: ``SoilMoistureNPD`` (Normalized
        Polarization Difference) and ``SoilMoistureSCA`` (Single Channel
        Algorithm), each with its own ``RetrievalQualityFlag{NPD,SCA}``.
        NPD is used here (see design-choices.md) since it matches the
        algorithm NSIDC-0451, this product's predecessor, used
        exclusively. ``Time`` is seconds since 1993-01-01T00:00:00 (TAI93
        convention, common to NASA/JAXA AMSR products) -- confirmed
        numerically live, not seconds since the Unix epoch. -9999.0 is
        the fill value for both the soil-moisture and QC fields (same
        convention as the NSIDC-0451 branch above).
        """
        try:
            data = f["HDFEOS/POINTS/AMSR-2 Level 2 Land Data/Data/Combined NPD and SCA Output Fields"][:]
            sm = data["SoilMoistureNPD"]
            lat = data["Latitude"]
            lon = data["Longitude"]
            time_raw = data["Time"]
        except (KeyError, ValueError) as exc:
            logger.warning(
                "from_amsr_ssm: AU_Land file %s is missing an expected "
                "field/group (%s).", path.name, exc,
            )
            return None

        valid = (sm != -9999.0) & ~np.isnan(sm)
        if not valid.any():
            logger.warning("from_amsr_ssm: all cells invalid in %s.", path.name)
            return None

        # TAI93: seconds since 1993-01-01T00:00:00, not the Unix epoch.
        time_vals = pd.Timestamp("1993-01-01") + pd.to_timedelta(time_raw[valid], unit="s")
        var_attrs = {
            "SOIL_MOISTURE": {
                "units": "m3 m-3",
                "standard_name": "volume_fraction_of_water_in_soil",
                "long_name": "AMSR-E/2 surface soil moisture (~0-1cm, X/Ka-band, NPD algorithm)",
            }
        }
        return DataTreeConverter._build_ssm_point_dataset(
            sm[valid].astype(float), lon[valid].astype(float), lat[valid].astype(float), time_vals.values,
            data_type="radiometer_ssm", var_attrs=var_attrs,
            platform_type="amsr_ssm", sensor="amsr",
            source="AMSR-E/AMSR2 Unified L2B Half-Orbit SSM (AU_Land/AU_Land_NRT_R02)",
            sensing_depth_cm="0-1", band="X/Ka", filename=path.name,
        )

    @staticmethod
    def _from_amsr_ssm_gportal_l3_grid(f: Any, path: Path) -> Optional[xr.Dataset]:
        """
        Parse a JAXA G-Portal AMSR2 L3 daily 0.1-degree global soil-moisture
        grid (``GW1AM2_YYYYMMDD_01D_EQM[AD]_L3SGSMCHF*.h5``) -- the format
        actually delivered by :class:`~sar_validation.downloaders.
        gportal_downloader.GPortalAMSR2Downloader`, distinct from both the
        NSIDC-0451 grid and the AU_Land swath formats above.

        **Confirmed against a real downloaded granule**
        (GW1AM2_20250701_01D_EQMA_L3SGSMCHF3300300.h5): ``Geophysical
        Data`` is a root-level ``(1800, 3600, 1)`` int16 dataset (its
        ``SCALE FACTOR``/``UNIT`` attrs give raw*0.1 = percent volumetric
        soil moisture -- confirmed by its own ``GeophysicalName`` file
        attribute, "Soil Moisture Content"); ``Time Information`` is a
        sibling ``(1800, 3600)`` int16 dataset in minutes-since-00:00-UTC
        of the granule's date (consistent with the file's own
        ``ObservationStartDateTime``/``ObservationEndDateTime`` attrs).
        There are no on-disk longitude/latitude fields -- the grid is a
        fixed global 0.1 degree EQR grid (JAXA GCOM-W's standard L3 grid
        definition): row 0 = 89.95 N downward, column 0 = -179.95 E
        eastward. ``-32768``/``-32767`` are sentinel codes (no-retrieval /
        missing-observation respectively, not physical readings) -- any
        negative raw value is treated as invalid.
        """
        try:
            sm_raw = np.asarray(f["Geophysical Data"][:, :, 0], dtype=np.int32)
            time_raw = np.asarray(f["Time Information"][:], dtype=np.int32)
        except KeyError as exc:
            logger.warning(
                "from_amsr_ssm: missing expected field in G-Portal L3 grid "
                "file %s (%s).", path.name, exc,
            )
            return None

        # Both the reading and its per-pixel observation time must be valid
        # -- a soil-moisture value collocation can't use without a real
        # timestamp is as useless as no reading at all, and pushing NaT
        # handling downstream into collocation's temporal-distance math
        # would be fragile. ~15% of otherwise-valid-sm cells in the
        # confirmed test granule lack a valid Time Information value.
        valid = (sm_raw >= 0) & (time_raw >= 0)
        if not valid.any():
            logger.warning("from_amsr_ssm: all cells invalid in %s.", path.name)
            return None

        date_match = re.search(r"_(\d{8})_", path.name)
        if date_match is None:
            logger.warning(
                "from_amsr_ssm: could not parse a YYYYMMDD date from filename "
                "%s.", path.name,
            )
            return None
        granule_date = pd.Timestamp(date_match.group(1))

        sm_scale = float(np.asarray(f["Geophysical Data"].attrs.get("SCALE FACTOR", [0.1]))[0])
        sm_percent = sm_raw[valid].astype(float) * sm_scale
        sm = sm_percent / 100.0   # percent -> m3 m-3 volumetric fraction

        ny, nx = sm_raw.shape
        lat_1d = 90.0 - (np.arange(ny) + 0.5) * (180.0 / ny)
        lon_1d = -180.0 + (np.arange(nx) + 0.5) * (360.0 / nx)
        lat2d, lon2d = np.meshgrid(lat_1d, lon_1d, indexing="ij")

        time_scale = float(np.asarray(f["Time Information"].attrs.get("SCALE FACTOR", [1.0]))[0])
        minutes = time_raw[valid].astype(float) * time_scale
        time_vals = (granule_date + pd.to_timedelta(minutes, unit="m")).values

        var_attrs = {
            "SOIL_MOISTURE": {
                "units": "m3 m-3",
                "standard_name": "volume_fraction_of_water_in_soil",
                "long_name": "AMSR-E/2 surface soil moisture (~0-1cm, X/Ka-band)",
            }
        }
        return DataTreeConverter._build_ssm_point_dataset(
            sm, lon2d[valid], lat2d[valid], time_vals,
            data_type="radiometer_ssm", var_attrs=var_attrs,
            platform_type="amsr_ssm", sensor="amsr",
            source="AMSR2 L3 Daily 0.1-degree Global Soil Moisture (JAXA G-Portal L3SGSMC)",
            sensing_depth_cm="0-1", band="X/Ka", filename=path.name,
            native_grid_deg=0.1,
        )

    @staticmethod
    def from_smap_ssm(path: Union[str, Path]) -> Optional[xr.Dataset]:
        """
        Open a SMAP Enhanced L2 Radiometer Soil Moisture product
        (SPL2SMP_E, HDF5) and return a standardised Dataset with a flat
        ``point`` dimension.

        CONFIRMED against a real downloaded product
        (``SMAP_L2_SM_P_E_55665_A_20250703T162602_R19240_001.h5``): fields
        do live under the ``Soil_Moisture_Retrieval_Data`` HDF5 group as
        ``soil_moisture``, ``longitude``, ``latitude``, ``tb_time_utc``
        (per-cell ISO-8601 timestamp string), and ``soil_moisture`` uses
        -9999.0 as its fill value (not NaN). Resulting SOIL_MOISTURE range
        for that file: 0.02-0.79 m3/m3 (116711 of 264701 cells retained).

        ``tb_time_utc`` carries its OWN, independent fill convention:
        confirmed 118 of 264701 cells in that same file used a literal
        ``"***"`` placeholder for the fractional-seconds digits (e.g.
        ``"2025-07-03T17:19:25.***Z"``), at cells where ``soil_moisture``
        itself was otherwise valid -- a strict ``pd.to_datetime`` on these
        raises ``ValueError`` and aborts the whole conversion. Parsed with
        ``errors="coerce"`` instead (turning unparseable strings into
        ``NaT``) and dropped via the same validity mask as the
        soil_moisture fill value.

        Parameters
        ----------
        path : str or Path
            Path to the downloaded HDF5 file.

        Returns
        -------
        xr.Dataset or None
            Dataset with ``data_type="radiometer_ssm"``, ``sensor="smap"``,
            or None on failure.
        """
        import h5py

        path = Path(path)
        if not path.exists():
            logger.warning("SMAP SSM file not found: %s", path)
            return None

        try:
            with h5py.File(path, "r") as f:
                if "Soil_Moisture_Retrieval_Data" not in f:
                    logger.warning(
                        "Missing Soil_Moisture_Retrieval_Data group in %s (available: %s).",
                        path.name, list(f.keys()),
                    )
                    return None
                grp = f["Soil_Moisture_Retrieval_Data"]
                required = ("soil_moisture", "longitude", "latitude", "tb_time_utc")
                if not all(field in grp for field in required):
                    logger.warning(
                        "Missing soil_moisture/longitude/latitude/tb_time_utc field(s) in %s (available: %s).",
                        path.name, list(grp.keys()),
                    )
                    return None
                sm = np.asarray(grp["soil_moisture"][:], dtype=float).ravel()
                lon = np.asarray(grp["longitude"][:], dtype=float).ravel()
                lat = np.asarray(grp["latitude"][:], dtype=float).ravel()
                time_raw = np.asarray(grp["tb_time_utc"][:]).ravel()
        except Exception as exc:
            logger.warning("Could not read SMAP SSM file %s: %s", path, exc)
            return None

        lon = ((lon + 180) % 360) - 180

        # tb_time_utc carries its own, independent fill convention: some
        # cells use a literal "***" placeholder for the fractional-seconds
        # digits (e.g. "2025-07-03T17:19:25.***Z") even when soil_moisture
        # itself is valid at that cell -- confirmed against a real
        # downloaded SPL2SMP_E granule (118 of 264701 cells). errors="coerce"
        # turns those into NaT rather than raising, so they can be dropped
        # via the same validity mask as the -9999.0 soil_moisture fill.
        time_strs = [
            t.decode() if isinstance(t, bytes) else str(t)
            for t in time_raw
        ]
        # Explicit format (rather than letting pandas infer it) keeps this
        # vectorized -- mixing valid timestamps with "***"-fill strings
        # defeats pandas' format auto-detection, which otherwise silently
        # falls back to a much slower per-element dateutil parse (real
        # files carry ~2.6e5 cells).
        time_parsed = pd.to_datetime(time_strs, format="%Y-%m-%dT%H:%M:%S.%fZ", errors="coerce")

        valid = (sm != -9999.0) & ~np.isnan(sm) & ~time_parsed.isna()
        if not valid.any():
            logger.warning("from_smap_ssm: all cells invalid in %s.", path.name)
            return None

        sm = sm[valid]
        lon = lon[valid]
        lat = lat[valid]
        time_arr = time_parsed[valid].values

        var_attrs = {
            "SOIL_MOISTURE": {
                "units": "m3 m-3",
                "standard_name": "volume_fraction_of_water_in_soil",
                "long_name": "SMAP surface soil moisture (~0-5cm, L-band)",
            }
        }
        return DataTreeConverter._build_ssm_point_dataset(
            sm, lon, lat, time_arr,
            data_type="radiometer_ssm", var_attrs=var_attrs,
            platform_type="smap_ssm", sensor="smap",
            source="SMAP Enhanced L2 Radiometer Soil Moisture (SPL2SMP_E)",
            sensing_depth_cm="0-5", band="L", filename=path.name,
        )

    @staticmethod
    def from_smos_ssm(path: Union[str, Path]) -> Optional[xr.Dataset]:
        """
        Open a SMOS soil-moisture product (ESA Online Dissemination,
        SM_OPER_MIR_SMUDP2, NetCDF) and return a standardised Dataset with
        a flat ``point`` dimension.

        CONFIRMED against a real downloaded product (fetched via the OADS
        portal, NRT_Open/MIR_SMNRT2, 2025-07-02): field names are
        **lowercase** (``soil_moisture``/``longitude``/``latitude``), NOT
        the capitalised names originally assumed here. There is also no
        single ``time`` variable -- per-point acquisition time is instead
        split across two integer fields, ``days_since_01-01-2000`` (days
        since the SMOS epoch 2000-01-01T00:00:00) and
        ``seconds_since_midnight`` (seconds within that day), combined
        below. Confirmed the real product is plain NetCDF at the top level
        (no ``.tgz``/``.HDR`` extraction needed).

        Confirmed soil_moisture has no ``-999.0``-style fill sentinel in
        real data (a full real granule's min/max were both plain physical
        values in ``[0, 1]``, no NaN) -- validity is primarily via
        ``~np.isnan`` (matching whatever fill/missing-value decoding
        ``xr.open_dataset``'s default ``mask_and_scale=True`` already
        applied), with the ``-999.0`` check kept alongside it only as
        cheap, harmless insurance in case some other file/swath does use
        that sentinel.

        Parameters
        ----------
        path : str or Path
            Path to the downloaded NetCDF file.

        Returns
        -------
        xr.Dataset or None
            Dataset with ``data_type="radiometer_ssm"``, ``sensor="smos"``,
            or None on failure.
        """
        path = Path(path)
        if not path.exists():
            logger.warning("SMOS SSM file not found: %s", path)
            return None

        try:
            raw = xr.open_dataset(path)
        except Exception as exc:
            logger.warning("Could not open %s: %s", path, exc)
            return None

        required = (
            "soil_moisture", "longitude", "latitude",
            "days_since_01-01-2000", "seconds_since_midnight",
        )
        if not all(field in raw.variables for field in required):
            logger.warning(
                "Missing soil_moisture/longitude/latitude/"
                "days_since_01-01-2000/seconds_since_midnight field(s) in "
                "%s (available: %s).",
                path.name, list(raw.variables),
            )
            raw.close()
            return None

        lon = ((raw["longitude"].values.ravel() + 180) % 360) - 180
        lat = raw["latitude"].values.ravel()
        smos_epoch = pd.Timestamp("2000-01-01")
        days = raw["days_since_01-01-2000"].values.ravel().astype("int64")
        seconds = raw["seconds_since_midnight"].values.ravel().astype("int64")
        time_arr = (
            smos_epoch
            + pd.to_timedelta(days, unit="D")
            + pd.to_timedelta(seconds, unit="s")
        )
        sm = raw["soil_moisture"].values.ravel().astype(float)

        valid = ~np.isnan(sm) & (sm != -999.0)
        if not valid.any():
            logger.warning("from_smos_ssm: all cells invalid in %s.", path.name)
            raw.close()
            return None

        var_attrs = {
            "SOIL_MOISTURE": {
                "units": "m3 m-3",
                "standard_name": "volume_fraction_of_water_in_soil",
                "long_name": "SMOS surface soil moisture (~0-5cm, L-band)",
            }
        }
        ds = DataTreeConverter._build_ssm_point_dataset(
            sm[valid], lon[valid], lat[valid], time_arr.values[valid],
            data_type="radiometer_ssm", var_attrs=var_attrs,
            platform_type="smos_ssm", sensor="smos",
            source="SMOS Soil Moisture (ESA Online Dissemination, SM_OPER_MIR_SMUDP2)",
            sensing_depth_cm="0-5", band="L", filename=path.name,
        )
        raw.close()
        return ds

    @staticmethod
    def from_c3s_ssm(
        nc_path: Union[str, Path],
        product_type: str,
    ) -> Optional[xr.Dataset]:
        """
        Open a C3S CDS satellite soil moisture NetCDF (0.25° daily global
        grid) and return a standardised Dataset with a flat ``point``
        dimension.

        Supports all three sensor-class variants produced by
        :class:`~sar_validation.downloaders.cds_soil_moisture_downloader.CDSSoilMoistureDownloader`:

        * ``"active"``   — ASCAT multi-scatterometer composite, units ``%``
        * ``"passive"``  — Multi-radiometer composite, units ``m3 m-3``
        * ``"combined"`` — Merged active + passive, units ``m3 m-3``

        The CDS NetCDF delivers a global 0.25° regular grid with dimensions
        ``(time, lat, lon)`` and a primary ``sm`` variable.  The grid is
        flattened to a flat ``point`` dimension so the result passes through
        the same layer-vs-layer collocation path as other SSM sources.

        Parameters
        ----------
        nc_path : str or Path
            Path to a NetCDF file as downloaded and extracted by
            :class:`~sar_validation.downloaders.cds_soil_moisture_downloader.CDSSoilMoistureDownloader`.
        product_type : str
            One of ``"active"``, ``"passive"``, or ``"combined"``.  Used to
            set the correct physical units on the output variable and to
            build the ``source`` attribute string.

        Returns
        -------
        xr.Dataset or None
            Dataset with ``data_type="cds_ssm"`` and ``platform_type="cds_ssm"``,
            or None on failure.
        """
        if product_type not in ("active", "passive", "combined"):
            logger.warning(
                "from_c3s_ssm: unknown product_type %r (expected active/passive/combined).",
                product_type,
            )
            return None

        nc_path = Path(nc_path)
        if not nc_path.exists():
            logger.warning("C3S SSM file not found: %s", nc_path)
            return None

        try:
            raw = xr.open_dataset(nc_path)
        except Exception as exc:
            logger.warning("Could not open C3S SSM file %s: %s", nc_path, exc)
            return None

        if "sm" not in raw.variables:
            logger.warning(
                "Missing 'sm' variable in %s (available: %s).",
                nc_path.name, list(raw.variables),
            )
            raw.close()
            return None

        # Units depend on product type:
        #   active   → [%] (percent saturation, ASCAT)
        #   passive  → [m3 m-3] (volumetric, radiometers)
        #   combined → [m3 m-3] (merged, follows passive convention)
        if product_type == "active":
            units = "%"
            long_name = "C3S CDS active (ASCAT) surface soil moisture composite"
            source_label = "C3S CDS ACTIVE SSM (multi-ASCAT 0.25°)"
        elif product_type == "passive":
            units = "m3 m-3"
            long_name = "C3S CDS passive (radiometer) surface soil moisture composite"
            source_label = "C3S CDS PASSIVE SSM (multi-radiometer 0.25°)"
        else:
            units = "m3 m-3"
            long_name = "C3S CDS combined (active+passive) surface soil moisture composite"
            source_label = "C3S CDS COMBINED SSM (0.25°)"

        sm_da = raw["sm"]

        # CDS files may have a leading time dimension of length 1; squeeze it.
        if "time" in sm_da.dims:
            time_coord = pd.to_datetime(raw["time"].values[0]) if raw["time"].size > 0 else pd.NaT
            sm_da = sm_da.isel(time=0, drop=True) if sm_da.dims[0] == "time" else sm_da
        else:
            time_coord = pd.NaT

        # Apply CF scale/offset if present (CDS v202505 does not use them,
        # but guard here for future versions).
        scale = float(sm_da.attrs.get("scale_factor", 1.0))
        offset = float(sm_da.attrs.get("add_offset", 0.0))
        fill = sm_da.attrs.get("_FillValue", None)

        sm_vals = sm_da.values.astype(float)
        if fill is not None:
            sm_vals = np.where(sm_vals == float(fill), np.nan, sm_vals)
        sm_vals = sm_vals * scale + offset

        # Build flat arrays; CDS dims are (lat, lon) after time squeeze.
        if "lat" in raw.coords and "lon" in raw.coords:
            lon_1d = raw["lon"].values.astype(float)
            lat_1d = raw["lat"].values.astype(float)
        elif "latitude" in raw.coords and "longitude" in raw.coords:
            lon_1d = raw["longitude"].values.astype(float)
            lat_1d = raw["latitude"].values.astype(float)
        else:
            logger.warning(
                "Cannot find lat/lon coordinates in %s (coords: %s).",
                nc_path.name, list(raw.coords),
            )
            raw.close()
            return None

        lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
        sm_flat = sm_vals.ravel()
        lon_flat = lon2d.ravel()
        lat_flat = lat2d.ravel()

        valid = ~np.isnan(sm_flat)
        if not valid.any():
            logger.warning("from_c3s_ssm: all cells NaN in %s.", nc_path.name)
            raw.close()
            return None

        n_valid = int(valid.sum())
        _ts = np.datetime64(time_coord, "ns") if time_coord is not pd.NaT else np.datetime64("NaT", "ns")
        time_arr = np.full(n_valid, _ts)

        var_attrs = {
            "SOIL_MOISTURE": {
                "units": units,
                "long_name": long_name,
            }
        }
        ds = DataTreeConverter._build_ssm_point_dataset(
            sm_flat[valid], lon_flat[valid], lat_flat[valid], time_arr,
            data_type="cds_ssm", var_attrs=var_attrs,
            platform_type="cds_ssm",
            source=source_label,
            sensing_depth_cm="0-5", band="multi", filename=nc_path.name,
            native_grid_deg=0.25,
        )
        raw.close()
        return ds

    @staticmethod
    def from_hycom(
        nc_paths: Union[str, Path, Sequence[Union[str, Path]]],
    ) -> Optional[xr.Dataset]:
        """
        Open one or more HyCOM segment NetCDF files (as downloaded by
        :class:`~sar_validation.downloaders.hycom_downloader.HycomDownloader`)
        and return one combined, GRIDDED Dataset (dims: ``time``, ``lat``,
        ``lon``) covering every requested segment.

        Unlike every other validation-source converter (except
        :meth:`from_era5`), the result is NOT flattened to a ``point``
        dimension -- ``ModelLayerCollocation`` interpolates this grid
        directly onto SAR pixel/point locations at collocation time.

        ``water_u``/``water_v`` are renamed to ``EWCT``/``NSCT`` here (not
        left raw, unlike ERA5 wind's ``u10``/``v10``) -- these are just
        renamed vector components, not a derived circular quantity, so
        renaming at conversion time is safe. This matches
        ``from_hf_radar_grid``'s existing ``water_u``/``water_v`` ->
        ``EWCT``/``NSCT`` convention.

        Parameters
        ----------
        nc_paths : Path or list of Path
            One or more HyCOM segment NetCDF files. Multiple files (e.g.
            a recipe window straddling the ESPC-D-V02/GOFS 3.1 cutover)
            are concatenated along ``time``.

        Returns
        -------
        xr.Dataset or None
            Dataset with ``data_type``/``platform_type`` set to
            ``"hycom"``, or None on failure.
        """
        paths = [Path(p) for p in ([nc_paths] if isinstance(nc_paths, (str, Path)) else nc_paths)]
        existing = sorted(p for p in paths if p.exists())
        if not existing:
            logger.warning("from_hycom: no files found among %s", paths)
            return None

        try:
            per_file = [xr.open_dataset(p) for p in existing]
            raw = per_file[0] if len(per_file) == 1 else xr.concat(per_file, dim="time")
            # `existing` is sorted ALPHABETICALLY by filename (above), not
            # chronologically -- HyCOM segment filenames embed the dataset
            # key right after the "hycom_" prefix ("hycom_espc_d_v02_..."
            # vs "hycom_gofs31_930_..."), so a straddling-cutover window's
            # ESPC-D-V02 file ('e' < 'g') sorts BEFORE its GOFS 3.1 file
            # even though ESPC-D-V02 is always the chronologically LATER
            # segment (only ever used at/after _HYCOM_CUTOVER_DATE -- see
            # hycom_downloader.py). xr.concat does not sort its inputs, so
            # without this the resulting time axis goes forward then jumps
            # backward at the cutover -- non-monotonic, which
            # model_collocation.py's np.searchsorted-based bracket search
            # has no correct behaviour for. sortby (not just a pre-sorted
            # `existing`) establishes the genuine invariant regardless of
            # input order, mirroring from_era5's own `sortby("lat")` fix
            # for CDS's descending latitude (see that method).
            raw = raw.sortby("time")
            raw = raw.load()
            for d in per_file:
                d.close()
        except Exception as exc:
            logger.warning("Could not open HyCOM file(s) %s: %s", paths, exc)
            return None

        missing = [v for v in ("water_u", "water_v") if v not in raw.variables]
        if missing:
            logger.warning(
                "from_hycom: missing variable(s) %s in %s (available: %s).",
                missing, paths, list(raw.variables),
            )
            raw.close()
            return None

        ewct = raw["water_u"].astype("float32")
        ewct.attrs.update({
            "units": "m s-1", "standard_name": "eastward_sea_water_velocity",
            "long_name": "HyCOM eastward sea water velocity (surface)",
        })
        nsct = raw["water_v"].astype("float32")
        nsct.attrs.update({
            "units": "m s-1", "standard_name": "northward_sea_water_velocity",
            "long_name": "HyCOM northward sea water velocity (surface)",
        })

        ds = xr.Dataset(
            {"EWCT": ewct, "NSCT": nsct},
            coords={"time": raw["time"], "lat": raw["lat"], "lon": raw["lon"]},
        )
        ds.attrs["data_type"] = "hycom"
        ds.attrs["platform_type"] = "hycom"
        ds.attrs["source"] = "HyCOM ocean model (surface currents)"
        raw.close()
        return ds

    @staticmethod
    def from_era5(
        nc_paths: Union[str, Path, Sequence[Union[str, Path]]],
        variable: str,
    ) -> Optional[xr.Dataset]:
        """
        Open one or more ERA5 daily NetCDF files (as downloaded by
        :class:`~sar_validation.downloaders.era5_downloader.ERA5Downloader`)
        and return one combined, GRIDDED Dataset (dims: ``time``, ``lat``,
        ``lon``) covering every requested day.

        Unlike every other validation-source converter, the result is NOT
        flattened to a ``point`` dimension -- the whole point of ERA5's
        collocation method (bilinear spatial + nearest-hour/hyperbolic
        temporal interpolation, see
        ``sar_validation.core.model_collocation``) is to interpolate this
        grid directly onto SAR pixel/point locations at collocation time.

        Parameters
        ----------
        nc_paths : Path or list of Path
            One or more ERA5 daily NetCDF files for the SAME *variable*.
            Multiple files are concatenated along ``time`` so the hyperbolic
            method can find bracketing hours across a day boundary.
        variable : str
            One of ``"wind"``, ``"waves"``, ``"soil_moisture"``.

        Returns
        -------
        xr.Dataset or None
            Dataset with ``data_type``/``platform_type`` set to
            ``"era5_wind"``/``"era5_waves"``/``"era5_soil_moisture"``, or
            None on failure.
        """
        if variable not in _ERA5_VARS:
            logger.warning("from_era5: unknown variable %r (expected wind/waves/soil_moisture).", variable)
            return None

        paths = [Path(p) for p in ([nc_paths] if isinstance(nc_paths, (str, Path)) else nc_paths)]
        existing = sorted(p for p in paths if p.exists())
        if not existing:
            logger.warning("from_era5: no files found among %s", paths)
            return None

        # xr.open_mfdataset requires the optional `dask` package, which
        # isn't a dependency of this project -- open each daily file
        # individually and concatenate along time instead, matching how
        # the rest of this codebase (e.g. from_c3s_ssm) avoids that
        # dependency.
        try:
            groups = _group_era5_paths_by_day(existing)
            per_day: List[xr.Dataset] = []
            for _, group_paths in sorted(groups.items()):
                if len(group_paths) == 1:
                    per_day.append(_normalize_era5_grib_coords(xr.open_dataset(group_paths[0])))
                else:
                    stitched = _stitch_antimeridian_window_files(group_paths)
                    if stitched is None:
                        for d in per_day:
                            d.close()
                        return None
                    per_day.append(_normalize_era5_grib_coords(stitched))

            raw = per_day[0] if len(per_day) == 1 else xr.concat(per_day, dim="time")
            # CDS always returns ERA5 latitude descending (north -> south,
            # e.g. 60.25, 60.00, ..., 34.75) -- model_collocation.py's
            # build_spatial_interpolator (a scipy RegularGridInterpolator)
            # requires a monotonic axis, and this toolbox relies on it
            # actually being ASCENDING (see that function's docstring).
            # scipy >= 1.10 also accepts descending axes transparently, so
            # this worked "by luck" on newer scipy -- sortby establishes a
            # genuinely ascending axis regardless of scipy version.
            raw = raw.sortby("lat")
            # ERA5 regional daily files are small (bbox-limited); load fully
            # into memory now so the Dataset returned below doesn't hold
            # lazy references into a file handle that's about to be closed.
            raw = raw.load()
            for d in per_day:
                d.close()
        except Exception as exc:
            logger.warning("Could not open ERA5 file(s) %s: %s", paths, exc)
            return None

        # lat/lon/time already normalized per-file by _normalize_era5_grib_coords
        # above (before concatenation, so "time" is guaranteed to be the
        # concat dim regardless of which raw name the CDS response used).

        spec = _ERA5_VARS[variable]
        missing = [v for v in spec["raw"] if v not in raw.variables]
        if missing:
            logger.warning(
                "from_era5: missing variable(s) %s in %s (available: %s).",
                missing, paths, list(raw.variables),
            )
            raw.close()
            return None

        data_vars = {}
        for var in spec["raw"]:
            da = raw[var].astype("float32")
            da.attrs.update(spec["cf"][var])
            data_vars[var] = da

        # Rename/derive to the canonical val_var codes _variable_map.py's
        # VARIABLE_PAIRS (and therefore statistics.py/visualization.py)
        # expect -- every OTHER wind/waves/soil_moisture validation source
        # is renamed to these same codes at conversion time (see e.g.
        # from_scatterometer_nc's WSPD/WDIR rename, from_radiometer_bytemap's
        # WindSat rotation). ERA5's raw CDS short names (swh, swvl1) never
        # matched them, so run_statistics() silently produced zero rows for
        # every era5_waves/era5_soil_moisture source -- confirmed against a
        # live CDS run 2026-08-07 (wind_era5.yaml: "no statistics produced").
        #
        # Wind is the one deliberate exception: u10/v10 are kept as raw
        # components here, NOT renamed/derived into WSPD/WDIR. WDIR is a
        # CIRCULAR quantity, and this Dataset is exactly what gets
        # bilinearly-spatially / hyperbolically-temporally interpolated at
        # collocation time (see model_collocation.py) -- interpolating an
        # already-derived direction as an ordinary linear scalar produces
        # wrong answers whenever the true value crosses the 0/360 seam
        # (e.g. blending 359 and 1 degrees naively yields ~180, not ~0).
        # model_collocation.py's `_derive_wind_wspd_wdir` instead derives
        # WSPD/WDIR from the FINAL, already-interpolated u10/v10 values,
        # after collocation -- so the eventual val_data/
        # collocation_results.nc output still ends up with the same
        # WSPD/WDIR columns every other wind validation source produces,
        # just computed at the right time. This is the one exception to
        # §2's "renamed at conversion time" invariant -- see
        # docs/design-choices.md §2 ("Canonical variable naming") and §5.7
        # ("ERA5 model validation").
        if variable == "waves":
            data_vars = {"VHM0": data_vars["swh"]}
        elif variable == "soil_moisture":
            data_vars = {"SOIL_MOISTURE": data_vars["swvl1"]}

        # land_sea_mask ("lsm" on the wire, requested only for wind -- see
        # era5_downloader.py's _CDS_VARIABLE_NAMES_BY_VARIABLE) is a
        # per-cell land-mask LOOKUP, not a per-hour model quantity to
        # interpolate/report at collocation points. It's kept as a
        # non-dimension COORDINATE (not a data_var), which means every
        # downstream consumer that iterates `era5_ds.data_vars`
        # (_model_values_at_points, _collocate_cell_averaging_grid in
        # sar_validation.core.model_collocation) already skips it
        # automatically -- it never leaks into val_data as a spurious
        # val_lsm statistics column. Collapsed from (time, lat, lon) to
        # (lat, lon) since it's time-invariant in reality (the CDS API
        # just echoes the same value at every requested hour).
        lsm_2d = None
        if variable == "wind" and "lsm" in raw.variables:
            lsm_da = raw["lsm"].astype("float32")
            if "time" in lsm_da.dims:
                lsm_da = lsm_da.isel(time=0, drop=True)
            lsm_2d = lsm_da

        ds = xr.Dataset(
            data_vars,
            coords={"time": raw["time"], "lat": raw["lat"], "lon": raw["lon"]},
        )
        if lsm_2d is not None:
            ds = ds.assign_coords(lsm=(("lat", "lon"), lsm_2d.values))
        ds.attrs["data_type"] = spec["data_type"]
        ds.attrs["platform_type"] = spec["data_type"]
        ds.attrs["source"] = f"ERA5 reanalysis ({variable}, Copernicus CDS)"
        raw.close()
        return ds

    @staticmethod
    def from_hf_radar_grid(
        nc_path: Union[str, Path],
        u_var: str = "water_u",
        v_var: str = "water_v",
        source_label: str = "NOAA HFRnet RTV",
    ) -> Optional[xr.Dataset]:
        """
        Open a gridded HF-radar-current NetCDF (dims ``time, lat, lon``) and
        return a standardised point-frame Dataset tagged
        ``data_type="hf_radar_grid"``.

        ``u_var``/``v_var`` name the eastward/northward current variables on
        the wire — NOAA HFRnet RTV ships ``water_u``/``water_v`` (the
        defaults); Copernicus Marine HFR radar-total products ship
        ``EWCT``/``NSCT`` directly.

        The regular grid is flattened to a ``point`` dimension (one point per
        cell per time) so it collocates through the ``layer_vs_layer`` path,
        exactly like the scatterometer converter. Ancillary uncertainty/QC
        fields are retained for reference: NOAA's ``DOPx``/``DOPy`` are
        combined into ``hfr_gdop``, its radial/site counts are kept as
        ``hfr_n_radials``/``hfr_n_sites``; Copernicus's ``GDOP`` is copied to
        ``hfr_gdop`` directly, ``EWCS``/``NSCS`` (the per-cell
        eastward/northward current standard deviations) to
        ``hfr_ewcs``/``hfr_nscs``, and each per-parameter QC flag
        (``CSPD_QC``, ``DDNS_QC``, ``GDOP_QC``, ``VART_QC``,
        ``POSITION_QC``) to its own ``hfr_qc_<param>`` field — these remain
        retained but unused. The overall ``QCflag`` is copied to ``hfr_qc``
        AND used to drop cells where it equals 4 ("bad"); NOAA's product has
        no equivalent flag (it filters upstream before publishing), so this
        has no effect on NOAA-sourced files.

        Returns
        -------
        xr.Dataset or None
            Dataset with ``data_type="hf_radar_grid"``, or None on failure.
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

        # Resolve coordinate names (tolerate lat/lon, latitude/longitude, LAT/LON).
        lat_name = next((n for n in ("lat", "latitude", "LAT") if n in raw.coords or n in raw), None)
        lon_name = next((n for n in ("lon", "longitude", "LON") if n in raw.coords or n in raw), None)
        time_name = next((n for n in ("time", "Time", "TIME") if n in raw.coords or n in raw), None)
        if not (lat_name and lon_name and u_var in raw and v_var in raw):
            logger.warning(
                "from_hf_radar_grid: %s missing lat/lon or %s/%s (have %s)",
                nc_path.name, u_var, v_var, list(raw.coords) + list(raw.data_vars),
            )
            raw.close()
            return None

        lats = np.asarray(raw[lat_name].values, dtype=float)
        lons = np.asarray(raw[lon_name].values, dtype=float)

        # Native grid resolution, derived from the file's own lat/lon
        # coordinate spacing (not assumed from a filename or hardcoded
        # constant) -- read downstream by collocation.py to set
        # aggregation_window_km per-node instead of a fixed 6.0. Same
        # degrees-to-km approximation collocation.py already uses
        # (km ~= deg * 111 * cos(latitude)), reversed.
        hfr_resolution_km: Optional[float] = None
        if len(lats) > 1 and len(lons) > 1:
            lat_spacing_deg = float(np.median(np.abs(np.diff(lats))))
            lon_spacing_deg = float(np.median(np.abs(np.diff(lons))))
            mean_lat = float(np.nanmean(lats))
            lat_spacing_km = lat_spacing_deg * 111.0
            lon_spacing_km = lon_spacing_deg * 111.0 * max(np.cos(np.radians(mean_lat)), 1e-6)
            if lat_spacing_km > 0 and lon_spacing_km > 0:
                hfr_resolution_km = (lat_spacing_km + lon_spacing_km) / 2.0

        if time_name is not None:
            times = pd.to_datetime(raw[time_name].values)
        else:
            times = pd.to_datetime([np.datetime64("NaT")])

        n_t, n_la, n_lo = len(times), len(lats), len(lons)

        # Broadcast (time, lat, lon) → flat point vectors.
        tt, la, lo = np.meshgrid(np.arange(n_t), lats, lons, indexing="ij")
        time_flat = np.repeat(times.values, n_la * n_lo)
        lat_flat = la.ravel()
        lon_flat = ((lo.ravel() + 180.0) % 360.0) - 180.0  # normalise to −180..180

        def _flat(varname):
            return np.asarray(raw[varname].values, dtype=float).reshape(n_t, n_la, n_lo).ravel()

        ewct = _flat(u_var)
        nsct = _flat(v_var)

        data_vars: Dict[str, tuple] = {
            "EWCT": ("point", ewct),
            "NSCT": ("point", nsct),
        }
        var_attrs: Dict[str, Dict] = {
            "EWCT": dict(raw[u_var].attrs),
            "NSCT": dict(raw[v_var].attrs),
        }

        # --- Retained ancillary fields (not used yet; design §3.7) ---
        if "DOPx" in raw and "DOPy" in raw:
            dopx, dopy = _flat("DOPx"), _flat("DOPy")
            data_vars["hfr_gdop"] = ("point", np.sqrt(dopx ** 2 + dopy ** 2))
            var_attrs["hfr_gdop"] = {
                "long_name": "geometric dilution of precision (sqrt(DOPx^2+DOPy^2))",
                "comment": "Retained for a future HF-radar QC/uncertainty filter.",
            }
        for src, dst in (("number_of_radials", "hfr_n_radials"),
                         ("number_of_sites", "hfr_n_sites")):
            if src in raw:
                data_vars[dst] = ("point", _flat(src))
                var_attrs[dst] = {"long_name": src.replace("_", " "),
                                  "comment": "Retained for a future HF-radar QC filter."}
        if "GDOP" in raw and "hfr_gdop" not in data_vars:
            data_vars["hfr_gdop"] = ("point", _flat("GDOP"))
            var_attrs["hfr_gdop"] = {
                "long_name": "geometric dilution of precision",
                "comment": "Retained for a future HF-radar QC/uncertainty filter.",
            }
        if "EWCS" in raw and "NSCS" in raw:
            data_vars["hfr_ewcs"] = ("point", _flat("EWCS"))
            data_vars["hfr_nscs"] = ("point", _flat("NSCS"))
            var_attrs["hfr_ewcs"] = {
                "long_name": "eastward current component std error", "units": "m s-1",
                "comment": "Retained for a future HF-radar QC/uncertainty filter.",
            }
            var_attrs["hfr_nscs"] = {
                "long_name": "northward current component std error", "units": "m s-1",
                "comment": "Retained for a future HF-radar QC/uncertainty filter.",
            }
        qc_flat = None
        if "QCflag" in raw:
            qc_flat = _flat("QCflag")
            data_vars["hfr_qc"] = ("point", qc_flat)
            var_attrs["hfr_qc"] = {
                "long_name": "HF-radar overall QC flag",
                "comment": "Cells where this equals 4 (\"bad\") are excluded below.",
            }
        # Per-parameter QC flags (Copernicus radar-total product): each one
        # is retained under its own field rather than folded into hfr_qc, so
        # a future QC phase can filter per-parameter instead of only on the
        # overall flag.
        for src, param in (
            ("CSPD_QC", "cspd"), ("DDNS_QC", "ddns"), ("GDOP_QC", "gdop"),
            ("VART_QC", "vart"), ("POSITION_QC", "position"),
        ):
            if src in raw:
                dst = f"hfr_qc_{param}"
                data_vars[dst] = ("point", _flat(src))
                var_attrs[dst] = {
                    "long_name": f"HF-radar {param} QC flag",
                    "comment": "Retained for a future HF-radar QC filter (design §3.7).",
                }

        # Drop points where both current components are NaN (masked
        # land/gaps), or where the overall QCflag marks the cell "bad" (4).
        # Per-parameter QC flags (CSPD_QC etc.) remain retained but unused.
        valid = np.isfinite(ewct) | np.isfinite(nsct)
        if qc_flat is not None:
            valid &= qc_flat != 4
        if not np.any(valid):
            logger.warning("from_hf_radar_grid: all cells NaN or QC-bad in %s.", nc_path.name)
            raw.close()
            return None

        ds = xr.Dataset(
            {k: ("point", v[valid]) for k, (_, v) in data_vars.items()},
            coords={
                "lon": ("point", lon_flat[valid]),
                "lat": ("point", lat_flat[valid]),
                "time": ("point", time_flat[valid]),
            },
        )
        for gattr in ("title", "institution"):
            if raw.attrs.get(gattr):
                ds.attrs[gattr] = str(raw.attrs[gattr])
        apply_cf_metadata(ds, "hf_radar", var_attrs)

        ds.attrs["data_type"]     = "hf_radar_grid"
        ds.attrs["platform_type"] = "radar"
        ds.attrs["source"]        = source_label
        ds.attrs["filename"]      = nc_path.name

        if hfr_resolution_km is not None:
            ds.attrs["hfr_resolution_km"] = hfr_resolution_km

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

        # Optional wind direction (polarimetric sensors like WindSat; AMSR2
        # has none). NetCDF direction is assumed already meteorological.
        wdir_full = None
        wdir_attrs = None
        for dname in ("wind_direction", "wind_dir"):
            if dname in raw and raw[dname].shape == wind.shape:
                wdir_full = raw[dname].values.ravel().astype(float)
                wdir_attrs = dict(raw[dname].attrs)
                break

        # Sensor tag (e.g. "amsr2") for the per-sensor collocation spec.
        sensor = str(raw.attrs.get("sensor", "")).strip().lower()
        if not sensor:
            m = re.search(r"RSS_([A-Za-z0-9]+)_", nc_path.name)
            sensor = m.group(1).lower() if m else "unknown"
        extra = {g: str(raw.attrs[g]) for g in ("title", "institution") if raw.attrs.get(g)}
        wspd_attrs = dict(wind.attrs)
        raw.close()

        return DataTreeConverter._finalize_radiometer_points(
            lon_full, lat_full, time_full, wspd_full, sensor,
            source=f"RSS radiometer ({sensor.upper()})", filename=nc_path.name,
            wdir_full=wdir_full, wspd_attrs=wspd_attrs, wdir_attrs=wdir_attrs,
            extra_global_attrs=extra, log_label=f"(nc wspd_var={wspd_name})",
        )

    @staticmethod
    def _finalize_radiometer_points(
        lon_full, lat_full, time_full, wspd_full, sensor, *,
        source: str, filename: str, wdir_full=None,
        wspd_attrs=None, wdir_attrs=None, extra_global_attrs=None, log_label: str = "",
    ) -> Optional[xr.Dataset]:
        """
        Shared tail for the radiometer converters.

        Normalises longitude to −180…180, drops cells without a valid wind
        retrieval or timestamp, builds the standardised ``point`` Dataset (with
        canonical ``WSPD`` and optional ``WDIR``), tags it, and stamps CF
        metadata. Both :meth:`from_radiometer_nc` (NetCDF) and
        :meth:`from_radiometer_bytemap` (RSS binary) funnel through here so the
        two formats produce identical node structure.
        """
        lon_full = ((np.asarray(lon_full, float) + 180) % 360) - 180
        lat_full = np.asarray(lat_full, float)
        wspd_full = np.asarray(wspd_full, float)
        time_full = np.asarray(time_full)

        # Keep only cells with a valid wind retrieval and a valid time. Wind is
        # already NaN over land/ice/rain (or a masked special code), so this
        # also removes those.
        keep = np.isfinite(wspd_full) & ~np.isnat(time_full)
        n_keep = int(keep.sum())
        if n_keep == 0:
            logger.info("from_radiometer: no valid wind cells in %s.", filename)
            return None

        data_vars: Dict[str, tuple] = {"WSPD": ("point", wspd_full[keep])}
        var_attrs: Dict[str, Dict] = {"WSPD": dict(wspd_attrs or {})}
        if wdir_full is not None:
            data_vars["WDIR"] = ("point", np.asarray(wdir_full, float)[keep])
            var_attrs["WDIR"] = dict(wdir_attrs or {})

        ds = xr.Dataset(
            data_vars,
            coords={
                "lon":  ("point", lon_full[keep]),
                "lat":  ("point", lat_full[keep]),
                "time": ("point", time_full[keep]),
            },
        )
        for gattr, val in (extra_global_attrs or {}).items():
            ds.attrs[gattr] = val
        apply_cf_metadata(ds, "radiometer", var_attrs)

        ds.attrs["data_type"]     = "radiometer"
        ds.attrs["platform_type"] = "radiometer"
        ds.attrs["sensor"]        = sensor
        ds.attrs["source"]        = source
        ds.attrs["filename"]      = filename

        logger.info(
            "from_radiometer: %s → %d valid wind points (sensor=%s) %s",
            filename, n_keep, sensor, log_label,
        )
        return ds

    #: Filename-prefix → sensor key for RSS binary bytemap products.
    _BYTEMAP_PREFIX_TO_SENSOR = {
        "f35":  "gmi",
        "f16":  "ssmis_f16",
        "f17":  "ssmis_f17",
        "f18":  "ssmis_f18",
        "wsat": "windsat",
    }

    @staticmethod
    def bytemap_sensor_from_filename(name: Union[str, Path]) -> Optional[str]:
        """Infer the RSS bytemap sensor key from a filename prefix (e.g.
        ``f35_20240601v8.2.gz`` → ``"gmi"``)."""
        prefix = Path(name).name.split("_")[0].lower()
        return DataTreeConverter._BYTEMAP_PREFIX_TO_SENSOR.get(prefix)

    @staticmethod
    def from_radiometer_bytemap(
        path: Union[str, Path],
        sensor: Optional[str] = None,
    ) -> Optional[xr.Dataset]:
        """
        Decode an RSS binary bytemap radiometer file (GMI / SSMIS / WindSat)
        and return the same standardised ``point`` Dataset as
        :meth:`from_radiometer_nc`.

        The gzipped 0.25° grid is decoded by
        :func:`sar_validation.downloaders._rss_bytemap.read_rss_bytemap`, then
        flattened to points. Per-cell time is built from the filename date plus
        the file's time-of-day variable (fractional hours, or minutes for
        WindSat). WindSat's wind direction is rotated 180°
        (oceanographic → meteorological) so it matches ``owiWindDirection`` and
        the in-situ ``WDIR`` code — reusing the convention applied in
        :meth:`from_scatterometer_nc`.

        Parameters
        ----------
        path : str or Path
            Path to the ``.gz`` bytemap file.
        sensor : str, optional
            Sensor key; inferred from the filename prefix when omitted.

        Returns
        -------
        xr.Dataset or None
            Dataset with ``data_type="radiometer"``, or None on failure.
        """
        from ..downloaders._rss_bytemap import BYTEMAP_LAYOUT, read_rss_bytemap

        path = Path(path)
        if not path.exists():
            logger.warning("Bytemap not found: %s", path)
            return None
        if sensor is None:
            sensor = DataTreeConverter.bytemap_sensor_from_filename(path.name)
        if sensor is None or sensor not in BYTEMAP_LAYOUT:
            logger.warning(
                "from_radiometer_bytemap: cannot resolve sensor for %s.", path.name
            )
            return None

        try:
            decoded, lon1d, lat1d = read_rss_bytemap(path, sensor)
        except Exception as exc:
            logger.warning("from_radiometer_bytemap: failed to read %s: %s", path.name, exc)
            return None

        layout = BYTEMAP_LAYOUT[sensor]
        wind = decoded[layout["wind"]]  # (pass, lat, lon)

        lat_grid, lon_grid = np.meshgrid(lat1d, lon1d, indexing="ij")
        lat_full = np.broadcast_to(lat_grid, wind.shape).ravel()
        lon_full = np.broadcast_to(lon_grid, wind.shape).ravel()
        wspd_full = wind.ravel()

        # Per-cell time = filename date + time-of-day (hours or minutes).
        tod = decoded[layout["time"]].ravel()
        factor_ns = {"hours": 3_600_000_000_000, "minutes": 60_000_000_000}[layout["time_unit"]]
        m = re.search(r"(\d{8})", path.stem)
        if m:
            d = m.group(1)
            base = np.datetime64(f"{d[0:4]}-{d[4:6]}-{d[6:8]}", "ns")
            offset = (np.nan_to_num(tod) * factor_ns).astype("timedelta64[ns]")
            time_full = np.where(
                np.isfinite(tod), base + offset, np.datetime64("NaT", "ns")
            )
        else:
            time_full = np.full(wspd_full.shape, np.datetime64("NaT", "ns"))

        # WindSat wind direction → canonical WDIR, rotated oceanographic →
        # meteorological (see from_scatterometer_nc).
        wdir_full = None
        wdir_attrs = None
        if layout.get("wdir"):
            wdir_full = (decoded[layout["wdir"]].ravel() + 180.0) % 360.0
            wdir_attrs = {
                "standard_name": "wind_from_direction",
                "long_name": "wind direction at 10 m (meteorological convention)",
                "units": "degree",
                "comment": (
                    "Rotated 180 degrees from the RSS WindSat oceanographic "
                    "convention by sar-l2-validation-toolbox."
                ),
            }

        return DataTreeConverter._finalize_radiometer_points(
            lon_full, lat_full, time_full, wspd_full, sensor,
            source=f"RSS radiometer bytemap ({sensor.upper()})", filename=path.name,
            wdir_full=wdir_full,
            wspd_attrs={
                "standard_name": "wind_speed",
                "long_name": f"{layout['wind']} 10 m wind speed",
                "units": "m s-1",
            },
            wdir_attrs=wdir_attrs,
            log_label=f"(bytemap wind={layout['wind']})",
        )

    @staticmethod
    def to_datatree(
        datasets: Mapping[str, Optional[xr.Dataset]],
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
        - If directory name contains "WV": reads RVL point measurements for currents, oswTotalHs for wind/waves
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
            WV mode extracts RVL for currents, oswTotalHs for wind/waves.
            Default is "wind".

        Returns
        -------
        xr.Dataset or None
            Dataset with oswTotalHs points (WV) or product-specific grids (IW/EW/SM),
            or None if no suitable data found.
        """
        safe_dir = Path(safe_dir)
        safe_name = safe_dir.name.upper()

        # Detect mode from SAFE directory name
        if "WV" in safe_name:
            # WV vignette OCN files carry oswTotalHs AND a 13x13 rvlRadVel grid.
            # Route currents to RVL extraction; wind/waves keep oswTotalHs.
            if product_type.lower() == "currents":
                return DataTreeConverter._extract_rvl_from_wv_safe(safe_dir)
            return DataTreeConverter.from_sar_l2_ocn_wv_safe(safe_dir)
        else:
            return DataTreeConverter._from_sar_l2_ocn_iw_safe(safe_dir, product_type=product_type)

    @staticmethod
    def from_sar_l2_ocn_wv_safe(
        safe_dir: Union[str, Path],
    ) -> Optional[xr.Dataset]:
        """
        Open Wave Mode (WV) data from one Sentinel-1 SAFE directory and extract
        oswTotalHs (integrated total significant wave height) point measurements.

        The WV mode produces multiple measurement files (~16 per SAFE product),
        each carrying a 1×1 vignette. This method extracts ``oswTotalHs`` (the
        integrated total significant wave height, matching the validation
        ``VHM0``) from every .nc file — falling back to the mean of the valid
        ``oswHs`` partitions when a product lacks ``oswTotalHs`` — and creates a
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

        # Extract oswTotalHs point measurements from all files
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

                # Wave height to validate against VHM0 = the product's
                # integrated total significant wave height (oswTotalHs), which
                # combines all wave systems. This is NOT the same as an
                # individual oswHs partition (oswHs holds one Hs per partition),
                # and it is not the root-sum-square of the partitions either —
                # it is integrated from the full spectrum. If a (legacy)
                # product lacks oswTotalHs, fall back to the mean of the valid
                # oswHs partitions (dropping the -1/NaN fill codes) rather than
                # picking a single partition.
                if "oswTotalHs" in ds_raw:
                    hs_val = ds_raw["oswTotalHs"].values.item()
                    hs = float(hs_val) if np.isfinite(hs_val) else np.nan
                    if not osw_attrs:
                        osw_attrs = {"oswTotalHs": dict(ds_raw["oswTotalHs"].attrs)}
                else:
                    parts = np.asarray(ds_raw["oswHs"].values, dtype=float).ravel()
                    valid = parts[np.isfinite(parts) & (parts > 0)]
                    hs = float(valid.mean()) if valid.size else np.nan
                    if not osw_attrs:
                        osw_attrs = {"oswTotalHs": {
                            "long_name": "total significant wave height "
                                         "(mean of oswHs partitions; oswTotalHs absent)",
                            "units": ds_raw["oswHs"].attrs.get("units", "m"),
                        }}
                    logger.debug(
                        "oswTotalHs absent in %s; using mean of %d valid oswHs "
                        "partitions", nc_path.name, valid.size,
                    )

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
            logger.warning("No valid oswTotalHs data extracted from %s", measurement_dir)
            return None

        # Create Dataset with point dimension
        data_vars = {
            "oswTotalHs": (["point"], point_hs),
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
        ds.attrs["measurement_type"] = "oswTotalHs"
        ds.attrs["num_points"] = len(point_hs)

        logger.info(
            "Extracted %d oswTotalHs points from WV product %s",
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
        keeps the grid structure using ("y", "x") dimensions. If flatten_to_points=True,
        the grid is flattened to points (169 points per file) for collocation.

        Parameters
        ----------
        measurement_dir : Path
            Path to the measurement/ directory within a SAFE archive.
        safe_dir : str or Path
            Path to the SAFE directory (used for logging and attributes).
        flatten_to_points : bool, optional
            If True, flatten 13×13 grids to 169 points with point dimension (default: False).
            If False, keep 2D grid structure with ("y", "x") dimensions.

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

                # RVL is 3-D (rvlAzSize, rvlRaSize, rvlSwath) for multi-swath
                # modes (IW/EW). Concatenate the sub-swaths side by side along
                # the range axis so the grid keeps EVERY sub-swath — slicing
                # [:, :, 0] would silently drop all but the first swath
                # (4 of 5 for EW, 2 of 3 for IW). Single-swath products (SM)
                # are already 2-D and pass through.
                def _swaths_to_grid(arr):
                    # Lay sub-swaths side by side along the range axis, in
                    # swath-index order (== ground-range order: iw1→iw3,
                    # ew1→ew5). A plain C-order reshape would interleave
                    # sub-swath columns instead, scrambling grid adjacency
                    # and smearing pcolormesh quads across the whole swath.
                    arr = np.asarray(arr)
                    if arr.ndim != 3:
                        return arr
                    return np.concatenate(
                        [arr[:, :, k] for k in range(arr.shape[2])], axis=1
                    )

                rvl_radvel = _swaths_to_grid(rvl_radvel_full)
                rvl_lats = _swaths_to_grid(rvl_lats_full)
                rvl_lons = _swaths_to_grid(rvl_lons_full)

                rvl_heading = (
                    _swaths_to_grid(ds_raw["rvlHeading"].values)
                    if "rvlHeading" in ds_raw
                    else np.full_like(rvl_radvel, np.nan)
                )
                rvl_incidence = (
                    _swaths_to_grid(ds_raw["rvlIncidenceAngle"].values)
                    if "rvlIncidenceAngle" in ds_raw
                    else np.full_like(rvl_radvel, np.nan)
                )
                rvl_radvel_std = (
                    _swaths_to_grid(ds_raw["rvlRadVelStd"].values)
                    if "rvlRadVelStd" in ds_raw
                    else np.full_like(rvl_radvel, np.nan)
                )

                # Land-flag masking. rvlLandFlag is set to 1 where land
                # coverage of the cell exceeds 10%. Land-contaminated cells
                # must not feed into currents validation, so rvlRadVel and
                # rvlRadVelStd are NaN'd out there — but the pre-mask mean is
                # kept as a QA stat since it is expected to be ~0 and a
                # meaningfully non-zero value signals a data-quality issue.
                # rvlHeading/rvlIncidenceAngle are geometry, not
                # measurements, and are left unmasked.
                land_pixel_count = 0
                land_pixel_fraction = float("nan")
                land_mean_radvel = float("nan")
                if "rvlLandFlag" in ds_raw:
                    rvl_landflag = _swaths_to_grid(ds_raw["rvlLandFlag"].values).astype(float)
                    land_mask = rvl_landflag == 1
                    total_classified = int(np.sum(~np.isnan(rvl_landflag)))
                    land_pixel_count = int(np.sum(land_mask))
                    if total_classified > 0:
                        land_pixel_fraction = land_pixel_count / total_classified
                    if land_pixel_count > 0:
                        land_mean_radvel = float(np.nanmean(rvl_radvel[land_mask]))
                        rvl_radvel = np.where(land_mask, np.nan, rvl_radvel)
                        rvl_radvel_std = np.where(land_mask, np.nan, rvl_radvel_std)
                        logger.warning(
                            "scene %s: %d/%d RVL cells land-flagged (%.1f%%) — "
                            "mean rvlRadVel over land = %.4f m/s (expected ~0)",
                            safe_dir.name, land_pixel_count, total_classified,
                            100 * land_pixel_fraction, land_mean_radvel,
                        )

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

                # Standard (y, x) naming to mirror the OWI grid, so is_wv_mode
                # detection and the grid collocation path treat RVL and OWI
                # grids identically.
                dims = ("y", "x")

                # Create Dataset with 2D grid structure
                data_vars: Dict[str, tuple] = {
                    "rvlRadVel": (dims, rvl_radvel),
                    "rvlRadVelStd": (dims, rvl_radvel_std),
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
                ds.attrs["rvl_land_pixel_count"] = land_pixel_count
                ds.attrs["rvl_land_pixel_fraction"] = land_pixel_fraction
                ds.attrs["rvl_land_mean_radvel"] = land_mean_radvel

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
            point_lons: list[float] = []
            point_lats: list[float] = []
            point_radvel: list[float] = []
            point_radvel_std: list[float] = []
            point_heading: list[float] = []
            point_incidence: list[float] = []
            point_times = []
            file_names = []
            rvl_attrs: Dict[str, Dict] = {}

            # Land-flag QA accumulated across every vignette file in this
            # scene (see the grid branch above for the rationale — same
            # masking rule, same QA stats, just summed across files here
            # since one WV scene is many small vignette files).
            land_pixel_count_total = 0
            total_classified_total = 0
            land_radvel_sum = 0.0
            land_radvel_finite_count_total = 0

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
                            for v in ("rvlRadVel", "rvlRadVelStd", "rvlHeading", "rvlIncidenceAngle")
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
                    rvl_radvel_std = (
                        ds_raw["rvlRadVelStd"].values.ravel()
                        if "rvlRadVelStd" in ds_raw
                        else np.full_like(rvl_radvel, np.nan)
                    )

                    # Land-flag masking (see the grid branch for rationale).
                    # rvlHeading/rvlIncidenceAngle are left unmasked.
                    if "rvlLandFlag" in ds_raw:
                        rvl_landflag = ds_raw["rvlLandFlag"].values.ravel().astype(float)
                        land_mask = rvl_landflag == 1
                        total_classified_total += int(np.sum(~np.isnan(rvl_landflag)))
                        file_land_count = int(np.sum(land_mask))
                        if file_land_count > 0:
                            land_pixel_count_total += file_land_count
                            land_radvel_sum += float(np.nansum(rvl_radvel[land_mask]))
                            land_radvel_finite_count_total += int(
                                np.sum(land_mask & ~np.isnan(rvl_radvel))
                            )
                            rvl_radvel = np.where(land_mask, np.nan, rvl_radvel)
                            rvl_radvel_std = np.where(land_mask, np.nan, rvl_radvel_std)

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
                    point_radvel_std.extend(rvl_radvel_std)
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
                "rvlRadVel": (("point",), np.asarray(point_radvel)),
                "rvlRadVelStd": (("point",), np.asarray(point_radvel_std)),
                "rvlHeading": (("point",), np.asarray(point_heading)),
                "rvlIncidenceAngle": (("point",), np.asarray(point_incidence)),
            }

            coords = {
                "lon": (["point"], point_lons),
                "lat": (["point"], point_lats),
                "time": (["point"], point_times),
                "filename": (["point"], file_names),
            }

            land_pixel_fraction = (
                land_pixel_count_total / total_classified_total
                if total_classified_total > 0 else float("nan")
            )
            land_mean_radvel = (
                land_radvel_sum / land_radvel_finite_count_total
                if land_radvel_finite_count_total > 0 else float("nan")
            )
            if land_pixel_count_total > 0:
                logger.warning(
                    "scene %s: %d/%d RVL cells land-flagged (%.1f%%) — "
                    "mean rvlRadVel over land = %.4f m/s (expected ~0)",
                    safe_dir.name, land_pixel_count_total, total_classified_total,
                    100 * land_pixel_fraction, land_mean_radvel,
                )

            ds = xr.Dataset(data_vars, coords=coords)
            apply_cf_metadata(ds, "sar", rvl_attrs)
            ds.attrs["data_type"] = "sar_l2_ocn"
            ds.attrs["source"] = "Sentinel-1"
            ds.attrs["safe_dir"] = safe_dir.name
            ds.attrs["measurement_type"] = "rvl"
            ds.attrs["num_points"] = len(point_radvel)
            ds.attrs["rvl_land_pixel_count"] = land_pixel_count_total
            ds.attrs["rvl_land_pixel_fraction"] = land_pixel_fraction
            ds.attrs["rvl_land_mean_radvel"] = land_mean_radvel

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
                ds_raw["owiNrcs"].values
                if "owiNrcs" in ds_raw
                else np.full_like(owi_windspeed, np.nan)
            )
            # owiNrcs is (owiAzSize, owiRaSize, owiPolarisation) — keep the
            # co-pol channel. Guard the slice so a product that ships a 2-D
            # owiNrcs does not raise IndexError (which the broad except below
            # would swallow, silently dropping the entire wind grid).
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

            # Land-flag masking. owiMask is a CF bitmask (flag_values
            # 0/1/2/4/8 = valid/land/ice/no_data/rfi) whose bits combine via
            # bitwise OR (e.g. 5 = land + no_data simultaneously). Land
            # pixels must not feed into wind validation, so owiWindSpeed and
            # owiWindDirection (a direction without a valid speed is
            # meaningless) are NaN'd out there. Only the land bit (1) is
            # checked -- ice/no_data/rfi are intentionally left unfiltered.
            owi_land_pixel_count = 0
            owi_land_pixel_fraction = float("nan")
            if "owiMask" in ds_raw:
                land_mask = (owi_mask & 1) != 0
                owi_land_pixel_count = int(np.sum(land_mask))
                owi_land_pixel_fraction = owi_land_pixel_count / land_mask.size
                if owi_land_pixel_count > 0:
                    owi_windspeed = np.where(land_mask, np.nan, owi_windspeed)
                    owi_winddir = np.where(land_mask, np.nan, owi_winddir)
                    logger.warning(
                        "scene %s: %d/%d OWI cells land-flagged (%.1f%%) via "
                        "owiMask -- owiWindSpeed/owiWindDirection NaN'd out",
                        safe_dir.name, owi_land_pixel_count, land_mask.size,
                        100 * owi_land_pixel_fraction,
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

            # Standard (y, x) naming for the flattened OWI grid.
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
            ds.attrs["owi_land_pixel_count"] = owi_land_pixel_count
            ds.attrs["owi_land_pixel_fraction"] = owi_land_pixel_fraction

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
            # RVL is the currents observable. Do NOT fall back to OWI wind — a
            # currents run must never silently produce wind data. If no RVL is
            # found, skip the scene (the caller drops None nodes) with a warning.
            ds_rvl = DataTreeConverter._extract_rvl_grid_data(
                measurement_dir, safe_dir, flatten_to_points=False
            )
            if ds_rvl is not None:
                ds_rvl.attrs["swath_mode"] = "IW/EW/SM"
                logger.info("Extracted RVL data from IW/EW/SM product %s", safe_dir.name)
                return ds_rvl
            logger.warning(
                "scene %s: no RVL/currents data — skipping (no OWI fallback for currents)",
                safe_dir.name,
            )
            return None

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
        - ``S1_L3_SSM/*.tif``          → ``sar/<stem>`` nodes
        - ``copernicus_insitu/*.csv``  → ``validation/<stem>`` nodes
        - ``osi_saf_winds/*.nc``       → ``validation/osi_saf_winds/<stem>`` nodes
        - ``scatterometer/*.nc``       → ``validation/scatterometer/<stem>`` nodes
        - ``scatterometer_hy2b/*.nc``       → ``validation/scatterometer_hy2b/<stem>`` nodes
        - ``scatterometer_hy2c/*.nc``       → ``validation/scatterometer_hy2c/<stem>`` nodes
        - ``scatterometer_oceansat3/*.nc``  → ``validation/scatterometer_oceansat3/<stem>`` nodes
        - ``hfr_noaa/*.nc``            → ``validation/hfr_noaa/<stem>`` nodes
        - ``hf_radar/*.nc``            → ``validation/hf_radar/<stem>`` nodes
        - ``hf_radar_historical/*.nc`` → ``validation/hf_radar_historical/<stem>`` nodes
        - ``adcp_historical/*.csv``    → ``validation/adcp_historical/<stem>`` nodes
        - ``argo_historical/*.csv``    → ``validation/argo_historical/<stem>`` nodes
        - ``drifter_historical/*.csv`` → ``validation/drifter_historical/<stem>`` nodes
        - ``glider_historical/*.csv``  → ``validation/glider_historical/<stem>`` nodes
        - ``ismn/*.csv``                → ``validation/ismn/<stem>`` nodes
        - ``ascat_ssm/*``               → ``validation/ascat_ssm/<stem>`` nodes
        - ``amsr_ssm/*.h5``             → ``validation/amsr_ssm/<stem>`` nodes
        - ``smap_ssm/*.h5``             → ``validation/smap_ssm/<stem>`` nodes
        - ``smos_ssm/*.nc``             → ``validation/smos_ssm/<stem>`` nodes
        - ``cds_ssm/*.nc``              → ``validation/cds_ssm/<stem>`` nodes
        - ``altimeter/*.nc``           → ``validation/altimeter/<stem>`` nodes
        - ``era5/*.nc``                 → single combined, GRIDDED ``validation/era5/era5`` node
        - ``hycom/*.nc``                → single combined, GRIDDED ``validation/hycom/hycom`` node

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

        # SAR data -- only the recipe's chosen source's own subdirectory is
        # scanned, so a stale sibling folder from a previous run using a
        # different source (e.g. S1_L3_SSM left over when this run wants
        # sentinel1_l2_ocn) is never picked up. See design-choices.md §8.11.
        if recipe is not None:
            from .sar_sources import SAR_SOURCES

            spec = SAR_SOURCES[recipe.config.sar_data.source]
            sar_subdir = base_dir / spec.output_subdir
            sar_paths: list[Path] = []
            if sar_subdir.exists():
                if spec.key == "sentinel1_clms_ssm":
                    from ..downloaders.sentinel1_soil_moisture_downloader import _SSM_FILENAME_MARKER
                    sar_paths = sorted(
                        p for p in list(sar_subdir.rglob("*.tif")) + list(sar_subdir.rglob("*.tiff"))
                        if _SSM_FILENAME_MARKER in p.name
                    )
                else:
                    sar_paths = sorted(sar_subdir.rglob(spec.file_glob))
            for sar_path in sar_paths:
                ds = spec.convert(sar_path, product_type)
                if ds is not None:
                    datasets[f"sar/{sar_path.stem}"] = ds
                    logger.info("Converted SAR product (%s): %s", spec.key, sar_path.name)
        else:
            # Legacy/test-only fallback: no recipe given, scan every known
            # SAR-shaped folder. Never exercised by the real pipeline --
            # cli.py always passes recipe.
            sar_dir = base_dir / "S1_L2_OCN"
            if sar_dir.exists():
                for safe_dir in sorted(d for d in sar_dir.iterdir()
                                       if d.is_dir() and d.suffix == ".SAFE"):
                    ds = DataTreeConverter.from_sar_l2_ocn_safe(safe_dir, product_type=product_type)
                    if ds is not None:
                        datasets[f"sar/{safe_dir.name}"] = ds
                        logger.info("Converted SAR SAFE: %s", safe_dir.name)

            ssm_dir = base_dir / "S1_L3_SSM"
            if ssm_dir.exists():
                from ..downloaders.sentinel1_soil_moisture_downloader import _SSM_FILENAME_MARKER

                tif_paths = [
                    p for p in list(ssm_dir.rglob("*.tif")) + list(ssm_dir.rglob("*.tiff"))
                    if _SSM_FILENAME_MARKER in p.name
                ]
                for tif_path in sorted(tif_paths):
                    ds = DataTreeConverter.from_sar_l3_ssm_geotiff(tif_path)
                    if ds is not None:
                        datasets[f"sar/{tif_path.stem}"] = ds
                        logger.info("Converted SSM GeoTIFF: %s", tif_path.name)

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

        # Delayed-mode in-situ current observations (Copernicus Marine
        # 013_044) — ADCP/Argo/drifter/glider, one dedicated folder per
        # instrument so provenance is distinguishable from the NRT
        # `copernicus_insitu` block above (which lumps everything under a
        # generic "insitu" source_type label).
        for instrument in (
            "adcp_historical", "argo_historical", "drifter_historical", "glider_historical",
        ):
            subdir = base_dir / instrument
            if subdir.exists():
                for csv_path in sorted(subdir.glob("*.csv")):
                    ds = _filtered(
                        DataTreeConverter.from_insitu_csv(csv_path, source_type=instrument),
                        csv_path.name,
                    )
                    if ds is not None:
                        datasets[f"validation/{instrument}/{csv_path.stem}"] = ds
                        logger.info("Converted %s CSV: %s", instrument, csv_path.name)

        # Manually-downloaded ISMN soil-moisture station CSVs (same
        # long-format schema as the Copernicus Marine in-situ CSVs above —
        # from_insitu_csv needs no changes). Not domain-filtered like the
        # satellite layer sources below: ISMNDownloader already writes only
        # in-bbox/in-window stations.
        ismn_dir = base_dir / "ismn"
        if ismn_dir.exists():
            for csv_path in sorted(ismn_dir.glob("*.csv")):
                ds = DataTreeConverter.from_insitu_csv(csv_path, source_type="ismn")
                if ds is not None:
                    datasets[f"validation/ismn/{csv_path.stem}"] = ds
                    logger.info("Converted ISMN CSV: %s", csv_path.name)

        # Scatterometer / OSI-SAF winds (standardised to point dimension).
        # scatterometer_hy2b/hy2c/oceansat3 are the KNMI OSI-SAF FTP,
        # recent-only 25km sources; from_scatterometer_nc handles them
        # unchanged (verified against real sample files — see design doc).
        for subdir_name in (
            "osi_saf_winds", "scatterometer",
            "scatterometer_hy2b", "scatterometer_hy2c", "scatterometer_oceansat3",
        ):
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

        # ASCAT Soil Moisture (SOMO12, historical-only — see design doc).
        # Files are read format-agnostically by the ``ascat`` package
        # (.nc/.bfr/.nat all funnel through from_ascat_ssm). EUMDAC delivers
        # each order alongside sidecar metadata files (confirmed against a
        # real download: EOPMetadata.xml, manifest.xml sitting in the same
        # flat directory as the .nat products) -- skip non-data extensions
        # rather than attempting (and noisily failing to) convert them.
        _ASCAT_DATA_SUFFIXES = (
            ".nc", ".nc.gz", ".bfr", ".bfr.gz", ".buf", ".buf.gz", ".nat", ".nat.gz",
        )
        subdir = base_dir / "ascat_ssm"
        if subdir.exists():
            for f in sorted(subdir.iterdir()):
                if not f.is_file():
                    continue
                name_lower = f.name.lower()
                if not name_lower.endswith(_ASCAT_DATA_SUFFIXES):
                    continue
                ds = _filtered(DataTreeConverter.from_ascat_ssm(f), f.name)
                if ds is not None:
                    datasets[f"validation/ascat_ssm/{f.stem}"] = ds
                    logger.info("Converted ASCAT SSM: %s", f.name)

        # H-SAF ASCAT SSM NRT (H29) -- flat obs-array netCDF files, always
        # ".nc" (see hsaf_downloader.py; H-SAF's own BUFR variant is never
        # requested by that downloader).
        subdir = base_dir / "hsaf_ssm"
        if subdir.exists():
            for f in sorted(subdir.glob("*.nc")):
                ds = _filtered(DataTreeConverter.from_hsaf_ssm(f), f.name)
                if ds is not None:
                    datasets[f"validation/hsaf_ssm/{f.stem}"] = ds
                    logger.info("Converted H-SAF SSM: %s", f.name)

        # AMSR-E/AMSR2 Daily Global Land Parameters (NSIDC-0451, HDF5, ``.h5``)
        # or AU_Land_NRT_R02/AU_Land (HDF-EOS5 swath, conventionally ``.he5``).
        subdir = base_dir / "amsr_ssm"
        if subdir.exists():
            for f in sorted(list(subdir.glob("*.h5")) + list(subdir.glob("*.he5"))):
                ds = _filtered(DataTreeConverter.from_amsr_ssm(f), f.name)
                if ds is not None:
                    datasets[f"validation/amsr_ssm/{f.stem}"] = ds
                    logger.info("Converted AMSR SSM: %s", f.name)

        # SMAP Enhanced L2 Radiometer Soil Moisture (SPL2SMP_E, HDF5).
        subdir = base_dir / "smap_ssm"
        if subdir.exists():
            for f in sorted(subdir.glob("*.h5")):
                ds = _filtered(DataTreeConverter.from_smap_ssm(f), f.name)
                if ds is not None:
                    datasets[f"validation/smap_ssm/{f.stem}"] = ds
                    logger.info("Converted SMAP SSM: %s", f.name)

        # SMOS soil moisture (ESA Online Dissemination FTPS, NetCDF).
        subdir = base_dir / "smos_ssm"
        if subdir.exists():
            nc_paths = sorted(subdir.glob("*.nc"))
            if not nc_paths and any(f.is_file() for f in subdir.iterdir()):
                # It is currently unconfirmed whether ESA serves this product
                # as NetCDF or as a .tgz archive (see from_smos_ssm's
                # docstring) -- SMOSDownloader accepts both and reports a
                # successful download either way, so a directory full of
                # .tgz (or any other non-.nc) files here would otherwise
                # silently yield zero SMOS collocations with no warning
                # surfaced anywhere.
                logger.warning(
                    "SMOS downloads present in %s but none match *.nc — if "
                    "these are .tgz archives, they are not yet extracted; "
                    "see from_smos_ssm's docstring.",
                    subdir,
                )
            for f in nc_paths:
                ds = _filtered(DataTreeConverter.from_smos_ssm(f), f.name)
                if ds is not None:
                    datasets[f"validation/smos_ssm/{f.stem}"] = ds
                    logger.info("Converted SMOS SSM: %s", f.name)

        # C3S CDS satellite soil moisture (Copernicus Climate Data Store,
        # global 0.25° daily grid, NetCDF). product_type ("active"/
        # "passive"/"combined") comes from the recipe's cds_ssm validation
        # source download_kwargs -- the same channel the orchestrator reads
        # to build the download request -- so units stay consistent with
        # what was actually downloaded. Defaults to "active" (matching
        # _download_cds_ssm's own default) when no recipe is given.
        subdir = base_dir / "cds_ssm"
        if subdir.exists():
            cds_product_type = "active"
            if recipe is not None:
                for src in recipe.config.validation_sources:
                    if src.source_type == "cds_ssm":
                        cds_product_type = src.download_kwargs.get("product_type", "active")
                        break
            for f in sorted(subdir.glob("*.nc")):
                ds = _filtered(DataTreeConverter.from_c3s_ssm(f, cds_product_type), f.name)
                if ds is not None:
                    datasets[f"validation/cds_ssm/{f.stem}"] = ds
                    logger.info("Converted C3S CDS SSM: %s", f.name)

        # NOAA HFRnet gridded RTV currents (flattened to points, tagged
        # hf_radar_grid). Domain-filtered like the scatterometer path.
        subdir = base_dir / "hfr_noaa"
        if subdir.exists():
            for nc_path in sorted(subdir.glob("*.nc")):
                ds = _filtered(
                    DataTreeConverter.from_hf_radar_grid(nc_path),
                    nc_path.name,
                )
                if ds is not None:
                    datasets[f"validation/hfr_noaa/{nc_path.stem}"] = ds
                    logger.info("Converted hfr_noaa (HF-radar grid): %s", nc_path.name)

        # Copernicus Marine HF-radar current grid (per-region radar-total
        # product). Same layer-node shape as hfr_noaa; EWCT/NSCT are already
        # the wire variable names so no rename is needed.
        subdir = base_dir / "hf_radar"
        if subdir.exists():
            for nc_path in sorted(subdir.glob("*.nc")):
                ds = _filtered(
                    DataTreeConverter.from_hf_radar_grid(
                        nc_path, u_var="EWCT", v_var="NSCT",
                        source_label="Copernicus Marine HFR radar-total",
                    ),
                    nc_path.name,
                )
                if ds is not None:
                    datasets[f"validation/hf_radar/{nc_path.stem}"] = ds
                    logger.info("Converted hf_radar (Copernicus HF-radar grid): %s", nc_path.name)

        # Copernicus Marine delayed-mode HF-radar current grid (013_044),
        # already normalized to the same shape as hf_radar/ by the downloader.
        subdir = base_dir / "hf_radar_historical"
        if subdir.exists():
            for nc_path in sorted(subdir.glob("*.nc")):
                ds = _filtered(
                    DataTreeConverter.from_hf_radar_grid(
                        nc_path, u_var="EWCT", v_var="NSCT",
                        source_label="Copernicus Marine HFR radar-total (delayed-mode)",
                    ),
                    nc_path.name,
                )
                if ds is not None:
                    datasets[f"validation/hf_radar_historical/{nc_path.stem}"] = ds
                    logger.info("Converted hf_radar_historical (Copernicus HF-radar grid): %s", nc_path.name)

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

        # Radiometer daily gridded products. Each file is a global 0.25° grid;
        # the converter flattens it to points and the domain filter crops to
        # the recipe bbox (>95% reduction, like the scatterometer). AMSR2 is
        # NetCDF (.nc → from_radiometer_nc); GMI/SSMIS/WindSat are RSS binary
        # bytemaps (.gz → from_radiometer_bytemap, sensor from filename prefix).
        subdir = base_dir / "radiometer"
        if subdir.exists():
            for f in sorted(list(subdir.glob("*.nc")) + list(subdir.glob("*.gz"))):
                if f.suffix == ".nc":
                    raw_ds = DataTreeConverter.from_radiometer_nc(f)
                else:
                    raw_ds = DataTreeConverter.from_radiometer_bytemap(f)
                ds = _filtered(raw_ds, f.name)
                if ds is not None:
                    datasets[f"validation/radiometer/{f.stem}"] = ds
                    logger.info("Converted radiometer: %s", f.name)

        # ERA5 reanalysis (Copernicus CDS) -- kept as a single combined,
        # GRIDDED node (not flattened to `point`, unlike every other
        # validation source) since ModelLayerCollocation interpolates it
        # directly onto SAR pixel locations at collocation time. Every
        # daily file downloaded for this run is opened together so the
        # hyperbolic method can find bracketing hours across a day
        # boundary. Not passed through _filtered() -- that helper assumes
        # a `point` dimension.
        subdir = base_dir / "era5"
        if subdir.exists():
            era5_variable = recipe.config.variable if recipe is not None else None
            era5_files = (
                sorted(subdir.glob(f"era5_{era5_variable}_*.nc"))
                if era5_variable is not None
                else []
            )
            if era5_files and era5_variable in ("wind", "waves", "soil_moisture"):
                ds = DataTreeConverter.from_era5(era5_files, era5_variable)
                if ds is not None:
                    datasets["validation/era5/era5"] = ds
                    logger.info("Converted ERA5 (%s): %d file(s)", era5_variable, len(era5_files))

        # HyCOM ocean model -- kept as a single combined, GRIDDED node
        # (not flattened to `point`), same rationale as ERA5. Only
        # relevant for currents recipes -- HyCOM has no wind/wave/soil-
        # moisture variable. Not passed through _filtered() -- that
        # helper assumes a `point` dimension.
        subdir = base_dir / "hycom"
        if subdir.exists():
            hycom_variable = recipe.config.variable if recipe is not None else None
            hycom_files = sorted(subdir.glob("hycom_*.nc")) if hycom_variable == "currents" else []
            if hycom_files:
                ds = DataTreeConverter.from_hycom(hycom_files)
                if ds is not None:
                    datasets["validation/hycom/hycom"] = ds
                    logger.info("Converted HyCOM: %d file(s)", len(hycom_files))

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
