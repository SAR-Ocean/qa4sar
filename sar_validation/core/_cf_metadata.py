"""
CF-convention metadata for datatree.nc and collocation_results.nc.

The raw input products are already CF-annotated (OSI-SAF scatterometer,
CMEMS altimeter L3, Sentinel-1 L2 OCN), so the converters capture each raw
variable's attributes and pass them through :func:`apply_cf_metadata`, which
sanitizes them and fills gaps from the tables below. The Copernicus in-situ
CSVs carry no attributes at all — their parameter codes are covered by
:data:`INSITU_VARIABLE_ATTRS`.

Product documentation for each source kind is recorded in the per-node
``references`` attribute (see :data:`PRODUCT_REFERENCES`).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import xarray as xr

__all__ = [
    "PRODUCT_REFERENCES",
    "COORD_ATTRS",
    "INSITU_VARIABLE_ATTRS",
    "sanitize_raw_attrs",
    "apply_cf_metadata",
    "annotate_collocation_ds",
]


#: Product documentation pages per source kind, stored in each node's
#: ``references`` global attribute.
PRODUCT_REFERENCES: Dict[str, str] = {
    "scatterometer": (
        "https://osi-saf.eumetsat.int/products/osi-104-b; "
        "https://osi-saf.eumetsat.int/products/osi-104-c"
    ),
    "scatterometer_ssm": "https://user.eumetsat.int/s3/eup-strapi-media/pdf_ten_0343_eps_ascatl2_pfs_f509981295.pdf",
    "altimeter": (
        "https://data.marine.copernicus.eu/product/"
        "WAVE_GLO_PHY_SWH_L3_NRT_014_001/services"
    ),
    "radiometer": "https://www.remss.com/missions/amsr/",
    "radiometer_ssm": "https://nsidc.org/data/nsidc-0451; https://nsidc.org/data/spl2smp_e",
    "hf_radar": "https://hfradar.ioos.us/",
    "sar": "https://s1.pages.eopf.copernicus.eu/s1-l12-rp/main/pfs/index.html",
    "insitu": (
        "https://data.marine.copernicus.eu/product/"
        "INSITU_GLO_PHYBGCWAV_DISCRETE_MYNRT_013_030/description"
    ),
}

#: CF attributes for the coordinate variables the converters emit. ``time``
#: deliberately has no ``units`` — xarray forbids a units attribute on
#: datetime64 variables (calendar/units belong to the on-disk encoding).
COORD_ATTRS: Dict[str, Dict[str, str]] = {
    "lon":   {"standard_name": "longitude", "long_name": "longitude", "units": "degrees_east"},
    "lat":   {"standard_name": "latitude",  "long_name": "latitude",  "units": "degrees_north"},
    "time":  {"standard_name": "time", "long_name": "time of observation"},
    "depth": {"standard_name": "depth", "long_name": "depth of observation",
              "units": "m", "positive": "down"},
}

#: CF attributes for the Copernicus Marine in-situ parameter codes used by
#: this toolbox (INSITU_GLO_PHYBGCWAV_DISCRETE_MYNRT_013_030). Also serves
#: as a fallback for the same codes produced by the scatterometer/altimeter
#: renames when the raw file carried no attributes.
INSITU_VARIABLE_ATTRS: Dict[str, Dict[str, str]] = {
    "WSPD": {"standard_name": "wind_speed",
             "long_name": "horizontal wind speed", "units": "m s-1"},
    "WDIR": {"standard_name": "wind_from_direction",
             "long_name": "wind direction (meteorological convention: direction the wind comes from)",
             "units": "degree"},
    "VHM0": {"standard_name": "sea_surface_wave_significant_height",
             "long_name": "spectral significant wave height (Hm0)", "units": "m"},
    "VAVH": {"standard_name": "sea_surface_wave_significant_height",
             "long_name": "significant wave height (H1/3)", "units": "m"},
    "VGHS": {"standard_name": "sea_surface_wave_significant_height",
             "long_name": "significant wave height", "units": "m"},
    "EWCT": {"standard_name": "eastward_sea_water_velocity",
             "long_name": "eastward current component", "units": "m s-1"},
    "NSCT": {"standard_name": "northward_sea_water_velocity",
             "long_name": "northward current component", "units": "m s-1"},
    "HCSP": {"standard_name": "sea_water_speed",
             "long_name": "horizontal current speed", "units": "m s-1"},
    "HCDT": {"standard_name": "sea_water_velocity_to_direction",
             "long_name": "current direction (direction the current flows towards)",
             "units": "degree"},
    "TEMP": {"standard_name": "sea_water_temperature",
             "long_name": "sea water temperature", "units": "degree_Celsius"},
    "PSAL": {"standard_name": "sea_water_practical_salinity",
             "long_name": "practical salinity", "units": "1e-3"},
    "SLEV": {"standard_name": "water_surface_height_above_reference_datum",
             "long_name": "observed sea level", "units": "m"},
    "SOIL_MOISTURE": {"standard_name": "volume_fraction_of_water_in_soil",
                      "long_name": "ISMN in-situ volumetric soil moisture",
                      "units": "m3 m-3"},
}

#: Descriptive attrs, kept when copying from a raw product variable.
_KEEP_RAW_ATTRS = ("standard_name", "long_name", "units", "comment")


def sanitize_raw_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reduce a raw variable's attributes to the descriptive CF subset.

    Packing/range attributes (``scale_factor``, ``add_offset``,
    ``valid_min``/``valid_max``, ``_FillValue``) are dropped — they describe
    the raw on-disk integer packing and are wrong for the unpacked float
    values the converters store.
    """
    return {k: attrs[k] for k in _KEEP_RAW_ATTRS if k in attrs}


def apply_cf_metadata(
    ds: xr.Dataset,
    source_kind: str,
    var_attrs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> xr.Dataset:
    """
    Stamp CF-convention metadata onto a converter-produced Dataset in place.

    Per variable (data variables and coordinates alike), attributes are
    taken from, in order of preference:

    1. ``var_attrs[name]`` — attributes captured from the raw product file,
       sanitized via :func:`sanitize_raw_attrs`;
    2. :data:`INSITU_VARIABLE_ATTRS` for known parameter codes;
    3. :data:`COORD_ATTRS` for the coordinate variables.

    Globals set: ``Conventions``, ``references`` (from
    :data:`PRODUCT_REFERENCES`), and a ``history`` line appended to any
    existing history.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to annotate (modified in place and returned).
    source_kind : str
        One of ``"scatterometer"``, ``"altimeter"``, ``"radiometer"``,
        ``"sar"``, ``"insitu"``.
    var_attrs : dict, optional
        Mapping of output variable name → raw attribute dict.
    """
    var_attrs = var_attrs or {}

    for name in list(ds.variables):
        merged: Dict[str, Any] = {}
        if name in COORD_ATTRS:
            merged.update(COORD_ATTRS[name])
        if name in INSITU_VARIABLE_ATTRS:
            merged.update(INSITU_VARIABLE_ATTRS[name])
        if name in var_attrs:
            merged.update(sanitize_raw_attrs(var_attrs[name]))
        if merged:
            ds[name].attrs.update(merged)

    ds.attrs["Conventions"] = "CF-1.8"
    if source_kind in PRODUCT_REFERENCES:
        refs = PRODUCT_REFERENCES[source_kind]
        existing_refs = ds.attrs.get("references")
        ds.attrs["references"] = f"{existing_refs}; {refs}" if existing_refs else refs

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = f"{stamp}: converted to point/grid Dataset by sar-l2-validation-toolbox"
    history = ds.attrs.get("history")
    ds.attrs["history"] = f"{history}\n{entry}" if history else entry
    return ds


#: Attrs for the geometry/offset variables of a collocation Dataset.
_COLLOCATION_GEOMETRY_ATTRS: Dict[str, Dict[str, str]] = {
    "sar_lon": {"standard_name": "longitude", "long_name": "SAR cell longitude", "units": "degrees_east"},
    "sar_lat": {"standard_name": "latitude", "long_name": "SAR cell latitude", "units": "degrees_north"},
    "val_lon": {"standard_name": "longitude", "long_name": "validation observation longitude", "units": "degrees_east"},
    "val_lat": {"standard_name": "latitude", "long_name": "validation observation latitude", "units": "degrees_north"},
    "spatial_distance_km": {
        "long_name": "great-circle distance between SAR cell and validation observation",
        "units": "km",
    },
    # No units attr here: a timedelta-like units string ("minutes") makes
    # xarray decode the float column as timedelta64 on re-open, silently
    # changing its dtype for every reader.
    "temporal_distance_minutes": {
        "long_name": "absolute time offset between SAR acquisition and validation observation, in minutes",
    },
    "time": {"standard_name": "time", "long_name": "SAR acquisition time"},
    "val_time": {"standard_name": "time", "long_name": "validation observation time"},
    "val_source": {"long_name": "validation platform type"},
    "val_id": {"long_name": "validation platform identifier"},
    "collocation_type": {"long_name": "collocation geometry (point_vs_point | point_vs_layer | layer_vs_layer)"},
    "sar_scene_name": {"long_name": "SAR scene (SAFE product) the observation was matched to"},
    "sar_y_idx": {"long_name": "SAR grid row index of the matched cell"},
    "sar_x_idx": {"long_name": "SAR grid column index of the matched cell"},
}


def annotate_collocation_ds(result_ds: xr.Dataset, datatree: xr.DataTree) -> xr.Dataset:
    """
    Copy CF metadata from datatree nodes onto a collocation Dataset in place.

    ``sar_<var>`` columns inherit the descriptive attrs of the first
    datatree variable named ``<var>`` (a recipe run has exactly one SAR
    product, so there's no ambiguity). ``val_<var>`` columns pool every
    validation source's rows together, distinguished only by the
    per-point ``val_source`` value -- when every source present shares the
    same units for that variable (the common case), the column gets one
    shared attrs value, same as before. When sources genuinely differ
    (e.g. soil moisture: ASCAT's "%" alongside ISMN/SMAP/SMOS's "m3 m-3"),
    a single column-level ``units`` string can't represent every row
    correctly, so this adds per-point ``val_units``/``val_long_name``
    companion variables instead of guessing -- the previous behavior
    (silently keeping whichever source's attrs were encountered first
    while walking ``datatree.subtree``) mislabeled every other source's
    rows. Geometry and provenance columns get fixed attrs from
    :data:`_COLLOCATION_GEOMETRY_ATTRS`.

    Platform-based resolution keys off each datatree node's ``platform_type``
    *attribute*, matched against each collocation row's ``val_source``
    *value*. For most converters those are the same string. The combined
    in-situ CSV converter (``from_insitu_csv``) is an exception: its node-level
    ``platform_type`` attr is a generic value (e.g. ``"insitu"``), while
    ``collocation.py`` derives each row's ``val_source`` from a *per-point*
    ``platform_type`` data variable with mapped values (``"mooring"``,
    ``"buoy"``, ``"drifter"``, ...). Those never match
    ``source_attrs_by_platform``, so ``known_attrs`` comes back empty for
    every source present. When that happens -- i.e. platform-based
    resolution found *nothing* for *any* source in this column -- fall back
    to the var-name-keyed lookup (``source_attrs``) used by the ``sar_``
    branch below. This is deliberately narrower than the var-name fallback
    that predates platform-based resolution: it only fires when
    ``known_attrs`` is completely empty, so it cannot mask a genuine
    mixed-units case where *some* sources resolved via platform_type and
    others didn't -- that scenario still falls through to the
    ``val_units``/``val_long_name`` branch below.
    """
    source_attrs: Dict[str, Dict[str, Any]] = {}
    source_attrs_by_platform: Dict[tuple, Dict[str, Any]] = {}
    for node in datatree.subtree:
        node_ds = node.to_dataset()
        platform_type = node_ds.attrs.get("platform_type")
        for raw_name, var in node_ds.variables.items():
            name = str(raw_name)
            if not var.attrs:
                continue
            if name not in source_attrs:
                source_attrs[name] = sanitize_raw_attrs(var.attrs)
            if platform_type:
                source_attrs_by_platform[(platform_type, name)] = sanitize_raw_attrs(var.attrs)

    for name in map(str, result_ds.variables):
        if name in _COLLOCATION_GEOMETRY_ATTRS:
            result_ds[name].attrs.update(_COLLOCATION_GEOMETRY_ATTRS[name])
            continue

        if name.startswith("val_") and "val_source" in result_ds:
            var_name = name[len("val_"):]
            platform_types_present = sorted(set(str(v) for v in result_ds["val_source"].values))
            attrs_per_platform = {
                pt: source_attrs_by_platform.get((pt, var_name)) for pt in platform_types_present
            }
            known_attrs = [a for a in attrs_per_platform.values() if a]
            distinct_units = {a.get("units") for a in known_attrs}

            if not known_attrs and var_name in source_attrs:
                result_ds[name].attrs.update(source_attrs[var_name])
            elif len(distinct_units) <= 1 and known_attrs:
                result_ds[name].attrs.update(known_attrs[0])
            elif known_attrs:
                val_source_values = result_ds["val_source"].values
                result_ds["val_units"] = (
                    "collocation",
                    [(attrs_per_platform.get(str(pt)) or {}).get("units", "") for pt in val_source_values],
                )
                result_ds["val_long_name"] = (
                    "collocation",
                    [(attrs_per_platform.get(str(pt)) or {}).get("long_name", "") for pt in val_source_values],
                )
                result_ds[name].attrs["units"] = "mixed — see val_units"
                result_ds[name].attrs.pop("long_name", None)
            continue

        if name.startswith("sar_") and name[len("sar_"):] in source_attrs:
            result_ds[name].attrs.update(source_attrs[name[len("sar_"):]])

    result_ds.attrs["Conventions"] = "CF-1.8"
    return result_ds
