"""Tests for sar_validation.core._cf_metadata."""

from __future__ import annotations

import numpy as np
import xarray as xr

from sar_validation.core._cf_metadata import annotate_collocation_ds
from sar_validation.core.datatree_converter import DataTreeConverter


def _datatree_with_mixed_units_sources():
    ismn_ds = xr.Dataset(
        {"SOIL_MOISTURE": ("point", np.array([0.1, 0.2]))},
        coords={"lon": ("point", np.array([-9.0, -8.5])),
                "lat": ("point", np.array([50.0, 50.5])),
                "time": ("point", np.array(["2026-07-10T12:00", "2026-07-10T12:05"], dtype="datetime64[ns]"))},
        attrs={"platform_type": "ismn"},
    )
    ismn_ds["SOIL_MOISTURE"].attrs = {
        "units": "m3 m-3", "long_name": "ISMN in-situ volumetric soil moisture",
    }
    ascat_ds = xr.Dataset(
        {"SOIL_MOISTURE": ("point", np.array([20.0, 30.0]))},
        coords={"lon": ("point", np.array([-9.2, -8.7])),
                "lat": ("point", np.array([50.2, 50.7])),
                "time": ("point", np.array(["2026-07-10T12:01", "2026-07-10T12:06"], dtype="datetime64[ns]"))},
        attrs={"platform_type": "ascat_ssm"},
    )
    ascat_ds["SOIL_MOISTURE"].attrs = {
        "units": "%", "long_name": "ASCAT surface soil moisture (~0-5cm, C-band)",
    }
    return DataTreeConverter.to_datatree({
        "validation/ismn": ismn_ds, "validation/ascat_ssm": ascat_ds,
    })


class TestAnnotateCollocationDsMixedUnits:
    def test_mixed_units_get_per_point_val_units_companion(self):
        datatree = _datatree_with_mixed_units_sources()
        result_ds = xr.Dataset({
            "val_SOIL_MOISTURE": ("collocation", np.array([0.1, 20.0, 0.2, 30.0])),
            "val_source": ("collocation", ["ismn", "ascat_ssm", "ismn", "ascat_ssm"]),
        })

        annotate_collocation_ds(result_ds, datatree)

        assert "val_units" in result_ds
        assert list(result_ds["val_units"].values) == ["m3 m-3", "%", "m3 m-3", "%"]
        assert "val_long_name" in result_ds
        assert result_ds["val_long_name"].values[1] == "ASCAT surface soil moisture (~0-5cm, C-band)"
        # The column-level attrs must not silently claim one specific
        # unit for the whole (mixed) column.
        assert result_ds["val_SOIL_MOISTURE"].attrs.get("units") != "m3 m-3"
        assert result_ds["val_SOIL_MOISTURE"].attrs.get("units") != "%"

    def test_uniform_units_keep_existing_single_attrs_behavior(self):
        """When every present val_source shares the same units (the common
        case -- e.g. wind's WSPD across mooring/altimeter/scatterometer),
        no val_units companion variable is added -- behavior is unchanged
        from before this fix."""
        mooring_ds = xr.Dataset(
            {"WSPD": ("point", np.array([5.0, 6.0]))},
            coords={"lon": ("point", np.array([-9.0, -8.5])),
                    "lat": ("point", np.array([50.0, 50.5])),
                    "time": ("point", np.array(["2026-07-10T12:00", "2026-07-10T12:05"], dtype="datetime64[ns]"))},
            attrs={"platform_type": "mooring"},
        )
        mooring_ds["WSPD"].attrs = {"units": "m s-1", "long_name": "horizontal wind speed"}
        altimeter_ds = xr.Dataset(
            {"WSPD": ("point", np.array([7.0, 8.0]))},
            coords={"lon": ("point", np.array([-9.2, -8.7])),
                    "lat": ("point", np.array([50.2, 50.7])),
                    "time": ("point", np.array(["2026-07-10T12:01", "2026-07-10T12:06"], dtype="datetime64[ns]"))},
            attrs={"platform_type": "altimeter"},
        )
        altimeter_ds["WSPD"].attrs = {"units": "m s-1", "long_name": "horizontal wind speed"}
        datatree = DataTreeConverter.to_datatree({
            "validation/mooring": mooring_ds, "validation/altimeter": altimeter_ds,
        })
        result_ds = xr.Dataset({
            "val_WSPD": ("collocation", np.array([5.0, 7.0])),
            "val_source": ("collocation", ["mooring", "altimeter"]),
        })

        annotate_collocation_ds(result_ds, datatree)

        assert "val_units" not in result_ds
        assert result_ds["val_WSPD"].attrs.get("units") == "m s-1"

    def test_sar_columns_unaffected_by_val_source_pooling(self):
        """sar_<var> columns are never pooled across multiple sources (one
        recipe run has exactly one SAR product) and must keep their
        existing single-attrs behavior unconditionally."""
        sar_ds = xr.Dataset(
            {"sarSSM": ("point", np.array([40.0, 45.0]))},
            coords={"lon": ("point", np.array([-9.0, -8.5])),
                    "lat": ("point", np.array([50.0, 50.5])),
                    "time": ("point", np.array(["2026-07-10T12:00", "2026-07-10T12:05"], dtype="datetime64[ns]"))},
        )
        sar_ds["sarSSM"].attrs = {
            "units": "%", "long_name": "Sentinel-1 CLMS surface soil moisture (percent saturation)",
        }
        datatree = DataTreeConverter.to_datatree({"sar/sceneA": sar_ds})
        result_ds = xr.Dataset({"sar_sarSSM": ("collocation", np.array([40.0, 45.0]))})

        annotate_collocation_ds(result_ds, datatree)

        assert result_ds["sar_sarSSM"].attrs.get("units") == "%"


class TestAnnotateCollocationDsCombinedInsituFallback:
    def test_generic_node_platform_type_falls_back_to_var_name_lookup(self):
        """Combined in-situ CSV nodes (``from_insitu_csv``) set a generic
        node-level ``platform_type`` attr (e.g. "insitu"), but
        ``collocation.py`` derives each row's ``val_source`` from a
        per-point ``platform_type`` variable with mapped values like
        "mooring"/"buoy" -- these never match
        ``source_attrs_by_platform``, so platform-based resolution finds
        nothing. This must fall back to the var-name-keyed lookup instead
        of leaving the column with no units/long_name at all."""
        insitu_ds = xr.Dataset(
            {"WSPD": ("point", np.array([5.0, 6.0]))},
            coords={"lon": ("point", np.array([-9.0, -8.5])),
                    "lat": ("point", np.array([50.0, 50.5])),
                    "time": ("point", np.array(["2026-07-10T12:00", "2026-07-10T12:05"], dtype="datetime64[ns]"))},
            attrs={"platform_type": "insitu"},
        )
        insitu_ds["WSPD"].attrs = {"units": "m s-1", "long_name": "horizontal wind speed"}
        datatree = DataTreeConverter.to_datatree({"validation/insitu": insitu_ds})
        result_ds = xr.Dataset({
            "val_WSPD": ("collocation", np.array([5.0, 6.0])),
            "val_source": ("collocation", ["mooring", "buoy"]),
        })

        annotate_collocation_ds(result_ds, datatree)

        assert "val_units" not in result_ds
        assert result_ds["val_WSPD"].attrs.get("units") == "m s-1"
        assert result_ds["val_WSPD"].attrs.get("long_name") == "horizontal wind speed"
