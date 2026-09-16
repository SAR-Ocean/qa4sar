"""Tests for converting reprocessed altimeter NetCDF files to Datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sar_validation.core.datatree_converter import DataTreeConverter, _decode_flag_variable


class TestDecodeFlagVariable:
    def test_decodes_values_to_their_flag_meanings(self):
        da = xr.DataArray(
            np.array([0, 3, 5], dtype="uint8"),
            attrs={
                "flag_values": np.array([0, 1, 2, 3, 4, 5], dtype="uint8"),
                "flag_meanings": "cryosat-2 jason-1 jason-2 jason-3 saral sentinel-3_a",
            },
        )

        decoded = _decode_flag_variable(da)

        assert list(decoded) == ["cryosat-2", "jason-3", "sentinel-3_a"]

    def test_returns_none_when_flag_meanings_missing(self):
        da = xr.DataArray(np.array([0, 1]), attrs={"flag_values": np.array([0, 1])})

        assert _decode_flag_variable(da) is None

    def test_returns_none_when_flag_values_missing(self):
        da = xr.DataArray(np.array([0, 1]), attrs={"flag_meanings": "a b"})

        assert _decode_flag_variable(da) is None

    def test_returns_none_when_lengths_do_not_match(self):
        da = xr.DataArray(
            np.array([0, 1]),
            attrs={"flag_values": np.array([0, 1, 2]), "flag_meanings": "a b"},
        )

        assert _decode_flag_variable(da) is None


def _write_reprocessed_altimeter_nc(path):
    """Write a 4-point NetCDF file matching the reprocessed altimeter
    product's real schema (two platforms, one point per platform with a
    missing uncertainty value)."""
    ds = xr.Dataset(
        {
            "swh":               ("time", np.array([2.7, 1.9, 2.5, 2.0])),
            "swh_adjusted":      ("time", np.array([2.65, 1.88, 2.4, 1.95])),
            "swh_denoised":      ("time", np.array([2.621078, 1.85, 2.3, 1.9])),
            "swh_uncertainty":   ("time", np.array([np.nan, 0.12, 0.1, 0.11])),
            "bathymetry":        ("time", np.array([-10.0, -55.0, -200.0, -300.0])),
            "distance_to_coast": ("time", np.array([2000.0, 15000.0, 5000.0, 8000.0])),
            "cycle":             ("time", np.array([40, 41, 42, 43], dtype="uint16")),
            "relative_pass":     ("time", np.array([227.0, 89.0, 15.0, 16.0], dtype="float32")),
            "satellite": (
                "time",
                np.array([3, 0, 3, 0], dtype="uint8"),
                {
                    "flag_values": np.array([0, 1, 2, 3, 4, 5], dtype="uint8"),
                    "flag_meanings": "cryosat-2 jason-1 jason-2 jason-3 saral sentinel-3_a",
                },
            ),
        },
        coords={
            "time": ("time", pd.date_range("2023-12-30T23:11:28", periods=4, freq="1s")),
            "lon":  ("time", np.array([-49.76, 10.2, -49.75, 10.3])),
            "lat":  ("time", np.array([-29.41, 54.1, -29.40, 54.2])),
        },
    )
    ds.to_netcdf(path)


class TestFromAltimeterReprocessed:
    def test_returns_none_for_missing_file(self, tmp_path):
        result = DataTreeConverter.from_altimeter_reprocessed(tmp_path / "missing.nc")
        assert result is None

    def test_renames_swh_denoised_to_vavh(self, tmp_path):
        nc_path = tmp_path / "ESACCI-SEASTATE-L3-SWH-MULTI_1D-20231230-fv01.nc"
        _write_reprocessed_altimeter_nc(nc_path)

        ds = DataTreeConverter.from_altimeter_reprocessed(nc_path)

        assert ds is not None
        assert "VAVH" in ds.data_vars
        assert "swh_denoised" not in ds.data_vars
        assert ds["VAVH"].values[0] == pytest.approx(2.621078)

    def test_renames_swh_uncertainty_to_vavh_uncertainty(self, tmp_path):
        nc_path = tmp_path / "test.nc"
        _write_reprocessed_altimeter_nc(nc_path)

        ds = DataTreeConverter.from_altimeter_reprocessed(nc_path)

        assert "VAVH_UNCERTAINTY" in ds.data_vars
        assert "swh_uncertainty" not in ds.data_vars
        assert np.isnan(ds["VAVH_UNCERTAINTY"].values[0])
        assert ds["VAVH_UNCERTAINTY"].values[1] == pytest.approx(0.12)

    def test_keeps_auxiliary_variables(self, tmp_path):
        nc_path = tmp_path / "test.nc"
        _write_reprocessed_altimeter_nc(nc_path)

        ds = DataTreeConverter.from_altimeter_reprocessed(nc_path)

        for name in ("swh", "swh_adjusted", "bathymetry", "distance_to_coast", "cycle", "relative_pass"):
            assert name in ds.data_vars

    def test_decodes_platform_id_from_satellite_flag_variable(self, tmp_path):
        nc_path = tmp_path / "test.nc"
        _write_reprocessed_altimeter_nc(nc_path)

        ds = DataTreeConverter.from_altimeter_reprocessed(nc_path)

        assert "platform_id" in ds.coords
        assert list(ds["platform_id"].values) == ["jason-3", "cryosat-2", "jason-3", "cryosat-2"]

    def test_dataset_attributes(self, tmp_path):
        nc_path = tmp_path / "test.nc"
        _write_reprocessed_altimeter_nc(nc_path)

        ds = DataTreeConverter.from_altimeter_reprocessed(nc_path)

        assert ds.attrs["data_type"] == "altimeter"
        assert ds.attrs["platform_type"] == "altimeter"
        assert ds.attrs["frequency"] == "reprocessed"

    def test_point_dimension_has_one_entry_per_observation(self, tmp_path):
        nc_path = tmp_path / "test.nc"
        _write_reprocessed_altimeter_nc(nc_path)

        ds = DataTreeConverter.from_altimeter_reprocessed(nc_path)

        assert ds.sizes["point"] == 4
