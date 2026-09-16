"""Tests for converting reprocessed altimeter NetCDF files to Datasets."""

from __future__ import annotations

import numpy as np
import xarray as xr

from sar_validation.core.datatree_converter import _decode_flag_variable


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
