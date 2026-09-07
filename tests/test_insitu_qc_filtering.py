"""
Tests for QC-flag filtering in ``DataTreeConverter.from_insitu_csv``.

Copernicus Marine's in-situ CSV export carries a ``value_qc`` column
alongside every ``value``, using CMEMS's 0-9 quality-control scale. A value
whose QC code is not 1, 2, 5, 7, or 8 -- or has no QC code at all -- is
removed from the converted Dataset, along with its own ``<PARAMETER>_QC``
column, so a parameter is always either present with a valid QC code
visible next to it, or absent entirely.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sar_validation.core.datatree_converter import DataTreeConverter


def _row(variable, value, value_qc, *, platform_id="P1",
         time="2026-01-01T00:00:00", lon=0.0, lat=50.0):
    return {
        "platform_id": platform_id,
        "platform_type": "MO",
        "time": time,
        "longitude": lon,
        "latitude": lat,
        "variable": variable,
        "value": value,
        "value_qc": value_qc,
    }


def _write_long_csv(tmp_path: Path, rows: list[dict], name: str = "insitu.csv") -> Path:
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class TestValidQcCodeKeepsValueAndCodeVisible:
    @pytest.mark.parametrize("qc_code", [1, 2, 5, 7, 8])
    def test_valid_qc_code_is_kept(self, tmp_path, qc_code):
        path = _write_long_csv(tmp_path, [_row("WSPD", 12.3, qc_code)])

        ds = DataTreeConverter.from_insitu_csv(path, source_type="mooring")

        assert ds is not None
        assert ds["WSPD"].values[0] == pytest.approx(12.3)
        assert ds["WSPD_QC"].values[0] == qc_code


class TestInvalidQcCodeNullsValueAndCode:
    @pytest.mark.parametrize("qc_code", [0, 3, 4, 6, 9])
    def test_invalid_qc_code_is_rejected(self, tmp_path, qc_code):
        path = _write_long_csv(tmp_path, [_row("VHM0", 2.5, qc_code)])

        ds = DataTreeConverter.from_insitu_csv(path, source_type="buoy")

        assert ds is not None
        assert np.isnan(ds["VHM0"].values[0])
        assert np.isnan(ds["VHM0_QC"].values[0])

    def test_missing_qc_code_is_rejected(self, tmp_path):
        path = _write_long_csv(tmp_path, [_row("WDIR", 180.0, np.nan)])

        ds = DataTreeConverter.from_insitu_csv(path, source_type="buoy")

        assert ds is not None
        assert np.isnan(ds["WDIR"].values[0])
        assert np.isnan(ds["WDIR_QC"].values[0])


class TestWindUsesTheSameGenericFilterAsEveryOtherParameter:
    def test_wspd_and_wdir_are_filtered_independently(self, tmp_path):
        path = _write_long_csv(tmp_path, [
            _row("WSPD", 12.3, 1),
            _row("WDIR", 180.0, 4),
        ])

        ds = DataTreeConverter.from_insitu_csv(path, source_type="buoy")

        assert ds is not None
        assert ds["WSPD"].values[0] == pytest.approx(12.3)
        assert ds["WSPD_QC"].values[0] == 1
        assert np.isnan(ds["WDIR"].values[0])
        assert np.isnan(ds["WDIR_QC"].values[0])


class TestCsvWithoutValueQcColumnIsUnaffected:
    def test_converts_unfiltered_when_no_qc_column_present(self, tmp_path):
        path = tmp_path / "insitu.csv"
        pd.DataFrame([{
            "platform_id": "P1", "platform_type": "MO",
            "time": "2026-01-01T00:00:00", "longitude": 0.0, "latitude": 50.0,
            "variable": "WSPD", "value": 12.3,
        }]).to_csv(path, index=False)

        ds = DataTreeConverter.from_insitu_csv(path, source_type="mooring")

        assert ds is not None
        assert ds["WSPD"].values[0] == pytest.approx(12.3)
        assert "WSPD_QC" not in ds


class TestWaveHeightPrecedenceAfterQcFiltering:
    def test_qc_bad_higher_precedence_column_falls_back_to_next(self, tmp_path):
        path = _write_long_csv(tmp_path, [
            _row("VHM0", 1.1, 4),
            _row("VAVH", 1.0, 1),
        ])

        ds = DataTreeConverter.from_insitu_csv(path, source_type="buoy")

        assert ds is not None
        assert np.isnan(ds["VHM0"].values[0])
        assert ds["VAVH"].values[0] == pytest.approx(1.0)
        assert ds["VAVH_QC"].values[0] == 1

    def test_qc_good_higher_precedence_column_still_wins(self, tmp_path):
        path = _write_long_csv(tmp_path, [
            _row("VHM0", 1.1, 1),
            _row("VAVH", 1.0, 1),
        ])

        ds = DataTreeConverter.from_insitu_csv(path, source_type="buoy")

        assert ds is not None
        assert ds["VHM0"].values[0] == pytest.approx(1.1)
        assert np.isnan(ds["VAVH"].values[0])
        assert np.isnan(ds["VAVH_QC"].values[0])
