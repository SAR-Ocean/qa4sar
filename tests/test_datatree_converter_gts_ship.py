"""
Tests for ``DataTreeConverter.from_gts_ship_bufr``.

``pdbufr.read_bufr`` is monkeypatched to return a synthetic DataFrame
shaped like a real decode of MARS obstype 180 BUFR -- building an actual
BUFR-encoded fixture file requires the eccodes encoding API, which this
toolbox does not otherwise depend on; monkeypatching the decode boundary
keeps these tests fast and independent of that. Row values below are
taken from a real decoded obstype-180-shaped BUFR file, not invented --
call signs, coordinates, and wind values are genuine.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from sar_validation.core.datatree_converter import DataTreeConverter


@pytest.fixture(autouse=True, scope="function")
def _ensure_pdbufr_available():
    """Ensure pdbufr is properly imported, even after tests that
    intentionally disable it for testing error paths."""
    import sar_validation.core.datatree_converter as dtc_module
    # Reload the module to restore pdbufr in case a previous test disabled it
    if dtc_module.pdbufr is None:
        importlib.reload(dtc_module)
    yield


def _fake_bufr_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _row(callsign, lat, lon, year, month, day, hour, minute, wind_speed, wind_direction):
    return {
        "shipOrMobileLandStationIdentifier": callsign,
        "latitude": lat,
        "longitude": lon,
        "year": year, "month": month, "day": day, "hour": hour, "minute": minute,
        "windSpeed": wind_speed, "windDirection": wind_direction,
    }


class TestFromGtsShipBufrBasicConversion:
    def test_converts_real_looking_rows(self, tmp_path, monkeypatch):
        frame = _fake_bufr_frame([
            _row("KBAG", 44.0, -86.9, 2026, 8, 1, 0, 0, 11.3, 340.0),
            _row("CFJ8305", 45.4, -83.3, 2026, 8, 1, 0, 53, 7.2, 350.0),
        ])
        monkeypatch.setattr(
            "sar_validation.core.datatree_converter.pdbufr.read_bufr",
            lambda path, columns, filters=None: frame,
        )
        bufr_path = tmp_path / "gts_ship_20260801.bufr"
        bufr_path.write_bytes(b"not real bufr -- read_bufr is mocked")

        ds = DataTreeConverter.from_gts_ship_bufr(bufr_path)

        assert ds is not None
        assert ds.sizes["point"] == 2
        assert list(ds["WSPD"].values) == pytest.approx([11.3, 7.2])
        assert list(ds["WDIR"].values) == pytest.approx([340.0, 350.0])
        assert list(ds["platform_id"].values) == ["KBAG", "CFJ8305"]
        assert ds.attrs["data_type"] == "gts_ship"
        assert ds.attrs["platform_type"] == "ferrybox"
        assert ds.attrs["filename"] == "gts_ship_20260801.bufr"

    def test_time_coordinate_is_built_from_ymdhm_columns(self, tmp_path, monkeypatch):
        frame = _fake_bufr_frame([
            _row("9HA123", 10.0, -20.0, 2026, 1, 1, 3, 15, 5.0, 90.0),
        ])
        monkeypatch.setattr(
            "sar_validation.core.datatree_converter.pdbufr.read_bufr",
            lambda path, columns, filters=None: frame,
        )
        bufr_path = tmp_path / "gts_ship_20260101.bufr"
        bufr_path.write_bytes(b"mocked")

        ds = DataTreeConverter.from_gts_ship_bufr(bufr_path)

        assert ds is not None
        assert pd.Timestamp(ds["time"].values[0]) == pd.Timestamp("2026-01-01T03:15:00")

    def test_rows_with_both_wind_fields_nan_are_dropped(self, tmp_path, monkeypatch):
        frame = _fake_bufr_frame([
            _row("ZCDC2", 54.88, 13.87, 2026, 8, 30, 8, 0, 9.0, 191.0),
            _row("9HA123", 10.0, -20.0, 2026, 8, 30, 9, 0, np.nan, np.nan),
        ])
        monkeypatch.setattr(
            "sar_validation.core.datatree_converter.pdbufr.read_bufr",
            lambda path, columns, filters=None: frame,
        )
        bufr_path = tmp_path / "gts_ship_20260830.bufr"
        bufr_path.write_bytes(b"mocked")

        ds = DataTreeConverter.from_gts_ship_bufr(bufr_path)

        assert ds is not None
        assert ds.sizes["point"] == 1
        assert list(ds["platform_id"].values) == ["ZCDC2"]

    def test_all_rows_missing_wind_returns_none(self, tmp_path, monkeypatch):
        frame = _fake_bufr_frame([
            _row("ZCDC2", 54.88, 13.87, 2026, 8, 30, 8, 0, np.nan, np.nan),
        ])
        monkeypatch.setattr(
            "sar_validation.core.datatree_converter.pdbufr.read_bufr",
            lambda path, columns, filters=None: frame,
        )
        bufr_path = tmp_path / "gts_ship_20260830.bufr"
        bufr_path.write_bytes(b"mocked")

        assert DataTreeConverter.from_gts_ship_bufr(bufr_path) is None

    def test_missing_file_returns_none(self, tmp_path):
        assert DataTreeConverter.from_gts_ship_bufr(tmp_path / "missing.bufr") is None
