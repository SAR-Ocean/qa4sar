"""
Tests for ``DataTreeConverter.from_gts_buoy_bufr``.

``pdbufr.read_bufr`` is monkeypatched to return a synthetic DataFrame
shaped like a real decode of MARS obstype 181/182 BUFR -- building an
actual BUFR-encoded fixture file requires the eccodes encoding API, which
this toolbox does not otherwise depend on; monkeypatching the decode
boundary keeps these tests fast and independent of that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sar_validation.core.datatree_converter import DataTreeConverter


def _fake_bufr_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _row(
    platform_id, lat, lon, year, month, day, hour, minute,
    wind_speed, wind_direction, name="Test Buoy",
):
    return {
        "marineObservingPlatformIdentifier": platform_id,
        "stationOrSiteName": name,
        "latitude": lat,
        "longitude": lon,
        "year": year, "month": month, "day": day, "hour": hour, "minute": minute,
        "windSpeed": wind_speed, "windDirection": wind_direction,
    }


class TestFromGtsBuoyBufrBasicConversion:
    def test_converts_real_looking_rows(self, tmp_path, monkeypatch):
        frame = _fake_bufr_frame([
            _row(6600021, 54.88, 13.87, 2026, 8, 30, 8, 0, 9.0, 191.0),
            _row(6600021, 54.88, 13.87, 2026, 8, 30, 10, 0, 11.1, 213.0),
        ])
        monkeypatch.setattr(
            "sar_validation.core.datatree_converter.pdbufr.read_bufr",
            lambda path, columns, filters=None: frame,
        )
        bufr_path = tmp_path / "gts_buoy_20260830.bufr"
        bufr_path.write_bytes(b"not real bufr -- read_bufr is mocked")

        ds = DataTreeConverter.from_gts_buoy_bufr(bufr_path)

        assert ds is not None
        assert ds.sizes["point"] == 2
        assert list(ds["WSPD"].values) == pytest.approx([9.0, 11.1])
        assert list(ds["WDIR"].values) == pytest.approx([191.0, 213.0])
        assert list(ds["platform_id"].values) == ["6600021", "6600021"]
        assert ds.attrs["data_type"] == "gts_buoy"
        assert ds.attrs["platform_type"] == "buoy"
        assert ds.attrs["filename"] == "gts_buoy_20260830.bufr"

    def test_time_coordinate_is_built_from_ymdhm_columns(self, tmp_path, monkeypatch):
        frame = _fake_bufr_frame([
            _row(1300002, 10.0, -20.0, 2026, 1, 1, 3, 15, 5.0, 90.0),
        ])
        monkeypatch.setattr(
            "sar_validation.core.datatree_converter.pdbufr.read_bufr",
            lambda path, columns, filters=None: frame,
        )
        bufr_path = tmp_path / "gts_buoy_20260101.bufr"
        bufr_path.write_bytes(b"mocked")

        ds = DataTreeConverter.from_gts_buoy_bufr(bufr_path)

        assert ds is not None
        assert pd.Timestamp(ds["time"].values[0]) == pd.Timestamp("2026-01-01T03:15:00")


class TestFromGtsBuoyBufrMissingWind:
    def test_rows_with_both_wind_fields_nan_are_dropped(self, tmp_path, monkeypatch):
        frame = _fake_bufr_frame([
            _row(6600021, 54.88, 13.87, 2026, 8, 30, 0, 0, np.nan, np.nan),
            _row(6600021, 54.88, 13.87, 2026, 8, 30, 8, 0, 9.0, 191.0),
        ])
        monkeypatch.setattr(
            "sar_validation.core.datatree_converter.pdbufr.read_bufr",
            lambda path, columns, filters=None: frame,
        )
        bufr_path = tmp_path / "gts_buoy_20260830.bufr"
        bufr_path.write_bytes(b"mocked")

        ds = DataTreeConverter.from_gts_buoy_bufr(bufr_path)

        assert ds is not None
        assert ds.sizes["point"] == 1
        assert float(ds["WSPD"].values[0]) == pytest.approx(9.0)

    def test_all_rows_missing_wind_returns_none(self, tmp_path, monkeypatch):
        frame = _fake_bufr_frame([
            _row(6600021, 54.88, 13.87, 2026, 8, 30, 0, 0, np.nan, np.nan),
        ])
        monkeypatch.setattr(
            "sar_validation.core.datatree_converter.pdbufr.read_bufr",
            lambda path, columns, filters=None: frame,
        )
        bufr_path = tmp_path / "gts_buoy_20260830.bufr"
        bufr_path.write_bytes(b"mocked")

        ds = DataTreeConverter.from_gts_buoy_bufr(bufr_path)

        assert ds is None


class TestFromGtsBuoyBufrMissingFile:
    def test_missing_file_returns_none(self, tmp_path):
        ds = DataTreeConverter.from_gts_buoy_bufr(tmp_path / "does_not_exist.bufr")
        assert ds is None


def _wave_row(platform_id, lat, lon, year, month, day, hour, minute, hs, name="Test Buoy"):
    return {
        "marineObservingPlatformIdentifier": platform_id,
        "stationOrSiteName": name,
        "latitude": lat, "longitude": lon,
        "year": year, "month": month, "day": day, "hour": hour, "minute": minute,
        "significantWaveHeight": hs,
    }


def _current_row(
    platform_id, lat, lon, year, month, day, hour, minute,
    depth, speed, direction, name="Test Buoy",
):
    return {
        "marineObservingPlatformIdentifier": platform_id,
        "stationOrSiteName": name,
        "latitude": lat, "longitude": lon,
        "year": year, "month": month, "day": day, "hour": hour, "minute": minute,
        "depthBelowSeaSurface": depth, "speedOfCurrent": speed, "directionOfCurrent": direction,
    }


class TestFromGtsBuoyBufrWaves:
    def test_converts_wave_height_rows(self, tmp_path, monkeypatch):
        frame = _fake_bufr_frame([
            _wave_row(6600021, 54.88, 13.87, 2026, 8, 30, 8, 0, 1.8),
            _wave_row(6600021, 54.88, 13.87, 2026, 8, 30, 10, 0, 2.1),
        ])
        monkeypatch.setattr(
            "sar_validation.core.datatree_converter.pdbufr.read_bufr",
            lambda path, columns, filters=None: frame,
        )
        bufr_path = tmp_path / "gts_buoy_20260830.bufr"
        bufr_path.write_bytes(b"mocked")

        ds = DataTreeConverter.from_gts_buoy_bufr(bufr_path, product_type="waves")

        assert ds is not None
        assert ds.sizes["point"] == 2
        assert list(ds["VAVH"].values) == pytest.approx([1.8, 2.1])
        assert "WSPD" not in ds.variables
        assert "EWCT" not in ds.variables

    def test_rows_missing_wave_height_are_dropped(self, tmp_path, monkeypatch):
        frame = _fake_bufr_frame([
            _wave_row(6600021, 54.88, 13.87, 2026, 8, 30, 8, 0, np.nan),
            _wave_row(6600021, 54.88, 13.87, 2026, 8, 30, 10, 0, 2.1),
        ])
        monkeypatch.setattr(
            "sar_validation.core.datatree_converter.pdbufr.read_bufr",
            lambda path, columns, filters=None: frame,
        )
        bufr_path = tmp_path / "gts_buoy_20260830.bufr"
        bufr_path.write_bytes(b"mocked")

        ds = DataTreeConverter.from_gts_buoy_bufr(bufr_path, product_type="waves")

        assert ds is not None
        assert ds.sizes["point"] == 1
        assert float(ds["VAVH"].values[0]) == pytest.approx(2.1)

    def test_all_rows_missing_wave_height_returns_none(self, tmp_path, monkeypatch):
        frame = _fake_bufr_frame([
            _wave_row(6600021, 54.88, 13.87, 2026, 8, 30, 8, 0, np.nan),
        ])
        monkeypatch.setattr(
            "sar_validation.core.datatree_converter.pdbufr.read_bufr",
            lambda path, columns, filters=None: frame,
        )
        bufr_path = tmp_path / "gts_buoy_20260830.bufr"
        bufr_path.write_bytes(b"mocked")

        ds = DataTreeConverter.from_gts_buoy_bufr(bufr_path, product_type="waves")

        assert ds is None


class TestFromGtsBuoyBufrCurrents:
    def test_shallowest_depth_is_kept_and_converted_to_ewct_nsct(self, tmp_path, monkeypatch):
        frame = _fake_bufr_frame([
            _current_row(6600021, 54.88, 13.87, 2026, 8, 30, 8, 0, 10.0, 0.3, 300.0),
            _current_row(6600021, 54.88, 13.87, 2026, 8, 30, 8, 0, 0.0, 0.5, 90.0),
        ])
        monkeypatch.setattr(
            "sar_validation.core.datatree_converter.pdbufr.read_bufr",
            lambda path, columns, filters=None: frame,
        )
        bufr_path = tmp_path / "gts_buoy_20260830.bufr"
        bufr_path.write_bytes(b"mocked")

        ds = DataTreeConverter.from_gts_buoy_bufr(bufr_path, product_type="currents")

        assert ds is not None
        assert ds.sizes["point"] == 1
        assert float(ds["EWCT"].values[0]) == pytest.approx(0.5 * np.sin(np.radians(90.0)))
        assert float(ds["NSCT"].values[0]) == pytest.approx(0.5 * np.cos(np.radians(90.0)))

    def test_rows_missing_current_speed_are_dropped(self, tmp_path, monkeypatch):
        frame = _fake_bufr_frame([
            _current_row(6600021, 54.88, 13.87, 2026, 8, 30, 8, 0, 0.0, np.nan, np.nan),
            _current_row(1300002, 10.0, -20.0, 2026, 8, 30, 8, 0, 0.0, 0.2, 45.0),
        ])
        monkeypatch.setattr(
            "sar_validation.core.datatree_converter.pdbufr.read_bufr",
            lambda path, columns, filters=None: frame,
        )
        bufr_path = tmp_path / "gts_buoy_20260830.bufr"
        bufr_path.write_bytes(b"mocked")

        ds = DataTreeConverter.from_gts_buoy_bufr(bufr_path, product_type="currents")

        assert ds is not None
        assert ds.sizes["point"] == 1
        assert list(ds["platform_id"].values) == ["1300002"]

    def test_shallowest_row_missing_speed_is_dropped_not_backfilled_from_deeper_row(
        self, tmp_path, monkeypatch,
    ):
        # The shallowest row (depth 0.0) carries no speed of its own, while
        # a deeper row (depth 10.0) for the same platform/time does carry
        # one. Selecting the shallowest depth must keep that row's own
        # fields intact: since it is missing its own required speed, the
        # whole observation is dropped, rather than the deeper row's speed
        # being spliced onto the shallow row's depth and direction.
        frame = _fake_bufr_frame([
            _current_row(6600021, 54.88, 13.87, 2026, 8, 30, 8, 0, 0.0, np.nan, 90.0),
            _current_row(6600021, 54.88, 13.87, 2026, 8, 30, 8, 0, 10.0, 0.3, 300.0),
        ])
        monkeypatch.setattr(
            "sar_validation.core.datatree_converter.pdbufr.read_bufr",
            lambda path, columns, filters=None: frame,
        )
        bufr_path = tmp_path / "gts_buoy_20260830.bufr"
        bufr_path.write_bytes(b"mocked")

        ds = DataTreeConverter.from_gts_buoy_bufr(bufr_path, product_type="currents")

        assert ds is None

    def test_row_with_no_depth_reported_at_all_is_still_kept(self, tmp_path, monkeypatch):
        # A single, non-replicated reading with no depth section at all
        # (depthBelowSeaSurface is NaN for its whole platform/time group)
        # must still resolve to that one available row rather than being
        # dropped as an artifact of the depth-selection step.
        frame = _fake_bufr_frame([
            _current_row(1300002, 10.0, -20.0, 2026, 8, 30, 8, 0, np.nan, 0.4, 120.0),
        ])
        monkeypatch.setattr(
            "sar_validation.core.datatree_converter.pdbufr.read_bufr",
            lambda path, columns, filters=None: frame,
        )
        bufr_path = tmp_path / "gts_buoy_20260830.bufr"
        bufr_path.write_bytes(b"mocked")

        ds = DataTreeConverter.from_gts_buoy_bufr(bufr_path, product_type="currents")

        assert ds is not None
        assert ds.sizes["point"] == 1
        assert list(ds["platform_id"].values) == ["1300002"]


class TestFromGtsBuoyBufrUnknownProductType:
    def test_unknown_product_type_raises(self, tmp_path):
        with pytest.raises(ValueError, match="product_type"):
            DataTreeConverter.from_gts_buoy_bufr(tmp_path / "x.bufr", product_type="soil_moisture")
