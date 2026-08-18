"""Tests for dry_collocation.py's SarFootprint model and SAR-side discovery."""

from __future__ import annotations

from datetime import datetime

import pytest


class TestSarFootprint:
    def test_constructs_with_all_fields(self):
        from sar_validation.core.dry_collocation import SarFootprint

        fp = SarFootprint(
            kind="polygon",
            bbox=(-10.0, 10.0, 35.0, 55.0),
            polygon=[(35.0, -10.0), (35.0, 10.0), (55.0, 10.0), (55.0, -10.0)],
            points=None,
            sensing_start=datetime(2026, 8, 1, 0, 0, 0),
            sensing_end=datetime(2026, 8, 1, 0, 1, 30),
            source_file="S1A_IW_OCN__2SDV_20260801T000000.SAFE",
        )

        assert fp.kind == "polygon"
        assert fp.bbox == (-10.0, 10.0, 35.0, 55.0)
        assert fp.polygon is not None and len(fp.polygon) == 4
        assert fp.points is None

    def test_is_frozen(self):
        from sar_validation.core.dry_collocation import SarFootprint

        fp = SarFootprint(
            kind="wv_points", bbox=(0.0, 1.0, 0.0, 1.0), polygon=None,
            points=[(0.5, 0.5)], sensing_start=datetime(2026, 1, 1),
            sensing_end=datetime(2026, 1, 1), source_file="x.SAFE",
        )
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            fp.kind = "polygon"


class TestDiscoverSentinel1OcnFootprintsDry:
    def test_non_wv_granule_becomes_a_polygon_footprint(self, monkeypatch):
        from sar_validation.core import dry_collocation

        fake_records = [
            {
                "Name": "S1A_IW_OCN__2SDV_20260801T060000_20260801T060030_000000_000000_0000.SAFE",
                "ContentDate_Start": "2026-08-01T06:00:00.000Z",
                "ContentDate_End": "2026-08-01T06:00:30.000Z",
                "GeoFootprint": {
                    "type": "Polygon",
                    "coordinates": [[[-10.0, 35.0], [10.0, 35.0], [10.0, 55.0], [-10.0, 55.0], [-10.0, 35.0]]],
                },
            },
        ]
        monkeypatch.setattr(
            dry_collocation, "_query_sentinel1_ocn_dry", lambda cfg: fake_records,
        )

        footprints = dry_collocation._discover_sentinel1_ocn_footprints_dry(cfg=object())

        assert len(footprints) == 1
        fp = footprints[0]
        assert fp.kind == "polygon"
        assert fp.polygon == [(35.0, -10.0), (35.0, 10.0), (55.0, 10.0), (55.0, -10.0)]
        assert fp.bbox == (-10.0, 10.0, 35.0, 55.0)
        assert fp.sensing_start.isoformat().startswith("2026-08-01T06:00:00")
        assert fp.source_file == fake_records[0]["Name"]

    def test_wv_granule_is_excluded_from_non_wv_discovery(self, monkeypatch):
        """WV-mode granules are handled by Task 4's separate function, not
        this one -- a WV Name (mode token 'WV') must be filtered out
        here, not turned into a (wrong) kind="polygon" footprint."""
        from sar_validation.core import dry_collocation

        fake_records = [
            {
                "Name": "S1A_WV_OCN__2SSV_20260801T060000_20260801T060200_000000_000000_0000.SAFE",
                "ContentDate_Start": "2026-08-01T06:00:00.000Z",
                "ContentDate_End": "2026-08-01T06:02:00.000Z",
                "GeoFootprint": {
                    "type": "Polygon",
                    "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
                },
            },
        ]
        monkeypatch.setattr(
            dry_collocation, "_query_sentinel1_ocn_dry", lambda cfg: fake_records,
        )

        footprints = dry_collocation._discover_sentinel1_ocn_footprints_dry(cfg=object())

        assert footprints == []

    def test_missing_geofootprint_still_produces_a_bbox_only_footprint(self, monkeypatch):
        """A product whose GeoFootprint is absent/None must fail toward a
        usable (if less precise) footprint, not be silently dropped --
        derive bbox from the recipe's own requested bbox as the
        conservative fallback (polygon=None)."""
        from sar_validation.core import dry_collocation

        fake_records = [
            {
                "Name": "S1A_IW_OCN__2SDV_20260801T060000_20260801T060030_000000_000000_0000.SAFE",
                "ContentDate_Start": "2026-08-01T06:00:00.000Z",
                "ContentDate_End": "2026-08-01T06:00:30.000Z",
                "GeoFootprint": None,
            },
        ]
        monkeypatch.setattr(
            dry_collocation, "_query_sentinel1_ocn_dry", lambda cfg: fake_records,
        )

        class _FakeCfg:
            min_lon, max_lon, min_lat, max_lat = -10.0, 10.0, 35.0, 55.0

        footprints = dry_collocation._discover_sentinel1_ocn_footprints_dry(cfg=_FakeCfg())

        assert len(footprints) == 1
        assert footprints[0].polygon is None
        assert footprints[0].bbox == (-10.0, 10.0, 35.0, 55.0)
