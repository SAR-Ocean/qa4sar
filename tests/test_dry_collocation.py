"""Tests for dry_collocation.py's SarFootprint model and SAR-side discovery."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

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
            geographic_bounds = SimpleNamespace(min_lon=-10.0, max_lon=10.0, min_lat=35.0, max_lat=55.0)

        footprints = dry_collocation._discover_sentinel1_ocn_footprints_dry(cfg=_FakeCfg())

        assert len(footprints) == 1
        assert footprints[0].polygon is None
        assert footprints[0].bbox == (-10.0, 10.0, 35.0, 55.0)


class TestDiscoverSentinel1WvFootprintsDry:
    """CONFIRMED LIVE 2026-08-18 (curl against the real CDSE OData API):
    CDSE catalogs an entire WV pass as ONE product, not one vignette per
    catalog entry. Each WV product's GeoFootprint is a "MultiPolygon"
    whose "coordinates" list already holds one small quad ring per
    vignette (125-145 per product in three real samples inspected) --
    directly in the catalog search response, no manifest.safe fetch
    needed. These tests use a synthetic 3-vignette MultiPolygon."""

    _WV_MULTIPOLYGON = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[10.0, 20.0], [10.2, 20.0], [10.2, 20.2], [10.0, 20.2], [10.0, 20.0]]],
            [[[11.0, 21.0], [11.2, 21.0], [11.2, 21.2], [11.0, 21.2], [11.0, 21.0]]],
            [[[12.0, 22.0], [12.2, 22.0], [12.2, 22.2], [12.0, 22.2], [12.0, 22.0]]],
        ],
    }

    def test_wv_granule_becomes_a_multi_point_footprint(self, monkeypatch):
        from sar_validation.core import dry_collocation

        fake_records = [
            {
                "Name": "S1A_WV_OCN__2SSV_20260801T060000_20260801T060200_000000_000000_0000.SAFE",
                "ContentDate_Start": "2026-08-01T06:00:00.000Z",
                "ContentDate_End": "2026-08-01T06:02:00.000Z",
                "GeoFootprint": self._WV_MULTIPOLYGON,
            },
        ]
        monkeypatch.setattr(
            dry_collocation, "_query_sentinel1_ocn_dry", lambda cfg: fake_records,
        )

        class _FakeCfg:
            # Wide enough to keep all 3 synthetic vignette centroids.
            geographic_bounds = SimpleNamespace(min_lon=0.0, max_lon=20.0, min_lat=0.0, max_lat=30.0)

        footprints = dry_collocation._discover_sentinel1_wv_footprints_dry(cfg=_FakeCfg())

        assert len(footprints) == 1
        fp = footprints[0]
        assert fp.kind == "wv_points"
        assert fp.polygon is None
        assert fp.points is not None and len(fp.points) == 3
        # Each vignette's centroid is the mean of its 4 (deduped) unique
        # [lon, lat] corners, expressed as (lat, lon).
        assert fp.points == [(20.1, 10.1), (21.1, 11.1), (22.1, 12.1)]
        # bbox is the enclosing box over all vignette centroids.
        assert fp.bbox == (10.1, 12.1, 20.1, 22.1)
        assert fp.sensing_start.isoformat().startswith("2026-08-01T06:00:00")
        assert fp.source_file == fake_records[0]["Name"]

    def test_vignettes_outside_recipe_bbox_are_filtered_out(self, monkeypatch):
        """A WV product's overall catalog envelope only guarantees it
        touches the recipe's requested bbox -- individual vignettes can
        still fall well outside it (confirmed live: one real product
        spanned ~700km along-track, 145 vignettes). Only centroids
        actually inside cfg.geographic_bounds should survive into
        points, and bbox must reflect just the survivors, not the full
        unfiltered set."""
        from sar_validation.core import dry_collocation

        fake_records = [
            {
                "Name": "S1A_WV_OCN__2SSV_20260801T060000_20260801T060200_000000_000000_0000.SAFE",
                "ContentDate_Start": "2026-08-01T06:00:00.000Z",
                "ContentDate_End": "2026-08-01T06:02:00.000Z",
                "GeoFootprint": self._WV_MULTIPOLYGON,
            },
        ]
        monkeypatch.setattr(
            dry_collocation, "_query_sentinel1_ocn_dry", lambda cfg: fake_records,
        )

        class _FakeCfg:
            # Only the first vignette's centroid (20.1, 10.1) falls
            # inside this bbox; the other two (21.1, 11.1) and
            # (22.1, 12.1) fall outside it.
            geographic_bounds = SimpleNamespace(min_lon=9.0, max_lon=10.5, min_lat=19.0, max_lat=20.5)

        footprints = dry_collocation._discover_sentinel1_wv_footprints_dry(cfg=_FakeCfg())

        assert len(footprints) == 1
        fp = footprints[0]
        assert fp.kind == "wv_points"
        assert fp.points == [(20.1, 10.1)]
        # bbox is the enclosing box over the surviving centroid only, not
        # the full unfiltered (10.1, 12.1, 20.1, 22.1) envelope.
        assert fp.bbox == (10.1, 10.1, 20.1, 20.1)

    def test_all_vignettes_outside_recipe_bbox_falls_back_to_empty_points(self, monkeypatch):
        """When every vignette centroid falls outside cfg.geographic_bounds,
        the record must still degrade to a valid (empty-points, cfg-bbox)
        footprint -- the same "not drop this granule" convention already
        used for a missing/malformed GeoFootprint -- rather than being
        silently skipped."""
        from sar_validation.core import dry_collocation

        fake_records = [
            {
                "Name": "S1A_WV_OCN__2SSV_20260801T060000_20260801T060200_000000_000000_0000.SAFE",
                "ContentDate_Start": "2026-08-01T06:00:00.000Z",
                "ContentDate_End": "2026-08-01T06:02:00.000Z",
                "GeoFootprint": self._WV_MULTIPOLYGON,
            },
        ]
        monkeypatch.setattr(
            dry_collocation, "_query_sentinel1_ocn_dry", lambda cfg: fake_records,
        )

        class _FakeCfg:
            # None of the 3 synthetic vignette centroids fall inside this
            # bbox, which is far away from all of them.
            geographic_bounds = SimpleNamespace(min_lon=-10.0, max_lon=-5.0, min_lat=-10.0, max_lat=-5.0)

        footprints = dry_collocation._discover_sentinel1_wv_footprints_dry(cfg=_FakeCfg())

        assert len(footprints) == 1
        fp = footprints[0]
        assert fp.kind == "wv_points"
        assert fp.points == []
        assert fp.bbox == (-10.0, -5.0, -10.0, -5.0)

    def test_non_wv_granule_is_excluded_from_wv_discovery(self, monkeypatch):
        """Inverse of test_wv_granule_is_excluded_from_non_wv_discovery --
        Task 3 and Task 4 partition the same underlying CDSE query results
        by mode, so a non-WV Name must be filtered out here."""
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

        footprints = dry_collocation._discover_sentinel1_wv_footprints_dry(cfg=object())

        assert footprints == []

    def test_missing_geofootprint_falls_back_to_empty_points_and_cfg_bbox(self, monkeypatch):
        """A WV record whose GeoFootprint is absent/None must fail toward
        a usable (if empty-points) footprint, not be silently dropped --
        derive bbox from the recipe's own requested bbox, mirroring Task
        3's non-WV fallback."""
        from sar_validation.core import dry_collocation

        fake_records = [
            {
                "Name": "S1A_WV_OCN__2SSV_20260801T060000_20260801T060200_000000_000000_0000.SAFE",
                "ContentDate_Start": "2026-08-01T06:00:00.000Z",
                "ContentDate_End": "2026-08-01T06:02:00.000Z",
                "GeoFootprint": None,
            },
        ]
        monkeypatch.setattr(
            dry_collocation, "_query_sentinel1_ocn_dry", lambda cfg: fake_records,
        )

        class _FakeCfg:
            geographic_bounds = SimpleNamespace(min_lon=-10.0, max_lon=10.0, min_lat=35.0, max_lat=55.0)

        footprints = dry_collocation._discover_sentinel1_wv_footprints_dry(cfg=_FakeCfg())

        assert len(footprints) == 1
        assert footprints[0].kind == "wv_points"
        assert footprints[0].polygon is None
        assert footprints[0].points == []
        assert footprints[0].bbox == (-10.0, 10.0, 35.0, 55.0)


class TestDiscoverRadarsat2FootprintsDry:
    def test_granule_becomes_bbox_only_polygon_footprint(self, monkeypatch):
        from sar_validation.core import dry_collocation

        fake_candidates = [
            ("RS2_OK1234_PK5678_DK9012_SCWA_20260801_060000_HH_SGF.nc", datetime(2026, 8, 1, 6, 0, 0)),
        ]
        monkeypatch.setattr(dry_collocation, "_list_radarsat2_candidates_dry", lambda cfg: fake_candidates)
        monkeypatch.setattr(
            dry_collocation, "_radarsat2_ncml_bbox", lambda url_path: (-70.0, -60.0, 40.0, 50.0),
        )

        footprints = dry_collocation._discover_radarsat2_footprints_dry(cfg=object())

        assert len(footprints) == 1
        fp = footprints[0]
        assert fp.kind == "polygon"
        assert fp.polygon is None
        assert fp.bbox == (-70.0, -60.0, 40.0, 50.0)
