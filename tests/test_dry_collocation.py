"""Tests for dry_collocation.py's SarFootprint model and SAR-side discovery."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

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
        """WV-mode granules are handled by a separate function, not this
        one -- a WV Name (mode token 'WV') must be filtered out here, not
        turned into a (wrong) kind="polygon" footprint."""
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
    """CDSE catalogs an entire WV pass as one product, not one vignette
    per catalog entry. Each WV product's GeoFootprint is a "MultiPolygon"
    whose "coordinates" list already holds one small quad ring per
    vignette, directly in the catalog search response -- no manifest.safe
    fetch needed. These tests use a synthetic 3-vignette MultiPolygon."""

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
        still fall well outside it. Only centroids actually inside
        cfg.geographic_bounds should survive into points, and bbox must
        reflect just the survivors, not the full unfiltered set."""
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
        WV and non-WV discovery partition the same underlying CDSE query
        results by mode, so a non-WV Name must be filtered out here."""
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


class TestListRadarsat2CandidatesDry:
    def test_antimeridian_crossing_bbox_searches_both_sides(self, monkeypatch):
        """An antimeridian-crossing cfg.geographic_bounds (min_lon >
        max_lon, per GeographicBounds' own convention) must be split into
        two non-crossing windows -- mirroring
        RADARSAT2WindDownloader.download()'s own split_antimeridian_bbox
        handling -- with candidates from both sides of the split kept, not
        just whichever side happens to satisfy a naive min_lon/max_lon
        comparison."""
        from sar_validation.core import dry_collocation

        # 172E/172W (not 175E/175W) deliberately -- both sit far enough
        # from the +-180 edge that they don't also get swept in by
        # _lon_within_padded_bbox's own antimeridian-wraparound pad
        # allowance (default pad is 5 degrees), so each candidate matches
        # exactly one of the two split windows, not both.
        catalog_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0">\n'
            '  <dataset name="SAR-Wind-HH-10N-172E_v3r0_rsat2_'
            's202608010100000_e202608010101160_c202608010200000.nc"\n'
            '           urlPath="sar-winds/radarsat2/2026/08/SAR-Wind-HH-10N-172E_v3r0_rsat2_'
            's202608010100000_e202608010101160_c202608010200000.nc" />\n'
            '  <dataset name="SAR-Wind-HH-10N-172W_v3r0_rsat2_'
            's202608020100000_e202608020101160_c202608020200000.nc"\n'
            '           urlPath="sar-winds/radarsat2/2026/08/SAR-Wind-HH-10N-172W_v3r0_rsat2_'
            's202608020100000_e202608020101160_c202608020200000.nc" />\n'
            '</catalog>\n'
        )

        class _FakeResp:
            def __init__(self, text):
                self._text = text

            def read(self):
                return self._text.encode()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(url, timeout=None):
            return _FakeResp(catalog_xml)

        monkeypatch.setattr(dry_collocation.urllib.request, "urlopen", fake_urlopen)

        class _FakeCfg:
            # min_lon > max_lon signals a bbox wrapping the antimeridian
            # -- covers 170E through 170W via the Pacific.
            geographic_bounds = SimpleNamespace(min_lon=170.0, max_lon=-170.0, min_lat=0.0, max_lat=20.0)
            temporal_bounds = SimpleNamespace(start="2026-08-01", end="2026-08-03")

        candidates = dry_collocation._list_radarsat2_candidates_dry(_FakeCfg())

        url_paths = [url_path for url_path, _ts in candidates]
        assert any("172E" in p for p in url_paths)
        assert any("172W" in p for p in url_paths)
        assert len(candidates) == 2


class TestSearchNisarSme2Dry:
    def test_both_candidates_searched_and_merged(self, monkeypatch):
        """NISAR SME2's underlying CMR collection changed mid-mission with
        no temporal overlap between its beta and provisional
        product-maturity levels (sar_sources.NISAR_SME2_CANDIDATES) --
        _search_nisar_sme2_dry must query BOTH candidates and merge their
        results, not just the first. Verified here by returning different
        fake granules keyed off the short_name kwarg earthaccess.search_data
        is called with."""
        from sar_validation.core import dry_collocation

        fake_earthaccess = MagicMock()
        fake_earthaccess.search_data.side_effect = (
            lambda short_name, version, bounding_box, temporal: [
                {"meta": {"native-id": f"{short_name}_granule"}}
            ]
        )

        class _FakeCfg:
            geographic_bounds = SimpleNamespace(min_lon=-10.0, max_lon=10.0, min_lat=35.0, max_lat=55.0)
            temporal_bounds = SimpleNamespace(start="2026-08-01", end="2026-08-02")

        monkeypatch.setitem(sys.modules, "earthaccess", fake_earthaccess)

        results = dry_collocation._search_nisar_sme2_dry(cfg=_FakeCfg())

        assert fake_earthaccess.search_data.call_count == 2
        native_ids = {r["meta"]["native-id"] for r in results}
        assert native_ids == {
            "NISAR_L3_SME2_BETA_V1_granule",
            "NISAR_L3_SME2_PROVISIONAL_V1_granule",
        }

    def test_authenticates_before_searching(self, monkeypatch):
        """Mirrors EarthdataSoilMoistureDownloader.download()'s own
        unconditional authenticate_earthdata() call before its
        per-candidate search loop -- a dry (no-download) search still
        needs valid NASA Earthdata Login credentials to query CMR, the
        same way _query_sentinel1_ocn_dry authenticates against CDSE
        before its own pure catalog query."""
        from sar_validation.core import dry_collocation

        calls = []

        fake_earthaccess = MagicMock()
        fake_earthaccess.search_data.side_effect = (
            lambda **kwargs: calls.append("search") or []
        )
        monkeypatch.setitem(sys.modules, "earthaccess", fake_earthaccess)

        from sar_validation.downloaders import base as _base

        monkeypatch.setattr(_base, "authenticate_earthdata", lambda: calls.append("auth"))

        class _FakeCfg:
            geographic_bounds = SimpleNamespace(min_lon=-10.0, max_lon=10.0, min_lat=35.0, max_lat=55.0)
            temporal_bounds = SimpleNamespace(start="2026-08-01", end="2026-08-02")

        dry_collocation._search_nisar_sme2_dry(cfg=_FakeCfg())

        # authenticate_earthdata() must run before any search_data() call,
        # mirroring EarthdataSoilMoistureDownloader.download()'s ordering.
        assert calls == ["auth", "search", "search"]


class TestDiscoverNisarSme2FootprintsDry:
    def test_granule_with_real_polygon_geometry(self, monkeypatch):
        from sar_validation.core import dry_collocation

        fake_granule = {
            "umm": {
                "TemporalExtent": {"RangeDateTime": {
                    "BeginningDateTime": "2026-08-01T06:00:00.000Z",
                    "EndingDateTime": "2026-08-01T06:05:00.000Z",
                }},
                "SpatialExtent": {"HorizontalSpatialDomain": {"Geometry": {
                    "GPolygons": [{"Boundary": {"Points": [
                        {"Longitude": -10.0, "Latitude": 35.0},
                        {"Longitude": 10.0, "Latitude": 35.0},
                        {"Longitude": 10.0, "Latitude": 55.0},
                        {"Longitude": -10.0, "Latitude": 55.0},
                    ]}}]
                }}},
            },
            "meta": {"native-id": "NISAR_L3_PR_SME2_003_005_A_014_..._001.h5"},
        }
        monkeypatch.setattr(dry_collocation, "_search_nisar_sme2_dry", lambda cfg: [fake_granule])

        footprints = dry_collocation._discover_nisar_sme2_footprints_dry(cfg=object())

        assert len(footprints) == 1
        fp = footprints[0]
        assert fp.kind == "polygon"
        assert fp.polygon == [(35.0, -10.0), (35.0, 10.0), (55.0, 10.0), (55.0, -10.0)]
        assert fp.bbox == (-10.0, 10.0, 35.0, 55.0)
        assert fp.sensing_start.isoformat().startswith("2026-08-01T06:00:00")
        assert fp.source_file == fake_granule["meta"]["native-id"]

    def test_closed_ring_drops_repeated_closing_vertex(self, monkeypatch):
        """NISAR SME2's GPolygons repeat their first vertex to close the
        ring, exactly like CDSE's GeoFootprint. That trailing duplicate
        must be dropped so SarFootprint.polygon holds one entry per
        distinct vertex, mirroring
        _polygon_and_bbox_from_geofootprint's convention."""
        from sar_validation.core import dry_collocation

        fake_granule = {
            "umm": {
                "TemporalExtent": {"RangeDateTime": {
                    "BeginningDateTime": "2026-08-01T06:00:00.000Z",
                    "EndingDateTime": "2026-08-01T06:05:00.000Z",
                }},
                "SpatialExtent": {"HorizontalSpatialDomain": {"Geometry": {
                    "GPolygons": [{"Boundary": {"Points": [
                        {"Latitude": 22.79518, "Longitude": -109.6084},
                        {"Latitude": 24.10005, "Longitude": -109.97092},
                        {"Latitude": 23.50525, "Longitude": -112.38056},
                        {"Latitude": 22.20781, "Longitude": -111.99627},
                        {"Latitude": 22.79518, "Longitude": -109.6084},
                    ]}}]
                }}},
            },
            "meta": {"native-id": "NISAR_L3_PR_SME2_real_granule.h5"},
        }
        monkeypatch.setattr(dry_collocation, "_search_nisar_sme2_dry", lambda cfg: [fake_granule])

        footprints = dry_collocation._discover_nisar_sme2_footprints_dry(cfg=object())

        assert len(footprints) == 1
        assert footprints[0].polygon is not None and len(footprints[0].polygon) == 4

    def test_missing_gpolygons_falls_back_to_bounding_rectangle(self, monkeypatch):
        """Defensive fallback: a granule whose Geometry has no GPolygons
        (a different/future NISAR collection, or an edge case) must still
        produce a usable bbox-only footprint from BoundingRectangles,
        rather than being dropped."""
        from sar_validation.core import dry_collocation

        fake_granule = {
            "umm": {
                "TemporalExtent": {"RangeDateTime": {
                    "BeginningDateTime": "2026-08-01T06:00:00.000Z",
                    "EndingDateTime": "2026-08-01T06:05:00.000Z",
                }},
                "SpatialExtent": {"HorizontalSpatialDomain": {"Geometry": {
                    "BoundingRectangles": [{
                        "WestBoundingCoordinate": -10.0,
                        "EastBoundingCoordinate": 10.0,
                        "SouthBoundingCoordinate": 35.0,
                        "NorthBoundingCoordinate": 55.0,
                    }]
                }}},
            },
            "meta": {"native-id": "NISAR_L3_PR_SME2_bbox_only.h5"},
        }
        monkeypatch.setattr(dry_collocation, "_search_nisar_sme2_dry", lambda cfg: [fake_granule])

        footprints = dry_collocation._discover_nisar_sme2_footprints_dry(cfg=object())

        assert len(footprints) == 1
        assert footprints[0].polygon is None
        assert footprints[0].bbox == (-10.0, 10.0, 35.0, 55.0)

    def test_missing_geometry_falls_back_to_cfg_bbox(self, monkeypatch):
        """When a granule has neither GPolygons nor BoundingRectangles,
        fall back to the recipe's own requested bbox -- the same
        "degrade, don't drop" convention used for Sentinel-1/RADARSAT-2."""
        from sar_validation.core import dry_collocation

        fake_granule = {
            "umm": {
                "TemporalExtent": {"RangeDateTime": {
                    "BeginningDateTime": "2026-08-01T06:00:00.000Z",
                    "EndingDateTime": "2026-08-01T06:05:00.000Z",
                }},
                "SpatialExtent": {"HorizontalSpatialDomain": {"Geometry": {}}},
            },
            "meta": {"native-id": "NISAR_L3_PR_SME2_no_geometry.h5"},
        }
        monkeypatch.setattr(dry_collocation, "_search_nisar_sme2_dry", lambda cfg: [fake_granule])

        class _FakeCfg:
            geographic_bounds = SimpleNamespace(min_lon=-10.0, max_lon=10.0, min_lat=35.0, max_lat=55.0)

        footprints = dry_collocation._discover_nisar_sme2_footprints_dry(cfg=_FakeCfg())

        assert len(footprints) == 1
        assert footprints[0].polygon is None
        assert footprints[0].bbox == (-10.0, 10.0, 35.0, 55.0)

    def test_malformed_gpolygons_degrades_to_cfg_bbox_without_aborting_batch(self, monkeypatch):
        """A granule with GPolygons present but missing the "Boundary"
        key (or any other malformed-geometry shape) must degrade that one
        granule to the cfg-bbox fallback, matching
        _polygon_and_bbox_from_geofootprint's "degrade, don't drop"
        convention -- not raise and abort discovery for every other
        granule in the same batch."""
        from sar_validation.core import dry_collocation

        malformed_granule = {
            "umm": {
                "TemporalExtent": {"RangeDateTime": {
                    "BeginningDateTime": "2026-08-01T06:00:00.000Z",
                    "EndingDateTime": "2026-08-01T06:05:00.000Z",
                }},
                "SpatialExtent": {"HorizontalSpatialDomain": {"Geometry": {
                    "GPolygons": [{}],  # missing "Boundary"
                }}},
            },
            "meta": {"native-id": "NISAR_L3_PR_SME2_malformed.h5"},
        }
        good_granule = {
            "umm": {
                "TemporalExtent": {"RangeDateTime": {
                    "BeginningDateTime": "2026-08-01T07:00:00.000Z",
                    "EndingDateTime": "2026-08-01T07:05:00.000Z",
                }},
                "SpatialExtent": {"HorizontalSpatialDomain": {"Geometry": {
                    "GPolygons": [{"Boundary": {"Points": [
                        {"Longitude": -10.0, "Latitude": 35.0},
                        {"Longitude": 10.0, "Latitude": 35.0},
                        {"Longitude": 10.0, "Latitude": 55.0},
                        {"Longitude": -10.0, "Latitude": 55.0},
                    ]}}]
                }}},
            },
            "meta": {"native-id": "NISAR_L3_PR_SME2_good.h5"},
        }
        monkeypatch.setattr(
            dry_collocation, "_search_nisar_sme2_dry", lambda cfg: [malformed_granule, good_granule],
        )

        class _FakeCfg:
            geographic_bounds = SimpleNamespace(min_lon=-20.0, max_lon=20.0, min_lat=30.0, max_lat=60.0)

        footprints = dry_collocation._discover_nisar_sme2_footprints_dry(cfg=_FakeCfg())

        assert len(footprints) == 2
        malformed_fp = next(fp for fp in footprints if fp.source_file == "NISAR_L3_PR_SME2_malformed.h5")
        assert malformed_fp.polygon is None
        assert malformed_fp.bbox == (-20.0, 20.0, 30.0, 60.0)

        good_fp = next(fp for fp in footprints if fp.source_file == "NISAR_L3_PR_SME2_good.h5")
        assert good_fp.polygon == [(35.0, -10.0), (35.0, 10.0), (55.0, 10.0), (55.0, -10.0)]
        assert good_fp.bbox == (-10.0, 10.0, 35.0, 55.0)


class TestClmsSsmFootprints:
    """CLMS SSM's own catalog footprint is the product tile's whole
    nominal region (e.g. all of Europe), not real per-day coverage --
    unlike every other SAR source in this module. The dry-search path
    therefore propagates Sentinel-1's own orbit (via
    orbit_coverage.orbit_overlap_windows) across each candidate tile's
    day to predict the real overpass corridor, trying each of the three
    registered satellites (1A/1B/1C) and taking the union -- deduplicated
    by (start, end) value, since a tile can be covered by more than one
    satellite and each would otherwise report the exact same matched
    window. The from-downloaded path instead reads the real non-NaN pixel
    extent directly off an already-downloaded GeoTIFF, which is more
    accurate than propagation since the real data is already on disk."""

    def test_dry_discovery_uses_orbit_overlap_windows_per_candidate_satellite(self, monkeypatch):
        from sar_validation.core import dry_collocation

        fake_records = [
            {
                "Name": "S1A_CLMS_SSM1km_20260801T000000.tif",
                "ContentDate_Start": "2026-08-01T00:00:00.000Z",
                "ContentDate_End": "2026-08-01T23:59:59.000Z",
            },
        ]
        monkeypatch.setattr(dry_collocation, "_query_clms_ssm_dry", lambda cfg: fake_records)

        call_args = []

        def _fake_windows(satellite, start, end, min_lon, max_lon, min_lat, max_lat, **kwargs):
            call_args.append(satellite)
            # Every candidate satellite reports the exact same matched
            # window here -- the union must dedupe these down to one
            # footprint, not one per satellite.
            return [(start, end)]

        monkeypatch.setattr(dry_collocation.orbit_coverage, "orbit_overlap_windows", _fake_windows)

        class _FakeCfg:
            geographic_bounds = SimpleNamespace(min_lon=-10.0, max_lon=30.0, min_lat=35.0, max_lat=60.0)

        footprints = dry_collocation._discover_clms_ssm_footprints_dry(cfg=_FakeCfg())

        assert len(footprints) == 1
        assert footprints[0].kind == "orbit_swath"
        assert footprints[0].polygon is None
        assert footprints[0].bbox == (-10.0, 30.0, 35.0, 60.0)
        assert "sentinel-1a" in call_args  # at least Sentinel-1A's orbit was checked
        assert "sentinel-1b" in call_args
        assert "sentinel-1c" in call_args

    def test_dry_discovery_unions_distinct_windows_across_satellites(self, monkeypatch):
        """When different satellites match genuinely different windows
        (not just the same one reported twice), every distinct window
        must survive as its own footprint -- the dedup is by (start, end)
        value equality, not a "keep only the first satellite's result"
        shortcut."""
        from datetime import datetime as dt

        from sar_validation.core import dry_collocation

        fake_records = [
            {
                "Name": "S1A_CLMS_SSM1km_20260801T000000.tif",
                "ContentDate_Start": "2026-08-01T00:00:00.000Z",
                "ContentDate_End": "2026-08-01T23:59:59.000Z",
            },
        ]
        monkeypatch.setattr(dry_collocation, "_query_clms_ssm_dry", lambda cfg: fake_records)

        _WINDOWS_BY_SAT = {
            "sentinel-1a": [(dt(2026, 8, 1, 5, 0), dt(2026, 8, 1, 5, 5))],
            "sentinel-1b": [(dt(2026, 8, 1, 17, 0), dt(2026, 8, 1, 17, 5))],
            "sentinel-1c": [(dt(2026, 8, 1, 5, 0), dt(2026, 8, 1, 5, 5))],  # duplicate of 1a's
        }

        def _fake_windows(satellite, start, end, min_lon, max_lon, min_lat, max_lat, **kwargs):
            return _WINDOWS_BY_SAT[satellite]

        monkeypatch.setattr(dry_collocation.orbit_coverage, "orbit_overlap_windows", _fake_windows)

        class _FakeCfg:
            geographic_bounds = SimpleNamespace(min_lon=-10.0, max_lon=30.0, min_lat=35.0, max_lat=60.0)

        footprints = dry_collocation._discover_clms_ssm_footprints_dry(cfg=_FakeCfg())

        windows = {(fp.sensing_start, fp.sensing_end) for fp in footprints}
        assert windows == {
            (dt(2026, 8, 1, 5, 0), dt(2026, 8, 1, 5, 5)),
            (dt(2026, 8, 1, 17, 0), dt(2026, 8, 1, 17, 5)),
        }

    def test_from_downloaded_reads_real_non_nan_pixel_extent(self, tmp_path, monkeypatch):
        import numpy as np
        import xarray as xr

        from sar_validation.core import dry_collocation

        # A 4x4 grid where only the top-left 2x2 block has real data --
        # the real footprint should be just that block's lat/lon range,
        # not the whole grid's.
        lon2d, lat2d = np.meshgrid([0.0, 1.0, 2.0, 3.0], [40.0, 41.0, 42.0, 43.0])
        sarssm = np.full((4, 4), np.nan)
        sarssm[0:2, 0:2] = 25.0  # valid data only in this block
        fake_ds = xr.Dataset(
            {"sarSSM": (("y", "x"), sarssm)},
            coords={"lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d), "time": datetime(2026, 8, 1)},
        )
        monkeypatch.setattr(
            dry_collocation.DataTreeConverter, "from_sar_l3_ssm_geotiff", staticmethod(lambda p: fake_ds),
        )

        fp = dry_collocation._clms_ssm_footprint_from_downloaded(tmp_path / "fake.tif")

        assert fp is not None
        assert fp.kind == "orbit_swath"
        assert fp.bbox == (0.0, 1.0, 40.0, 41.0)  # only the valid 2x2 block's extent
        assert fp.polygon is None
        assert fp.source_file == str(tmp_path / "fake.tif")

    def test_from_downloaded_returns_none_when_geotiff_missing(self, monkeypatch):
        """from_sar_l3_ssm_geotiff itself returns None when the file
        doesn't exist -- that must propagate as None here too, not raise
        (mirrors the converter's own "missing file" contract)."""
        from sar_validation.core import dry_collocation

        monkeypatch.setattr(
            dry_collocation.DataTreeConverter, "from_sar_l3_ssm_geotiff", staticmethod(lambda p: None),
        )

        fp = dry_collocation._clms_ssm_footprint_from_downloaded("/nonexistent/fake.tif")

        assert fp is None

    def test_from_downloaded_returns_none_when_all_pixels_nan(self, monkeypatch):
        """An all-NaN raster (e.g. a tile with zero valid retrievals for
        the day) has no real pixel extent to report -- must degrade to
        None, not a degenerate/nonsensical bbox."""
        import numpy as np
        import xarray as xr

        from sar_validation.core import dry_collocation

        lon2d, lat2d = np.meshgrid([0.0, 1.0], [40.0, 41.0])
        fake_ds = xr.Dataset(
            {"sarSSM": (("y", "x"), np.full((2, 2), np.nan))},
            coords={"lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d), "time": datetime(2026, 8, 1)},
        )
        monkeypatch.setattr(
            dry_collocation.DataTreeConverter, "from_sar_l3_ssm_geotiff", staticmethod(lambda p: fake_ds),
        )

        fp = dry_collocation._clms_ssm_footprint_from_downloaded("fake.tif")

        assert fp is None


class TestTopLevelDispatchers:
    def test_discover_sar_footprints_dry_dispatches_on_source_key(self, monkeypatch):
        from sar_validation.core import dry_collocation

        sentinel = object()
        monkeypatch.setattr(dry_collocation, "_discover_sentinel1_ocn_footprints_dry", lambda cfg: [sentinel])
        monkeypatch.setattr(dry_collocation, "_discover_sentinel1_wv_footprints_dry", lambda cfg: [])

        class _FakeSarDataSpec:
            source = "sentinel1_l2_ocn"

        result = dry_collocation.discover_sar_footprints_dry(_FakeSarDataSpec(), cfg=object())

        assert result == [sentinel]

    def test_discover_sar_footprints_dry_unknown_source_raises(self):
        from sar_validation.core import dry_collocation

        class _FakeSarDataSpec:
            source = "not_a_real_source"

        with pytest.raises(ValueError, match="not_a_real_source"):
            dry_collocation.discover_sar_footprints_dry(_FakeSarDataSpec(), cfg=object())

    def test_sar_footprints_from_downloaded_clms_ssm(self, tmp_path, monkeypatch):
        from sar_validation.core import dry_collocation

        fp = object()
        monkeypatch.setattr(dry_collocation, "_clms_ssm_footprint_from_downloaded", lambda p: fp)

        class _FakeSarSourceSpec:
            key = "sentinel1_clms_ssm"

        result = dry_collocation.sar_footprints_from_downloaded(
            [tmp_path / "a.tif"], _FakeSarSourceSpec(),
        )

        assert result == [fp]


class TestPredictSourceDispatch:
    def test_unregistered_source_type_returns_unknown_not_an_exception(self):
        """predict_source must never raise for the caller (the report
        loop iterates every configured source) -- an
        unrecognized source_type is itself an "unknown" verdict, not a
        crash."""
        from sar_validation.core.dry_collocation import predict_source

        class _FakeSource:
            source_type = "not_a_real_source_type"

        result = predict_source(_FakeSource(), cfg=object(), sar_footprints=[])

        assert result.verdict == "unknown"
        assert result.source_type == "not_a_real_source_type"
        assert "not_a_real_source_type" in result.detail


class TestPointInFootprint:
    def test_polygon_kind_uses_real_point_in_polygon(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0),
            polygon=[(35.0, -10.0), (35.0, 10.0), (55.0, 10.0), (55.0, -10.0)],
            points=None, sensing_start=datetime(2026, 8, 1), sensing_end=datetime(2026, 8, 1),
            source_file="s1.SAFE",
        )
        # A point inside the bbox but outside a smaller real polygon would
        # be wrongly accepted by a bbox-only check -- prove the real
        # orbit_coverage._point_in_polygon is actually being called by
        # monkeypatching it to force a "no" and confirming the result
        # follows it, not the bbox.
        monkeypatch.setattr(dry_collocation.orbit_coverage, "_point_in_polygon", lambda lat, lon, polygon: False)

        assert dry_collocation._point_in_footprint(45.0, 0.0, footprint) is False

    def test_wv_points_kind_uses_vignette_search_radius(self):
        from sar_validation.core.dry_collocation import SarFootprint, _point_in_footprint

        footprint = SarFootprint(
            kind="wv_points", bbox=(-1.0, 1.0, -1.0, 1.0), polygon=None,
            points=[(0.0, 0.0)], sensing_start=datetime(2026, 8, 1), sensing_end=datetime(2026, 8, 1),
            source_file="wv.SAFE",
        )
        # Within the ~20km vignette radius of (0,0).
        assert _point_in_footprint(0.05, 0.05, footprint) is True
        # Far outside any vignette's radius.
        assert _point_in_footprint(10.0, 10.0, footprint) is False

    def test_orbit_swath_kind_falls_back_to_bbox(self):
        from sar_validation.core.dry_collocation import SarFootprint, _point_in_footprint

        footprint = SarFootprint(
            kind="orbit_swath", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1), sensing_end=datetime(2026, 8, 1), source_file="ssm.tif",
        )
        assert _point_in_footprint(45.0, 0.0, footprint) is True
        assert _point_in_footprint(0.0, 0.0, footprint) is False


class TestOrbitCorridorPredicate:
    """_predict_orbit_corridor_source resolves its time tolerance via
    _resolve_temporal_padding_minutes(cfg, source_type) -- the real
    tolerance-resolution function orchestrator.py uses -- not a flat
    cfg.time_tolerance_minutes attribute (RecipeConfig has no such
    attribute), so every test here mocks that resolver rather than
    relying on a fake cfg shape."""

    def test_collocated_when_a_candidate_overlaps_a_footprint_window(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        candidate_start = datetime(2026, 8, 1, 6, 0, 0)
        candidate_end = datetime(2026, 8, 1, 6, 3, 0)

        def _fake_list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end):
            return [("h103_fake.nc", candidate_start, candidate_end)]

        def _fake_orbit_overlap_windows(satellite, start, end, min_lon, max_lon, min_lat, max_lat, **kwargs):
            return [(start, end)]  # full overlap

        monkeypatch.setattr(dry_collocation.orbit_coverage, "orbit_overlap_windows", _fake_orbit_overlap_windows)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = dry_collocation._predict_orbit_corridor_source(
            source=object(), cfg=object(), sar_footprints=[footprint],
            satellite_resolver=lambda name: "metop-b", list_candidates_dry=_fake_list_candidates_dry,
            source_type="ascat_ssm",
        )

        assert result.verdict == "collocated"
        assert result.bucket == "orbit-corridor"

    def test_none_predicted_when_no_candidate_overlaps_any_footprint(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        def _fake_list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end):
            return [("h103_fake.nc", datetime(2026, 8, 1, 6, 0, 0), datetime(2026, 8, 1, 6, 3, 0))]

        def _fake_orbit_overlap_windows(satellite, start, end, min_lon, max_lon, min_lat, max_lat, **kwargs):
            return []  # never overlaps the bbox at all

        monkeypatch.setattr(dry_collocation.orbit_coverage, "orbit_overlap_windows", _fake_orbit_overlap_windows)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = dry_collocation._predict_orbit_corridor_source(
            source=object(), cfg=object(), sar_footprints=[footprint],
            satellite_resolver=lambda name: "metop-b", list_candidates_dry=_fake_list_candidates_dry,
            source_type="ascat_ssm",
        )

        assert result.verdict == "none-predicted"

    def test_wv_points_footprint_checked_per_vignette_never_batched(self, monkeypatch):
        """A wv_points footprint with N points must drive N separate
        orbit_overlap_windows calls (one per vignette, zero-width bbox),
        never one call against an enclosing box."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        call_count = {"n": 0}

        def _fake_list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end):
            return [("h103_fake.nc", datetime(2026, 8, 1, 6, 0, 0), datetime(2026, 8, 1, 6, 3, 0))]

        def _fake_orbit_overlap_windows(satellite, start, end, min_lon, max_lon, min_lat, max_lat, **kwargs):
            call_count["n"] += 1
            return []

        monkeypatch.setattr(dry_collocation.orbit_coverage, "orbit_overlap_windows", _fake_orbit_overlap_windows)

        footprint = SarFootprint(
            kind="wv_points", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None,
            points=[(40.0, 0.0), (41.0, 1.0), (42.0, 2.0)],
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="wv.SAFE",
        )

        dry_collocation._predict_orbit_corridor_source(
            source=object(), cfg=object(), sar_footprints=[footprint],
            satellite_resolver=lambda name: "metop-b", list_candidates_dry=_fake_list_candidates_dry,
            source_type="ascat_ssm",
        )

        assert call_count["n"] == 3  # one call per vignette point, not one for the whole footprint

    def test_satellite_resolver_called_per_candidate_not_once_per_source(self, monkeypatch):
        """Two candidates from different satellites in the same listing
        must each get their own orbit_overlap_windows call with their own
        resolved satellite key -- a single fixed satellite for the whole
        predicate would silently mis-refine one of them."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        def _fake_list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end):
            return [
                ("metopb_fake.nc", datetime(2026, 8, 1, 6, 0, 0), datetime(2026, 8, 1, 6, 3, 0)),
                ("metopc_fake.nc", datetime(2026, 8, 1, 6, 0, 0), datetime(2026, 8, 1, 6, 3, 0)),
            ]

        seen_satellites = []

        def _fake_orbit_overlap_windows(satellite, start, end, min_lon, max_lon, min_lat, max_lat, **kwargs):
            seen_satellites.append(satellite)
            return []

        def _resolver(candidate_name):
            return "metop-b" if "metopb" in candidate_name else "metop-c"

        monkeypatch.setattr(dry_collocation.orbit_coverage, "orbit_overlap_windows", _fake_orbit_overlap_windows)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        dry_collocation._predict_orbit_corridor_source(
            source=object(), cfg=object(), sar_footprints=[footprint],
            satellite_resolver=_resolver, list_candidates_dry=_fake_list_candidates_dry,
            source_type="ascat_ssm",
        )

        assert seen_satellites == ["metop-b", "metop-c"]

    def test_no_footprints_is_unknown(self):
        from sar_validation.core import dry_collocation

        result = dry_collocation._predict_orbit_corridor_source(
            source=object(), cfg=object(), sar_footprints=[],
            satellite_resolver=lambda name: "metop-b", list_candidates_dry=lambda *a, **k: [],
            source_type="ascat_ssm",
        )

        assert result.verdict == "unknown"


class TestPredictAscatSsm:
    def test_footprint_before_cutoff_only_checked_via_eumdac(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        calls = []

        def _fake_predict_orbit_corridor_source(
            source, cfg, sar_footprints, *, satellite_resolver, list_candidates_dry, source_type,
        ):
            calls.append(list_candidates_dry)
            return dry_collocation.SourcePrediction(
                source_type=source_type, bucket="orbit-corridor", verdict="collocated", detail="ok",
            )

        monkeypatch.setattr(dry_collocation, "_predict_orbit_corridor_source", _fake_predict_orbit_corridor_source)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2024, 1, 1, 6, 0, 0), sensing_end=datetime(2024, 1, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = dry_collocation._predict_ascat_ssm(source=object(), cfg=object(), sar_footprints=[footprint])

        assert result.verdict == "collocated"
        assert calls == [dry_collocation._eumdac_ascat_ssm_list_candidates_dry]

    def test_footprint_within_hsaf_window_only_checked_via_hsaf(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        calls = []

        def _fake_predict_orbit_corridor_source(
            source, cfg, sar_footprints, *, satellite_resolver, list_candidates_dry, source_type,
        ):
            calls.append(list_candidates_dry)
            return dry_collocation.SourcePrediction(
                source_type=source_type, bucket="orbit-corridor", verdict="none-predicted", detail="ok",
            )

        monkeypatch.setattr(dry_collocation, "_predict_orbit_corridor_source", _fake_predict_orbit_corridor_source)

        now = datetime.now(timezone.utc)
        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=now - timedelta(days=1), sensing_end=now - timedelta(days=1) + timedelta(minutes=1),
            source_file="s1.SAFE",
        )

        result = dry_collocation._predict_ascat_ssm(source=object(), cfg=object(), sar_footprints=[footprint])

        assert result.verdict == "none-predicted"
        assert calls == [dry_collocation._hsaf_list_candidates_dry]

    def test_collocated_if_either_branch_predicts_collocated(self, monkeypatch):
        """A footprint whose padded window straddles both archives must
        be collocated if EITHER branch finds something -- never require
        both."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)
        # Force "always eligible for EUMDAC" for this test.
        monkeypatch.setattr(dry_collocation, "_ASCAT_COVERAGE_CUTOFF", "2099-01-01")

        def _fake_predict_orbit_corridor_source(
            source, cfg, sar_footprints, *, satellite_resolver, list_candidates_dry, source_type,
        ):
            is_hsaf = list_candidates_dry is dry_collocation._hsaf_list_candidates_dry
            verdict = "collocated" if is_hsaf else "none-predicted"
            return dry_collocation.SourcePrediction(
                source_type=source_type, bucket="orbit-corridor", verdict=verdict, detail="ok",
                matched_windows=[(datetime(2026, 8, 1), datetime(2026, 8, 1))] if verdict == "collocated" else None,
            )

        monkeypatch.setattr(dry_collocation, "_predict_orbit_corridor_source", _fake_predict_orbit_corridor_source)

        now = datetime.now(timezone.utc)
        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=now - timedelta(days=1), sensing_end=now - timedelta(days=1) + timedelta(minutes=1),
            source_file="s1.SAFE",
        )

        result = dry_collocation._predict_ascat_ssm(source=object(), cfg=object(), sar_footprints=[footprint])

        assert result.verdict == "collocated"

    def test_footprint_in_neither_window_is_unknown_not_none_predicted(self, monkeypatch):
        """A footprint older than the EUMDAC cutoff-check window and
        older than H-SAF's rolling window must be 'unknown' (fail
        toward inclusion), never a confident 'none-predicted'."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)
        # Force "never eligible for EUMDAC" for this test.
        monkeypatch.setattr(dry_collocation, "_ASCAT_COVERAGE_CUTOFF", "2020-01-01")

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2023, 1, 1, 6, 0, 0), sensing_end=datetime(2023, 1, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = dry_collocation._predict_ascat_ssm(source=object(), cfg=object(), sar_footprints=[footprint])

        assert result.verdict == "unknown"

    def test_registered_under_ascat_ssm_source_type(self):
        from sar_validation.core import dry_collocation

        assert dry_collocation._PREDICATES["ascat_ssm"] is dry_collocation._predict_ascat_ssm


class TestPredictScatterometerFtpSources:
    """One predict_source integration test per newly registered
    scatterometer_hy2b/hy2c/oceansat3 source_type -- exercises the real
    _PREDICATES dispatch (predict_source), not _predict_orbit_corridor_source
    directly. Each of these downloaders' own FTP listing only ever
    contains one satellite's data, so satellite_resolver is a fixed
    constant, unlike ascat_ssm's EUMDAC branch."""

    @pytest.mark.parametrize(
        "source_type,list_candidates_dry_name",
        [
            pytest.param("scatterometer_hy2b", "_hy2b_list_candidates_dry", id="hy2b"),
            pytest.param("scatterometer_hy2c", "_hy2c_list_candidates_dry", id="hy2c"),
            pytest.param("scatterometer_oceansat3", "_oceansat3_list_candidates_dry", id="oceansat3"),
        ],
    )
    def test_collocated_via_predict_source(self, monkeypatch, source_type, list_candidates_dry_name):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint, predict_source

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        candidate_start = datetime(2026, 8, 1, 6, 0, 0)
        candidate_end = datetime(2026, 8, 1, 6, 3, 0)

        def _fake_list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end):
            return [("fake_candidate.nc", candidate_start, candidate_end)]

        monkeypatch.setattr(dry_collocation, list_candidates_dry_name, _fake_list_candidates_dry)

        def _fake_orbit_overlap_windows(satellite, start, end, min_lon, max_lon, min_lat, max_lat, **kwargs):
            return [(start, end)]  # full overlap

        monkeypatch.setattr(dry_collocation.orbit_coverage, "orbit_overlap_windows", _fake_orbit_overlap_windows)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = predict_source(
            SimpleNamespace(source_type=source_type), cfg=object(), sar_footprints=[footprint],
        )

        assert result.verdict == "collocated"
        assert result.bucket == "orbit-corridor"
        assert result.source_type == source_type

    @pytest.mark.parametrize(
        "source_type",
        ["scatterometer_hy2b", "scatterometer_hy2c", "scatterometer_oceansat3"],
    )
    def test_registered_under_own_source_type(self, source_type):
        from sar_validation.core import dry_collocation

        predicate = dry_collocation._PREDICATES[source_type]
        assert predicate is getattr(dry_collocation, f"_predict_{source_type}")


class TestPredictSmosSsm:
    """predict_source integration test for the newly registered
    smos_ssm source_type -- exercises the real _PREDICATES dispatch, not
    _predict_orbit_corridor_source directly."""

    def test_collocated_via_predict_source(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint, predict_source

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        candidate_start = datetime(2026, 8, 1, 6, 0, 0)
        candidate_end = datetime(2026, 8, 1, 6, 3, 0)

        def _fake_list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end):
            return [("fake_candidate.nc", candidate_start, candidate_end)]

        monkeypatch.setattr(dry_collocation, "_smos_list_candidates_dry", _fake_list_candidates_dry)

        def _fake_orbit_overlap_windows(satellite, start, end, min_lon, max_lon, min_lat, max_lat, **kwargs):
            return [(start, end)]  # full overlap

        monkeypatch.setattr(dry_collocation.orbit_coverage, "orbit_overlap_windows", _fake_orbit_overlap_windows)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = predict_source(
            SimpleNamespace(source_type="smos_ssm"), cfg=object(), sar_footprints=[footprint],
        )

        assert result.verdict == "collocated"
        assert result.bucket == "orbit-corridor"
        assert result.source_type == "smos_ssm"

    def test_registered_under_smos_ssm_source_type(self):
        from sar_validation.core import dry_collocation

        assert dry_collocation._PREDICATES["smos_ssm"] is dry_collocation._predict_smos_ssm


class TestBboxOverlapsFootprint:
    def test_overlapping_bbox_returns_true(self):
        from sar_validation.core.dry_collocation import SarFootprint, _bbox_overlaps_footprint

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1), sensing_end=datetime(2026, 8, 1), source_file="s1.SAFE",
        )
        assert _bbox_overlaps_footprint(5.0, 15.0, 40.0, 50.0, footprint) is True

    def test_disjoint_bbox_returns_false(self):
        from sar_validation.core.dry_collocation import SarFootprint, _bbox_overlaps_footprint

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1), sensing_end=datetime(2026, 8, 1), source_file="s1.SAFE",
        )
        assert _bbox_overlaps_footprint(50.0, 60.0, 0.0, 10.0, footprint) is False
