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


class TestAntimeridianAwareLonBounds:
    """A naive min()/max() over raw longitudes is only correct when the
    true span doesn't cross +/-180: points actually clustered near the
    dateline (e.g. 179.9 and -179.9, physically ~20km apart) would
    otherwise produce min=-179.9, max=179.9 -- read as a normal
    (non-wrapping) bbox, that's backwards, covering almost the entire
    globe instead of the narrow true region."""

    def test_non_wrapping_span_returns_naive_min_max(self):
        from sar_validation.core.dry_collocation import _antimeridian_aware_lon_bounds

        assert _antimeridian_aware_lon_bounds([-10.0, 0.0, 10.0]) == (-10.0, 10.0)

    def test_points_clustered_near_the_dateline_produce_a_wrapping_bbox(self):
        from sar_validation.core.dry_collocation import _antimeridian_aware_lon_bounds

        min_lon, max_lon = _antimeridian_aware_lon_bounds([179.9, -179.9])
        assert min_lon > max_lon  # this codebase's own wrap-convention signal
        assert min_lon == pytest.approx(179.9)
        assert max_lon == pytest.approx(-179.9)

    def test_wide_scattered_cluster_still_finds_the_narrow_true_span(self):
        """Several points scattered on both sides of the dateline (e.g.
        a whole footprint union, not just two extremes) must still
        resolve to the true minimal-spanning wrap bbox, not just handle
        the two-point case."""
        from sar_validation.core.dry_collocation import _antimeridian_aware_lon_bounds

        lons = [178.0, 179.5, -179.8, -178.5]
        min_lon, max_lon = _antimeridian_aware_lon_bounds(lons)
        assert min_lon == pytest.approx(178.0)
        assert max_lon == pytest.approx(-178.5)

    def test_empty_input_raises(self):
        from sar_validation.core.dry_collocation import _antimeridian_aware_lon_bounds

        with pytest.raises(ValueError):
            _antimeridian_aware_lon_bounds([])


class TestQuerySentinel1OcnDry:
    """_query_sentinel1_ocn_dry backs both _discover_sentinel1_ocn_
    footprints_dry and _discover_sentinel1_wv_footprints_dry -- every
    SAR-side discovery --dry-collocation does for Sentinel-1. Mirrors
    SARDownloader.query's own identical antimeridian-splitting tests
    (test_downloaders.py's TestSARDownloaderAntimeridian): query_products
    builds a WKT POLYGON directly from min_lon/max_lon with no
    antimeridian awareness of its own, so an unsplit wrapping bbox
    (min_lon > max_lon) produces a self-intersecting/inverted polygon --
    CDSE then returns geographically wrong results, not the region
    actually requested."""

    def _record(self, id_):
        return {
            "Id": id_, "Name": f"S1A_IW_OCN__2SDV_2026070{id_}T000000",
            "ContentDate_Start": "2026-07-02T00:00:00Z",
            "ContentDate_End": "2026-07-02T00:00:10Z",
            "ContentLength_GB": 1.0, "Online": True, "GeoFootprint": None,
        }

    def _cfg(self, min_lon, max_lon):
        return SimpleNamespace(
            geographic_bounds=SimpleNamespace(min_lon=min_lon, max_lon=max_lon, min_lat=-15.0, max_lat=30.0),
            temporal_bounds=SimpleNamespace(start="2026-07-02", end="2026-07-03"),
        )

    def _patch_client(self, monkeypatch, fake_client):
        from sar_validation.core import dry_collocation

        monkeypatch.setattr(dry_collocation, "authenticate_cdse", lambda: ("user", "pass"))
        monkeypatch.setattr(dry_collocation, "CopernicusODataClient", lambda user, pwd: fake_client)

    def test_crossing_bbox_splits_into_two_queries(self, monkeypatch):
        from sar_validation.core import dry_collocation

        fake_client = MagicMock()
        fake_client.query_products.side_effect = [[self._record("1")], [self._record("2")]]
        self._patch_client(monkeypatch, fake_client)

        records = dry_collocation._query_sentinel1_ocn_dry(self._cfg(135.0, -120.0))

        assert fake_client.query_products.call_count == 2
        first_kwargs = fake_client.query_products.call_args_list[0].kwargs
        second_kwargs = fake_client.query_products.call_args_list[1].kwargs
        assert (first_kwargs["min_lon"], first_kwargs["max_lon"]) == (135.0, 180.0)
        assert (second_kwargs["min_lon"], second_kwargs["max_lon"]) == (-180.0, -120.0)
        assert sorted(r["Id"] for r in records) == ["1", "2"]

    def test_dedupes_product_returned_by_both_windows(self, monkeypatch):
        from sar_validation.core import dry_collocation

        fake_client = MagicMock()
        dup = self._record("dup")
        fake_client.query_products.side_effect = [[dup], [dup]]
        self._patch_client(monkeypatch, fake_client)

        records = dry_collocation._query_sentinel1_ocn_dry(self._cfg(135.0, -120.0))

        assert len(records) == 1

    def test_non_crossing_bbox_queries_once(self, monkeypatch):
        from sar_validation.core import dry_collocation

        fake_client = MagicMock()
        fake_client.query_products.return_value = []
        self._patch_client(monkeypatch, fake_client)

        dry_collocation._query_sentinel1_ocn_dry(self._cfg(-20.0, 0.0))

        assert fake_client.query_products.call_count == 1
        kwargs = fake_client.query_products.call_args.kwargs
        assert (kwargs["min_lon"], kwargs["max_lon"]) == (-20.0, 0.0)


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

        cfg = SimpleNamespace(sar_data=SimpleNamespace(swath_mode=["IW", "EW"]))
        footprints = dry_collocation._discover_sentinel1_ocn_footprints_dry(cfg=cfg)

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
        turned into a (wrong) kind="polygon" footprint. swath_mode
        includes both "IW" and "WV" so the exclusion under test is the
        real per-record mode filter, not just an early return from
        swath_mode having no non-WV entries at all."""
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

        cfg = SimpleNamespace(sar_data=SimpleNamespace(swath_mode=["IW", "WV"]))
        footprints = dry_collocation._discover_sentinel1_ocn_footprints_dry(cfg=cfg)

        assert footprints == []

    def test_mode_not_in_recipes_swath_mode_is_excluded(self, monkeypatch):
        """A recipe whose own sar_data.swath_mode restricts to ["WV", "SM"]
        must never get IW/EW footprints predicted against it -- those are
        scenes the real (non-dry) download path would never touch,
        inflating every downstream validation-source prediction checked
        against them. An EW record must not survive when swath_mode is
        ["WV", "SM"]."""
        from sar_validation.core import dry_collocation

        fake_records = [
            {
                "Name": "S1A_EW_OCN__2SDH_20260801T060000_20260801T060030_000000_000000_0000.SAFE",
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

        cfg = SimpleNamespace(sar_data=SimpleNamespace(swath_mode=["WV", "SM"]))
        footprints = dry_collocation._discover_sentinel1_ocn_footprints_dry(cfg=cfg)

        assert footprints == []

    def test_mode_in_recipes_swath_mode_is_kept(self, monkeypatch):
        """Inverse of the regression test above -- an SM record must
        survive when swath_mode is ["WV", "SM"]."""
        from sar_validation.core import dry_collocation

        fake_records = [
            {
                "Name": "S1A_SM_OCN__2SDH_20260801T060000_20260801T060030_000000_000000_0000.SAFE",
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

        cfg = SimpleNamespace(sar_data=SimpleNamespace(swath_mode=["WV", "SM"]))
        footprints = dry_collocation._discover_sentinel1_ocn_footprints_dry(cfg=cfg)

        assert len(footprints) == 1
        assert footprints[0].source_file == fake_records[0]["Name"]

    def test_only_wv_in_swath_mode_skips_the_catalog_search_entirely(self, monkeypatch):
        """When swath_mode has no non-WV entries at all, there's nothing
        this function could ever return -- it must short-circuit before
        even querying the catalog, not just filter an empty result down
        to []."""
        from sar_validation.core import dry_collocation

        query_called = []
        monkeypatch.setattr(
            dry_collocation, "_query_sentinel1_ocn_dry", lambda cfg: query_called.append(1) or [],
        )

        cfg = SimpleNamespace(sar_data=SimpleNamespace(swath_mode=["WV"]))
        footprints = dry_collocation._discover_sentinel1_ocn_footprints_dry(cfg=cfg)

        assert footprints == []
        assert query_called == []

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
            sar_data = SimpleNamespace(swath_mode=["IW", "EW"])

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
            sar_data = SimpleNamespace(swath_mode=["WV"])

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
            sar_data = SimpleNamespace(swath_mode=["WV"])

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
            sar_data = SimpleNamespace(swath_mode=["WV"])

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

        cfg = SimpleNamespace(sar_data=SimpleNamespace(swath_mode=["WV"]))
        footprints = dry_collocation._discover_sentinel1_wv_footprints_dry(cfg=cfg)

        assert footprints == []

    def test_wv_not_in_swath_mode_skips_the_catalog_search_entirely(self, monkeypatch):
        """Regression test companion to
        TestDiscoverSentinel1OcnFootprintsDry's identical one: a recipe
        whose sar_data.swath_mode excludes "WV" must never get WV
        footprints predicted against it -- short-circuits before even
        querying the catalog."""
        from sar_validation.core import dry_collocation

        query_called = []
        monkeypatch.setattr(
            dry_collocation, "_query_sentinel1_ocn_dry", lambda cfg: query_called.append(1) or [],
        )

        cfg = SimpleNamespace(sar_data=SimpleNamespace(swath_mode=["IW", "EW", "SM"]))
        footprints = dry_collocation._discover_sentinel1_wv_footprints_dry(cfg=cfg)

        assert footprints == []
        assert query_called == []

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
            sar_data = SimpleNamespace(swath_mode=["WV"])

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
    registered satellites (1A/1B/1C) and taking the union of their
    matched windows' own earliest start and latest end -- one footprint
    per catalog record (i.e. per real daily mosaic file), matching how
    many actual SAR scenes a real run treats this source as having (see
    sar_sources.py's own "scenes are daily, ... continent-wide mosaics"
    comment), not one footprint per individual orbit pass. The
    from-downloaded path instead reads the real non-NaN pixel extent
    directly off an already-downloaded GeoTIFF, which is more accurate
    than propagation since the real data is already on disk."""

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
            # window here -- the union must produce one footprint for
            # this one catalog record, not one per satellite.
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

    def test_dry_discovery_produces_one_footprint_per_record_spanning_every_matched_window(self, monkeypatch):
        """A europe-wide bbox is typically crossed by Sentinel-1 many
        times a day across three satellites -- each distinct pass window
        must still collapse into exactly one footprint per catalog
        record (one real daily mosaic file, not one per orbit pass),
        spanning from the earliest matched window's start to the latest
        matched window's end."""
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

        assert len(footprints) == 1
        assert footprints[0].sensing_start == dt(2026, 8, 1, 5, 0)
        assert footprints[0].sensing_end == dt(2026, 8, 1, 17, 5)

    def test_dry_discovery_skips_a_record_with_no_matched_window_at_all(self, monkeypatch):
        """A record every candidate satellite genuinely never crosses
        (an empty list, not a fail-open whole-window match -- see
        orbit_overlap_windows' own docstring) produces no footprint:
        there would be no real SAR data in the requested area for that
        day to validate against."""
        from sar_validation.core import dry_collocation

        fake_records = [
            {
                "Name": "S1A_CLMS_SSM1km_20260801T000000.tif",
                "ContentDate_Start": "2026-08-01T00:00:00.000Z",
                "ContentDate_End": "2026-08-01T23:59:59.000Z",
            },
        ]
        monkeypatch.setattr(dry_collocation, "_query_clms_ssm_dry", lambda cfg: fake_records)
        monkeypatch.setattr(dry_collocation.orbit_coverage, "orbit_overlap_windows", lambda *a, **kw: [])

        class _FakeCfg:
            geographic_bounds = SimpleNamespace(min_lon=-10.0, max_lon=30.0, min_lat=35.0, max_lat=60.0)

        footprints = dry_collocation._discover_clms_ssm_footprints_dry(cfg=_FakeCfg())

        assert footprints == []

    def test_dry_discovery_produces_one_footprint_per_record_across_multiple_records(self, monkeypatch):
        """A 3-day recipe's 3 real catalog records (one daily mosaic
        each) must produce exactly 3 footprints, regardless of how many
        individual orbit passes each day's propagation matches --
        matching what a real run treats as this source's own scene
        count (see sar_sources.py)."""
        from sar_validation.core import dry_collocation

        fake_records = [
            {
                "Name": f"S1A_CLMS_SSM1km_2026080{day}T000000.tif",
                "ContentDate_Start": f"2026-08-0{day}T00:00:00.000Z",
                "ContentDate_End": f"2026-08-0{day}T23:59:59.000Z",
            }
            for day in (1, 2, 3)
        ]
        monkeypatch.setattr(dry_collocation, "_query_clms_ssm_dry", lambda cfg: fake_records)

        def _fake_windows(satellite, start, end, min_lon, max_lon, min_lat, max_lat, **kwargs):
            # Several distinct passes per satellite per day, mirroring
            # the real many-passes-per-day behavior this fix targets.
            return [
                (start + timedelta(hours=h), start + timedelta(hours=h, minutes=5))
                for h in (4, 9, 14, 19)
            ]

        monkeypatch.setattr(dry_collocation.orbit_coverage, "orbit_overlap_windows", _fake_windows)

        class _FakeCfg:
            geographic_bounds = SimpleNamespace(min_lon=-10.0, max_lon=30.0, min_lat=35.0, max_lat=60.0)

        footprints = dry_collocation._discover_clms_ssm_footprints_dry(cfg=_FakeCfg())

        assert len(footprints) == 3
        assert {fp.source_file for fp in footprints} == {r["Name"] for r in fake_records}

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
            [tmp_path / "a.tif"], _FakeSarSourceSpec(), "soil_moisture",
        )

        assert result == [fp]


class TestSarFootprintsFromDownloadedRemainingSources:
    """sentinel1_l2_ocn / nisar_sme2 / radarsat2 all share the same
    generic from-downloaded path -- SARSourceSpec.convert() has already
    normalized each source's native format into a (lon, lat, time)
    xarray.Dataset, so no per-source branching is needed here beyond
    detecting WV mode's "point" dimension (see collocation.py's own
    is_wv_mode check)."""

    def test_sentinel1_ocn_uses_converted_dataset_geometry(self, tmp_path):
        import numpy as np
        import xarray as xr

        from sar_validation.core import dry_collocation

        fake_ds = xr.Dataset(
            {"oswWindSpeed": (("y", "x"), [[5.0]])},
            coords={
                "lon": (("y", "x"), [[10.0]]), "lat": (("y", "x"), [[45.0]]),
                # A real SARSourceSpec.convert() always produces a real
                # numpy datetime64[ns] "time" coordinate (never a bare
                # Python datetime object) -- using np.datetime64 here
                # (rather than a plain datetime(...) literal, which
                # xarray stores as an object-dtype scalar with different
                # .values.item() behavior) is what actually exercises the
                # real conversion path below and would have caught the
                # "sensing_start/sensing_end silently become a raw int
                # nanosecond count" regression.
                "time": np.datetime64("2026-08-01T06:00:00", "ns"),
            },
        )

        class _FakeSarSourceSpec:
            key = "sentinel1_l2_ocn"

            @staticmethod
            def convert(path, product_type):
                assert product_type == "wind"
                return fake_ds

        result = dry_collocation.sar_footprints_from_downloaded(
            [tmp_path / "a.nc"], _FakeSarSourceSpec(), "wind",
        )

        assert len(result) == 1
        assert result[0].kind == "polygon"
        assert result[0].bbox == (10.0, 10.0, 45.0, 45.0)
        # isinstance, not just equality: a raw int nanosecond count would
        # fail this even though downstream code silently tolerates it
        # right up until a predicate does datetime arithmetic on it.
        assert isinstance(result[0].sensing_start, datetime)
        assert isinstance(result[0].sensing_end, datetime)
        assert result[0].sensing_start == datetime(2026, 8, 1, 6, 0, 0)
        assert result[0].sensing_end == datetime(2026, 8, 1, 6, 0, 0)
        assert result[0].source_file == str(tmp_path / "a.nc")

    def test_conversion_failure_is_skipped_not_raised(self, tmp_path):
        from sar_validation.core import dry_collocation

        class _FakeSarSourceSpec:
            key = "sentinel1_l2_ocn"

            @staticmethod
            def convert(path, product_type):
                raise RuntimeError("boom")

        result = dry_collocation.sar_footprints_from_downloaded(
            [tmp_path / "a.nc"], _FakeSarSourceSpec(), "wind",
        )

        assert result == []

    def test_none_dataset_is_skipped_not_raised(self, tmp_path):
        """convert() returning None (e.g. a currents run whose scene has
        no RVL data -- see _from_sar_l2_ocn_iw_safe) must be skipped like
        a caught exception, not treated as a footprint."""
        from sar_validation.core import dry_collocation

        class _FakeSarSourceSpec:
            key = "sentinel1_l2_ocn"

            @staticmethod
            def convert(path, product_type):
                return None

        result = dry_collocation.sar_footprints_from_downloaded(
            [tmp_path / "a.nc"], _FakeSarSourceSpec(), "currents",
        )

        assert result == []

    def test_wv_mode_produces_multi_point_footprint_spanning_its_time_range(self, tmp_path):
        """WV mode's converted Dataset carries one (lon, lat, time) triple
        per vignette along a "point" dimension (see
        DataTreeConverter.from_sar_l2_ocn_wv_safe) -- a real WV SAFE
        product bundles ~16 vignettes at DIFFERENT acquisition times, so
        ds["time"].values is a multi-element array, not a scalar.
        sensing_start/sensing_end must span that range rather than assume
        a single scalar time."""
        import numpy as np
        import xarray as xr

        from sar_validation.core import dry_collocation

        fake_ds = xr.Dataset(
            {"oswTotalHs": (["point"], [1.0, 2.0])},
            coords={
                "lon": (["point"], [10.0, 11.0]),
                "lat": (["point"], [45.0, 46.0]),
                "time": (["point"], np.array(
                    ["2026-08-01T06:00:00", "2026-08-01T06:05:00"], dtype="datetime64[ns]",
                )),
            },
        )

        class _FakeSarSourceSpec:
            key = "sentinel1_l2_ocn"

            @staticmethod
            def convert(path, product_type):
                return fake_ds

        result = dry_collocation.sar_footprints_from_downloaded(
            [tmp_path / "a.SAFE"], _FakeSarSourceSpec(), "waves",
        )

        assert len(result) == 1
        fp = result[0]
        assert fp.kind == "wv_points"
        assert fp.points == [(45.0, 10.0), (46.0, 11.0)]
        assert fp.bbox == (10.0, 11.0, 45.0, 46.0)
        assert fp.sensing_start == datetime(2026, 8, 1, 6, 0, 0)
        assert fp.sensing_end == datetime(2026, 8, 1, 6, 5, 0)

    def test_wv_mode_single_vignette_does_not_crash(self, tmp_path):
        """A single-vignette WV product's ds["time"].values is a
        1-element array -- must not be treated any differently from the
        multi-vignette case above."""
        import xarray as xr

        from sar_validation.core import dry_collocation

        fake_ds = xr.Dataset(
            {"oswTotalHs": (["point"], [1.0])},
            coords={
                "lon": (["point"], [10.0]),
                "lat": (["point"], [45.0]),
                "time": (["point"], [datetime(2026, 8, 1, 6, 0, 0)]),
            },
        )

        class _FakeSarSourceSpec:
            key = "sentinel1_l2_ocn"

            @staticmethod
            def convert(path, product_type):
                return fake_ds

        result = dry_collocation.sar_footprints_from_downloaded(
            [tmp_path / "a.SAFE"], _FakeSarSourceSpec(), "waves",
        )

        assert len(result) == 1
        assert result[0].kind == "wv_points"
        assert result[0].points == [(45.0, 10.0)]
        assert result[0].sensing_start == result[0].sensing_end == datetime(2026, 8, 1, 6, 0, 0)

    def test_missing_lon_lat_after_convert_is_skipped_not_raised(self, tmp_path):
        """A converted Dataset missing lon/lat (e.g. an unusual NISAR SME2
        or RADARSAT-2 file) raises KeyError from the geometry-extraction
        code that runs AFTER convert() -- that must be caught and skip
        just this one file, like a convert() failure itself, not
        propagate out of sar_footprints_from_downloaded (which would hit
        _collocation_predictions()'s blanket except Exception and disable
        gating for the ENTIRE run, not just this one bad file)."""
        import xarray as xr

        from sar_validation.core import dry_collocation

        bad_ds = xr.Dataset(
            {"sarSSM": (("y", "x"), [[0.2]])},
            coords={"time": datetime(2026, 8, 1, 12, 0, 0)},  # no lon/lat
        )
        good_ds = xr.Dataset(
            {"sarSSM": (("y", "x"), [[0.2]])},
            coords={
                "lon": (("y", "x"), [[20.0]]), "lat": (("y", "x"), [[50.0]]),
                "time": datetime(2026, 8, 1, 12, 0, 0),
            },
        )

        class _FakeSarSourceSpec:
            key = "nisar_sme2"

            @staticmethod
            def convert(path, product_type):
                return bad_ds if "bad" in str(path) else good_ds

        result = dry_collocation.sar_footprints_from_downloaded(
            [tmp_path / "bad.h5", tmp_path / "good.h5"], _FakeSarSourceSpec(), "soil_moisture",
        )

        assert len(result) == 1
        assert result[0].source_file == str(tmp_path / "good.h5")

    def test_non_wv_length_one_time_array_does_not_raise(self, tmp_path):
        """A non-WV source whose converted Dataset's "time" value is a
        length-1 array (rather than a true 0-d scalar) must not raise --
        pd.Timestamp(...) rejects an array, unlike
        pd.to_datetime(np.atleast_1d(...)) (mirroring orchestrator.py's
        own _compute_sar_scene_times, which handles this exact same
        defensively for the exact same converted files)."""
        import numpy as np
        import xarray as xr

        from sar_validation.core import dry_collocation

        fake_ds = xr.Dataset(
            {"sarSSM": (("y", "x"), [[0.2]])},
            coords={
                "lon": (("y", "x"), [[20.0]]), "lat": (("y", "x"), [[50.0]]),
                "time": ("time", np.array(["2026-08-01T12:00:00"], dtype="datetime64[ns]")),
            },
        )

        class _FakeSarSourceSpec:
            key = "radarsat2"

            @staticmethod
            def convert(path, product_type):
                return fake_ds

        result = dry_collocation.sar_footprints_from_downloaded(
            [tmp_path / "a.h5"], _FakeSarSourceSpec(), "wind",
        )

        assert len(result) == 1
        assert result[0].sensing_start == datetime(2026, 8, 1, 12, 0, 0)
        assert result[0].sensing_end == datetime(2026, 8, 1, 12, 0, 0)

    def test_nat_time_footprint_is_dropped_not_appended(self, tmp_path):
        """A converted Dataset whose time resolves to NaT (e.g. a NISAR
        SME2 granule with a missing/unparseable zeroDopplerStartTime --
        see datatree_converter.py) must NOT produce a footprint at all --
        an unfiltered NaT footprint can make _predict_global_composite's
        day-range loop produce zero days, a false "none-predicted"
        verdict that wrongly SKIPS a real download (the one fail-closed
        risk in this feature)."""
        import numpy as np
        import xarray as xr

        from sar_validation.core import dry_collocation

        nat_ds = xr.Dataset(
            {"sarSSM": (("y", "x"), [[0.2]])},
            coords={
                "lon": (("y", "x"), [[20.0]]), "lat": (("y", "x"), [[50.0]]),
                "time": np.datetime64("NaT", "ns"),
            },
        )
        good_ds = xr.Dataset(
            {"sarSSM": (("y", "x"), [[0.2]])},
            coords={
                "lon": (("y", "x"), [[21.0]]), "lat": (("y", "x"), [[51.0]]),
                "time": datetime(2026, 8, 1, 12, 0, 0),
            },
        )

        class _FakeSarSourceSpec:
            key = "nisar_sme2"

            @staticmethod
            def convert(path, product_type):
                return nat_ds if path.name == "nat_scene.h5" else good_ds

        result = dry_collocation.sar_footprints_from_downloaded(
            [tmp_path / "nat_scene.h5", tmp_path / "good.h5"], _FakeSarSourceSpec(), "soil_moisture",
        )

        assert len(result) == 1
        assert result[0].source_file == str(tmp_path / "good.h5")

    @pytest.mark.parametrize("key", ["nisar_sme2", "radarsat2"])
    def test_nisar_and_radarsat2_route_through_the_same_generic_grid_path(self, tmp_path, key):
        """nisar_sme2 and radarsat2's convert() callbacks both produce a
        (y, x)-gridded Dataset (see from_nisar_sme2 / from_radarsat2_wind)
        just like sentinel1_l2_ocn's non-WV path -- this was the
        NotImplementedError gap this task closes for those two sources."""
        import xarray as xr

        from sar_validation.core import dry_collocation

        fake_ds = xr.Dataset(
            {"sarSSM": (("y", "x"), [[0.2]])},
            coords={
                "lon": (("y", "x"), [[20.0]]), "lat": (("y", "x"), [[50.0]]),
                "time": datetime(2026, 8, 1, 12, 0, 0),
            },
        )

        class _FakeSarSourceSpec:
            pass

        _FakeSarSourceSpec.key = key
        _FakeSarSourceSpec.convert = staticmethod(lambda path, product_type: fake_ds)

        result = dry_collocation.sar_footprints_from_downloaded(
            [tmp_path / "a.h5"], _FakeSarSourceSpec(), "soil_moisture",
        )

        assert len(result) == 1
        assert result[0].kind == "polygon"
        assert result[0].bbox == (20.0, 20.0, 50.0, 50.0)


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

    def test_stop_on_first_match_forwarded_to_a_supporting_predicate(self, monkeypatch):
        """A predicate that declares stop_on_first_match (e.g. smap_ssm,
        catalog-precise-bucket) must actually receive True when
        predict_source is called with stop_on_first_match=True -- this is
        the real-run gating path's own entry point into the predicate
        dispatch (see DataOrchestrator._collocation_predictions)."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint, predict_source

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        seen_bboxes = []

        def _tracking_list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end):
            seen_bboxes.append((min_lon, max_lon, min_lat, max_lat))
            return [("fake_candidate", datetime(2026, 8, 1, 6, 0, 0), datetime(2026, 8, 1, 6, 3, 0))]

        monkeypatch.setattr(dry_collocation, "_smap_ssm_list_candidates_dry", _tracking_list_candidates_dry)

        footprint1 = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )
        footprint2 = SarFootprint(
            kind="polygon", bbox=(20.0, 30.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 2, 6, 0, 30), sensing_end=datetime(2026, 8, 2, 6, 1, 0),
            source_file="s2.SAFE",
        )

        result = predict_source(
            SimpleNamespace(source_type="smap_ssm"), cfg=object(),
            sar_footprints=[footprint1, footprint2], stop_on_first_match=True,
        )

        assert result.verdict == "collocated"
        assert seen_bboxes == [(-10.0, 10.0, 35.0, 55.0)]  # second footprint never queried

    def test_stop_on_first_match_ignored_by_a_non_supporting_predicate(self, monkeypatch):
        """A predicate that does not declare stop_on_first_match (e.g.
        ismn) must keep working exactly as before when predict_source is
        called with stop_on_first_match=True -- the flag is only ever
        forwarded to predicates that opted in (see
        _predicate_accepts_stop_on_first_match)."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import predict_source

        monkeypatch.setattr(dry_collocation, "_build_ismn_downloader", lambda cfg: MagicMock(
            station_date_ranges_dry=lambda *a, **k: None,
        ))

        result = predict_source(
            SimpleNamespace(source_type="ismn"), cfg=object(), sar_footprints=[], stop_on_first_match=True,
        )

        assert result.verdict == "unknown"
        assert result.source_type == "ismn"


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


def _patch_orbit_matching(monkeypatch, dry_collocation, fake_orbit_overlap_windows):
    """_predict_orbit_corridor_source's exhaustive (stop_on_first_match=
    False, the default) path calls orbit_coverage.sample_ground_track +
    match_ground_track instead of orbit_overlap_windows directly (see
    sample_ground_track's own docstring for why: propagation is shared
    across every footprint checked against the same satellite, matching
    is not). This patches both so a test's existing
    fake_orbit_overlap_windows(satellite, start, end, min_lon, max_lon,
    min_lat, max_lat, **kwargs) fake -- written against the old
    single-call contract -- keeps working unchanged: sample_ground_track
    returns an inert stub (never actually consulted by the fake matcher
    below), and match_ground_track delegates straight through to
    fake_orbit_overlap_windows with the same arguments it would have
    received directly."""
    monkeypatch.setattr(
        dry_collocation.orbit_coverage, "sample_ground_track",
        lambda satellite, start, end, *a, **k: [(start, 0.0, 0.0), (end, 0.0, 0.0)],
    )

    def _fake_match_ground_track(samples, satellite, start, end, min_lon, max_lon, min_lat, max_lat, **kwargs):
        return fake_orbit_overlap_windows(satellite, start, end, min_lon, max_lon, min_lat, max_lat, **kwargs)

    monkeypatch.setattr(dry_collocation.orbit_coverage, "match_ground_track", _fake_match_ground_track)


class TestRunOrbitMatchJobsConcurrency:
    """_run_orbit_match_jobs' sequential fallback (below
    _ORBIT_MATCH_PARALLEL_MIN_JOBS jobs) must not write
    samples_by_satellite/margin_km to the module-level globals
    _orbit_match_pool_initializer sets -- those are only safe for
    ProcessPoolExecutor workers (separate processes, separate memory).
    predict_collocation's own ThreadPoolExecutor can run several
    orbit-corridor sources (e.g. altimeter alongside scatterometer_hy2b/
    hy2c/oceansat3) concurrently on threads that share one process's
    memory -- a global write there would let one source's own call
    clobber another's samples/margin_km mid-flight, causing spurious
    "Prediction raised an exception" failures (a real KeyError from one
    thread reading a different thread's satellite key) that come and go
    between runs of the exact same recipe."""

    def _job(self, satellite, fp_idx, lat=0.0, lon=0.0):
        from datetime import datetime as dt

        start, end = dt(2026, 8, 1, 6, 0, 0), dt(2026, 8, 1, 6, 3, 0)
        return (satellite, start, end, (0.0, 0.0, 0.0, 0.0), None, (lat, lon), start, end, fp_idx)

    def test_sequential_fallback_never_touches_the_module_globals(self, monkeypatch):
        from sar_validation.core import dry_collocation

        monkeypatch.setattr(dry_collocation, "_orbit_match_samples", {"sentinel": "untouched"})
        monkeypatch.setattr(dry_collocation, "_orbit_match_margin_km", -1.0)
        monkeypatch.setattr(
            dry_collocation.orbit_coverage, "match_ground_track",
            lambda samples, satellite, start, end, *a, **k: [(start, end)],
        )

        jobs = [self._job("sat-a", fp_idx=0)]  # well under _ORBIT_MATCH_PARALLEL_MIN_JOBS
        dry_collocation._run_orbit_match_jobs(jobs, {"sat-a": [(None, 0.0, 0.0)]}, margin_km=5.0)

        assert dry_collocation._orbit_match_samples == {"sentinel": "untouched"}
        assert dry_collocation._orbit_match_margin_km == -1.0

    def test_concurrent_calls_with_different_satellites_do_not_cross_contaminate(self, monkeypatch):
        """Two _run_orbit_match_jobs calls for DIFFERENT satellites, run
        on separate threads and forced to overlap via a barrier (so
        thread B's own call is guaranteed to be executing while thread
        A's is still in flight), must each only ever see their own
        samples_by_satellite -- not raise, and not return the other's
        satellite's result."""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from sar_validation.core import dry_collocation

        barrier = threading.Barrier(2)

        def _fake_match_ground_track(samples, satellite, start, end, *a, **k):
            barrier.wait(timeout=5)  # force both threads to be mid-call simultaneously
            return [(start, end)]

        monkeypatch.setattr(dry_collocation.orbit_coverage, "match_ground_track", _fake_match_ground_track)

        def _run(satellite, fp_idx):
            jobs = [self._job(satellite, fp_idx=fp_idx)]
            return dry_collocation._run_orbit_match_jobs(
                jobs, {satellite: [(None, 0.0, 0.0)]}, margin_km=5.0,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(_run, "sat-a", 0)
            future_b = executor.submit(_run, "sat-b", 1)
            result_a = future_a.result(timeout=10)
            result_b = future_b.result(timeout=10)

        assert [fp for fp, _s, _e in result_a] == [0]
        assert [fp for fp, _s, _e in result_b] == [1]


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

        _patch_orbit_matching(monkeypatch, dry_collocation, _fake_orbit_overlap_windows)

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

        _patch_orbit_matching(monkeypatch, dry_collocation, _fake_orbit_overlap_windows)

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

        _patch_orbit_matching(monkeypatch, dry_collocation, _fake_orbit_overlap_windows)

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

    def test_wv_points_footprint_forwards_target_point_not_zero_width_bbox(self, monkeypatch):
        """A WV vignette is a genuine point, not an area -- match_ground_
        track's bbox/polygon containment sweep can only ever match a
        zero-width bbox on an exact floating-point coordinate equality,
        i.e. never in practice. _predict_orbit_corridor_source must pass
        the vignette's own (lat, lon) through as target_point (a real
        distance check), for both the fast (stop_on_first_match=True)
        and exhaustive paths."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)
        monkeypatch.setattr(
            dry_collocation.orbit_coverage, "sample_ground_track",
            lambda satellite, start, end, *a, **k: [(start, 0.0, 0.0), (end, 0.0, 0.0)],
        )

        seen_target_points = []

        def _fake_match_ground_track(samples, satellite, start, end, min_lon, max_lon, min_lat, max_lat, **kwargs):
            seen_target_points.append(kwargs.get("target_point"))
            return [(start, end)]

        monkeypatch.setattr(dry_collocation.orbit_coverage, "match_ground_track", _fake_match_ground_track)

        def _fake_list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end):
            return [("h103_fake.nc", datetime(2026, 8, 1, 6, 0, 0), datetime(2026, 8, 1, 6, 3, 0))]

        footprint = SarFootprint(
            kind="wv_points", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None,
            points=[(40.0, 0.0)],
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="wv.SAFE",
        )

        for stop_on_first_match in (True, False):
            seen_target_points.clear()
            result = dry_collocation._predict_orbit_corridor_source(
                source=object(), cfg=object(), sar_footprints=[footprint],
                satellite_resolver=lambda name: "metop-b", list_candidates_dry=_fake_list_candidates_dry,
                source_type="ascat_ssm", stop_on_first_match=stop_on_first_match,
            )
            assert seen_target_points == [(40.0, 0.0)]
            assert result.verdict == "collocated"

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

        _patch_orbit_matching(monkeypatch, dry_collocation, _fake_orbit_overlap_windows)

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

    def test_stop_on_first_match_skips_remaining_footprints(self, monkeypatch):
        """stop_on_first_match=True (the real-run gating path -- see
        DataOrchestrator._collocation_predictions) must stop probing as
        soon as one candidate's refined window overlaps -- the second
        footprint's own candidate listing must never even be requested."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        def _fake_orbit_overlap_windows(satellite, start, end, min_lon, max_lon, min_lat, max_lat, **kwargs):
            return [(start, end)]  # full overlap

        _patch_orbit_matching(monkeypatch, dry_collocation, _fake_orbit_overlap_windows)

        seen_bboxes = []

        def _tracking_list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end):
            seen_bboxes.append((min_lon, max_lon, min_lat, max_lat))
            return [("h103_fake.nc", datetime(2026, 8, 1, 6, 0, 0), datetime(2026, 8, 1, 6, 3, 0))]

        footprint1 = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )
        footprint2 = SarFootprint(
            kind="polygon", bbox=(20.0, 30.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 2, 6, 0, 30), sensing_end=datetime(2026, 8, 2, 6, 1, 0),
            source_file="s2.SAFE",
        )

        result = dry_collocation._predict_orbit_corridor_source(
            source=object(), cfg=object(), sar_footprints=[footprint1, footprint2],
            satellite_resolver=lambda name: "metop-b", list_candidates_dry=_tracking_list_candidates_dry,
            source_type="ascat_ssm", stop_on_first_match=True,
        )

        assert result.verdict == "collocated"
        assert seen_bboxes == [(-10.0, 10.0, 35.0, 55.0)]
        assert "stopped at first match" in result.detail
        assert "--dry-collocation-detail" in result.detail
        # The leading count is a lower bound, not an exact total, once
        # the loop stopped at the first confirmed match -- see
        # _count_prefix's own docstring.
        assert "at least 1 of 2 SAR footprint(s)" in result.detail

    def test_exhaustive_detail_text_has_no_stop_early_note(self, monkeypatch):
        """The exhaustive path's own count is already the real total, so
        its detail text must not carry the "stopped at first match" note
        stop_on_first_match=True gets, nor the "at least" qualifier on
        its own count."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        def _fake_orbit_overlap_windows(satellite, start, end, min_lon, max_lon, min_lat, max_lat, **kwargs):
            return [(start, end)]

        _patch_orbit_matching(monkeypatch, dry_collocation, _fake_orbit_overlap_windows)

        def _fake_list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end):
            return [("h103_fake.nc", datetime(2026, 8, 1, 6, 0, 0), datetime(2026, 8, 1, 6, 3, 0))]

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = dry_collocation._predict_orbit_corridor_source(
            source=object(), cfg=object(), sar_footprints=[footprint],
            satellite_resolver=lambda name: "metop-b", list_candidates_dry=_fake_list_candidates_dry,
            source_type="ascat_ssm", stop_on_first_match=False,
        )

        assert result.verdict == "collocated"
        assert "stopped at first match" not in result.detail
        assert "at least" not in result.detail
        assert "1 of 1 SAR footprint(s)" in result.detail

    def test_sample_ground_track_failure_fails_open_without_crashing(self, monkeypatch):
        """The exhaustive path's own fail-open branch (sample_ground_track
        raising for a satellite) must not crash on its own job-tuple
        unpacking -- each job carries (cand_start, cand_end, target_bbox,
        target_polygon, target_point, padded_start, padded_end), a
        7-tuple, not the 6-tuple an area-only target would have."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        def _failing_sample_ground_track(satellite, start, end, *a, **k):
            raise RuntimeError("propagation failed")

        monkeypatch.setattr(dry_collocation.orbit_coverage, "sample_ground_track", _failing_sample_ground_track)

        def _fake_list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end):
            return [("h103_fake.nc", datetime(2026, 8, 1, 6, 0, 0), datetime(2026, 8, 1, 6, 3, 0))]

        footprint = SarFootprint(
            kind="wv_points", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None,
            points=[(40.0, 0.0)],
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="wv.SAFE",
        )

        result = dry_collocation._predict_orbit_corridor_source(
            source=object(), cfg=object(), sar_footprints=[footprint],
            satellite_resolver=lambda name: "metop-b", list_candidates_dry=_fake_list_candidates_dry,
            source_type="ascat_ssm", stop_on_first_match=False,
        )

        assert result.verdict == "collocated"  # fails open -- whole candidate window kept

    def test_default_stop_on_first_match_false_stays_exhaustive(self, monkeypatch):
        """The --dry-collocation preview path relies on the default
        (False) to keep scanning every footprint, since matched_windows'
        length feeds the report's own "N matched window(s)" detail text."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        def _fake_orbit_overlap_windows(satellite, start, end, min_lon, max_lon, min_lat, max_lat, **kwargs):
            return [(start, end)]  # full overlap

        _patch_orbit_matching(monkeypatch, dry_collocation, _fake_orbit_overlap_windows)

        seen_bboxes = []

        def _tracking_list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end):
            seen_bboxes.append((min_lon, max_lon, min_lat, max_lat))
            return [("h103_fake.nc", datetime(2026, 8, 1, 6, 0, 0), datetime(2026, 8, 1, 6, 3, 0))]

        footprint1 = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )
        footprint2 = SarFootprint(
            kind="polygon", bbox=(20.0, 30.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 2, 6, 0, 30), sensing_end=datetime(2026, 8, 2, 6, 1, 0),
            source_file="s2.SAFE",
        )

        result = dry_collocation._predict_orbit_corridor_source(
            source=object(), cfg=object(), sar_footprints=[footprint1, footprint2],
            satellite_resolver=lambda name: "metop-b", list_candidates_dry=_tracking_list_candidates_dry,
            source_type="ascat_ssm",
        )

        assert result.verdict == "collocated"
        assert len(seen_bboxes) == 2

    def test_exhaustive_count_dedupes_by_footprint_not_by_vignette_point(self, monkeypatch):
        """One wv_points footprint with several vignette points that ALL
        match (one real satellite pass grazing several nearby vignettes
        in a row) must report as one matched footprint in its own count,
        even though matched_windows itself keeps every raw hit -- see
        _predict_orbit_corridor_source's own matched_footprint_indices
        comment for why per-vignette counting alone is misleading."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        def _fake_orbit_overlap_windows(satellite, start, end, min_lon, max_lon, min_lat, max_lat, **kwargs):
            return [(start, end)]  # full overlap

        _patch_orbit_matching(monkeypatch, dry_collocation, _fake_orbit_overlap_windows)

        def _fake_list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end):
            return [("h103_fake.nc", datetime(2026, 8, 1, 6, 0, 0), datetime(2026, 8, 1, 6, 3, 0))]

        footprint = SarFootprint(
            kind="wv_points", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None,
            points=[(40.0, 0.0), (41.0, 1.0), (42.0, 2.0)],
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="wv.SAFE",
        )

        result = dry_collocation._predict_orbit_corridor_source(
            source=object(), cfg=object(), sar_footprints=[footprint],
            satellite_resolver=lambda name: "metop-b", list_candidates_dry=_fake_list_candidates_dry,
            source_type="ascat_ssm",
        )

        assert result.verdict == "collocated"
        assert len(result.matched_windows or []) == 3  # every vignette's own hit still recorded
        assert "1 of 1 SAR footprint(s)" in result.detail
        assert "up to 3 candidate pass(es) total" in result.detail


class TestCatalogPrecisePredicate:
    """_predict_catalog_precise_source resolves its time tolerance via
    _resolve_temporal_padding_minutes(cfg, source_type) -- same
    convention as _predict_orbit_corridor_source (see
    TestOrbitCorridorPredicate) -- so every test here mocks that
    resolver rather than relying on a fake cfg shape."""

    def test_collocated_when_a_footprint_yields_any_candidate(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        def _fake_list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end):
            return [("granule_x", datetime(2026, 8, 1, 6, 0, 0), datetime(2026, 8, 1, 6, 5, 0))]

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = dry_collocation._predict_catalog_precise_source(
            source=object(), cfg=object(), sar_footprints=[footprint],
            list_candidates_dry=_fake_list_candidates_dry, source_type="scatterometer",
        )

        assert result.verdict == "collocated"
        assert result.bucket == "catalog-precise"

    def test_none_predicted_when_no_footprint_yields_any_candidate(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        def _fake_list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end):
            return []

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = dry_collocation._predict_catalog_precise_source(
            source=object(), cfg=object(), sar_footprints=[footprint],
            list_candidates_dry=_fake_list_candidates_dry, source_type="scatterometer",
        )

        assert result.verdict == "none-predicted"

    def test_wv_points_queried_individually_not_batched(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        call_count = {"n": 0}

        def _fake_list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end):
            call_count["n"] += 1
            return []

        footprint = SarFootprint(
            kind="wv_points", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None,
            points=[(40.0, 0.0), (41.0, 1.0)],
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="wv.SAFE",
        )

        dry_collocation._predict_catalog_precise_source(
            source=object(), cfg=object(), sar_footprints=[footprint],
            list_candidates_dry=_fake_list_candidates_dry, source_type="scatterometer",
        )

        assert call_count["n"] == 2  # one query per vignette

    def test_no_footprints_is_unknown(self):
        from sar_validation.core import dry_collocation

        result = dry_collocation._predict_catalog_precise_source(
            source=object(), cfg=object(), sar_footprints=[],
            list_candidates_dry=lambda *a, **k: [], source_type="scatterometer",
        )

        assert result.verdict == "unknown"

    def test_listing_failure_is_unknown_not_none_predicted(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        def _raising_list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end):
            raise RuntimeError("network error")

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = dry_collocation._predict_catalog_precise_source(
            source=object(), cfg=object(), sar_footprints=[footprint],
            list_candidates_dry=_raising_list_candidates_dry, source_type="scatterometer",
        )

        assert result.verdict == "unknown"

    def test_stop_on_first_match_skips_remaining_footprints(self, monkeypatch):
        """stop_on_first_match=True (the real-run gating path) must stop
        querying as soon as the first footprint yields a candidate -- the
        second footprint's own listing must never be requested."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        seen_bboxes = []

        def _tracking_list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end):
            seen_bboxes.append((min_lon, max_lon, min_lat, max_lat))
            return [("granule_x", datetime(2026, 8, 1, 6, 0, 0), datetime(2026, 8, 1, 6, 5, 0))]

        footprint1 = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )
        footprint2 = SarFootprint(
            kind="polygon", bbox=(20.0, 30.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 2, 6, 0, 30), sensing_end=datetime(2026, 8, 2, 6, 1, 0),
            source_file="s2.SAFE",
        )

        result = dry_collocation._predict_catalog_precise_source(
            source=object(), cfg=object(), sar_footprints=[footprint1, footprint2],
            list_candidates_dry=_tracking_list_candidates_dry, source_type="scatterometer",
            stop_on_first_match=True,
        )

        assert result.verdict == "collocated"
        assert seen_bboxes == [(-10.0, 10.0, 35.0, 55.0)]
        assert "stopped at first match" in result.detail
        assert "--dry-collocation-detail" in result.detail

    def test_default_stop_on_first_match_false_stays_exhaustive(self, monkeypatch):
        """The --dry-collocation preview path relies on the default
        (False) to keep querying every footprint, since matched_windows'
        length feeds the report's own "N candidate(s)" detail text."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        seen_bboxes = []

        def _tracking_list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end):
            seen_bboxes.append((min_lon, max_lon, min_lat, max_lat))
            return [("granule_x", datetime(2026, 8, 1, 6, 0, 0), datetime(2026, 8, 1, 6, 5, 0))]

        footprint1 = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )
        footprint2 = SarFootprint(
            kind="polygon", bbox=(20.0, 30.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 2, 6, 0, 30), sensing_end=datetime(2026, 8, 2, 6, 1, 0),
            source_file="s2.SAFE",
        )

        result = dry_collocation._predict_catalog_precise_source(
            source=object(), cfg=object(), sar_footprints=[footprint1, footprint2],
            list_candidates_dry=_tracking_list_candidates_dry, source_type="scatterometer",
        )

        assert result.verdict == "collocated"
        assert len(seen_bboxes) == 2

    def test_exhaustive_count_dedupes_by_footprint_not_by_vignette_point(self, monkeypatch):
        """Same rationale as _predict_orbit_corridor_source's identical
        test: one wv_points footprint with several vignette points that
        all get a candidate must report as one matched footprint, even
        though matched_windows keeps every raw candidate hit."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        def _fake_list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end):
            return [("granule_x", datetime(2026, 8, 1, 6, 0, 0), datetime(2026, 8, 1, 6, 5, 0))]

        footprint = SarFootprint(
            kind="wv_points", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None,
            points=[(40.0, 0.0), (41.0, 1.0), (42.0, 2.0)],
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="wv.SAFE",
        )

        result = dry_collocation._predict_catalog_precise_source(
            source=object(), cfg=object(), sar_footprints=[footprint],
            list_candidates_dry=_fake_list_candidates_dry, source_type="scatterometer",
        )

        assert result.verdict == "collocated"
        assert len(result.matched_windows or []) == 3  # every vignette's own hit still recorded
        assert "1 of 1 SAR footprint(s)" in result.detail
        assert "up to 3 candidate file(s) total" in result.detail


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

    def test_stop_on_first_match_skips_hsaf_branch_once_eumdac_confirms(self, monkeypatch):
        """When stop_on_first_match=True and the EUMDAC branch alone
        already confirms 'collocated', the H-SAF branch must not run at
        all -- a real-run gating caller only needs one confirmed hit."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        calls = []

        def _fake_predict_orbit_corridor_source(
            source, cfg, sar_footprints, *, satellite_resolver, list_candidates_dry, source_type,
            stop_on_first_match=False,
        ):
            calls.append((list_candidates_dry, stop_on_first_match))
            return dry_collocation.SourcePrediction(
                source_type=source_type, bucket="orbit-corridor", verdict="collocated", detail="ok",
            )

        monkeypatch.setattr(dry_collocation, "_predict_orbit_corridor_source", _fake_predict_orbit_corridor_source)

        # Recent enough to be eligible for BOTH the EUMDAC and H-SAF branches.
        now = datetime.now(timezone.utc)
        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=now - timedelta(days=1), sensing_end=now - timedelta(days=1) + timedelta(minutes=1),
            source_file="s1.SAFE",
        )
        monkeypatch.setattr(dry_collocation, "_ASCAT_COVERAGE_CUTOFF", now.date().isoformat())

        result = dry_collocation._predict_ascat_ssm(
            source=object(), cfg=object(), sar_footprints=[footprint], stop_on_first_match=True,
        )

        assert result.verdict == "collocated"
        assert calls == [(dry_collocation._eumdac_ascat_ssm_list_candidates_dry, True)]


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

        # Within the OSI-SAF FTP server's own rolling MAX_AGE_DAYS-day
        # retention window (relative to wall-clock now, not a fixed
        # historical date -- see _predict_scatterometer_ftp_source's own
        # staleness pre-check, which would otherwise short-circuit before
        # ever reaching list_candidates_dry).
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        candidate_start = now - timedelta(hours=1)
        candidate_end = now - timedelta(minutes=57)

        def _fake_list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end):
            return [("fake_candidate.nc", candidate_start, candidate_end)]

        monkeypatch.setattr(dry_collocation, list_candidates_dry_name, _fake_list_candidates_dry)

        def _fake_orbit_overlap_windows(satellite, start, end, min_lon, max_lon, min_lat, max_lat, **kwargs):
            return [(start, end)]  # full overlap

        _patch_orbit_matching(monkeypatch, dry_collocation, _fake_orbit_overlap_windows)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=candidate_start + timedelta(minutes=30), sensing_end=candidate_end + timedelta(minutes=30),
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

    @pytest.mark.parametrize(
        "source_type,list_candidates_dry_name",
        [
            pytest.param("scatterometer_hy2b", "_hy2b_list_candidates_dry", id="hy2b"),
            pytest.param("scatterometer_hy2c", "_hy2c_list_candidates_dry", id="hy2c"),
            pytest.param("scatterometer_oceansat3", "_oceansat3_list_candidates_dry", id="oceansat3"),
        ],
    )
    def test_recipe_older_than_retention_window_gets_a_specific_message(
        self, monkeypatch, source_type, list_candidates_dry_name,
    ):
        """A recipe whose every SAR footprint predates the OSI-SAF FTP
        server's own rolling MAX_AGE_DAYS-day retention must say so
        directly -- not the generic "no predicted overlap" a genuine
        geographic non-match would produce, which would incorrectly
        suggest the satellite's ground track was checked and never
        crossed the target area. list_candidates_dry must never even be
        called: there is nothing this predicate could learn from it."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint, predict_source

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        listing_called = []
        monkeypatch.setattr(
            dry_collocation, list_candidates_dry_name,
            lambda *a, **k: listing_called.append(1) or [],
        )

        old = datetime(2026, 1, 1, 0, 0, 0)  # far older than any real retention window
        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=old, sensing_end=old + timedelta(minutes=1),
            source_file="s1.SAFE",
        )

        result = predict_source(
            SimpleNamespace(source_type=source_type), cfg=object(), sar_footprints=[footprint],
        )

        assert result.verdict == "unknown"
        assert "download window" in result.detail
        assert "rolling" in result.detail
        assert listing_called == []


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

        _patch_orbit_matching(monkeypatch, dry_collocation, _fake_orbit_overlap_windows)

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


class TestPredictCatalogPreciseSources:
    """predict_source integration test per newly registered
    catalog-precise source_type -- exercises the real _PREDICATES
    dispatch, not _predict_catalog_precise_source directly."""

    @pytest.mark.parametrize(
        "source_type,list_candidates_dry_name",
        [
            pytest.param("scatterometer_ascat", "_scatterometer_list_candidates_dry", id="scatterometer_ascat"),
            pytest.param("smap_ssm", "_smap_ssm_list_candidates_dry", id="smap_ssm"),
        ],
    )
    def test_collocated_via_predict_source(self, monkeypatch, source_type, list_candidates_dry_name):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint, predict_source

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        def _fake_list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end):
            return [("fake_candidate", datetime(2026, 8, 1, 6, 0, 0), datetime(2026, 8, 1, 6, 3, 0))]

        monkeypatch.setattr(dry_collocation, list_candidates_dry_name, _fake_list_candidates_dry)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = predict_source(
            SimpleNamespace(source_type=source_type), cfg=object(), sar_footprints=[footprint],
        )

        assert result.verdict == "collocated"
        assert result.bucket == "catalog-precise"
        assert result.source_type == source_type

    @pytest.mark.parametrize("source_type", ["scatterometer_ascat", "smap_ssm"])
    def test_registered_under_own_source_type(self, source_type):
        from sar_validation.core import dry_collocation

        predicate = dry_collocation._PREDICATES[source_type]
        assert predicate is getattr(dry_collocation, f"_predict_{source_type}")


class TestPredictAltimeterTolerance:
    """Regression coverage for a bug where _predict_altimeter resolved its
    time tolerance via the bare "altimeter" source_type. That key has no
    DEFAULT_LAYER_TYPE_SPECS entry (only "altimeter_1hz"/"altimeter_5hz"
    do, both 180 minutes) -- so it silently fell back to the generic
    30-minute point_vs_layer default instead of the 180-minute tolerance
    orchestrator.py's real _download_altimeter uses. A real altimeter pass
    30-180 minutes from a SAR scene would then be missed by the predictor
    and wrongly reported "none-predicted". Unlike the rest of this
    module's tests, _resolve_temporal_padding_minutes is deliberately NOT
    mocked in test_padded_window_reflects_180_minutes_end_to_end below --
    it needs to exercise the real key lookup to catch this class of bug."""

    def test_resolves_tolerance_via_altimeter_1hz_and_5hz_keys(self, monkeypatch):
        """Pins the exact keys passed to _resolve_temporal_padding_minutes
        -- a flat mocked return value alone wouldn't catch a caller still
        passing the wrong (bare "altimeter") key."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        calls = []

        def _fake_resolve(cfg, *source_types):
            calls.append(source_types)
            return 90

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", _fake_resolve)
        monkeypatch.setattr(dry_collocation, "_altimeter_orbit_candidates_dry", lambda *a, **k: [])

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        dry_collocation._predict_altimeter(source=object(), cfg=object(), sar_footprints=[footprint])

        assert calls == [("altimeter_1hz", "altimeter_5hz")]

    def test_padded_window_reflects_180_minutes_end_to_end(self, monkeypatch):
        """Uses the REAL _resolve_temporal_padding_minutes (not mocked)
        against a realistically-shaped cfg (recipe.py's own CollocationType,
        whose defaults already carry DEFAULT_LAYER_TYPE_SPECS), and asserts
        the padded start/end actually passed to list_candidates_dry reflect
        180-minute padding -- pinning the bug's true end-to-end symptom,
        not just the numeric tolerance in isolation."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint
        from sar_validation.core.recipe import CollocationType

        cfg = SimpleNamespace(collocation=CollocationType(), validation_sources=[])

        captured = {}

        def _fake_list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end):
            captured["start"] = start
            captured["end"] = end
            return []

        monkeypatch.setattr(dry_collocation, "_altimeter_orbit_candidates_dry", _fake_list_candidates_dry)

        sensing_start = datetime(2026, 8, 1, 6, 0, 30)
        sensing_end = datetime(2026, 8, 1, 6, 1, 0)
        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=sensing_start, sensing_end=sensing_end, source_file="s1.SAFE",
        )

        dry_collocation._predict_altimeter(source=object(), cfg=cfg, sar_footprints=[footprint])

        expected_start = (sensing_start - timedelta(minutes=180)).isoformat()
        expected_end = (sensing_end + timedelta(minutes=180)).isoformat()
        assert captured["start"] == expected_start
        assert captured["end"] == expected_end

        # A narrower 30-minute padding must not match -- a stronger check
        # than only asserting equality with the correct value.
        wrong_start = (sensing_start - timedelta(minutes=30)).isoformat()
        assert captured["start"] != wrong_start


class TestAltimeterSatelliteResolver:
    def test_maps_every_satellite_code_altimeter_downloader_defines(self):
        """Every code in AltimeterDownloader.SATELLITES_1HZ must resolve
        to a real orbit_coverage.SATELLITE_ORBIT_SPECS key -- an
        unmapped code would silently fail open (whole candidate window
        kept, no real geographic refinement) for that mission."""
        from sar_validation.core import dry_collocation, orbit_coverage
        from sar_validation.downloaders.altimeter_downloader import SATELLITES_1HZ

        for sat_code in SATELLITES_1HZ:
            resolved = dry_collocation._altimeter_satellite_resolver(sat_code)
            assert resolved in orbit_coverage.SATELLITE_ORBIT_SPECS, (
                f"{sat_code!r} resolves to {resolved!r}, not a real orbit spec key"
            )

    def test_unknown_code_resolves_to_unknown_not_a_crash(self):
        from sar_validation.core import dry_collocation

        assert dry_collocation._altimeter_satellite_resolver("not_a_real_code") == "unknown"

    def test_hy2b_hy2c_resolve_to_their_own_altimeter_entry_not_the_scatterometer_one(self):
        """HaiYang-2B/2C carry two separate instruments -- a wide-swath
        HSCAT scatterometer (900km half-width, the plain "hy2b"/"hy2c"
        orbit_coverage.py entries, used by the scatterometer_hy2b/
        scatterometer_hy2c source types) and a narrow nadir altimeter
        (~8km, like every other altimeter mission). This predicate must
        resolve to the dedicated altimeter entry -- reusing the plain
        "hy2b"/"hy2c" keys here would silently give those two missions
        alone a ~1000km-wide search corridor instead of ~8km."""
        from sar_validation.core import dry_collocation, orbit_coverage

        assert dry_collocation._altimeter_satellite_resolver("h2b") == "hy2b-altimeter"
        assert dry_collocation._altimeter_satellite_resolver("h2c") == "hy2c-altimeter"

        for key in ("hy2b-altimeter", "hy2c-altimeter"):
            assert orbit_coverage.SATELLITE_ORBIT_SPECS[key].swath_half_width_km < 50.0, (
                f"{key} should model the narrow altimeter payload, not the scatterometer's wide swath"
            )

    def test_every_altimeter_mission_has_a_narrow_swath_not_a_wide_one(self):
        """Broader version of the hy2b/hy2c check above: every mission
        this predicate can resolve to should be a genuinely narrow,
        nadir-pointing instrument -- if a future satellite is added whose
        only existing orbit_coverage.py entry models a different,
        wider-swath payload, this catches it too."""
        from sar_validation.core import dry_collocation, orbit_coverage
        from sar_validation.downloaders.altimeter_downloader import SATELLITES_1HZ

        for sat_code in SATELLITES_1HZ:
            resolved = dry_collocation._altimeter_satellite_resolver(sat_code)
            spec = orbit_coverage.SATELLITE_ORBIT_SPECS[resolved]
            assert spec.swath_half_width_km < 50.0, (
                f"{sat_code!r} -> {resolved!r} has swath_half_width_km={spec.swath_half_width_km}, "
                f"too wide for a nadir altimeter -- likely reusing a different instrument's orbit spec"
            )


class TestAltimeterOrbitMarginKm:
    def test_predict_altimeter_passes_a_narrow_margin_not_the_wide_swath_default(self, monkeypatch):
        """_predict_orbit_corridor_source's own margin_km default (100km)
        is sized for a genuine wide-swath instrument's orbit/TLE
        uncertainty buffer. Applied unchanged to altimeter's ~8km nadir
        footprint, it over-predicts collocations by roughly an order of
        magnitude. _predict_altimeter must override it down to
        _ALTIMETER_ORBIT_MARGIN_KM."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)
        monkeypatch.setattr(
            dry_collocation, "_altimeter_orbit_candidates_dry",
            lambda *a, **k: [("j3", datetime(2026, 8, 1, 6, 0, 0), datetime(2026, 8, 1, 6, 3, 0))],
        )

        seen_margins = []

        def _fake_orbit_overlap_windows(satellite, start, end, min_lon, max_lon, min_lat, max_lat, **kwargs):
            seen_margins.append(kwargs.get("margin_km"))
            return []

        _patch_orbit_matching(monkeypatch, dry_collocation, _fake_orbit_overlap_windows)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        dry_collocation._predict_altimeter(source=object(), cfg=object(), sar_footprints=[footprint])

        assert seen_margins == [dry_collocation._ALTIMETER_ORBIT_MARGIN_KM]
        assert dry_collocation._ALTIMETER_ORBIT_MARGIN_KM < 100.0

    def test_other_orbit_corridor_sources_keep_the_shared_default_margin(self, monkeypatch):
        """ASCAT/HY-2/AMSR2/SMOS are genuine wide-swath instruments (250km
        to 900km half-width) -- they must NOT be narrowed down to
        altimeter's own margin; only altimeter overrides it."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        def _fake_list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end):
            return [("h103_fake.nc", datetime(2026, 8, 1, 6, 0, 0), datetime(2026, 8, 1, 6, 3, 0))]

        seen_margins = []

        def _fake_orbit_overlap_windows(satellite, start, end, min_lon, max_lon, min_lat, max_lat, **kwargs):
            seen_margins.append(kwargs.get("margin_km"))
            return []

        _patch_orbit_matching(monkeypatch, dry_collocation, _fake_orbit_overlap_windows)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        dry_collocation._predict_orbit_corridor_source(
            source=object(), cfg=object(), sar_footprints=[footprint],
            satellite_resolver=lambda name: "metop-b", list_candidates_dry=_fake_list_candidates_dry,
            source_type="ascat_ssm",
        )

        assert seen_margins == [100.0]


class TestAltimeterOrbitCandidatesDry:
    """_altimeter_orbit_candidates_dry is the network-free replacement for
    the old catalog-precise predicate's live copernicusmarine listing --
    see _predict_altimeter's own docstring for why."""

    def test_returns_one_candidate_per_in_scope_satellite_no_network(self):
        """Every mission in SATELLITES_1HZ whose AVAILABILITY_START is on
        or before the requested window's end, and whose AVAILABILITY_END
        (if any) is on or after the window's start, must appear exactly
        once -- this function must never touch the network (no mocking
        needed to prove it: a real copernicusmarine call would need
        auth/network and this test has neither). Window is chosen inside
        every mission's own active period (including h2c's, which ends
        2026-05-20), so this isn't conflated with AVAILABILITY_END's own
        exclusion, covered separately below."""
        from sar_validation.core.dry_collocation import _altimeter_orbit_candidates_dry
        from sar_validation.downloaders.altimeter_downloader import SATELLITES_1HZ

        candidates = _altimeter_orbit_candidates_dry(
            -10.0, 10.0, 35.0, 55.0, "2026-04-01T00:00:00", "2026-04-01T06:00:00",
        )

        names = [name for name, _s, _e in candidates]
        assert set(names) == set(SATELLITES_1HZ)
        assert len(names) == len(set(names))  # no duplicates

    def test_excludes_satellite_whose_availability_starts_after_the_window(self, monkeypatch):
        import sar_validation.downloaders.altimeter_downloader as altimeter_downloader
        from sar_validation.core import dry_collocation

        fake_availability = {"1hz": {**altimeter_downloader.AVAILABILITY_START["1hz"], "j3": "2030-01-01T00:00:00"}}
        monkeypatch.setattr(altimeter_downloader, "AVAILABILITY_START", fake_availability)

        candidates = dry_collocation._altimeter_orbit_candidates_dry(
            -10.0, 10.0, 35.0, 55.0, "2026-04-01T00:00:00", "2026-04-01T06:00:00",
        )

        names = [name for name, _s, _e in candidates]
        assert "j3" not in names
        assert "al" in names  # every other mission still in scope

    def test_excludes_satellite_whose_availability_ended_before_the_window(self):
        """A satellite that stopped producing data before the requested
        window (e.g. h2c, frozen since 2026-05-20 -- see
        AVAILABILITY_END) is still a real orbiting object SGP4 can
        predict a ground-track crossing for, but that crossing has no
        real observation behind it -- excluding it here is what prevents
        an unexplainable false positive no margin_km value could fix
        (the predicted crossing can be geometrically closer than a
        genuine match from a still-active mission)."""
        from sar_validation.core.dry_collocation import _altimeter_orbit_candidates_dry

        candidates = _altimeter_orbit_candidates_dry(
            -10.0, 10.0, 35.0, 55.0, "2026-07-01T00:00:00", "2026-07-01T06:00:00",
        )

        names = [name for name, _s, _e in candidates]
        assert "h2c" not in names
        assert "h2b" in names  # a still-active mission stays in scope

    def test_includes_satellite_whose_window_starts_before_its_availability_ended(self):
        """A window that only partially overlaps a satellite's active
        period (starts before its end date) must still include it -- only
        a window starting entirely after AVAILABILITY_END excludes it."""
        from sar_validation.core.dry_collocation import _altimeter_orbit_candidates_dry

        candidates = _altimeter_orbit_candidates_dry(
            -10.0, 10.0, 35.0, 55.0, "2026-05-15T00:00:00", "2026-05-25T00:00:00",
        )

        names = [name for name, _s, _e in candidates]
        assert "h2c" in names


class TestPredictAltimeterOrbitCorridor:
    """predict_source integration test for the altimeter source_type,
    mirroring TestPredictSmosSsm's pattern -- exercises the real
    _PREDICATES dispatch, not _predict_orbit_corridor_source directly.
    Altimeter uses the orbit-corridor bucket (real SGP4 propagation), not
    the catalog-precise bucket (which would need one network call per
    footprint per satellite)."""

    def test_registered_under_altimeter_source_type(self):
        from sar_validation.core import dry_collocation

        assert dry_collocation._PREDICATES["altimeter"] is dry_collocation._predict_altimeter

    def test_collocated_via_predict_source(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint, predict_source

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        candidate_start = datetime(2026, 8, 1, 6, 0, 0)
        candidate_end = datetime(2026, 8, 1, 6, 3, 0)

        def _fake_list_candidates_dry(min_lon, max_lon, min_lat, max_lat, start, end):
            return [("j3", candidate_start, candidate_end)]

        monkeypatch.setattr(dry_collocation, "_altimeter_orbit_candidates_dry", _fake_list_candidates_dry)

        def _fake_orbit_overlap_windows(satellite, start, end, min_lon, max_lon, min_lat, max_lat, **kwargs):
            return [(start, end)]  # full overlap

        _patch_orbit_matching(monkeypatch, dry_collocation, _fake_orbit_overlap_windows)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = predict_source(
            SimpleNamespace(source_type="altimeter"), cfg=object(), sar_footprints=[footprint],
        )

        assert result.verdict == "collocated"
        assert result.bucket == "orbit-corridor"
        assert result.source_type == "altimeter"

    def test_none_predicted_when_no_mission_orbit_crosses(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint, predict_source

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)
        monkeypatch.setattr(
            dry_collocation, "_altimeter_orbit_candidates_dry",
            lambda *a, **k: [("j3", datetime(2026, 8, 1, 6, 0, 0), datetime(2026, 8, 1, 6, 3, 0))],
        )
        _patch_orbit_matching(monkeypatch, dry_collocation, lambda *a, **k: [])

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = predict_source(
            SimpleNamespace(source_type="altimeter"), cfg=object(), sar_footprints=[footprint],
        )

        assert result.verdict == "none-predicted"
        assert result.bucket == "orbit-corridor"


class TestPredictAmsrSsm:
    """Combined predicate for source_type='amsr_ssm': NASA Earthdata/CMR
    (catalog-precise) checked first, JAXA G-Portal SFTP (orbit-corridor)
    a real second source. Unlike ascat_ssm there is no date-based
    eligibility split -- both branches apply to every footprint, every
    time (see _predict_amsr_ssm's own docstring)."""

    def test_collocated_when_only_earthdata_branch_finds_something(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        def _fake_catalog_precise(source, cfg, sar_footprints, *, list_candidates_dry, source_type):
            return dry_collocation.SourcePrediction(
                source_type=source_type, bucket="catalog-precise", verdict="collocated", detail="earthdata ok",
            )

        def _fake_orbit_corridor(
            source, cfg, sar_footprints, *, satellite_resolver, list_candidates_dry, source_type, **kwargs,
        ):
            return dry_collocation.SourcePrediction(
                source_type=source_type, bucket="orbit-corridor", verdict="none-predicted", detail="gportal empty",
            )

        monkeypatch.setattr(dry_collocation, "_predict_catalog_precise_source", _fake_catalog_precise)
        monkeypatch.setattr(dry_collocation, "_predict_orbit_corridor_source", _fake_orbit_corridor)
        monkeypatch.setattr(dry_collocation.orbit_coverage, "orbit_overlaps_bbox", lambda *a, **kw: True)
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = dry_collocation._predict_amsr_ssm(source=object(), cfg=object(), sar_footprints=[footprint])

        assert result.verdict == "collocated"
        assert result.bucket == "catalog-precise"

    def test_collocated_when_only_gportal_branch_finds_something(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        def _fake_catalog_precise(source, cfg, sar_footprints, *, list_candidates_dry, source_type):
            return dry_collocation.SourcePrediction(
                source_type=source_type, bucket="catalog-precise", verdict="none-predicted", detail="earthdata empty",
            )

        def _fake_orbit_corridor(
            source, cfg, sar_footprints, *, satellite_resolver, list_candidates_dry, source_type, **kwargs,
        ):
            return dry_collocation.SourcePrediction(
                source_type=source_type, bucket="orbit-corridor", verdict="collocated", detail="gportal ok",
            )

        monkeypatch.setattr(dry_collocation, "_predict_catalog_precise_source", _fake_catalog_precise)
        monkeypatch.setattr(dry_collocation, "_predict_orbit_corridor_source", _fake_orbit_corridor)
        monkeypatch.setattr(dry_collocation.orbit_coverage, "orbit_overlaps_bbox", lambda *a, **kw: True)
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = dry_collocation._predict_amsr_ssm(source=object(), cfg=object(), sar_footprints=[footprint])

        assert result.verdict == "collocated"

    def test_only_the_matching_branch_contributes_to_the_collocated_detail_text(self, monkeypatch):
        """Regression test for a real usability bug: when only one branch
        found something, Earthdata's own "no candidates found" detail
        must not be joined in next to G-Portal's "collocated" detail --
        it reads as contradictory even though the overall verdict is
        unambiguous."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        def _fake_catalog_precise(source, cfg, sar_footprints, *, list_candidates_dry, source_type):
            return dry_collocation.SourcePrediction(
                source_type=source_type, bucket="catalog-precise", verdict="none-predicted",
                detail="No candidates found across 43 SAR footprint(s).",
            )

        def _fake_orbit_corridor(
            source, cfg, sar_footprints, *, satellite_resolver, list_candidates_dry, source_type, **kwargs,
        ):
            return dry_collocation.SourcePrediction(
                source_type=source_type, bucket="orbit-corridor", verdict="collocated",
                detail="1 of 43 SAR footprint(s) with a predicted overlap.",
            )

        monkeypatch.setattr(dry_collocation, "_predict_catalog_precise_source", _fake_catalog_precise)
        monkeypatch.setattr(dry_collocation, "_predict_orbit_corridor_source", _fake_orbit_corridor)
        monkeypatch.setattr(dry_collocation.orbit_coverage, "orbit_overlaps_bbox", lambda *a, **kw: True)
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = dry_collocation._predict_amsr_ssm(source=object(), cfg=object(), sar_footprints=[footprint])

        assert result.verdict == "collocated"
        assert result.detail == "1 of 43 SAR footprint(s) with a predicted overlap."
        assert "No candidates found" not in result.detail

    def test_collocated_when_both_branches_find_something(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        def _fake_catalog_precise(source, cfg, sar_footprints, *, list_candidates_dry, source_type):
            return dry_collocation.SourcePrediction(
                source_type=source_type, bucket="catalog-precise", verdict="collocated", detail="earthdata ok",
            )

        def _fake_orbit_corridor(
            source, cfg, sar_footprints, *, satellite_resolver, list_candidates_dry, source_type, **kwargs,
        ):
            return dry_collocation.SourcePrediction(
                source_type=source_type, bucket="orbit-corridor", verdict="collocated", detail="gportal ok",
            )

        monkeypatch.setattr(dry_collocation, "_predict_catalog_precise_source", _fake_catalog_precise)
        monkeypatch.setattr(dry_collocation, "_predict_orbit_corridor_source", _fake_orbit_corridor)
        monkeypatch.setattr(dry_collocation.orbit_coverage, "orbit_overlaps_bbox", lambda *a, **kw: True)
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = dry_collocation._predict_amsr_ssm(source=object(), cfg=object(), sar_footprints=[footprint])

        assert result.verdict == "collocated"

    def test_none_predicted_only_when_both_branches_find_nothing(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        def _fake_catalog_precise(source, cfg, sar_footprints, *, list_candidates_dry, source_type):
            return dry_collocation.SourcePrediction(
                source_type=source_type, bucket="catalog-precise", verdict="none-predicted", detail="earthdata empty",
            )

        def _fake_orbit_corridor(
            source, cfg, sar_footprints, *, satellite_resolver, list_candidates_dry, source_type, **kwargs,
        ):
            return dry_collocation.SourcePrediction(
                source_type=source_type, bucket="orbit-corridor", verdict="none-predicted", detail="gportal empty",
            )

        monkeypatch.setattr(dry_collocation, "_predict_catalog_precise_source", _fake_catalog_precise)
        monkeypatch.setattr(dry_collocation, "_predict_orbit_corridor_source", _fake_orbit_corridor)
        monkeypatch.setattr(dry_collocation.orbit_coverage, "orbit_overlaps_bbox", lambda *a, **kw: True)
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = dry_collocation._predict_amsr_ssm(source=object(), cfg=object(), sar_footprints=[footprint])

        assert result.verdict == "none-predicted"

    def test_unknown_propagates_even_if_other_branch_is_none_predicted(self, monkeypatch):
        """An 'unknown' verdict from either branch must never be silently
        downgraded to 'none-predicted' just because the other branch
        answered confidently -- fail toward inclusion."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        def _fake_catalog_precise(source, cfg, sar_footprints, *, list_candidates_dry, source_type):
            return dry_collocation.SourcePrediction(
                source_type=source_type, bucket="catalog-precise", verdict="unknown", detail="earthdata listing failed",
            )

        def _fake_orbit_corridor(
            source, cfg, sar_footprints, *, satellite_resolver, list_candidates_dry, source_type, **kwargs,
        ):
            return dry_collocation.SourcePrediction(
                source_type=source_type, bucket="orbit-corridor", verdict="none-predicted", detail="gportal empty",
            )

        monkeypatch.setattr(dry_collocation, "_predict_catalog_precise_source", _fake_catalog_precise)
        monkeypatch.setattr(dry_collocation, "_predict_orbit_corridor_source", _fake_orbit_corridor)
        monkeypatch.setattr(dry_collocation.orbit_coverage, "orbit_overlaps_bbox", lambda *a, **kw: True)
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = dry_collocation._predict_amsr_ssm(source=object(), cfg=object(), sar_footprints=[footprint])

        assert result.verdict == "unknown"

    def test_registered_under_amsr_ssm_source_type(self):
        from sar_validation.core import dry_collocation

        assert dry_collocation._PREDICATES["amsr_ssm"] is dry_collocation._predict_amsr_ssm

    def test_stop_on_first_match_skips_gportal_branch_once_earthdata_confirms(self, monkeypatch):
        """When stop_on_first_match=True and the Earthdata branch alone
        already confirms 'collocated', the G-Portal (orbit-corridor)
        branch must not run at all."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        gportal_called = []

        def _fake_catalog_precise(source, cfg, sar_footprints, *, list_candidates_dry, source_type, **kwargs):
            return dry_collocation.SourcePrediction(
                source_type=source_type, bucket="catalog-precise", verdict="collocated", detail="earthdata ok",
            )

        def _fake_orbit_corridor(
            source, cfg, sar_footprints, *, satellite_resolver, list_candidates_dry, source_type, **kwargs,
        ):
            gportal_called.append(True)
            return dry_collocation.SourcePrediction(
                source_type=source_type, bucket="orbit-corridor", verdict="collocated", detail="gportal ok",
            )

        monkeypatch.setattr(dry_collocation, "_predict_catalog_precise_source", _fake_catalog_precise)
        monkeypatch.setattr(dry_collocation, "_predict_orbit_corridor_source", _fake_orbit_corridor)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = dry_collocation._predict_amsr_ssm(
            source=object(), cfg=object(), sar_footprints=[footprint], stop_on_first_match=True,
        )

        assert result.verdict == "collocated"
        assert gportal_called == []

    def test_gportal_branch_skipped_entirely_when_orbit_predicts_no_overlap(self, monkeypatch):
        """G-Portal's own SFTP directory-tree walk is real, sequential
        network I/O -- a footprint GCOM-W1's orbit provably never passes
        over must never even reach it, regardless of what Earthdata
        found."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        gportal_called = []

        def _fake_catalog_precise(source, cfg, sar_footprints, *, list_candidates_dry, source_type):
            return dry_collocation.SourcePrediction(
                source_type=source_type, bucket="catalog-precise", verdict="none-predicted", detail="earthdata empty",
            )

        def _fake_orbit_corridor(
            source, cfg, sar_footprints, *, satellite_resolver, list_candidates_dry, source_type, **kwargs,
        ):
            gportal_called.append(True)
            return dry_collocation.SourcePrediction(
                source_type=source_type, bucket="orbit-corridor", verdict="collocated", detail="gportal ok",
            )

        monkeypatch.setattr(dry_collocation, "_predict_catalog_precise_source", _fake_catalog_precise)
        monkeypatch.setattr(dry_collocation, "_predict_orbit_corridor_source", _fake_orbit_corridor)
        monkeypatch.setattr(dry_collocation.orbit_coverage, "orbit_overlaps_bbox", lambda *a, **kw: False)
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = dry_collocation._predict_amsr_ssm(source=object(), cfg=object(), sar_footprints=[footprint])

        assert result.verdict == "none-predicted"
        assert gportal_called == []

    def test_gportal_branch_respects_stop_on_first_match_like_earthdatas(self, monkeypatch):
        """The G-Portal branch is forwarded the caller's own
        stop_on_first_match exactly like Earthdata's branch, once orbit
        predicts an overlap -- --dry-collocation-detail
        (stop_on_first_match=False) gets G-Portal's real, exhaustive
        per-footprint count, not a forced early stop."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        received_kwargs = []

        def _fake_catalog_precise(source, cfg, sar_footprints, *, list_candidates_dry, source_type, **kwargs):
            return dry_collocation.SourcePrediction(
                source_type=source_type, bucket="catalog-precise", verdict="none-predicted", detail="earthdata empty",
            )

        def _fake_orbit_corridor(
            source, cfg, sar_footprints, *, satellite_resolver, list_candidates_dry, source_type, **kwargs,
        ):
            received_kwargs.append(kwargs)
            return dry_collocation.SourcePrediction(
                source_type=source_type, bucket="orbit-corridor", verdict="collocated", detail="gportal ok",
            )

        monkeypatch.setattr(dry_collocation, "_predict_catalog_precise_source", _fake_catalog_precise)
        monkeypatch.setattr(dry_collocation, "_predict_orbit_corridor_source", _fake_orbit_corridor)
        monkeypatch.setattr(dry_collocation.orbit_coverage, "orbit_overlaps_bbox", lambda *a, **kw: True)
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        exhaustive_result = dry_collocation._predict_amsr_ssm(
            source=object(), cfg=object(), sar_footprints=[footprint], stop_on_first_match=False,
        )
        fast_result = dry_collocation._predict_amsr_ssm(
            source=object(), cfg=object(), sar_footprints=[footprint], stop_on_first_match=True,
        )

        assert exhaustive_result.verdict == "collocated"
        assert fast_result.verdict == "collocated"
        # The exhaustive call omits stop_on_first_match entirely (relying
        # on _predict_orbit_corridor_source's own False default), mirroring
        # Earthdata's identical branching just above.
        assert received_kwargs == [{}, {"stop_on_first_match": True}]


class TestEarthdataAmsrSsmListCandidatesDryDatasetSelection:
    """_earthdata_amsr_ssm_list_candidates_dry must pick the same CMR
    dataset orchestrator.py's _download_amsr_ssm would pick for this
    call's own window end -- NSIDC-0451 on/before the cutoff, AU_Land
    after."""

    def test_picks_nsidc_0451_on_or_before_cutoff(self, monkeypatch):
        from sar_validation.core import dry_collocation

        seen_datasets = []

        class _FakeDownloader:
            def __init__(self, dataset, output_dir):
                seen_datasets.append(dataset)

            def list_candidates_dry(self, *a, **k):
                return []

        monkeypatch.setattr(dry_collocation, "EarthdataSoilMoistureDownloader", _FakeDownloader)

        dry_collocation._earthdata_amsr_ssm_list_candidates_dry(
            -10.0, 10.0, 35.0, 55.0, "2023-01-01T00:00:00", "2023-12-31T06:00:00",
        )

        assert seen_datasets == ["NSIDC-0451"]

    def test_picks_au_land_after_cutoff(self, monkeypatch):
        from sar_validation.core import dry_collocation

        seen_datasets = []

        class _FakeDownloader:
            def __init__(self, dataset, output_dir):
                seen_datasets.append(dataset)

            def list_candidates_dry(self, *a, **k):
                return []

        monkeypatch.setattr(dry_collocation, "EarthdataSoilMoistureDownloader", _FakeDownloader)

        dry_collocation._earthdata_amsr_ssm_list_candidates_dry(
            -10.0, 10.0, 35.0, 55.0, "2026-01-01T00:00:00", "2026-08-01T06:00:00",
        )

        assert seen_datasets == ["AU_Land"]


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


class TestPredictIsmn:
    """_predict_ismn resolves its time tolerance via
    _resolve_temporal_padding_minutes(cfg, "ismn") -- the real
    tolerance-resolution function orchestrator.py uses -- not a flat
    cfg.time_tolerance_minutes attribute, so every test here mocks that
    resolver rather than relying on a fake cfg shape. Tests that reach
    _point_in_footprint also need a real cfg.collocation.sar_footprint_radius_km,
    supplied via recipe.py's own CollocationType."""

    def test_unknown_when_no_archive_present(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        class _FakeIsmnDownloader:
            def station_date_ranges_dry(self, *a, **k):
                return None

        monkeypatch.setattr(dry_collocation, "_build_ismn_downloader", lambda cfg: _FakeIsmnDownloader())
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 0), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = dry_collocation._predict_ismn(source=object(), cfg=object(), sar_footprints=[footprint])

        assert result.verdict == "unknown"
        assert "ismn.earth" in result.message

    def test_collocated_when_a_station_range_covers_a_footprint_window(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint
        from sar_validation.core.recipe import CollocationType

        class _FakeIsmnDownloader:
            def station_date_ranges_dry(self, *a, **k):
                return {"Net_Stn": (45.0, 0.0, datetime(2026, 7, 1), datetime(2026, 9, 1))}

        monkeypatch.setattr(dry_collocation, "_build_ismn_downloader", lambda cfg: _FakeIsmnDownloader())
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 0), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        cfg = SimpleNamespace(collocation=CollocationType())
        result = dry_collocation._predict_ismn(source=object(), cfg=cfg, sar_footprints=[footprint])

        assert result.verdict == "collocated"
        assert "Net_Stn" in result.matched_stations

    def test_station_inside_bbox_but_outside_real_polygon_is_not_matched(self, monkeypatch):
        """The two-tier precision this task adds: station_date_ranges_dry's
        own bbox argument is only the coarse pre-filter (it can't be
        anything finer -- it doesn't know about footprint shape at all).
        _predict_ismn must apply _point_in_footprint as the fine
        refinement, so a station inside the footprint's bbox but outside
        its true (smaller) polygon must NOT count as collocated."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint
        from sar_validation.core.recipe import CollocationType

        class _FakeIsmnDownloader:
            def station_date_ranges_dry(self, *a, **k):
                # (55, 5) sits inside the footprint's bbox (-10..10, 35..55)
                # but outside the triangular polygon below.
                return {"Outside_Tri": (55.0, 5.0, datetime(2026, 7, 1), datetime(2026, 9, 1))}

        monkeypatch.setattr(dry_collocation, "_build_ismn_downloader", lambda cfg: _FakeIsmnDownloader())
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0),
            # A right triangle spanning only the lower-left half of the bbox.
            polygon=[(35.0, -10.0), (35.0, 10.0), (55.0, -10.0)],
            points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 0), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        cfg = SimpleNamespace(collocation=CollocationType())
        result = dry_collocation._predict_ismn(source=object(), cfg=cfg, sar_footprints=[footprint])

        assert result.verdict == "none-predicted"

    def test_unknown_when_no_sar_footprints_supplied(self, monkeypatch):
        from sar_validation.core import dry_collocation

        result = dry_collocation._predict_ismn(source=object(), cfg=object(), sar_footprints=[])

        assert result.verdict == "unknown"

    def test_collocated_when_footprint_is_tz_aware_and_station_range_is_naive(self, monkeypatch):
        """Regression test: a real SarFootprint's sensing_start/sensing_end
        is tz-aware (every real discover_sar_footprints_dry construction
        site uses datetime.fromisoformat(...replace("Z", "+00:00"))), while
        a real ISMNDownloader.station_date_ranges_dry's (lat, lon, earliest,
        latest) tuples are naive (parsed via bare datetime.strptime on each
        .stm file's own timestamp column). Before _windows_overlap routed
        through _to_naive_utc, comparing an aware footprint window against
        a naive station range raised "can't compare offset-naive and
        offset-aware datetimes" and crashed predict_source outright --
        found via a live sanity check against a real local ISMN archive."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint
        from sar_validation.core.recipe import CollocationType

        class _FakeIsmnDownloader:
            def station_date_ranges_dry(self, *a, **k):
                return {"Net_Stn": (45.0, 0.0, datetime(2026, 7, 1), datetime(2026, 9, 1))}

        monkeypatch.setattr(dry_collocation, "_build_ismn_downloader", lambda cfg: _FakeIsmnDownloader())
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 0, tzinfo=timezone.utc),
            sensing_end=datetime(2026, 8, 1, 6, 1, 0, tzinfo=timezone.utc),
            source_file="s1.SAFE",
        )

        cfg = SimpleNamespace(collocation=CollocationType())
        result = dry_collocation._predict_ismn(source=object(), cfg=cfg, sar_footprints=[footprint])

        assert result.verdict == "collocated"
        assert "Net_Stn" in result.matched_stations

    def test_station_date_ranges_dry_called_once_not_per_footprint(self, monkeypatch):
        """Performance regression test: station_date_ranges_dry rescans
        the real ISMN archive's .stm files from scratch on every call
        (confirmed non-trivial I/O against a real archive) -- with
        multiple footprints, it must be called exactly once (against the
        union bbox), not once per footprint."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint
        from sar_validation.core.recipe import CollocationType

        calls = []

        class _FakeIsmnDownloader:
            def station_date_ranges_dry(self, min_lon, max_lon, min_lat, max_lat, *a, **k):
                calls.append((min_lon, max_lon, min_lat, max_lat))
                return {"Net_Stn": (45.0, 0.0, datetime(2026, 7, 1), datetime(2026, 9, 1))}

        monkeypatch.setattr(dry_collocation, "_build_ismn_downloader", lambda cfg: _FakeIsmnDownloader())
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        footprints = [
            SarFootprint(
                kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
                sensing_start=datetime(2026, 8, 1, 6, 0, 0), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
                source_file="s1.SAFE",
            ),
            SarFootprint(
                kind="polygon", bbox=(20.0, 30.0, 60.0, 70.0), polygon=None, points=None,
                sensing_start=datetime(2026, 8, 1, 7, 0, 0), sensing_end=datetime(2026, 8, 1, 7, 1, 0),
                source_file="s2.SAFE",
            ),
            SarFootprint(
                kind="polygon", bbox=(-30.0, -20.0, 10.0, 20.0), polygon=None, points=None,
                sensing_start=datetime(2026, 8, 1, 8, 0, 0), sensing_end=datetime(2026, 8, 1, 8, 1, 0),
                source_file="s3.SAFE",
            ),
        ]

        cfg = SimpleNamespace(collocation=CollocationType())
        dry_collocation._predict_ismn(source=object(), cfg=cfg, sar_footprints=footprints)

        assert len(calls) == 1
        # The bounding envelope over all three footprints' own bboxes.
        assert calls[0] == (-30.0, 30.0, 10.0, 70.0)

    def test_union_bbox_still_matches_station_only_in_one_corner_footprint(self, monkeypatch):
        """Two footprints in different corners of a region; a station only
        overlaps one of them. Even though the single archive scan now
        covers the union bbox (a superset of either footprint's own
        bbox), the per-footprint _point_in_footprint refinement must
        still correctly attribute the station's match to only the
        footprint it actually falls inside."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint
        from sar_validation.core.recipe import CollocationType

        class _FakeIsmnDownloader:
            def station_date_ranges_dry(self, *a, **k):
                # Sits inside the first (south-west) footprint's bbox only.
                return {"SW_Stn": (5.0, -5.0, datetime(2026, 7, 1), datetime(2026, 9, 1))}

        monkeypatch.setattr(dry_collocation, "_build_ismn_downloader", lambda cfg: _FakeIsmnDownloader())
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *source_types: 90)

        sw_footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 0.0, 0.0, 10.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 0), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="sw.SAFE",
        )
        ne_footprint = SarFootprint(
            kind="polygon", bbox=(40.0, 50.0, 40.0, 50.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 7, 0, 0), sensing_end=datetime(2026, 8, 1, 7, 1, 0),
            source_file="ne.SAFE",
        )

        cfg = SimpleNamespace(collocation=CollocationType())
        result = dry_collocation._predict_ismn(
            source=object(), cfg=cfg, sar_footprints=[sw_footprint, ne_footprint],
        )

        assert result.verdict == "collocated"
        assert result.matched_stations == ["SW_Stn"]


class TestToNaiveUtc:
    def test_aware_datetime_converted_to_naive_utc(self):
        from sar_validation.core import dry_collocation

        aware = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = dry_collocation._to_naive_utc(aware)

        assert result == datetime(2026, 1, 1, 12, 0, 0)
        assert result.tzinfo is None

    def test_naive_datetime_passes_through_unchanged(self):
        from sar_validation.core import dry_collocation

        naive = datetime(2026, 1, 1, 12, 0, 0)
        assert dry_collocation._to_naive_utc(naive) == naive

    def test_non_utc_aware_datetime_is_converted_not_just_stripped(self):
        from sar_validation.core import dry_collocation

        aware = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone(timedelta(hours=5)))
        result = dry_collocation._to_naive_utc(aware)

        assert result == datetime(2026, 1, 1, 7, 0, 0)


class TestPredictInsitu:
    """_predict_insitu is registered under all five real
    orchestrator._INSITU_TYPES keys (mooring, buoy, drifter, ferrybox,
    tidal_gauge) -- there is no source_type="insitu" anywhere in this
    codebase. predict_source is called once per individual validation
    source, so the predicate must filter its own station_ranges_dry
    call down to just [source.source_type], never the full five-type
    batch InSituDownloader.download() itself accepts.

    Uses real per-station coordinates (station_ranges_dry) refined via
    _point_in_footprint, not the boolean check_availability_dry -- see
    station_ranges_dry's own docstring for why a bbox-only check (a
    wv_points footprint's bbox spans the bounding envelope of dozens of
    scattered vignette points) over-matches."""

    def _cfg(self):
        return SimpleNamespace(
            variable="waves", collocation=SimpleNamespace(sar_footprint_radius_km=14.0),
        )

    def _footprint(self):
        from sar_validation.core.dry_collocation import SarFootprint

        return SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 0), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

    def _footprint_at(self, lon):
        """A narrow-bbox footprint centered on lon, distinguishable from
        its siblings by location -- for tests needing several footprints
        that only some real stations fall within."""
        from sar_validation.core.dry_collocation import SarFootprint

        return SarFootprint(
            kind="polygon", bbox=(lon - 0.5, lon + 0.5, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 30), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

    @pytest.mark.parametrize(
        "source_type", ["mooring", "buoy", "drifter", "ferrybox", "tidal_gauge"],
    )
    def test_registered_under_each_real_insitu_source_type(self, source_type):
        from sar_validation.core import dry_collocation

        assert dry_collocation._PREDICATES[source_type] is dry_collocation._predict_insitu

    def test_no_source_type_insitu_key_registered(self):
        """The original plan draft assumed a single source_type="insitu"
        -- that key must not exist anywhere in _PREDICATES."""
        from sar_validation.core import dry_collocation

        assert "insitu" not in dry_collocation._PREDICATES

    @pytest.mark.parametrize(
        "source_type", ["mooring", "buoy", "drifter", "ferrybox", "tidal_gauge"],
    )
    def test_station_ranges_dry_filtered_to_single_source_type_not_full_batch(
        self, monkeypatch, source_type,
    ):
        """The core correction this task makes: even though a real
        InSituDownloader.download() call batches every requested platform
        type into one source_types=[...] list, predict_source is called
        once per validation source -- station_ranges_dry must only ever
        be asked about the one source_type it was invoked for."""
        from sar_validation.core import dry_collocation
        from sar_validation.downloaders.insitu_downloader import InSituDownloader

        seen_source_types = []

        def _fake_station_ranges_dry(
            self, min_lon, max_lon, min_lat, max_lat, start, end,
            source_types=None, dataset_part=None, variables=None,
        ):
            seen_source_types.append(source_types)
            return {"S1": (45.0, 0.0, datetime(2026, 8, 1, 5, 0, 0), datetime(2026, 8, 1, 7, 0, 0))}

        monkeypatch.setattr(InSituDownloader, "station_ranges_dry", _fake_station_ranges_dry)
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        result = dry_collocation.predict_source(
            SimpleNamespace(source_type=source_type), cfg=self._cfg(), sar_footprints=[self._footprint()],
        )

        assert seen_source_types == [[source_type]]
        assert all(st != ["mooring", "buoy", "drifter", "ferrybox", "tidal_gauge"] for st in seen_source_types)
        assert result.verdict == "collocated"
        assert result.source_type == source_type
        assert result.bucket == "ground-point"

    def test_none_predicted_when_no_data_available(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.downloaders.insitu_downloader import InSituDownloader

        monkeypatch.setattr(InSituDownloader, "station_ranges_dry", lambda self, *a, **k: {})
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        result = dry_collocation.predict_source(
            SimpleNamespace(source_type="buoy"), cfg=self._cfg(), sar_footprints=[self._footprint()],
        )

        assert result.verdict == "none-predicted"

    def test_unknown_when_availability_check_raises(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.downloaders.insitu_downloader import InSituDownloader

        def _raise(self, *a, **k):
            raise RuntimeError("copernicusmarine unreachable")

        monkeypatch.setattr(InSituDownloader, "station_ranges_dry", _raise)
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        result = dry_collocation.predict_source(
            SimpleNamespace(source_type="tidal_gauge"), cfg=self._cfg(), sar_footprints=[self._footprint()],
        )

        assert result.verdict == "unknown"

    def test_collocated_when_a_real_station_falls_within_the_footprint(self, monkeypatch):
        """The core thing station_ranges_dry + _point_in_footprint
        replaces the old boolean check_availability_dry with: a station
        whose real coordinates fall within the footprint's shape (and
        whose own observation window overlaps the footprint's padded
        window) makes it match."""
        from sar_validation.core import dry_collocation
        from sar_validation.downloaders.insitu_downloader import InSituDownloader

        monkeypatch.setattr(
            InSituDownloader, "station_ranges_dry",
            lambda self, *a, **k: {
                "S1": (45.0, 0.0, datetime(2026, 8, 1, 5, 0, 0), datetime(2026, 8, 1, 7, 0, 0)),
            },
        )
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        result = dry_collocation.predict_source(
            SimpleNamespace(source_type="ferrybox"), cfg=self._cfg(), sar_footprints=[self._footprint()],
        )

        assert result.verdict == "collocated"
        assert result.matched_stations == ["S1"]

    def test_none_predicted_when_station_is_outside_the_real_vignette_location(self, monkeypatch):
        """A wv_points footprint's own bbox is the bounding envelope of
        every scattered vignette point, which can span thousands of km --
        a station technically inside that huge box but nowhere near the
        real (single, tiny) vignette here must not count as collocated."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint
        from sar_validation.downloaders.insitu_downloader import InSituDownloader

        footprint = SarFootprint(
            kind="wv_points", bbox=(-70.0, -20.0, 0.0, 40.0), polygon=None,
            points=[(20.0, -50.0)],  # (lat, lon) -- the one real vignette
            sensing_start=datetime(2026, 8, 1, 6, 0, 0), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="wv.SAFE",
        )
        monkeypatch.setattr(
            InSituDownloader, "station_ranges_dry",
            lambda self, *a, **k: {
                # Inside the wide bbox, but thousands of km from the real vignette.
                "S1": (35.0, -25.0, datetime(2026, 8, 1, 5, 0, 0), datetime(2026, 8, 1, 7, 0, 0)),
            },
        )
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        result = dry_collocation.predict_source(
            SimpleNamespace(source_type="mooring"), cfg=self._cfg(), sar_footprints=[footprint],
        )

        assert result.verdict == "none-predicted"

    def test_none_predicted_when_station_observation_window_does_not_overlap(self, monkeypatch):
        """A station geographically within the footprint but whose real
        observations all fall outside the footprint's own padded time
        window must not count either."""
        from sar_validation.core import dry_collocation
        from sar_validation.downloaders.insitu_downloader import InSituDownloader

        monkeypatch.setattr(
            InSituDownloader, "station_ranges_dry",
            lambda self, *a, **k: {
                "S1": (45.0, 0.0, datetime(2026, 8, 3, 0, 0, 0), datetime(2026, 8, 3, 1, 0, 0)),
            },
        )
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        result = dry_collocation.predict_source(
            SimpleNamespace(source_type="mooring"), cfg=self._cfg(), sar_footprints=[self._footprint()],
        )

        assert result.verdict == "none-predicted"

    def test_unknown_when_no_sar_footprints_supplied(self, monkeypatch):
        from sar_validation.core import dry_collocation

        result = dry_collocation.predict_source(
            SimpleNamespace(source_type="mooring"), cfg=self._cfg(), sar_footprints=[],
        )

        assert result.verdict == "unknown"

    def test_single_network_call_regardless_of_footprint_count(self, monkeypatch):
        """station_ranges_dry must be called exactly once, regardless of
        footprint count -- not once per footprint, which would mean one
        copernicusmarine round-trip per footprint per source type. Every
        footprint's own match is decided locally against that one
        result."""
        from sar_validation.core import dry_collocation
        from sar_validation.downloaders.insitu_downloader import InSituDownloader

        calls = []

        def _fake_station_ranges_dry(self, *a, **k):
            calls.append(a)
            return {}

        monkeypatch.setattr(InSituDownloader, "station_ranges_dry", _fake_station_ranges_dry)
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        footprints = [self._footprint() for _ in range(67)]
        result = dry_collocation.predict_source(
            SimpleNamespace(source_type="mooring"), cfg=self._cfg(), sar_footprints=footprints,
        )

        assert result.verdict == "none-predicted"
        assert len(calls) == 1

    def test_exhaustive_by_default_reports_every_matched_footprint(self, monkeypatch):
        """By default (stop_on_first_match=False, the --dry-collocation
        preview path's own convention -- see predict_source's docstring)
        every footprint must be checked against the fetched stations, and
        each real hit must contribute its own entry to matched_windows --
        not just "at least one", matching altimeter's own "N matched
        window(s)" reporting."""
        from sar_validation.core import dry_collocation
        from sar_validation.downloaders.insitu_downloader import InSituDownloader

        # 3 footprints; stations sit at footprint 1 and footprint 3's own
        # locations only -- a match on a non-last footprint is the case
        # an unconditional early break would silently truncate.
        footprints = [self._footprint_at(-9.5), self._footprint_at(-7.5), self._footprint_at(-5.5)]
        ranges = {
            "S1": (45.0, -9.5, datetime(2026, 8, 1, 5, 0, 0), datetime(2026, 8, 1, 7, 0, 0)),
            "S2": (45.0, -5.5, datetime(2026, 8, 1, 5, 0, 0), datetime(2026, 8, 1, 7, 0, 0)),
        }
        monkeypatch.setattr(InSituDownloader, "station_ranges_dry", lambda self, *a, **k: ranges)
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        result = dry_collocation.predict_source(
            SimpleNamespace(source_type="mooring"), cfg=self._cfg(), sar_footprints=footprints,
        )

        assert result.verdict == "collocated"
        assert len(result.matched_windows or []) == 2  # footprints 1 and 3
        assert result.matched_stations == ["S1", "S2"]
        assert "2 of 3" in result.detail

    def test_stop_on_first_match_true_stops_scanning_footprints_early(self, monkeypatch):
        """stop_on_first_match=True (the real-run gating path -- see
        DataOrchestrator._collocation_predictions) must stop scanning
        footprints as soon as one is confirmed -- purely a local-loop
        optimization now, since station_ranges_dry itself is already
        just one call regardless."""
        from sar_validation.core import dry_collocation
        from sar_validation.downloaders.insitu_downloader import InSituDownloader

        footprints = [self._footprint_at(-9.5), self._footprint_at(-7.5), self._footprint_at(-5.5)]
        ranges = {
            "S1": (45.0, -9.5, datetime(2026, 8, 1, 5, 0, 0), datetime(2026, 8, 1, 7, 0, 0)),
            "S2": (45.0, -5.5, datetime(2026, 8, 1, 5, 0, 0), datetime(2026, 8, 1, 7, 0, 0)),
        }
        monkeypatch.setattr(InSituDownloader, "station_ranges_dry", lambda self, *a, **k: ranges)
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        result = dry_collocation.predict_source(
            SimpleNamespace(source_type="mooring"), cfg=self._cfg(), sar_footprints=footprints,
            stop_on_first_match=True,
        )

        assert result.verdict == "collocated"
        assert len(result.matched_windows or []) == 1  # stopped after footprint 1's own match
        assert "stopped at first match" in result.detail
        assert "--dry-collocation-detail" in result.detail

    def test_queries_only_variables_relevant_to_the_recipe(self, monkeypatch):
        """A pure-currents drifter (EWCT/NSCT only) has nothing comparable
        to validate against on a "waves" recipe. station_ranges_dry must
        be asked for variables_for_recipe(cfg.variable), not every
        variable this dataset carries."""
        from sar_validation.core import dry_collocation
        from sar_validation.downloaders.insitu_downloader import InSituDownloader, variables_for_recipe

        seen_variables = []

        def _fake_station_ranges_dry(self, *a, variables=None, **k):
            seen_variables.append(variables)
            return {}

        monkeypatch.setattr(InSituDownloader, "station_ranges_dry", _fake_station_ranges_dry)
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        dry_collocation.predict_source(
            SimpleNamespace(source_type="mooring"), cfg=self._cfg(), sar_footprints=[self._footprint()],
        )

        assert seen_variables == [variables_for_recipe("waves")]
        assert seen_variables != [None]


class TestPredictInsituCurrentsHistorical:
    """_predict_insitu_currents_historical is registered under the four
    real orchestrator.py source_type keys (adcp_historical,
    argo_historical, drifter_historical, glider_historical) -- there is
    no source_type="insitu_currents_historical" anywhere in this
    codebase. Unlike _predict_insitu (a shared InSituDownloader instance
    filtered per-call by source_types=[...]), each of these four keys
    must construct its own InSituCurrentsHistoricalDownloader instance
    with instrument=<key minus "_historical">, since instrument is a
    required constructor argument there, not a query filter."""

    def _cfg(self):
        return SimpleNamespace(
            variable="waves", collocation=SimpleNamespace(sar_footprint_radius_km=14.0),
        )

    def _old_footprint(self):
        """Sensing time well past _MIN_AGE_DAYS (182 days) so the
        delayed-mode archive eligibility check passes."""
        from sar_validation.core.dry_collocation import SarFootprint

        old = datetime.now(timezone.utc) - timedelta(days=400)
        return SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=old, sensing_end=old + timedelta(minutes=1),
            source_file="s1.SAFE",
        )

    def _old_footprint_at(self, lon):
        """A narrow-bbox old-enough footprint centered on lon --
        distinguishable from its siblings by location, for tests needing
        several footprints that only some real stations fall within."""
        from sar_validation.core.dry_collocation import SarFootprint

        old = datetime.now(timezone.utc) - timedelta(days=400)
        return SarFootprint(
            kind="polygon", bbox=(lon - 0.5, lon + 0.5, 35.0, 55.0), polygon=None, points=None,
            sensing_start=old, sensing_end=old + timedelta(minutes=1),
            source_file="s1.SAFE",
        )

    def _recent_footprint(self):
        """Sensing time inside the archive's real lag window -- download()
        itself would skip this without touching the network."""
        from sar_validation.core.dry_collocation import SarFootprint

        recent = datetime.now(timezone.utc) - timedelta(days=10)
        return SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=recent, sensing_end=recent + timedelta(minutes=1),
            source_file="s1.SAFE",
        )

    @pytest.mark.parametrize(
        "source_type", ["adcp_historical", "argo_historical", "drifter_historical", "glider_historical"],
    )
    def test_registered_under_each_real_source_type(self, source_type):
        from sar_validation.core import dry_collocation

        assert (
            dry_collocation._PREDICATES[source_type]
            is dry_collocation._predict_insitu_currents_historical
        )

    def test_no_bare_insitu_currents_historical_key_registered(self):
        """The original plan draft assumed a single
        source_type="insitu_currents_historical" -- that key must not
        exist anywhere in _PREDICATES."""
        from sar_validation.core import dry_collocation

        assert "insitu_currents_historical" not in dry_collocation._PREDICATES

    @pytest.mark.parametrize(
        "source_type, expected_instrument",
        [
            ("adcp_historical", "adcp"),
            ("argo_historical", "argo"),
            ("drifter_historical", "drifter"),
            ("glider_historical", "glider"),
        ],
    )
    def test_constructs_downloader_with_correct_instrument(
        self, monkeypatch, source_type, expected_instrument,
    ):
        """The core correction this task makes: instrument must be
        derived by stripping "_historical" off source.source_type, never
        left as the raw source_type string and never hardcoded to one
        fixed instrument regardless of which key was invoked."""
        from sar_validation.core import dry_collocation
        from sar_validation.downloaders.insitu_currents_historical_downloader import (
            InSituCurrentsHistoricalDownloader,
        )

        seen_instruments = []
        real_init = InSituCurrentsHistoricalDownloader.__init__

        def _spy_init(self, instrument, *a, **k):
            seen_instruments.append(instrument)
            real_init(self, instrument, *a, **k)

        old = datetime.now(timezone.utc) - timedelta(days=400)
        monkeypatch.setattr(InSituCurrentsHistoricalDownloader, "__init__", _spy_init)
        monkeypatch.setattr(
            InSituCurrentsHistoricalDownloader, "station_ranges_dry",
            lambda self, *a, **k: {
                "S1": (45.0, 0.0, old - timedelta(minutes=30), old + timedelta(minutes=90)),
            },
        )
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        result = dry_collocation.predict_source(
            SimpleNamespace(source_type=source_type), cfg=self._cfg(),
            sar_footprints=[self._old_footprint()],
        )

        assert seen_instruments == [expected_instrument]
        assert expected_instrument != source_type
        assert result.verdict == "collocated"
        assert result.source_type == source_type
        assert result.bucket == "ground-point"

    def test_none_predicted_when_no_data_available(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.downloaders.insitu_currents_historical_downloader import (
            InSituCurrentsHistoricalDownloader,
        )

        monkeypatch.setattr(
            InSituCurrentsHistoricalDownloader, "station_ranges_dry", lambda self, *a, **k: {},
        )
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        result = dry_collocation.predict_source(
            SimpleNamespace(source_type="argo_historical"), cfg=self._cfg(),
            sar_footprints=[self._old_footprint()],
        )

        assert result.verdict == "none-predicted"

    def test_unknown_when_availability_check_raises(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.downloaders.insitu_currents_historical_downloader import (
            InSituCurrentsHistoricalDownloader,
        )

        def _raise(self, *a, **k):
            raise RuntimeError("copernicusmarine unreachable")

        monkeypatch.setattr(InSituCurrentsHistoricalDownloader, "station_ranges_dry", _raise)
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        result = dry_collocation.predict_source(
            SimpleNamespace(source_type="glider_historical"), cfg=self._cfg(),
            sar_footprints=[self._old_footprint()],
        )

        assert result.verdict == "unknown"

    def test_unknown_when_every_footprint_is_too_recent(self, monkeypatch):
        """download() itself skips (no network call) for any window whose
        end date is younger than _MIN_AGE_DAYS -- the predicate must
        mirror that real eligibility gate as "unknown" (not a confident
        "none-predicted"), since a too-recent footprint says nothing
        about whether data would eventually appear once the archive
        catches up."""
        from sar_validation.core import dry_collocation
        from sar_validation.downloaders.insitu_currents_historical_downloader import (
            InSituCurrentsHistoricalDownloader,
        )

        calls = []
        monkeypatch.setattr(
            InSituCurrentsHistoricalDownloader, "station_ranges_dry",
            lambda self, *a, **k: calls.append(1) or {},
        )
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        result = dry_collocation.predict_source(
            SimpleNamespace(source_type="drifter_historical"), cfg=self._cfg(),
            sar_footprints=[self._recent_footprint()],
        )

        assert result.verdict == "unknown"
        assert calls == []

    def test_collocated_when_a_real_station_falls_within_the_footprint(self, monkeypatch):
        """The core thing station_ranges_dry + _point_in_footprint
        replaces the old boolean check_availability_dry with -- see
        TestPredictInsitu's identical test."""
        from sar_validation.core import dry_collocation
        from sar_validation.downloaders.insitu_currents_historical_downloader import (
            InSituCurrentsHistoricalDownloader,
        )

        old = datetime.now(timezone.utc) - timedelta(days=400)
        monkeypatch.setattr(
            InSituCurrentsHistoricalDownloader, "station_ranges_dry",
            lambda self, *a, **k: {
                "S1": (45.0, 0.0, old - timedelta(minutes=30), old + timedelta(minutes=90)),
            },
        )
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        result = dry_collocation.predict_source(
            SimpleNamespace(source_type="adcp_historical"), cfg=self._cfg(),
            sar_footprints=[self._old_footprint()],
        )

        assert result.verdict == "collocated"
        assert result.matched_stations == ["S1"]

    def test_single_network_call_regardless_of_footprint_count(self, monkeypatch):
        """Regression test mirroring TestPredictInsitu's own: the
        predicate now makes exactly one real network call regardless of
        eligible footprint count (down from up to one per footprint)."""
        from sar_validation.core import dry_collocation
        from sar_validation.downloaders.insitu_currents_historical_downloader import (
            InSituCurrentsHistoricalDownloader,
        )

        calls = []

        def _fake_station_ranges_dry(self, *a, **k):
            calls.append(a)
            return {}

        monkeypatch.setattr(InSituCurrentsHistoricalDownloader, "station_ranges_dry", _fake_station_ranges_dry)
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        footprints = [self._old_footprint() for _ in range(67)]
        result = dry_collocation.predict_source(
            SimpleNamespace(source_type="glider_historical"), cfg=self._cfg(), sar_footprints=footprints,
        )

        assert result.verdict == "none-predicted"
        assert len(calls) == 1

    def test_exhaustive_by_default_reports_every_matched_footprint(self, monkeypatch):
        """See _predict_insitu's identical regression test -- the sibling
        predicate had the same unconditional-stop-at-first-hit bug."""
        from sar_validation.core import dry_collocation
        from sar_validation.downloaders.insitu_currents_historical_downloader import (
            InSituCurrentsHistoricalDownloader,
        )

        # 3 footprints; stations sit at footprint 1 and footprint 3's own
        # locations only -- a match on a non-last footprint is the case
        # an unconditional early break would silently truncate.
        footprints = [
            self._old_footprint_at(-9.5), self._old_footprint_at(-7.5), self._old_footprint_at(-5.5),
        ]
        old = datetime.now(timezone.utc) - timedelta(days=400)
        ranges = {
            "S1": (45.0, -9.5, old - timedelta(minutes=30), old + timedelta(minutes=90)),
            "S2": (45.0, -5.5, old - timedelta(minutes=30), old + timedelta(minutes=90)),
        }
        monkeypatch.setattr(InSituCurrentsHistoricalDownloader, "station_ranges_dry", lambda self, *a, **k: ranges)
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        result = dry_collocation.predict_source(
            SimpleNamespace(source_type="adcp_historical"), cfg=self._cfg(), sar_footprints=footprints,
        )

        assert result.verdict == "collocated"
        assert len(result.matched_windows or []) == 2
        assert result.matched_stations == ["S1", "S2"]
        assert "2 of 3" in result.detail

    def test_stop_on_first_match_true_stops_scanning_footprints_early(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.downloaders.insitu_currents_historical_downloader import (
            InSituCurrentsHistoricalDownloader,
        )

        footprints = [
            self._old_footprint_at(-9.5), self._old_footprint_at(-7.5), self._old_footprint_at(-5.5),
        ]
        old = datetime.now(timezone.utc) - timedelta(days=400)
        ranges = {
            "S1": (45.0, -9.5, old - timedelta(minutes=30), old + timedelta(minutes=90)),
            "S2": (45.0, -5.5, old - timedelta(minutes=30), old + timedelta(minutes=90)),
        }
        monkeypatch.setattr(InSituCurrentsHistoricalDownloader, "station_ranges_dry", lambda self, *a, **k: ranges)
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        result = dry_collocation.predict_source(
            SimpleNamespace(source_type="adcp_historical"), cfg=self._cfg(), sar_footprints=footprints,
            stop_on_first_match=True,
        )

        assert result.verdict == "collocated"
        assert len(result.matched_windows or []) == 1
        assert "stopped at first match" in result.detail
        assert "--dry-collocation-detail" in result.detail


class TestHfRadarCandidateRegions:
    """_hf_radar_candidate_regions is the per-region area-vs-area
    refinement HF-radar's predicate needs -- a footprint sitting only in
    one corner of a large multi-region country must still be attributed
    to the right specific region(s), not whichever region the recipe's
    own (much larger) nominal bbox happens to overlap most."""

    _REGIONS = {
        "West": {"bbox": (-130.0, -115.0, 30.0, 50.0)},
        "East": {"bbox": (-98.0, -60.0, 22.0, 46.0)},
        "Elsewhere": {"bbox": (100.0, 110.0, 0.0, 10.0)},
    }

    def test_only_genuinely_overlapping_regions_are_returned(self):
        from sar_validation.core.dry_collocation import SarFootprint, _hf_radar_candidate_regions

        footprint = SarFootprint(
            kind="polygon", bbox=(-126.0, -120.0, 33.0, 38.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1), sensing_end=datetime(2026, 8, 1), source_file="s1.SAFE",
        )

        candidates = _hf_radar_candidate_regions(self._REGIONS, footprint)

        names = [name for name, _bbox in candidates]
        assert names == ["West"]

    def test_intersected_bbox_is_clamped_to_region_bounds(self):
        from sar_validation.core.dry_collocation import SarFootprint, _hf_radar_candidate_regions

        # Footprint spans across West's eastern edge (-115.0) into open space.
        footprint = SarFootprint(
            kind="polygon", bbox=(-118.0, -110.0, 33.0, 38.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1), sensing_end=datetime(2026, 8, 1), source_file="s1.SAFE",
        )

        candidates = _hf_radar_candidate_regions(self._REGIONS, footprint)

        assert len(candidates) == 1
        name, bbox = candidates[0]
        assert name == "West"
        # Clamped to West's own eastern edge (-115.0), not the footprint's own -110.0.
        assert bbox == (-118.0, -115.0, 33.0, 38.0)

    def test_footprint_overlapping_two_regions_returns_both(self):
        from sar_validation.core.dry_collocation import SarFootprint, _hf_radar_candidate_regions

        footprint = SarFootprint(
            kind="polygon", bbox=(-120.0, -90.0, 25.0, 35.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1), sensing_end=datetime(2026, 8, 1), source_file="s1.SAFE",
        )

        candidates = _hf_radar_candidate_regions(self._REGIONS, footprint)

        names = {name for name, _bbox in candidates}
        assert names == {"West", "East"}

    def test_no_overlapping_region_returns_empty(self):
        from sar_validation.core.dry_collocation import SarFootprint, _hf_radar_candidate_regions

        footprint = SarFootprint(
            kind="polygon", bbox=(0.0, 5.0, 0.0, 5.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1), sensing_end=datetime(2026, 8, 1), source_file="s1.SAFE",
        )

        assert _hf_radar_candidate_regions(self._REGIONS, footprint) == []


class TestPredictHfRadarShared:
    """_predict_hf_radar is the shared predicate parameterized across the
    three HF-radar sources with their own region table (hf_radar,
    hf_radar_historical, hf_radar_noaa) -- mirrors _predict_ismn's
    tolerance-resolution mocking convention."""

    _REGIONS = {"West": {"bbox": (-130.0, -115.0, 30.0, 50.0)}}

    def _footprint(self):
        from sar_validation.core.dry_collocation import SarFootprint

        return SarFootprint(
            kind="polygon", bbox=(-126.0, -120.0, 33.0, 38.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 0), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

    def test_unknown_when_no_sar_footprints_supplied(self, monkeypatch):
        from sar_validation.core import dry_collocation

        result = dry_collocation._predict_hf_radar(
            source=object(), cfg=object(), sar_footprints=[],
            check_availability_dry=lambda *a, **k: True,
            regions_table=self._REGIONS, source_type="hf_radar",
        )

        assert result.verdict == "unknown"

    def test_collocated_when_candidate_region_check_true(self, monkeypatch):
        from sar_validation.core import dry_collocation

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        seen_bboxes = []

        def _fake_check(min_lon, max_lon, min_lat, max_lat, start, end):
            seen_bboxes.append((min_lon, max_lon, min_lat, max_lat))
            return True

        result = dry_collocation._predict_hf_radar(
            source=object(), cfg=object(), sar_footprints=[self._footprint()],
            check_availability_dry=_fake_check,
            regions_table=self._REGIONS, source_type="hf_radar",
        )

        assert result.verdict == "collocated"
        assert result.source_type == "hf_radar"
        assert result.bucket == "ground-point"
        # Called with the region-clamped bbox (West's own bounds), not the
        # footprint's raw bbox.
        assert seen_bboxes == [(-126.0, -120.0, 33.0, 38.0)]

    def test_none_predicted_when_no_candidate_region_has_data(self, monkeypatch):
        from sar_validation.core import dry_collocation

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        result = dry_collocation._predict_hf_radar(
            source=object(), cfg=object(), sar_footprints=[self._footprint()],
            check_availability_dry=lambda *a, **k: False,
            regions_table=self._REGIONS, source_type="hf_radar",
        )

        assert result.verdict == "none-predicted"

    def test_none_predicted_when_footprint_overlaps_no_known_region(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        footprint = SarFootprint(
            kind="polygon", bbox=(0.0, 5.0, 0.0, 5.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 0), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = dry_collocation._predict_hf_radar(
            source=object(), cfg=object(), sar_footprints=[footprint],
            check_availability_dry=lambda *a, **k: True,
            regions_table=self._REGIONS, source_type="hf_radar",
        )

        assert result.verdict == "none-predicted"

    def test_unknown_when_check_raises(self, monkeypatch):
        from sar_validation.core import dry_collocation

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        def _raise(*a, **k):
            raise RuntimeError("copernicusmarine unreachable")

        result = dry_collocation._predict_hf_radar(
            source=object(), cfg=object(), sar_footprints=[self._footprint()],
            check_availability_dry=_raise,
            regions_table=self._REGIONS, source_type="hf_radar",
        )

        assert result.verdict == "unknown"

    def test_resolves_tolerance_via_hf_radar_grid_key(self, monkeypatch):
        from sar_validation.core import dry_collocation

        seen_keys = []

        def _fake_resolve(cfg, *source_types):
            seen_keys.append(source_types)
            return 90

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", _fake_resolve)

        dry_collocation._predict_hf_radar(
            source=object(), cfg=object(), sar_footprints=[self._footprint()],
            check_availability_dry=lambda *a, **k: True,
            regions_table=self._REGIONS, source_type="hf_radar_noaa",
        )

        assert seen_keys == [("hf_radar_grid",)]

    def test_stops_after_first_confirmed_hit_across_regions(self, monkeypatch):
        """Once one candidate region's check_availability_dry returns True,
        no further region (for that footprint, or any later footprint)
        should be probed -- this predicate runs on the default path of
        every real recipe run once gating is active, so its live-probe
        cost must be bounded by the first confirmed hit, not exhaustive.
        Uses its own multi-region table (self._REGIONS on this class only
        has one entry -- see TestHfRadarCandidateRegions for the
        multi-region fixture this borrows the bboxes from)."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        regions = {
            "West": {"bbox": (-130.0, -115.0, 30.0, 50.0)},
            "East": {"bbox": (-98.0, -60.0, 22.0, 46.0)},
            "Elsewhere": {"bbox": (100.0, 110.0, 0.0, 10.0)},
        }

        # Overlaps both "West" and "East".
        multi_region_footprint = SarFootprint(
            kind="polygon", bbox=(-120.0, -90.0, 25.0, 35.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 0), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )
        second_footprint = SarFootprint(
            kind="polygon", bbox=(100.0, 105.0, 1.0, 5.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 2, 6, 0, 0), sensing_end=datetime(2026, 8, 2, 6, 1, 0),
            source_file="s2.SAFE",
        )

        calls = []

        def _fake_check(min_lon, max_lon, min_lat, max_lat, start, end):
            calls.append((min_lon, max_lon, min_lat, max_lat))
            return True

        result = dry_collocation._predict_hf_radar(
            source=object(), cfg=object(), sar_footprints=[multi_region_footprint, second_footprint],
            check_availability_dry=_fake_check,
            regions_table=regions, source_type="hf_radar",
        )

        assert result.verdict == "collocated"
        # Only the first overlapping region of the first footprint was
        # ever checked -- neither the second overlapping region nor the
        # second footprint's own region ("Elsewhere") triggered a call.
        assert len(calls) == 1


class TestPredictHfRadarRegistrations:
    """One predict_source integration test per real HF-radar
    source_type, exercising the actual _PREDICATES dispatch."""

    @pytest.mark.parametrize(
        "source_type,check_dry_name",
        [
            pytest.param("hf_radar", "_hf_radar_copernicus_check_dry", id="hf_radar"),
            pytest.param("hf_radar_historical", "_hf_radar_historical_check_dry", id="hf_radar_historical"),
            pytest.param("hf_radar_noaa", "_hf_radar_noaa_check_dry", id="hf_radar_noaa"),
        ],
    )
    def test_collocated_via_predict_source(self, monkeypatch, source_type, check_dry_name):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint, predict_source

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)
        monkeypatch.setattr(dry_collocation, check_dry_name, lambda *a, **k: True)

        footprint = SarFootprint(
            kind="polygon", bbox=(-126.0, -120.0, 33.0, 38.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 0), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = predict_source(
            SimpleNamespace(source_type=source_type), cfg=object(), sar_footprints=[footprint],
        )

        assert result.verdict == "collocated"
        assert result.source_type == source_type
        assert result.bucket == "ground-point"

    def test_registered_under_own_source_type_hf_radar(self):
        from sar_validation.core import dry_collocation

        assert dry_collocation._PREDICATES["hf_radar"] is dry_collocation._predict_hf_radar_copernicus

    @pytest.mark.parametrize(
        "source_type", ["hf_radar_historical", "hf_radar_noaa", "hf_radar_us"],
    )
    def test_registered_under_own_source_type(self, source_type):
        from sar_validation.core import dry_collocation

        predicate = dry_collocation._PREDICATES[source_type]
        assert predicate is getattr(dry_collocation, f"_predict_{source_type}")

    def test_no_thredds_only_source_type_registered(self):
        """noaa_hfradar_thredds_downloader.py is used internally by
        hf_radar_us's own waterfall, not dispatched on its own -- there is
        no separate NOAA-THREDDS-only source_type in orchestrator.py's
        _dispatch_source registry."""
        from sar_validation.core import dry_collocation

        assert "hf_radar_thredds" not in dry_collocation._PREDICATES
        assert "noaa_hfradar_thredds" not in dry_collocation._PREDICATES


class TestPredictHfRadarUs:
    """_predict_hf_radar_us delegates directly to
    HFRadarUSDownloader.check_availability_dry per footprint's own
    (un-refined) bbox -- unlike its three siblings, it does not iterate a
    local region table, since the delegated waterfall already resolves
    its own region internally."""

    def _footprint(self):
        from sar_validation.core.dry_collocation import SarFootprint

        return SarFootprint(
            kind="polygon", bbox=(-126.0, -120.0, 33.0, 38.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 0), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

    def test_unknown_when_no_sar_footprints_supplied(self):
        from sar_validation.core import dry_collocation

        result = dry_collocation._predict_hf_radar_us(source=object(), cfg=object(), sar_footprints=[])

        assert result.verdict == "unknown"

    def test_collocated_via_predict_source(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import predict_source

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        seen_bboxes = []

        def _fake_check(min_lon, max_lon, min_lat, max_lat, start, end):
            seen_bboxes.append((min_lon, max_lon, min_lat, max_lat))
            return True

        monkeypatch.setattr(dry_collocation, "_hf_radar_us_check_dry", _fake_check)

        result = predict_source(
            SimpleNamespace(source_type="hf_radar_us"), cfg=object(), sar_footprints=[self._footprint()],
        )

        assert result.verdict == "collocated"
        assert result.source_type == "hf_radar_us"
        assert result.bucket == "ground-point"
        # Called with the footprint's own raw bbox -- no per-region
        # candidate-table refinement at this layer.
        assert seen_bboxes == [(-126.0, -120.0, 33.0, 38.0)]

    def test_none_predicted_when_check_returns_false(self, monkeypatch):
        from sar_validation.core import dry_collocation

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)
        monkeypatch.setattr(dry_collocation, "_hf_radar_us_check_dry", lambda *a, **k: False)

        result = dry_collocation._predict_hf_radar_us(
            source=object(), cfg=object(), sar_footprints=[self._footprint()],
        )

        assert result.verdict == "none-predicted"

    def test_unknown_when_check_raises(self, monkeypatch):
        from sar_validation.core import dry_collocation

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        def _raise(*a, **k):
            raise RuntimeError("waterfall exploded")

        monkeypatch.setattr(dry_collocation, "_hf_radar_us_check_dry", _raise)

        result = dry_collocation._predict_hf_radar_us(
            source=object(), cfg=object(), sar_footprints=[self._footprint()],
        )

        assert result.verdict == "unknown"

    def test_stops_after_first_confirmed_hit_across_footprints(self, monkeypatch):
        """Once the first footprint's own check confirms availability, the
        second footprint must not be probed at all -- bounds this
        predicate's live-probe cost on the real-run gating path."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        calls = []

        def _fake_check(min_lon, max_lon, min_lat, max_lat, start, end):
            calls.append((min_lon, max_lon, min_lat, max_lat))
            return True

        monkeypatch.setattr(dry_collocation, "_hf_radar_us_check_dry", _fake_check)

        second_footprint = SarFootprint(
            kind="polygon", bbox=(0.0, 5.0, 0.0, 5.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 2, 6, 0, 0), sensing_end=datetime(2026, 8, 2, 6, 1, 0),
            source_file="s2.SAFE",
        )

        result = dry_collocation._predict_hf_radar_us(
            source=object(), cfg=object(), sar_footprints=[self._footprint(), second_footprint],
        )

        assert result.verdict == "collocated"
        assert len(calls) == 1


class TestPredictGlobalComposite:
    """_predict_global_composite is the shared predicate for the
    global-composite bucket (RSS radiometer, CDS SSM): both sources
    publish one daily global-coverage file, so only the calendar day(s) a
    footprint's (padded) sensing window touches matters -- no spatial
    (bbox/polygon) refinement. Every test here mocks
    _resolve_temporal_padding_minutes (matching every other predicate's
    own tests in this module) since cfg is a bare sentinel with no real
    ``collocation`` config attached."""

    def _footprint(self, day=datetime(2026, 8, 1, 6, 0, 0)):
        from sar_validation.core.dry_collocation import SarFootprint

        return SarFootprint(
            kind="polygon", bbox=(-10.0, 5.0, 50.0, 62.0), polygon=None, points=None,
            sensing_start=day, sensing_end=day + timedelta(minutes=1),
            source_file="s1.SAFE",
        )

    def test_unknown_when_no_sar_footprints_supplied(self):
        from sar_validation.core.dry_collocation import _predict_global_composite

        result = _predict_global_composite(
            source=object(), cfg=object(), sar_footprints=[],
            check_exists_dry=lambda day: True, source_type="radiometer",
        )

        assert result.verdict == "unknown"
        assert result.bucket == "global-composite"

    def test_collocated_when_any_footprint_day_exists(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import _predict_global_composite

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        seen_days = []

        def _check(day):
            seen_days.append(day)
            return day == datetime(2026, 8, 2).date()

        result = _predict_global_composite(
            source=object(), cfg=object(),
            sar_footprints=[self._footprint(datetime(2026, 8, 1, 6)), self._footprint(datetime(2026, 8, 2, 6))],
            check_exists_dry=_check, source_type="radiometer",
        )

        assert result.verdict == "collocated"
        assert result.bucket == "global-composite"
        assert seen_days == [datetime(2026, 8, 1).date(), datetime(2026, 8, 2).date()]

    def test_stops_after_first_confirmed_day(self, monkeypatch):
        """Once any day exists, no later day (of the same or a later
        footprint) should be probed -- bounds this predicate's live-probe
        cost on the real-run gating path."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import _predict_global_composite

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        seen_days = []

        def _check(day):
            seen_days.append(day)
            return True

        result = _predict_global_composite(
            source=object(), cfg=object(),
            sar_footprints=[self._footprint(datetime(2026, 8, 1, 6)), self._footprint(datetime(2026, 8, 2, 6))],
            check_exists_dry=_check, source_type="radiometer",
        )

        assert result.verdict == "collocated"
        assert seen_days == [datetime(2026, 8, 1).date()]

    def test_none_predicted_when_no_footprint_day_exists(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import _predict_global_composite

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        result = _predict_global_composite(
            source=object(), cfg=object(), sar_footprints=[self._footprint()],
            check_exists_dry=lambda day: False, source_type="cds_ssm",
        )

        assert result.verdict == "none-predicted"
        assert result.source_type == "cds_ssm"

    def test_unknown_when_check_raises(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import _predict_global_composite

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        def _raise(day):
            raise RuntimeError("boom")

        result = _predict_global_composite(
            source=object(), cfg=object(), sar_footprints=[self._footprint()],
            check_exists_dry=_raise, source_type="radiometer",
        )

        assert result.verdict == "unknown"

    def test_bbox_and_polygon_never_consulted(self, monkeypatch):
        """Spatial refinement isn't meaningful for a daily global-coverage
        file -- check_exists_dry only ever receives a calendar day, never
        the footprint's bbox/polygon."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint, _predict_global_composite

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 5.0, 50.0, 62.0), polygon=[(50.0, -10.0), (62.0, 5.0)], points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 0), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )
        call_args = []

        def _check(*args):
            call_args.append(args)
            return True

        _predict_global_composite(
            source=object(), cfg=object(), sar_footprints=[footprint],
            check_exists_dry=_check, source_type="radiometer",
        )

        assert call_args == [(datetime(2026, 8, 1).date(),)]

    def test_padded_window_crossing_day_boundary_checks_both_days(self, monkeypatch):
        """A footprint whose sensing window is close to a UTC day boundary
        must have BOTH the unpadded day and the padded-in adjacent day
        checked -- the real download path (orchestrator._padded_temporal_bounds)
        applies this same padding, so a footprint at 23:50 with a 30-minute
        tolerance can genuinely pull data from the next day; checking only
        the footprint's own unpadded sensing_start date would risk a false
        "none-predicted" for that adjacent day's real data."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint, _predict_global_composite

        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 30)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 5.0, 50.0, 62.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 23, 50, 0), sensing_end=datetime(2026, 8, 1, 23, 51, 0),
            source_file="s1.SAFE",
        )
        seen_days = []

        def _check(day):
            seen_days.append(day)
            return day == datetime(2026, 8, 2).date()

        result = _predict_global_composite(
            source=object(), cfg=object(), sar_footprints=[footprint],
            check_exists_dry=_check, source_type="radiometer",
        )

        assert seen_days == [datetime(2026, 8, 1).date(), datetime(2026, 8, 2).date()]
        assert result.verdict == "collocated"


class TestRadiometerCheckDry:
    def test_passes_sensors_from_download_kwargs(self, monkeypatch):
        from sar_validation.core import dry_collocation

        captured = {}

        class _FakeRadiometerDownloader:
            def __init__(self, output_dir):
                captured["output_dir"] = output_dir

            def check_exists_dry(self, day, sensors=None):
                captured["day"] = day
                captured["sensors"] = sensors
                return True

        monkeypatch.setattr(dry_collocation, "RadiometerDownloader", _FakeRadiometerDownloader)

        source = SimpleNamespace(download_kwargs={"sensors": ["amsr2"]})
        result = dry_collocation._radiometer_check_dry(source, datetime(2026, 8, 1).date())

        assert result is True
        assert captured["sensors"] == ["amsr2"]
        assert captured["day"] == datetime(2026, 8, 1).date()

    def test_defaults_sensors_to_none_when_not_in_download_kwargs(self, monkeypatch):
        from sar_validation.core import dry_collocation

        captured = {}

        class _FakeRadiometerDownloader:
            def __init__(self, output_dir):
                pass

            def check_exists_dry(self, day, sensors=None):
                captured["sensors"] = sensors
                return False

        monkeypatch.setattr(dry_collocation, "RadiometerDownloader", _FakeRadiometerDownloader)

        source = SimpleNamespace(download_kwargs={})
        dry_collocation._radiometer_check_dry(source, datetime(2026, 8, 1).date())

        assert captured["sensors"] is None


class TestCdsSsmCheckDry:
    def test_passes_product_type_from_download_kwargs(self, monkeypatch):
        from sar_validation.core import dry_collocation

        captured = {}

        class _FakeCDSSoilMoistureDownloader:
            def __init__(self, product_type, output_dir):
                captured["product_type"] = product_type

            def check_availability_dry(self, day):
                captured["day"] = day
                return True

        monkeypatch.setattr(dry_collocation, "CDSSoilMoistureDownloader", _FakeCDSSoilMoistureDownloader)

        source = SimpleNamespace(download_kwargs={"product_type": "passive"})
        result = dry_collocation._cds_ssm_check_dry(source, datetime(2026, 8, 1).date())

        assert result is True
        assert captured["product_type"] == "passive"

    def test_defaults_product_type_to_active(self, monkeypatch):
        from sar_validation.core import dry_collocation

        captured = {}

        class _FakeCDSSoilMoistureDownloader:
            def __init__(self, product_type, output_dir):
                captured["product_type"] = product_type

            def check_availability_dry(self, day):
                return False

        monkeypatch.setattr(dry_collocation, "CDSSoilMoistureDownloader", _FakeCDSSoilMoistureDownloader)

        source = SimpleNamespace(download_kwargs={})
        dry_collocation._cds_ssm_check_dry(source, datetime(2026, 8, 1).date())

        assert captured["product_type"] == "active"


class TestPredictGlobalCompositeRegistrations:
    """One predict_source integration test per real global-composite
    source_type, exercising the actual _PREDICATES dispatch."""

    @pytest.mark.parametrize(
        "source_type,check_dry_name",
        [
            pytest.param("radiometer", "_radiometer_check_dry", id="radiometer"),
            pytest.param("cds_ssm", "_cds_ssm_check_dry", id="cds_ssm"),
        ],
    )
    def test_collocated_via_predict_source(self, monkeypatch, source_type, check_dry_name):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint, predict_source

        monkeypatch.setattr(dry_collocation, check_dry_name, lambda source, day: True)
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 5.0, 50.0, 62.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 0), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = predict_source(
            SimpleNamespace(source_type=source_type, download_kwargs={}),
            cfg=object(), sar_footprints=[footprint],
        )

        assert result.verdict == "collocated"
        assert result.source_type == source_type
        assert result.bucket == "global-composite"

    @pytest.mark.parametrize("source_type", ["radiometer", "cds_ssm"])
    def test_none_predicted_via_predict_source(self, monkeypatch, source_type):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint, predict_source

        check_dry_name = f"_{source_type}_check_dry"
        monkeypatch.setattr(dry_collocation, check_dry_name, lambda source, day: False)
        monkeypatch.setattr(dry_collocation, "_resolve_temporal_padding_minutes", lambda cfg, *st: 90)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 5.0, 50.0, 62.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 0), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = predict_source(
            SimpleNamespace(source_type=source_type, download_kwargs={}),
            cfg=object(), sar_footprints=[footprint],
        )

        assert result.verdict == "none-predicted"

    def test_registered_under_own_source_type_radiometer(self):
        from sar_validation.core import dry_collocation

        assert dry_collocation._PREDICATES["radiometer"] is dry_collocation._predict_radiometer

    def test_registered_under_own_source_type_cds_ssm(self):
        from sar_validation.core import dry_collocation

        assert dry_collocation._PREDICATES["cds_ssm"] is dry_collocation._predict_cds_ssm


class TestPredictModelSource:
    """The models bucket (ERA5, HYCOM): a temporal-coverage-window check
    only -- no spatial refinement, since global/regional grid coverage is
    assumed."""

    def test_collocated_when_footprint_within_documented_coverage(self):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2019, 6, 1, 6, 0, 0), sensing_end=datetime(2019, 6, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = dry_collocation._predict_model_source(
            source=object(), cfg=object(), sar_footprints=[footprint],
            coverage_start=datetime(2018, 12, 4), coverage_end=None, source_type="hycom",
        )

        assert result.verdict == "collocated"
        assert result.source_type == "hycom"
        assert result.bucket == "model"

    def test_none_predicted_when_footprint_before_coverage_start(self):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2017, 1, 1, 6, 0, 0), sensing_end=datetime(2017, 1, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = dry_collocation._predict_model_source(
            source=object(), cfg=object(), sar_footprints=[footprint],
            coverage_start=datetime(2018, 12, 4), coverage_end=None, source_type="hycom",
        )

        assert result.verdict == "none-predicted"

    def test_none_predicted_when_footprint_after_coverage_end(self):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2015, 1, 1, 6, 0, 0), sensing_end=datetime(2015, 1, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = dry_collocation._predict_model_source(
            source=object(), cfg=object(), sar_footprints=[footprint],
            coverage_start=None, coverage_end=datetime(2010, 1, 1), source_type="era5",
        )

        assert result.verdict == "none-predicted"

    def test_collocated_when_coverage_start_is_none(self):
        """era5's own registration passes coverage_start=None -- it has no
        documented lower bound -- so even a very old footprint must count
        as within coverage."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(1960, 1, 1, 6, 0, 0), sensing_end=datetime(1960, 1, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = dry_collocation._predict_model_source(
            source=object(), cfg=object(), sar_footprints=[footprint],
            coverage_start=None, coverage_end=None, source_type="era5",
        )

        assert result.verdict == "collocated"

    def test_unknown_when_no_footprints_supplied(self):
        from sar_validation.core import dry_collocation

        result = dry_collocation._predict_model_source(
            source=object(), cfg=object(), sar_footprints=[],
            coverage_start=None, coverage_end=None, source_type="era5",
        )

        assert result.verdict == "unknown"

    def test_collocated_when_footprint_is_tz_aware_and_coverage_bound_is_naive(self):
        """Regression test: coverage_start/coverage_end (e.g. HYCOM's own
        module-level _HYCOM_MIN_DATE) are plain naive datetime constants,
        while a real SarFootprint's sensing_start/sensing_end is tz-aware.
        Before this comparison routed through _to_naive_utc, this raised
        "can't compare offset-naive and offset-aware datetimes" and
        crashed predict_source outright -- found via a live sanity check."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2019, 6, 1, 6, 0, 0, tzinfo=timezone.utc),
            sensing_end=datetime(2019, 6, 1, 6, 1, 0, tzinfo=timezone.utc),
            source_file="s1.SAFE",
        )

        result = dry_collocation._predict_model_source(
            source=object(), cfg=object(), sar_footprints=[footprint],
            coverage_start=datetime(2018, 12, 4), coverage_end=None, source_type="hycom",
        )

        assert result.verdict == "collocated"

    def test_none_predicted_when_tz_aware_footprint_is_before_naive_coverage_start(self):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2010, 1, 1, 6, 0, 0, tzinfo=timezone.utc),
            sensing_end=datetime(2010, 1, 1, 6, 1, 0, tzinfo=timezone.utc),
            source_file="s1.SAFE",
        )

        result = dry_collocation._predict_model_source(
            source=object(), cfg=object(), sar_footprints=[footprint],
            coverage_start=datetime(2018, 12, 4), coverage_end=None, source_type="hycom",
        )

        assert result.verdict == "none-predicted"


class TestPredictModelSourceLiveProbe:
    """The recent-date live-probe extension: for a footprint within
    _MODEL_RECENT_PROBE_WINDOW_DAYS of "now", the coverage-window check
    alone isn't enough -- an explicit live_probe callable must also
    confirm before the verdict is trusted as "collocated"."""

    @staticmethod
    def _recent_footprint():
        from sar_validation.core.dry_collocation import SarFootprint

        recent_start = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)
        return SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=recent_start, sensing_end=recent_start + timedelta(minutes=1),
            source_file="s1.SAFE",
        )

    @staticmethod
    def _old_footprint():
        from sar_validation.core.dry_collocation import SarFootprint

        return SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2020, 1, 1, 6, 0, 0), sensing_end=datetime(2020, 1, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

    def test_collocated_when_recent_footprint_confirmed_by_live_probe(self):
        from sar_validation.core import dry_collocation

        result = dry_collocation._predict_model_source(
            source=object(), cfg=object(), sar_footprints=[self._recent_footprint()],
            coverage_start=None, coverage_end=None, source_type="era5",
            live_probe=lambda fp: True,
        )

        assert result.verdict == "collocated"

    def test_unknown_when_recent_footprint_not_yet_confirmed_by_live_probe(self):
        from sar_validation.core import dry_collocation

        result = dry_collocation._predict_model_source(
            source=object(), cfg=object(), sar_footprints=[self._recent_footprint()],
            coverage_start=None, coverage_end=None, source_type="era5",
            live_probe=lambda fp: False,
        )

        assert result.verdict == "unknown"

    def test_unknown_when_live_probe_raises(self):
        from sar_validation.core import dry_collocation

        def _boom(fp):
            raise RuntimeError("network error")

        result = dry_collocation._predict_model_source(
            source=object(), cfg=object(), sar_footprints=[self._recent_footprint()],
            coverage_start=None, coverage_end=None, source_type="era5",
            live_probe=_boom,
        )

        assert result.verdict == "unknown"

    def test_older_footprint_never_invokes_live_probe(self):
        """A footprint well outside the recent-probe window must be
        trusted from the coverage-window check alone -- the live probe
        exists to cover recent, possibly-not-yet-published data, not
        every request."""
        from sar_validation.core import dry_collocation

        calls = []

        def _probe(fp):
            calls.append(fp)
            return True

        result = dry_collocation._predict_model_source(
            source=object(), cfg=object(), sar_footprints=[self._old_footprint()],
            coverage_start=None, coverage_end=None, source_type="era5",
            live_probe=_probe,
        )

        assert result.verdict == "collocated"
        assert calls == []

    def test_live_probe_stops_after_first_confirmation(self):
        """A single confirmed recent footprint is enough for the overall
        verdict -- the loop must not keep calling live_probe (a real,
        slow network round-trip for e.g. HYCOM) for every remaining
        recent, in-coverage footprint once one has already confirmed."""
        from sar_validation.core import dry_collocation

        recent_footprints = [self._recent_footprint() for _ in range(5)]
        calls = []

        def _probe(fp):
            calls.append(fp)
            return True

        result = dry_collocation._predict_model_source(
            source=object(), cfg=object(), sar_footprints=recent_footprints,
            coverage_start=None, coverage_end=None, source_type="era5",
            live_probe=_probe,
        )

        assert result.verdict == "collocated"
        assert len(calls) == 1

    def test_older_confirmed_footprint_outweighs_recent_unconfirmed_one(self):
        """One confirmed footprint (old, no probe needed) is enough for an
        overall "collocated" verdict, even alongside a recent footprint
        whose live probe came back empty."""
        from sar_validation.core import dry_collocation

        result = dry_collocation._predict_model_source(
            source=object(), cfg=object(),
            sar_footprints=[self._old_footprint(), self._recent_footprint()],
            coverage_start=None, coverage_end=None, source_type="era5",
            live_probe=lambda fp: False,
        )

        assert result.verdict == "collocated"


class TestHycomLiveProbe:
    def test_true_when_has_coverage_reports_a_granule(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        captured = {}

        class _FakeHycomDownloader:
            def __init__(self, output_dir):
                captured["output_dir"] = output_dir

            def has_coverage(self, dataset_key, seg_start, seg_end, clip_at_cutover=False):
                captured["args"] = (dataset_key, seg_start, seg_end, clip_at_cutover)
                return True

        monkeypatch.setattr(dry_collocation, "HycomDownloader", _FakeHycomDownloader)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2025, 1, 1, 6, 0, 0), sensing_end=datetime(2025, 1, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        assert dry_collocation._hycom_live_probe(footprint) is True
        assert captured["args"][0] == "espc_d_v02"

    def test_false_when_has_coverage_reports_no_granule(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        class _FakeHycomDownloader:
            def __init__(self, output_dir):
                pass

            def has_coverage(self, dataset_key, seg_start, seg_end, clip_at_cutover=False):
                return False

        monkeypatch.setattr(dry_collocation, "HycomDownloader", _FakeHycomDownloader)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2025, 1, 1, 6, 0, 0), sensing_end=datetime(2025, 1, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        assert dry_collocation._hycom_live_probe(footprint) is False

    def test_straddling_window_checks_both_segments(self, monkeypatch):
        """A footprint straddling _HYCOM_CUTOVER_DATE resolves to two
        segments (see _resolve_hycom_segments) -- has_coverage must be
        checked for each, and any() confirming is enough."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        seen_keys = []

        class _FakeHycomDownloader:
            def __init__(self, output_dir):
                pass

            def has_coverage(self, dataset_key, seg_start, seg_end, clip_at_cutover=False):
                seen_keys.append(dataset_key)
                return dataset_key == "espc_d_v02"

        monkeypatch.setattr(dry_collocation, "HycomDownloader", _FakeHycomDownloader)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2024, 8, 9, 12, 0, 0), sensing_end=datetime(2024, 8, 10, 12, 0, 0),
            source_file="s1.SAFE",
        )

        assert dry_collocation._hycom_live_probe(footprint) is True
        assert sorted(seen_keys) == ["espc_d_v02", "gofs31_930"]

    def test_tz_aware_footprint_does_not_crash_resolve_hycom_segments(self, monkeypatch):
        """Regression test: _resolve_hycom_segments compares its arguments
        against the naive _HYCOM_MIN_DATE/_HYCOM_CUTOVER_DATE constants.
        Before _hycom_live_probe normalized via _to_naive_utc first, a
        tz-aware footprint (the real shape every SarFootprint from
        discover_sar_footprints_dry has) raised "can't compare
        offset-naive and offset-aware datetimes" -- found via a live
        sanity check against the real tds.hycom.org OPeNDAP endpoint."""
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        class _FakeHycomDownloader:
            def __init__(self, output_dir):
                pass

            def has_coverage(self, dataset_key, seg_start, seg_end, clip_at_cutover=False):
                assert seg_start.tzinfo is None
                assert seg_end.tzinfo is None
                return True

        monkeypatch.setattr(dry_collocation, "HycomDownloader", _FakeHycomDownloader)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2025, 1, 1, 6, 0, 0, tzinfo=timezone.utc),
            sensing_end=datetime(2025, 1, 1, 6, 1, 0, tzinfo=timezone.utc),
            source_file="s1.SAFE",
        )

        assert dry_collocation._hycom_live_probe(footprint) is True


class TestEra5LiveProbe:
    def test_uses_cfg_variable_and_footprint_day(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        captured = {}

        class _FakeERA5Downloader:
            def __init__(self, variable, output_dir):
                captured["variable"] = variable

            def check_availability_dry(self, day):
                captured["day"] = day
                return True

        monkeypatch.setattr(dry_collocation, "ERA5Downloader", _FakeERA5Downloader)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 0), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )
        cfg = SimpleNamespace(variable="wind")

        assert dry_collocation._era5_live_probe(cfg, footprint) is True
        assert captured["variable"] == "wind"
        assert captured["day"] == datetime(2026, 8, 1).date()

    def test_false_when_check_availability_dry_reports_no_data(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint

        class _FakeERA5Downloader:
            def __init__(self, variable, output_dir):
                pass

            def check_availability_dry(self, day):
                return False

        monkeypatch.setattr(dry_collocation, "ERA5Downloader", _FakeERA5Downloader)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2026, 8, 1, 6, 0, 0), sensing_end=datetime(2026, 8, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )
        cfg = SimpleNamespace(variable="soil_moisture")

        assert dry_collocation._era5_live_probe(cfg, footprint) is False


class TestPredictModelSourceRegistrations:
    """One predict_source integration test per real models-bucket
    source_type, exercising the actual _PREDICATES dispatch."""

    def test_registered_under_own_source_type_hycom(self):
        from sar_validation.core import dry_collocation

        assert "hycom" in dry_collocation._PREDICATES

    def test_registered_under_own_source_type_era5(self):
        from sar_validation.core import dry_collocation

        assert "era5" in dry_collocation._PREDICATES

    def test_hycom_collocated_via_predict_source(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint, predict_source

        monkeypatch.setattr(dry_collocation, "_hycom_live_probe", lambda fp: True)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2019, 6, 1, 6, 0, 0), sensing_end=datetime(2019, 6, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = predict_source(
            SimpleNamespace(source_type="hycom"), cfg=object(), sar_footprints=[footprint],
        )

        assert result.verdict == "collocated"
        assert result.source_type == "hycom"
        assert result.bucket == "model"

    def test_hycom_none_predicted_before_coverage_start_via_predict_source(self):
        from sar_validation.core.dry_collocation import SarFootprint, predict_source

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(2010, 1, 1, 6, 0, 0), sensing_end=datetime(2010, 1, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = predict_source(
            SimpleNamespace(source_type="hycom"), cfg=object(), sar_footprints=[footprint],
        )

        assert result.verdict == "none-predicted"

    def test_era5_collocated_via_predict_source(self, monkeypatch):
        from sar_validation.core import dry_collocation
        from sar_validation.core.dry_collocation import SarFootprint, predict_source

        monkeypatch.setattr(dry_collocation, "_era5_live_probe", lambda cfg, fp: True)

        footprint = SarFootprint(
            kind="polygon", bbox=(-10.0, 10.0, 35.0, 55.0), polygon=None, points=None,
            sensing_start=datetime(1970, 1, 1, 6, 0, 0), sensing_end=datetime(1970, 1, 1, 6, 1, 0),
            source_file="s1.SAFE",
        )

        result = predict_source(
            SimpleNamespace(source_type="era5"),
            cfg=SimpleNamespace(variable="wind"), sar_footprints=[footprint],
        )

        assert result.verdict == "collocated"
        assert result.source_type == "era5"
        assert result.bucket == "model"


class TestPredictCollocation:
    def test_aggregates_one_prediction_per_validation_source(self, monkeypatch):
        from sar_validation.core import dry_collocation

        class _FakeSource:
            def __init__(self, source_type):
                self.source_type = source_type

        class _FakeCfg:
            validation_sources = [_FakeSource("ismn"), _FakeSource("era5")]

        fake_predictions = {
            "ismn": dry_collocation.SourcePrediction(
                source_type="ismn", bucket="ground-point", verdict="collocated", detail="x"
            ),
            "era5": dry_collocation.SourcePrediction(
                source_type="era5", bucket="model", verdict="none-predicted", detail="y"
            ),
        }
        monkeypatch.setattr(
            dry_collocation, "predict_source",
            lambda source, cfg, sar_footprints, **kwargs: fake_predictions[source.source_type],
        )

        report = dry_collocation.predict_collocation(_FakeCfg(), sar_footprints=[], recipe_path="r.yaml")

        assert report.recipe_path == "r.yaml"
        assert report.sar_footprint_count == 0
        assert len(report.predictions) == 2
        assert {p.source_type for p in report.predictions} == {"ismn", "era5"}
        # Every prediction reflects the real (fake) predicate result, not a
        # silently-swallowed "unknown" from a signature mismatch between
        # predict_collocation's own stop_on_first_match forwarding and this
        # fake -- i.e. the fake was genuinely called, not caught by
        # predict_collocation's per-source try/except.
        assert {p.verdict for p in report.predictions} == {"collocated", "none-predicted"}

    def test_a_single_source_raising_becomes_unknown_not_a_crash(self, monkeypatch):
        """One source's predicate misbehaving must not take down the
        whole report -- every OTHER source's prediction must still
        appear."""
        from sar_validation.core import dry_collocation

        class _FakeSource:
            def __init__(self, source_type):
                self.source_type = source_type

        class _FakeCfg:
            validation_sources = [_FakeSource("ismn"), _FakeSource("era5")]

        def _fake_predict_source(source, cfg, sar_footprints, **kwargs):
            if source.source_type == "ismn":
                raise RuntimeError("boom")
            return dry_collocation.SourcePrediction(
                source_type="era5", bucket="model", verdict="collocated", detail="y"
            )

        monkeypatch.setattr(dry_collocation, "predict_source", _fake_predict_source)

        report = dry_collocation.predict_collocation(_FakeCfg(), sar_footprints=[])

        assert len(report.predictions) == 2
        ismn_pred = next(p for p in report.predictions if p.source_type == "ismn")
        assert ismn_pred.verdict == "unknown"

    def test_stop_on_first_match_forwarded_to_every_source(self, monkeypatch):
        """The real-run gating path (DataOrchestrator._collocation_predictions)
        calls predict_collocation(..., stop_on_first_match=True) -- confirm
        that value reaches predict_source once per configured validation
        source, not just the first."""
        from sar_validation.core import dry_collocation

        class _FakeSource:
            def __init__(self, source_type):
                self.source_type = source_type

        class _FakeCfg:
            validation_sources = [_FakeSource("ismn"), _FakeSource("era5")]

        seen = []

        def _fake_predict_source(source, cfg, sar_footprints, *, stop_on_first_match=False):
            seen.append((source.source_type, stop_on_first_match))
            return dry_collocation.SourcePrediction(
                source_type=source.source_type, bucket="x", verdict="unknown", detail="x",
            )

        monkeypatch.setattr(dry_collocation, "predict_source", _fake_predict_source)

        dry_collocation.predict_collocation(_FakeCfg(), sar_footprints=[], stop_on_first_match=True)

        assert seen == [("ismn", True), ("era5", True)]

    def test_stop_on_first_match_defaults_to_false(self, monkeypatch):
        """The --dry-collocation preview path calls predict_collocation
        without stop_on_first_match at all -- confirm that still resolves
        to False at predict_source, preserving today's exhaustive
        behavior."""
        from sar_validation.core import dry_collocation

        class _FakeSource:
            source_type = "ismn"

        class _FakeCfg:
            validation_sources = [_FakeSource()]

        seen = []

        def _fake_predict_source(source, cfg, sar_footprints, *, stop_on_first_match=False):
            seen.append(stop_on_first_match)
            return dry_collocation.SourcePrediction(
                source_type=source.source_type, bucket="x", verdict="unknown", detail="x",
            )

        monkeypatch.setattr(dry_collocation, "predict_source", _fake_predict_source)

        dry_collocation.predict_collocation(_FakeCfg(), sar_footprints=[])

        assert seen == [False]


class TestReportRendering:
    def test_console_table_includes_every_prediction(self):
        from sar_validation.core.dry_collocation import CollocationReport, SourcePrediction, render_console_table

        report = CollocationReport(
            recipe_path="r.yaml", sar_footprint_count=2,
            predictions=[
                SourcePrediction(
                    source_type="ismn", bucket="ground-point", verdict="collocated", detail="2 station(s)"
                ),
                SourcePrediction(
                    source_type="era5", bucket="model", verdict="none-predicted", detail="outside coverage"
                ),
            ],
        )

        table = render_console_table(report)

        assert "ismn" in table
        assert "collocated" in table
        assert "era5" in table
        assert "none-predicted" in table

    def test_json_round_trips_every_field(self):
        import json

        from sar_validation.core.dry_collocation import CollocationReport, SourcePrediction, report_to_json

        window_start = datetime(2026, 8, 1, 6, 0, 0)
        window_end = datetime(2026, 8, 1, 6, 5, 30)
        report = CollocationReport(
            recipe_path="r.yaml", sar_footprint_count=1,
            predictions=[
                SourcePrediction(
                    source_type="ismn", bucket="ground-point", verdict="collocated",
                    detail="1 station(s) with data in the predicted window(s).", message="see README",
                    matched_windows=[(window_start, window_end)], matched_stations=["station_a"],
                ),
            ],
        )

        parsed = json.loads(report_to_json(report))

        assert parsed["recipe_path"] == "r.yaml"
        assert parsed["sar_footprint_count"] == 1
        assert parsed["predictions"][0]["source_type"] == "ismn"
        assert parsed["predictions"][0]["message"] == "see README"
        # The datetime -> ISO-8601 serialization branch (report_to_json's
        # own _default callback) is only ever exercised when
        # matched_windows is non-empty -- pin the exact ISO-8601 strings
        # for both elements of the tuple, not just that JSON encoding
        # didn't raise.
        assert parsed["predictions"][0]["matched_windows"] == [
            [window_start.isoformat(), window_end.isoformat()]
        ]
