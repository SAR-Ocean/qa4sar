"""Tests for the shared NOAA HF-radar region table (_noaa_hfr_regions)."""

from __future__ import annotations

import pytest


class TestRegionTableShape:
    def test_has_six_regions(self):
        from sar_validation.downloaders._noaa_hfr_regions import NOAA_HFR_REGIONS

        assert set(NOAA_HFR_REGIONS) == {
            "US_WEST", "US_EAST_GULF", "US_HAWAII", "US_PRVI",
            "US_GREAT_LAKES", "US_GULF_OF_ALASKA",
        }

    def test_thredds_codes(self):
        from sar_validation.downloaders._noaa_hfr_regions import NOAA_HFR_REGIONS

        assert NOAA_HFR_REGIONS["US_WEST"]["thredds_code"] == "USWC"
        assert NOAA_HFR_REGIONS["US_EAST_GULF"]["thredds_code"] == "USEGC"
        assert NOAA_HFR_REGIONS["US_HAWAII"]["thredds_code"] == "USHI"
        assert NOAA_HFR_REGIONS["US_PRVI"]["thredds_code"] == "PRVI"
        assert NOAA_HFR_REGIONS["US_GREAT_LAKES"]["thredds_code"] == "GLNA"
        assert NOAA_HFR_REGIONS["US_GULF_OF_ALASKA"]["thredds_code"] == "GAK"

    def test_great_lakes_and_gulf_of_alaska_have_no_erddap_dataset(self):
        from sar_validation.downloaders._noaa_hfr_regions import NOAA_HFR_REGIONS

        assert NOAA_HFR_REGIONS["US_GREAT_LAKES"]["erddap_datasets"] is None
        assert NOAA_HFR_REGIONS["US_GULF_OF_ALASKA"]["erddap_datasets"] is None

    def test_us_west_erddap_datasets_include_500m(self):
        from sar_validation.downloaders._noaa_hfr_regions import NOAA_HFR_REGIONS

        datasets = NOAA_HFR_REGIONS["US_WEST"]["erddap_datasets"]
        assert datasets[0.5] == "ucsdHfrW500"
        assert datasets[1] == "ucsdHfrW1"
        assert datasets[2] == "ucsdHfrW2"
        assert datasets[6] == "ucsdHfrW6"

    def test_us_hawaii_erddap_datasets_is_1km_only(self):
        from sar_validation.downloaders._noaa_hfr_regions import NOAA_HFR_REGIONS

        assert NOAA_HFR_REGIONS["US_HAWAII"]["erddap_datasets"] == {1: "ucsdHfrH1"}

    def test_default_resolution_km_is_6_except_hawaii(self):
        from sar_validation.downloaders._noaa_hfr_regions import NOAA_HFR_REGIONS

        for name, region in NOAA_HFR_REGIONS.items():
            expected = 1 if name == "US_HAWAII" else 6
            assert region["default_resolution_km"] == expected, name


class TestMatchNoaaHfrRegion:
    def test_us_west_center_matches(self):
        from sar_validation.downloaders._noaa_hfr_regions import match_noaa_hfr_region

        name, region = match_noaa_hfr_region(-125.0, -119.0, 33.0, 38.0)
        assert name == "US_WEST"

    def test_hawaii_center_matches(self):
        from sar_validation.downloaders._noaa_hfr_regions import match_noaa_hfr_region

        name, _ = match_noaa_hfr_region(-159.0, -154.0, 19.0, 22.0)
        assert name == "US_HAWAII"

    def test_great_lakes_center_matches(self):
        from sar_validation.downloaders._noaa_hfr_regions import match_noaa_hfr_region

        name, _ = match_noaa_hfr_region(-85.2, -84.3, 45.7, 45.95)
        assert name == "US_GREAT_LAKES"

    def test_gulf_of_alaska_center_matches(self):
        from sar_validation.downloaders._noaa_hfr_regions import match_noaa_hfr_region

        name, _ = match_noaa_hfr_region(-155.0, -145.0, 55.0, 60.0)
        assert name == "US_GULF_OF_ALASKA"

    def test_no_match_raises_value_error(self):
        from sar_validation.downloaders._noaa_hfr_regions import match_noaa_hfr_region

        with pytest.raises(ValueError):
            match_noaa_hfr_region(2.0, 8.0, 53.0, 55.0)  # German Bight


class TestRegionBboxOverlaps:
    def test_bbox_fully_inside_region_overlaps(self):
        from sar_validation.downloaders._noaa_hfr_regions import NOAA_HFR_REGIONS, region_bbox_overlaps

        assert region_bbox_overlaps(NOAA_HFR_REGIONS["US_WEST"], -125.0, -119.0, 33.0, 38.0)

    def test_bbox_partially_overlapping_region_edge_overlaps(self):
        from sar_validation.downloaders._noaa_hfr_regions import NOAA_HFR_REGIONS, region_bbox_overlaps

        # Straddles US_WEST's western edge (-130.36) without its center being inside.
        assert region_bbox_overlaps(NOAA_HFR_REGIONS["US_WEST"], -140.0, -128.0, 33.0, 38.0)

    def test_bbox_entirely_outside_region_does_not_overlap(self):
        from sar_validation.downloaders._noaa_hfr_regions import NOAA_HFR_REGIONS, region_bbox_overlaps

        assert not region_bbox_overlaps(NOAA_HFR_REGIONS["US_WEST"], 2.0, 8.0, 53.0, 55.0)


class TestFinestResolutionKm:
    def test_us_west_finest_is_500m(self):
        from sar_validation.downloaders._noaa_hfr_regions import NOAA_HFR_REGIONS, finest_resolution_km

        assert finest_resolution_km(NOAA_HFR_REGIONS["US_WEST"]) == 0.5

    def test_us_great_lakes_finest_is_500m(self):
        from sar_validation.downloaders._noaa_hfr_regions import NOAA_HFR_REGIONS, finest_resolution_km

        assert finest_resolution_km(NOAA_HFR_REGIONS["US_GREAT_LAKES"]) == 0.5

    def test_us_east_gulf_finest_is_1km(self):
        from sar_validation.downloaders._noaa_hfr_regions import NOAA_HFR_REGIONS, finest_resolution_km

        assert finest_resolution_km(NOAA_HFR_REGIONS["US_EAST_GULF"]) == 1

    def test_us_hawaii_finest_is_1km(self):
        from sar_validation.downloaders._noaa_hfr_regions import NOAA_HFR_REGIONS, finest_resolution_km

        assert finest_resolution_km(NOAA_HFR_REGIONS["US_HAWAII"]) == 1

    def test_us_prvi_finest_is_2km(self):
        from sar_validation.downloaders._noaa_hfr_regions import NOAA_HFR_REGIONS, finest_resolution_km

        assert finest_resolution_km(NOAA_HFR_REGIONS["US_PRVI"]) == 2

    def test_us_gulf_of_alaska_finest_is_2km(self):
        from sar_validation.downloaders._noaa_hfr_regions import NOAA_HFR_REGIONS, finest_resolution_km

        assert finest_resolution_km(NOAA_HFR_REGIONS["US_GULF_OF_ALASKA"]) == 2


class TestResolutionToken:
    def test_500m(self):
        from sar_validation.downloaders._noaa_hfr_regions import _resolution_token

        assert _resolution_token(0.5) == "500m"

    def test_whole_km_values(self):
        from sar_validation.downloaders._noaa_hfr_regions import _resolution_token

        assert _resolution_token(1) == "1km"
        assert _resolution_token(2.0) == "2km"
        assert _resolution_token(6) == "6km"
