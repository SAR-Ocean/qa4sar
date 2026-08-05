"""Tests for the shared NOAA HF-radar region table (_noaa_hfr_regions)."""

from __future__ import annotations

import pytest


class TestMatchNoaaHfrRegion:
    @pytest.mark.parametrize(
        "min_lon,max_lon,min_lat,max_lat,expected_region",
        [
            (-125.0, -119.0, 33.0, 38.0, "US_WEST"),
            (-159.0, -154.0, 19.0, 22.0, "US_HAWAII"),
            (-85.2, -84.3, 45.7, 45.95, "US_GREAT_LAKES"),
            (-155.0, -145.0, 55.0, 60.0, "US_GULF_OF_ALASKA"),
        ],
        ids=["us_west", "hawaii", "great_lakes", "gulf_of_alaska"],
    )
    def test_center_matches(self, min_lon, max_lon, min_lat, max_lat, expected_region):
        from sar_validation.downloaders._noaa_hfr_regions import match_noaa_hfr_region

        name, _ = match_noaa_hfr_region(min_lon, max_lon, min_lat, max_lat)
        assert name == expected_region

    def test_no_match_raises_value_error(self):
        from sar_validation.downloaders._noaa_hfr_regions import match_noaa_hfr_region

        with pytest.raises(ValueError):
            match_noaa_hfr_region(2.0, 8.0, 53.0, 55.0)  # German Bight


class TestRegionBboxOverlaps:
    @pytest.mark.parametrize(
        "min_lon,max_lon,min_lat,max_lat,expected_overlap",
        [
            (-125.0, -119.0, 33.0, 38.0, True),
            # Straddles US_WEST's western edge (-130.36) without its center being inside.
            (-140.0, -128.0, 33.0, 38.0, True),
            (2.0, 8.0, 53.0, 55.0, False),
        ],
        ids=["fully_inside", "partially_overlapping_edge", "entirely_outside"],
    )
    def test_overlap(self, min_lon, max_lon, min_lat, max_lat, expected_overlap):
        from sar_validation.downloaders._noaa_hfr_regions import NOAA_HFR_REGIONS, region_bbox_overlaps

        result = region_bbox_overlaps(NOAA_HFR_REGIONS["US_WEST"], min_lon, max_lon, min_lat, max_lat)
        assert result == expected_overlap


class TestResolutionToken:
    @pytest.mark.parametrize(
        "resolution_km,expected_token",
        [
            (0.5, "500m"),
            (1, "1km"),
            (2.0, "2km"),
            (6, "6km"),
        ],
    )
    def test_resolution_token(self, resolution_km, expected_token):
        from sar_validation.downloaders._noaa_hfr_regions import _resolution_token

        assert _resolution_token(resolution_km) == expected_token
