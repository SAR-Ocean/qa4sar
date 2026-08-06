"""Tests for the RADARSAT-2 SAR wind THREDDS downloader."""

from __future__ import annotations

from datetime import datetime

import pytest

_NEW_ERA_CATALOG_XML = """<?xml version="1.0" encoding="UTF-8"?>
<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0">
  <dataset name="SAR-Wind-HH-64N-174E_v3r0_rsat2_s202606040552510_e202606040554070_c202606041745293.nc"
           urlPath="sar-winds/radarsat2/2026/06/SAR-Wind-HH-64N-174E_v3r0_rsat2_s202606040552510_e202606040554070_c202606041745293.nc" />
  <dataset name="SAR-Wind-HH-65N-168W_v3r0_rsat2_s202606030441440_e202606030443000_c202606030846560.nc"
           urlPath="sar-winds/radarsat2/2026/06/SAR-Wind-HH-65N-168W_v3r0_rsat2_s202606030441440_e202606030443000_c202606030846560.nc" />
  <dataset name="SAR-Wind-HH-10S-30E_v3r0_rsat2_s202606100200000_e202606100201160_c202606100300000.nc"
           urlPath="sar-winds/radarsat2/2026/06/SAR-Wind-HH-10S-30E_v3r0_rsat2_s202606100200000_e202606100201160_c202606100300000.nc" />
</catalog>
"""


class TestParseGranuleNameOldEra:
    def test_west_north(self):
        from sar_validation.downloaders.radarsat2_wind_downloader import _parse_granule_name

        ts, lon, lat = _parse_granule_name(
            "RSAT2_GSS_2019_06_01_02_01_52_0612669712_131.54W_71.53N_HH_C5_GFS05CDF_wind_level2_norcs.nc"
        )
        assert ts == datetime(2019, 6, 1, 2, 1, 52)
        assert lon == pytest.approx(-131.54)
        assert lat == pytest.approx(71.53)

    def test_east_south(self):
        from sar_validation.downloaders.radarsat2_wind_downloader import _parse_granule_name

        ts, lon, lat = _parse_granule_name(
            "RSAT2_KSAT_2023_06_16_10_00_00_0612669999_142.65E_34.43S_HH_C5_FIXED_wind_level2_norcs.nc"
        )
        assert ts == datetime(2023, 6, 16, 10, 0, 0)
        assert lon == pytest.approx(142.65)
        assert lat == pytest.approx(-34.43)

    def test_lon_over_180_normalized(self):
        from sar_validation.downloaders.radarsat2_wind_downloader import _parse_granule_name

        _, lon, _ = _parse_granule_name(
            "RSAT2_GSS_2019_06_01_05_23_12_0612681792_185.56E_71.31N_HH_C5_GFS05CDF_wind_level2_norcs.nc"
        )
        assert lon == pytest.approx(-174.44)  # 185.56 - 360


class TestParseGranuleNameNewEra:
    def test_north_east(self):
        from sar_validation.downloaders.radarsat2_wind_downloader import _parse_granule_name

        ts, lon, lat = _parse_granule_name(
            "SAR-Wind-HH-64N-174E_v3r0_rsat2_s202606040552510_e202606040554070_c202606041745293.nc"
        )
        assert ts == datetime(2026, 6, 4, 5, 52, 51)
        assert lon == pytest.approx(174.0)
        assert lat == pytest.approx(64.0)

    def test_south_west(self):
        from sar_validation.downloaders.radarsat2_wind_downloader import _parse_granule_name

        _, lon, lat = _parse_granule_name(
            "SAR-Wind-HH-45S-120W_v2r0_rsat2_s202401150430000_e202401150431160_c202401150600000.nc"
        )
        assert lon == pytest.approx(-120.0)
        assert lat == pytest.approx(-45.0)


class TestParseGranuleNameUnmatched:
    def test_unrelated_filename_returns_none(self):
        from sar_validation.downloaders.radarsat2_wind_downloader import _parse_granule_name

        assert _parse_granule_name("readme.txt") is None


class TestListRadarsat2GranulesBboxFilter:
    def test_keeps_only_candidates_within_padded_bbox(self):
        from sar_validation.downloaders.radarsat2_wind_downloader import _list_radarsat2_granules

        granules = _list_radarsat2_granules(
            _NEW_ERA_CATALOG_XML, datetime(2026, 6, 1), datetime(2026, 6, 30),
            min_lon=170, max_lon=180, min_lat=60, max_lat=68,
        )
        urls = [url for _, url, _, _ in granules]
        # Only 64N-174E is inside this non-wrapping window -- 65N-168W's
        # signed center (-168) is on the other side of the antimeridian
        # (see test_pad_lets_a_near_but_outside_center_through below for
        # that candidate's own dedicated, valid window) and 10S-30E is
        # simply far away.
        assert len(urls) == 1
        assert "64N-174E" in urls[0]

    def test_time_window_excludes_out_of_range_granule(self):
        from sar_validation.downloaders.radarsat2_wind_downloader import _list_radarsat2_granules

        granules = _list_radarsat2_granules(
            _NEW_ERA_CATALOG_XML, datetime(2026, 6, 4, 0, 0), datetime(2026, 6, 4, 23, 59),
            min_lon=170, max_lon=180, min_lat=60, max_lat=68,
        )
        assert len(granules) == 1
        assert "64N-174E" in granules[0][1]

    def test_pad_lets_a_near_but_outside_center_through(self):
        """65N/168W's signed center is -168, which is outside the raw bbox
        [-179, -170] by ~2 degrees -- but within the 5-degree pad, so it
        must still come through as a download candidate (Task 3's precise
        NCML-based check is what makes the final keep/drop call, not this
        coarse pre-filter)."""
        from sar_validation.downloaders.radarsat2_wind_downloader import _list_radarsat2_granules

        granules = _list_radarsat2_granules(
            _NEW_ERA_CATALOG_XML, datetime(2026, 6, 1), datetime(2026, 6, 30),
            min_lon=-179, max_lon=-170, min_lat=60, max_lat=68,
        )
        assert any("65N-168W" in url for _, url, _, _ in granules)

    def test_malformed_filename_skipped_not_crashed(self):
        from sar_validation.downloaders.radarsat2_wind_downloader import _list_radarsat2_granules

        catalog = (
            '<?xml version="1.0"?>'
            '<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0">'
            '<dataset name="unrelated_file.nc" urlPath="sar-winds/radarsat2/2026/06/unrelated_file.nc" />'
            "</catalog>"
        )
        granules = _list_radarsat2_granules(
            catalog, datetime(2026, 6, 1), datetime(2026, 6, 30),
            min_lon=170, max_lon=180, min_lat=60, max_lat=68,
        )
        assert granules == []

    def test_end_exclusive_excludes_boundary_granule(self):
        from sar_validation.downloaders.radarsat2_wind_downloader import _list_radarsat2_granules

        granules = _list_radarsat2_granules(
            _NEW_ERA_CATALOG_XML, datetime(2026, 6, 1), datetime(2026, 6, 4),
            min_lon=170, max_lon=180, min_lat=60, max_lat=68,
            end_exclusive=True,
        )
        assert not any("64N-174E" in url for _, url, _, _ in granules)  # timestamped 2026-06-04 05:52:51


# Trimmed fragments of real NCML documents (fetched live 2026-08-05) --
# only the parts _parse_ncml_bbox actually reads.
_NEW_ERA_NCML_XML = """<?xml version="1.0" encoding="UTF-8"?>
<netcdf xmlns="http://www.unidata.ucar.edu/namespaces/netcdf/ncml-2.2" location="Not provided because of security concerns.">
  <attribute name="title" value="SAR_Wind_HH_64N_174E" />
  <attribute name="geospatial_bounds" value="POLYGON((170.64201 60.946674, 180.30565 61.870308, 178.9892 66.32337, 167.73218 65.3223, 170.64201 60.946674))" />
  <attribute name="geospatial_lat_max" type="float" value="66.32337" />
  <attribute name="geospatial_lat_min" type="float" value="60.946674" />
  <attribute name="geospatial_lon_max" type="float" value="180.30565" />
  <attribute name="geospatial_lon_min" type="float" value="167.73218" />
  <group name="NCISOMetadata">
    <attribute name="geospatial_lon_min" value="167.73218" type="float" />
    <attribute name="geospatial_lat_min" value="60.95103" type="float" />
    <attribute name="geospatial_lon_max" value="180.29509" type="float" />
    <attribute name="geospatial_lat_max" value="66.32278" type="float" />
  </group>
</netcdf>
"""

_OLD_ERA_NCML_XML = """<?xml version="1.0" encoding="UTF-8"?>
<netcdf xmlns="http://www.unidata.ucar.edu/namespaces/netcdf/ncml-2.2" location="Not provided because of security concerns.">
  <attribute name="title" value="SAR_Wind" />
  <group name="CFMetadata">
    <attribute name="geospatial_lon_min" value="-141.28207" type="float" />
    <attribute name="geospatial_lat_min" value="68.67533" type="float" />
    <attribute name="geospatial_lon_max" value="-123.27931" type="float" />
    <attribute name="geospatial_lat_max" value="74.19328" type="float" />
  </group>
</netcdf>
"""


class TestParseNcmlBbox:
    def test_new_era_uses_root_level_attributes(self):
        from sar_validation.downloaders.radarsat2_wind_downloader import _parse_ncml_bbox

        bbox = _parse_ncml_bbox(_NEW_ERA_NCML_XML)
        assert bbox == pytest.approx((167.73218, 180.30565, 60.946674, 66.32337))

    def test_old_era_falls_back_to_cfmetadata_group(self):
        """Old-era raw files carry no geospatial_*_min/max attributes at
        all (confirmed live), but THREDDS' NCML service still computes
        and reports them, nested under <group name="CFMetadata">."""
        from sar_validation.downloaders.radarsat2_wind_downloader import _parse_ncml_bbox

        bbox = _parse_ncml_bbox(_OLD_ERA_NCML_XML)
        assert bbox == pytest.approx((-141.28207, -123.27931, 68.67533, 74.19328))

    def test_missing_attributes_returns_none(self):
        from sar_validation.downloaders.radarsat2_wind_downloader import _parse_ncml_bbox

        ncml = (
            '<?xml version="1.0"?>'
            '<netcdf xmlns="http://www.unidata.ucar.edu/namespaces/netcdf/ncml-2.2">'
            '<attribute name="title" value="no bbox here" />'
            "</netcdf>"
        )
        assert _parse_ncml_bbox(ncml) is None
