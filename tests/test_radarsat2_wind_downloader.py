"""Tests for the RADARSAT-2 SAR wind THREDDS downloader."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# Built via string concatenation (not a triple-quoted literal) so every
# physical line stays within this repo's 120-char ruff limit -- the real
# urlPath values themselves are already >120 chars, so they're each split
# across two adjacent literals with no inserted characters between them.
_NEW_ERA_CATALOG_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0">\n'
    '  <dataset name="SAR-Wind-HH-64N-174E_v3r0_rsat2_s202606040552510_e202606040554070_c202606041745293.nc"\n'
    '           urlPath="sar-winds/radarsat2/2026/06/'
    'SAR-Wind-HH-64N-174E_v3r0_rsat2_s202606040552510_e202606040554070_c202606041745293.nc" />\n'
    '  <dataset name="SAR-Wind-HH-65N-168W_v3r0_rsat2_s202606030441440_e202606030443000_c202606030846560.nc"\n'
    '           urlPath="sar-winds/radarsat2/2026/06/'
    'SAR-Wind-HH-65N-168W_v3r0_rsat2_s202606030441440_e202606030443000_c202606030846560.nc" />\n'
    '  <dataset name="SAR-Wind-HH-10S-30E_v3r0_rsat2_s202606100200000_e202606100201160_c202606100300000.nc"\n'
    '           urlPath="sar-winds/radarsat2/2026/06/'
    'SAR-Wind-HH-10S-30E_v3r0_rsat2_s202606100200000_e202606100201160_c202606100300000.nc" />\n'
    '</catalog>\n'
)


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


# A single-dataset catalog with a granule centered at 178W (-178), i.e.
# only 2 degrees past the 180 antimeridian edge -- used to reproduce the
# shipped wind_radarsat2_example.yaml recipe's bbox (min_lon=165,
# max_lon=180) against a candidate that should survive the coarse pad
# filter (the pad is 5 degrees) but, pre-fix, numerically cannot: the
# padded window's upper edge (180 + 5 = 185) is never reached by a
# normalized (-180..180) longitude, no matter how close to the
# antimeridian the real candidate is on the other side.
_NEAR_ANTIMERIDIAN_CATALOG_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0">\n'
    '  <dataset name="SAR-Wind-HH-64N-178W_v3r0_rsat2_s202606050552510_e202606050554070_c202606051745293.nc"\n'
    '           urlPath="sar-winds/radarsat2/2026/06/'
    'SAR-Wind-HH-64N-178W_v3r0_rsat2_s202606050552510_e202606050554070_c202606051745293.nc" />\n'
    '</catalog>\n'
)


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

    def test_pad_wraps_at_antimeridian_for_near_edge_window(self):
        """The shipped wind_radarsat2_example.yaml recipe uses a
        non-antimeridian-crossing bbox whose upper edge sits exactly at
        180 (min_lon=165, max_lon=180). A real candidate centered at
        178W (-178) is only 2 degrees past that edge -- well inside the
        5-degree pad -- but a naive `lon <= max_lon + _BBOX_PAD_DEG`
        check compares it against 185, a value no normalized longitude
        can ever reach. The coarse filter must special-case this
        wraparound so the candidate still comes through as a download
        candidate."""
        from sar_validation.downloaders.radarsat2_wind_downloader import _list_radarsat2_granules

        granules = _list_radarsat2_granules(
            _NEAR_ANTIMERIDIAN_CATALOG_XML, datetime(2026, 6, 1), datetime(2026, 6, 30),
            min_lon=165, max_lon=180, min_lat=60, max_lat=68,
        )
        assert any("64N-178W" in url for _, url, _, _ in granules)

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
# only the parts _parse_ncml_bbox actually reads. The real documents also
# carry a `location="Not provided..."` attribute on <netcdf> and a WKT
# `geospatial_bounds` polygon attribute; both are dropped here (neither
# is read by the parser) partly because they're unused and partly
# because keeping them pushes these lines past this repo's 120-char
# ruff limit.
_NEW_ERA_NCML_XML = """<?xml version="1.0" encoding="UTF-8"?>
<netcdf xmlns="http://www.unidata.ucar.edu/namespaces/netcdf/ncml-2.2">
  <attribute name="title" value="SAR_Wind_HH_64N_174E" />
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
<netcdf xmlns="http://www.unidata.ucar.edu/namespaces/netcdf/ncml-2.2">
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


class TestRadarsat2WindDownloaderDownload:
    def test_dry_run_queries_catalog_but_never_ncml_or_fileserver(self, tmp_path, capsys):
        """dry-run must answer whether real scenes exist for the requested
        bbox/time -- so it does fetch each touched month's (lightweight)
        catalog.xml -- but must never reach the per-candidate NCML check
        or a full fileServer download, and must always return []."""
        from sar_validation.downloaders.radarsat2_wind_downloader import RADARSAT2WindDownloader

        requested_urls = []

        def fake_urlopen(url, timeout=None):
            requested_urls.append(url)
            cm = MagicMock()
            cm.__enter__.return_value.read.return_value = _NEW_ERA_CATALOG_XML.encode()
            return cm

        with patch(
            "sar_validation.downloaders.radarsat2_wind_downloader.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            dl = RADARSAT2WindDownloader(output_dir=tmp_path, dry_run=True)
            out = dl.download(170, 180, 60, 68, "2026-06-01", "2026-06-30")

        assert out == []
        assert requested_urls  # the catalog.xml call did happen
        assert not any("/ncml/" in u or "/fileServer/" in u for u in requested_urls)
        assert not any(tmp_path.glob("*.nc"))
        out_text = capsys.readouterr().out
        assert "1 candidate scene" in out_text  # only 64N-174E is inside [170,180]
        assert "64N-174E" in out_text

    def test_month_404_skipped_not_raised(self, tmp_path):
        import urllib.error

        from sar_validation.downloaders.radarsat2_wind_downloader import RADARSAT2WindDownloader

        def fake_urlopen(url, timeout=None):
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

        with patch(
            "sar_validation.downloaders.radarsat2_wind_downloader.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            dl = RADARSAT2WindDownloader(output_dir=tmp_path, dry_run=False)
            out = dl.download(170, 180, 60, 68, "2014-01-01", "2014-01-31")
        assert out == []

    def test_matching_granule_downloaded_after_ncml_confirms_overlap(self, tmp_path):
        from sar_validation.downloaders.radarsat2_wind_downloader import RADARSAT2WindDownloader

        def fake_urlopen(url, timeout=None):
            cm = MagicMock()
            if "/fileServer/" in url:
                cm.__enter__.return_value.read.return_value = b"fake-netcdf-bytes"
            elif "/ncml/" in url:
                cm.__enter__.return_value.read.return_value = _NEW_ERA_NCML_XML.encode()
            else:
                cm.__enter__.return_value.read.return_value = _NEW_ERA_CATALOG_XML.encode()
            return cm

        with patch(
            "sar_validation.downloaders.radarsat2_wind_downloader.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            dl = RADARSAT2WindDownloader(output_dir=tmp_path, dry_run=False)
            out = dl.download(170, 180, 60, 68, "2026-06-01", "2026-06-30")

        assert len(out) == 1  # only 64N-174E is inside the non-wrapping [170,180] window
        assert "64N-174E" in out[0].name

    def test_fresh_download_prints_confirmation(self, tmp_path, capsys):
        """A freshly-downloaded granule must print a confirmation line, the
        same way every other downloader in this pipeline reports progress
        -- previously nothing was printed for a first-time RADARSAT-2
        download (only the "Already downloaded" cache-hit case printed)."""
        from sar_validation.downloaders.radarsat2_wind_downloader import RADARSAT2WindDownloader

        def fake_urlopen(url, timeout=None):
            cm = MagicMock()
            if "/fileServer/" in url:
                cm.__enter__.return_value.read.return_value = b"fake-netcdf-bytes"
            elif "/ncml/" in url:
                cm.__enter__.return_value.read.return_value = _NEW_ERA_NCML_XML.encode()
            else:
                cm.__enter__.return_value.read.return_value = _NEW_ERA_CATALOG_XML.encode()
            return cm

        with patch(
            "sar_validation.downloaders.radarsat2_wind_downloader.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            dl = RADARSAT2WindDownloader(output_dir=tmp_path, dry_run=False)
            out = dl.download(170, 180, 60, 68, "2026-06-01", "2026-06-30")

        assert len(out) == 1
        out_text = capsys.readouterr().out
        assert "Downloaded" in out_text
        assert "64N-174E" in out_text

    def test_date_only_end_excludes_scene_after_that_days_midnight(self, tmp_path):
        """A recipe end bound of "2026-06-05" (date-only) normalizes to
        that date's literal midnight -- the same instant normalize_datetime
        and the in-situ downloader already treat it as -- not "through the
        end of that day". A granule timestamped 2026-06-05T05:24:00 (i.e.
        after that midnight) must therefore be excluded."""
        from sar_validation.downloaders.radarsat2_wind_downloader import RADARSAT2WindDownloader

        inside_name = (
            "SAR-Wind-HH-64N-174E_v3r0_rsat2_s202606040552510_"
            "e202606040554070_c202606041745293.nc"
        )
        after_midnight_name = (
            "SAR-Wind-HH-66N-181E_v3r0_rsat2_s202606050524000_"
            "e202606050525160_c202606050613357.nc"
        )
        catalog = (
            '<?xml version="1.0"?>'
            '<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0">'
            f'<dataset name="{inside_name}" urlPath="sar-winds/radarsat2/2026/06/{inside_name}" />'
            f'<dataset name="{after_midnight_name}" '
            f'urlPath="sar-winds/radarsat2/2026/06/{after_midnight_name}" />'
            "</catalog>"
        )

        def fake_urlopen(url, timeout=None):
            cm = MagicMock()
            if "/fileServer/" in url:
                cm.__enter__.return_value.read.return_value = b"fake-netcdf-bytes"
            elif "/ncml/" in url:
                cm.__enter__.return_value.read.return_value = _NEW_ERA_NCML_XML.encode()
            else:
                cm.__enter__.return_value.read.return_value = catalog.encode()
            return cm

        with patch(
            "sar_validation.downloaders.radarsat2_wind_downloader.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            dl = RADARSAT2WindDownloader(output_dir=tmp_path, dry_run=False)
            out = dl.download(165, 180, 60, 68, "2026-06-01", "2026-06-05")

        names = {p.name for p in out}
        assert inside_name in names
        assert after_midnight_name not in names
        assert out[0].exists()
        assert out[0].read_bytes() == b"fake-netcdf-bytes"

    def test_far_away_granule_never_reaches_any_network_check(self, tmp_path):
        """The 10S/30E fixture granule is far outside the requested bbox +
        the coarse pad -- it must never even reach the NCML or fileServer
        steps."""
        from sar_validation.downloaders.radarsat2_wind_downloader import RADARSAT2WindDownloader

        requested_urls = []

        def fake_urlopen(url, timeout=None):
            requested_urls.append(url)
            cm = MagicMock()
            if "/fileServer/" in url:
                cm.__enter__.return_value.read.return_value = b"fake-netcdf-bytes"
            elif "/ncml/" in url:
                cm.__enter__.return_value.read.return_value = _NEW_ERA_NCML_XML.encode()
            else:
                cm.__enter__.return_value.read.return_value = _NEW_ERA_CATALOG_XML.encode()
            return cm

        with patch(
            "sar_validation.downloaders.radarsat2_wind_downloader.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            dl = RADARSAT2WindDownloader(output_dir=tmp_path, dry_run=False)
            dl.download(170, 180, 60, 68, "2026-06-01", "2026-06-30")

        assert not any("30E" in u for u in requested_urls)

    def test_ncml_confirms_no_overlap_skips_download_entirely(self, tmp_path):
        """A candidate that passes the coarse filename-center pre-filter
        but whose precise NCML-reported bbox does not overlap the
        requested bbox must never be downloaded at all -- no fileServer
        request issued, no file written."""
        from sar_validation.downloaders.radarsat2_wind_downloader import RADARSAT2WindDownloader

        name = (
            "SAR-Wind-HH-64N-174E_v3r0_rsat2_s202606040552510_"
            "e202606040554070_c202606041745293.nc"
        )
        catalog = (
            '<?xml version="1.0"?>'
            '<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0">'
            f'<dataset name="{name}" urlPath="sar-winds/radarsat2/2026/06/{name}" />'
            "</catalog>"
        )
        # NCML reports the real footprint as 172-176E -- it never reaches
        # the requested bbox (177-180E), even though the filename center
        # (174E) is within the coarse pad of that same requested bbox.
        ncml = (
            '<?xml version="1.0"?>'
            '<netcdf xmlns="http://www.unidata.ucar.edu/namespaces/netcdf/ncml-2.2">'
            '<attribute name="geospatial_lon_min" value="172.0" type="float" />'
            '<attribute name="geospatial_lon_max" value="176.0" type="float" />'
            '<attribute name="geospatial_lat_min" value="63.0" type="float" />'
            '<attribute name="geospatial_lat_max" value="65.0" type="float" />'
            "</netcdf>"
        )
        fileserver_calls = []

        def fake_urlopen(url, timeout=None):
            cm = MagicMock()
            if "/fileServer/" in url:
                fileserver_calls.append(url)
                cm.__enter__.return_value.read.return_value = b"fake-netcdf-bytes"
            elif "/ncml/" in url:
                cm.__enter__.return_value.read.return_value = ncml.encode()
            else:
                cm.__enter__.return_value.read.return_value = catalog.encode()
            return cm

        with patch(
            "sar_validation.downloaders.radarsat2_wind_downloader.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            dl = RADARSAT2WindDownloader(output_dir=tmp_path, dry_run=False)
            out = dl.download(177, 180, 63, 65, "2026-06-01", "2026-06-30")

        assert out == []
        assert fileserver_calls == []
        assert not any(tmp_path.glob("SAR-Wind-*.nc"))

    def test_ncml_fetch_failure_fails_open_and_still_downloads(self, tmp_path):
        """A transient NCML metadata-service error must not silently drop
        a real candidate -- the full scene is still downloaded (the cost
        of a false positive here is one unnecessary download, not a
        dropped real granule)."""
        import urllib.error

        from sar_validation.downloaders.radarsat2_wind_downloader import RADARSAT2WindDownloader

        name = (
            "SAR-Wind-HH-64N-174E_v3r0_rsat2_s202606040552510_"
            "e202606040554070_c202606041745293.nc"
        )
        catalog = (
            '<?xml version="1.0"?>'
            '<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0">'
            f'<dataset name="{name}" urlPath="sar-winds/radarsat2/2026/06/{name}" />'
            "</catalog>"
        )

        def fake_urlopen(url, timeout=None):
            if "/ncml/" in url:
                raise urllib.error.URLError("connection reset")
            cm = MagicMock()
            if "/fileServer/" in url:
                cm.__enter__.return_value.read.return_value = b"fake-netcdf-bytes"
            else:
                cm.__enter__.return_value.read.return_value = catalog.encode()
            return cm

        with patch(
            "sar_validation.downloaders.radarsat2_wind_downloader.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            dl = RADARSAT2WindDownloader(output_dir=tmp_path, dry_run=False)
            out = dl.download(170, 180, 60, 68, "2026-06-01", "2026-06-30")

        assert len(out) == 1
        assert out[0].exists()

    def test_already_downloaded_skipped_without_force(self, tmp_path):
        """An already-downloaded file is returned without any NCML or
        fileServer request -- it was already verified when first
        downloaded."""
        from sar_validation.downloaders.radarsat2_wind_downloader import RADARSAT2WindDownloader

        name = (
            "SAR-Wind-HH-64N-174E_v3r0_rsat2_s202606040552510_"
            "e202606040554070_c202606041745293.nc"
        )
        catalog = (
            '<?xml version="1.0"?>'
            '<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0">'
            f'<dataset name="{name}" urlPath="sar-winds/radarsat2/2026/06/{name}" />'
            "</catalog>"
        )
        existing = tmp_path / name
        existing.write_bytes(b"already-here")

        def fake_urlopen(url, timeout=None):
            assert "/fileServer/" not in url and "/ncml/" not in url  # neither should be reached
            cm = MagicMock()
            cm.__enter__.return_value.read.return_value = catalog.encode()
            return cm

        with patch(
            "sar_validation.downloaders.radarsat2_wind_downloader.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            dl = RADARSAT2WindDownloader(output_dir=tmp_path, dry_run=False, force_download=False)
            out = dl.download(170, 180, 60, 68, "2026-06-01", "2026-06-30")

        assert existing in out

    def test_network_calls_use_prefer_ipv4_dns(self, tmp_path):
        from sar_validation.downloaders.radarsat2_wind_downloader import RADARSAT2WindDownloader

        def fake_urlopen(url, timeout=None):
            cm = MagicMock()
            if "/fileServer/" in url:
                cm.__enter__.return_value.read.return_value = b"fake-netcdf-bytes"
            elif "/ncml/" in url:
                cm.__enter__.return_value.read.return_value = _NEW_ERA_NCML_XML.encode()
            else:
                cm.__enter__.return_value.read.return_value = _NEW_ERA_CATALOG_XML.encode()
            return cm

        with patch(
            "sar_validation.downloaders.radarsat2_wind_downloader.prefer_ipv4_dns"
        ) as mock_prefer:
            mock_prefer.return_value.__exit__.return_value = False
            with patch(
                "sar_validation.downloaders.radarsat2_wind_downloader.urllib.request.urlopen",
                side_effect=fake_urlopen,
            ):
                dl = RADARSAT2WindDownloader(output_dir=tmp_path, dry_run=False)
                dl.download(170, 180, 60, 68, "2026-06-01", "2026-06-30")

        # 1 catalog fetch + 1 NCML fetch + 1 fileServer download (only
        # 64N-174E is inside the [170,180] window)
        assert mock_prefer.call_count == 3

    def test_found_count_set_from_catalog_even_in_dry_run(self, tmp_path):
        from sar_validation.downloaders.radarsat2_wind_downloader import RADARSAT2WindDownloader

        def fake_urlopen(url, timeout=None):
            cm = MagicMock()
            cm.__enter__.return_value.read.return_value = _NEW_ERA_CATALOG_XML.encode()
            return cm

        with patch(
            "sar_validation.downloaders.radarsat2_wind_downloader.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            dl = RADARSAT2WindDownloader(output_dir=tmp_path, dry_run=True)
            out = dl.download(170, 180, 60, 68, "2026-06-01", "2026-06-30")

        assert out == []
        assert dl.found_count == 1  # only 64N-174E is inside [170,180]

    def test_found_count_zero_when_month_404s(self, tmp_path):
        import urllib.error

        from sar_validation.downloaders.radarsat2_wind_downloader import RADARSAT2WindDownloader

        def fake_urlopen(url, timeout=None):
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

        with patch(
            "sar_validation.downloaders.radarsat2_wind_downloader.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            dl = RADARSAT2WindDownloader(output_dir=tmp_path, dry_run=False)
            dl.download(170, 180, 60, 68, "2014-01-01", "2014-01-31")

        assert dl.found_count == 0
