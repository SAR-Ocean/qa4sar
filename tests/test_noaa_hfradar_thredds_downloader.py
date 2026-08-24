"""Tests for the NOAA THREDDS archive HF-radar downloader."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

_OLD_ERA_CATALOG_XML = """<?xml version="1.0" encoding="UTF-8"?>
<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0">
  <dataset name="202401312300_hfr_uswc_6km_rtv_uwls_SIO.nc"
           urlPath="ioos/hfradar/rtv/2024/202401/USWC/202401312300_hfr_uswc_6km_rtv_uwls_SIO.nc" />
  <dataset name="202401312300_hfr_uswc_6km_rtv_uwls_NDBC.nc"
           urlPath="ioos/hfradar/rtv/2024/202401/USWC/202401312300_hfr_uswc_6km_rtv_uwls_NDBC.nc" />
  <dataset name="202401312300_hfr_uswc_6km_rtv_uwls_25hr_average_SIO.nc"
           urlPath="ioos/hfradar/rtv/2024/202401/USWC/202401312300_hfr_uswc_6km_rtv_uwls_25hr_average_SIO.nc" />
  <dataset name="202401312200_hfr_uswc_6km_rtv_uwls_NDBC.nc"
           urlPath="ioos/hfradar/rtv/2024/202401/USWC/202401312200_hfr_uswc_6km_rtv_uwls_NDBC.nc" />
  <dataset name="202401312300_hfr_uswc_1km_rtv_uwls_NDBC.nc"
           urlPath="ioos/hfradar/rtv/2024/202401/USWC/202401312300_hfr_uswc_1km_rtv_uwls_NDBC.nc" />
  <dataset name="202402010000_hfr_uswc_6km_rtv_uwls_NDBC.nc"
           urlPath="ioos/hfradar/rtv/2024/202402/USWC/202402010000_hfr_uswc_6km_rtv_uwls_NDBC.nc" />
</catalog>
"""

_NEW_ERA_CATALOG_XML = """<?xml version="1.0" encoding="UTF-8"?>
<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0">
  <dataset name="rtv-uswc-6km-uwls_v1r0_hfr_s202606302300000_e202606302300000_c202607011730210.nc"
    urlPath="rtv/2026/202606/USWC/rtv-uswc-6km-uwls_v1r0_hfr_s202606302300000_e202606302300000_c202607011730210.nc" />
  <dataset name="rtv-uswc-6km-uwls_v1r0_hfr_s202606302200000_e202606302200000_c202607010100213.nc"
    urlPath="rtv/2026/202606/USWC/rtv-uswc-6km-uwls_v1r0_hfr_s202606302200000_e202606302200000_c202607010100213.nc" />
  <dataset name="rtv-uswc-500m-uwls_v1r0_hfr_s202606302300000_e202606302300000_c202607011730210.nc"
    urlPath="rtv/2026/202606/USWC/rtv-uswc-500m-uwls_v1r0_hfr_s202606302300000_e202606302300000_c202607011730210.nc" />
</catalog>
"""


class TestListThreddsGranules:
    @pytest.mark.parametrize(
        "catalog,resolution,start,end,end_exclusive,expected_count,expected_check",
        [
            pytest.param(
                _OLD_ERA_CATALOG_XML, 6,
                datetime(2024, 1, 31, 0, 0), datetime(2024, 1, 31, 23, 59), False,
                2,  # the two 6km _NDBC entries (22:00 and 23:00), not SIO/25hr/1km
                lambda granules: (
                    all(u.endswith("_NDBC.nc") for _, u in granules)
                    and all("6km" in u for _, u in granules)
                ),
                id="keeps_only_ndbc_at_requested_resolution",
            ),
            pytest.param(
                _OLD_ERA_CATALOG_XML, 6,
                datetime(2024, 1, 31, 23, 0), datetime(2024, 1, 31, 23, 0), False,
                1,
                lambda granules: (
                    granules[0][0] == datetime(2024, 1, 31, 23, 0)
                    and "202401312300" in granules[0][1]
                ),
                id="timestamp_parsed_from_leading_12_digits",
            ),
            pytest.param(
                _OLD_ERA_CATALOG_XML, 6,
                datetime(2024, 1, 31, 22, 30), datetime(2024, 1, 31, 23, 30), False,
                1,  # only 23:00 is inside [22:30, 23:30]; 22:00 is excluded
                None,
                id="window_filters_out_of_range_granule",
            ),
            pytest.param(
                # Simulates _download_window's date-only-end handling: a
                # request for end="2024-01-31" widens the matching bound to
                # datetime(2024, 2, 1, 0, 0) so the whole of Jan 31 is
                # covered, but with end_exclusive=True the Feb 1 00:00
                # granule itself must not leak in.
                _OLD_ERA_CATALOG_XML, 6,
                datetime(2024, 1, 31, 0, 0), datetime(2024, 2, 1, 0, 0), True,
                2,  # the two 6km _NDBC entries from Jan 31 (22:00, 23:00)
                lambda granules: not any("202402010000" in u for _, u in granules),
                id="end_exclusive_excludes_next_day_midnight_granule",
            ),
            pytest.param(
                # Default (end_exclusive=False) behavior is unchanged: the
                # same widened bound now inclusively matches the Feb 1 00:00
                # granule too.
                _OLD_ERA_CATALOG_XML, 6,
                datetime(2024, 1, 31, 0, 0), datetime(2024, 2, 1, 0, 0), False,
                3,  # the two Jan 31 6km _NDBC entries plus Feb 1 00:00
                lambda granules: any("202402010000" in u for _, u in granules),
                id="end_inclusive_default_includes_next_day_midnight_granule",
            ),
            pytest.param(
                _NEW_ERA_CATALOG_XML, 6,
                datetime(2026, 6, 30, 0, 0), datetime(2026, 6, 30, 23, 59), False,
                2,
                lambda granules: all("6km" in u and "500m" not in u for _, u in granules),
                id="keeps_only_requested_resolution",
            ),
            pytest.param(
                _NEW_ERA_CATALOG_XML, 6,
                datetime(2026, 6, 30, 23, 0), datetime(2026, 6, 30, 23, 0), False,
                1,
                lambda granules: granules[0][0] == datetime(2026, 6, 30, 23, 0),
                id="timestamp_parsed_from_s_token",
            ),
        ],
    )
    def test_list_thredds_granules(
        self, catalog, resolution, start, end, end_exclusive, expected_count, expected_check,
    ):
        from sar_validation.downloaders.noaa_hfradar_thredds_downloader import _list_thredds_granules

        granules = _list_thredds_granules(catalog, resolution, start, end, end_exclusive=end_exclusive)
        assert len(granules) == expected_count
        if expected_check is not None:
            assert expected_check(granules)


def _make_thredds_nc(
    tmp_path, name, n_lat=3, n_lon=4, lat_bounds=(33.0, 38.0), lon_bounds=(-125.0, -119.0),
):
    """Build a synthetic THREDDS granule. Includes non-gridded variables
    (time_bnds(time, nv), wgs84(time,) scalar-per-time int8) alongside the
    gridded u/v, matching real NOAA THREDDS granules -- these are what
    .where(..., drop=True) used to broadcast/upcast incorrectly (see
    test_merged_output_trimmed_to_request_bbox)."""
    rng = np.random.default_rng(3)
    lats = np.linspace(lat_bounds[0], lat_bounds[1], n_lat)
    lons = np.linspace(lon_bounds[0], lon_bounds[1], n_lon)
    shape = (1, n_lat, n_lon)
    ds = xr.Dataset(
        {
            "u": (("time", "lat", "lon"), rng.uniform(-0.5, 0.5, shape)),
            "v": (("time", "lat", "lon"), rng.uniform(-0.5, 0.5, shape)),
            "time_bnds": (("time", "nv"), np.zeros((1, 2))),
            "wgs84": (("time",), np.zeros((1,), dtype="int8")),
        },
        coords={"time": [0], "nv": [0, 1], "lat": lats, "lon": lons},
    )
    path = tmp_path / name
    ds.to_netcdf(path)
    return path


class TestNoaaThreddsHfRadarDownloaderCheckAvailabilityDry:
    def test_true_when_a_granule_matches(self, tmp_path):
        from sar_validation.downloaders.noaa_hfradar_thredds_downloader import (
            NOAATHREDDSHFRadarDownloader,
        )

        def fake_urlopen(url, timeout=None):
            cm = MagicMock()
            cm.__enter__.return_value.read.return_value = _OLD_ERA_CATALOG_XML.encode()
            return cm

        dl = NOAATHREDDSHFRadarDownloader(output_dir=tmp_path, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_thredds_downloader.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ) as m:
            result = dl.check_availability_dry(-125, -119, 33, 38, "2024-01-31", "2024-01-31")

        assert result is True
        # Only the catalog.xml was fetched -- never a fileServer granule.
        assert not any("/fileServer/" in c.args[0] for c in m.call_args_list)
        assert not any(tmp_path.glob("*.nc"))

    def test_false_when_catalog_reachable_but_no_granule_matches(self, tmp_path):
        from sar_validation.downloaders.noaa_hfradar_thredds_downloader import (
            NOAATHREDDSHFRadarDownloader,
        )

        empty_catalog = (
            '<?xml version="1.0"?>'
            '<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"></catalog>'
        )

        def fake_urlopen(url, timeout=None):
            cm = MagicMock()
            cm.__enter__.return_value.read.return_value = empty_catalog.encode()
            return cm

        dl = NOAATHREDDSHFRadarDownloader(output_dir=tmp_path, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_thredds_downloader.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            result = dl.check_availability_dry(-125, -119, 33, 38, "2024-01-31", "2024-01-31")

        assert result is False

    def test_false_when_every_month_404s(self, tmp_path):
        import urllib.error

        from sar_validation.downloaders.noaa_hfradar_thredds_downloader import (
            NOAATHREDDSHFRadarDownloader,
        )

        def fake_urlopen(url, timeout=None):
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

        dl = NOAATHREDDSHFRadarDownloader(output_dir=tmp_path, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_thredds_downloader.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            result = dl.check_availability_dry(-125, -119, 33, 38, "2024-01-31", "2024-01-31")

        assert result is False

    def test_no_matching_region_raises_value_error(self, tmp_path):
        from sar_validation.downloaders.noaa_hfradar_thredds_downloader import (
            NOAATHREDDSHFRadarDownloader,
        )

        dl = NOAATHREDDSHFRadarDownloader(output_dir=tmp_path, resolution_km=6)
        with pytest.raises(ValueError):
            dl.check_availability_dry(2.0, 8.0, 53.0, 55.0, "2024-01-31", "2024-01-31")


class TestNoaaThreddsHfRadarDownloaderDownload:
    def test_no_match_raises_value_error(self, tmp_path):
        from sar_validation.downloaders.noaa_hfradar_thredds_downloader import (
            NOAATHREDDSHFRadarDownloader,
        )

        dl = NOAATHREDDSHFRadarDownloader(output_dir=tmp_path, dry_run=True)
        with pytest.raises(ValueError):
            dl.download(2.0, 8.0, 53.0, 55.0, "2024-01-31", "2024-01-31")

    def test_dry_run_queries_catalog_but_never_downloads(self, tmp_path, capsys):
        """dry-run must answer whether real granules exist for the
        requested bbox/time -- so it does fetch each touched month's
        (lightweight) catalog.xml -- but must never reach a full
        fileServer download, and must always return []."""
        from sar_validation.downloaders.noaa_hfradar_thredds_downloader import (
            NOAATHREDDSHFRadarDownloader,
        )

        requested_urls = []

        def fake_urlopen(url, timeout=None):
            requested_urls.append(url)
            cm = MagicMock()
            cm.__enter__.return_value.read.return_value = _OLD_ERA_CATALOG_XML.encode()
            return cm

        dl = NOAATHREDDSHFRadarDownloader(output_dir=tmp_path, dry_run=True, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_thredds_downloader.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            out = dl.download(-125, -119, 33, 38, "2024-01-31", "2024-01-31")

        assert out == []
        assert requested_urls  # the catalog.xml call did happen
        assert not any("/fileServer/" in u for u in requested_urls)
        assert not any(tmp_path.glob("*.nc"))
        out_text = capsys.readouterr().out
        assert "granule(s) matched" in out_text

    def test_no_matching_granules_but_catalog_reachable_returns_empty_list(self, tmp_path):
        from sar_validation.downloaders.noaa_hfradar_thredds_downloader import (
            NOAATHREDDSHFRadarDownloader,
        )

        empty_catalog = (
            '<?xml version="1.0"?>'
            '<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"></catalog>'
        )

        def fake_urlopen(url, timeout=None):
            cm = MagicMock()
            cm.__enter__.return_value.read.return_value = empty_catalog.encode()
            return cm

        with patch(
            "sar_validation.downloaders.noaa_hfradar_thredds_downloader.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            dl = NOAATHREDDSHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
            out = dl.download(-125, -119, 33, 38, "2024-01-31", "2024-01-31")
        assert out == []

    def test_every_month_404_raises_value_error(self, tmp_path):
        import urllib.error

        from sar_validation.downloaders.noaa_hfradar_thredds_downloader import (
            NOAATHREDDSHFRadarDownloader,
        )

        def fake_urlopen(url, timeout=None):
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

        with patch(
            "sar_validation.downloaders.noaa_hfradar_thredds_downloader.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            dl = NOAATHREDDSHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
            with pytest.raises(ValueError, match="No THREDDS"):
                dl.download(-125, -119, 33, 38, "2005-01-31", "2005-01-31")

    def test_matching_granule_downloaded_renamed_and_merged(self, tmp_path):
        from sar_validation.downloaders.noaa_hfradar_thredds_downloader import (
            NOAATHREDDSHFRadarDownloader,
        )

        catalog = _NEW_ERA_CATALOG_XML
        # Build a real synthetic granule up front so the fake response for
        # /fileServer/ URLs has real netCDF bytes for the merge step to read.
        granule_bytes = _make_thredds_nc(tmp_path, "granule_src.nc").read_bytes()

        def fake_urlopen(url, timeout=None):
            cm = MagicMock()
            if "/fileServer/" in url:
                cm.__enter__.return_value.read.return_value = granule_bytes
            else:
                cm.__enter__.return_value.read.return_value = catalog.encode()
            return cm

        with patch(
            "sar_validation.downloaders.noaa_hfradar_thredds_downloader.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            dl = NOAATHREDDSHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
            out = dl.download(-125, -119, 33, 38, "2026-06-30", "2026-06-30")

        assert len(out) == 1
        merged = xr.open_dataset(out[0])
        assert "water_u" in merged and "water_v" in merged
        assert "u" not in merged and "v" not in merged
        assert "USWC" in out[0].name

    def test_merged_output_trimmed_to_request_bbox(self, tmp_path):
        """THREDDS serves whole-region grids (unlike ERDDAP's server-side
        subsetting), so the downloader must trim the merged output to the
        requested bbox client-side. Build a granule whose raw grid extends
        well beyond the requested window and confirm the output only
        contains lat/lon within the requested bounds, and that trimming
        actually shrank the extent (not a no-op)."""
        from sar_validation.downloaders.noaa_hfradar_thredds_downloader import (
            NOAATHREDDSHFRadarDownloader,
        )

        catalog = _NEW_ERA_CATALOG_XML
        # Raw granule grid spans well beyond the bbox we'll request below.
        raw_lat_bounds = (30.0, 40.0)
        raw_lon_bounds = (-128.0, -116.0)
        granule_bytes = _make_thredds_nc(
            tmp_path, "granule_src.nc", n_lat=11, n_lon=13,
            lat_bounds=raw_lat_bounds, lon_bounds=raw_lon_bounds,
        ).read_bytes()

        def fake_urlopen(url, timeout=None):
            cm = MagicMock()
            if "/fileServer/" in url:
                cm.__enter__.return_value.read.return_value = granule_bytes
            else:
                cm.__enter__.return_value.read.return_value = catalog.encode()
            return cm

        req_min_lon, req_max_lon, req_min_lat, req_max_lat = -124.0, -120.0, 34.0, 37.0
        with patch(
            "sar_validation.downloaders.noaa_hfradar_thredds_downloader.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            dl = NOAATHREDDSHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
            out = dl.download(
                req_min_lon, req_max_lon, req_min_lat, req_max_lat, "2026-06-30", "2026-06-30",
            )

        assert len(out) == 1
        merged = xr.open_dataset(out[0])
        assert float(merged.lat.min()) >= req_min_lat
        assert float(merged.lat.max()) <= req_max_lat
        assert float(merged.lon.min()) >= req_min_lon
        assert float(merged.lon.max()) <= req_max_lon
        # Trimming actually happened: the output's extent is strictly
        # smaller than the raw (ungapped) granule's extent.
        assert float(merged.lat.max()) - float(merged.lat.min()) < raw_lat_bounds[1] - raw_lat_bounds[0]
        assert float(merged.lon.max()) - float(merged.lon.min()) < raw_lon_bounds[1] - raw_lon_bounds[0]
        # Regression check: non-gridded variables must be left untouched by
        # the bbox trim, not broadcast across lat/lon (the .where() bug) --
        # dims and dtypes must match what _make_thredds_nc wrote.
        assert merged["time_bnds"].dims == ("time", "nv")
        assert merged["wgs84"].dims == ("time",)
        assert merged["wgs84"].dtype == np.int8

    def test_zero_overlap_bbox_raises_clear_value_error(self, tmp_path):
        """A request bbox with zero overlap with the granule's actual grid
        (e.g. a fine-resolution request targeting a sub-area outside the
        granule's real footprint) must raise a clear, explicit ValueError
        explaining why -- not silently write a corrupted/empty file, and not
        crash with .where()'s confusing dimension-mismatch error."""
        from sar_validation.downloaders.noaa_hfradar_thredds_downloader import (
            NOAATHREDDSHFRadarDownloader,
        )

        catalog = _NEW_ERA_CATALOG_XML
        # Granule's real grid spans lat 33-38; request a bbox whose center
        # still falls within US_WEST's overall region bbox (lon -130.36..
        # -115.8056, lat 30.25..49.99204, so match_noaa_hfr_region succeeds
        # and the ValueError comes from the new post-merge check, not region
        # matching) but is entirely north of the granule's actual grid --
        # zero overlap on the lat axis.
        granule_bytes = _make_thredds_nc(
            tmp_path, "granule_src.nc", lat_bounds=(33.0, 38.0), lon_bounds=(-125.0, -119.0),
        ).read_bytes()

        def fake_urlopen(url, timeout=None):
            cm = MagicMock()
            if "/fileServer/" in url:
                cm.__enter__.return_value.read.return_value = granule_bytes
            else:
                cm.__enter__.return_value.read.return_value = catalog.encode()
            return cm

        with patch(
            "sar_validation.downloaders.noaa_hfradar_thredds_downloader.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            dl = NOAATHREDDSHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
            with pytest.raises(ValueError, match="grid points fall within the requested bbox"):
                dl.download(-125, -119, 40.0, 42.0, "2026-06-30", "2026-06-30")

    def test_temp_dir_cleaned_up_when_a_granule_fetch_raises_partway_through(self, tmp_path):
        """A network error mid-loop (second of two granules) must not leave
        .thredds_tmp* behind: a later run at a different resolution could
        find foreign leftovers, and a stale non-empty tmp_dir would make a
        plain .rmdir() raise OSError on a later, unrelated, non-error run --
        an exception hf_radar_us_downloader's narrow `except ValueError`
        does not catch, disabling the whole Copernicus fallback."""
        import urllib.error

        from sar_validation.downloaders.noaa_hfradar_thredds_downloader import (
            NOAATHREDDSHFRadarDownloader,
        )

        # Two granules in range so a second call can be made to fail.
        catalog = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0">\n'
            '  <dataset name="rtv-uswc-6km-uwls_v1r0_hfr_s202606302200000_e202606302200000_'
            'c202607010100213.nc"\n'
            '    urlPath="rtv/2026/202606/USWC/rtv-uswc-6km-uwls_v1r0_hfr_s202606302200000_'
            'e202606302200000_c202607010100213.nc" />\n'
            '  <dataset name="rtv-uswc-6km-uwls_v1r0_hfr_s202606302300000_e202606302300000_'
            'c202607011730210.nc"\n'
            '    urlPath="rtv/2026/202606/USWC/rtv-uswc-6km-uwls_v1r0_hfr_s202606302300000_'
            'e202606302300000_c202607011730210.nc" />\n'
            "</catalog>\n"
        )
        granule_bytes = _make_thredds_nc(tmp_path, "granule_src.nc").read_bytes()
        granule_call_count = [0]

        def fake_urlopen(url, timeout=None):
            if "/fileServer/" in url:
                granule_call_count[0] += 1
                if granule_call_count[0] == 2:
                    raise urllib.error.URLError("connection reset")
                cm = MagicMock()
                cm.__enter__.return_value.read.return_value = granule_bytes
                return cm
            cm = MagicMock()
            cm.__enter__.return_value.read.return_value = catalog.encode()
            return cm

        with patch(
            "sar_validation.downloaders.noaa_hfradar_thredds_downloader.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            dl = NOAATHREDDSHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
            with pytest.raises(urllib.error.URLError):
                dl.download(-125, -119, 33, 38, "2026-06-30", "2026-06-30")

        assert not (tmp_path / ".thredds_tmp").exists()

    @pytest.mark.parametrize(
        "url_predicate,start,end,needs_granule_bytes",
        [
            pytest.param(
                lambda url: "/fileServer/" in url,
                "2026-06-30", "2026-06-30",
                True,
                id="granule_download_uses_a_bounded_timeout",
            ),
            pytest.param(
                lambda url: "/fileServer/" not in url,
                "2024-01-31", "2024-01-31",
                False,
                id="catalog_fetch_uses_a_bounded_timeout_of_15",
            ),
        ],
    )
    def test_bounded_timeout(self, tmp_path, url_predicate, start, end, needs_granule_bytes):
        """Regression test: urlretrieve (the original implementation) has no
        timeout parameter at all, and a stalled connection would hang the
        download indefinitely. The fix moved to urlopen(..., timeout=...)
        for granule fetches specifically so a hung connection is bounded.
        Lowered from 60 to 15: combined with prefer_ipv4_dns(), a genuinely
        broken IPv6 path now fails fast per address instead of eating up to
        6 * timeout seconds before reaching a working IPv4 address. The
        pre-existing timeout=30 catalog urlopen() call site is lowered to
        15 too, for the same IPv6-black-hole reason."""
        from sar_validation.downloaders.noaa_hfradar_thredds_downloader import (
            NOAATHREDDSHFRadarDownloader,
        )

        catalog = _NEW_ERA_CATALOG_XML
        granule_bytes = _make_thredds_nc(tmp_path, "granule_src.nc").read_bytes()
        matched_calls = []

        def fake_urlopen(url, timeout=None):
            cm = MagicMock()
            if url_predicate(url):
                matched_calls.append(timeout)
                cm.__enter__.return_value.read.return_value = (
                    granule_bytes if needs_granule_bytes else catalog.encode()
                )
            else:
                cm.__enter__.return_value.read.return_value = catalog.encode()
            return cm

        with patch(
            "sar_validation.downloaders.noaa_hfradar_thredds_downloader.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            dl = NOAATHREDDSHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
            dl.download(-125, -119, 33, 38, start, end)

        assert matched_calls  # at least one matching call was actually made
        assert all(t == 15 for t in matched_calls)

    @pytest.mark.parametrize(
        "catalog,granule_matches,start,end,expected_call_count,expected_out_count",
        [
            pytest.param(
                (
                    '<?xml version="1.0"?>'
                    '<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0">'
                    "</catalog>"
                ),
                False,
                "2024-01-31", "2024-01-31",
                1, 0,
                id="catalog_fetch_wraps_network_call_in_prefer_ipv4_dns",
            ),
            pytest.param(
                _NEW_ERA_CATALOG_XML,
                True,
                "2026-06-30T23:00:00", "2026-06-30T23:00:00",
                2, 1,
                id="granule_download_wraps_network_call_in_prefer_ipv4_dns",
            ),
        ],
    )
    def test_prefer_ipv4_dns_wiring(
        self, tmp_path, catalog, granule_matches, start, end, expected_call_count, expected_out_count,
    ):
        """Wiring test: both the catalog and granule urlopen() call sites
        must actually use prefer_ipv4_dns(), not just have it importable.
        The empty-catalog scenario exercises only the catalog call site
        (no granules match, so no fileServer fetch happens); the matching-
        granule scenario exercises both call sites, so prefer_ipv4_dns()
        must be entered exactly twice if both call sites are wired -- only
        once would indicate the granule site is not."""
        from sar_validation.downloaders.noaa_hfradar_thredds_downloader import (
            NOAATHREDDSHFRadarDownloader,
        )

        granule_bytes = (
            _make_thredds_nc(tmp_path, "granule_src.nc").read_bytes() if granule_matches else None
        )

        def fake_urlopen(url, timeout=None):
            cm = MagicMock()
            if "/fileServer/" in url:
                cm.__enter__.return_value.read.return_value = granule_bytes
            else:
                cm.__enter__.return_value.read.return_value = catalog.encode()
            return cm

        with patch(
            "sar_validation.downloaders.noaa_hfradar_thredds_downloader.prefer_ipv4_dns"
        ) as mock_prefer:
            mock_prefer.return_value.__exit__.return_value = False
            with patch(
                "sar_validation.downloaders.noaa_hfradar_thredds_downloader.urllib.request.urlopen",
                side_effect=fake_urlopen,
            ):
                dl = NOAATHREDDSHFRadarDownloader(
                    output_dir=tmp_path, dry_run=False, resolution_km=6,
                )
                out = dl.download(-125, -119, 33, 38, start, end)

        assert len(out) == expected_out_count
        assert mock_prefer.call_count == expected_call_count
        assert mock_prefer.return_value.__enter__.call_count == expected_call_count
        assert mock_prefer.return_value.__exit__.call_count == expected_call_count
