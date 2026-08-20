"""Tests for CDSSoilMoistureDownloader."""

from __future__ import annotations

import sys
import zipfile
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


class TestCDSSoilMoistureDownloaderDryRun:
    def test_dry_run_returns_empty_without_network_call(self, tmp_path):
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        dl = CDSSoilMoistureDownloader(product_type="active", output_dir=tmp_path, dry_run=True)
        paths = dl.download(
            min_lon=-10.0, max_lon=30.0, min_lat=35.0, max_lat=60.0,
            start="2026-01-01", end="2026-01-03",
        )
        # dry_run: no network call, no files written, returns empty
        assert paths == []
        assert not list(tmp_path.glob("*.nc"))

    def test_dry_run_skips_existing_files_and_returns_them(self, tmp_path):
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        # Pre-create a file; it should be returned even in dry_run (already cached)
        existing = tmp_path / "c3s_ssm_active_20260101.nc"
        existing.write_text("fake")

        dl = CDSSoilMoistureDownloader(product_type="active", output_dir=tmp_path, dry_run=True)
        paths = dl.download(
            min_lon=-10.0, max_lon=30.0, min_lat=35.0, max_lat=60.0,
            start="2026-01-01", end="2026-01-03",
        )
        # 2026-01-01 already cached → returned; 2026-01-02 dry_run → skipped
        assert len(paths) == 1
        assert paths[0].name == "c3s_ssm_active_20260101.nc"


class TestCDSSoilMoistureDownloaderDateRangeBoundary:
    """Reproduces a real pipeline bug: a recipe's padded collocation window
    (built by orchestrator._padded_temporal_bounds, which always returns a
    full datetime, never a bare date) can extend a few hours into the next
    calendar day even when the literal recipe end time doesn't -- e.g. a
    19:00-20:00 SAR pass with a 360-minute tolerance pads out to 02:00 the
    next day. The day that padded end datetime falls on must still be
    downloaded, or every point in that source's only file lands outside the
    later collocation time-filter and the whole source silently vanishes
    from the report -- confirmed against a real
    recipes/soil_moisture_cds_nisar_test.yaml run where only
    c3s_ssm_passive_20260709.nc was fetched despite the padded window
    reaching into 2026-07-10."""

    def test_padded_end_crossing_midnight_includes_the_next_day(self, tmp_path, monkeypatch):
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        requested_days: list[date] = []

        def fake_download_day(self, day):
            requested_days.append(day)
            return None

        monkeypatch.setattr(CDSSoilMoistureDownloader, "_download_day", fake_download_day)

        dl = CDSSoilMoistureDownloader(product_type="passive", output_dir=tmp_path)
        dl.download(
            min_lon=-10.0, max_lon=30.0, min_lat=35.0, max_lat=60.0,
            start="2026-07-09T13:00:00", end="2026-07-10T02:00:00",
        )

        assert requested_days == [date(2026, 7, 9), date(2026, 7, 10)]

    def test_end_exactly_on_a_day_boundary_still_includes_that_day(self, tmp_path, monkeypatch):
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        requested_days: list[date] = []
        monkeypatch.setattr(
            CDSSoilMoistureDownloader, "_download_day",
            lambda self, day: requested_days.append(day) or None,
        )

        dl = CDSSoilMoistureDownloader(product_type="active", output_dir=tmp_path)
        dl.download(
            min_lon=-10.0, max_lon=30.0, min_lat=35.0, max_lat=60.0,
            start="2026-07-09T00:00:00", end="2026-07-09T00:00:00",
        )

        assert requested_days == [date(2026, 7, 9)]


class TestCDSSoilMoistureDownloaderFailurePropagation:
    """Previously, when every requested day's CDS API call itself errored
    (e.g. a 400 Bad Request from an invalid facet combination),
    download() silently returned [] with no exception -- orchestrator.py's
    _run_download treats that as an ordinary "status": "success" with 0
    files, so the actual cause (each day's own WARNING log line) never
    reached the run's final "Warnings from this run" summary. A recipe
    could report zero CDS data with no visible explanation at all."""

    def test_raises_when_every_day_errors_at_the_api_level(self, tmp_path, monkeypatch):
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        def fake_download_day(self, day):
            self._had_request_failure = True
            return None

        monkeypatch.setattr(CDSSoilMoistureDownloader, "_download_day", fake_download_day)

        dl = CDSSoilMoistureDownloader(product_type="active", output_dir=tmp_path)
        with pytest.raises(RuntimeError, match="every requested day"):
            dl.download(
                min_lon=-10.0, max_lon=30.0, min_lat=35.0, max_lat=60.0,
                start="2024-01-01", end="2024-01-02",
            )

    def test_no_raise_when_some_days_succeed(self, tmp_path, monkeypatch):
        """A mix of successes and API-level failures must not raise --
        per-day independence (see the module docstring) means a partial
        result is still useful and must be returned normally."""
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        def fake_download_day(self, day):
            if day == date(2024, 1, 1):
                self._had_request_failure = True
                return None
            return self._nc_path_for_day(day)

        monkeypatch.setattr(CDSSoilMoistureDownloader, "_download_day", fake_download_day)

        dl = CDSSoilMoistureDownloader(product_type="active", output_dir=tmp_path)
        result = dl.download(
            min_lon=-10.0, max_lon=30.0, min_lat=35.0, max_lat=60.0,
            start="2024-01-01", end="2024-01-02",
        )
        assert result == [dl._nc_path_for_day(date(2024, 1, 2))]

    def test_no_raise_when_zero_files_is_a_genuinely_empty_response(self, tmp_path, monkeypatch):
        """Every day returning None WITHOUT ever setting
        _had_request_failure (i.e. every CDS API call itself succeeded,
        just with no extractable data) must be treated as the existing
        benign "no data for this window" outcome, not a failure."""
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        monkeypatch.setattr(
            CDSSoilMoistureDownloader, "_download_day", lambda self, day: None,
        )

        dl = CDSSoilMoistureDownloader(product_type="active", output_dir=tmp_path)
        result = dl.download(
            min_lon=-10.0, max_lon=30.0, min_lat=35.0, max_lat=60.0,
            start="2024-01-01", end="2024-01-02",
        )
        assert result == []


class TestCDSSoilMoistureDownloaderNcPath:
    def test_nc_path_naming_active(self, tmp_path):
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        dl = CDSSoilMoistureDownloader(product_type="active", output_dir=tmp_path)
        assert dl._nc_path_for_day(date(2026, 1, 5)).name == "c3s_ssm_active_20260105.nc"

    def test_nc_path_naming_passive(self, tmp_path):
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        dl = CDSSoilMoistureDownloader(product_type="passive", output_dir=tmp_path)
        assert dl._nc_path_for_day(date(2019, 12, 31)).name == "c3s_ssm_passive_20191231.nc"

    def test_nc_path_naming_combined(self, tmp_path):
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        dl = CDSSoilMoistureDownloader(product_type="combined", output_dir=tmp_path)
        assert dl._nc_path_for_day(date(2024, 6, 15)).name == "c3s_ssm_combined_20240615.nc"


class TestCDSSoilMoistureDownloaderBuildRequest:
    def test_build_request_active(self, tmp_path):
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        dl = CDSSoilMoistureDownloader(product_type="active", output_dir=tmp_path)
        req = dl._build_request(date(2026, 3, 15))
        assert req["type_of_sensor"] == ["active"]
        assert req["type_of_record"] == ["cdr"]
        assert req["year"] == ["2026"]
        assert req["month"] == ["03"]
        assert req["day"] == ["15"]
        assert "variable" in req
        assert "time_aggregation" in req

    def test_build_request_honors_explicit_type_of_record(self, tmp_path):
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        dl = CDSSoilMoistureDownloader(product_type="active", output_dir=tmp_path)
        req = dl._build_request(date(2026, 3, 15), type_of_record="icdr")
        assert req["type_of_record"] == ["icdr"]

    def test_build_request_passive(self, tmp_path):
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        dl = CDSSoilMoistureDownloader(product_type="passive", output_dir=tmp_path)
        req = dl._build_request(date(2019, 7, 4))
        assert req["type_of_sensor"] == ["passive"]
        assert req["month"] == ["07"]
        assert req["day"] == ["04"]

    def test_build_request_combined(self, tmp_path):
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        dl = CDSSoilMoistureDownloader(product_type="combined", output_dir=tmp_path)
        req = dl._build_request(date(2020, 1, 1))
        assert req["type_of_sensor"] == ["combined"]

    def test_build_request_active_uses_saturation_variable(self, tmp_path):
        """ASCAT ('active') soil moisture is percent-saturation, not
        volumetric -- confirmed against CDS's own live constraints.json
        (https://cds.climate.copernicus.eu/api/catalogue/v1/collections/
        satellite-soil-moisture/constraints.json): every active+daily+icdr
        combination requires variable='surface_soil_moisture_saturation',
        never '..._volumetric'. Requesting the volumetric variable for
        'active' is rejected by the live API with a 400 Bad Request --
        confirmed live, this exact request previously downloaded 0 files
        for every 'active' day with no error surfaced past a WARNING log,
        silently starving Sentinel-1 CLMS SSM recipes of C3S CDS data."""
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        dl = CDSSoilMoistureDownloader(product_type="active", output_dir=tmp_path)
        req = dl._build_request(date(2026, 3, 15))
        assert req["variable"] == ["surface_soil_moisture_saturation"]

    def test_build_request_passive_uses_volumetric_variable(self, tmp_path):
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        dl = CDSSoilMoistureDownloader(product_type="passive", output_dir=tmp_path)
        req = dl._build_request(date(2026, 3, 15))
        assert req["variable"] == ["surface_soil_moisture_volumetric"]

    def test_build_request_combined_uses_volumetric_variable(self, tmp_path):
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        dl = CDSSoilMoistureDownloader(product_type="combined", output_dir=tmp_path)
        req = dl._build_request(date(2026, 3, 15))
        assert req["variable"] == ["surface_soil_moisture_volumetric"]

    def test_build_request_submits_every_known_version_by_default(self, tmp_path):
        """A single hardcoded "latest" version (v202505) is only valid for
        2025-2026 per CDS's own live constraints.json -- confirmed live,
        requesting it for an older date (e.g. 2024-01-01) gets rejected
        with a 400 "Request has not produced a valid combination of
        values", even though older versions of this exact same dataset do
        cover that date. Every known version must be submitted together
        (an OR-style multi-value facet) so CDS's own constraint solver
        picks whichever applies to the requested day, regardless of which
        year is requested."""
        from sar_validation.downloaders.cds_soil_moisture_downloader import (
            _CDS_VERSIONS,
            CDSSoilMoistureDownloader,
        )

        dl = CDSSoilMoistureDownloader(product_type="active", output_dir=tmp_path)
        req = dl._build_request(date(2024, 1, 1))
        assert req["version"] == _CDS_VERSIONS
        assert len(_CDS_VERSIONS) > 1

    def test_build_request_honors_explicit_version_override(self, tmp_path):
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        dl = CDSSoilMoistureDownloader(
            product_type="active", output_dir=tmp_path, version="v202312",
        )
        req = dl._build_request(date(2024, 1, 1))
        assert req["version"] == ["v202312"]


class TestCDSSoilMoistureDownloaderCdrIcdrFallback:
    """_download_day() must prefer the finalized CDR record and only fall
    back to the faster, near-real-time ICDR record when the CDR request
    itself fails -- e.g. CDR hasn't been published yet for a very recent
    day. Requesting ICDR unconditionally (the previous behavior) never
    asks for the better-quality finalized product even when it's already
    available."""

    def test_requests_cdr_first_and_succeeds_without_fallback(self, tmp_path):
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        dl = CDSSoilMoistureDownloader(product_type="active", output_dir=tmp_path)
        fake_remote = MagicMock()
        fake_client = MagicMock()
        fake_client.retrieve.return_value = fake_remote

        def fake_download(dest):
            with zipfile.ZipFile(dest, "w") as zf:
                zf.writestr("c3s_ssm_active_20260315.nc", b"fake")

        fake_remote.download.side_effect = fake_download

        with patch.dict(sys.modules, {"cdsapi": MagicMock(Client=MagicMock(return_value=fake_client))}):
            result = dl._download_day(date(2026, 3, 15))

        assert result is not None
        assert fake_client.retrieve.call_count == 1
        request = fake_client.retrieve.call_args[0][1]
        assert request["type_of_record"] == ["cdr"]
        assert dl._had_request_failure is False

    def test_falls_back_to_icdr_when_cdr_request_fails(self, tmp_path):
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        dl = CDSSoilMoistureDownloader(product_type="active", output_dir=tmp_path)
        fake_client = MagicMock()

        def fake_download(dest):
            with zipfile.ZipFile(dest, "w") as zf:
                zf.writestr("c3s_ssm_active_20260315.nc", b"fake")

        succeeding_remote = MagicMock()
        succeeding_remote.download.side_effect = fake_download
        fake_client.retrieve.side_effect = [
            RuntimeError("CDR not yet published for this day"),
            succeeding_remote,
        ]

        with patch.dict(sys.modules, {"cdsapi": MagicMock(Client=MagicMock(return_value=fake_client))}):
            result = dl._download_day(date(2026, 3, 15))

        assert result is not None
        assert fake_client.retrieve.call_count == 2
        first_request = fake_client.retrieve.call_args_list[0][0][1]
        second_request = fake_client.retrieve.call_args_list[1][0][1]
        assert first_request["type_of_record"] == ["cdr"]
        assert second_request["type_of_record"] == ["icdr"]
        assert dl._had_request_failure is False

    def test_fails_when_both_cdr_and_icdr_requests_fail(self, tmp_path):
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        dl = CDSSoilMoistureDownloader(product_type="active", output_dir=tmp_path)
        fake_client = MagicMock()
        fake_client.retrieve.side_effect = [
            RuntimeError("CDR not yet published"),
            RuntimeError("ICDR also unavailable"),
        ]

        with patch.dict(sys.modules, {"cdsapi": MagicMock(Client=MagicMock(return_value=fake_client))}):
            result = dl._download_day(date(2026, 3, 15))

        assert result is None
        assert fake_client.retrieve.call_count == 2
        assert dl._had_request_failure is True


class TestCDSSoilMoistureDownloaderCheckAvailabilityDry:
    """check_availability_dry is a fast, unauthenticated existence probe for
    dry-collocation prediction -- queries the CDS catalogue's live
    collection-metadata endpoint (``ecmwf.datastores.Client.get_collection``)
    for this dataset's real temporal extent, rather than submitting a real
    CDR/ICDR processing job the way ``_download_day`` does."""

    @staticmethod
    def _patch_datastores(monkeypatch, begin=None, end=None, client_cls=None):
        """Install a fake ``ecmwf.datastores`` module in sys.modules whose
        Client(...).get_collection(...) returns a fake Collection exposing
        begin_datetime/end_datetime. Returns (fake_client_cls,
        fake_client_instance) so tests can assert on construction/calls."""
        fake_collection = MagicMock(begin_datetime=begin, end_datetime=end)
        fake_client_instance = MagicMock()
        fake_client_instance.get_collection.return_value = fake_collection
        fake_client_cls = client_cls or MagicMock(return_value=fake_client_instance)

        fake_datastores_module = MagicMock()
        fake_datastores_module.Client = fake_client_cls

        fake_ecmwf_module = MagicMock()
        fake_ecmwf_module.datastores = fake_datastores_module

        monkeypatch.setitem(sys.modules, "ecmwf", fake_ecmwf_module)
        monkeypatch.setitem(sys.modules, "ecmwf.datastores", fake_datastores_module)
        return fake_client_cls, fake_client_instance

    def test_true_when_day_falls_within_live_extent(self, monkeypatch, tmp_path):
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        self._patch_datastores(
            monkeypatch,
            begin=datetime(1978, 11, 1, tzinfo=timezone.utc),
            end=datetime(2026, 8, 10, tzinfo=timezone.utc),
        )

        dl = CDSSoilMoistureDownloader(product_type="active", output_dir=tmp_path)
        assert dl.check_availability_dry(date(2026, 3, 15)) is True

    def test_false_when_day_falls_outside_live_extent(self, monkeypatch, tmp_path):
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        self._patch_datastores(
            monkeypatch,
            begin=datetime(1978, 11, 1, tzinfo=timezone.utc),
            end=datetime(2026, 8, 10, tzinfo=timezone.utc),
        )

        dl = CDSSoilMoistureDownloader(product_type="active", output_dir=tmp_path)
        assert dl.check_availability_dry(date(2026, 9, 1)) is False

    def test_queries_the_right_dataset_and_never_downloads(self, monkeypatch, tmp_path):
        from sar_validation.downloaders.cds_soil_moisture_downloader import (
            _CDS_DATASET,
            CDSSoilMoistureDownloader,
        )

        _fake_cls, fake_client = self._patch_datastores(
            monkeypatch,
            begin=datetime(1978, 11, 1, tzinfo=timezone.utc),
            end=datetime(2026, 8, 10, tzinfo=timezone.utc),
        )

        dl = CDSSoilMoistureDownloader(product_type="active", output_dir=tmp_path)
        dl.check_availability_dry(date(2026, 3, 15))

        fake_client.get_collection.assert_called_once_with(_CDS_DATASET)
        # No blocking CDR/ICDR processing job is ever submitted.
        fake_client.retrieve.assert_not_called()

    def test_raises_when_catalogue_lookup_fails(self, monkeypatch, tmp_path):
        """A network/auth/API error while querying the catalogue (e.g. no
        connectivity, or the endpoint itself erroring) must propagate --
        never be swallowed into False -- so _predict_global_composite's own
        exception handling is what converts it to an 'unknown' verdict,
        never a false 'none-predicted'."""
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        failing_client_cls = MagicMock(side_effect=RuntimeError("connection refused"))
        self._patch_datastores(monkeypatch, client_cls=failing_client_cls)

        dl = CDSSoilMoistureDownloader(product_type="active", output_dir=tmp_path)
        with pytest.raises(RuntimeError, match="connection refused"):
            dl.check_availability_dry(date(2026, 3, 15))

    def test_raises_when_ecmwf_datastores_not_installed(self, monkeypatch, tmp_path):
        """Previously, a missing cdsapi/ecmwf-datastores-client dependency,
        or any cdsapi auth/connection failure (e.g. no ~/.cdsapirc
        configured -- the single most likely real-world case), was swallowed
        into a false 'no data' result. It must now propagate as an
        ImportError so the caller's own exception handling produces
        'unknown', never a false 'none-predicted'."""
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        dl = CDSSoilMoistureDownloader(product_type="active", output_dir=tmp_path)
        monkeypatch.setitem(sys.modules, "ecmwf.datastores", None)

        with pytest.raises(ImportError):
            dl.check_availability_dry(date(2026, 3, 15))

    def test_raises_when_catalogue_extent_is_missing(self, monkeypatch, tmp_path):
        """A malformed/incomplete catalogue response (no usable
        begin/end datetime) can't answer the "does data exist" question
        either -- this must raise, not silently return False."""
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        self._patch_datastores(monkeypatch, begin=None, end=None)

        dl = CDSSoilMoistureDownloader(product_type="active", output_dir=tmp_path)
        with pytest.raises(RuntimeError, match="temporal extent"):
            dl.check_availability_dry(date(2026, 3, 15))


class TestCDSSoilMoistureDownloaderExtractNc:
    def test_extract_nc_renames_to_stable_filename(self, tmp_path):
        """_extract_nc pulls the first .nc from the zip and renames it to the
        canonical c3s_ssm_<product_type>_<YYYYMMDD>.nc path. Caller must delete zip."""
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        dl = CDSSoilMoistureDownloader(product_type="active", output_dir=tmp_path)
        day = date(2026, 2, 1)

        # Build a zip with one .nc inside
        zip_path = tmp_path / "cds_response.zip"
        nc_inside_name = "ESACCI-SOILMOISTURE-L3S-SSMV-ACTIVE-20260201000000-fv07.1.nc"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(nc_inside_name, b"fake_nc_content")

        result = dl._extract_nc(zip_path, day)

        expected = tmp_path / "c3s_ssm_active_20260201.nc"
        assert result == expected
        assert expected.exists()
        # Zip is not deleted by _extract_nc; caller must delete it
        assert zip_path.exists()

    def test_extract_nc_returns_none_if_no_nc_in_zip(self, tmp_path):
        from sar_validation.downloaders.cds_soil_moisture_downloader import CDSSoilMoistureDownloader

        dl = CDSSoilMoistureDownloader(product_type="active", output_dir=tmp_path)
        day = date(2026, 2, 1)

        zip_path = tmp_path / "empty.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("readme.txt", "no nc here")

        result = dl._extract_nc(zip_path, day)
        assert result is None

