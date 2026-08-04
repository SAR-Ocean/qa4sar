"""Tests for CDSSoilMoistureDownloader."""

from __future__ import annotations

import zipfile
from datetime import date


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
        assert req["type_of_record"] == ["icdr"]
        assert req["year"] == ["2026"]
        assert req["month"] == ["03"]
        assert req["day"] == ["15"]
        assert "variable" in req
        assert "time_aggregation" in req

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

