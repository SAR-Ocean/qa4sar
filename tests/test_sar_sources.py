"""Tests for the SAR_SOURCES registry (sar_validation.core.sar_sources)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestSarSourcesRegistryShape:
    def test_registry_has_the_three_registered_sources(self):
        from sar_validation.core.sar_sources import SAR_SOURCES

        assert set(SAR_SOURCES) == {"sentinel1_l2_ocn", "sentinel1_clms_ssm", "nisar_sme2"}

    def test_sentinel1_l2_ocn_applies_to_wind_waves_currents(self):
        from sar_validation.core.sar_sources import SAR_SOURCES

        spec = SAR_SOURCES["sentinel1_l2_ocn"]
        assert spec.variables == frozenset({"wind", "waves", "currents"})
        assert spec.output_subdir == "S1_L2_OCN"
        assert spec.file_glob == "*.SAFE"

    def test_sentinel1_clms_ssm_applies_to_soil_moisture_only(self):
        from sar_validation.core.sar_sources import SAR_SOURCES

        spec = SAR_SOURCES["sentinel1_clms_ssm"]
        assert spec.variables == frozenset({"soil_moisture"})
        assert spec.output_subdir == "S1_L3_SSM"
        assert spec.file_glob == "*.tif*"
        assert spec.default_min_depth == 0.0
        assert spec.default_max_depth == 0.05
        assert spec.default_time_tolerance_minutes == 720
        assert spec.default_aggregation_window_km == 1.0
        assert spec.default_spatial_tolerance_km == 2.0

    def test_sentinel1_l2_ocn_has_no_soil_moisture_defaults(self):
        from sar_validation.core.sar_sources import SAR_SOURCES

        spec = SAR_SOURCES["sentinel1_l2_ocn"]
        assert spec.default_min_depth is None
        assert spec.default_max_depth is None
        assert spec.default_time_tolerance_minutes is None


class TestSentinel1L2OcnDownloaderWiring:
    def test_build_downloader_returns_sar_downloader(self, tmp_path):
        from sar_validation.core.sar_sources import SAR_SOURCES

        spec = SAR_SOURCES["sentinel1_l2_ocn"]
        with patch("sar_validation.downloaders.sar_downloader.SARDownloader") as mock_cls:
            mock_cls.return_value = MagicMock()
            dl = spec.build_downloader(tmp_path, False, True)
        mock_cls.assert_called_once_with(output_dir=tmp_path, dry_run=False, force_download=True)
        assert dl is mock_cls.return_value

    def test_extra_download_kwargs_maps_swath_mode_and_max_downloads(self):
        from sar_validation.core.recipe import SARDataSpec
        from sar_validation.core.sar_sources import SAR_SOURCES

        spec = SAR_SOURCES["sentinel1_l2_ocn"]
        sd = SARDataSpec(swath_mode=["IW", "EW"], max_downloads=5, download_kwargs={"top": 50})
        kwargs = spec.extra_download_kwargs(sd)
        assert kwargs == {"modes": ["IW", "EW"], "limit": 5, "top": 50}

    def test_extra_download_kwargs_none_swath_mode_becomes_none(self):
        from sar_validation.core.recipe import SARDataSpec
        from sar_validation.core.sar_sources import SAR_SOURCES

        spec = SAR_SOURCES["sentinel1_l2_ocn"]
        sd = SARDataSpec(swath_mode=[], max_downloads=None)
        kwargs = spec.extra_download_kwargs(sd)
        assert kwargs["modes"] is None
        assert kwargs["limit"] is None


class TestSentinel1ClmsSsmDownloaderWiring:
    def test_build_downloader_returns_soil_moisture_downloader(self, tmp_path):
        from sar_validation.core.sar_sources import SAR_SOURCES

        spec = SAR_SOURCES["sentinel1_clms_ssm"]
        with patch(
            "sar_validation.downloaders.soil_moisture_downloader.SoilMoistureDownloader"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            dl = spec.build_downloader(tmp_path, True, False)
        mock_cls.assert_called_once_with(output_dir=tmp_path, dry_run=True, force_download=False)
        assert dl is mock_cls.return_value

    def test_extra_download_kwargs_ignores_swath_mode(self):
        from sar_validation.core.recipe import SARDataSpec
        from sar_validation.core.sar_sources import SAR_SOURCES

        spec = SAR_SOURCES["sentinel1_clms_ssm"]
        sd = SARDataSpec(swath_mode=["IW"], max_downloads=3, download_kwargs={"top": 10})
        kwargs = spec.extra_download_kwargs(sd)
        assert kwargs == {"top": 10}


class TestConvertCallbacks:
    def test_sentinel1_l2_ocn_convert_calls_from_sar_l2_ocn_safe(self, tmp_path):
        from sar_validation.core.sar_sources import SAR_SOURCES

        spec = SAR_SOURCES["sentinel1_l2_ocn"]
        safe_dir = tmp_path / "S1A_IW_OCN.SAFE"
        safe_dir.mkdir()
        with patch(
            "sar_validation.core.datatree_converter.DataTreeConverter.from_sar_l2_ocn_safe"
        ) as mock_fn:
            mock_fn.return_value = "fake_dataset"
            result = spec.convert(safe_dir, "currents")
        mock_fn.assert_called_once_with(safe_dir, product_type="currents")
        assert result == "fake_dataset"

    def test_sentinel1_clms_ssm_convert_calls_from_sar_l3_ssm_geotiff(self, tmp_path):
        from sar_validation.core.sar_sources import SAR_SOURCES

        spec = SAR_SOURCES["sentinel1_clms_ssm"]
        tif_path = tmp_path / "ssm.tif"
        with patch(
            "sar_validation.core.datatree_converter.DataTreeConverter.from_sar_l3_ssm_geotiff"
        ) as mock_fn:
            mock_fn.return_value = "fake_dataset"
            result = spec.convert(tif_path, "wind")
        mock_fn.assert_called_once_with(tif_path)
        assert result == "fake_dataset"


class TestNisarSme2RegistryEntry:
    def test_applies_to_soil_moisture_only(self):
        from sar_validation.core.sar_sources import SAR_SOURCES

        spec = SAR_SOURCES["nisar_sme2"]
        assert spec.variables == frozenset({"soil_moisture"})
        assert spec.output_subdir == "NISAR_L3_SME2"
        assert spec.file_glob == "*.h5"

    def test_template_defaults(self):
        from sar_validation.core.sar_sources import SAR_SOURCES

        spec = SAR_SOURCES["nisar_sme2"]
        assert spec.default_min_depth == 0.0
        assert spec.default_max_depth == 0.05
        assert spec.default_time_tolerance_minutes == 360
        assert spec.default_aggregation_window_km == 0.2
        assert spec.default_spatial_tolerance_km == 2.0

    def test_build_downloader_returns_earthdata_downloader(self, tmp_path):
        from sar_validation.core.sar_sources import SAR_SOURCES

        spec = SAR_SOURCES["nisar_sme2"]
        with patch(
            "sar_validation.downloaders.earthdata_soil_moisture_downloader.EarthdataSoilMoistureDownloader"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            dl = spec.build_downloader(tmp_path, False, True)
        assert dl is mock_cls.return_value
        _, kwargs = mock_cls.call_args
        assert kwargs["output_dir"] == tmp_path
        assert kwargs["dry_run"] is False

    def test_build_downloader_queries_both_beta_and_provisional_collections(self, tmp_path):
        """NISAR SME2's underlying CMR collection changed mid-mission with
        no temporal overlap (confirmed against real CMR data and a
        real-world coverage gap a user hit) -- both must be passed to
        EarthdataSoilMoistureDownloader so a request can find data from
        either era."""
        from sar_validation.core.sar_sources import SAR_SOURCES

        spec = SAR_SOURCES["nisar_sme2"]
        with patch(
            "sar_validation.downloaders.earthdata_soil_moisture_downloader.EarthdataSoilMoistureDownloader"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            spec.build_downloader(tmp_path, False, True)
        _, kwargs = mock_cls.call_args
        assert kwargs["dataset"] == [
            ("NISAR_L3_SME2_BETA_V1", "1"),
            ("NISAR_L3_SME2_PROVISIONAL_V1", "1"),
        ]

    def test_extra_download_kwargs_passes_through_download_kwargs_only(self):
        from sar_validation.core.recipe import SARDataSpec
        from sar_validation.core.sar_sources import SAR_SOURCES

        spec = SAR_SOURCES["nisar_sme2"]
        sd = SARDataSpec(source="nisar_sme2", swath_mode=["IW"], download_kwargs={"version": "001"})
        kwargs = spec.extra_download_kwargs(sd)
        assert kwargs == {"version": "001"}
