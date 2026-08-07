"""Tests for the SAR_SOURCES registry (sar_validation.core.sar_sources)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestSentinel1L2OcnDownloaderWiring:
    def test_build_downloader_returns_sar_downloader(self, tmp_path):
        from sar_validation.core.sar_sources import SAR_SOURCES

        spec = SAR_SOURCES["sentinel1_l2_ocn"]
        with patch("sar_validation.downloaders.sentinel1_l2_ocn_downloader.SARDownloader") as mock_cls:
            mock_cls.return_value = MagicMock()
            dl = spec.build_downloader(tmp_path, False, True)
        mock_cls.assert_called_once_with(output_dir=tmp_path, dry_run=False, force_download=True)
        assert dl is mock_cls.return_value

    @pytest.mark.parametrize(
        "swath_mode,max_downloads,download_kwargs,expected_kwargs",
        [
            (["IW", "EW"], 5, {"top": 50}, {"modes": ["IW", "EW"], "limit": 5, "top": 50}),
            ([], None, {}, {"modes": None, "limit": None}),
        ],
        ids=["maps_swath_mode_and_max_downloads", "none_swath_mode_becomes_none"],
    )
    def test_extra_download_kwargs(self, swath_mode, max_downloads, download_kwargs, expected_kwargs):
        from sar_validation.core.recipe import SARDataSpec
        from sar_validation.core.sar_sources import SAR_SOURCES

        spec = SAR_SOURCES["sentinel1_l2_ocn"]
        sd = SARDataSpec(swath_mode=swath_mode, max_downloads=max_downloads, download_kwargs=download_kwargs)
        kwargs = spec.extra_download_kwargs(sd)
        assert kwargs == expected_kwargs


class TestSentinel1ClmsSsmDownloaderWiring:
    def test_build_downloader_returns_soil_moisture_downloader(self, tmp_path):
        from sar_validation.core.sar_sources import SAR_SOURCES

        spec = SAR_SOURCES["sentinel1_clms_ssm"]
        with patch(
            "sar_validation.downloaders.sentinel1_soil_moisture_downloader.SoilMoistureDownloader"
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
    def test_build_downloader_returns_earthdata_downloader_querying_both_collections(self, tmp_path):
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
            dl = spec.build_downloader(tmp_path, False, True)
        assert dl is mock_cls.return_value
        _, kwargs = mock_cls.call_args
        assert kwargs["output_dir"] == tmp_path
        assert kwargs["dry_run"] is False
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


class TestSarSourceSatelliteField:
    """Each registry entry records which satellite family it belongs to,
    so --sar-source can accept a satellite name (e.g. "sentinel1") instead
    of forcing users to know the internal product-specific key."""

    @pytest.mark.parametrize("key, satellite", [
        ("sentinel1_l2_ocn", "sentinel1"),
        ("sentinel1_clms_ssm", "sentinel1"),
        ("nisar_sme2", "nisar"),
        ("radarsat2", "radarsat2"),
    ])
    def test_satellite_field(self, key, satellite):
        from sar_validation.core.sar_sources import SAR_SOURCES

        assert SAR_SOURCES[key].satellite == satellite


class TestAvailableSatellites:
    def test_lists_sentinel1_nisar_and_radarsat2(self):
        from sar_validation.core.sar_sources import AVAILABLE_SATELLITES

        assert AVAILABLE_SATELLITES == ["nisar", "radarsat2", "sentinel1"]


class TestResolveSarSource:
    """resolve_sar_source(name, variable) is the single place --sar-source
    CLI values get turned into an internal SAR_SOURCES key -- accepting
    either a satellite family name (the new, user-facing convention) or an
    exact internal key (kept working for backward compatibility /
    recipe.yaml's own stored sar_data.source values)."""

    @pytest.mark.parametrize("name, variable, expected", [
        ("sentinel1", "wind", "sentinel1_l2_ocn"),
        ("sentinel1", "waves", "sentinel1_l2_ocn"),
        ("sentinel1", "currents", "sentinel1_l2_ocn"),
        ("sentinel1", "soil_moisture", "sentinel1_clms_ssm"),
        ("nisar", "soil_moisture", "nisar_sme2"),
        ("sentinel1_clms_ssm", "soil_moisture", "sentinel1_clms_ssm"),
    ])
    def test_resolves(self, name, variable, expected):
        from sar_validation.core.sar_sources import resolve_sar_source

        assert resolve_sar_source(name, variable) == expected

    @pytest.mark.parametrize("name, variable, match", [
        ("nisar", "wind", "no product"),
        ("sentinel1_clms_ssm", "wind", "only valid for"),
        ("bogus", "wind", "sentinel1"),
    ])
    def test_raises(self, name, variable, match):
        from sar_validation.core.sar_sources import resolve_sar_source

        with pytest.raises(ValueError, match=match):
            resolve_sar_source(name, variable)


class TestRegistryCompleteness:
    def test_registry_has_the_four_registered_sources(self):
        from sar_validation.core.sar_sources import SAR_SOURCES

        assert set(SAR_SOURCES) == {
            "sentinel1_l2_ocn", "sentinel1_clms_ssm", "nisar_sme2", "radarsat2",
        }


class TestRadarsat2RegistryEntry:
    def test_applies_to_wind_only(self):
        from sar_validation.core.sar_sources import SAR_SOURCES

        spec = SAR_SOURCES["radarsat2"]
        assert spec.variables == frozenset({"wind"})
        assert spec.output_subdir == "RADARSAT2_WIND"
        assert spec.file_glob == "*.nc"

    def test_no_soil_moisture_defaults(self):
        from sar_validation.core.sar_sources import SAR_SOURCES

        spec = SAR_SOURCES["radarsat2"]
        assert spec.default_min_depth is None
        assert spec.default_time_tolerance_minutes is None

    def test_build_downloader_returns_radarsat2_downloader(self, tmp_path):
        from sar_validation.core.sar_sources import SAR_SOURCES

        spec = SAR_SOURCES["radarsat2"]
        with patch(
            "sar_validation.downloaders.radarsat2_wind_downloader.RADARSAT2WindDownloader"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            dl = spec.build_downloader(tmp_path, False, True)
        mock_cls.assert_called_once_with(output_dir=tmp_path, dry_run=False, force_download=True)
        assert dl is mock_cls.return_value

    def test_extra_download_kwargs_passes_through_download_kwargs_only(self):
        from sar_validation.core.recipe import SARDataSpec
        from sar_validation.core.sar_sources import SAR_SOURCES

        spec = SAR_SOURCES["radarsat2"]
        sd = SARDataSpec(source="radarsat2", swath_mode=["IW"], download_kwargs={"foo": "bar"})
        kwargs = spec.extra_download_kwargs(sd)
        assert kwargs == {"foo": "bar"}

    def test_convert_calls_from_radarsat2_wind(self, tmp_path):
        from sar_validation.core.sar_sources import SAR_SOURCES

        spec = SAR_SOURCES["radarsat2"]
        nc_path = tmp_path / "scene.nc"
        with patch(
            "sar_validation.core.datatree_converter.DataTreeConverter.from_radarsat2_wind"
        ) as mock_fn:
            mock_fn.return_value = "fake_dataset"
            result = spec.convert(nc_path, "wind")
        mock_fn.assert_called_once_with(nc_path, "wind")
        assert result == "fake_dataset"
