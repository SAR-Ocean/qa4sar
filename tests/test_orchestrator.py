"""Tests for DataOrchestrator's SAR product_level branching."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from sar_validation.core.orchestrator import DataOrchestrator
from sar_validation.core.recipe import (
    GeographicBounds,
    Recipe,
    RecipeConfig,
    SARDataSpec,
    TemporalBounds,
    ValidationDataSource,
)


def _recipe(product_level: str) -> Recipe:
    cfg = RecipeConfig(
        name="test",
        variable="soil_moisture" if product_level == "L3_SSM" else "wind",
        geographic_bounds=GeographicBounds(-10.0, 20.0, 40.0, 55.0),
        temporal_bounds=TemporalBounds("2026-01-01", "2026-01-02"),
        sar_data=SARDataSpec(product_level=product_level),
    )
    return Recipe(cfg)


class TestDownloadSarProductLevelBranch:
    def test_l3_ssm_uses_soil_moisture_downloader(self, tmp_path):
        recipe = _recipe("L3_SSM")
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        orchestrator.base_dir = tmp_path

        with patch(
            "sar_validation.downloaders.soil_moisture_downloader.SoilMoistureDownloader"
        ) as mock_cls:
            mock_instance = MagicMock()
            mock_instance.download.return_value = []
            mock_cls.return_value = mock_instance

            ok = orchestrator._download_sar()

        assert ok is True
        mock_cls.assert_called_once()
        assert mock_cls.call_args.kwargs["output_dir"] == tmp_path / "S1_L3_SSM"

    def test_l2_ocn_uses_sar_downloader(self, tmp_path):
        recipe = _recipe("L2_OCN")
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        orchestrator.base_dir = tmp_path

        with patch(
            "sar_validation.downloaders.sar_downloader.SARDownloader"
        ) as mock_cls:
            mock_instance = MagicMock()
            mock_instance.download.return_value = []
            mock_cls.return_value = mock_instance

            ok = orchestrator._download_sar()

        assert ok is True
        mock_cls.assert_called_once()
        assert mock_cls.call_args.kwargs["output_dir"] == tmp_path / "S1_L2_OCN"

    def test_l3_ssm_failure_is_recorded_in_metadata(self, tmp_path):
        recipe = _recipe("L3_SSM")
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        orchestrator.base_dir = tmp_path

        with patch(
            "sar_validation.downloaders.soil_moisture_downloader.SoilMoistureDownloader"
        ) as mock_cls:
            mock_cls.side_effect = RuntimeError("boom")
            ok = orchestrator._download_sar()

        assert ok is False
        assert orchestrator.metadata["downloads"]["sar"]["status"] == "failed"
        assert "boom" in orchestrator.metadata["errors"][0]


class TestDownloadIsmnDispatch:
    def test_ismn_source_type_dispatches_to_ismn_downloader(self, tmp_path):
        cfg = RecipeConfig(
            name="test", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-10.0, 20.0, 40.0, 55.0),
            temporal_bounds=TemporalBounds("2026-01-01", "2026-01-02"),
            validation_sources=[
                ValidationDataSource(
                    source_type="ismn", min_depth=0.0, max_depth=0.05,
                    download_kwargs={"ismn_archive_path": "/tmp/archive.zip"},
                )
            ],
        )
        recipe = Recipe(cfg)
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        orchestrator.base_dir = tmp_path

        with patch(
            "sar_validation.downloaders.ismn_downloader.ISMNDownloader"
        ) as mock_cls:
            mock_instance = MagicMock()
            mock_instance.download.return_value = []
            mock_cls.return_value = mock_instance

            ok = orchestrator._dispatch_source(cfg.validation_sources[0])

        assert ok is True
        call_kwargs = mock_instance.download.call_args.kwargs
        assert call_kwargs["archive_path"] == "/tmp/archive.zip"
        assert call_kwargs["min_depth"] == 0.0
        assert call_kwargs["max_depth"] == 0.05

    def test_unconfigured_ismn_archive_path_forwards_none(self, tmp_path):
        cfg = RecipeConfig(
            name="test", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-10.0, 20.0, 40.0, 55.0),
            temporal_bounds=TemporalBounds("2026-01-01", "2026-01-02"),
            validation_sources=[ValidationDataSource(source_type="ismn")],
        )
        recipe = Recipe(cfg)
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        orchestrator.base_dir = tmp_path

        with patch(
            "sar_validation.downloaders.ismn_downloader.ISMNDownloader"
        ) as mock_cls:
            mock_instance = MagicMock()
            mock_instance.download.return_value = []
            mock_cls.return_value = mock_instance

            ok = orchestrator._dispatch_source(cfg.validation_sources[0])

        assert ok is True
        assert mock_instance.download.call_args.kwargs["archive_path"] is None


class TestIsmnDownloadStatusReporting:
    """No ISMN archive collected must not be reported as 'success' -- that
    misled Lotte into thinking a run had worked when the ISMN step had
    silently collected zero files (still awaiting a manually-downloaded
    archive)."""

    def _recipe(self) -> Recipe:
        cfg = RecipeConfig(
            name="test", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-10.0, 20.0, 40.0, 55.0),
            temporal_bounds=TemporalBounds("2026-01-01", "2026-01-02"),
            validation_sources=[
                ValidationDataSource(source_type="ismn", min_depth=0.0, max_depth=0.05),
            ],
        )
        return Recipe(cfg)

    def test_zero_files_reports_awaiting_manual_archive(self, tmp_path):
        recipe = self._recipe()
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.base_dir = tmp_path

        with patch(
            "sar_validation.downloaders.ismn_downloader.ISMNDownloader"
        ) as mock_cls:
            mock_instance = MagicMock()
            mock_instance.download.return_value = []
            mock_cls.return_value = mock_instance

            ok = orchestrator._download_ismn(recipe.config.validation_sources[0])

        assert ok is True
        assert orchestrator.metadata["downloads"]["ismn"]["status"] == "awaiting_manual_archive"

    def test_nonzero_files_reports_success(self, tmp_path):
        recipe = self._recipe()
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.base_dir = tmp_path

        with patch(
            "sar_validation.downloaders.ismn_downloader.ISMNDownloader"
        ) as mock_cls:
            mock_instance = MagicMock()
            mock_instance.download.return_value = [tmp_path / "ismn_station.csv"]
            mock_cls.return_value = mock_instance

            ok = orchestrator._download_ismn(recipe.config.validation_sources[0])

        assert ok is True
        assert orchestrator.metadata["downloads"]["ismn"]["status"] == "success"

    def test_dry_run_reports_dry_run_status_even_with_zero_files(self, tmp_path):
        recipe = self._recipe()
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        orchestrator.base_dir = tmp_path

        with patch(
            "sar_validation.downloaders.ismn_downloader.ISMNDownloader"
        ) as mock_cls:
            mock_instance = MagicMock()
            mock_instance.download.return_value = []
            mock_cls.return_value = mock_instance

            orchestrator._download_ismn(recipe.config.validation_sources[0])

        assert orchestrator.metadata["downloads"]["ismn"]["status"] == "dry_run"
