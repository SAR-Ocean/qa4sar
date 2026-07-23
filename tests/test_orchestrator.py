"""Tests for DataOrchestrator's SAR product_level branching."""

from __future__ import annotations

import logging
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


class TestCleanupIfEmpty:
    def test_removes_dir_with_no_files_including_nested_empty_subdirs(self, tmp_path):
        recipe = _recipe("L2_OCN")
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        empty_dir = tmp_path / "empty"
        (empty_dir / "nested").mkdir(parents=True)

        orchestrator._cleanup_if_empty(empty_dir)

        assert not empty_dir.exists()

    def test_keeps_dir_containing_a_file(self, tmp_path):
        recipe = _recipe("L2_OCN")
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        populated_dir = tmp_path / "populated"
        populated_dir.mkdir()
        (populated_dir / "data.nc").write_text("x")

        orchestrator._cleanup_if_empty(populated_dir)

        assert populated_dir.exists()
        assert (populated_dir / "data.nc").exists()

    def test_noop_when_dir_does_not_exist(self, tmp_path):
        recipe = _recipe("L2_OCN")
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        missing_dir = tmp_path / "missing"

        orchestrator._cleanup_if_empty(missing_dir)  # must not raise

        assert not missing_dir.exists()


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


class TestScatterometerHandlerCleansUpEmptyOutputDir:
    def test_removes_output_dir_when_downloader_produced_no_files(self, tmp_path):
        recipe = _recipe("L2_OCN")
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.base_dir = tmp_path

        with patch(
            "sar_validation.downloaders.scatterometer_downloader.ScatterometerDownloader"
        ) as mock_cls:
            mock_instance = MagicMock()

            def fake_download(**kwargs):
                # Real downloaders create output_dir unconditionally before
                # they know whether any data will land in it -- simulate
                # that here.
                (tmp_path / "osi_saf_winds").mkdir(parents=True, exist_ok=True)
                return []

            mock_instance.download.side_effect = fake_download
            mock_cls.return_value = mock_instance

            ok = orchestrator._download_scatterometer(None)

        assert ok is True
        assert not (tmp_path / "osi_saf_winds").exists()

    def test_keeps_output_dir_when_downloader_produced_a_file(self, tmp_path):
        recipe = _recipe("L2_OCN")
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.base_dir = tmp_path

        with patch(
            "sar_validation.downloaders.scatterometer_downloader.ScatterometerDownloader"
        ) as mock_cls:
            mock_instance = MagicMock()

            def fake_download(**kwargs):
                out = tmp_path / "osi_saf_winds"
                out.mkdir(parents=True, exist_ok=True)
                (out / "scat.nc").write_text("x")
                return [out / "scat.nc"]

            mock_instance.download.side_effect = fake_download
            mock_cls.return_value = mock_instance

            ok = orchestrator._download_scatterometer(None)

        assert ok is True
        assert (tmp_path / "osi_saf_winds" / "scat.nc").exists()


class TestCombinedCurrentsHistoricalStatusMessage:
    def _recipe_with(self, *source_types: str) -> Recipe:
        cfg = RecipeConfig(
            name="test", variable="currents",
            geographic_bounds=GeographicBounds(-10.0, 20.0, 40.0, 55.0),
            temporal_bounds=TemporalBounds("2020-01-01", "2020-01-02"),
            validation_sources=[
                ValidationDataSource(source_type=st) for st in source_types
            ],
        )
        return Recipe(cfg)

    def test_all_empty_logs_one_combined_warning(self, caplog):
        recipe = self._recipe_with("adcp_historical", "argo_historical")
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.metadata["downloads"]["adcp_historical"] = {"status": "success", "file_count": 0}
        orchestrator.metadata["downloads"]["argo_historical"] = {"status": "success", "file_count": 0}

        with caplog.at_level(logging.WARNING):
            orchestrator._report_combined_currents_status()

        combined = [
            r for r in caplog.records
            if "No delayed-mode in-situ current data found" in r.message
        ]
        assert len(combined) == 1
        assert "adcp" in combined[0].message
        assert "argo" in combined[0].message

    def test_all_empty_also_records_a_notice(self, caplog):
        """The message must persist into download_metadata.json (via
        metadata["notices"]) too, not just the live console log -- so it
        survives past the point where later pipeline steps (convert,
        collocate, stats, plot) scroll it out of the terminal. Must NOT go
        into metadata["errors"]: that list makes _is_already_downloaded
        (cli.py) treat a run as failed and always re-download, which "no
        data available" should not trigger."""
        recipe = self._recipe_with("adcp_historical", "argo_historical")
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.metadata["downloads"]["adcp_historical"] = {"status": "success", "file_count": 0}
        orchestrator.metadata["downloads"]["argo_historical"] = {"status": "success", "file_count": 0}

        orchestrator._report_combined_currents_status()

        assert len(orchestrator.metadata["notices"]) == 1
        assert "adcp" in orchestrator.metadata["notices"][0]
        assert "argo" in orchestrator.metadata["notices"][0]
        assert orchestrator.metadata["errors"] == []

    def test_one_nonempty_suppresses_combined_warning(self, caplog):
        recipe = self._recipe_with("adcp_historical", "argo_historical")
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.metadata["downloads"]["adcp_historical"] = {"status": "success", "file_count": 2}
        orchestrator.metadata["downloads"]["argo_historical"] = {"status": "success", "file_count": 0}

        with caplog.at_level(logging.WARNING):
            orchestrator._report_combined_currents_status()

        assert not any(
            "No delayed-mode in-situ current data found" in r.message for r in caplog.records
        )

    def test_message_names_only_instruments_present_in_recipe(self, caplog):
        recipe = self._recipe_with("drifter_historical")
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.metadata["downloads"]["drifter_historical"] = {"status": "success", "file_count": 0}

        with caplog.at_level(logging.WARNING):
            orchestrator._report_combined_currents_status()

        combined = [
            r for r in caplog.records
            if "No delayed-mode in-situ current data found" in r.message
        ]
        assert len(combined) == 1
        assert combined[0].message.strip().startswith(
            "No delayed-mode in-situ current data found (drifter)"
        )

    def test_dry_run_never_logs_combined_warning(self, caplog):
        recipe = self._recipe_with("adcp_historical")
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        orchestrator.metadata["downloads"]["adcp_historical"] = {"status": "dry_run", "file_count": 0}

        with caplog.at_level(logging.WARNING):
            orchestrator._report_combined_currents_status()

        assert not any(
            "No delayed-mode in-situ current data found" in r.message for r in caplog.records
        )

    def test_failed_instrument_suppresses_combined_warning(self, caplog):
        """A real failure (network error, etc.) is a different problem than
        'no data' and must not be folded into the combined message."""
        recipe = self._recipe_with("adcp_historical")
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.metadata["downloads"]["adcp_historical"] = {"status": "failed", "error": "boom"}

        with caplog.at_level(logging.WARNING):
            orchestrator._report_combined_currents_status()

        assert not any(
            "No delayed-mode in-situ current data found" in r.message for r in caplog.records
        )
