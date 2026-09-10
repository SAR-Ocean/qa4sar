"""Tests for the orchestrator's 'buoy_gts' source_type dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from sar_validation.core.orchestrator import DataOrchestrator
from sar_validation.core.recipe import (
    GeographicBounds,
    Recipe,
    RecipeConfig,
    TemporalBounds,
    ValidationDataSource,
)


def _make_recipe(tmp_path, source_types):
    config = RecipeConfig(
        name="test-gts-buoy",
        variable="wind",
        geographic_bounds=GeographicBounds(min_lon=-10, max_lon=10, min_lat=40, max_lat=55),
        temporal_bounds=TemporalBounds(start="2026-08-30T00:00:00", end="2026-08-30T12:00:00"),
        validation_sources=[ValidationDataSource(source_type=t) for t in source_types],
    )
    return Recipe(config=config)


class TestBuoyGtsDispatch:
    def test_buoy_gts_source_type_calls_gts_buoy_downloader(self, tmp_path):
        recipe = _make_recipe(tmp_path, ["buoy_gts"])
        orch = DataOrchestrator(recipe, dry_run=True)
        orch.base_dir = tmp_path

        with patch(
            "sar_validation.downloaders.gts_buoy_downloader.GTSBuoyDownloader"
        ) as mock_cls:
            mock_dl = MagicMock()
            mock_dl.download.return_value = []
            mock_cls.return_value = mock_dl

            source = recipe.config.validation_sources[0]
            ok = orch._dispatch_source(source)

        assert ok is True
        mock_cls.assert_called_once()
        mock_dl.download.assert_called_once()
        assert orch.metadata["downloads"]["buoy_gts"]["status"] == "dry_run"

    def test_buoy_gts_output_dir_is_gts_buoy_subdir(self, tmp_path):
        recipe = _make_recipe(tmp_path, ["buoy_gts"])
        orch = DataOrchestrator(recipe, dry_run=True)
        orch.base_dir = tmp_path

        with patch(
            "sar_validation.downloaders.gts_buoy_downloader.GTSBuoyDownloader"
        ) as mock_cls:
            mock_dl = MagicMock()
            mock_dl.download.return_value = []
            mock_cls.return_value = mock_dl

            source = recipe.config.validation_sources[0]
            orch._dispatch_source(source)

        _, kwargs = mock_cls.call_args
        assert kwargs["output_dir"] == tmp_path / "gts_buoy"
