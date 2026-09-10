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


class _FakePrediction:
    def __init__(self, verdict: str) -> None:
        self.verdict = verdict


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


class TestBuoyWaterfallDispatch:
    def test_buoy_waterfall_downloads_both_gts_and_copernicus_insitu(self, tmp_path):
        recipe = _make_recipe(tmp_path, ["buoy_waterfall"])
        orch = DataOrchestrator(recipe, dry_run=True)
        orch.base_dir = tmp_path
        orch._previous_downloads = {"sar": {"status": "success", "found_count": 1}}

        with (
            patch.object(DataOrchestrator, "_download_insitu", return_value=True) as mock_insitu,
            patch.object(DataOrchestrator, "_download_gts_buoy", return_value=True) as mock_gts,
        ):
            ok = orch.download_all()

        assert ok is True
        mock_insitu.assert_called_once()
        assert mock_insitu.call_args[0][0] == ["buoy_waterfall"]
        mock_gts.assert_called_once()
        dispatched_source = mock_gts.call_args[0][0]
        assert dispatched_source.source_type == "buoy_waterfall"

    def test_buoy_gts_alone_is_not_double_dispatched(self, tmp_path):
        """A plain "buoy_gts" source must still be dispatched exactly once,
        through _dispatch_source's own handler -- the new explicit
        "buoy_waterfall" loop must not also match it."""
        recipe = _make_recipe(tmp_path, ["buoy_gts"])
        orch = DataOrchestrator(recipe, dry_run=True)
        orch.base_dir = tmp_path
        orch._previous_downloads = {"sar": {"status": "success", "found_count": 1}}

        with patch.object(DataOrchestrator, "_download_gts_buoy", return_value=True) as mock_gts:
            ok = orch.download_all()

        assert ok is True
        mock_gts.assert_called_once()

    def test_buoy_waterfall_gts_side_skipped_when_no_collocation_predicted(self, tmp_path, monkeypatch):
        """The buoy_waterfall GTS download must honor the same
        collocation-prediction skip gate as every other dispatch path in
        download_all(), instead of dispatching unconditionally."""
        recipe = _make_recipe(tmp_path, ["buoy_waterfall"])
        orch = DataOrchestrator(recipe, dry_run=True)
        orch.base_dir = tmp_path
        orch._previous_downloads = {"sar": {"status": "success", "found_count": 1}}
        monkeypatch.setattr(
            orch, "_collocation_predictions",
            lambda: {"buoy_waterfall": _FakePrediction(verdict="none-predicted")},
        )

        with patch.object(DataOrchestrator, "_download_gts_buoy", return_value=True) as mock_gts:
            ok = orch.download_all()

        assert ok is True
        mock_gts.assert_not_called()
        assert orch.metadata["downloads"]["buoy_waterfall"]["status"] == "skipped"
        assert "collocation" in orch.metadata["downloads"]["buoy_waterfall"]["reason"]

    def test_buoy_waterfall_gts_side_skipped_when_already_succeeded(self, tmp_path):
        """A buoy_waterfall GTS download that already succeeded in a
        previous run must not be re-dispatched, matching the
        _already_succeeded gate used by every other dispatch path."""
        recipe = _make_recipe(tmp_path, ["buoy_waterfall"])
        orch = DataOrchestrator(recipe, dry_run=True)
        orch.base_dir = tmp_path
        orch._previous_downloads = {
            "sar": {"status": "success", "found_count": 1},
            "buoy_waterfall": {"status": "success", "file_count": 3},
        }

        with patch.object(DataOrchestrator, "_download_gts_buoy", return_value=True) as mock_gts:
            ok = orch.download_all()

        assert ok is True
        mock_gts.assert_not_called()
        assert orch.metadata["downloads"]["buoy_waterfall"] == {"status": "success", "file_count": 3}

    def test_buoy_waterfall_gts_metadata_recorded_under_own_source_type(self, tmp_path):
        """A buoy_waterfall source's GTS download must record its success
        under the "buoy_waterfall" metadata key, not under a different
        validation source's key, since _already_succeeded looks up a
        prior run's status by the calling source's own source_type."""
        recipe = _make_recipe(tmp_path, ["buoy_waterfall"])
        orch = DataOrchestrator(recipe, dry_run=True)
        orch.base_dir = tmp_path
        orch._previous_downloads = {"sar": {"status": "success", "found_count": 1}}

        with (
            patch.object(DataOrchestrator, "_download_insitu", return_value=True),
            patch(
                "sar_validation.downloaders.gts_buoy_downloader.GTSBuoyDownloader"
            ) as mock_cls,
        ):
            mock_dl = MagicMock()
            mock_dl.download.return_value = []
            mock_cls.return_value = mock_dl

            ok = orch.download_all()

        assert ok is True
        assert orch.metadata["downloads"]["buoy_waterfall"]["status"] == "dry_run"
        assert "buoy_gts" not in orch.metadata["downloads"]

    def test_buoy_waterfall_gts_real_success_gates_next_run(self, tmp_path):
        """A buoy_waterfall GTS success recorded under the
        "buoy_waterfall" metadata key, as a real download produces, must
        cause a subsequent run's _already_succeeded gate to skip
        re-dispatching GTSBuoyDownloader entirely."""
        recipe = _make_recipe(tmp_path, ["buoy_waterfall"])
        orch = DataOrchestrator(recipe, dry_run=True)
        orch.base_dir = tmp_path
        orch._previous_downloads = {
            "sar": {"status": "success", "found_count": 1},
            "buoy_waterfall": {"status": "success", "files": []},
        }

        with (
            patch.object(DataOrchestrator, "_download_insitu", return_value=True),
            patch(
                "sar_validation.downloaders.gts_buoy_downloader.GTSBuoyDownloader"
            ) as mock_cls,
        ):
            ok = orch.download_all()

        assert ok is True
        mock_cls.assert_not_called()
        assert orch.metadata["downloads"]["buoy_waterfall"] == {"status": "success", "files": []}
