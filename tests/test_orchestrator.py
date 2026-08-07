"""Tests for DataOrchestrator's SAR product_level branching."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from sar_validation.core.orchestrator import DataOrchestrator
from sar_validation.core.recipe import (
    GeographicBounds,
    Recipe,
    RecipeConfig,
    SARDataSpec,
    TemporalBounds,
    ValidationDataSource,
)


def _recipe(source: str) -> Recipe:
    cfg = RecipeConfig(
        name="test",
        variable="soil_moisture" if source == "sentinel1_clms_ssm" else "wind",
        geographic_bounds=GeographicBounds(-10.0, 20.0, 40.0, 55.0),
        temporal_bounds=TemporalBounds("2026-01-01", "2026-01-02"),
        sar_data=SARDataSpec(source=source),
    )
    return Recipe(cfg)


class TestCleanupIfEmpty:
    def test_removes_dir_with_no_files_including_nested_empty_subdirs(self, tmp_path):
        recipe = _recipe("sentinel1_l2_ocn")
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        empty_dir = tmp_path / "empty"
        (empty_dir / "nested").mkdir(parents=True)

        orchestrator._cleanup_if_empty(empty_dir)

        assert not empty_dir.exists()

    def test_keeps_dir_containing_a_file(self, tmp_path):
        recipe = _recipe("sentinel1_l2_ocn")
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        populated_dir = tmp_path / "populated"
        populated_dir.mkdir()
        (populated_dir / "data.nc").write_text("x")

        orchestrator._cleanup_if_empty(populated_dir)

        assert populated_dir.exists()
        assert (populated_dir / "data.nc").exists()

    def test_noop_when_dir_does_not_exist(self, tmp_path):
        recipe = _recipe("sentinel1_l2_ocn")
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        missing_dir = tmp_path / "missing"

        orchestrator._cleanup_if_empty(missing_dir)  # must not raise

        assert not missing_dir.exists()


class TestRunDownload:
    def test_success_records_files_and_status(self, tmp_path):
        orch = DataOrchestrator(_recipe("sentinel1_l2_ocn"), dry_run=True)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "f.nc").touch()

        class FakeDl:
            def download(self, **kwargs):
                return [out_dir / "f.nc"]

        ok = orch._run_download(
            "test_key", out_dir, lambda: FakeDl(), lambda: {}, "Test",
        )
        assert ok is True
        assert orch.metadata["downloads"]["test_key"]["status"] == "dry_run"
        assert orch.metadata["downloads"]["test_key"]["files"] == [str(out_dir / "f.nc")]

    def test_failure_records_error(self, tmp_path):
        orch = DataOrchestrator(_recipe("sentinel1_l2_ocn"), dry_run=True)
        out_dir = tmp_path / "out"

        class FailingDl:
            def download(self, **kwargs):
                raise RuntimeError("boom")

        ok = orch._run_download(
            "test_key", out_dir, lambda: FailingDl(), lambda: {}, "Test",
        )
        assert ok is False
        assert orch.metadata["downloads"]["test_key"]["status"] == "failed"
        assert "Test download failed: boom" in orch.metadata["downloads"]["test_key"]["error"]

    def test_result_to_metadata_override(self, tmp_path):
        orch = DataOrchestrator(_recipe("sentinel1_l2_ocn"), dry_run=True)
        out_dir = tmp_path / "out"

        class FakeDl:
            def download(self, **kwargs):
                return [1, 2, 3]

        ok = orch._run_download(
            "test_key", out_dir, lambda: FakeDl(), lambda: {}, "Test",
            result_to_metadata=lambda result, dl: {"file_count": len(result)},
        )
        assert ok is True
        assert orch.metadata["downloads"]["test_key"]["file_count"] == 3
        assert "files" not in orch.metadata["downloads"]["test_key"]

    def test_build_kwargs_exception_records_failure_not_raised(self, tmp_path):
        """A bad recipe (e.g. download_kwargs isn't actually a dict) can make
        kwargs construction itself raise. That must be caught the same as a
        downloader/.download() failure -- recorded as a per-source failure --
        rather than propagating uncaught through _dispatch_source."""
        orch = DataOrchestrator(_recipe("sentinel1_l2_ocn"), dry_run=True)
        out_dir = tmp_path / "out"

        class FakeDl:
            def download(self, **kwargs):
                return []

        def build_kwargs():
            raise ValueError("bad kwargs")

        ok = orch._run_download(
            "test_key", out_dir, lambda: FakeDl(), build_kwargs, "Test",
        )
        assert ok is False
        assert orch.metadata["downloads"]["test_key"]["status"] == "failed"
        assert "Test download failed: bad kwargs" in orch.metadata["downloads"]["test_key"]["error"]


class TestDownloadSarSourceBranch:
    @pytest.mark.parametrize(
        "source,patch_path,expected_output_subdir",
        [
            pytest.param(
                "sentinel1_clms_ssm",
                "sar_validation.downloaders.sentinel1_soil_moisture_downloader.SoilMoistureDownloader",
                "S1_L3_SSM",
                id="l3_ssm_uses_soil_moisture_downloader",
            ),
            pytest.param(
                "sentinel1_l2_ocn",
                "sar_validation.downloaders.sentinel1_l2_ocn_downloader.SARDownloader",
                "S1_L2_OCN",
                id="l2_ocn_uses_sar_downloader",
            ),
        ],
    )
    def test_dispatches_to_the_downloader_matching_the_sar_source(
        self, tmp_path, source, patch_path, expected_output_subdir
    ):
        recipe = _recipe(source)
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        orchestrator.base_dir = tmp_path

        with patch(patch_path) as mock_cls:
            mock_instance = MagicMock()
            mock_instance.download.return_value = []
            mock_cls.return_value = mock_instance

            ok = orchestrator._download_sar()

        assert ok is True
        mock_cls.assert_called_once()
        assert mock_cls.call_args.kwargs["output_dir"] == tmp_path / expected_output_subdir

    def test_l3_ssm_failure_is_recorded_in_metadata(self, tmp_path):
        recipe = _recipe("sentinel1_clms_ssm")
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        orchestrator.base_dir = tmp_path

        with patch(
            "sar_validation.downloaders.sentinel1_soil_moisture_downloader.SoilMoistureDownloader"
        ) as mock_cls:
            mock_cls.side_effect = RuntimeError("boom")
            ok = orchestrator._download_sar()

        assert ok is False
        assert orchestrator.metadata["downloads"]["sar"]["status"] == "failed"
        assert "boom" in orchestrator.metadata["errors"][0]


class TestDownloadIsmnDispatch:
    @pytest.mark.parametrize(
        "validation_source_kwargs,expected_archive_path",
        [
            pytest.param(
                {
                    "min_depth": 0.0, "max_depth": 0.05,
                    "download_kwargs": {"ismn_archive_path": "/tmp/archive.zip"},
                },
                "/tmp/archive.zip",
                id="configured_archive_path",
            ),
            pytest.param({}, None, id="unconfigured_archive_path_forwards_none"),
        ],
    )
    def test_ismn_source_dispatches_with_expected_archive_path(
        self, tmp_path, validation_source_kwargs, expected_archive_path
    ):
        cfg = RecipeConfig(
            name="test", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-10.0, 20.0, 40.0, 55.0),
            temporal_bounds=TemporalBounds("2026-01-01", "2026-01-02"),
            validation_sources=[
                ValidationDataSource(source_type="ismn", **validation_source_kwargs),
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
        assert call_kwargs["archive_path"] == expected_archive_path
        if "min_depth" in validation_source_kwargs:
            assert call_kwargs["min_depth"] == validation_source_kwargs["min_depth"]
            assert call_kwargs["max_depth"] == validation_source_kwargs["max_depth"]


class TestDownloadAscatSsmDispatch:
    def test_ascat_ssm_source_type_dispatches_to_ascat_ssm_downloader(self, tmp_path):
        cfg = RecipeConfig(
            name="test", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-10.0, 20.0, 40.0, 55.0),
            temporal_bounds=TemporalBounds("2026-01-01", "2026-01-02"),
            validation_sources=[ValidationDataSource(source_type="ascat_ssm")],
        )
        recipe = Recipe(cfg)
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        orchestrator.base_dir = tmp_path

        with patch(
            "sar_validation.downloaders.ascat_soil_moisture_downloader.ASCATSoilMoistureDownloader"
        ) as mock_cls:
            mock_instance = MagicMock()
            mock_instance.download.return_value = []
            mock_cls.return_value = mock_instance

            ok = orchestrator._dispatch_source(cfg.validation_sources[0])

        assert ok is True
        mock_cls.assert_called_once()
        assert mock_cls.call_args.kwargs["output_dir"] == tmp_path / "ascat_ssm"
        assert orchestrator.metadata["downloads"]["ascat_ssm"]["status"] == "dry_run"


class TestDownloadTemporalPadding:
    """A multi-day soil_moisture recipe's first/last SAR scene needs
    validation data from *outside* the literal requested date range to
    fill its own +-time_tolerance collocation window -- e.g. day 1's
    window extends 12h before the requested start. Previously,
    ascat_ssm/amsr_ssm/smap_ssm download requests used the recipe's raw
    start/end with no margin, so the first/last SAR scene in a run got
    silently starved of validation data relative to scenes in the middle
    of the range (visible as day 1/day 3 having far fewer matches, and
    far less "unmatched" clutter on the diagnostics plot, than day 2).
    Fixed by padding every validation-source download's start/end by that
    source's own resolved collocation time-tolerance."""

    def _recipe(self, **collocation_kwargs) -> Recipe:
        from sar_validation.core.recipe import CollocationType

        cfg = RecipeConfig(
            name="test", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-10.0, 20.0, 40.0, 55.0),
            temporal_bounds=TemporalBounds("2026-02-01", "2026-02-03"),
            sar_data=SARDataSpec(source="sentinel1_clms_ssm"),
            validation_sources=[
                ValidationDataSource(source_type="ascat_ssm"),
                ValidationDataSource(source_type="ismn"),
            ],
            collocation=CollocationType(**collocation_kwargs) if collocation_kwargs else CollocationType(),
        )
        return Recipe(cfg)

    def test_resolve_temporal_padding_minutes_uses_default_layer_type_spec(self):
        from sar_validation.core.orchestrator import _resolve_temporal_padding_minutes

        cfg = self._recipe().config
        # ascat_ssm's spec lives under "scatterometer_ssm" (its data_type
        # tag), not its own source_type -- built into
        # DEFAULT_LAYER_TYPE_SPECS regardless of recipe overrides.
        assert _resolve_temporal_padding_minutes(cfg, "ascat_ssm") == 720.0
        assert _resolve_temporal_padding_minutes(cfg, "amsr_ssm") == 720.0
        assert _resolve_temporal_padding_minutes(cfg, "smap_ssm") == 720.0
        assert _resolve_temporal_padding_minutes(cfg, "smos_ssm") == 720.0

    def test_resolve_temporal_padding_minutes_falls_back_to_point_vs_layer_default(self):
        from sar_validation.core.orchestrator import _resolve_temporal_padding_minutes

        cfg = self._recipe().config
        # "ismn" has no DEFAULT_LAYER_TYPE_SPECS entry -- falls back to the
        # (unconfigured, so 30-min default) point_vs_layer tolerance.
        assert _resolve_temporal_padding_minutes(cfg, "ismn") == 30.0

    def test_resolve_temporal_padding_minutes_per_source_override_wins(self):
        from sar_validation.core.orchestrator import _resolve_temporal_padding_minutes
        from sar_validation.core.recipe import CollocationType

        cfg = RecipeConfig(
            name="test", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-10.0, 20.0, 40.0, 55.0),
            temporal_bounds=TemporalBounds("2026-02-01", "2026-02-03"),
            validation_sources=[
                ValidationDataSource(
                    source_type="ascat_ssm", collocation_kwargs={"time_tolerance_minutes": 45},
                ),
            ],
            collocation=CollocationType(),
        )
        assert _resolve_temporal_padding_minutes(cfg, "ascat_ssm") == 45.0

    def test_padded_temporal_bounds_pads_symmetrically(self):
        from sar_validation.core.orchestrator import _padded_temporal_bounds

        cfg = self._recipe().config
        start, end = _padded_temporal_bounds(cfg, "ascat_ssm")
        assert start == "2026-01-31T12:00:00"
        assert end == "2026-02-03T12:00:00"

    def test_ascat_ssm_download_receives_padded_bounds(self, tmp_path):
        """The actual downloader call must see the padded start/end, not
        the recipe's literal requested range."""
        cfg = self._recipe().config
        recipe = Recipe(cfg)
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        orchestrator.base_dir = tmp_path

        with patch(
            "sar_validation.downloaders.ascat_soil_moisture_downloader.ASCATSoilMoistureDownloader"
        ) as mock_cls:
            mock_instance = MagicMock()
            mock_instance.download.return_value = []
            mock_cls.return_value = mock_instance

            orchestrator._dispatch_source(cfg.validation_sources[0])

        call_kwargs = mock_instance.download.call_args.kwargs
        assert call_kwargs["start"] == "2026-01-31T12:00:00"
        assert call_kwargs["end"] == "2026-02-03T12:00:00"

    def test_sar_download_is_not_padded(self, tmp_path):
        """SAR scenes define the reference times other sources are padded
        around -- padding the SAR request itself would fetch scenes
        outside what the recipe actually asked for."""
        cfg = self._recipe().config
        recipe = Recipe(cfg)
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        orchestrator.base_dir = tmp_path

        with patch(
            "sar_validation.downloaders.sentinel1_soil_moisture_downloader.SoilMoistureDownloader"
        ) as mock_cls:
            mock_instance = MagicMock()
            mock_instance.download.return_value = []
            mock_cls.return_value = mock_instance

            orchestrator._download_sar()

        call_kwargs = mock_instance.download.call_args.kwargs
        assert call_kwargs["start"] == "2026-02-01"
        assert call_kwargs["end"] == "2026-02-03"


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

    @pytest.mark.parametrize(
        "dry_run,has_files,expected_status",
        [
            pytest.param(False, False, "awaiting_manual_archive", id="zero_files"),
            pytest.param(False, True, "success", id="nonzero_files"),
            pytest.param(True, False, "dry_run", id="dry_run_with_zero_files"),
        ],
    )
    def test_status_reflects_dry_run_and_file_count(
        self, tmp_path, dry_run, has_files, expected_status
    ):
        recipe = self._recipe()
        orchestrator = DataOrchestrator(recipe, dry_run=dry_run)
        orchestrator.base_dir = tmp_path
        download_return_value = [tmp_path / "ismn_station.csv"] if has_files else []

        with patch(
            "sar_validation.downloaders.ismn_downloader.ISMNDownloader"
        ) as mock_cls:
            mock_instance = MagicMock()
            mock_instance.download.return_value = download_return_value
            mock_cls.return_value = mock_instance

            ok = orchestrator._download_ismn(recipe.config.validation_sources[0])

        assert orchestrator.metadata["downloads"]["ismn"]["status"] == expected_status
        if not dry_run:
            assert ok is True


class TestScatterometerHandlerCleansUpEmptyOutputDir:
    @pytest.mark.parametrize(
        "produces_file,expect_dir_removed",
        [
            pytest.param(False, True, id="no_files_produced"),
            pytest.param(True, False, id="file_produced"),
        ],
    )
    def test_output_dir_cleanup_depends_on_whether_files_were_produced(
        self, tmp_path, produces_file, expect_dir_removed
    ):
        recipe = _recipe("sentinel1_l2_ocn")
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
                out = tmp_path / "osi_saf_winds"
                out.mkdir(parents=True, exist_ok=True)
                if produces_file:
                    (out / "scat.nc").write_text("x")
                    return [out / "scat.nc"]
                return []

            mock_instance.download.side_effect = fake_download
            mock_cls.return_value = mock_instance

            ok = orchestrator._download_scatterometer(None)

        assert ok is True
        if expect_dir_removed:
            assert not (tmp_path / "osi_saf_winds").exists()
        else:
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

    @pytest.mark.parametrize(
        "source_types,downloads,expected_message_substrs",
        [
            pytest.param(
                ("adcp_historical", "argo_historical"),
                {
                    "adcp_historical": {"status": "success", "file_count": 0},
                    "argo_historical": {"status": "success", "file_count": 0},
                },
                ("adcp", "argo"),
                id="both_empty",
            ),
            pytest.param(
                ("drifter_historical",),
                {"drifter_historical": {"status": "success", "file_count": 0}},
                ("drifter",),
                id="single_source",
            ),
        ],
    )
    def test_all_empty_logs_one_combined_warning(
        self, caplog, source_types, downloads, expected_message_substrs
    ):
        recipe = self._recipe_with(*source_types)
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.metadata["downloads"].update(downloads)

        with caplog.at_level(logging.WARNING):
            orchestrator._report_combined_currents_status()

        combined = [
            r for r in caplog.records
            if "No delayed-mode in-situ current data found" in r.message
        ]
        assert len(combined) == 1
        for substr in expected_message_substrs:
            assert substr in combined[0].message
        if len(source_types) == 1:
            # The message must name only the instrument(s) actually present
            # in the recipe -- not every instrument the combined check knows
            # about.
            assert combined[0].message.strip().startswith(
                "No delayed-mode in-situ current data found (drifter)"
            )

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

    @pytest.mark.parametrize(
        "downloads,dry_run",
        [
            pytest.param(
                {
                    "adcp_historical": {"status": "success", "file_count": 2},
                    "argo_historical": {"status": "success", "file_count": 0},
                },
                False,
                id="one_nonempty_suppresses",
            ),
            pytest.param(
                {"adcp_historical": {"status": "dry_run", "file_count": 0}},
                True,
                id="dry_run_never_logs",
            ),
            pytest.param(
                {"adcp_historical": {"status": "failed", "error": "boom"}},
                False,
                # A real failure (network error, etc.) is a different
                # problem than 'no data' and must not be folded into the
                # combined message.
                id="failed_instrument_suppresses",
            ),
        ],
    )
    def test_combined_warning_suppressed_unless_all_present_sources_are_empty(
        self, caplog, downloads, dry_run
    ):
        source_types = tuple(downloads.keys())
        recipe = self._recipe_with(*source_types)
        orchestrator = DataOrchestrator(recipe, dry_run=dry_run)
        orchestrator.metadata["downloads"].update(downloads)

        with caplog.at_level(logging.WARNING):
            orchestrator._report_combined_currents_status()

        assert not any(
            "No delayed-mode in-situ current data found" in r.message for r in caplog.records
        )


class TestCombinedHfRadarUsStatusMessage:
    def _recipe(self) -> Recipe:
        cfg = RecipeConfig(
            name="test", variable="currents",
            geographic_bounds=GeographicBounds(-125.0, -119.0, 33.0, 38.0),
            temporal_bounds=TemporalBounds("2026-01-01", "2026-01-02"),
            validation_sources=[ValidationDataSource(source_type="hf_radar_us")],
        )
        return Recipe(cfg)

    def test_all_backends_empty_logs_and_records_one_notice(self, caplog):
        recipe = self._recipe()
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.metadata["downloads"]["hf_radar_us"] = {
            "status": "success", "file_count": 0,
            "attempted_backends": ["erddap", "thredds", "copernicus"],
        }

        with caplog.at_level(logging.WARNING):
            orchestrator._report_combined_hf_radar_us_status()

        combined = [r for r in caplog.records if "No US HF-radar data found" in r.message]
        assert len(combined) == 1
        assert "erddap" in combined[0].message
        assert "thredds" in combined[0].message
        assert "copernicus" in combined[0].message
        assert len(orchestrator.metadata["notices"]) == 1
        assert orchestrator.metadata["errors"] == []

    @pytest.mark.parametrize(
        "downloads_entry,dry_run",
        [
            pytest.param(
                {"status": "success", "file_count": 3, "attempted_backends": ["erddap"]},
                False,
                id="nonempty_result",
            ),
            pytest.param(None, False, id="no_entry_at_all"),
            pytest.param(
                {"status": "dry_run", "file_count": 0, "attempted_backends": ["erddap"]},
                True,
                id="dry_run_never_logs",
            ),
        ],
    )
    def test_no_warning_unless_all_backends_report_zero(self, caplog, downloads_entry, dry_run):
        recipe = self._recipe()
        orchestrator = DataOrchestrator(recipe, dry_run=dry_run)
        if downloads_entry is not None:
            orchestrator.metadata["downloads"]["hf_radar_us"] = downloads_entry

        with caplog.at_level(logging.WARNING):
            orchestrator._report_combined_hf_radar_us_status()

        assert not any("No US HF-radar data found" in r.message for r in caplog.records)
        assert orchestrator.metadata["notices"] == []


class TestCombinedHfRadarStatusMessage:
    def _recipe_with(self, *source_types: str) -> Recipe:
        cfg = RecipeConfig(
            name="test", variable="currents",
            geographic_bounds=GeographicBounds(-10.0, 20.0, 40.0, 55.0),
            temporal_bounds=TemporalBounds("2020-01-01", "2020-01-02"),
            validation_sources=[ValidationDataSource(source_type=st) for st in source_types],
        )
        return Recipe(cfg)

    def test_both_empty_logs_one_combined_notice(self, caplog):
        recipe = self._recipe_with("hf_radar", "hf_radar_historical")
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.metadata["downloads"]["hf_radar_historical"] = {"status": "success", "file_count": 0}
        orchestrator.metadata["downloads"]["hf_radar"] = {"status": "success", "file_count": 0}

        with caplog.at_level(logging.WARNING):
            orchestrator._report_combined_hf_radar_status()

        combined = [r for r in caplog.records if "No HF-radar data found" in r.message]
        assert len(combined) == 1
        assert len(orchestrator.metadata["notices"]) == 1

    @pytest.mark.parametrize(
        "source_types,downloads,expected_fires",
        [
            pytest.param(
                ("hf_radar", "hf_radar_historical"),
                {
                    "hf_radar_historical": {"status": "success", "file_count": 4},
                    "hf_radar": {"status": "skipped", "reason": "covered by hf_radar_historical"},
                },
                False,
                id="hf_radar_skipped_but_historical_had_data",
            ),
            pytest.param(
                ("hf_radar",),
                {"hf_radar": {"status": "success", "file_count": 0}},
                True,
                id="only_hf_radar_in_recipe_and_empty_still_logs",
            ),
        ],
    )
    def test_combined_notice_fires_based_on_hf_radar_status(
        self, caplog, source_types, downloads, expected_fires
    ):
        recipe = self._recipe_with(*source_types)
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.metadata["downloads"].update(downloads)

        with caplog.at_level(logging.WARNING):
            orchestrator._report_combined_hf_radar_status()

        fired = any("No HF-radar data found" in r.message for r in caplog.records)
        assert fired is expected_fires



class TestPerSourceDownloadGating:
    def _recipe_with_ascat_and_smos(self):
        cfg = RecipeConfig(
            name="test", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-10.0, 20.0, 40.0, 55.0),
            temporal_bounds=TemporalBounds("2026-01-01", "2026-01-02"),
            validation_sources=[
                ValidationDataSource(source_type="ascat_ssm"),
                ValidationDataSource(source_type="smos_ssm"),
            ],
        )
        return Recipe(cfg)

    def test_already_succeeded_source_is_not_redispatched(self, tmp_path):
        import json

        recipe = self._recipe_with_ascat_and_smos()
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.base_dir = tmp_path
        # Simulate a previous run: ascat_ssm succeeded, smos_ssm failed.
        (tmp_path / "download_metadata.json").write_text(json.dumps({
            "downloads": {
                "ascat_ssm": {"status": "success", "files": ["a.nat"]},
                "smos_ssm": {"status": "failed", "error": "timed out"},
            },
            "errors": ["SMOS SSM download failed: timed out"],
            "notices": [],
        }))
        # Re-construct so __init__ picks up the metadata file just written.
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.base_dir = tmp_path
        orchestrator._previous_downloads = orchestrator._load_previous_downloads()

        with patch.object(DataOrchestrator, "_download_ascat_ssm") as mock_ascat, \
             patch.object(DataOrchestrator, "_download_smos_ssm", return_value=False) as mock_smos, \
             patch.object(DataOrchestrator, "_download_sar", return_value=True):
            orchestrator.download_all()

        mock_ascat.assert_not_called()
        mock_smos.assert_called_once()
        assert orchestrator.metadata["downloads"]["ascat_ssm"]["status"] == "success"

    def test_force_download_bypasses_gating(self, tmp_path):
        import json

        recipe = self._recipe_with_ascat_and_smos()
        orchestrator = DataOrchestrator(recipe, dry_run=False, force_download=True)
        orchestrator.base_dir = tmp_path
        (tmp_path / "download_metadata.json").write_text(json.dumps({
            "downloads": {"ascat_ssm": {"status": "success", "files": ["a.nat"]}},
            "errors": [], "notices": [],
        }))
        orchestrator = DataOrchestrator(recipe, dry_run=False, force_download=True)
        orchestrator.base_dir = tmp_path
        orchestrator._previous_downloads = orchestrator._load_previous_downloads()

        with patch.object(DataOrchestrator, "_download_ascat_ssm", return_value=True) as mock_ascat, \
             patch.object(DataOrchestrator, "_download_smos_ssm", return_value=True), \
             patch.object(DataOrchestrator, "_download_sar", return_value=True):
            orchestrator.download_all()

        mock_ascat.assert_called_once()

    def test_no_previous_metadata_dispatches_everything(self, tmp_path):
        recipe = self._recipe_with_ascat_and_smos()
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.base_dir = tmp_path  # no download_metadata.json written

        with patch.object(DataOrchestrator, "_download_ascat_ssm", return_value=True) as mock_ascat, \
             patch.object(DataOrchestrator, "_download_smos_ssm", return_value=True) as mock_smos, \
             patch.object(DataOrchestrator, "_download_sar", return_value=True):
            orchestrator.download_all()

        mock_ascat.assert_called_once()
        mock_smos.assert_called_once()


class TestAmsrCoverageCutoffNotice:
    def _recipe_with_amsr(self, end_date: str):
        cfg = RecipeConfig(
            name="test", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-10.0, 20.0, 40.0, 55.0),
            temporal_bounds=TemporalBounds("2026-01-01", end_date),
            validation_sources=[ValidationDataSource(source_type="amsr_ssm")],
        )
        return Recipe(cfg)

    def test_notice_added_when_request_exceeds_known_coverage(self, tmp_path):
        recipe = self._recipe_with_amsr("2026-01-02")
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.base_dir = tmp_path

        with patch(
            "sar_validation.downloaders.earthdata_soil_moisture_downloader.EarthdataSoilMoistureDownloader"
        ) as mock_cls, patch(
            "sar_validation.downloaders.gportal_downloader.GPortalAMSR2Downloader"
        ) as mock_gportal_cls:
            mock_instance = MagicMock()
            mock_instance.download.return_value = []
            mock_cls.return_value = mock_instance
            mock_gportal_cls.return_value.download.return_value = []

            orchestrator._download_amsr_ssm(recipe.config.validation_sources[0])

        assert any(
            "AMSR-E/2" in n and "2025-09-01" in n for n in orchestrator.metadata["notices"]
        )
        assert orchestrator.metadata["downloads"]["amsr_ssm"]["status"] == "success"
        # A notice, not an error -- must not trip _is_already_downloaded's gate.
        assert orchestrator.metadata["errors"] == []

    def test_no_notice_when_within_coverage_and_files_found(self, tmp_path):
        cfg = RecipeConfig(
            name="test", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-10.0, 20.0, 40.0, 55.0),
            temporal_bounds=TemporalBounds("2020-01-01", "2020-01-02"),
            validation_sources=[ValidationDataSource(source_type="amsr_ssm")],
        )
        recipe = Recipe(cfg)
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.base_dir = tmp_path

        with patch(
            "sar_validation.downloaders.earthdata_soil_moisture_downloader.EarthdataSoilMoistureDownloader"
        ) as mock_cls:
            mock_instance = MagicMock()
            mock_instance.download.return_value = [tmp_path / "granule.h5"]
            mock_cls.return_value = mock_instance

            orchestrator._download_amsr_ssm(recipe.config.validation_sources[0])

        assert orchestrator.metadata["notices"] == []

    @pytest.mark.parametrize(
        "end_date,start_date_override,expected_dataset",
        [
            pytest.param("2025-07-04", None, "AU_Land", id="post_2023_dates"),
            pytest.param("2023-06-01", "2023-05-01", "NSIDC-0451", id="historical_dates"),
        ],
    )
    def test_selects_dataset_based_on_requested_dates(
        self, tmp_path, end_date, start_date_override, expected_dataset
    ):
        recipe = self._recipe_with_amsr(end_date)
        if start_date_override is not None:
            # Override start so it's before end.
            recipe.config.temporal_bounds.start = start_date_override
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.base_dir = tmp_path

        with patch(
            "sar_validation.downloaders.earthdata_soil_moisture_downloader.EarthdataSoilMoistureDownloader"
        ) as mock_cls, patch(
            "sar_validation.downloaders.gportal_downloader.GPortalAMSR2Downloader"
        ) as mock_gportal_cls:
            mock_instance = MagicMock()
            mock_instance.download.return_value = []
            mock_cls.return_value = mock_instance
            mock_gportal_cls.return_value.download.return_value = []

            orchestrator._download_amsr_ssm(recipe.config.validation_sources[0])

        assert mock_cls.call_args.kwargs["dataset"] == expected_dataset

    def test_notice_added_when_gportal_fallback_also_finds_nothing(self, tmp_path):
        """Within AMSR's known coverage window (no coverage-cutoff notice),
        if NASA Earthdata returns 0 granules AND the G-Portal SFTP fallback
        also finds 0 files, the run must not silently report "success" with
        no explanation -- confirmed against a real recipe run where G-Portal
        logged "Found 0 AMSR2 file(s) in window." with no resulting notice
        at all, indistinguishable from a genuinely successful download."""
        cfg = RecipeConfig(
            name="test", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-10.0, 20.0, 40.0, 55.0),
            temporal_bounds=TemporalBounds("2025-07-01", "2025-07-03"),
            validation_sources=[ValidationDataSource(source_type="amsr_ssm")],
        )
        recipe = Recipe(cfg)
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.base_dir = tmp_path

        with patch(
            "sar_validation.downloaders.earthdata_soil_moisture_downloader.EarthdataSoilMoistureDownloader"
        ) as mock_cls, patch(
            "sar_validation.downloaders.gportal_downloader.GPortalAMSR2Downloader"
        ) as mock_gportal_cls:
            mock_instance = MagicMock()
            mock_instance.download.return_value = []
            mock_cls.return_value = mock_instance
            mock_gportal_cls.return_value.download.return_value = []

            orchestrator._download_amsr_ssm(recipe.config.validation_sources[0])

        assert any(
            "G-Portal" in n and "0 files" in n for n in orchestrator.metadata["notices"]
        )
        assert orchestrator.metadata["downloads"]["amsr_ssm"]["status"] == "success"
        assert orchestrator.metadata["errors"] == []


class TestDownloadEra5:
    def _recipe(self, variable="wind"):
        from sar_validation.core.recipe import (
            GeographicBounds,
            Recipe,
            RecipeConfig,
            TemporalBounds,
            ValidationDataSource,
        )
        cfg = RecipeConfig(
            name="t", variable=variable,
            geographic_bounds=GeographicBounds(-10.0, 10.0, 40.0, 55.0),
            temporal_bounds=TemporalBounds("2026-07-12T18:00:00", "2026-07-12T23:00:00"),
            validation_sources=[ValidationDataSource(source_type="era5")],
        )
        return Recipe(cfg)

    def test_dispatches_to_download_era5(self, tmp_path, monkeypatch):
        from sar_validation.core.orchestrator import DataOrchestrator

        recipe = self._recipe()
        orch = DataOrchestrator(recipe, dry_run=True)
        called = {}

        def fake_download_era5(self, source):
            called["source_type"] = source.source_type
            return True

        monkeypatch.setattr(DataOrchestrator, "_download_era5", fake_download_era5)
        result = orch._dispatch_source(recipe.config.validation_sources[0])
        assert result is True
        assert called["source_type"] == "era5"

    def test_download_era5_builds_downloader_with_recipe_variable(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        from sar_validation.core.orchestrator import DataOrchestrator

        recipe = self._recipe(variable="waves")
        orch = DataOrchestrator(recipe, dry_run=True)

        fake_dl = MagicMock()
        fake_dl.download.return_value = []
        fake_cls = MagicMock(return_value=fake_dl)

        # _download_era5 does a local `from ..downloaders.era5_downloader
        # import ERA5Downloader` -- patch the name on that module so the
        # local import picks up the fake at call time.
        import sar_validation.downloaders.era5_downloader as dl_mod
        monkeypatch.setattr(dl_mod, "ERA5Downloader", fake_cls)

        ok = orch._download_era5(recipe.config.validation_sources[0])
        assert ok is True
        assert fake_cls.call_args.kwargs["variable"] == "waves"
