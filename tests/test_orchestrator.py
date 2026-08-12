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
        # Dates comfortably inside the EUMDAC coverage window (<=
        # _ASCAT_COVERAGE_CUTOFF = 2025-07-15) so the waterfall routes to
        # ASCATSoilMoistureDownloader, not HSAFDownloader.
        cfg = RecipeConfig(
            name="test", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-10.0, 20.0, 40.0, 55.0),
            temporal_bounds=TemporalBounds("2025-06-01", "2025-06-02"),
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
        the recipe's literal requested range.

        Uses its own config (not the shared self._recipe() helper, whose
        2026 dates now fall in the EUMDAC/H-SAF coverage gap) with dates
        comfortably inside the EUMDAC coverage window so the waterfall
        routes to ASCATSoilMoistureDownloader, matching what this test
        mocks."""
        from sar_validation.core.recipe import CollocationType

        cfg = RecipeConfig(
            name="test", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-10.0, 20.0, 40.0, 55.0),
            temporal_bounds=TemporalBounds("2025-06-01", "2025-06-03"),
            sar_data=SARDataSpec(source="sentinel1_clms_ssm"),
            validation_sources=[
                ValidationDataSource(source_type="ascat_ssm"),
                ValidationDataSource(source_type="ismn"),
            ],
            collocation=CollocationType(),
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

            orchestrator._dispatch_source(cfg.validation_sources[0])

        call_kwargs = mock_instance.download.call_args.kwargs
        assert call_kwargs["start"] == "2025-05-31T12:00:00"
        assert call_kwargs["end"] == "2025-06-03T12:00:00"

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


class TestModelSourceTemporalPadding:
    """era5/hycom (model_vs_layer, via ModelLayerCollocation) previously
    silently ignored their own DEFAULT_LAYER_TYPE_SPECS-tuned tolerance at
    download time -- "era5" (the ValidationDataSource.source_type every
    recipe actually uses) has no DEFAULT_LAYER_TYPE_SPECS entry of its
    own (only "era5_wind"/"era5_waves"/"era5_soil_moisture" do), so it
    fell through to the generic 30-min point_vs_layer fallback."""

    def test_era5_resolves_to_its_own_variable_specific_default(self):
        from sar_validation.core.orchestrator import _resolve_temporal_padding_minutes

        for variable, expected in (("wind", 120.0), ("waves", 120.0), ("soil_moisture", 720.0)):
            cfg = RecipeConfig(
                name="test", variable=variable,
                geographic_bounds=GeographicBounds(-10.0, 20.0, 40.0, 55.0),
                temporal_bounds=TemporalBounds("2026-02-01", "2026-02-03"),
                validation_sources=[ValidationDataSource(source_type="era5")],
            )
            assert _resolve_temporal_padding_minutes(cfg, "era5") == expected

    def test_hycom_resolves_to_its_own_default(self):
        from sar_validation.core.orchestrator import _resolve_temporal_padding_minutes

        cfg = RecipeConfig(
            name="test", variable="currents",
            geographic_bounds=GeographicBounds(-10.0, 20.0, 40.0, 55.0),
            temporal_bounds=TemporalBounds("2026-02-01", "2026-02-03"),
            validation_sources=[ValidationDataSource(source_type="hycom")],
        )
        assert _resolve_temporal_padding_minutes(cfg, "hycom") == 360.0

    def test_partial_era5_wind_override_keeps_default_time_tolerance(self):
        """A recipe overriding only e.g. "method" for "era5_wind" (as
        every wind_era5-style recipe does) must not lose the default's
        time_tolerance_minutes -- regression test for the
        layer_specs.update(...) shallow-merge bug."""
        from sar_validation.core.orchestrator import _resolve_temporal_padding_minutes
        from sar_validation.core.recipe import CollocationType, LayerVsLayerCollocation

        cfg = RecipeConfig(
            name="test", variable="wind",
            geographic_bounds=GeographicBounds(-10.0, 20.0, 40.0, 55.0),
            temporal_bounds=TemporalBounds("2026-02-01", "2026-02-03"),
            validation_sources=[ValidationDataSource(source_type="era5")],
            collocation=CollocationType(
                layer_vs_layer=LayerVsLayerCollocation(
                    layer_type_specs={"era5_wind": {"method": "cell-averaging"}},
                ),
            ),
        )
        assert _resolve_temporal_padding_minutes(cfg, "era5") == 120.0

    def test_warns_when_hycom_override_is_below_the_bracket_safe_minimum(self, caplog):
        import logging

        from sar_validation.core.orchestrator import _resolve_temporal_padding_minutes
        from sar_validation.core.recipe import CollocationType, LayerVsLayerCollocation

        cfg = RecipeConfig(
            name="test", variable="currents",
            geographic_bounds=GeographicBounds(-10.0, 20.0, 40.0, 55.0),
            temporal_bounds=TemporalBounds("2026-02-01", "2026-02-03"),
            validation_sources=[ValidationDataSource(source_type="hycom")],
            collocation=CollocationType(
                layer_vs_layer=LayerVsLayerCollocation(
                    layer_type_specs={"hycom": {"time_tolerance_minutes": 90}},
                ),
            ),
        )
        with caplog.at_level(logging.WARNING):
            resolved = _resolve_temporal_padding_minutes(cfg, "hycom")
        assert resolved == 90.0  # the recipe's explicit choice is still honored
        assert any("below the 360-minute minimum" in r.message for r in caplog.records)

    def test_no_warning_when_default_is_used(self, caplog):
        import logging

        from sar_validation.core.orchestrator import _resolve_temporal_padding_minutes

        cfg = RecipeConfig(
            name="test", variable="currents",
            geographic_bounds=GeographicBounds(-10.0, 20.0, 40.0, 55.0),
            temporal_bounds=TemporalBounds("2026-02-01", "2026-02-03"),
            validation_sources=[ValidationDataSource(source_type="hycom")],
        )
        with caplog.at_level(logging.WARNING):
            _resolve_temporal_padding_minutes(cfg, "hycom")
        assert not caplog.records

    def _model_recipe(self, source_type: str, variable: str = "currents") -> Recipe:
        cfg = RecipeConfig(
            name="test", variable=variable,
            geographic_bounds=GeographicBounds(-10.0, 20.0, 40.0, 55.0),
            temporal_bounds=TemporalBounds("2026-02-01", "2026-02-03"),
            sar_data=SARDataSpec(source="sentinel1_l2_ocn"),
            validation_sources=[ValidationDataSource(source_type=source_type)],
        )
        return Recipe(cfg)

    def test_hycom_download_receives_resolved_time_tolerance_and_unpadded_bounds(self, tmp_path):
        """HycomDownloader now does its OWN bracket-margin widening (see
        its time_tolerance_minutes parameter) -- the orchestrator must
        pass the resolved value through, and the literal (unpadded)
        recipe window, not a separately (and now redundantly) padded one."""
        recipe = self._model_recipe("hycom")
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        orchestrator.base_dir = tmp_path

        with patch("sar_validation.downloaders.hycom_downloader.HycomDownloader") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.download.return_value = []
            mock_cls.return_value = mock_instance

            orchestrator._dispatch_source(recipe.config.validation_sources[0])

        assert mock_cls.call_args.kwargs["time_tolerance_minutes"] == 360.0
        call_kwargs = mock_instance.download.call_args.kwargs
        assert call_kwargs["start"] == "2026-02-01"
        assert call_kwargs["end"] == "2026-02-03"

    def test_era5_download_receives_resolved_time_tolerance_and_unpadded_bounds(self, tmp_path):
        recipe = self._model_recipe("era5", variable="wind")
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        orchestrator.base_dir = tmp_path

        with patch("sar_validation.downloaders.era5_downloader.ERA5Downloader") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.download.return_value = []
            mock_cls.return_value = mock_instance

            orchestrator._dispatch_source(recipe.config.validation_sources[0])

        assert mock_cls.call_args.kwargs["time_tolerance_minutes"] == 120.0
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


class TestLoadPreviousVariableAndEra5Gating:
    """_load_previous_variable()/_already_succeeded("era5") together detect
    the case where two recipes with identical geographic/temporal bounds
    (so they share base_dir) request different recipe.config.variable
    values -- e.g. wind_era5.yaml vs waves_era5.yaml. Scoped to era5 only,
    matching cli.py's _is_already_downloaded's own era5-only scoping (see
    TestIsAlreadyDownloaded in test_cli.py)."""

    def _era5_recipe(self, variable):
        cfg = RecipeConfig(
            name="t", variable=variable,
            geographic_bounds=GeographicBounds(-10.0, 10.0, 40.0, 55.0),
            temporal_bounds=TemporalBounds("2026-01-01", "2026-01-02"),
            validation_sources=[ValidationDataSource(source_type="era5")],
        )
        return Recipe(cfg)

    def test_load_previous_variable_reads_top_level_field(self, tmp_path):
        import json

        (tmp_path / "download_metadata.json").write_text(json.dumps({
            "variable": "wind", "downloads": {}, "errors": [],
        }))
        recipe = self._era5_recipe("wind")
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.base_dir = tmp_path
        assert orchestrator._load_previous_variable() == "wind"

    def test_load_previous_variable_none_when_metadata_missing(self, tmp_path):
        recipe = self._era5_recipe("wind")
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.base_dir = tmp_path
        assert orchestrator._load_previous_variable() is None

    def test_already_succeeded_era5_true_when_variable_matches(self, tmp_path):
        import json

        (tmp_path / "download_metadata.json").write_text(json.dumps({
            "variable": "wind",
            "downloads": {"era5": {"status": "success"}},
            "errors": [],
        }))
        recipe = self._era5_recipe("wind")
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.base_dir = tmp_path
        orchestrator._previous_downloads = orchestrator._load_previous_downloads()
        orchestrator._previous_variable = orchestrator._load_previous_variable()
        assert orchestrator._already_succeeded("era5") is True

    def test_already_succeeded_era5_false_when_variable_differs(self, tmp_path):
        import json

        (tmp_path / "download_metadata.json").write_text(json.dumps({
            "variable": "wind",
            "downloads": {"era5": {"status": "success"}},
            "errors": [],
        }))
        recipe = self._era5_recipe("waves")
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.base_dir = tmp_path
        orchestrator._previous_downloads = orchestrator._load_previous_downloads()
        orchestrator._previous_variable = orchestrator._load_previous_variable()
        assert orchestrator._already_succeeded("era5") is False

    def test_already_succeeded_non_era5_source_ignores_variable_mismatch(self, tmp_path):
        """A non-era5 source_type recorded successfully must not be
        affected by a top-level variable mismatch -- the mismatch check in
        _already_succeeded is explicitly scoped to source_type == "era5"."""
        import json

        (tmp_path / "download_metadata.json").write_text(json.dumps({
            "variable": "wind",
            "downloads": {"sar": {"status": "success"}},
            "errors": [],
        }))
        recipe = self._era5_recipe("waves")
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.base_dir = tmp_path
        orchestrator._previous_downloads = orchestrator._load_previous_downloads()
        orchestrator._previous_variable = orchestrator._load_previous_variable()
        assert orchestrator._already_succeeded("sar") is True


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


class TestSarEmptyStopsPipeline:
    def _recipe_with_validation_source(self) -> Recipe:
        cfg = RecipeConfig(
            name="test", variable="wind",
            geographic_bounds=GeographicBounds(-10.0, 20.0, 40.0, 55.0),
            temporal_bounds=TemporalBounds("2026-01-01", "2026-01-02"),
            sar_data=SARDataSpec(source="sentinel1_l2_ocn"),
            validation_sources=[ValidationDataSource(source_type="scatterometer")],
        )
        return Recipe(cfg)

    def test_zero_sar_products_skips_validation_downloads(self, tmp_path):
        recipe = self._recipe_with_validation_source()
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.base_dir = tmp_path

        with patch(
            "sar_validation.downloaders.sentinel1_l2_ocn_downloader.SARDownloader"
        ) as mock_sar_cls, patch(
            "sar_validation.downloaders.scatterometer_downloader.ScatterometerDownloader"
        ) as mock_scat_cls:
            mock_sar = MagicMock()
            mock_sar.download.return_value = []
            mock_sar.found_count = 0
            mock_sar_cls.return_value = mock_sar

            ok = orchestrator.download_all()

        mock_scat_cls.assert_not_called()
        assert ok is True
        assert orchestrator.metadata["sar_data_found"] is False
        assert any("No SAR data found" in n for n in orchestrator.metadata["notices"])

    def test_nonzero_sar_products_still_downloads_validation_sources(self, tmp_path):
        recipe = self._recipe_with_validation_source()
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.base_dir = tmp_path

        with patch(
            "sar_validation.downloaders.sentinel1_l2_ocn_downloader.SARDownloader"
        ) as mock_sar_cls, patch(
            "sar_validation.downloaders.scatterometer_downloader.ScatterometerDownloader"
        ) as mock_scat_cls:
            mock_sar = MagicMock()
            mock_sar.download.return_value = [tmp_path / "scene.SAFE"]
            mock_sar.found_count = 1
            mock_sar_cls.return_value = mock_sar

            mock_scat = MagicMock()
            mock_scat.download.return_value = []
            mock_scat_cls.return_value = mock_scat

            orchestrator.download_all()

        mock_scat_cls.assert_called_once()
        assert orchestrator.metadata["sar_data_found"] is True

    def test_dry_run_also_stops_when_zero_sar_products_found(self, tmp_path):
        recipe = self._recipe_with_validation_source()
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        orchestrator.base_dir = tmp_path

        with patch(
            "sar_validation.downloaders.sentinel1_l2_ocn_downloader.SARDownloader"
        ) as mock_sar_cls, patch(
            "sar_validation.downloaders.scatterometer_downloader.ScatterometerDownloader"
        ) as mock_scat_cls:
            mock_sar = MagicMock()
            mock_sar.download.return_value = []
            mock_sar.found_count = 0
            mock_sar_cls.return_value = mock_sar

            orchestrator.download_all()

        mock_scat_cls.assert_not_called()

    def test_sar_failure_does_not_trigger_the_stop(self, tmp_path):
        """A SAR download that raises must keep today's behavior (continue
        to validation downloads, ok=False) -- zero-found and failed are
        different outcomes."""
        recipe = self._recipe_with_validation_source()
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.base_dir = tmp_path

        with patch(
            "sar_validation.downloaders.sentinel1_l2_ocn_downloader.SARDownloader"
        ) as mock_sar_cls, patch(
            "sar_validation.downloaders.scatterometer_downloader.ScatterometerDownloader"
        ) as mock_scat_cls:
            mock_sar = MagicMock()
            mock_sar.download.side_effect = RuntimeError("boom")
            mock_sar_cls.return_value = mock_sar

            mock_scat = MagicMock()
            mock_scat.download.return_value = []
            mock_scat_cls.return_value = mock_scat

            ok = orchestrator.download_all()

        mock_scat_cls.assert_called_once()
        assert ok is False
        assert orchestrator.metadata["sar_data_found"] is True

    def test_already_succeeded_cache_restore_also_triggers_the_gate(self, tmp_path):
        """A previous run's cached SAR entry with found_count == 0 must
        stop download_all() the same way a fresh zero-result SAR download
        does -- not just skip re-downloading SAR itself. Exercises the
        `_already_succeeded("sar")` branch specifically, which is a
        different code path from `previous_sar_data_found()` below (that
        one is read by cli.py's resume shortcut, which never calls
        download_all() at all)."""
        import json

        (tmp_path / "download_metadata.json").write_text(json.dumps({
            "variable": "wind",
            "downloads": {"sar": {"status": "success", "found_count": 0, "files": []}},
        }))
        recipe = self._recipe_with_validation_source()
        recipe.config.output_dir = str(tmp_path)
        orchestrator = DataOrchestrator(recipe, dry_run=False)

        with patch(
            "sar_validation.downloaders.scatterometer_downloader.ScatterometerDownloader"
        ) as mock_scat_cls:
            ok = orchestrator.download_all()

        mock_scat_cls.assert_not_called()
        assert ok is True
        assert orchestrator.metadata["sar_data_found"] is False

    def test_already_succeeded_cache_restore_checks_disk_for_old_schema_empty_files(self, tmp_path):
        """Old-schema metadata (recorded before found_count existed) can
        have status=success, files=[] even though real products were
        downloaded -- SARDownloader.download() only appends *newly*
        downloaded files to the list it returns; a product already on
        disk (skipped as a duplicate) is never appended, so a fully
        successful run where every match was already cached still ends
        up with files=[]. Confirmed live 2026-08-11: a copied-over
        currents_useastcoast.yaml download_metadata.json from before
        found_count existed was misread as "no SAR data found" despite
        the SAFE files sitting right there in S1_L2_OCN/. Real files on
        disk matching the source's file_glob must override the
        ambiguous empty list."""
        import json

        (tmp_path / "download_metadata.json").write_text(json.dumps({
            "variable": "wind",
            "downloads": {"sar": {"status": "success", "files": []}},
        }))
        (tmp_path / "S1_L2_OCN" / "S1A_IW_OCN__2SDV_X.SAFE").mkdir(parents=True)
        recipe = self._recipe_with_validation_source()
        recipe.config.output_dir = str(tmp_path)
        orchestrator = DataOrchestrator(recipe, dry_run=False)

        with patch(
            "sar_validation.downloaders.scatterometer_downloader.ScatterometerDownloader"
        ) as mock_scat_cls:
            ok = orchestrator.download_all()

        mock_scat_cls.assert_called_once()
        assert ok is True
        assert orchestrator.metadata["sar_data_found"] is True

    def test_disk_check_backfills_found_count_into_saved_metadata(self, tmp_path):
        """The disk-check fallback must heal the stale old-schema entry
        in place, not just read through it -- otherwise every future run
        re-triggers the same disk scan forever instead of the metadata
        ever becoming self-consistent. Requested explicitly: 'backfill
        the found_count to make sure download_metadata.json stays as
        consistent as possible.'"""
        import json

        (tmp_path / "download_metadata.json").write_text(json.dumps({
            "variable": "wind",
            "downloads": {"sar": {"status": "success", "files": []}},
        }))
        (tmp_path / "S1_L2_OCN" / "S1A_IW_OCN__2SDV_X.SAFE").mkdir(parents=True)
        (tmp_path / "S1_L2_OCN" / "S1B_IW_OCN__2SDV_Y.SAFE").mkdir(parents=True)
        recipe = self._recipe_with_validation_source()
        recipe.config.output_dir = str(tmp_path)
        orchestrator = DataOrchestrator(recipe, dry_run=False)

        with patch("sar_validation.downloaders.scatterometer_downloader.ScatterometerDownloader"):
            orchestrator.download_all()

        sar_entry = orchestrator.metadata["downloads"]["sar"]
        assert sar_entry["found_count"] == 2
        assert len(sar_entry["files"]) == 2
        assert all(str(tmp_path / "S1_L2_OCN") in f for f in sar_entry["files"])

        # And the healed entry is what actually gets written to disk.
        saved = json.loads((tmp_path / "download_metadata.json").read_text())
        assert saved["downloads"]["sar"]["found_count"] == 2

    def test_previous_sar_data_found_reads_cached_metadata(self, tmp_path):
        import json

        (tmp_path / "download_metadata.json").write_text(json.dumps({
            "variable": "wind",
            "downloads": {"sar": {"status": "success", "found_count": 0, "files": []}},
        }))
        recipe = self._recipe_with_validation_source()
        recipe.config.output_dir = str(tmp_path)
        orchestrator = DataOrchestrator(recipe, dry_run=False)

        assert orchestrator.previous_sar_data_found() is False

    def test_previous_sar_data_found_falls_back_to_files_len(self, tmp_path):
        """Back-compat: metadata written before this feature has no
        found_count key at all; falls back to len(files)."""
        import json

        (tmp_path / "download_metadata.json").write_text(json.dumps({
            "variable": "wind",
            "downloads": {"sar": {"status": "success", "files": ["a.SAFE"]}},
        }))
        recipe = self._recipe_with_validation_source()
        recipe.config.output_dir = str(tmp_path)
        orchestrator = DataOrchestrator(recipe, dry_run=False)

        assert orchestrator.previous_sar_data_found() is True

    def test_previous_sar_data_found_checks_disk_when_found_count_missing_and_files_empty(
        self, tmp_path,
    ):
        """Same old-schema gap as test_already_succeeded_cache_restore_
        checks_disk_for_old_schema_empty_files above, but for the CLI
        resume-shortcut code path (previous_sar_data_found), which never
        calls download_all()."""
        import json

        (tmp_path / "download_metadata.json").write_text(json.dumps({
            "variable": "wind",
            "downloads": {"sar": {"status": "success", "files": []}},
        }))
        (tmp_path / "S1_L2_OCN" / "S1A_IW_OCN__2SDV_X.SAFE").mkdir(parents=True)
        recipe = self._recipe_with_validation_source()
        recipe.config.output_dir = str(tmp_path)
        orchestrator = DataOrchestrator(recipe, dry_run=False)

        assert orchestrator.previous_sar_data_found() is True

    def test_previous_sar_data_found_still_false_when_nothing_on_disk_either(self, tmp_path):
        """Same old-schema entry, but genuinely no SAR products anywhere
        on disk -- the new disk-check fallback must not fabricate a
        match just because a directory was created."""
        import json

        (tmp_path / "download_metadata.json").write_text(json.dumps({
            "variable": "wind",
            "downloads": {"sar": {"status": "success", "files": []}},
        }))
        recipe = self._recipe_with_validation_source()
        recipe.config.output_dir = str(tmp_path)
        orchestrator = DataOrchestrator(recipe, dry_run=False)

        assert orchestrator.previous_sar_data_found() is False

    def test_previous_sar_data_found_true_when_no_previous_run(self, tmp_path):
        recipe = self._recipe_with_validation_source()
        recipe.config.output_dir = str(tmp_path)
        orchestrator = DataOrchestrator(recipe, dry_run=False)

        assert orchestrator.previous_sar_data_found() is True


class TestDownloadHycom:
    def _recipe(self):
        from sar_validation.core.recipe import (
            GeographicBounds,
            Recipe,
            RecipeConfig,
            TemporalBounds,
            ValidationDataSource,
        )
        cfg = RecipeConfig(
            name="t", variable="currents",
            geographic_bounds=GeographicBounds(-10.0, 10.0, 40.0, 55.0),
            temporal_bounds=TemporalBounds("2025-01-01T00:00:00", "2025-01-01T06:00:00"),
            validation_sources=[ValidationDataSource(source_type="hycom")],
        )
        return Recipe(cfg)

    def test_dispatches_to_download_hycom(self, tmp_path, monkeypatch):
        from sar_validation.core.orchestrator import DataOrchestrator

        recipe = self._recipe()
        orch = DataOrchestrator(recipe, dry_run=True)
        called = {}

        def fake_download_hycom(self, source):
            called["source_type"] = source.source_type
            return True

        monkeypatch.setattr(DataOrchestrator, "_download_hycom", fake_download_hycom)
        result = orch._dispatch_source(recipe.config.validation_sources[0])
        assert result is True
        assert called["source_type"] == "hycom"

    def test_download_hycom_builds_downloader_and_calls_download(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        from sar_validation.core.orchestrator import DataOrchestrator

        recipe = self._recipe()
        orch = DataOrchestrator(recipe, dry_run=True)

        fake_dl = MagicMock()
        fake_dl.download.return_value = []
        fake_cls = MagicMock(return_value=fake_dl)

        import sar_validation.downloaders.hycom_downloader as dl_mod
        monkeypatch.setattr(dl_mod, "HycomDownloader", fake_cls)

        ok = orch._download_hycom(recipe.config.validation_sources[0])
        assert ok is True
        fake_cls.assert_called_once()
        fake_dl.download.assert_called_once()
        call_kwargs = fake_dl.download.call_args.kwargs
        assert call_kwargs["min_lon"] == -10.0
        assert call_kwargs["max_lon"] == 10.0


class TestDownloadAscatSsmWaterfall:
    """_download_ascat_ssm now splits [start, end] across two downloaders:
    ASCATSoilMoistureDownloader (EUMDAC/SOMO12) for dates <=
    _ASCAT_COVERAGE_CUTOFF, HSAFDownloader (H-SAF H29 NRT) for the rolling
    last-60-days on-line archive. A gap between the two is a warning, not
    a silent drop -- see design-choices.md and this feature's spec doc."""

    def _make_orchestrator(self, tmp_path, start, end, monkeypatch):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import (
            GeographicBounds,
            Recipe,
            RecipeConfig,
            SARDataSpec,
            TemporalBounds,
            ValidationDataSource,
        )

        config = RecipeConfig(
            name="test", variable="soil_moisture",
            geographic_bounds=GeographicBounds(min_lon=-10, max_lon=10, min_lat=40, max_lat=55),
            temporal_bounds=TemporalBounds(start=start, end=end),
            sar_data=SARDataSpec(source="sentinel1_clms_ssm"),
            validation_sources=[ValidationDataSource(source_type="ascat_ssm")],
            output_dir=str(tmp_path),
        )
        recipe = Recipe(config=config)
        # DataOrchestrator's constructor takes no base_dir kwarg -- it
        # derives self.base_dir from recipe.config.output_dir (set above)
        # via _setup_base_dir().
        return DataOrchestrator(recipe, dry_run=True)

    def test_range_entirely_before_cutoff_only_calls_eumdac(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "sar_validation.downloaders.ascat_soil_moisture_downloader.ASCATSoilMoistureDownloader.download",
            lambda self, **kw: calls.append(("eumdac", kw)) or [],
        )
        monkeypatch.setattr(
            "sar_validation.downloaders.hsaf_downloader.HSAFDownloader.download",
            lambda self, **kw: calls.append(("hsaf", kw)) or [],
        )
        orch = self._make_orchestrator(tmp_path, "2024-01-01", "2024-01-02", monkeypatch)
        source = orch.recipe.config.validation_sources[0]

        orch._download_ascat_ssm(source)

        assert [c[0] for c in calls] == ["eumdac"]

    def test_range_after_cutoff_only_calls_hsaf(self, tmp_path, monkeypatch):
        import datetime as dt

        today = dt.date.today()
        start = (today - dt.timedelta(days=30)).isoformat()
        end = today.isoformat()
        calls = []
        monkeypatch.setattr(
            "sar_validation.downloaders.ascat_soil_moisture_downloader.ASCATSoilMoistureDownloader.download",
            lambda self, **kw: calls.append(("eumdac", kw)) or [],
        )
        monkeypatch.setattr(
            "sar_validation.downloaders.hsaf_downloader.HSAFDownloader.download",
            lambda self, **kw: calls.append(("hsaf", kw)) or [],
        )
        orch = self._make_orchestrator(tmp_path, start, end, monkeypatch)
        source = orch.recipe.config.validation_sources[0]

        orch._download_ascat_ssm(source)

        assert [c[0] for c in calls] == ["hsaf"]

    def test_gap_between_cutoff_and_hsaf_window_produces_notice(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sar_validation.downloaders.ascat_soil_moisture_downloader.ASCATSoilMoistureDownloader.download",
            lambda self, **kw: [],
        )
        monkeypatch.setattr(
            "sar_validation.downloaders.hsaf_downloader.HSAFDownloader.download",
            lambda self, **kw: [],
        )
        orch = self._make_orchestrator(tmp_path, "2025-09-01", "2025-09-02", monkeypatch)
        source = orch.recipe.config.validation_sources[0]

        orch._download_ascat_ssm(source)

        assert any("gap" in n.lower() or "coverage" in n.lower() for n in orch.metadata["notices"])

    def test_overlap_range_both_succeed_is_not_a_failure(self, tmp_path, monkeypatch):
        """A date range spanning both eras (before cutoff -> recent) attempts
        both downloaders. When both succeed and return files, the run must
        not be flagged as failed and no error should be recorded."""
        import datetime as dt

        today = dt.date.today().isoformat()
        calls = []

        def eumdac_download(self, **kw):
            calls.append(("eumdac", kw))
            return [tmp_path / "eumdac_file.nc"]

        def hsaf_download(self, **kw):
            calls.append(("hsaf", kw))
            return [tmp_path / "hsaf_file.nc"]

        monkeypatch.setattr(
            "sar_validation.downloaders.ascat_soil_moisture_downloader.ASCATSoilMoistureDownloader.download",
            eumdac_download,
        )
        monkeypatch.setattr(
            "sar_validation.downloaders.hsaf_downloader.HSAFDownloader.download",
            hsaf_download,
        )
        orch = self._make_orchestrator(tmp_path, "2025-07-10", today, monkeypatch)
        source = orch.recipe.config.validation_sources[0]

        orch._download_ascat_ssm(source)

        assert {c[0] for c in calls} == {"eumdac", "hsaf"}
        assert orch.metadata["downloads"]["ascat_ssm"]["status"] != "failed"
        assert orch.metadata["errors"] == []

    def test_overlap_range_one_branch_fails_is_partial_not_hard_failure(self, tmp_path, monkeypatch):
        """Same overlap range, but the EUMDAC branch raises while H-SAF
        succeeds and returns real data. This must be reported as a partial
        failure (a notice), not a hard failure (an error + status=failed) --
        the run still has real, usable data from the succeeding branch."""
        import datetime as dt

        today = dt.date.today().isoformat()

        def eumdac_download(self, **kw):
            raise RuntimeError("transient EUMDAC hiccup")

        def hsaf_download(self, **kw):
            return [tmp_path / "hsaf_file.nc"]

        monkeypatch.setattr(
            "sar_validation.downloaders.ascat_soil_moisture_downloader.ASCATSoilMoistureDownloader.download",
            eumdac_download,
        )
        monkeypatch.setattr(
            "sar_validation.downloaders.hsaf_downloader.HSAFDownloader.download",
            hsaf_download,
        )
        orch = self._make_orchestrator(tmp_path, "2025-07-10", today, monkeypatch)
        source = orch.recipe.config.validation_sources[0]

        orch._download_ascat_ssm(source)

        assert orch.metadata["downloads"]["ascat_ssm"]["status"] != "failed"
        assert orch.metadata["errors"] == []
        assert any(
            "partial" in n.lower() or "one of the eumdac" in n.lower()
            for n in orch.metadata["notices"]
        )
