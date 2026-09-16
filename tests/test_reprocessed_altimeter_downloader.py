"""Tests for ReprocessedAltimeterDownloader (Copernicus Marine multi-year
along-track altimeter data)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from sar_validation.downloaders.reprocessed_altimeter_downloader import (
    COVERAGE_END,
    MISSIONS,
    ReprocessedAltimeterDownloader,
)


def _fake_response_get(files):
    """Minimal stand-in for copernicusmarine.get()'s ResponseGet -- just
    enough surface (.files, each with .file_path) for download() to read."""
    response = MagicMock()
    response.files = [MagicMock(file_path=Path(f)) for f in files]
    return response


class TestMissionsTable:
    def test_jason_3_registered_with_matching_orbit_key(self):
        assert MISSIONS["jason-3"]["orbit_key"] == "jason-3"
        assert MISSIONS["jason-3"]["start"] == "2016-02-17"
        assert MISSIONS["jason-3"]["end"] == "2023-12-31"

    def test_ers_1_registered_with_matching_orbit_key(self):
        assert MISSIONS["ers-1"]["orbit_key"] == "ers-1"
        assert MISSIONS["ers-1"]["start"] == "1991-08-03"
        assert MISSIONS["ers-1"]["end"] == "1996-06-02"

    def test_topex_poseidon_key_matches_the_files_own_satellite_flag_meaning(self):
        """The product's own "satellite" flag variable uses
        "topex-poseidon" (confirmed against a real downloaded file), not
        "topex" -- the MISSIONS table key must match exactly, since it
        doubles as the platform_id value the converter decodes per
        point."""
        assert "topex-poseidon" in MISSIONS
        assert MISSIONS["topex-poseidon"]["orbit_key"] == "topex"

    def test_sentinel_mission_keys_match_the_files_own_underscore_spelling(self):
        """Confirmed against a real downloaded file's flag_meanings:
        "sentinel-3_a"/"sentinel-3_b"/"sentinel-6_a", not
        "sentinel-3a"/"sentinel-3b"/"sentinel-6a" (which remain valid as
        orbit_coverage.py's own key spelling, via orbit_key)."""
        assert MISSIONS["sentinel-3_a"]["orbit_key"] == "sentinel-3a"
        assert MISSIONS["sentinel-3_b"]["orbit_key"] == "sentinel-3b"
        assert MISSIONS["sentinel-6_a"]["orbit_key"] == "sentinel-6a"

    def test_coverage_end_matches_product_documentation(self):
        assert COVERAGE_END == "2023-12-31"


class TestDownload:
    def test_skips_day_when_no_mission_orbit_crosses_bbox(self, tmp_path):
        dl = ReprocessedAltimeterDownloader(output_dir=tmp_path, dry_run=False)

        fake_module = MagicMock()
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            with patch(
                "sar_validation.core.orbit_coverage.orbit_overlaps_bbox",
                return_value=False,
            ):
                downloaded = dl.download(
                    min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                    start="2023-12-30", end="2023-12-30",
                )

        fake_module.get.assert_not_called()
        assert downloaded == []

    def test_downloads_whole_day_file_when_any_mission_orbit_crosses(self, tmp_path):
        dl = ReprocessedAltimeterDownloader(output_dir=tmp_path, dry_run=False)

        fake_module = MagicMock()
        fake_module.get.return_value = _fake_response_get(
            [str(tmp_path / "ESACCI-SEASTATE-L3-SWH-MULTI_1D-20231230-fv01.nc")]
        )

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            with patch(
                "sar_validation.core.orbit_coverage.orbit_overlaps_bbox",
                return_value=True,
            ):
                downloaded = dl.download(
                    min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                    start="2023-12-30", end="2023-12-30",
                )

        assert len(downloaded) == 1
        assert downloaded[0].name == "ESACCI-SEASTATE-L3-SWH-MULTI_1D-20231230-fv01.nc"
        fake_module.get.assert_called_once()
        call_kwargs = fake_module.get.call_args.kwargs
        assert call_kwargs["dataset_id"] == "cci_obs-wave_glo_phy-swh_my_l3_PT1S-i"
        assert call_kwargs["filter"] == "*20231230*"

    def test_clips_requested_window_to_coverage_end(self, tmp_path):
        """A window entirely after 2023-12-31 has no reprocessed data --
        must return without ever calling get()."""
        dl = ReprocessedAltimeterDownloader(output_dir=tmp_path, dry_run=False)

        fake_module = MagicMock()
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            downloaded = dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2024-01-02", end="2024-01-03",
            )

        fake_module.get.assert_not_called()
        assert downloaded == []

    def test_no_matching_remote_files_is_not_an_error(self, tmp_path):
        """copernicusmarine.get() returns an empty files list (not an
        exception) when nothing matches its filter -- must be treated as
        "no data this day", not a crash."""
        dl = ReprocessedAltimeterDownloader(output_dir=tmp_path, dry_run=False)

        fake_module = MagicMock()
        fake_module.get.return_value = _fake_response_get([])

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            with patch(
                "sar_validation.core.orbit_coverage.orbit_overlaps_bbox",
                return_value=True,
            ):
                downloaded = dl.download(
                    min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                    start="2023-12-30", end="2023-12-30",
                )

        assert downloaded == []

    def test_force_download_passes_overwrite_to_get(self, tmp_path):
        dl = ReprocessedAltimeterDownloader(output_dir=tmp_path, dry_run=False, force_download=True)

        fake_module = MagicMock()
        fake_module.get.return_value = _fake_response_get(
            [str(tmp_path / "ESACCI-SEASTATE-L3-SWH-MULTI_1D-20231230-fv01.nc")]
        )

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            with patch(
                "sar_validation.core.orbit_coverage.orbit_overlaps_bbox",
                return_value=True,
            ):
                dl.download(
                    min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                    start="2023-12-30", end="2023-12-30",
                )

        call_kwargs = fake_module.get.call_args.kwargs
        assert call_kwargs["overwrite"] is True
        assert call_kwargs["skip_existing"] is False

    def test_dry_run_does_not_call_get(self, tmp_path):
        dl = ReprocessedAltimeterDownloader(output_dir=tmp_path, dry_run=True)

        fake_module = MagicMock()
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            with patch(
                "sar_validation.core.orbit_coverage.orbit_overlaps_bbox",
                return_value=True,
            ):
                downloaded = dl.download(
                    min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                    start="2023-12-30", end="2023-12-30",
                )

        fake_module.get.assert_not_called()
        assert downloaded == []
