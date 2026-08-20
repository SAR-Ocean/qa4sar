"""Tests for AltimeterDownloader (Copernicus Marine along-track data) --
this file only covers list_candidates_dry, added for the dry-collocation
predictor. `import copernicusmarine` inside the method under test is
faked via `patch.dict("sys.modules", {"copernicusmarine": fake_module})`,
matching test_downloaders.py's TestAltimeterDownloaderAntimeridian
convention for this same class's download() method.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np

from sar_validation.downloaders.altimeter_downloader import AltimeterDownloader


class _FakeDataArray:
    def __init__(self, values):
        self.values = values


class _FakeDataset:
    """Minimal stand-in for the xarray.Dataset copernicusmarine.open_dataset
    returns -- just enough surface (.sizes, ["time"].values) for
    list_candidates_dry to read."""

    def __init__(self, time_values):
        self.sizes = {"time": len(time_values)}
        self._time = _FakeDataArray(np.array(time_values, dtype="datetime64[ns]"))

    def __getitem__(self, key):
        assert key == "time"
        return self._time


class TestListCandidatesDry:
    def test_returns_real_time_range_without_subsetting(self, tmp_path):
        dl = AltimeterDownloader(output_dir=tmp_path, dry_run=False)

        fake_module = MagicMock()
        fake_module.open_dataset.return_value = _FakeDataset(
            ["2026-06-01T10:00:00", "2026-06-01T10:05:00"]
        )

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            candidates = dl.list_candidates_dry(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-06-01", end="2026-06-02",
                frequencies=["1hz"], satellites=["al"],
            )

        fake_module.subset.assert_not_called()
        assert len(candidates) == 1
        dataset_id, t_min, t_max = candidates[0]
        assert dataset_id == "cmems_obs-wave_glo_phy-swh_nrt_al-l3_PT1S"
        assert t_min == datetime(2026, 6, 1, 10, 0, 0)
        assert t_max == datetime(2026, 6, 1, 10, 5, 0)

    def test_dataset_with_no_data_in_window_is_skipped(self, tmp_path):
        dl = AltimeterDownloader(output_dir=tmp_path, dry_run=False)

        fake_module = MagicMock()
        fake_module.open_dataset.return_value = _FakeDataset([])

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            candidates = dl.list_candidates_dry(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-06-01", end="2026-06-02",
                frequencies=["1hz"], satellites=["al"],
            )

        assert candidates == []

    def test_open_dataset_failure_for_one_satellite_does_not_abort_others(self, tmp_path):
        """A raised exception (e.g. an out-of-bounds selection) for one
        satellite must not prevent the remaining satellites/frequencies
        from being checked -- mirrors download()'s own "no data in this
        region/time window" per-satellite handling."""
        dl = AltimeterDownloader(output_dir=tmp_path, dry_run=False)

        fake_module = MagicMock()

        def _fake_open_dataset(dataset_id, **kwargs):
            if "al" in dataset_id:
                raise RuntimeError("no matching data")
            return _FakeDataset(["2026-06-01T10:00:00"])

        fake_module.open_dataset.side_effect = _fake_open_dataset

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            candidates = dl.list_candidates_dry(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-06-01", end="2026-06-02",
                frequencies=["1hz"], satellites=["al", "c2"],
            )

        assert len(candidates) == 1
        assert candidates[0][0] == "cmems_obs-wave_glo_phy-swh_nrt_c2-l3_PT1S"

    def test_frequency_before_its_availability_start_is_skipped(self, tmp_path):
        dl = AltimeterDownloader(output_dir=tmp_path, dry_run=False)

        fake_module = MagicMock()

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            candidates = dl.list_candidates_dry(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-01-01", end="2026-01-02",
                frequencies=["5hz"], satellites=["al"],
            )

        fake_module.open_dataset.assert_not_called()
        assert candidates == []

    def test_raises_when_every_candidate_raises(self, tmp_path):
        """A network/auth error on every single candidate is not the same
        as a definitive "no data" answer -- unlike the legitimate
        out-of-bounds case, this must not silently return []. The caller
        (_predict_catalog_precise_source) relies on this raising to
        produce "unknown" rather than a false "none-predicted"."""
        dl = AltimeterDownloader(output_dir=tmp_path, dry_run=False)

        fake_module = MagicMock()
        fake_module.open_dataset.side_effect = RuntimeError("connection refused")

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            try:
                dl.list_candidates_dry(
                    min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                    start="2026-06-01", end="2026-06-02",
                    frequencies=["1hz"], satellites=["al", "c2"],
                )
            except RuntimeError:
                pass
            else:
                raise AssertionError("expected list_candidates_dry to raise")

    def test_mixed_network_error_and_definitive_empty_result_does_not_raise(self, tmp_path):
        """One candidate raising (network error) alongside another that
        definitively resolves (even to empty/no data) is still a real
        signal -- not every candidate failed, so this must return
        normally (an empty list here, since the one definitive candidate
        found no data) rather than raising."""
        dl = AltimeterDownloader(output_dir=tmp_path, dry_run=False)

        fake_module = MagicMock()

        def _fake_open_dataset(dataset_id, **kwargs):
            if "al" in dataset_id:
                raise RuntimeError("connection refused")
            return _FakeDataset([])

        fake_module.open_dataset.side_effect = _fake_open_dataset

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            candidates = dl.list_candidates_dry(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-06-01", end="2026-06-02",
                frequencies=["1hz"], satellites=["al", "c2"],
            )

        assert candidates == []
