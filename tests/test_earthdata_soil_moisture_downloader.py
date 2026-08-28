"""Tests for EarthdataSoilMoistureDownloader (SMAP/AMSR2 via NASA
Earthdata/CMR) -- this file only covers list_candidates_dry, added for
the dry-collocation predictor. Mocking follows
dry_collocation.py's own TestSearchNisarSme2Dry convention: `import
earthaccess` inside the method under test is faked via
`monkeypatch.setitem(sys.modules, "earthaccess", fake_earthaccess)`, and
`authenticate_earthdata` is patched on `sar_validation.downloaders.base`
(where the local `from .base import authenticate_earthdata` resolves it
at call time).
"""

from __future__ import annotations

import sys
from datetime import datetime
from unittest.mock import MagicMock

from sar_validation.downloaders.earthdata_soil_moisture_downloader import EarthdataSoilMoistureDownloader


def _fake_granule(native_id: str, begin: str, end: str) -> dict:
    return {
        "meta": {"native-id": native_id},
        "umm": {"TemporalExtent": {"RangeDateTime": {"BeginningDateTime": begin, "EndingDateTime": end}}},
    }


class TestListCandidatesDry:
    def test_returns_matches_without_downloading(self, tmp_path, monkeypatch):
        dl = EarthdataSoilMoistureDownloader(dataset="SPL2SMP_E", version="006", output_dir=tmp_path)

        fake_earthaccess = MagicMock()
        fake_earthaccess.search_data.return_value = [
            _fake_granule("SMAP_granule_1", "2026-01-01T00:30:00Z", "2026-01-01T00:33:00Z"),
        ]
        monkeypatch.setitem(sys.modules, "earthaccess", fake_earthaccess)

        from sar_validation.downloaders import base as _base

        monkeypatch.setattr(_base, "authenticate_earthdata", lambda: None)

        candidates = dl.list_candidates_dry(
            min_lon=-10.0, max_lon=10.0, min_lat=35.0, max_lat=55.0,
            start="2026-01-01T00:00:00", end="2026-01-01T01:00:00",
        )

        fake_earthaccess.download.assert_not_called()
        assert len(candidates) == 1
        granule_id, begin, end = candidates[0]
        assert granule_id == "SMAP_granule_1"
        assert begin == datetime(2026, 1, 1, 0, 30, 0, tzinfo=begin.tzinfo)
        assert end == datetime(2026, 1, 1, 0, 33, 0, tzinfo=end.tzinfo)

    def test_searches_every_configured_candidate_and_merges(self, tmp_path, monkeypatch):
        """A downloader configured with multiple (short_name, version)
        candidates (e.g. a mission that moved CMR collections) must
        search every one and merge the results, mirroring download()'s
        own per-candidate loop."""
        dl = EarthdataSoilMoistureDownloader(
            dataset=[("SHORT_A", "001"), ("SHORT_B", "002")], output_dir=tmp_path,
        )

        fake_earthaccess = MagicMock()
        fake_earthaccess.search_data.side_effect = (
            lambda short_name, version, bounding_box, temporal: [
                _fake_granule(f"{short_name}_granule", "2026-01-01T00:30:00Z", "2026-01-01T00:33:00Z"),
            ]
        )
        monkeypatch.setitem(sys.modules, "earthaccess", fake_earthaccess)

        from sar_validation.downloaders import base as _base

        monkeypatch.setattr(_base, "authenticate_earthdata", lambda: None)

        candidates = dl.list_candidates_dry(
            min_lon=-10.0, max_lon=10.0, min_lat=35.0, max_lat=55.0,
            start="2026-01-01T00:00:00", end="2026-01-01T01:00:00",
        )

        assert fake_earthaccess.search_data.call_count == 2
        assert {c[0] for c in candidates} == {"SHORT_A_granule", "SHORT_B_granule"}

    def test_authenticates_before_searching(self, tmp_path, monkeypatch):
        dl = EarthdataSoilMoistureDownloader(dataset="SPL2SMP_E", version="006", output_dir=tmp_path)

        calls = []
        fake_earthaccess = MagicMock()
        fake_earthaccess.search_data.side_effect = lambda **kwargs: calls.append("search") or []
        monkeypatch.setitem(sys.modules, "earthaccess", fake_earthaccess)

        from sar_validation.downloaders import base as _base

        monkeypatch.setattr(_base, "authenticate_earthdata", lambda: calls.append("auth"))

        dl.list_candidates_dry(
            min_lon=-10.0, max_lon=10.0, min_lat=35.0, max_lat=55.0,
            start="2026-01-01T00:00:00", end="2026-01-01T01:00:00",
        )

        assert calls == ["auth", "search"]

    def test_malformed_granule_is_skipped_not_raised(self, tmp_path, monkeypatch):
        dl = EarthdataSoilMoistureDownloader(dataset="SPL2SMP_E", version="006", output_dir=tmp_path)

        fake_earthaccess = MagicMock()
        fake_earthaccess.search_data.return_value = [
            {"meta": {}, "umm": {}},  # missing TemporalExtent entirely
            _fake_granule("SMAP_granule_ok", "2026-01-01T00:30:00Z", "2026-01-01T00:33:00Z"),
        ]
        monkeypatch.setitem(sys.modules, "earthaccess", fake_earthaccess)

        from sar_validation.downloaders import base as _base

        monkeypatch.setattr(_base, "authenticate_earthdata", lambda: None)

        candidates = dl.list_candidates_dry(
            min_lon=-10.0, max_lon=10.0, min_lat=35.0, max_lat=55.0,
            start="2026-01-01T00:00:00", end="2026-01-01T01:00:00",
        )

        assert len(candidates) == 1
        assert candidates[0][0] == "SMAP_granule_ok"
