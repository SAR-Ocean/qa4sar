"""Tests for InSituDownloader (Copernicus Marine in-situ platforms)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from sar_validation.downloaders import insitu_downloader
from sar_validation.downloaders.insitu_downloader import (
    ALL_VARIABLES,
    InSituDownloader,
    variables_for_recipe,
)

_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT = 165.0, 180.0, 60.0, 68.0


@pytest.fixture(autouse=True)
def _reset_fetch_stations_cache():
    """_fetch_stations_dry's own shared cache is module-level, keyed by
    query parameters many tests here reuse verbatim (the same default
    bbox/window) -- without resetting it, a test asserting on real data
    could leak its own cached result into a later test asserting on
    empty data for the identical parameters, or vice versa."""
    insitu_downloader._fetch_stations_cache.clear()
    insitu_downloader._fetch_stations_locks.clear()
    yield
    insitu_downloader._fetch_stations_cache.clear()
    insitu_downloader._fetch_stations_locks.clear()


class _FakeCopernicusMarineModule:
    """Minimal stand-in for the copernicusmarine module, exposing just the
    read_dataframe() call _fetch_stations_dry uses. platform_type
    defaults to "MO" (mooring) -- callers that need to test source_types
    filtering pass a different code. rows (a list of dicts, one per
    observation row) overrides the single-row default entirely, for
    station_ranges_dry tests needing multiple stations/observations."""

    def __init__(self, has_data: bool, platform_type: str = "MO", rows: "list[dict] | None" = None):
        self._has_data = has_data
        self._platform_type = platform_type
        self._rows = rows

    def read_dataframe(self, **kwargs):
        if not self._has_data:
            return pd.DataFrame()
        if self._rows is not None:
            return pd.DataFrame(self._rows)
        return pd.DataFrame({
            "variable": ["VHM0"],
            "platform_type": [self._platform_type],
            "platform_id": ["12345"],
            "time": ["2026-08-01T00:30:00"],
            "longitude": [0.0],
            "latitude": [45.0],
            "value": [1.0],
        })


class TestNoDataOutcome:
    def test_no_file_written_at_all_is_treated_as_no_data_not_a_failure(self, tmp_path, caplog):
        """Real-world Copernicus Marine behaviour: subset() reports success (status='000',
        file_status='DOWNLOADED') and raises no exception, yet writes no
        output file at all for a genuinely empty result. This used to fall
        through to a FileNotFoundError, which orchestrator.py's
        _download_insitu logs as an ERROR-level "In-situ download failed"
        -- alarming and misleading for what is actually just "no in-situ
        platforms in this region/period". Must return None (like the
        sibling insitu_currents_historical_downloader.py already does for
        the identical copernicusmarine quirk), not raise."""
        dl = InSituDownloader(output_dir=tmp_path)
        fake_module = MagicMock()

        def fake_subset(**kwargs):
            pass  # "succeeds" but writes nothing, per the real observed behaviour

        fake_module.subset.side_effect = fake_subset
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}), \
             caplog.at_level(logging.DEBUG):
            out = dl.download(
                _MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT,
                "2026-06-01", "2026-06-05",
                source_types=["mooring", "buoy"],
            )

        assert out == []
        assert list(tmp_path.glob("*.csv")) == []
        assert any("No in-situ observations" in r.message for r in caplog.records)

    def test_real_result_still_saved_normally(self, tmp_path, monkeypatch):
        """Sanity check that the no-data handling above doesn't also
        swallow a genuine successful download. copernicusmarine.subset()
        (as called here, with no output_directory kwarg) writes into the
        process CWD -- chdir into tmp_path so that write lands somewhere
        disposable instead of the real repo root."""
        monkeypatch.chdir(tmp_path)
        dl = InSituDownloader(output_dir=tmp_path / "out")
        fake_module = MagicMock()

        expected_name = (
            "cmems_obs-ins_glo_phybgcwav_mynrt_na_irr_"
            "WSPD-WDIR-VAVH-VGHS-VHM0-HCDT-HCSP-EWCT-NSCT_"
            "165.00E-180.00E_60.00N-68.00N_20.00-20.00m_2026-06-01-2026-06-05.csv"
        )

        def fake_subset(**kwargs):
            from pathlib import Path
            Path(expected_name).write_text("platform_type,value\nMO,1.0\n")

        fake_module.subset.side_effect = fake_subset
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            out = dl.download(
                _MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT,
                "2026-06-01", "2026-06-05",
            )

        assert len(out) == 1
        assert out[0].exists()


class TestCheckAvailabilityDry:
    def test_check_availability_dry_returns_true_when_data_exists(self, monkeypatch, tmp_path):
        fake_copernicusmarine = _FakeCopernicusMarineModule(has_data=True)
        monkeypatch.setattr(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader._get_copernicusmarine",
            lambda self: fake_copernicusmarine,
        )

        dl = InSituDownloader(output_dir=tmp_path)
        result = dl.check_availability_dry(
            min_lon=-10.0, max_lon=10.0, min_lat=35.0, max_lat=55.0,
            start="2026-08-01T00:00:00", end="2026-08-01T01:00:00",
        )

        assert result is True

    def test_check_availability_dry_returns_false_when_no_data(self, monkeypatch, tmp_path):
        fake_copernicusmarine = _FakeCopernicusMarineModule(has_data=False)
        monkeypatch.setattr(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader._get_copernicusmarine",
            lambda self: fake_copernicusmarine,
        )

        dl = InSituDownloader(output_dir=tmp_path)
        result = dl.check_availability_dry(
            min_lon=-10.0, max_lon=10.0, min_lat=35.0, max_lat=55.0,
            start="2026-08-01T00:00:00", end="2026-08-01T01:00:00",
        )

        assert result is False

    def test_check_availability_dry_filters_by_source_types(self, monkeypatch, tmp_path):
        """Data exists, but only from a mooring ("MO") platform -- a
        caller asking for "buoy" ("DB") specifically must not see it as
        available, even though the raw fetch was non-empty."""
        fake_copernicusmarine = _FakeCopernicusMarineModule(has_data=True, platform_type="MO")
        monkeypatch.setattr(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader._get_copernicusmarine",
            lambda self: fake_copernicusmarine,
        )

        dl = InSituDownloader(output_dir=tmp_path)
        result = dl.check_availability_dry(
            min_lon=-10.0, max_lon=10.0, min_lat=35.0, max_lat=55.0,
            start="2026-08-01T00:00:00", end="2026-08-01T01:00:00",
            source_types=["buoy"],
        )

        assert result is False

    def test_check_availability_dry_matching_source_type_returns_true(self, monkeypatch, tmp_path):
        fake_copernicusmarine = _FakeCopernicusMarineModule(has_data=True, platform_type="MO")
        monkeypatch.setattr(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader._get_copernicusmarine",
            lambda self: fake_copernicusmarine,
        )

        dl = InSituDownloader(output_dir=tmp_path)
        result = dl.check_availability_dry(
            min_lon=-10.0, max_lon=10.0, min_lat=35.0, max_lat=55.0,
            start="2026-08-01T00:00:00", end="2026-08-01T01:00:00",
            source_types=["mooring"],
        )

        assert result is True


class TestStationRangesDry:
    """station_ranges_dry keeps real per-station coordinates instead of
    check_availability_dry's boolean collapse -- see dry_collocation.py's
    _predict_insitu, which uses this to apply the same real
    point-vs-footprint-shape refinement _predict_ismn already does."""

    def test_empty_result_returns_empty_dict(self, monkeypatch, tmp_path):
        fake_copernicusmarine = _FakeCopernicusMarineModule(has_data=False)
        monkeypatch.setattr(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader._get_copernicusmarine",
            lambda self: fake_copernicusmarine,
        )

        dl = InSituDownloader(output_dir=tmp_path)
        ranges = dl.station_ranges_dry(
            min_lon=-10.0, max_lon=10.0, min_lat=35.0, max_lat=55.0,
            start="2026-08-01T00:00:00", end="2026-08-01T01:00:00",
        )

        assert ranges == {}

    def test_single_station_single_observation(self, monkeypatch, tmp_path):
        fake_copernicusmarine = _FakeCopernicusMarineModule(has_data=True)
        monkeypatch.setattr(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader._get_copernicusmarine",
            lambda self: fake_copernicusmarine,
        )

        dl = InSituDownloader(output_dir=tmp_path)
        ranges = dl.station_ranges_dry(
            min_lon=-10.0, max_lon=10.0, min_lat=35.0, max_lat=55.0,
            start="2026-08-01T00:00:00", end="2026-08-01T01:00:00",
        )

        assert list(ranges.keys()) == ["12345"]
        lat, lon, earliest, latest = ranges["12345"]
        assert lat == 45.0
        assert lon == 0.0
        assert earliest == latest  # a single observation

    def test_multiple_observations_for_one_station_collapse_to_its_own_range(self, monkeypatch, tmp_path):
        """A station reporting several times over the window must appear
        once, with earliest/latest spanning its own real observations --
        not once per row."""
        fake_copernicusmarine = _FakeCopernicusMarineModule(has_data=True, rows=[
            {"platform_id": "A1", "platform_type": "MO", "longitude": 1.0, "latitude": 40.0,
             "time": "2026-08-01T00:00:00"},
            {"platform_id": "A1", "platform_type": "MO", "longitude": 1.0, "latitude": 40.0,
             "time": "2026-08-01T00:30:00"},
            {"platform_id": "A1", "platform_type": "MO", "longitude": 1.0, "latitude": 40.0,
             "time": "2026-08-01T01:00:00"},
        ])
        monkeypatch.setattr(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader._get_copernicusmarine",
            lambda self: fake_copernicusmarine,
        )

        dl = InSituDownloader(output_dir=tmp_path)
        ranges = dl.station_ranges_dry(
            min_lon=-10.0, max_lon=10.0, min_lat=35.0, max_lat=55.0,
            start="2026-08-01T00:00:00", end="2026-08-01T02:00:00",
        )

        assert list(ranges.keys()) == ["A1"]
        _lat, _lon, earliest, latest = ranges["A1"]
        assert earliest.isoformat() == "2026-08-01T00:00:00"
        assert latest.isoformat() == "2026-08-01T01:00:00"

    def test_multiple_distinct_stations_each_get_their_own_entry(self, monkeypatch, tmp_path):
        fake_copernicusmarine = _FakeCopernicusMarineModule(has_data=True, rows=[
            {"platform_id": "A1", "platform_type": "MO", "longitude": 1.0, "latitude": 40.0,
             "time": "2026-08-01T00:00:00"},
            {"platform_id": "B2", "platform_type": "DB", "longitude": -5.0, "latitude": 20.0,
             "time": "2026-08-01T00:15:00"},
        ])
        monkeypatch.setattr(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader._get_copernicusmarine",
            lambda self: fake_copernicusmarine,
        )

        dl = InSituDownloader(output_dir=tmp_path)
        ranges = dl.station_ranges_dry(
            min_lon=-10.0, max_lon=10.0, min_lat=15.0, max_lat=55.0,
            start="2026-08-01T00:00:00", end="2026-08-01T02:00:00",
        )

        assert set(ranges.keys()) == {"A1", "B2"}
        assert ranges["A1"][:2] == (40.0, 1.0)
        assert ranges["B2"][:2] == (20.0, -5.0)

    def test_filters_by_source_types(self, monkeypatch, tmp_path):
        """Only "MO" (mooring) rows exist -- a caller asking for "buoy"
        ("DB") specifically must not see that station at all."""
        fake_copernicusmarine = _FakeCopernicusMarineModule(has_data=True, platform_type="MO")
        monkeypatch.setattr(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader._get_copernicusmarine",
            lambda self: fake_copernicusmarine,
        )

        dl = InSituDownloader(output_dir=tmp_path)
        ranges = dl.station_ranges_dry(
            min_lon=-10.0, max_lon=10.0, min_lat=35.0, max_lat=55.0,
            start="2026-08-01T00:00:00", end="2026-08-01T01:00:00",
            source_types=["buoy"],
        )

        assert ranges == {}


class TestFetchStationsCache:
    """_fetch_stations_dry's own shared cache: --dry-collocation's five
    real in-situ source types (mooring/buoy/ferrybox/drifter/tidal_gauge)
    all run concurrently via predict_collocation's own ThreadPoolExecutor,
    every one of them requesting the exact same bbox/window/dataset_part/
    variables for a single recipe run (cfg.variable is recipe-wide, not
    per-source) -- without this cache, that concurrency means five
    simultaneous, otherwise-identical network requests (and five
    simultaneous auth.marine.copernicus.eu token requests) instead of
    one, which is enough concurrent load to trip that server's own read
    timeout under real-world load."""

    def _counting_fake(self, has_data=True):
        calls = []

        class _Fake:
            def read_dataframe(self, **kwargs):
                calls.append(kwargs)
                if not has_data:
                    return pd.DataFrame()
                return pd.DataFrame({
                    "variable": ["VHM0"], "platform_type": ["MO"], "platform_id": ["12345"],
                    "time": ["2026-08-01T00:30:00"], "longitude": [0.0], "latitude": [45.0],
                    "value": [1.0],
                })

        return _Fake(), calls

    def test_two_callers_with_identical_parameters_share_one_real_fetch(self, monkeypatch, tmp_path):
        """Two source_types requesting the exact same bbox/window/
        dataset_part/variables (only source_types itself differs, which
        is applied locally after the fetch, not part of the server-side
        query) must trigger copernicusmarine.read_dataframe() only once."""
        fake, calls = self._counting_fake()
        monkeypatch.setattr(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader._get_copernicusmarine",
            lambda self: fake,
        )

        dl = InSituDownloader(output_dir=tmp_path)
        dl.station_ranges_dry(
            min_lon=-10.0, max_lon=10.0, min_lat=35.0, max_lat=55.0,
            start="2026-08-01T00:00:00", end="2026-08-01T01:00:00",
            source_types=["mooring"],
        )
        dl.station_ranges_dry(
            min_lon=-10.0, max_lon=10.0, min_lat=35.0, max_lat=55.0,
            start="2026-08-01T00:00:00", end="2026-08-01T01:00:00",
            source_types=["buoy"],
        )

        assert len(calls) == 1

    def test_different_parameters_are_not_shared(self, monkeypatch, tmp_path):
        fake, calls = self._counting_fake()
        monkeypatch.setattr(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader._get_copernicusmarine",
            lambda self: fake,
        )

        dl = InSituDownloader(output_dir=tmp_path)
        dl.station_ranges_dry(
            min_lon=-10.0, max_lon=10.0, min_lat=35.0, max_lat=55.0,
            start="2026-08-01T00:00:00", end="2026-08-01T01:00:00",
        )
        dl.station_ranges_dry(
            min_lon=-20.0, max_lon=20.0, min_lat=35.0, max_lat=55.0,  # different bbox
            start="2026-08-01T00:00:00", end="2026-08-01T01:00:00",
        )

        assert len(calls) == 2

    def test_concurrent_callers_with_identical_parameters_still_share_one_fetch(self, monkeypatch, tmp_path):
        """The real scenario this cache exists for: multiple threads
        (predict_collocation's own ThreadPoolExecutor) requesting the
        same parameters genuinely concurrently, not just sequentially --
        forced to overlap via a barrier so the second thread is
        guaranteed to be waiting on the first's own in-flight fetch."""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        barrier = threading.Barrier(2)
        calls = []

        class _SlowFake:
            def read_dataframe(self, **kwargs):
                calls.append(kwargs)
                barrier.wait(timeout=5)  # force both threads to be mid-call simultaneously
                return pd.DataFrame({
                    "variable": ["VHM0"], "platform_type": ["MO"], "platform_id": ["12345"],
                    "time": ["2026-08-01T00:30:00"], "longitude": [0.0], "latitude": [45.0],
                    "value": [1.0],
                })

        monkeypatch.setattr(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader._get_copernicusmarine",
            lambda self: _SlowFake(),
        )

        dl = InSituDownloader(output_dir=tmp_path)

        def _call(source_type):
            return dl.station_ranges_dry(
                min_lon=-10.0, max_lon=10.0, min_lat=35.0, max_lat=55.0,
                start="2026-08-01T00:00:00", end="2026-08-01T01:00:00",
                source_types=[source_type],
            )

        # Only the leader's own read_dataframe call should ever reach the
        # barrier -- if the cache failed to dedupe, the second thread
        # would ALSO call read_dataframe and hit the barrier itself
        # (which this test's own single-use barrier would then hang on,
        # not silently pass), so a hang here is itself a failure signal.
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(_call, "mooring")
            barrier.wait(timeout=5)
            future_b = executor.submit(_call, "buoy")
            result_a = future_a.result(timeout=5)
            result_b = future_b.result(timeout=5)

        assert len(calls) == 1  # the real proof of dedup: only one leader ever reached the barrier
        # The fake data is a single platform_type="MO" row -- the mooring
        # filter (locally applied post-fetch, from the one shared result)
        # correctly keeps it, the buoy filter correctly excludes it.
        assert list(result_a.keys()) == ["12345"]
        assert result_b == {}


class TestVariablesForRecipe:
    """variables_for_recipe narrows the query to just the physical
    quantities relevant to a recipe's own cfg.variable -- see
    dry_collocation.py's _predict_insitu, which uses this to avoid
    counting e.g. a pure-currents drifter (EWCT/NSCT only) as a
    collocation for a "waves" recipe."""

    def test_waves_maps_to_wave_height_codes(self):
        assert set(variables_for_recipe("waves")) == {"VHM0", "VAVH", "VGHS"}

    def test_wind_maps_to_wind_codes(self):
        assert set(variables_for_recipe("wind")) == {"WSPD", "WDIR"}

    def test_currents_maps_to_current_codes(self):
        assert set(variables_for_recipe("currents")) == {"EWCT", "NSCT", "HCDT", "HCSP"}

    def test_unrecognized_variable_falls_back_to_every_variable(self):
        """Fail-toward-inclusion: a recipe.variable this dataset has no
        dedicated mapping for (e.g. "soil_moisture", which this marine
        in-situ dataset doesn't carry at all) must not silently exclude
        everything -- falls back to the full ALL_VARIABLES set."""
        assert set(variables_for_recipe("soil_moisture")) == set(ALL_VARIABLES)

    def test_every_mapped_code_is_a_real_all_variables_member(self):
        """Every code in every mapping must actually be one of this
        dataset's real variables -- a typo here would silently make a
        recipe's own in-situ availability check always empty."""
        from sar_validation.downloaders.insitu_downloader import RECIPE_VARIABLE_TO_INSITU_VARIABLES

        for variable, codes in RECIPE_VARIABLE_TO_INSITU_VARIABLES.items():
            for code in codes:
                assert code in ALL_VARIABLES, f"{variable!r} maps to {code!r}, not a real ALL_VARIABLES member"
