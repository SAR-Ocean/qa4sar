"""Tests for RadiometerDownloader."""

from __future__ import annotations

from datetime import date

import pytest


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class TestRadiometerDownloaderCheckExistsDry:
    def test_check_exists_dry_uses_head_not_get(self, monkeypatch):
        from sar_validation.downloaders.radiometer_downloader import RadiometerDownloader

        head_calls = []

        def _fake_head(url, timeout=None):
            head_calls.append(url)
            return _FakeResponse(200)

        monkeypatch.setattr("requests.head", _fake_head)
        # Fail the test loudly if a real GET is attempted -- this method must
        # never stream a body.
        monkeypatch.setattr(
            "requests.get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not GET")),
        )

        dl = RadiometerDownloader(output_dir=None)
        result = dl.check_exists_dry(day=date(2026, 8, 1), sensors=["amsr2"])

        assert result is True
        assert len(head_calls) >= 1

    def test_returns_false_when_every_candidate_404s(self, monkeypatch):
        from sar_validation.downloaders.radiometer_downloader import RadiometerDownloader

        monkeypatch.setattr("requests.head", lambda url, timeout=None: _FakeResponse(404))

        dl = RadiometerDownloader(output_dir=None)
        result = dl.check_exists_dry(day=date(2026, 8, 1), sensors=["amsr2"])

        assert result is False

    def test_defaults_to_all_supported_sensors_when_omitted(self, monkeypatch):
        from sar_validation.downloaders.radiometer_downloader import SENSORS, RadiometerDownloader, SUPPORTED_SENSORS

        head_calls = []
        monkeypatch.setattr(
            "requests.head",
            lambda url, timeout=None: head_calls.append(url) or _FakeResponse(404),
        )

        dl = RadiometerDownloader(output_dir=None)
        result = dl.check_exists_dry(day=date(2026, 8, 1))

        assert result is False
        # Every supported sensor must have been probed (one HEAD per its
        # primary candidate, plus one more for any sensor with an rt
        # fallback), not just a default sensor.
        expected_candidate_count = sum(
            2 if SENSORS[s].get("rt_url_path") else 1 for s in SUPPORTED_SENSORS
        )
        assert len(head_calls) == expected_candidate_count

    def test_skips_sensor_before_its_availability_start(self, monkeypatch):
        """amsr2's availability_start is 2012-07-02T00:00:00 -- a day before
        that must skip amsr2 entirely (no HEAD calls at all), never issue a
        HEAD for a URL that can't possibly exist."""
        from sar_validation.downloaders.radiometer_downloader import RadiometerDownloader

        head_calls = []
        monkeypatch.setattr(
            "requests.head",
            lambda url, timeout=None: head_calls.append(url) or _FakeResponse(200),
        )

        dl = RadiometerDownloader(output_dir=None)
        result = dl.check_exists_dry(day=date(2012, 7, 1), sensors=["amsr2"])

        assert result is False
        assert head_calls == []

    def test_includes_sensor_exactly_on_its_availability_start_day(self, monkeypatch):
        """The boundary day itself (2012-07-02, matching amsr2's
        availability_start of 2012-07-02T00:00:00) must still be checked --
        a naive bare-date-string vs. full-ISO-datetime-string comparison
        would wrongly treat this day as before availability."""
        from sar_validation.downloaders.radiometer_downloader import RadiometerDownloader

        head_calls = []
        monkeypatch.setattr(
            "requests.head",
            lambda url, timeout=None: head_calls.append(url) or _FakeResponse(200),
        )

        dl = RadiometerDownloader(output_dir=None)
        result = dl.check_exists_dry(day=date(2012, 7, 2), sensors=["amsr2"])

        assert result is True
        assert len(head_calls) >= 1

    def test_falls_back_to_rt_candidate_when_primary_404s(self, monkeypatch):
        from sar_validation.downloaders.radiometer_downloader import RadiometerDownloader

        def _fake_head(url, timeout=None):
            if "-rt." in url:
                return _FakeResponse(200)
            return _FakeResponse(404)

        monkeypatch.setattr("requests.head", _fake_head)

        dl = RadiometerDownloader(output_dir=None)
        result = dl.check_exists_dry(day=date(2026, 8, 1), sensors=["amsr2"])

        assert result is True

    def test_unknown_sensor_key_is_skipped_not_raised(self, monkeypatch):
        from sar_validation.downloaders.radiometer_downloader import RadiometerDownloader

        monkeypatch.setattr("requests.head", lambda url, timeout=None: _FakeResponse(200))

        dl = RadiometerDownloader(output_dir=None)
        result = dl.check_exists_dry(day=date(2026, 8, 1), sensors=["not_a_real_sensor"])

        assert result is False

    def test_network_error_on_one_candidate_does_not_raise(self, monkeypatch):
        import requests

        from sar_validation.downloaders.radiometer_downloader import RadiometerDownloader

        def _fake_head(url, timeout=None):
            raise requests.RequestException("boom")

        monkeypatch.setattr("requests.head", _fake_head)

        dl = RadiometerDownloader(output_dir=None)
        result = dl.check_exists_dry(day=date(2026, 8, 1), sensors=["amsr2"])

        assert result is False
