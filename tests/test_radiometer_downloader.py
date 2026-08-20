"""Tests for RadiometerDownloader."""

from __future__ import annotations

from datetime import date


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class TestRadiometerDownloaderCheckExistsDry:
    def test_check_exists_dry_uses_head_not_get(self, monkeypatch, tmp_path):
        from sar_validation.downloaders.radiometer_downloader import RadiometerDownloader

        head_calls = []

        def _fake_head(url, timeout=None, allow_redirects=None):
            head_calls.append((url, allow_redirects))
            return _FakeResponse(200)

        monkeypatch.setattr("requests.head", _fake_head)
        # Fail the test loudly if a real GET is attempted -- this method must
        # never stream a body.
        monkeypatch.setattr(
            "requests.get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not GET")),
        )

        dl = RadiometerDownloader(output_dir=tmp_path)
        result = dl.check_exists_dry(day=date(2026, 8, 1), sensors=["amsr2"])

        assert result is True
        assert len(head_calls) >= 1
        # allow_redirects=True is passed explicitly -- requests defaults HEAD
        # to no redirect-following, which would otherwise silently look like
        # "not found" if the server ever redirects.
        assert all(allow_redirects is True for _url, allow_redirects in head_calls)

    def test_returns_false_when_every_candidate_404s(self, monkeypatch, tmp_path):
        from sar_validation.downloaders.radiometer_downloader import RadiometerDownloader

        monkeypatch.setattr("requests.head", lambda url, timeout=None, allow_redirects=None: _FakeResponse(404))

        dl = RadiometerDownloader(output_dir=tmp_path)
        result = dl.check_exists_dry(day=date(2026, 8, 1), sensors=["amsr2"])

        assert result is False

    def test_defaults_to_all_supported_sensors_when_omitted(self, monkeypatch, tmp_path):
        from sar_validation.downloaders.radiometer_downloader import SENSORS, SUPPORTED_SENSORS, RadiometerDownloader

        head_calls = []
        monkeypatch.setattr(
            "requests.head",
            lambda url, timeout=None, allow_redirects=None: head_calls.append(url) or _FakeResponse(404),
        )

        dl = RadiometerDownloader(output_dir=tmp_path)
        result = dl.check_exists_dry(day=date(2026, 8, 1))

        assert result is False
        # Every supported sensor must have been probed (one HEAD per its
        # primary candidate, plus one more for any sensor with an rt
        # fallback), not just a default sensor.
        expected_candidate_count = sum(
            2 if SENSORS[s].get("rt_url_path") else 1 for s in SUPPORTED_SENSORS
        )
        assert len(head_calls) == expected_candidate_count

    def test_skips_sensor_before_its_availability_start(self, monkeypatch, tmp_path):
        """amsr2's availability_start is 2012-07-02T00:00:00 -- a day before
        that must skip amsr2 entirely (no HEAD calls at all), never issue a
        HEAD for a URL that can't possibly exist."""
        from sar_validation.downloaders.radiometer_downloader import RadiometerDownloader

        head_calls = []
        monkeypatch.setattr(
            "requests.head",
            lambda url, timeout=None, allow_redirects=None: head_calls.append(url) or _FakeResponse(200),
        )

        dl = RadiometerDownloader(output_dir=tmp_path)
        result = dl.check_exists_dry(day=date(2012, 7, 1), sensors=["amsr2"])

        assert result is False
        assert head_calls == []

    def test_includes_sensor_exactly_on_its_availability_start_day(self, monkeypatch, tmp_path):
        """The boundary day itself (2012-07-02, matching amsr2's
        availability_start of 2012-07-02T00:00:00) must still be checked --
        a naive bare-date-string vs. full-ISO-datetime-string comparison
        would wrongly treat this day as before availability."""
        from sar_validation.downloaders.radiometer_downloader import RadiometerDownloader

        head_calls = []
        monkeypatch.setattr(
            "requests.head",
            lambda url, timeout=None, allow_redirects=None: head_calls.append(url) or _FakeResponse(200),
        )

        dl = RadiometerDownloader(output_dir=tmp_path)
        result = dl.check_exists_dry(day=date(2012, 7, 2), sensors=["amsr2"])

        assert result is True
        assert len(head_calls) >= 1

    def test_falls_back_to_rt_candidate_when_primary_404s(self, monkeypatch, tmp_path):
        from sar_validation.downloaders.radiometer_downloader import RadiometerDownloader

        def _fake_head(url, timeout=None, allow_redirects=None):
            if "-rt." in url:
                return _FakeResponse(200)
            return _FakeResponse(404)

        monkeypatch.setattr("requests.head", _fake_head)

        dl = RadiometerDownloader(output_dir=tmp_path)
        result = dl.check_exists_dry(day=date(2026, 8, 1), sensors=["amsr2"])

        assert result is True

    def test_unknown_sensor_key_is_skipped_not_raised(self, monkeypatch, tmp_path):
        from sar_validation.downloaders.radiometer_downloader import RadiometerDownloader

        monkeypatch.setattr("requests.head", lambda url, timeout=None, allow_redirects=None: _FakeResponse(200))

        dl = RadiometerDownloader(output_dir=tmp_path)
        result = dl.check_exists_dry(day=date(2026, 8, 1), sensors=["not_a_real_sensor"])

        assert result is False

    def test_raises_when_every_candidate_raises_network_error(self, monkeypatch, tmp_path):
        """A network error (DNS failure, connection refused, timeout) on
        EVERY candidate is not a definitive "doesn't exist" answer -- unlike
        download()'s streamed GET, which only cares whether it managed to
        fetch a file, this dry-check must distinguish "confirmed absent"
        from "couldn't check". If every attempted candidate raises, this
        must raise (or otherwise signal "unknown"), never silently return
        False -- a false False here would make a real download look
        incorrectly skippable."""
        import requests

        from sar_validation.downloaders.radiometer_downloader import RadiometerDownloader

        def _fake_head(url, timeout=None, allow_redirects=None):
            raise requests.RequestException("boom")

        monkeypatch.setattr("requests.head", _fake_head)

        dl = RadiometerDownloader(output_dir=tmp_path)
        try:
            dl.check_exists_dry(day=date(2026, 8, 1), sensors=["amsr2"])
        except Exception:
            pass
        else:
            raise AssertionError(
                "check_exists_dry must raise (or otherwise signal 'unknown') when every "
                "candidate URL request fails with a network error, not silently return False."
            )

    def test_false_when_some_candidates_error_but_others_get_definitive_404(self, monkeypatch, tmp_path):
        """A mix of a network error on one candidate and a real (404)
        response on another is still a genuine "not found" -- at least one
        candidate produced a definitive answer, so False is legitimate here,
        unlike the all-errors case above."""
        import requests

        from sar_validation.downloaders.radiometer_downloader import RadiometerDownloader

        def _fake_head(url, timeout=None, allow_redirects=None):
            if "-rt." in url:
                raise requests.RequestException("boom")
            return _FakeResponse(404)

        monkeypatch.setattr("requests.head", _fake_head)

        dl = RadiometerDownloader(output_dir=tmp_path)
        result = dl.check_exists_dry(day=date(2026, 8, 1), sensors=["amsr2"])

        assert result is False
