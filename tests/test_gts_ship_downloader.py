"""Tests for ShipDownloader."""

from __future__ import annotations

from sar_validation.downloaders.gts_ship_downloader import ShipDownloader


class TestShipDownloaderRequestShape:
    def test_single_day_window_issues_one_request(self, tmp_path, monkeypatch):
        calls = []

        def fake_execute(self, request, target):
            calls.append((request, target))
            target.write_bytes(b"")

        monkeypatch.setattr(ShipDownloader, "_execute_mars_request", fake_execute)

        dl = ShipDownloader(output_dir=tmp_path)
        paths = dl.download(
            min_lon=-10, max_lon=10, min_lat=40, max_lat=55,
            start="2026-08-30T00:00:00", end="2026-08-30T23:59:00",
        )

        assert len(calls) == 1
        request, target = calls[0]
        assert request["class"] == "od"
        assert request["type"] == "ob"
        assert request["stream"] == "oper"
        assert request["obstype"] == "180"
        assert request["date"] == "20260830"
        assert request["time"] == "0000"
        assert request["range"] == "1439"
        assert paths == [target]

    def test_multi_day_window_issues_one_request_per_day(self, tmp_path, monkeypatch):
        calls = []

        def fake_execute(self, request, target):
            calls.append(request)
            target.write_bytes(b"")

        monkeypatch.setattr(ShipDownloader, "_execute_mars_request", fake_execute)

        dl = ShipDownloader(output_dir=tmp_path)
        dl.download(
            min_lon=-10, max_lon=10, min_lat=40, max_lat=55,
            start="2026-08-30T18:00:00", end="2026-09-01T06:00:00",
        )

        dates = [c["date"] for c in calls]
        assert dates == ["20260830", "20260831", "20260901"]

    def test_target_filename_encodes_day(self, tmp_path, monkeypatch):
        def fake_execute(self, request, target):
            target.write_bytes(b"")

        monkeypatch.setattr(ShipDownloader, "_execute_mars_request", fake_execute)

        dl = ShipDownloader(output_dir=tmp_path)
        paths = dl.download(
            min_lon=-10, max_lon=10, min_lat=40, max_lat=55,
            start="2026-08-30T00:00:00", end="2026-08-30T12:00:00",
        )

        assert paths[0].name == "gts_ship_20260830.bufr"
        assert paths[0].parent == tmp_path


class TestShipDownloaderExistingFile:
    def test_existing_day_file_is_skipped(self, tmp_path, monkeypatch):
        existing = tmp_path / "gts_ship_20260830.bufr"
        existing.write_bytes(b"already here")

        def fake_execute(self, request, target):
            raise AssertionError("must not re-request an already-downloaded day")

        monkeypatch.setattr(ShipDownloader, "_execute_mars_request", fake_execute)

        dl = ShipDownloader(output_dir=tmp_path)
        paths = dl.download(
            min_lon=-10, max_lon=10, min_lat=40, max_lat=55,
            start="2026-08-30T00:00:00", end="2026-08-30T12:00:00",
        )

        assert paths == [existing]

    def test_force_download_re_requests_an_existing_day(self, tmp_path, monkeypatch):
        existing = tmp_path / "gts_ship_20260830.bufr"
        existing.write_bytes(b"stale")

        calls = []

        def fake_execute(self, request, target):
            calls.append(request)
            target.write_bytes(b"fresh")

        monkeypatch.setattr(ShipDownloader, "_execute_mars_request", fake_execute)

        dl = ShipDownloader(output_dir=tmp_path, force_download=True)
        dl.download(
            min_lon=-10, max_lon=10, min_lat=40, max_lat=55,
            start="2026-08-30T00:00:00", end="2026-08-30T12:00:00",
        )

        assert len(calls) == 1


class TestShipDownloaderDryRun:
    def test_dry_run_never_executes_a_request(self, tmp_path, monkeypatch):
        def fake_execute(self, request, target):
            raise AssertionError("dry_run must not call MARS")

        monkeypatch.setattr(ShipDownloader, "_execute_mars_request", fake_execute)

        dl = ShipDownloader(output_dir=tmp_path, dry_run=True)
        paths = dl.download(
            min_lon=-10, max_lon=10, min_lat=40, max_lat=55,
            start="2026-08-30T00:00:00", end="2026-08-30T12:00:00",
        )

        assert paths == []
        assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


class TestShipDownloaderTimeout:
    def test_a_hung_request_times_out_and_abandons_remaining_days(self, tmp_path, monkeypatch):
        import threading

        release = threading.Event()
        calls = []

        def fake_execute(self, request, target):
            calls.append(request["date"])
            release.wait()  # never released within the test -- simulates a hung MARS job

        monkeypatch.setattr(ShipDownloader, "_execute_mars_request", fake_execute)
        monkeypatch.setattr(
            "sar_validation.downloaders.gts_ship_downloader._MARS_REQUEST_TIMEOUT_SECONDS", 0.05,
        )

        dl = ShipDownloader(output_dir=tmp_path)
        paths = dl.download(
            min_lon=-10, max_lon=10, min_lat=40, max_lat=55,
            start="2026-08-30T00:00:00", end="2026-09-01T00:00:00",
        )

        assert calls == ["20260830"]
        assert paths == []
        release.set()  # let the background thread finish so it does not leak into other tests
