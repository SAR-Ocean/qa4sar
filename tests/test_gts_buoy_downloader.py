"""Tests for GTSBuoyDownloader."""

from __future__ import annotations

import pytest

from sar_validation.downloaders.gts_buoy_downloader import GTSBuoyDownloader


class TestGTSBuoyDownloaderRequestShape:
    def test_single_day_window_issues_one_request(self, tmp_path, monkeypatch):
        calls = []

        def fake_execute(self, request, target):
            calls.append((request, target))
            target.write_bytes(b"")

        monkeypatch.setattr(GTSBuoyDownloader, "_execute_mars_request", fake_execute)

        dl = GTSBuoyDownloader(output_dir=tmp_path)
        paths = dl.download(
            min_lon=-10, max_lon=10, min_lat=40, max_lat=55,
            start="2026-08-30T00:00:00", end="2026-08-30T23:59:00",
        )

        assert len(calls) == 1
        request, target = calls[0]
        assert request["class"] == "od"
        assert request["type"] == "ob"
        assert request["stream"] == "oper"
        assert request["obstype"] == "181/182"
        assert request["date"] == "20260830"
        assert request["time"] == "0000"
        assert request["range"] == "1439"
        assert paths == [target]

    def test_multi_day_window_issues_one_request_per_day(self, tmp_path, monkeypatch):
        calls = []

        def fake_execute(self, request, target):
            calls.append(request)
            target.write_bytes(b"")

        monkeypatch.setattr(GTSBuoyDownloader, "_execute_mars_request", fake_execute)

        dl = GTSBuoyDownloader(output_dir=tmp_path)
        dl.download(
            min_lon=-10, max_lon=10, min_lat=40, max_lat=55,
            start="2026-08-30T18:00:00", end="2026-09-01T06:00:00",
        )

        dates = [c["date"] for c in calls]
        assert dates == ["20260830", "20260831", "20260901"]

    def test_target_filename_encodes_day_and_bbox(self, tmp_path, monkeypatch):
        def fake_execute(self, request, target):
            target.write_bytes(b"")

        monkeypatch.setattr(GTSBuoyDownloader, "_execute_mars_request", fake_execute)

        dl = GTSBuoyDownloader(output_dir=tmp_path)
        paths = dl.download(
            min_lon=-10, max_lon=10, min_lat=40, max_lat=55,
            start="2026-08-30T00:00:00", end="2026-08-30T12:00:00",
        )

        assert paths[0].name == "gts_buoy_20260830.bufr"
        assert paths[0].parent == tmp_path


class TestGTSBuoyDownloaderExistingFile:
    def test_existing_day_file_is_skipped(self, tmp_path, monkeypatch):
        existing = tmp_path / "gts_buoy_20260830.bufr"
        existing.write_bytes(b"already here")

        def fake_execute(self, request, target):
            raise AssertionError("must not re-request an already-downloaded day")

        monkeypatch.setattr(GTSBuoyDownloader, "_execute_mars_request", fake_execute)

        dl = GTSBuoyDownloader(output_dir=tmp_path)
        paths = dl.download(
            min_lon=-10, max_lon=10, min_lat=40, max_lat=55,
            start="2026-08-30T00:00:00", end="2026-08-30T12:00:00",
        )

        assert paths == [existing]
        assert existing.read_bytes() == b"already here"


class TestGTSBuoyDownloaderDryRun:
    def test_dry_run_makes_no_request_and_returns_no_files(self, tmp_path, monkeypatch):
        def fake_execute(self, request, target):
            raise AssertionError("dry_run must not call _execute_mars_request")

        monkeypatch.setattr(GTSBuoyDownloader, "_execute_mars_request", fake_execute)

        dl = GTSBuoyDownloader(output_dir=tmp_path, dry_run=True)
        paths = dl.download(
            min_lon=-10, max_lon=10, min_lat=40, max_lat=55,
            start="2026-08-30T00:00:00", end="2026-08-30T12:00:00",
        )

        assert paths == []
        assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


class TestGTSBuoyDownloaderRequestFailure:
    def test_failed_request_removes_partial_file_and_raises(self, tmp_path, monkeypatch):
        def fake_execute(self, request, target):
            target.write_bytes(b"partial")
            raise RuntimeError("MARS request failed")

        monkeypatch.setattr(GTSBuoyDownloader, "_execute_mars_request", fake_execute)

        dl = GTSBuoyDownloader(output_dir=tmp_path)
        with pytest.raises(RuntimeError, match="MARS request failed"):
            dl.download(
                min_lon=-10, max_lon=10, min_lat=40, max_lat=55,
                start="2026-08-30T00:00:00", end="2026-08-30T12:00:00",
            )

        assert not (tmp_path / "gts_buoy_20260830.bufr").exists()
