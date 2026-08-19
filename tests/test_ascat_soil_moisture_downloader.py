"""Tests for ASCATSoilMoistureDownloader's dry-collocation candidate listing.

download()'s own tests (search/dedup, antimeridian, force-download) live in
test_downloaders.py alongside the sibling scatterometer downloader's tests --
this file only covers list_candidates_dry, added for the dry-collocation
predictor. Mocking follows test_downloaders.py's own convention: `import
eumdac` inside the method under test is faked via
`patch.dict("sys.modules", {"eumdac": fake_eumdac})`, not by patching
`eumdac.DataStore` directly.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from sar_validation.downloaders.ascat_soil_moisture_downloader import ASCATSoilMoistureDownloader


class TestListCandidatesDry:
    def test_returns_matches_without_downloading(self, tmp_path):
        """Mirrors download()'s own EUMDAC search/dedup logic exactly, but
        never calls datastore.get_product."""
        dl = ASCATSoilMoistureDownloader(output_dir=tmp_path, dry_run=False)
        dl._token = "fake-token"

        fake_product = MagicMock()
        fake_product.__str__.return_value = "ASCA_SMR_02_M01_20260101003000Z_..."
        fake_product.sensing_start = datetime(2026, 1, 1, 0, 30, 0)
        fake_product.sensing_end = datetime(2026, 1, 1, 0, 33, 0)

        fake_eumdac = MagicMock()
        fake_collection = MagicMock()
        fake_collection.search.return_value = [fake_product]
        fake_datastore = MagicMock()
        fake_datastore.get_collection.return_value = fake_collection
        fake_eumdac.DataStore.return_value = fake_datastore

        with patch.dict("sys.modules", {"eumdac": fake_eumdac}):
            candidates = dl.list_candidates_dry(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-01-01T00:00:00", end="2026-01-01T01:00:00",
            )

        fake_datastore.get_product.assert_not_called()
        assert len(candidates) == 1
        product_id, sensing_start, sensing_end = candidates[0]
        assert product_id == "ASCA_SMR_02_M01_20260101003000Z_..."
        assert sensing_start == datetime(2026, 1, 1, 0, 30, 0)
        assert sensing_end == datetime(2026, 1, 1, 0, 33, 0)

    def test_dedupes_products_seen_in_multiple_antimeridian_windows(self, tmp_path):
        dl = ASCATSoilMoistureDownloader(output_dir=tmp_path, dry_run=False)
        dl._token = "fake-token"

        fake_product = MagicMock()
        fake_product.__str__.return_value = "dup"
        fake_product.sensing_start = datetime(2026, 1, 1, 0, 30, 0)
        fake_product.sensing_end = datetime(2026, 1, 1, 0, 33, 0)

        fake_eumdac = MagicMock()
        fake_collection = MagicMock()
        fake_collection.search.side_effect = [[fake_product], [fake_product]]
        fake_datastore = MagicMock()
        fake_datastore.get_collection.return_value = fake_collection
        fake_eumdac.DataStore.return_value = fake_datastore

        with patch.dict("sys.modules", {"eumdac": fake_eumdac}):
            candidates = dl.list_candidates_dry(
                min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0,
                start="2026-07-02", end="2026-07-03",
            )

        assert fake_collection.search.call_count == 2
        assert len(candidates) == 1

    def test_no_products_returns_empty(self, tmp_path):
        dl = ASCATSoilMoistureDownloader(output_dir=tmp_path, dry_run=False)
        dl._token = "fake-token"

        fake_eumdac = MagicMock()
        fake_collection = MagicMock()
        fake_collection.search.return_value = []
        fake_datastore = MagicMock()
        fake_datastore.get_collection.return_value = fake_collection
        fake_eumdac.DataStore.return_value = fake_datastore

        with patch.dict("sys.modules", {"eumdac": fake_eumdac}):
            candidates = dl.list_candidates_dry(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-01-01T00:00:00", end="2026-01-01T01:00:00",
            )

        assert candidates == []
