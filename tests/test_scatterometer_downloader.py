"""Tests for ScatterometerDownloader (EUMDAC OSI-104 ASCAT winds) --
this file only covers list_candidates_dry, added for the dry-collocation
predictor. Mocking follows test_downloaders.py's own convention: `import
eumdac` inside the method under test is faked via
`patch.dict("sys.modules", {"eumdac": fake_eumdac})`, not by patching
`eumdac.DataStore` directly.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from sar_validation.downloaders.scatterometer_downloader import ScatterometerDownloader


class TestListCandidatesDry:
    def test_returns_matches_without_downloading(self, tmp_path):
        """Mirrors download()'s own EUMDAC search/dedup/satellite-filter
        logic exactly, but never calls datastore.get_product."""
        dl = ScatterometerDownloader(output_dir=tmp_path, dry_run=False)
        dl._token = "fake-token"

        fake_product = MagicMock()
        fake_product.__str__.return_value = "ASCA_SZR_02_M02_20260101003000Z_metopb_..."
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
        assert product_id == "ASCA_SZR_02_M02_20260101003000Z_metopb_..."
        assert sensing_start == datetime(2026, 1, 1, 0, 30, 0)
        assert sensing_end == datetime(2026, 1, 1, 0, 33, 0)

    def test_dedupes_products_seen_in_multiple_antimeridian_windows(self, tmp_path):
        dl = ScatterometerDownloader(output_dir=tmp_path, dry_run=False)
        dl._token = "fake-token"

        fake_product = MagicMock()
        fake_product.__str__.return_value = "dup_metopb"
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

    def test_products_not_matching_any_known_satellite_are_dropped(self, tmp_path):
        """download() only downloads products whose ID contains one of
        SATELLITES ("metopb"/"metopc") -- list_candidates_dry must apply
        the same filter, so it never reports a candidate download()
        itself would silently skip."""
        dl = ScatterometerDownloader(output_dir=tmp_path, dry_run=False)
        dl._token = "fake-token"

        unknown_product = MagicMock()
        unknown_product.__str__.return_value = "ASCA_SZR_02_M02_20260101003000Z_unknownsat_..."
        unknown_product.sensing_start = datetime(2026, 1, 1, 0, 30, 0)
        unknown_product.sensing_end = datetime(2026, 1, 1, 0, 33, 0)

        known_product = MagicMock()
        known_product.__str__.return_value = "ASCA_SZR_02_M03_20260101003000Z_metopc_..."
        known_product.sensing_start = datetime(2026, 1, 1, 0, 30, 0)
        known_product.sensing_end = datetime(2026, 1, 1, 0, 33, 0)

        fake_eumdac = MagicMock()
        fake_collection = MagicMock()
        fake_collection.search.return_value = [unknown_product, known_product]
        fake_datastore = MagicMock()
        fake_datastore.get_collection.return_value = fake_collection
        fake_eumdac.DataStore.return_value = fake_datastore

        with patch.dict("sys.modules", {"eumdac": fake_eumdac}):
            candidates = dl.list_candidates_dry(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-01-01T00:00:00", end="2026-01-01T01:00:00",
            )

        assert len(candidates) == 1
        assert candidates[0][0] == "ASCA_SZR_02_M03_20260101003000Z_metopc_..."
