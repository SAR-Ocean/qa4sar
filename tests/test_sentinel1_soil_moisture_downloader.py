"""Tests for SoilMoistureDownloader (Sentinel-1 CLMS SSM, 1km, Europe, daily)."""

from __future__ import annotations

import zipfile
from unittest.mock import MagicMock

from sar_validation.downloaders.sentinel1_soil_moisture_downloader import (
    DATASET_IDENTIFIER,
    SoilMoistureDownloader,
)

# Bbox comfortably inside PRODUCT_EXTENT (-11, 50, 35, 72)
_EU_MIN_LON, _EU_MAX_LON, _EU_MIN_LAT, _EU_MAX_LAT = -10.0, 20.0, 40.0, 55.0
# Bbox nowhere near PRODUCT_EXTENT
_FAR_MIN_LON, _FAR_MAX_LON, _FAR_MIN_LAT, _FAR_MAX_LAT = 150.0, 160.0, -40.0, -30.0

# CDSE serves both a COG and a NetCDF variant per date/tile; only the "_cog"
# one should survive query()'s filtering (see SoilMoistureDownloader.query).
_PRODUCT_NAME = "c_gls_SSM1km_202601010000_CEURO_S1CSAR_V1.1.1_cog"


def _fake_record(id_="abc", name=_PRODUCT_NAME):
    return {
        "Id": id_,
        "Name": name,
        "ContentDate_Start": "2026-01-01T00:00:00Z",
        "ContentDate_End": "2026-01-01T00:00:10Z",
        "ContentLength_GB": 0.01,
        "Online": True,
    }


class TestSoilMoistureDownloaderEuropeGuard:
    def test_bbox_outside_extent_returns_empty_without_network_call(self, tmp_path):
        dl = SoilMoistureDownloader(output_dir=tmp_path)
        # If the guard didn't short-circuit, this would try to authenticate
        # and raise/hang — assert query() never touches the client at all.
        df = dl.query(
            min_lon=_FAR_MIN_LON, max_lon=_FAR_MAX_LON,
            min_lat=_FAR_MIN_LAT, max_lat=_FAR_MAX_LAT,
            start="2026-01-01", end="2026-01-02",
        )
        assert df.empty
        assert dl._client is None

    def test_bbox_inside_extent_proceeds_to_query(self, tmp_path):
        dl = SoilMoistureDownloader(output_dir=tmp_path)
        fake_client = MagicMock()
        fake_client.query_clms_products.return_value = [_fake_record()]
        dl._client = fake_client

        df = dl.query(
            min_lon=_EU_MIN_LON, max_lon=_EU_MAX_LON,
            min_lat=_EU_MIN_LAT, max_lat=_EU_MAX_LAT,
            start="2026-01-01", end="2026-01-02",
        )
        assert len(df) == 1
        fake_client.query_clms_products.assert_called_once()
        kwargs = fake_client.query_clms_products.call_args.kwargs
        assert kwargs["dataset_identifier"] == DATASET_IDENTIFIER
        assert (kwargs["min_lon"], kwargs["max_lon"]) == (_EU_MIN_LON, _EU_MAX_LON)

    def test_only_cog_variant_kept_when_both_formats_returned(self, tmp_path):
        """CDSE serves both a COG and a NetCDF product per date/tile — only
        the COG one (the converter reads GeoTIFF via rioxarray, not NetCDF)
        should survive query()'s filtering."""
        dl = SoilMoistureDownloader(output_dir=tmp_path)
        fake_client = MagicMock()
        fake_client.query_clms_products.return_value = [
            _fake_record(id_="cog-id", name="c_gls_SSM1km_202601010000_CEURO_S1CSAR_V1.1.1_cog"),
            _fake_record(id_="nc-id", name="c_gls_SSM1km_202601010000_CEURO_S1CSAR_V1.1.1_nc"),
        ]
        dl._client = fake_client

        df = dl.query(
            min_lon=_EU_MIN_LON, max_lon=_EU_MAX_LON,
            min_lat=_EU_MIN_LAT, max_lat=_EU_MAX_LAT,
            start="2026-01-01", end="2026-01-02",
        )
        assert len(df) == 1
        assert df.iloc[0]["Id"] == "cog-id"


class TestSoilMoistureDownloaderQueryAntimeridian:
    def test_crossing_bbox_splits_into_two_windows(self, tmp_path):
        # PRODUCT_EXTENT never crosses the antimeridian, but the split
        # helper must still be exercised for a bbox that does.
        dl = SoilMoistureDownloader(output_dir=tmp_path)
        fake_client = MagicMock()
        fake_client.query_clms_products.return_value = []
        dl._client = fake_client

        dl.query(
            min_lon=40.0, max_lon=-170.0, min_lat=30.0, max_lat=50.0,
            start="2026-01-01", end="2026-01-02",
        )
        # Overlaps PRODUCT_EXTENT via the [40, 180] window, so the guard
        # passes and the split still produces two windows.
        assert fake_client.query_clms_products.call_count == 2


class TestSoilMoistureDownloaderDownload:
    def test_dry_run_returns_empty_and_skips_download(self, tmp_path):
        dl = SoilMoistureDownloader(output_dir=tmp_path, dry_run=True)
        fake_client = MagicMock()
        fake_client.query_clms_products.return_value = [_fake_record()]
        dl._client = fake_client

        paths = dl.download(
            min_lon=_EU_MIN_LON, max_lon=_EU_MAX_LON,
            min_lat=_EU_MIN_LAT, max_lat=_EU_MAX_LAT,
            start="2026-01-01", end="2026-01-02",
        )
        assert paths == []
        fake_client.download_product.assert_not_called()

    def test_skips_product_whose_file_already_exists(self, tmp_path):
        dl = SoilMoistureDownloader(output_dir=tmp_path, dry_run=False)
        fake_client = MagicMock()
        fake_client.query_clms_products.return_value = [_fake_record()]
        dl._client = fake_client
        (tmp_path / _fake_record()["Name"]).write_bytes(b"")

        dl.download(
            min_lon=_EU_MIN_LON, max_lon=_EU_MAX_LON,
            min_lat=_EU_MIN_LAT, max_lat=_EU_MAX_LAT,
            start="2026-01-01", end="2026-01-02",
        )
        fake_client.download_product.assert_not_called()

    def _fake_download_product_zip(self, output_dir, product_name):
        """Simulate CDSE's real response: a .zip containing a per-product
        subfolder with a '-SSM_' soil-moisture GeoTIFF and a sibling
        '-NOISE_' uncertainty-layer GeoTIFF."""
        zip_path = output_dir / f"{product_name}.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(
                f"{product_name}/c_gls_SSM1km-SSM_202601010000_CEURO_S1CSAR_V1.1.1.tiff",
                b"fake-ssm-bytes",
            )
            zf.writestr(
                f"{product_name}/c_gls_SSM1km-NOISE_202601010000_CEURO_S1CSAR_V1.1.1.tiff",
                b"fake-noise-bytes",
            )
        return zip_path

    def test_downloaded_zip_is_unzipped_and_only_ssm_file_returned(self, tmp_path):
        dl = SoilMoistureDownloader(output_dir=tmp_path, dry_run=False)
        fake_client = MagicMock()
        fake_client.query_clms_products.return_value = [_fake_record()]
        fake_client.download_product.side_effect = (
            lambda product_id, output_dir, product_name:
                self._fake_download_product_zip(output_dir, product_name)
        )
        dl._client = fake_client

        paths = dl.download(
            min_lon=_EU_MIN_LON, max_lon=_EU_MAX_LON,
            min_lat=_EU_MIN_LAT, max_lat=_EU_MAX_LAT,
            start="2026-01-01", end="2026-01-02",
        )

        assert len(paths) == 1
        assert paths[0].name == "c_gls_SSM1km-SSM_202601010000_CEURO_S1CSAR_V1.1.1.tiff"
        assert paths[0].exists()
        # The zip itself is removed after extraction (like SARDownloader).
        assert not (tmp_path / f"{_PRODUCT_NAME}.zip").exists()
        # The sibling NOISE file is extracted to disk but not returned.
        noise_path = tmp_path / _PRODUCT_NAME / "c_gls_SSM1km-NOISE_202601010000_CEURO_S1CSAR_V1.1.1.tiff"
        assert noise_path.exists()
        assert noise_path not in paths
