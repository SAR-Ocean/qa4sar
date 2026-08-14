"""
Download Sentinel-1 CLMS Surface Soil Moisture (1 km, Europe, daily)
rasters from Copernicus Dataspace (CDSE).

Library usage::

    from sar_validation.downloaders.sentinel1_soil_moisture_downloader import SoilMoistureDownloader
    dl = SoilMoistureDownloader(output_dir=Path("data/run1/S1_L3_SSM"))
    dl.download(min_lon=-10, max_lon=20, min_lat=40, max_lat=55,
                start="2026-01-01", end="2026-01-02")
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

from .base import (
    CopernicusODataClient,
    authenticate_cdse,
    normalize_datetime,
    split_antimeridian_bbox,
)

logger = logging.getLogger(__name__)

__all__ = ["SoilMoistureDownloader", "DATASET_IDENTIFIER", "PRODUCT_EXTENT"]

#: CLMS OData ``datasetIdentifier`` attribute value for the 1 km daily
#: Europe Surface Soil Moisture product. CONFIRMED: a real recipe run
#: (bbox lon[-10,30] lat[35,60], 2024-01-01..05) successfully queried and
#: downloaded real CEURO SSM1km products with this identifier.
DATASET_IDENTIFIER = "ssm_europe_1km_daily_v1"

#: Documented geographic extent of the CLMS SSM 1 km Europe (CEURO) product
#: (lon_min, lon_max, lat_min, lat_max). CONFIRMED from a real downloaded
#: product's embedded GDAL tags (geospatial_lon_min/max,
#: geospatial_lat_min/max): lon -11 to 50, lat 35 to 72.
PRODUCT_EXTENT = (-11.0, 50.0, 35.0, 72.0)

#: CDSE serves two container-format variants of this product per date/tile
#: (COG GeoTIFF and NetCDF, distinguished by this suffix in the product
#: Name); only the COG variant is queried/downloaded, since the converter
#: (``from_sar_l3_ssm_geotiff``) reads GeoTIFF via rioxarray, not NetCDF.
_COG_NAME_SUFFIX = "_cog"

#: Within an extracted COG product folder, the actual soil-moisture
#: GeoTIFF's filename contains this substring (as opposed to the sibling
#: ``-NOISE_`` uncertainty-layer GeoTIFF, which is not used) — confirmed
#: from a real downloaded product's file names, e.g.
#: ``c_gls_SSM1km-SSM_202401020000_CEURO_S1CSAR_V1.2.1.tiff``.
_SSM_FILENAME_MARKER = "-SSM_"


def _bbox_overlaps_extent(min_lon: float, max_lon: float, min_lat: float, max_lat: float) -> bool:
    """
    True if the request bbox overlaps ``PRODUCT_EXTENT`` at all.

    Handles antimeridian-crossing bboxes: if the request bbox crosses the
    antimeridian, check if either of the two split windows overlaps with
    the (non-crossing) PRODUCT_EXTENT.
    """
    ext_min_lon, ext_max_lon, ext_min_lat, ext_max_lat = PRODUCT_EXTENT
    lat_overlap = min_lat <= ext_max_lat and max_lat >= ext_min_lat

    if min_lon <= max_lon:
        # Normal (non-crossing) bbox
        lon_overlap = min_lon <= ext_max_lon and max_lon >= ext_min_lon
    else:
        # Antimeridian-crossing bbox: check if either window overlaps
        # Window 1: [min_lon, 180]
        window1_overlap = min_lon <= ext_max_lon and 180 >= ext_min_lon
        # Window 2: [-180, max_lon]
        window2_overlap = -180 <= ext_max_lon and max_lon >= ext_min_lon
        lon_overlap = window1_overlap or window2_overlap

    return lon_overlap and lat_overlap


class SoilMoistureDownloader:
    """
    Download Sentinel-1 CLMS Surface Soil Moisture (1 km, Europe, daily)
    GeoTIFF rasters from Copernicus Dataspace.

    Mirrors ``SARDownloader``'s shape (query/download, dry_run,
    force_download, top), but queries the CLMS collection via
    ``CopernicusODataClient.query_clms_products`` instead of the
    SENTINEL-1 OCN ``productType`` filter. Each product is downloaded as a
    ``.zip`` (like ``SARDownloader``'s SAFE products) and unzipped; unlike
    ``SARDownloader``, only the soil-moisture GeoTIFF inside is kept (the
    sibling noise-layer GeoTIFF is not returned).

    Parameters
    ----------
    output_dir : Path
        Directory to save downloaded GeoTIFFs.
    dry_run : bool
        If True, print what would be downloaded without actually downloading.
    username, password : str, optional
        CDSE credentials. If omitted, resolved from environment / credentials file.
    force_download : bool
        If True, re-download files even if already present in output_dir.
    """

    def __init__(
        self,
        output_dir: Path,
        dry_run: bool = False,
        username: Optional[str] = None,
        password: Optional[str] = None,
        force_download: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self._username = username
        self._password = password
        self._client: Optional[CopernicusODataClient] = None
        self.force_download = force_download

    def _get_client(self) -> CopernicusODataClient:
        if self._client is None:
            user, pwd = authenticate_cdse(self._username, self._password)
            self._client = CopernicusODataClient(user, pwd)
        return self._client

    def query(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start: str,
        end: str,
        top: int = 100,
    ) -> pd.DataFrame:
        """
        Query available CLMS Surface Soil Moisture products.

        Returns an empty DataFrame (with a logged warning, no network
        call, no client authentication) if the request bbox doesn't
        overlap the product's documented Europe extent. Results are
        filtered to the COG-format variant only (see ``_COG_NAME_SUFFIX``)
        — CDSE serves both a COG and a NetCDF product per date/tile, and
        only the COG one is downloaded.
        """
        if not _bbox_overlaps_extent(min_lon, max_lon, min_lat, max_lat):
            logger.warning(
                "Requested bbox (%.2f, %.2f, %.2f, %.2f) does not overlap "
                "the CLMS SSM 1km Europe product extent %s — skipping query.",
                min_lon, max_lon, min_lat, max_lat, PRODUCT_EXTENT,
            )
            return pd.DataFrame()

        start_norm = normalize_datetime(start) + ".000Z"
        end_norm   = normalize_datetime(end)   + ".000Z"

        client = self._get_client()
        frames = []
        for lo, hi in split_antimeridian_bbox(min_lon, max_lon):
            records = client.query_clms_products(
                dataset_identifier=DATASET_IDENTIFIER,
                start_date=start_norm,
                end_date=end_norm,
                min_lon=lo,
                max_lon=hi,
                min_lat=min_lat,
                max_lat=max_lat,
                top=top,
            )
            frames.append(pd.DataFrame(records))
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if df.empty:
            return df
        df = df.drop_duplicates(subset="Id", keep="first").reset_index(drop=True)
        df = df[df["Name"].str.contains(_COG_NAME_SUFFIX, case=False, na=False)]
        return df.reset_index(drop=True)

    def download(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start: str,
        end: str,
        top: int = 100,
    ) -> list[Path]:
        """
        Query and download CLMS Surface Soil Moisture GeoTIFFs.

        Each product is downloaded as a ``.zip`` (containing a ``-SSM_``
        soil-moisture GeoTIFF and a sibling ``-NOISE_`` uncertainty-layer
        GeoTIFF) and unzipped into ``output_dir``; only the ``-SSM_`` file
        path is returned.

        Returns
        -------
        list[Path]
            Paths to the extracted soil-moisture GeoTIFF files.
        """
        df = self.query(
            min_lon=min_lon, max_lon=max_lon,
            min_lat=min_lat, max_lat=max_lat,
            start=start, end=end, top=top,
        )
        self.found_count = len(df)

        if df.empty:
            print("No CLMS Surface Soil Moisture products found.")
            return []

        print(f"Found {len(df)} product(s).")
        for _, row in df.iterrows():
            print(f"  {row['Name']}  ({row['ContentLength_GB']:.2f} GB)  online={row['Online']}")

        if self.dry_run:
            print(f"[DRY RUN] Would download {len(df)} product(s) to {self.output_dir}")
            return []

        client = self._get_client()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []

        for i, (_, row) in enumerate(df.iterrows(), start=1):
            product_name = row["Name"]
            if not self.force_download and (self.output_dir / product_name).exists():
                print(f"[{i}/{len(df)}] Already downloaded: {product_name}")
                continue
            print(f"[{i}/{len(df)}] Downloading {product_name} …")
            try:
                path = client.download_product(row["Id"], self.output_dir, product_name)
                if path.suffix == ".zip":
                    with zipfile.ZipFile(path, "r") as zf:
                        zf.extractall(self.output_dir)
                    path.unlink()
                    extracted_dir = self.output_dir / product_name
                    ssm_tiffs = sorted(
                        p for p in extracted_dir.glob("*.tif*")
                        if _SSM_FILENAME_MARKER in p.name
                    )
                    if not ssm_tiffs:
                        print(f"  ERROR: no {_SSM_FILENAME_MARKER} file found in {extracted_dir}")
                        continue
                    downloaded.extend(ssm_tiffs)
                    print(f"  Unzipped to {extracted_dir}")
                else:
                    downloaded.append(path)
                    print(f"  Saved to {path}")
            except Exception as exc:
                print(f"  ERROR: {exc}")

        return downloaded
