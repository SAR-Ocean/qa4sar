"""
Shared base utilities for all downloaders.

Provides:
- CopernicusODataClient  — CDSE OData REST client (used for SAR downloads)
- authenticate_cdse      — Read CDSE credentials from file / env
- authenticate_eumdac    — Read EUMDAC credentials from file / env
- normalize_datetime     — ISO datetime normalisation helper
- format_dir_datetime    — Directory-name-safe datetime string
- build_output_dir       — Canonical output directory path
"""

from __future__ import annotations

import base64
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import requests

__all__ = [
    "CopernicusODataClient",
    "authenticate_cdse",
    "authenticate_eumdac",
    "normalize_datetime",
    "is_date_recent",
    "build_output_dir",
    "split_antimeridian_bbox",
    "copernicus_marine_download_kwargs",
]


# ---------------------------------------------------------------------------
# Copernicus Dataspace (CDSE) OData client
# ---------------------------------------------------------------------------

class CopernicusODataClient:
    """
    Thin REST client for the Copernicus Dataspace (CDSE) OData v1 catalogue.

    Used by the SAR downloader to query and download Sentinel-1 L2_OCN products.
    """

    CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    DOWNLOAD_URL  = "https://download.dataspace.copernicus.eu/odata/v1/Products"
    TOKEN_URL     = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
        "/protocol/openid-connect/token"
    )

    def __init__(self, username: str, password: str) -> None:
        self.username = username
        self.password = password
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self.session = requests.Session()
        self._authenticate()

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _authenticate(self) -> None:
        data = {
            "grant_type": "password",
            "username": self.username,
            "password": self.password,
            "client_id": "cdse-public",
        }
        resp = requests.post(self.TOKEN_URL, data=data, timeout=30)
        if resp.status_code == 401:
            raise PermissionError(
                "CDSE authentication failed — check username / password."
            )
        resp.raise_for_status()
        self._store_token(resp.json())

    def _refresh(self) -> None:
        if not self._refresh_token:
            self._authenticate()
            return
        data = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
            "client_id": "cdse-public",
        }
        resp = requests.post(self.TOKEN_URL, data=data, timeout=30)
        if resp.status_code == 200:
            self._store_token(resp.json())
        else:
            self._authenticate()

    def _store_token(self, token_data: dict) -> None:
        self._access_token = token_data["access_token"]
        self._refresh_token = token_data.get("refresh_token")
        expires_in = token_data.get("expires_in", 3600)
        self._token_expiry = datetime.now() + timedelta(seconds=expires_in)
        self.session.headers.update({"Authorization": f"Bearer {self._access_token}"})

    def _ensure_token(self) -> None:
        if self._token_expiry and datetime.now() > self._token_expiry - timedelta(minutes=1):
            self._refresh()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query_products(
        self,
        collection: str,
        product_type: str,
        start_date: str,
        end_date: str,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        top: int = 100,
    ) -> list[dict]:
        """
        Query products from the CDSE catalogue.

        Returns a list of product dicts with keys:
            Id, Name, ContentDate_Start, ContentDate_End, ContentLength_GB, Online
        """
        self._ensure_token()

        polygon = (
            f"POLYGON(({min_lon} {min_lat},{min_lon} {max_lat},"
            f"{max_lon} {max_lat},{max_lon} {min_lat},{min_lon} {min_lat}))"
        )
        filters = [
            f"Collection/Name eq '{collection}'",
            (
                "Attributes/OData.CSC.StringAttribute/any("
                f"att:att/Name eq 'productType' and "
                f"att/OData.CSC.StringAttribute/Value eq '{product_type}')"
            ),
            f"ContentDate/Start gt {start_date}",
            f"ContentDate/Start lt {end_date}",
            f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}')",
        ]
        params = {
            "$filter": " and ".join(filters),
            "$orderby": "ContentDate/Start desc",
            "$top": str(top),   # query params serialise to strings anyway
            "$expand": "Attributes",
        }
        resp = self.session.get(self.CATALOGUE_URL, params=params, timeout=60)
        resp.raise_for_status()

        records = []
        for prod in resp.json().get("value", []):
            records.append(
                {
                    "Id": prod.get("Id"),
                    "Name": prod.get("Name"),
                    "ContentDate_Start": prod.get("ContentDate", {}).get("Start"),
                    "ContentDate_End": prod.get("ContentDate", {}).get("End"),
                    "ContentLength_GB": prod.get("ContentLength", 0) / (1024 ** 3),
                    "Online": prod.get("Online"),
                }
            )
        return records

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_product(
        self,
        product_id: str,
        output_dir: Path,
        product_name: str = "",
        chunk_size: int = 8192,
    ) -> Path:
        """Download a single product by ID. Returns the local file path."""
        self._ensure_token()
        url = f"{self.DOWNLOAD_URL}({product_id})/$value"
        resp = self.session.get(url, stream=True, timeout=60)
        resp.raise_for_status()

        # Determine filename
        cd = resp.headers.get("content-disposition", "")
        if "filename=" in cd:
            filename = cd.split("filename=")[-1].strip('"')
        else:
            filename = f"{product_id}.zip"

        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / filename
        with open(filepath, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
        return filepath


# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------

def authenticate_cdse(
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Resolve CDSE (Copernicus Dataspace) credentials.

    Priority order:
      1. Explicit arguments
      2. Environment variables  COPERNICUS_USERNAME / COPERNICUS_PASSWORD
      3. ~/.config/cdse/credentials  (base64-encoded or plain key=value)

    Raises RuntimeError if no credentials are found.
    """
    if username and password:
        return username, password

    username = username or os.environ.get("COPERNICUS_USERNAME")
    password = password or os.environ.get("COPERNICUS_PASSWORD")
    if username and password:
        return username, password

    cred_file = Path.home() / ".config" / "cdse" / "credentials"
    if cred_file.exists():
        raw = cred_file.read_text().strip()
        try:
            content = base64.b64decode(raw).decode("utf-8")
        except Exception:
            content = raw
        for line in content.splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#") and not line.startswith("["):
                key, _, val = line.partition("=")
                key = key.strip().lower()
                val = val.strip()
                if key == "username":
                    username = val
                elif key == "password":
                    password = val
        if username and password:
            return username, password

    raise RuntimeError(
        "CDSE credentials not found.\n"
        "Options:\n"
        "  1. Store in ~/.config/cdse/credentials (key=value, optionally base64-encoded)\n"
        "  2. Set COPERNICUS_USERNAME / COPERNICUS_PASSWORD environment variables\n"
        "  3. Pass --username / --password on the command line\n"
        "Register at: https://dataspace.copernicus.eu"
    )


def authenticate_eumdac(
    username: Optional[str] = None,
    password: Optional[str] = None,
):
    """
    Resolve EUMDAC credentials and return an eumdac.AccessToken.

    Priority order:
      1. Explicit arguments
      2. Environment variables  EUMDAC_USERNAME / EUMDAC_PASSWORD
      3. ~/.eumdac/credentials  (comma-separated: username,password)

    Requires the ``eumdac`` package to be installed.
    Raises RuntimeError if no credentials are found.
    """
    try:
        import eumdac  # noqa: F401 — checked here so error is clear
    except ImportError as exc:
        raise ImportError(
            "eumdac is required for scatterometer downloads.\n"
            "Install it with:  pip install eumdac"
        ) from exc

    if not username:
        username = os.environ.get("EUMDAC_USERNAME")
    if not password:
        password = os.environ.get("EUMDAC_PASSWORD")

    cred_file = Path.home() / ".eumdac" / "credentials"
    if (not username or not password) and cred_file.exists():
        content = cred_file.read_text().strip()
        parts = content.split(",")
        if len(parts) >= 2:
            username = username or parts[0].strip()
            password = password or parts[1].strip()

    if not username or not password:
        raise RuntimeError(
            "EUMDAC credentials not found.\n"
            "Options:\n"
            "  1. Store in ~/.eumdac/credentials as 'username,password'\n"
            "  2. Set EUMDAC_USERNAME / EUMDAC_PASSWORD environment variables\n"
            "Register at: https://eoportal.eumetsat.int"
        )

    import eumdac

    token = eumdac.AccessToken((username, password))
    return token


# ---------------------------------------------------------------------------
# Datetime / directory helpers
# ---------------------------------------------------------------------------

def normalize_datetime(dt_str: str) -> str:
    """
    Normalize a datetime string to ISO format (YYYY-MM-DDTHH:MM:SS).
    
    Accepts:
    - Date only: "2026-01-01" → "2026-01-01T00:00:00"
    - With T separator: "2026-01-01T12:34:56" → unchanged
    - With space separator: "2026-01-01 12:34:56" → "2026-01-01T12:34:56"
    - With HHMMSS (no colons): "2026-01-01 120000" → "2026-01-01T12:00:00"
    - With Z suffix: "2026-01-01T12:34:56Z" → "2026-01-01T12:34:56"
    """
    dt_str = dt_str.strip().rstrip("Z").split(".")[0]  # Remove Z and milliseconds
    
    # Replace space with T if present
    if " " in dt_str:
        dt_str = dt_str.replace(" ", "T")
    
    # Split date and time parts
    if "T" in dt_str:
        date_part, time_part = dt_str.split("T", 1)
        # Check if time_part is 6 digits (HHMMSS format without colons)
        if len(time_part) == 6 and time_part.isdigit():
            # Convert HHMMSS to HH:MM:SS
            time_part = f"{time_part[0:2]}:{time_part[2:4]}:{time_part[4:6]}"
        dt_str = f"{date_part}T{time_part}"
    elif len(dt_str) == 10:
        # Date only
        dt_str += "T00:00:00"
    
    return dt_str


def is_date_recent(dt_str: str, threshold_days: int = 30) -> bool:
    """
    Check if a datetime string is within the recent threshold.
    
    Parameters
    ----------
    dt_str : str
        DateTime string (will be normalized to ISO format).
    threshold_days : int
        Number of days from today to consider "recent" (default: 30).
    
    Returns
    -------
    bool
        True if the date is within threshold_days from today, False otherwise.
    """
    dt_norm = normalize_datetime(dt_str)
    # Parse ISO format: YYYY-MM-DDTHH:MM:SS
    try:
        dt_obj = datetime.fromisoformat(dt_norm)
        days_diff = (datetime.now() - dt_obj).days
        return 0 <= days_diff <= threshold_days
    except ValueError:
        return False


def build_output_dir(
    start_dt: str,
    end_dt: str,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    base: str | Path = "data",
) -> Path:
    """
    Build a canonical output directory path.

    Format:  data/YYYY-MM-DD-HHMMSS-YYYY-MM-DD-HHMMSS_minlon_maxlon_minlat_maxlat/
    """

    def _fmt(dt: str) -> str:
        dt = normalize_datetime(dt)
        date_part, _, time_part = dt.partition("T")
        return f"{date_part}-{time_part.replace(':', '')[:6]}"

    dir_name = (
        f"{_fmt(start_dt)}-{_fmt(end_dt)}_"
        f"{min_lon:.2f}_{max_lon:.2f}_{min_lat:.2f}_{max_lat:.2f}"
    )
    return Path(base) / dir_name


def split_antimeridian_bbox(min_lon: float, max_lon: float) -> list[tuple[float, float]]:
    """
    Split a longitude range into 1 or 2 non-crossing (lon_min, lon_max) windows.

    A recipe's ``GeographicBounds.min_lon > max_lon`` means the bbox wraps
    through the antimeridian (180 degrees) rather than being invalid, e.g.
    ``min_lon=135, max_lon=-120`` covers the Pacific from 135E to 120W.
    Returns the box unchanged (as a single window) when ``min_lon <=
    max_lon``; otherwise returns the two windows ``[min_lon, 180]`` and
    ``[-180, max_lon]`` that together cover the same region without either
    window itself crossing the antimeridian.
    """
    if min_lon <= max_lon:
        return [(min_lon, max_lon)]
    return [(min_lon, 180.0), (-180.0, max_lon)]


def copernicus_marine_download_kwargs(force_download: bool) -> dict:
    """
    Return the ``skip_existing``/``overwrite`` kwargs for a
    ``copernicusmarine.subset()``/``.get()`` call, matching this toolbox's
    ``--force-download`` semantics.

    ``copernicusmarine`` has no ``force_download`` parameter (the two real,
    mutually-exclusive options are ``overwrite`` and ``skip_existing``) —
    this is the single place that translates the toolbox's boolean flag into
    the real API, so no downloader has to reason about the mapping itself.
    """
    return {"skip_existing": not force_download, "overwrite": force_download}
