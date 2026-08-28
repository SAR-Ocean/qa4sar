"""
Shared base utilities for all downloaders.

Provides:
- CopernicusODataClient  — CDSE OData REST client (used for SAR downloads)
- authenticate_cdse      — Read CDSE credentials from file / env
- authenticate_eumdac    — Read EUMDAC credentials from env / OS keyring
- authenticate_osi_saf_ftp — Read OSI-SAF wind FTP credentials from env / OS keyring
- authenticate_hsaf_ftp  — Read H-SAF FTP credentials from env / OS keyring
- authenticate_space_track — Read Space-Track.org credentials from env / OS keyring
- authenticate_gportal   — Read JAXA G-Portal credentials from env / OS keyring / prompt
- authenticate_smos_ftp  — Read SMOS Online Dissemination FTPS credentials from env / OS keyring
- authenticate_earthdata — Resolve NASA Earthdata Login credentials from env /
  OS keyring / ~/.netrc and log in via earthaccess
- set_credential         — Store a username/password pair in the OS keyring
  (used by ``sar-validate --set-credential``)
- normalize_datetime     — ISO datetime normalisation helper
- is_date_recent         — True if a date/datetime string falls within a
  recent threshold
- build_output_dir       — Canonical output directory path
- split_antimeridian_bbox — Split an antimeridian-crossing bbox into 1-2
  non-crossing longitude windows
- months_touched         — Every (year, month) pair an inclusive date
  range touches
- copernicus_marine_download_kwargs — skip_existing/overwrite kwargs for a
  copernicusmarine call, matching this toolbox's --force-download semantics
- prefer_ipv4_dns        — Context manager that reorders socket.getaddrinfo()
  results IPv4-first, to avoid wasting time on hosts with black-holed IPv6
"""


from __future__ import annotations

import base64
import json
import logging
import os
import re
import socket
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple

import keyring
import keyring.errors
import requests

logger = logging.getLogger(__name__)

__all__ = [
    "CopernicusODataClient",
    "authenticate_cdse",
    "authenticate_eumdac",
    "authenticate_osi_saf_ftp",
    "authenticate_hsaf_ftp",
    "authenticate_space_track",
    "authenticate_gportal",
    "authenticate_smos_ftp",
    "authenticate_earthdata",
    "set_credential",
    "normalize_datetime",
    "is_date_recent",
    "build_output_dir",
    "split_antimeridian_bbox",
    "months_touched",
    "copernicus_marine_download_kwargs",
    "prefer_ipv4_dns",
]

# ---------------------------------------------------------------------------
# OS keyring service names — one per migrated credential set. Each service
# stores two entries: "username" and "password" (via keyring.set_password /
# keyring.get_password).
# ---------------------------------------------------------------------------
_KEYRING_SERVICES = {
    "eumdac": "sar-validation-eumdac",
    "osi_saf": "sar-validation-osi-saf",
    "gportal": "sar-validation-gportal",
    "smos": "sar-validation-smos",
    "earthdata": "sar-validation-earthdata",
    "hsaf": "sar-validation-hsaf",
    "space_track": "sar-validation-space-track",
}


def _keyring_get(service: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Read (username, password) from the OS keyring for *service*.

    Treats "no OS keyring backend available" (e.g. headless CI runners,
    which raise keyring.errors.NoKeyringError or similar) identically to
    "nothing stored" -- callers fall through to the next credential source
    rather than crashing.
    """
    try:
        username = keyring.get_password(service, "username")
        password = keyring.get_password(service, "password")
    except keyring.errors.KeyringError:
        return None, None
    return username, password


def _keyring_set_quiet(service: str, username: str, password: str) -> None:
    """
    Best-effort store of (username, password) into the OS keyring.

    Used for the automatic legacy-file migration path, where a missing
    keyring backend must not turn a successful credential read into a
    crash -- it just means the migration silently doesn't happen this run.
    """
    try:
        keyring.set_password(service, "username", username)
        keyring.set_password(service, "password", password)
    except keyring.errors.KeyringError:
        pass


def _resolve_from_keyring_or_legacy_file(
    name: str,
    cred_file: Path,
    parse_legacy_file: Callable[[str], Tuple[Optional[str], Optional[str]]],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve (username, password) from the OS keyring, migrating from a
    legacy plaintext credentials file into the keyring on first use.

    Priority: keyring first; if the keyring has nothing stored for
    *name* AND the legacy *cred_file* still exists on disk, read it,
    write its contents into the keyring (one-time migration), print a
    console notice, and return those values. The legacy file itself is
    never deleted -- that's left to the user.
    """
    service = _KEYRING_SERVICES[name]
    username, password = _keyring_get(service)
    if username and password:
        return username, password

    if cred_file.exists():
        file_username, file_password = parse_legacy_file(cred_file.read_text())
        if file_username and file_password:
            _keyring_set_quiet(service, file_username, file_password)
            if cred_file.name == ".netrc":
                # ~/.netrc is a shared OS-standard file used by many tools
                # (curl, wget, earthaccess itself, other NASA clients) and
                # may hold other machines' unrelated credentials -- unlike
                # the bespoke single-purpose legacy files below, it is not
                # safe to suggest deleting it.
                logger.warning(
                    "Migrated %s credentials from %s to the OS keyring "
                    "(service=%r). %s is left in place since it's a "
                    "shared file other tools may still use.",
                    name, cred_file, service, cred_file,
                )
            else:
                logger.warning(
                    "Migrated %s credentials from %s to the OS keyring "
                    "(service=%r). You can now safely delete %s.",
                    name, cred_file, service, cred_file,
                )
            return file_username, file_password

    return username, password


def _parse_comma_separated_legacy_file(content: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse the EUMDAC legacy file format: 'username,password'.
    """
    parts = content.strip().split(",")
    if len(parts) >= 2:
        return parts[0].strip(), parts[1].strip()
    return None, None


def _parse_json_legacy_file(content: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse the OSI-SAF/G-Portal/SMOS legacy file format:
    '{"username": ..., "password": ...}'.
    """
    creds = json.loads(content)
    return creds.get("username"), creds.get("password")


def _parse_netrc_legacy_file(content: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse a ~/.netrc file's urs.earthdata.nasa.gov machine entry
    (login/password), using the stdlib netrc module.
    """
    import netrc as netrc_module
    import os
    import tempfile

    # netrc.netrc() only accepts a file path, not a string -- write the
    # content to a throwaway temp file so the stdlib parser can be reused
    # instead of hand-rolling a parser for netrc's format.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".netrc", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        parsed = netrc_module.netrc(tmp_path)
    except (netrc_module.NetrcParseError, OSError):
        return None, None
    finally:
        os.unlink(tmp_path)

    entry = parsed.authenticators("urs.earthdata.nasa.gov")
    if entry is None:
        return None, None
    login, _account, password = entry
    return login, password


def set_credential(name: str, username: str, password: str) -> None:
    """
    Store *username*/*password* in the OS keyring for a credential set.

    Used by ``sar-validate --set-credential {eumdac,osi_saf,gportal,smos,earthdata}``.

    Parameters
    ----------
    name : str
        One of "eumdac", "osi_saf", "gportal", "smos", "earthdata".
    username, password : str
        Values to store.

    Raises
    ------
    ValueError
        If *name* is not one of the recognised credential sets.
    keyring.errors.KeyringError
        If no OS keyring backend is available. This function backs 
        an explicit user action, so the caller (the CLI) needs a real 
        error to report rather than a silent no-op.
    """
    try:
        service = _KEYRING_SERVICES[name]
    except KeyError:
        raise ValueError(
            f"Unknown credential set {name!r}. Choose one of: "
            f"{', '.join(sorted(_KEYRING_SERVICES))}"
        ) from None
    keyring.set_password(service, "username", username)
    keyring.set_password(service, "password", password)


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
            Id, Name, ContentDate_Start, ContentDate_End, ContentLength_GB,
            Online, GeoFootprint (raw CDSE GeoJSON-style geometry, or None)
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
            # ge/le, not gt/lt: a daily product's ContentDate/Start commonly
            # lands exactly on a requested window boundary (e.g. a recipe's
            # start date normalizes to an exact midnight timestamp) -- a
            # strict gt/lt silently drops that boundary-matching product.
            f"ContentDate/Start ge {start_date}",
            f"ContentDate/Start le {end_date}",
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
                    "GeoFootprint": prod.get("GeoFootprint"),
                }
            )
        return records

    def query_clms_products(
        self,
        dataset_identifier: str,
        start_date: str,
        end_date: str,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        top: int = 100,
    ) -> list[dict]:
        """
        Query CLMS (Copernicus Land Monitoring Service) products from the
        CDSE catalogue, e.g. Surface Soil Moisture rasters.

        Same polygon/pagination machinery as :meth:`query_products`, but
        filters on the ``datasetIdentifier`` attribute (CLMS's own
        product-family key) instead of ``productType`` (used by
        SENTINEL-1 OCN).

        Returns a list of product dicts with keys:
            Id, Name, ContentDate_Start, ContentDate_End, ContentLength_GB, Online
        """
        self._ensure_token()

        polygon = (
            f"POLYGON(({min_lon} {min_lat},{min_lon} {max_lat},"
            f"{max_lon} {max_lat},{max_lon} {min_lat},{min_lon} {min_lat}))"
        )
        filters = [
            "Collection/Name eq 'CLMS'",
            (
                "Attributes/OData.CSC.StringAttribute/any("
                f"att:att/Name eq 'datasetIdentifier' and "
                f"att/OData.CSC.StringAttribute/Value eq '{dataset_identifier}')"
            ),
            # ge/le, not gt/lt -- see the identical comment in query_products.
            f"ContentDate/Start ge {start_date}",
            f"ContentDate/Start le {end_date}",
            f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}')",
        ]
        params = {
            "$filter": " and ".join(filters),
            "$orderby": "ContentDate/Start desc",
            "$top": str(top),
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
      3. OS keyring (service "sar-validation-eumdac"; see set_credential /
         ``sar-validate --set-credential eumdac``). If nothing is stored
         there yet but the legacy ~/.eumdac/credentials file
         (comma-separated: username,password) still exists, it is read
         once, migrated into the keyring, and a console notice is printed.

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

    if not username or not password:
        cred_file = Path.home() / ".eumdac" / "credentials"
        kr_username, kr_password = _resolve_from_keyring_or_legacy_file(
            "eumdac", cred_file, _parse_comma_separated_legacy_file
        )
        username = username or kr_username
        password = password or kr_password

    if not username or not password:
        raise RuntimeError(
            "EUMDAC credentials not found.\n"
            "Options:\n"
            "  1. Run `sar-validate --set-credential eumdac` to store credentials "
            "in your OS keyring\n"
            "  2. Set EUMDAC_USERNAME / EUMDAC_PASSWORD environment variables\n"
            "Register at: https://eoportal.eumetsat.int"
        )

    import eumdac

    token = eumdac.AccessToken((username, password))
    return token


def authenticate_earthdata(
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> None:
    """
    Resolve NASA Earthdata Login credentials and log in via ``earthaccess``.

    Priority order:
      1. Explicit arguments
      2. Environment variables  EARTHDATA_USERNAME / EARTHDATA_PASSWORD
      3. OS keyring (service "sar-validation-earthdata"; see set_credential /
         ``sar-validate --set-credential earthdata``). If nothing is stored
         there yet but ~/.netrc has a urs.earthdata.nasa.gov entry, it is
         read once, migrated into the keyring, and a console notice is
         printed.

    Unlike the other ``authenticate_*`` helpers, this performs the actual
    login. ``earthaccess.login()`` is the toolbox's whole NASA Earthdata 
    client. If nothing resolves from any of the above, falls back to a bare
    ``earthaccess.login()``, preserving its own netrc/interactive-prompt 
    resolution.
    """
    import os

    import earthaccess

    if not username:
        username = os.environ.get("EARTHDATA_USERNAME")
    if not password:
        password = os.environ.get("EARTHDATA_PASSWORD")

    if not username or not password:
        cred_file = Path.home() / ".netrc"
        kr_username, kr_password = _resolve_from_keyring_or_legacy_file(
            "earthdata", cred_file, _parse_netrc_legacy_file,
        )
        username = username or kr_username
        password = password or kr_password

    if username and password:
        os.environ["EARTHDATA_USERNAME"] = username
        os.environ["EARTHDATA_PASSWORD"] = password
        earthaccess.login(strategy="environment")
    else:
        earthaccess.login()


def authenticate_osi_saf_ftp(
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Resolve OSI-SAF wind FTP (ftppro.knmi.nl) credentials.

    Priority order:
      1. Explicit arguments
      2. Environment variables  OSI_SAF_FTP_USERNAME / OSI_SAF_FTP_PASSWORD
      3. OS keyring (service "sar-validation-osi-saf"; see set_credential /
         ``sar-validate --set-credential osi_saf``). If nothing is stored
         there yet but the legacy ~/.eumetsat_osi_saf_wind_credentials
         file (JSON: {"username": ..., "password": ...}) still exists, it
         is read once, migrated into the keyring, and a console notice is
         printed.

    Raises RuntimeError if no credentials are found.
    """
    if username and password:
        return username, password

    username = username or os.environ.get("OSI_SAF_FTP_USERNAME")
    password = password or os.environ.get("OSI_SAF_FTP_PASSWORD")
    if username and password:
        return username, password

    cred_file = Path.home() / ".eumetsat_osi_saf_wind_credentials"
    kr_username, kr_password = _resolve_from_keyring_or_legacy_file(
        "osi_saf", cred_file, _parse_json_legacy_file
    )
    username = username or kr_username
    password = password or kr_password
    if username and password:
        return username, password

    raise RuntimeError(
        "OSI-SAF FTP credentials not found.\n"
        "Options:\n"
        "  1. Run `sar-validate --set-credential osi_saf` to store credentials "
        "in your OS keyring\n"
        "  2. Set OSI_SAF_FTP_USERNAME / OSI_SAF_FTP_PASSWORD environment variables\n"
        "  3. Pass --username / --password on the command line\n"
        "Register at: https://osi-saf.eumetsat.int/register"
    )


def authenticate_hsaf_ftp(
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Resolve H-SAF FTP (ftphsaf.meteoam.it) credentials.

    Priority order:
      1. Explicit arguments
      2. Environment variables  HSAF_FTP_USERNAME / HSAF_FTP_PASSWORD
      3. OS keyring (service "sar-validation-hsaf"; see set_credential /
         ``sar-validate --set-credential hsaf``).

    Raises RuntimeError if no credentials are found.
    """
    if username and password:
        return username, password

    username = username or os.environ.get("HSAF_FTP_USERNAME")
    password = password or os.environ.get("HSAF_FTP_PASSWORD")
    if username and password:
        return username, password

    cred_file = Path.home() / ".hsaf_ftp_credentials"
    kr_username, kr_password = _resolve_from_keyring_or_legacy_file(
        "hsaf", cred_file, _parse_json_legacy_file
    )
    username = username or kr_username
    password = password or kr_password
    if username and password:
        return username, password

    raise RuntimeError(
        "H-SAF FTP credentials not found.\n"
        "Options:\n"
        "  1. Run `sar-validate --set-credential hsaf` to store credentials "
        "in your OS keyring\n"
        "  2. Set HSAF_FTP_USERNAME / HSAF_FTP_PASSWORD environment variables\n"
        "  3. Pass --username / --password on the command line\n"
        "Register at: https://hsaf.meteoam.it/User/Register"
    )


def authenticate_space_track(
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Resolve Space-Track.org credentials (used by orbit_coverage.get_tle
    to look up historical TLEs for the orbit-based geographic pre-filter).

    Priority order:
      1. Explicit arguments
      2. Environment variables  SPACE_TRACK_USERNAME / SPACE_TRACK_PASSWORD
      3. OS keyring (service "sar-validation-space-track"; see
         set_credential / ``sar-validate --set-credential space_track``).

    Raises RuntimeError if no credentials are found.
    """
    if username and password:
        return username, password

    username = username or os.environ.get("SPACE_TRACK_USERNAME")
    password = password or os.environ.get("SPACE_TRACK_PASSWORD")
    if username and password:
        return username, password

    cred_file = Path.home() / ".space_track_credentials"
    kr_username, kr_password = _resolve_from_keyring_or_legacy_file(
        "space_track", cred_file, _parse_json_legacy_file
    )
    username = username or kr_username
    password = password or kr_password
    if username and password:
        return username, password

    raise RuntimeError(
        "Space-Track credentials not found.\n"
        "Options:\n"
        "  1. Run `sar-validate --set-credential space_track` to store credentials "
        "in your OS keyring\n"
        "  2. Set SPACE_TRACK_USERNAME / SPACE_TRACK_PASSWORD environment variables\n"
        "  3. Pass --username / --password on the command line\n"
        "Register at: https://www.space-track.org/auth/createAccount"
    )


def authenticate_gportal(
    username: Optional[str] = None,
    password: Optional[str] = None,
    allow_prompt: bool = True,
) -> Tuple[str, str]:
    """
    Resolve JAXA G-Portal SFTP (ftp.gportal.jaxa.jp:2051) credentials.

    Priority order:
      1. Explicit arguments
      2. Environment variables  GPORTAL_USERNAME / GPORTAL_PASSWORD
      3. OS keyring (service "sar-validation-gportal"; see set_credential /
         ``sar-validate --set-credential gportal``). If nothing is stored
         there yet but the legacy ~/.jaxa_gportal_credentials file (JSON:
         {"username": ..., "password": ...}) still exists, it is read
         once, migrated into the keyring, and a console notice is printed.
      4. Interactive terminal prompt (username via input(), password via
         getpass.getpass()) -- deliberately NOT persisted to disk, unlike
         every step above. This is the only authenticate_* helper in this
         module that prompts instead of raising when nothing resolves;
         that is intentional and specific to G-Portal, not a convention
         to retrofit onto the others. Only reached when *allow_prompt* is
         True (the default, for direct/CLI use of the downloader).

    G-Portal has no SSH-key registration option for this account, so 
    account+password is the only authentication method available here.

    Parameters
    ----------
    allow_prompt : bool
        When False, step 4 is skipped -- raises RuntimeError instead of
        prompting if nothing resolved from steps 1-3.
    """
    if username and password:
        return username, password

    username = username or os.environ.get("GPORTAL_USERNAME")
    password = password or os.environ.get("GPORTAL_PASSWORD")
    if username and password:
        return username, password

    cred_file = Path.home() / ".jaxa_gportal_credentials"
    kr_username, kr_password = _resolve_from_keyring_or_legacy_file(
        "gportal", cred_file, _parse_json_legacy_file
    )
    username = username or kr_username
    password = password or kr_password
    if username and password:
        return username, password

    if not allow_prompt:
        raise RuntimeError(
            "G-Portal credentials not found.\n"
            "Options:\n"
            "  1. Run `sar-validate --set-credential gportal` to store credentials "
            "in your OS keyring\n"
            "  2. Set GPORTAL_USERNAME / GPORTAL_PASSWORD environment variables\n"
            "  3. Pass --username / --password on the command line\n"
            "Register at: https://gportal.jaxa.jp"
        )

    import getpass

    if not username:
        username = input("G-Portal username: ")
    if not password:
        password = getpass.getpass("G-Portal password: ")
    return username, password


def authenticate_smos_ftp(
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Resolve SMOS Online Dissemination FTPS (smos-diss.eo.esa.int) credentials.

    Priority order:
      1. Explicit arguments
      2. Environment variables  SMOS_FTP_USERNAME / SMOS_FTP_PASSWORD
      3. OS keyring (service "sar-validation-smos"; see set_credential /
         ``sar-validate --set-credential smos``). If nothing is stored
         there yet but the legacy ~/.esa_smos_credentials file (JSON:
         {"username": ..., "password": ...}) still exists, it is read
         once, migrated into the keyring, and a console notice is printed.

    Raises RuntimeError if no credentials are found.
    """
    if username and password:
        return username, password

    username = username or os.environ.get("SMOS_FTP_USERNAME")
    password = password or os.environ.get("SMOS_FTP_PASSWORD")
    if username and password:
        return username, password

    cred_file = Path.home() / ".esa_smos_credentials"
    kr_username, kr_password = _resolve_from_keyring_or_legacy_file(
        "smos", cred_file, _parse_json_legacy_file
    )
    username = username or kr_username
    password = password or kr_password
    if username and password:
        return username, password

    raise RuntimeError(
        "SMOS FTPS credentials not found.\n"
        "Options:\n"
        "  1. Run `sar-validate --set-credential smos` to store credentials "
        "in your OS keyring\n"
        "  2. Set SMOS_FTP_USERNAME / SMOS_FTP_PASSWORD environment variables\n"
        "  3. Pass --username / --password on the command line\n"
        "Register at: https://smos-diss.eo.esa.int"
    )


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
    - With UTC offset: "2026-01-01T12:34:56+00:00" → "2026-01-01T12:34:56"
    """
    dt_str = dt_str.strip()
    # Strip a trailing numeric UTC offset (e.g. "+00:00", "-05:00") the
    # same way a literal "Z" suffix is already stripped below -- every
    # offset produced anywhere in this codebase is UTC (e.g. a tz-aware
    # SarFootprint.sensing_start/sensing_end's own .isoformat() renders
    # as "...+00:00", never "Z"), so dropping it is equivalent to an
    # already-UTC value losing its (redundant) tzinfo. Gated on length so
    # a bare "YYYY-MM-DD" date's own hyphens are never mistaken for a
    # sign -- an offset can only appear after a time component.
    if len(dt_str) > 10:
        dt_str = re.sub(r"[+-]\d{2}:?\d{2}$", "", dt_str)
    dt_str = dt_str.rstrip("Z").split(".")[0]  # Remove Z and milliseconds

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


def months_touched(start: datetime, end: datetime) -> list[tuple[int, int]]:
    """
    Return every (year, month) pair touched by the inclusive [start, end]
    range, e.g. (2024, 11, 15) to (2025, 2, 3) -> [(2024, 11), (2024, 12),
    (2025, 1), (2025, 2)]. Used by THREDDS-archive downloaders (NOAA
    HF-radar, RADARSAT-2 wind) to know which monthly catalog.xml files to
    fetch.
    """
    months: list[tuple[int, int]] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

@contextmanager
def prefer_ipv4_dns() -> Iterator[None]:
    """Temporarily reorder ``socket.getaddrinfo()`` results so IPv4
    addresses (``socket.AF_INET``) come before IPv6 addresses
    (``socket.AF_INET6``), for the duration of the ``with`` block.

    Some hosts have broken IPv6 reachability that a plain DNS/connect timeout
    does not protect against. ``socket.create_connection`` -- used internally 
    by ``urllib.request.urlopen``/``http.client`` -- iterates
    ``getaddrinfo(AF_UNSPEC)`` results in the order returned, which is
    IPv6-before-IPv4 by default on many systems. Every request to such a
    host therefore wastes up to ``len(ipv6_addresses) * timeout`` seconds
    cycling through dead candidates before ever reaching a working IPv4
    one. This function does not filter out IPv6, but tries IPv4 first.
    Combined with a shorter per-address timeout, this function makes the 
    common case (IPv4) fast instead of slow.

    Uses a stable sort (``key=lambda r: r[0] != socket.AF_INET``) so each
    family's original relative order is preserved -- only the IPv4-vs-IPv6
    grouping changes.

    Restores the original ``socket.getaddrinfo`` afterward, even if an
    exception is raised inside the ``with`` block.

    This codebase's downloaders run sequentially, single-threaded, so a
    temporary global monkeypatch of ``socket.getaddrinfo`` is safe here.
    """
    original_getaddrinfo = socket.getaddrinfo

    def _ipv4_first_getaddrinfo(*args, **kwargs):
        results = original_getaddrinfo(*args, **kwargs)
        return sorted(results, key=lambda r: r[0] != socket.AF_INET)

    socket.getaddrinfo = _ipv4_first_getaddrinfo
    try:
        yield
    finally:
        socket.getaddrinfo = original_getaddrinfo


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
