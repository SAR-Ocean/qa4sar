"""
Download SMOS soil-moisture products from ESA's Online Dissemination
service (smos-diss.eo.esa.int) via the OADS web portal (HTTPS +
SAML2/WSO2 SSO login), not FTPS — FTPS was dropped after confirming the
FTP server hangs indefinitely from this toolbox's network path.

Browses the NRT_Open product tree by date via OADS's tree-browse form.
Unlike CATDS/CDN mirrors, this is ESA's own dissemination point serving
the operational L2 SM product (SM_OPER_MIR_SMUDP2), volumetric units
(m3/m3).

No server-side bbox filtering — files are downloaded per day and cropped
to the recipe domain downstream, in ``convert_downloaded_data``.

Library usage::

    from sar_validation.downloaders.smos_downloader import SMOSDownloader
    dl = SMOSDownloader(output_dir=Path("data/run1/smos_ssm"))
    dl.download(min_lon=-10, max_lon=10, min_lat=40, max_lat=55,
                start="2026-01-01", end="2026-01-02")

CLI usage::

    python -m sar_validation.downloaders.smos_downloader \\
        --min-lon -10 --max-lon 10 --min-lat 40 --max-lat 55 \\
        --start 2026-01-01 --end 2026-01-02
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests

from .base import authenticate_smos_ftp, build_output_dir, normalize_datetime

__all__ = ["SMOSDownloader"]

OADS_BASE_URL = "https://smos-diss.eo.esa.int/oads/access"
OADS_TREE_URL = f"{OADS_BASE_URL}/collection/NRT_Open/tree"
OADS_LOGIN_URL = f"{OADS_BASE_URL}/login"
#: OADS "Type" tree-form value for the NRT L2 Soil Moisture product.
OADS_PRODUCT_TYPE = "MIR_SMNRT2"

#: Standard ESA SMOS filename convention:
#: SM_OPER_MIR_SMUDP2_<start>_<stop>_<orbit>_<counter>_<version>.zip
#: (operational) or the NRT variant SM_OPER_MIR_SMNRT2_... -- both carry
#: real start/stop sensing timestamps in YYYYMMDDTHHMMSS format. If a
#: real OADS filename doesn't match this (unconfirmed against a live
#: listing as of this writing), _filter_by_orbit_overlap falls back to
#: a whole-day window instead -- see that method.
_SENSING_WINDOW_RE = re.compile(r"MIR_SM(?:UDP2|NRT2)_(\d{8}T\d{6})_(\d{8}T\d{6})_")


def _parse_sensing_window(filename: str) -> "Optional[tuple[datetime, datetime]]":
    """Extract the embedded (start, stop) sensing timestamps from a
    standard SMOS filename, or None if the filename doesn't match the
    expected convention."""
    m = _SENSING_WINDOW_RE.search(filename)
    if not m:
        return None
    try:
        start = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        stop = datetime.strptime(m.group(2), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        return start, stop
    except ValueError:
        return None


class SMOSDownloader:
    """
    Download SMOS soil-moisture products from ESA's Online Dissemination
    service via the OADS web portal.

    Parameters
    ----------
    output_dir : Path
        Directory to save downloaded files.
    dry_run : bool
        If True, print what would be downloaded without actually downloading.
    username, password : str, optional
        OADS/ESA account credentials. If omitted, resolved from environment /
        credentials file.
    """

    def __init__(
        self,
        output_dir: Path,
        dry_run: bool = False,
        username: Optional[str] = None,
        password: Optional[str] = None,
        orbit_prefilter: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self._username = username
        self._password = password
        self.orbit_prefilter = orbit_prefilter

    def _list_products_for_day(self, session: "requests.Session", day) -> list:
        """
        List SMOS NRT L2 SM products available for *day* via the OADS
        tree-browse form (``p0``=type, ``p1``=year, ``p2``=month,
        ``p3``=day), parsed from the returned HTML.

        Returns
        -------
        list[dict]
            One dict per product: ``{"filename": str, "download_href": str}``.
            ``download_href`` is the (login-gated) relative URL found next
            to "Download Product" — resolved to an absolute URL and
            actually fetched by :meth:`_download_product` after login.
        """
        resp = session.post(
            OADS_TREE_URL,
            data={
                "p0": OADS_PRODUCT_TYPE,
                "p1": f"{day.year:04d}",
                "p2": f"{day.month:02d}",
                "p3": f"{day.day:02d}",
            },
            timeout=60,
        )
        resp.raise_for_status()

        products = []
        # Each product is a <h5 class="productTitle">filename</h5> followed
        # by a "Download Product" <a href="...">; matched non-greedily so
        # multiple productContainer blocks don't bleed into each other.
        # href's closing quote is followed by [^>]* (not immediately ">")
        # because the *authenticated* page's real link carries an extra
        # target="_blank" attribute in between -- confirmed against a real
        # authenticated fetch, which also revealed the href itself differs
        # from the unauthenticated page this was originally written
        # against: authenticated hrefs are direct download links
        # (/oads/data/NRT_Open/<filename>), not the login-gated redirect
        # (/oads/access/login?...) an unauthenticated fetch sees.
        for title_match in re.finditer(
            r'class=["\']productTitle["\']>([^<]+)</h5>.*?<a href=["\']([^"\']+)["\'][^>]*>Download Product</a>',
            resp.text, re.DOTALL,
        ):
            filename, href = title_match.group(1), title_match.group(2)
            products.append({
                "filename": filename,
                "download_href": href.replace("&amp;", "&"),
            })

        if not products and "No products" not in resp.text:
            # Zero matches but the page doesn't say "no products" either --
            # the regex likely doesn't match this (possibly authenticated,
            # possibly-different-from-unauthenticated) page's real layout.
            # Save it so the mismatch can actually be diagnosed instead of
            # silently reporting 0 downloads with no explanation.
            debug_path = self.output_dir / f"smos_list_debug_{day.isoformat()}.html"
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                debug_path.write_text(resp.text)
                print(
                    f"  {day.isoformat()}: 0 products parsed but the page "
                    f"didn't say 'no products' either -- response saved to "
                    f"{debug_path} for inspection."
                )
            except OSError:
                pass

        return products

    def _login(self, session: "requests.Session", username: str, password: str) -> None:
        """
        Authenticate *session* against ESA EOIAM's SAML2/WSO2 SSO, so
        subsequent requests through it can reach OADS's login-gated
        "Download Product" links.

        Three-hop flow, matching the ``loginForm``/``samlsso`` pattern
        served by ``eoiam-idp.eo.esa.int`` (confirmed live, unauthenticated,
        against ``/oads/access/login``): (1) GET the OADS login URL, which
        redirects to the IdP and embeds a ``sessionDataKey`` in a hidden
        login form; (2) POST username/password plus that key to the IdP's
        ``samlsso`` endpoint; (3) the IdP's response is itself an
        auto-submitting HTML form carrying a ``SAMLResponse``/``RelayState``
        pair, which must be POSTed to the service provider's ACS endpoint
        to actually establish the authenticated session cookie.
        """
        resp = session.get(OADS_LOGIN_URL, timeout=60)
        resp.raise_for_status()

        # Match form with id="loginForm" and action in either order. Quotes
        # are matched as ["'] rather than hardcoded to " — confirmed live
        # against the real IdP response that attribute quoting is NOT
        # consistent, not even within one page (action="..."/id="..." use
        # double quotes, but the sessionDataKey <input>'s value='...' below
        # uses single quotes) and not even for the same attribute name
        # across different hops (this hop's "name=" happens to be
        # double-quoted; the SAMLResponse hop's below is single-quoted) —
        # so every quoted attribute in this method, "name=" included, is
        # matched quote-agnostically rather than assuming any one style.
        action_match = re.search(
            r'<form[^>]*action=["\']([^"\']+)["\'][^>]*id="loginForm"'
            r'|<form[^>]*id="loginForm"[^>]*action=["\']([^"\']+)["\']',
            resp.text
        )
        key_match = re.search(r'name=["\']sessionDataKey["\']\s+value=["\']([^"\']+)["\']', resp.text)
        if not action_match or not key_match:
            raise RuntimeError(
                "SMOS OADS login failed: could not find the EOIAM login "
                "form/sessionDataKey in the IdP's response. The login page "
                "layout may have changed."
            )

        # Extract action URL from whichever group matched. The IdP serves
        # this as a relative URL (confirmed live: action="../samlsso"), so
        # it must be resolved against the actual post-redirect page URL
        # (resp.url, populated by requests following the OADS_LOGIN_URL
        # redirect chain) rather than posted as-is — passing a relative URL
        # directly to session.post() raises requests.exceptions.MissingSchema.
        action_url = action_match.group(1) if action_match.group(1) else action_match.group(2)
        action_url = urljoin(resp.url, action_url)

        idp_resp = session.post(
            action_url,
            data={
                "tocommonauth": "true",
                "username": username,
                "password": password,
                "sessionDataKey": key_match.group(1),
            },
            timeout=60,
        )
        idp_resp.raise_for_status()

        acs_action_match = re.search(r'<form[^>]*action=["\']([^"\']+)["\']', idp_resp.text)
        saml_response_match = re.search(
            r'name=["\']SAMLResponse["\']\s+value=["\']([^"\']+)["\']', idp_resp.text,
        )
        if not acs_action_match or not saml_response_match:
            debug_path = self.output_dir / "smos_saml_debug.html"
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                debug_path.write_text(idp_resp.text)
                debug_note = f" The IdP's response body was saved to {debug_path} for inspection."
            except OSError:
                debug_note = ""
            raise RuntimeError(
                "SMOS OADS login failed: no SAMLResponse returned after "
                f"POSTing credentials (IdP responded {idp_resp.status_code} "
                f"from {idp_resp.url}). This usually means either the "
                "credentials were rejected (check SMOS_FTP_USERNAME/"
                "SMOS_FTP_PASSWORD) or the IdP returned something other "
                "than the expected auto-submit SAMLResponse form (e.g. an "
                "MFA/OTP prompt, an account-locked notice, or a changed "
                f"page layout).{debug_note}"
            )
        relay_state_match = re.search(
            r'name=["\']RelayState["\']\s+value=["\']([^"\']*)["\']', idp_resp.text,
        )

        acs_data = {"SAMLResponse": saml_response_match.group(1)}
        if relay_state_match:
            acs_data["RelayState"] = relay_state_match.group(1)
        acs_action_url = urljoin(idp_resp.url, acs_action_match.group(1))
        acs_resp = session.post(acs_action_url, data=acs_data, timeout=60)
        acs_resp.raise_for_status()

    def _filter_by_orbit_overlap(
        self, products: list, day, min_lon: float, max_lon: float, min_lat: float, max_lat: float,
    ) -> list:
        """Drop products whose sensing window shows no predicted orbit
        overlap with the requested bbox -- see
        orbit_coverage.orbit_overlaps_bbox. Uses each filename's real
        embedded start/stop timestamps when parseable (see
        _parse_sensing_window); falls back to the whole day [00:00:00Z,
        23:59:59Z] otherwise -- a real, expected fallback (not a
        defensive "can't happen" branch), since the real OADS filename
        format hasn't been directly confirmed by this codebase yet."""
        from ..core.orbit_coverage import orbit_overlaps_bbox

        kept = []
        dropped = 0
        for product in products:
            window = _parse_sensing_window(product["filename"])
            if window is not None:
                start, end = window
            else:
                start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
                end = start + timedelta(hours=23, minutes=59, seconds=59)
            if orbit_overlaps_bbox("smos", start, end, min_lon, max_lon, min_lat, max_lat):
                kept.append(product)
            else:
                dropped += 1
        if dropped:
            print(f"Orbit pre-filter: skipped {dropped} file(s) with no predicted overlap.")
        return kept

    def download(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start: str,
        end: str,
    ) -> list[Path]:
        """
        Download SMOS SSM products for every day in ``[start, end]`` via
        ESA's OADS web portal (see module docstring — FTPS was replaced
        after confirming the FTP server hangs indefinitely from this
        toolbox's network path, while OADS responds normally).

        Returns
        -------
        list[Path]
            Paths to the downloaded files.
        """
        start_dt = normalize_datetime(start)
        end_dt = normalize_datetime(end)

        if self.dry_run:
            print(
                f"[DRY RUN] Would download SMOS SSM data\n"
                f"  Region: lon [{min_lon},{max_lon}] lat [{min_lat},{max_lat}] (cropped downstream)\n"
                f"  Time:   {start_dt} -> {end_dt}\n"
                f"  Server: smos-diss.eo.esa.int (OADS)\n"
                f"  Output: {self.output_dir}"
            )
            return []

        username, password = authenticate_smos_ftp(self._username, self._password)

        day = datetime.fromisoformat(start_dt).date()
        last = datetime.fromisoformat(end_dt).date()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        session = requests.Session()
        self._login(session, username, password)

        downloaded: list[Path] = []
        while day <= last:
            products = self._list_products_for_day(session, day)
            if self.orbit_prefilter:
                products = self._filter_by_orbit_overlap(
                    products, day, min_lon, max_lon, min_lat, max_lat,
                )
            for product in products:
                fname = product["filename"]
                if not fname.endswith((".nc", ".tgz")):
                    continue
                dest = self.output_dir / fname
                if dest.exists():
                    print(f"  {fname}: already present, skipping.")
                    downloaded.append(dest)
                    continue
                print(f"  Downloading {fname} …")
                href = product["download_href"]
                url = href if href.startswith("http") else f"{OADS_BASE_URL.rsplit('/oads', 1)[0]}{href}"
                resp = session.get(url, timeout=120)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                downloaded.append(dest)
            day += timedelta(days=1)

        print(f"Downloaded {len(downloaded)} SMOS SSM file(s).")
        return downloaded


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download SMOS soil-moisture data from ESA Online Dissemination (OADS).",
    )
    p.add_argument("--params-file", metavar="FILE")
    p.add_argument("--min-lon", type=float)
    p.add_argument("--max-lon", type=float)
    p.add_argument("--min-lat", type=float)
    p.add_argument("--max-lat", type=float)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--username", default=None)
    p.add_argument("--password", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--no-orbit-prefilter", dest="orbit_prefilter", action="store_false", default=True,
        help="Disable the orbit-based geographic pre-filter (default: enabled).",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if args.params_file:
        with open(args.params_file) as f:
            params = json.load(f)
        min_lon = params["minimum_longitude"]
        max_lon = params["maximum_longitude"]
        min_lat = params["minimum_latitude"]
        max_lat = params["maximum_latitude"]
        start = params["start_datetime"]
        end = params["end_datetime"]
    else:
        for attr in ("min_lon", "max_lon", "min_lat", "max_lat", "start", "end"):
            if getattr(args, attr) is None:
                print(f"Error: --{attr.replace('_','-')} is required (or use --params-file)")
                sys.exit(1)
        min_lon, max_lon = args.min_lon, args.max_lon
        min_lat, max_lat = args.min_lat, args.max_lat
        start, end = args.start, args.end

    output_dir = Path(args.output_dir) if args.output_dir else (
        build_output_dir(start, end, min_lon, max_lon, min_lat, max_lat) / "smos_ssm"
    )

    dl = SMOSDownloader(
        output_dir=output_dir,
        dry_run=args.dry_run,
        username=args.username,
        password=args.password,
        orbit_prefilter=args.orbit_prefilter,
    )
    dl.download(
        min_lon=min_lon, max_lon=max_lon,
        min_lat=min_lat, max_lat=max_lat,
        start=start, end=end,
    )


if __name__ == "__main__":
    main()
