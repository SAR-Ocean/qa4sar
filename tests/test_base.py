"""Tests for CopernicusODataClient's OData query filter construction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from sar_validation.downloaders.base import CopernicusODataClient


def _make_client() -> CopernicusODataClient:
    fake_token_resp = MagicMock()
    fake_token_resp.status_code = 200
    fake_token_resp.json.return_value = {"access_token": "tok", "expires_in": 3600}
    fake_token_resp.raise_for_status.return_value = None
    with patch("requests.post", return_value=fake_token_resp):
        return CopernicusODataClient("user", "pass")


def _fake_query_resp():
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"value": []}
    return resp


class TestQueryProductsDateFilterIsInclusive:
    """A daily product's ContentDate/Start commonly lands exactly on a
    requested window boundary (e.g. a recipe's start/end date normalizes
    to an exact midnight timestamp). A strict gt/lt filter silently drops
    that boundary-matching product -- confirmed against a real recipe run
    (2025-07-01..03) that only found the 07-02 CLMS SSM product, not
    07-01, because 07-01's ContentDate/Start equalled the requested start
    exactly. The filter must use ge/le so an exact-boundary match is kept."""

    def test_query_products_uses_inclusive_bounds(self):
        client = _make_client()
        with patch.object(client.session, "get", return_value=_fake_query_resp()) as mock_get:
            client.query_products(
                collection="SENTINEL-1", product_type="OCN",
                start_date="2025-07-01T00:00:00.000Z",
                end_date="2025-07-03T00:00:00.000Z",
                min_lon=-10.0, max_lon=30.0, min_lat=35.0, max_lat=60.0,
            )

        filter_str = mock_get.call_args.kwargs["params"]["$filter"]
        assert "ContentDate/Start ge 2025-07-01T00:00:00.000Z" in filter_str
        assert "ContentDate/Start le 2025-07-03T00:00:00.000Z" in filter_str
        assert "ContentDate/Start gt" not in filter_str
        assert "ContentDate/Start lt" not in filter_str

    def test_query_clms_products_uses_inclusive_bounds(self):
        client = _make_client()
        with patch.object(client.session, "get", return_value=_fake_query_resp()) as mock_get:
            client.query_clms_products(
                dataset_identifier="ssm_europe_1km_daily_v1",
                start_date="2025-07-01T00:00:00.000Z",
                end_date="2025-07-03T00:00:00.000Z",
                min_lon=-10.0, max_lon=30.0, min_lat=35.0, max_lat=60.0,
            )

        filter_str = mock_get.call_args.kwargs["params"]["$filter"]
        assert "ContentDate/Start ge 2025-07-01T00:00:00.000Z" in filter_str
        assert "ContentDate/Start le 2025-07-03T00:00:00.000Z" in filter_str
        assert "ContentDate/Start gt" not in filter_str
        assert "ContentDate/Start lt" not in filter_str
