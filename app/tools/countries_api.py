"""
HTTP wrapper around the REST Countries API
"""

from __future__ import annotations
import logging
from typing import Any
import httpx
from app.config import get_settings

logger = logging.getLogger(__name__)


class CountryNotFoundError(Exception):
    """Raised when the REST Countries API returns a 404."""


class CountriesAPIError(Exception):
    """Raised for any unexpected API or network error."""


# Module-level client
_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    """Return (or lazily create) the shared async HTTP client."""
    global _client
    if _client is None or _client.is_closed:
        settings = get_settings()
        transport = httpx.AsyncHTTPTransport(retries=settings.max_retries)
        _client = httpx.AsyncClient(
            base_url=settings.rest_countries_base_url,
            timeout=settings.http_timeout_seconds,
            transport=transport,
            headers={"Accept": "application/json"},
        )
    return _client


async def close_client() -> None:
    """Close the shared HTTP client."""
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None


async def fetch_country(country_name: str) -> list[dict[str, Any]]:
    """
    Fetch country data from the REST Countries API.

    Raises:

    CountryNotFoundError when HTTP 404
    CountriesAPIError for any other HTTP or network error
    """
    if not country_name or not country_name.strip():
        raise ValueError("country_name must not be empty")

    client = get_client()
    url = f"/name/{country_name.strip()}"

    try:
        response = await client.get(url)
    except httpx.TimeoutException as exc:
        raise CountriesAPIError(
            f"Request timed out while fetching '{country_name}'"
        ) from exc
    except httpx.RequestError as exc:
        raise CountriesAPIError(
            f"Network error while fetching '{country_name}': {exc}"
        ) from exc

    if response.status_code == 404:
        raise CountryNotFoundError(
            f"No country found matching '{country_name}'. "
            "Please check the spelling and try again."
        )

    if response.status_code != 200:
        raise CountriesAPIError(
            f"REST Countries API returned HTTP {response.status_code} "
            f"for '{country_name}'."
        )

    data: list[dict[str, Any]] = response.json()
    logger.debug("Received %d result(s) for '%s'", len(data), country_name)
    return data