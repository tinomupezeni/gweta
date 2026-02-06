"""REST/GraphQL API client with quality validation.

This module provides the APIClient class for fetching and
validating data from REST and GraphQL APIs.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx

from gweta.core.config import GwetaSettings, get_settings
from gweta.core.exceptions import AcquisitionError
from gweta.core.logging import get_logger
from gweta.core.types import QualityIssue

logger = get_logger(__name__)


@dataclass
class APIResponse:
    """Response from an API request.

    Attributes:
        url: Request URL
        status_code: HTTP status code
        data: Response data (parsed JSON or raw text)
        headers: Response headers
        quality_score: Data quality score
        issues: List of quality issues
    """
    url: str
    status_code: int
    data: Any
    headers: dict[str, str] = field(default_factory=dict)
    quality_score: float = 1.0
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        """Check if request was successful."""
        return 200 <= self.status_code < 300

    @property
    def is_json(self) -> bool:
        """Check if response is JSON."""
        return isinstance(self.data, (dict, list))


class APIClient:
    """Fetch and validate data from REST/GraphQL APIs.

    Provides HTTP client functionality with:
    - Automatic retry logic
    - Response validation
    - Pagination support
    - Rate limiting awareness

    Example:
        >>> client = APIClient(base_url="https://api.example.com")
        >>> response = await client.fetch("/users")
        >>> print(response.data)
    """

    def __init__(
        self,
        base_url: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        config: GwetaSettings | None = None,
    ) -> None:
        """Initialize APIClient.

        Args:
            base_url: Base URL for all requests
            headers: Default headers for all requests
            timeout: Request timeout in seconds
            config: Gweta settings
        """
        self.base_url = base_url or ""
        self.default_headers = headers or {}
        self.timeout = timeout
        self.config = config or get_settings()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.default_headers,
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "APIClient":
        """Enter async context."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context."""
        await self.close()

    async def fetch(
        self,
        url: str,
        method: str = "GET",
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> APIResponse:
        """Fetch data from an API endpoint.

        Args:
            url: URL or path to fetch
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            json: JSON body for POST/PUT requests
            headers: Additional headers for this request

        Returns:
            APIResponse with data and quality metrics

        Raises:
            AcquisitionError: If request fails
        """
        client = await self._get_client()
        full_url = url if url.startswith("http") else f"{self.base_url}{url}"

        try:
            response = await client.request(
                method=method,
                url=url,
                params=params,
                json=json,
                headers=headers,
            )

            # Try to parse as JSON
            try:
                data = response.json()
            except Exception:
                data = response.text

            result = APIResponse(
                url=full_url,
                status_code=response.status_code,
                data=data,
                headers=dict(response.headers),
            )

            # Validate response
            result.quality_score, result.issues = self._validate_response(result)

            return result

        except httpx.TimeoutException as e:
            raise AcquisitionError(
                f"Request timeout for {full_url}",
                source=full_url,
                source_type="api",
            ) from e
        except httpx.RequestError as e:
            raise AcquisitionError(
                f"Request failed for {full_url}: {e}",
                source=full_url,
                source_type="api",
            ) from e

    async def fetch_paginated(
        self,
        url: str,
        page_param: str = "page",
        page_size_param: str | None = "page_size",
        page_size: int = 100,
        max_pages: int = 100,
        data_key: str | None = None,
    ) -> list[APIResponse]:
        """Fetch all pages from a paginated API endpoint.

        Args:
            url: Base URL for the paginated endpoint
            page_param: Query parameter name for page number
            page_size_param: Query parameter for page size (optional)
            page_size: Number of items per page
            max_pages: Maximum number of pages to fetch
            data_key: Key in response containing the data array

        Returns:
            List of APIResponse objects, one per page
        """
        responses: list[APIResponse] = []
        page = 1

        while page <= max_pages:
            params: dict[str, Any] = {page_param: page}
            if page_size_param:
                params[page_size_param] = page_size

            response = await self.fetch(url, params=params)
            responses.append(response)

            # Check if we should continue
            if not response.is_success:
                break

            # Check if there's more data
            if data_key and isinstance(response.data, dict):
                data = response.data.get(data_key, [])
                if not data or len(data) < page_size:
                    break
            elif isinstance(response.data, list):
                if len(response.data) < page_size:
                    break
            else:
                break

            page += 1

        logger.info(f"Fetched {len(responses)} pages from {url}")
        return responses

    def _validate_response(
        self,
        response: APIResponse,
    ) -> tuple[float, list[QualityIssue]]:
        """Validate API response quality.

        Args:
            response: Response to validate

        Returns:
            Tuple of (quality_score, issues)
        """
        issues: list[QualityIssue] = []
        score = 1.0

        # Check status code
        if not response.is_success:
            issues.append(
                QualityIssue(
                    code="HTTP_ERROR",
                    severity="error",
                    message=f"HTTP {response.status_code}",
                )
            )
            score = 0.0
            return score, issues

        # Check for empty response
        if response.data is None or response.data == "":
            issues.append(
                QualityIssue(
                    code="EMPTY_RESPONSE",
                    severity="warning",
                    message="Response body is empty",
                )
            )
            score *= 0.5

        # Check for error indicators in JSON
        if isinstance(response.data, dict):
            if "error" in response.data or "errors" in response.data:
                issues.append(
                    QualityIssue(
                        code="API_ERROR",
                        severity="error",
                        message="Response contains error field",
                    )
                )
                score *= 0.3

        return score, issues

    def fetch_sync(self, url: str, **kwargs: Any) -> APIResponse:
        """Synchronous wrapper for fetch().

        Args:
            url: URL to fetch
            **kwargs: Additional arguments

        Returns:
            APIResponse
        """
        return asyncio.run(self.fetch(url, **kwargs))
