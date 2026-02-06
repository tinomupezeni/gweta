"""Single URL fetcher for lightweight requests.

This module provides a simple URL fetcher using httpx
for cases where a full browser crawl is not needed.
"""

from dataclasses import dataclass
from typing import Any

import httpx
from bs4 import BeautifulSoup

from gweta.core.config import get_settings
from gweta.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FetchResult:
    """Result from fetching a single URL."""
    url: str
    status_code: int
    content: str
    content_type: str
    title: str | None = None
    success: bool = True
    error: str | None = None


class URLFetcher:
    """Lightweight URL fetcher using httpx.

    Use this for simple static pages where JavaScript
    rendering is not needed.

    Example:
        >>> fetcher = URLFetcher()
        >>> result = await fetcher.fetch("https://example.com")
        >>> print(result.content)
    """

    def __init__(
        self,
        timeout: float = 30.0,
        follow_redirects: bool = True,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize URLFetcher.

        Args:
            timeout: Request timeout in seconds
            follow_redirects: Whether to follow redirects
            headers: Default headers for requests
        """
        self.timeout = timeout
        self.follow_redirects = follow_redirects
        self.default_headers = headers or {
            "User-Agent": "Gweta/0.1 (+https://github.com/gweta/gweta)"
        }
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=self.follow_redirects,
                headers=self.default_headers,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def fetch(self, url: str) -> FetchResult:
        """Fetch a single URL.

        Args:
            url: URL to fetch

        Returns:
            FetchResult with content and metadata
        """
        client = await self._get_client()

        try:
            response = await client.get(url)
            content_type = response.headers.get("content-type", "")

            # Extract text content
            if "text/html" in content_type:
                content, title = self._extract_html(response.text)
            else:
                content = response.text
                title = None

            return FetchResult(
                url=url,
                status_code=response.status_code,
                content=content,
                content_type=content_type,
                title=title,
                success=response.is_success,
            )

        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return FetchResult(
                url=url,
                status_code=0,
                content="",
                content_type="",
                success=False,
                error=str(e),
            )

    def _extract_html(self, html: str) -> tuple[str, str | None]:
        """Extract text content from HTML.

        Args:
            html: Raw HTML content

        Returns:
            Tuple of (text_content, title)
        """
        soup = BeautifulSoup(html, "html.parser")

        # Get title
        title = soup.title.string if soup.title else None

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Get text
        text = soup.get_text(separator="\n", strip=True)

        return text, title
