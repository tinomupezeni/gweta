"""Sitemap-based URL discovery and crawling.

This module provides the SitemapFetcher class for discovering
URLs from XML sitemaps for systematic crawling.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from xml.etree import ElementTree as ET

import httpx

from gweta.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SitemapURL:
    """A URL entry from a sitemap."""
    loc: str
    lastmod: datetime | None = None
    changefreq: str | None = None
    priority: float | None = None


@dataclass
class SitemapResult:
    """Results from parsing a sitemap."""
    url: str
    urls: list[SitemapURL] = field(default_factory=list)
    nested_sitemaps: list[str] = field(default_factory=list)
    success: bool = True
    error: str | None = None


class SitemapFetcher:
    """Fetch and parse XML sitemaps for URL discovery.

    Supports:
    - Standard XML sitemaps
    - Sitemap index files
    - Recursive sitemap discovery

    Example:
        >>> fetcher = SitemapFetcher()
        >>> result = await fetcher.fetch("https://example.com/sitemap.xml")
        >>> for url in result.urls:
        ...     print(url.loc)
    """

    # XML namespace for sitemaps
    NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    def __init__(
        self,
        timeout: float = 30.0,
        max_urls: int = 10000,
    ) -> None:
        """Initialize SitemapFetcher.

        Args:
            timeout: Request timeout in seconds
            max_urls: Maximum URLs to extract
        """
        self.timeout = timeout
        self.max_urls = max_urls
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": "Gweta/0.1"},
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def fetch(
        self,
        url: str,
        recursive: bool = True,
    ) -> SitemapResult:
        """Fetch and parse a sitemap.

        Args:
            url: URL of the sitemap
            recursive: Whether to follow nested sitemaps

        Returns:
            SitemapResult with discovered URLs
        """
        client = await self._get_client()
        result = SitemapResult(url=url)

        try:
            response = await client.get(url)
            if not response.is_success:
                result.success = False
                result.error = f"HTTP {response.status_code}"
                return result

            # Parse XML
            root = ET.fromstring(response.text)

            # Check if it's a sitemap index
            if root.tag.endswith("sitemapindex"):
                result.nested_sitemaps = self._parse_sitemap_index(root)

                # Recursively fetch nested sitemaps
                if recursive:
                    for nested_url in result.nested_sitemaps[: self.max_urls]:
                        nested_result = await self.fetch(nested_url, recursive=False)
                        result.urls.extend(nested_result.urls)
                        if len(result.urls) >= self.max_urls:
                            break
            else:
                result.urls = self._parse_urlset(root)

            logger.info(f"Extracted {len(result.urls)} URLs from {url}")

        except ET.ParseError as e:
            result.success = False
            result.error = f"XML parse error: {e}"
        except Exception as e:
            result.success = False
            result.error = str(e)

        return result

    def _parse_sitemap_index(self, root: ET.Element) -> list[str]:
        """Parse sitemap index to get nested sitemap URLs."""
        urls: list[str] = []

        for sitemap in root.findall(".//sm:sitemap", self.NS):
            loc = sitemap.find("sm:loc", self.NS)
            if loc is not None and loc.text:
                urls.append(loc.text)

        return urls

    def _parse_urlset(self, root: ET.Element) -> list[SitemapURL]:
        """Parse urlset to get page URLs."""
        urls: list[SitemapURL] = []

        for url_elem in root.findall(".//sm:url", self.NS):
            loc = url_elem.find("sm:loc", self.NS)
            if loc is None or not loc.text:
                continue

            lastmod_elem = url_elem.find("sm:lastmod", self.NS)
            lastmod = None
            if lastmod_elem is not None and lastmod_elem.text:
                try:
                    lastmod = datetime.fromisoformat(
                        lastmod_elem.text.replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

            changefreq_elem = url_elem.find("sm:changefreq", self.NS)
            changefreq = changefreq_elem.text if changefreq_elem is not None else None

            priority_elem = url_elem.find("sm:priority", self.NS)
            priority = None
            if priority_elem is not None and priority_elem.text:
                try:
                    priority = float(priority_elem.text)
                except ValueError:
                    pass

            urls.append(
                SitemapURL(
                    loc=loc.text,
                    lastmod=lastmod,
                    changefreq=changefreq,
                    priority=priority,
                )
            )

            if len(urls) >= self.max_urls:
                break

        return urls
