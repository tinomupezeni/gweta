"""RSS/Atom feed fetcher for content ingestion.

This module provides the RSSFetcher class for extracting
content from RSS and Atom feeds.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from xml.etree import ElementTree as ET

import httpx

from gweta.core.logging import get_logger
from gweta.core.types import Chunk

logger = get_logger(__name__)


@dataclass
class FeedItem:
    """A single item from an RSS/Atom feed."""
    title: str
    link: str
    content: str
    published: datetime | None = None
    author: str | None = None
    categories: list[str] = field(default_factory=list)


@dataclass
class FeedResult:
    """Results from parsing a feed."""
    url: str
    title: str | None = None
    description: str | None = None
    items: list[FeedItem] = field(default_factory=list)
    feed_type: str = "unknown"  # rss or atom
    success: bool = True
    error: str | None = None


class RSSFetcher:
    """Fetch and parse RSS/Atom feeds.

    Extracts content from feeds for RAG ingestion,
    with automatic detection of RSS vs Atom format.

    Example:
        >>> fetcher = RSSFetcher()
        >>> result = await fetcher.fetch("https://example.com/feed.xml")
        >>> for item in result.items:
        ...     print(item.title)
    """

    # Atom namespace
    ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

    def __init__(
        self,
        timeout: float = 30.0,
        max_items: int = 100,
    ) -> None:
        """Initialize RSSFetcher.

        Args:
            timeout: Request timeout in seconds
            max_items: Maximum items to extract
        """
        self.timeout = timeout
        self.max_items = max_items
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

    async def fetch(self, url: str) -> FeedResult:
        """Fetch and parse a feed.

        Args:
            url: URL of the RSS/Atom feed

        Returns:
            FeedResult with extracted items
        """
        client = await self._get_client()
        result = FeedResult(url=url)

        try:
            response = await client.get(url)
            if not response.is_success:
                result.success = False
                result.error = f"HTTP {response.status_code}"
                return result

            # Parse XML
            root = ET.fromstring(response.text)

            # Detect feed type
            if root.tag.endswith("feed"):
                result.feed_type = "atom"
                self._parse_atom(root, result)
            elif root.tag == "rss" or root.find("channel") is not None:
                result.feed_type = "rss"
                self._parse_rss(root, result)
            else:
                result.success = False
                result.error = "Unknown feed format"

            logger.info(f"Extracted {len(result.items)} items from {url}")

        except ET.ParseError as e:
            result.success = False
            result.error = f"XML parse error: {e}"
        except Exception as e:
            result.success = False
            result.error = str(e)

        return result

    def _parse_rss(self, root: ET.Element, result: FeedResult) -> None:
        """Parse RSS feed."""
        channel = root.find("channel")
        if channel is None:
            return

        # Feed metadata
        title_elem = channel.find("title")
        result.title = title_elem.text if title_elem is not None else None

        desc_elem = channel.find("description")
        result.description = desc_elem.text if desc_elem is not None else None

        # Items
        for item in channel.findall("item")[: self.max_items]:
            title = item.find("title")
            link = item.find("link")
            desc = item.find("description")
            content = item.find("{http://purl.org/rss/1.0/modules/content/}encoded")
            pubdate = item.find("pubDate")
            author = item.find("author") or item.find("{http://purl.org/dc/elements/1.1/}creator")

            published = None
            if pubdate is not None and pubdate.text:
                try:
                    # Try common RSS date formats
                    from email.utils import parsedate_to_datetime
                    published = parsedate_to_datetime(pubdate.text)
                except Exception:
                    pass

            categories = [
                cat.text for cat in item.findall("category")
                if cat.text
            ]

            result.items.append(
                FeedItem(
                    title=title.text if title is not None else "",
                    link=link.text if link is not None else "",
                    content=(content.text if content is not None else
                             desc.text if desc is not None else ""),
                    published=published,
                    author=author.text if author is not None else None,
                    categories=categories,
                )
            )

    def _parse_atom(self, root: ET.Element, result: FeedResult) -> None:
        """Parse Atom feed."""
        # Feed metadata
        title_elem = root.find("atom:title", self.ATOM_NS)
        result.title = title_elem.text if title_elem is not None else None

        subtitle_elem = root.find("atom:subtitle", self.ATOM_NS)
        result.description = subtitle_elem.text if subtitle_elem is not None else None

        # Entries
        for entry in root.findall("atom:entry", self.ATOM_NS)[: self.max_items]:
            title = entry.find("atom:title", self.ATOM_NS)
            link = entry.find("atom:link[@rel='alternate']", self.ATOM_NS)
            if link is None:
                link = entry.find("atom:link", self.ATOM_NS)
            content = entry.find("atom:content", self.ATOM_NS)
            summary = entry.find("atom:summary", self.ATOM_NS)
            published = entry.find("atom:published", self.ATOM_NS)
            updated = entry.find("atom:updated", self.ATOM_NS)
            author = entry.find("atom:author/atom:name", self.ATOM_NS)

            pub_date = None
            date_elem = published or updated
            if date_elem is not None and date_elem.text:
                try:
                    pub_date = datetime.fromisoformat(
                        date_elem.text.replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

            categories = [
                cat.get("term", "")
                for cat in entry.findall("atom:category", self.ATOM_NS)
                if cat.get("term")
            ]

            result.items.append(
                FeedItem(
                    title=title.text if title is not None else "",
                    link=link.get("href", "") if link is not None else "",
                    content=(content.text if content is not None else
                             summary.text if summary is not None else ""),
                    published=pub_date,
                    author=author.text if author is not None else None,
                    categories=categories,
                )
            )

    def to_chunks(
        self,
        result: FeedResult,
        include_metadata: bool = True,
    ) -> list[Chunk]:
        """Convert feed items to chunks.

        Args:
            result: FeedResult to convert
            include_metadata: Whether to include item metadata

        Returns:
            List of Chunk objects
        """
        chunks: list[Chunk] = []

        for item in result.items:
            if not item.content.strip():
                continue

            metadata: dict[str, Any] = {
                "source_type": "feed",
                "feed_url": result.url,
                "feed_type": result.feed_type,
            }

            if include_metadata:
                metadata["title"] = item.title
                metadata["link"] = item.link
                if item.published:
                    metadata["published"] = item.published.isoformat()
                if item.author:
                    metadata["author"] = item.author
                if item.categories:
                    metadata["categories"] = item.categories

            chunks.append(
                Chunk(
                    text=item.content,
                    source=item.link or result.url,
                    metadata=metadata,
                    quality_score=1.0,
                )
            )

        return chunks
