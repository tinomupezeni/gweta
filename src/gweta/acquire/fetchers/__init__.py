"""Pluggable fetchers for different data sources.

This module provides specialized fetchers for:
- Single URL fetching
- Sitemap-based crawling
- RSS/Atom feed ingestion
"""

from gweta.acquire.fetchers.url import URLFetcher
from gweta.acquire.fetchers.sitemap import SitemapFetcher
from gweta.acquire.fetchers.rss import RSSFetcher

__all__ = [
    "URLFetcher",
    "SitemapFetcher",
    "RSSFetcher",
]
