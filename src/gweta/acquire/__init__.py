"""Acquisition layer for Gweta.

This module provides data acquisition from various sources:
- Web crawling (via Crawl4AI wrapper)
- PDF extraction (via pdfplumber)
- API fetching (REST/GraphQL)
- Database connectivity (via SQLAlchemy)
- URL, sitemap, and RSS/Atom feed fetching
"""

from gweta.acquire.crawler import GwetaCrawler, CrawlResult, PageQualityScore
from gweta.acquire.pdf import PDFExtractor, PDFExtractionResult, PDFPage, PDFTable
from gweta.acquire.api import APIClient, APIResponse
from gweta.acquire.database import DatabaseSource, QueryResult, QuerySanitizer
from gweta.acquire.fetchers import URLFetcher, SitemapFetcher, RSSFetcher

__all__ = [
    # Crawler
    "GwetaCrawler",
    "CrawlResult",
    "PageQualityScore",
    # PDF
    "PDFExtractor",
    "PDFExtractionResult",
    "PDFPage",
    "PDFTable",
    # API
    "APIClient",
    "APIResponse",
    # Database
    "DatabaseSource",
    "QueryResult",
    "QuerySanitizer",
    # Fetchers
    "URLFetcher",
    "SitemapFetcher",
    "RSSFetcher",
]
