"""MCP tools for Gweta.

This module defines the MCP tools that expose
Gweta's capabilities to AI agents.
"""

from typing import Any

from gweta.core.logging import get_logger

logger = get_logger(__name__)


def register_tools(mcp: Any) -> None:
    """Register all MCP tools.

    Args:
        mcp: FastMCP server instance
    """

    @mcp.tool()
    async def crawl_and_ingest(
        url: str,
        depth: int = 2,
        target_collection: str = "default",
        authority_tier: int = 3,
        rules: str | None = None,
    ) -> dict[str, Any]:
        """Crawl a website, validate extracted content, and load
        quality chunks into the target vector database collection.

        Args:
            url: Starting URL to crawl
            depth: How many links deep to follow (1-5)
            target_collection: Name of the vector DB collection
            authority_tier: Source authority level (1=blog, 5=legislation)
            rules: Optional domain rule set name

        Returns:
            Quality report with crawl statistics
        """
        from gweta.acquire.crawler import GwetaCrawler
        from gweta.ingest.stores.chroma import ChromaStore

        crawler = GwetaCrawler()
        result = await crawler.crawl(url, depth=depth)

        store = ChromaStore(collection_name=target_collection)
        if result.chunks:
            await store.add(result.chunks)

        return {
            "url": url,
            "pages_crawled": result.pages_crawled,
            "pages_passed": result.pages_passed,
            "pages_failed": result.pages_failed,
            "chunks_loaded": len(result.chunks),
            "chunks_rejected": len(result.rejected_chunks),
            "quality_score": result.quality_score,
        }

    @mcp.tool()
    async def validate_chunks(
        chunks: list[dict[str, Any]],
        rules: str | None = None,
    ) -> dict[str, Any]:
        """Validate a list of chunks without loading them.

        Args:
            chunks: List of chunk dicts with 'text' and 'metadata'
            rules: Optional domain rule set name

        Returns:
            Validation report with pass/fail per chunk
        """
        from gweta.core.types import Chunk
        from gweta.validate.chunks import ChunkValidator

        validator = ChunkValidator()

        # Convert dicts to Chunk objects
        chunk_objects = [
            Chunk(
                text=c.get("text", ""),
                metadata=c.get("metadata", {}),
                source=c.get("source", ""),
            )
            for c in chunks
        ]

        report = validator.validate_batch(chunk_objects)

        return {
            "total_chunks": report.total_chunks,
            "passed": report.passed,
            "failed": report.failed,
            "avg_quality_score": report.avg_quality_score,
            "issues_by_type": report.issues_by_type,
        }

    @mcp.tool()
    async def check_health(
        collection: str,
        golden_dataset: str | None = None,
    ) -> dict[str, Any]:
        """Check health of a knowledge base collection.

        Returns quality scores, stale sources, coverage gaps,
        and specific chunks that fail validation.

        Args:
            collection: Name of the collection to check
            golden_dataset: Optional path to golden dataset for testing
        """
        from pathlib import Path

        from gweta.ingest.stores.chroma import ChromaStore
        from gweta.validate.health import HealthChecker

        store = ChromaStore(collection_name=collection)
        checker = HealthChecker(store)

        golden_path = Path(golden_dataset) if golden_dataset else None
        report = await checker.full_health_check(golden_dataset=golden_path)

        return {
            "collection": report.collection,
            "total_chunks": report.total_chunks,
            "avg_quality_score": report.avg_quality_score,
            "staleness": {
                "stale_chunks": report.staleness.stale_chunks if report.staleness else 0,
                "fresh_chunks": report.staleness.fresh_chunks if report.staleness else 0,
                "stale_sources": report.staleness.stale_sources if report.staleness else [],
            } if report.staleness else None,
            "duplicates": {
                "duplicate_groups": report.duplicates.duplicate_groups if report.duplicates else 0,
                "duplicate_chunks": report.duplicates.duplicate_chunks if report.duplicates else 0,
            } if report.duplicates else None,
            "recommendations": report.recommendations,
        }

    @mcp.tool()
    async def crawl_site(
        url: str,
        depth: int = 2,
        output_format: str = "markdown",
    ) -> dict[str, Any]:
        """Crawl a website and return validated content without
        loading it. Useful for preview/review.

        Args:
            url: Starting URL to crawl
            depth: Crawl depth
            output_format: Output format (markdown or json)
        """
        from gweta.acquire.crawler import GwetaCrawler

        crawler = GwetaCrawler()
        result = await crawler.crawl(url, depth=depth)

        if output_format == "markdown":
            content = "\n\n---\n\n".join(c.text for c in result.chunks)
        else:
            content = [c.to_dict() for c in result.chunks]

        return {
            "url": url,
            "pages_crawled": result.pages_crawled,
            "quality_score": result.quality_score,
            "content": content,
        }

    @mcp.tool()
    async def ingest_from_database(
        dsn: str,
        query: str,
        target_collection: str = "default",
        text_column: str = "content",
        metadata_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Read data from a database, validate it, and load into
        the knowledge base.

        Args:
            dsn: Database connection string
            query: SQL query to extract data
            target_collection: Target vector DB collection
            text_column: Column containing text to embed
            metadata_columns: Columns to preserve as metadata
        """
        from gweta.acquire.database import DatabaseSource
        from gweta.ingest.stores.chroma import ChromaStore

        store = ChromaStore(collection_name=target_collection)

        async with DatabaseSource(dsn) as db:
            chunks = await db.extract_and_validate(
                query=query,
                text_column=text_column,
                metadata_columns=metadata_columns,
            )

            if chunks:
                await store.add(chunks)

        return {
            "source": "database",
            "chunks_loaded": len(chunks),
            "target_collection": target_collection,
        }

    @mcp.tool()
    async def query_database(
        dsn: str,
        query: str,
    ) -> dict[str, Any]:
        """Execute a read-only query against a database.
        Returns results as structured data.

        Args:
            dsn: Database connection string
            query: SQL query (must be read-only)
        """
        from gweta.acquire.database import DatabaseSource

        async with DatabaseSource(dsn, read_only=True) as db:
            result = await db.query(query)

        return {
            "columns": result.columns,
            "rows": result.rows[:100],  # Limit for response size
            "row_count": result.row_count,
            "execution_time": result.execution_time,
        }

    @mcp.tool()
    async def extract_pdf(
        path: str,
        extract_tables: bool = True,
        target_collection: str | None = None,
    ) -> dict[str, Any]:
        """Extract content from a PDF file with quality validation.

        Args:
            path: Path to PDF file
            extract_tables: Whether to extract tables
            target_collection: Optional collection to ingest to

        Returns:
            Extraction results with quality metrics
        """
        from pathlib import Path as PathLib

        from gweta.acquire.pdf import PDFExtractor

        extractor = PDFExtractor()
        result = await extractor.extract(
            source=PathLib(path),
            extract_tables=extract_tables,
            create_chunks=True,
        )

        # Optionally ingest to collection
        if target_collection and result.chunks:
            from gweta.ingest.stores.chroma import ChromaStore
            store = ChromaStore(collection_name=target_collection)
            await store.add(result.chunks)

        return {
            "path": path,
            "total_pages": result.total_pages,
            "tables_extracted": len(result.tables),
            "chunks_created": len(result.chunks),
            "quality_score": result.quality_score,
            "issues": [
                {"code": i.code, "severity": i.severity, "message": i.message}
                for i in result.issues
            ],
            "ingested_to": target_collection,
        }

    @mcp.tool()
    async def fetch_api(
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fetch data from a REST API endpoint.

        Args:
            url: API endpoint URL
            method: HTTP method (GET, POST, etc.)
            headers: Request headers
            params: Query parameters
            json_body: JSON body for POST/PUT

        Returns:
            API response with data and quality metrics
        """
        from gweta.acquire.api import APIClient

        async with APIClient() as client:
            response = await client.fetch(
                url=url,
                method=method,
                headers=headers,
                params=params,
                json=json_body,
            )

        return {
            "url": response.url,
            "status_code": response.status_code,
            "is_success": response.is_success,
            "is_json": response.is_json,
            "quality_score": response.quality_score,
            "data": response.data,
            "issues": [
                {"code": i.code, "severity": i.severity, "message": i.message}
                for i in response.issues
            ],
        }

    @mcp.tool()
    async def fetch_sitemap(
        url: str,
        recursive: bool = True,
        max_urls: int = 1000,
    ) -> dict[str, Any]:
        """Fetch and parse a sitemap to discover URLs.

        Args:
            url: Sitemap URL
            recursive: Follow nested sitemaps
            max_urls: Maximum URLs to return

        Returns:
            List of discovered URLs with metadata
        """
        from gweta.acquire.fetchers.sitemap import SitemapFetcher

        fetcher = SitemapFetcher(max_urls=max_urls)
        result = await fetcher.fetch(url, recursive=recursive)
        await fetcher.close()

        return {
            "url": url,
            "success": result.success,
            "error": result.error,
            "urls_found": len(result.urls),
            "nested_sitemaps": result.nested_sitemaps,
            "urls": [
                {
                    "loc": u.loc,
                    "lastmod": u.lastmod.isoformat() if u.lastmod else None,
                    "priority": u.priority,
                }
                for u in result.urls[:100]  # Limit response size
            ],
        }

    @mcp.tool()
    async def fetch_rss(
        url: str,
        create_chunks: bool = True,
        target_collection: str | None = None,
    ) -> dict[str, Any]:
        """Fetch and parse an RSS/Atom feed.

        Args:
            url: Feed URL
            create_chunks: Convert items to chunks
            target_collection: Optional collection to ingest to

        Returns:
            Feed content with items
        """
        from gweta.acquire.fetchers.rss import RSSFetcher

        fetcher = RSSFetcher()
        result = await fetcher.fetch(url)
        await fetcher.close()

        chunks = fetcher.to_chunks(result) if create_chunks else []

        # Optionally ingest
        if target_collection and chunks:
            from gweta.ingest.stores.chroma import ChromaStore
            store = ChromaStore(collection_name=target_collection)
            await store.add(chunks)

        return {
            "url": url,
            "title": result.title,
            "feed_type": result.feed_type,
            "success": result.success,
            "error": result.error,
            "items_found": len(result.items),
            "chunks_created": len(chunks),
            "ingested_to": target_collection,
            "items": [
                {
                    "title": item.title,
                    "link": item.link,
                    "published": item.published.isoformat() if item.published else None,
                }
                for item in result.items[:20]  # Limit response size
            ],
        }

    logger.info("Registered MCP tools")
