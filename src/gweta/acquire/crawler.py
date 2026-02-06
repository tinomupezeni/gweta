"""Quality-aware web crawler wrapping Crawl4AI.

This module provides the GwetaCrawler class that wraps Crawl4AI
with pre-crawl validation, post-crawl quality scoring, and
integration with the source authority registry.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from gweta.core.config import GwetaSettings, get_settings
from gweta.core.exceptions import CrawlError
from gweta.core.logging import get_logger
from gweta.core.registry import SourceAuthorityRegistry
from gweta.core.types import Chunk, QualityIssue

logger = get_logger(__name__)


@dataclass
class CrawlError_Result:
    """Details about a crawl error."""
    url: str
    error: str
    status_code: int | None = None


@dataclass
class CrawlResult:
    """Results from a crawl operation.

    Attributes:
        url: Starting URL that was crawled
        pages_crawled: Total number of pages attempted
        pages_passed: Pages that passed quality validation
        pages_failed: Pages that failed quality validation
        quality_score: Average quality score across all pages
        chunks: List of validated chunks ready for ingestion
        rejected_chunks: Chunks that failed validation (for review)
        errors: List of crawl errors encountered
        duration_seconds: Total crawl duration
    """
    url: str
    pages_crawled: int = 0
    pages_passed: int = 0
    pages_failed: int = 0
    quality_score: float = 0.0
    chunks: list[Chunk] = field(default_factory=list)
    rejected_chunks: list[Chunk] = field(default_factory=list)
    errors: list[CrawlError_Result] = field(default_factory=list)
    duration_seconds: float = 0.0

    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"Crawl Results for {self.url}\n"
            f"  Pages: {self.pages_crawled} crawled, "
            f"{self.pages_passed} passed, {self.pages_failed} failed\n"
            f"  Quality Score: {self.quality_score:.2f}\n"
            f"  Chunks: {len(self.chunks)} accepted, "
            f"{len(self.rejected_chunks)} rejected\n"
            f"  Duration: {self.duration_seconds:.1f}s"
        )

    async def load_to(self, store: Any) -> None:
        """Load validated chunks to a vector store.

        Args:
            store: Vector store adapter with add() method
        """
        if self.chunks:
            await store.add(self.chunks)


@dataclass
class PageQualityScore:
    """Quality score for a single crawled page."""
    url: str
    extraction_score: float = 1.0
    content_completeness: float = 1.0
    javascript_rendered: bool = False
    has_main_content: bool = True
    boilerplate_ratio: float = 0.0
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Check if page passed quality validation."""
        return not any(i.severity == "error" for i in self.issues)

    @property
    def overall_score(self) -> float:
        """Calculate overall quality score."""
        return (
            self.extraction_score * 0.4
            + self.content_completeness * 0.3
            + (1.0 - self.boilerplate_ratio) * 0.3
        )


class GwetaCrawler:
    """Quality-aware web crawler wrapping Crawl4AI.

    Provides crawling with:
    - Pre-crawl validation (authority registry check)
    - Post-crawl quality scoring
    - Automatic chunking with quality metadata
    - PDF follow and extraction

    Example:
        >>> crawler = GwetaCrawler(quality_threshold=0.6)
        >>> result = await crawler.crawl("https://example.com", depth=2)
        >>> print(result.summary())
        >>> await result.load_to(chroma_store)
    """

    def __init__(
        self,
        config: GwetaSettings | None = None,
        authority_registry: SourceAuthorityRegistry | Path | str | None = None,
        quality_threshold: float | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """Initialize GwetaCrawler.

        Args:
            config: Gweta settings (uses defaults if not provided)
            authority_registry: Registry for source authority (path or instance)
            quality_threshold: Minimum quality score to accept chunks
            chunk_size: Size of chunks in characters
            chunk_overlap: Overlap between chunks
        """
        self.config = config or get_settings()
        self.quality_threshold = quality_threshold or self.config.min_quality_score
        self.chunk_size = chunk_size or self.config.default_chunk_size
        self.chunk_overlap = chunk_overlap or self.config.default_chunk_overlap

        # Load authority registry
        if isinstance(authority_registry, SourceAuthorityRegistry):
            self.registry = authority_registry
        elif isinstance(authority_registry, (Path, str)):
            self.registry = SourceAuthorityRegistry.from_yaml(Path(authority_registry))
        else:
            self.registry = SourceAuthorityRegistry.get_default()

        self._crawler = None

    async def _get_crawler(self) -> Any:
        """Lazy-load the Crawl4AI crawler."""
        if self._crawler is None:
            try:
                from crawl4ai import AsyncWebCrawler
                self._crawler = AsyncWebCrawler()
            except ImportError as e:
                raise ImportError(
                    "Crawl4AI is required for web crawling. "
                    "Install it with: pip install crawl4ai"
                ) from e
        return self._crawler

    async def crawl(
        self,
        url: str,
        depth: int = 2,
        follow_pdfs: bool = True,
        allowed_domains: list[str] | None = None,
        rules: str | None = None,
        max_pages: int = 100,
    ) -> CrawlResult:
        """Crawl a website with quality validation.

        Args:
            url: Starting URL to crawl
            depth: How many links deep to follow (1-5)
            follow_pdfs: Whether to download and extract linked PDFs
            allowed_domains: List of domains to restrict crawling to
            rules: Optional domain rule set name for validation
            max_pages: Maximum number of pages to crawl

        Returns:
            CrawlResult with validated chunks and quality metrics

        Raises:
            CrawlError: If crawling fails
            ValueError: If URL is blocked by authority registry
        """
        start_time = datetime.now()

        # Pre-crawl validation
        if not self.registry.is_allowed(url):
            raise CrawlError(
                f"URL {url} is blocked by authority registry",
                url=url,
            )

        # Clamp depth
        depth = max(1, min(depth, self.config.max_crawl_depth))

        result = CrawlResult(url=url)
        crawled_urls: set[str] = set()

        try:
            crawler = await self._get_crawler()

            # For now, implement basic single-page crawl
            # Full recursive crawling will be implemented in Phase 2
            await self._crawl_page(
                crawler=crawler,
                url=url,
                result=result,
                crawled_urls=crawled_urls,
                depth=depth,
                current_depth=0,
                max_pages=max_pages,
                follow_pdfs=follow_pdfs,
                allowed_domains=allowed_domains,
            )

        except Exception as e:
            logger.error(f"Crawl failed for {url}: {e}")
            result.errors.append(CrawlError_Result(url=url, error=str(e)))

        # Calculate final metrics
        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        if result.chunks:
            total_score = sum(c.quality_score or 0 for c in result.chunks)
            result.quality_score = total_score / len(result.chunks)

        logger.info(f"Crawl completed: {result.summary()}")
        return result

    async def _crawl_page(
        self,
        crawler: Any,
        url: str,
        result: CrawlResult,
        crawled_urls: set[str],
        depth: int,
        current_depth: int,
        max_pages: int,
        follow_pdfs: bool,
        allowed_domains: list[str] | None,
    ) -> None:
        """Crawl a single page and extract content.

        This is a placeholder implementation. Full recursive crawling
        with link extraction will be implemented in Phase 2.
        """
        if url in crawled_urls:
            return
        if len(crawled_urls) >= max_pages:
            return
        if current_depth > depth:
            return

        crawled_urls.add(url)
        result.pages_crawled += 1

        try:
            # Use Crawl4AI to fetch and extract content
            crawl_result = await crawler.arun(url=url)

            if crawl_result.success:
                # Score the page quality
                page_score = self._score_page(url, crawl_result.markdown)

                if page_score.passed:
                    result.pages_passed += 1

                    # Create chunks from the content
                    chunks = self._create_chunks(
                        text=crawl_result.markdown,
                        url=url,
                        page_score=page_score,
                    )

                    # Separate accepted and rejected chunks
                    for chunk in chunks:
                        if (chunk.quality_score or 0) >= self.quality_threshold:
                            result.chunks.append(chunk)
                        else:
                            result.rejected_chunks.append(chunk)
                else:
                    result.pages_failed += 1
                    logger.warning(f"Page failed quality check: {url}")
            else:
                result.pages_failed += 1
                result.errors.append(
                    CrawlError_Result(
                        url=url,
                        error=crawl_result.error_message or "Unknown error",
                    )
                )

        except Exception as e:
            result.pages_failed += 1
            result.errors.append(CrawlError_Result(url=url, error=str(e)))
            logger.error(f"Failed to crawl {url}: {e}")

    def _score_page(self, url: str, content: str) -> PageQualityScore:
        """Score the quality of a crawled page.

        Args:
            url: Page URL
            content: Extracted markdown content

        Returns:
            PageQualityScore with quality metrics
        """
        issues: list[QualityIssue] = []
        score = PageQualityScore(url=url)

        # Check content length
        if len(content) < self.config.min_text_length:
            issues.append(
                QualityIssue(
                    code="CONTENT_TOO_SHORT",
                    severity="error",
                    message=f"Content length ({len(content)}) below minimum ({self.config.min_text_length})",
                )
            )
            score.content_completeness = 0.3

        # Estimate boilerplate ratio (simplified)
        lines = content.split("\n")
        short_lines = sum(1 for line in lines if len(line.strip()) < 20)
        if lines:
            score.boilerplate_ratio = short_lines / len(lines)

        if score.boilerplate_ratio > 0.7:
            issues.append(
                QualityIssue(
                    code="HIGH_BOILERPLATE",
                    severity="warning",
                    message=f"High boilerplate ratio: {score.boilerplate_ratio:.1%}",
                )
            )

        score.issues = issues
        return score

    def _create_chunks(
        self,
        text: str,
        url: str,
        page_score: PageQualityScore,
    ) -> list[Chunk]:
        """Create chunks from page content.

        Args:
            text: Page content as markdown
            url: Source URL
            page_score: Quality score for the page

        Returns:
            List of Chunk objects with quality metadata
        """
        # Simple recursive splitting for now
        # Will integrate with proper chunker in Phase 1 Week 3
        chunks: list[Chunk] = []

        # Split by paragraphs first, then by size
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, url, page_score))
                current_chunk = para
            else:
                current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para

        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, url, page_score))

        return chunks

    def _create_chunk(
        self,
        text: str,
        url: str,
        page_score: PageQualityScore,
    ) -> Chunk:
        """Create a single chunk with metadata.

        Args:
            text: Chunk text content
            url: Source URL
            page_score: Quality score from parent page

        Returns:
            Chunk with quality metadata
        """
        source_info = self.registry.get_source(url)

        return Chunk(
            text=text.strip(),
            source=url,
            quality_score=page_score.overall_score,
            metadata={
                "source_url": url,
                "authority_tier": source_info.authority_tier,
                "source_name": source_info.name,
                "crawled_at": datetime.now().isoformat(),
            },
        )

    def crawl_sync(
        self,
        url: str,
        **kwargs: Any,
    ) -> CrawlResult:
        """Synchronous wrapper for crawl().

        Args:
            url: Starting URL to crawl
            **kwargs: Additional arguments passed to crawl()

        Returns:
            CrawlResult with validated chunks
        """
        return asyncio.run(self.crawl(url, **kwargs))
