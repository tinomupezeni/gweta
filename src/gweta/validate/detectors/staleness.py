"""Staleness and freshness tracking.

This module tracks content freshness and detects
stale data that needs refreshing.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from gweta.core.types import Chunk, Source


@dataclass
class StalenessReport:
    """Report on content staleness.

    Attributes:
        total_chunks: Total chunks analyzed
        stale_chunks: Number of stale chunks
        fresh_chunks: Number of fresh chunks
        unknown_chunks: Chunks without freshness info
        stale_sources: Sources that are stale
        oldest_content: Oldest content date found
    """
    total_chunks: int = 0
    stale_chunks: int = 0
    fresh_chunks: int = 0
    unknown_chunks: int = 0
    stale_sources: list[str] = field(default_factory=list)
    oldest_content: datetime | None = None

    @property
    def staleness_ratio(self) -> float:
        """Calculate ratio of stale content."""
        if self.total_chunks == 0:
            return 0.0
        return self.stale_chunks / self.total_chunks


class StalenessChecker:
    """Check content freshness and detect stale data.

    Tracks when content was last updated and flags
    content that exceeds freshness thresholds.

    Example:
        >>> checker = StalenessChecker(default_freshness_days=90)
        >>> report = checker.check_chunks(chunks)
        >>> for source in report.stale_sources:
        ...     print(f"Stale: {source}")
    """

    def __init__(
        self,
        default_freshness_days: int = 90,
        source_freshness: dict[str, int] | None = None,
    ) -> None:
        """Initialize StalenessChecker.

        Args:
            default_freshness_days: Default freshness window in days
            source_freshness: Per-source freshness windows
        """
        self.default_freshness_days = default_freshness_days
        self.source_freshness = source_freshness or {}

    def get_freshness_window(self, source: str) -> timedelta:
        """Get freshness window for a source.

        Args:
            source: Source identifier

        Returns:
            Timedelta for freshness window
        """
        days = self.source_freshness.get(source, self.default_freshness_days)
        return timedelta(days=days)

    def is_stale(
        self,
        chunk: Chunk,
        reference_date: datetime | None = None,
    ) -> bool:
        """Check if a chunk is stale.

        Args:
            chunk: Chunk to check
            reference_date: Date to compare against (default: now)

        Returns:
            True if chunk is stale
        """
        reference_date = reference_date or datetime.now()

        # Try to get content date from metadata
        content_date = self._extract_date(chunk)
        if content_date is None:
            # Use chunk creation date as fallback
            content_date = chunk.created_at

        freshness = self.get_freshness_window(chunk.source)
        age = reference_date - content_date

        return age > freshness

    def _extract_date(self, chunk: Chunk) -> datetime | None:
        """Extract content date from chunk metadata.

        Looks for common date fields in metadata.
        """
        date_fields = [
            "crawled_at",
            "published",
            "date",
            "last_modified",
            "updated_at",
            "created_at",
        ]

        for field in date_fields:
            value = chunk.metadata.get(field)
            if value:
                if isinstance(value, datetime):
                    return value
                if isinstance(value, str):
                    try:
                        return datetime.fromisoformat(value.replace("Z", "+00:00"))
                    except ValueError:
                        continue
        return None

    def check_chunks(
        self,
        chunks: list[Chunk],
        reference_date: datetime | None = None,
    ) -> StalenessReport:
        """Check staleness of multiple chunks.

        Args:
            chunks: Chunks to check
            reference_date: Date to compare against

        Returns:
            StalenessReport with results
        """
        reference_date = reference_date or datetime.now()
        report = StalenessReport(total_chunks=len(chunks))

        stale_sources: set[str] = set()
        oldest: datetime | None = None

        for chunk in chunks:
            content_date = self._extract_date(chunk)

            if content_date is None:
                report.unknown_chunks += 1
                continue

            if oldest is None or content_date < oldest:
                oldest = content_date

            if self.is_stale(chunk, reference_date):
                report.stale_chunks += 1
                stale_sources.add(chunk.source)
            else:
                report.fresh_chunks += 1

        report.stale_sources = list(stale_sources)
        report.oldest_content = oldest

        return report

    def check_sources(
        self,
        sources: list[Source],
        reference_date: datetime | None = None,
    ) -> list[Source]:
        """Get list of stale sources.

        Args:
            sources: Sources to check
            reference_date: Date to compare against

        Returns:
            List of stale Source objects
        """
        reference_date = reference_date or datetime.now()
        stale: list[Source] = []

        for source in sources:
            if source.is_stale:
                stale.append(source)

        return stale
