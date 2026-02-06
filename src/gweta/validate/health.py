"""Layer 4: Knowledge base health monitoring.

This module provides ongoing health monitoring for
existing knowledge bases, including staleness detection,
duplicate analysis, and coverage gaps.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from gweta.core.logging import get_logger
from gweta.core.registry import SourceAuthorityRegistry
from gweta.core.types import Chunk
from gweta.validate.detectors.duplicates import DuplicateDetector
from gweta.validate.detectors.staleness import StalenessChecker, StalenessReport

logger = get_logger(__name__)


@dataclass
class DuplicateReport:
    """Report on duplicate content."""
    total_checked: int = 0
    duplicate_groups: int = 0
    duplicate_chunks: int = 0
    groups: list[list[str]] = field(default_factory=list)


@dataclass
class CoverageReport:
    """Report on topic coverage."""
    expected_topics: list[str] = field(default_factory=list)
    covered_topics: list[str] = field(default_factory=list)
    missing_topics: list[str] = field(default_factory=list)
    coverage_ratio: float = 0.0


@dataclass
class HealthReport:
    """Comprehensive health report for a knowledge base.

    Attributes:
        timestamp: When the report was generated
        collection: Collection/index name
        total_chunks: Total chunks in the collection
        avg_quality_score: Average quality score
        staleness: Staleness report
        duplicates: Duplicate report
        coverage: Coverage report (if topics provided)
        golden_results: Golden dataset test results (if provided)
        recommendations: List of actionable recommendations
    """
    timestamp: datetime
    collection: str
    total_chunks: int = 0
    avg_quality_score: float = 0.0
    staleness: StalenessReport | None = None
    duplicates: DuplicateReport | None = None
    coverage: CoverageReport | None = None
    golden_results: Any = None  # GoldenTestReport if available
    recommendations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Knowledge Base Health Report",
            f"Collection: {self.collection}",
            f"Generated: {self.timestamp.isoformat()}",
            f"",
            f"Overview:",
            f"  Total Chunks: {self.total_chunks}",
            f"  Avg Quality Score: {self.avg_quality_score:.2f}",
        ]

        if self.staleness:
            lines.extend([
                f"",
                f"Freshness:",
                f"  Stale Chunks: {self.staleness.stale_chunks}",
                f"  Fresh Chunks: {self.staleness.fresh_chunks}",
                f"  Staleness Ratio: {self.staleness.staleness_ratio:.1%}",
            ])

        if self.duplicates:
            lines.extend([
                f"",
                f"Duplicates:",
                f"  Duplicate Groups: {self.duplicates.duplicate_groups}",
                f"  Duplicate Chunks: {self.duplicates.duplicate_chunks}",
            ])

        if self.coverage:
            lines.extend([
                f"",
                f"Coverage:",
                f"  Expected Topics: {len(self.coverage.expected_topics)}",
                f"  Covered Topics: {len(self.coverage.covered_topics)}",
                f"  Coverage Ratio: {self.coverage.coverage_ratio:.1%}",
            ])

        if self.recommendations:
            lines.extend([
                f"",
                f"Recommendations:",
            ])
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)


class HealthChecker:
    """Layer 4: Ongoing knowledge base health monitoring.

    Performs comprehensive health checks including:
    - Staleness detection
    - Duplicate analysis
    - Coverage gap analysis
    - Quality drift monitoring

    Example:
        >>> checker = HealthChecker(store)
        >>> report = await checker.full_health_check()
        >>> print(report.summary())
    """

    def __init__(
        self,
        store: Any,
        authority_registry: SourceAuthorityRegistry | None = None,
    ) -> None:
        """Initialize HealthChecker.

        Args:
            store: Vector store adapter
            authority_registry: Source authority registry
        """
        self.store = store
        self.registry = authority_registry or SourceAuthorityRegistry.get_default()
        self._staleness_checker = StalenessChecker()
        self._duplicate_detector = DuplicateDetector()

    async def check_staleness(self) -> StalenessReport:
        """Find sources past their freshness window.

        Returns:
            StalenessReport with stale sources
        """
        chunks = await self._get_all_chunks()
        return self._staleness_checker.check_chunks(chunks)

    async def check_duplicates(self) -> DuplicateReport:
        """Find duplicate/near-duplicate chunks.

        Returns:
            DuplicateReport with duplicate groups
        """
        chunks = await self._get_all_chunks()
        report = DuplicateReport(total_checked=len(chunks))

        # Build duplicate index
        self._duplicate_detector.clear()
        for chunk in chunks:
            self._duplicate_detector.add(chunk.id, chunk.text)

        # Get groups
        groups = self._duplicate_detector.get_duplicate_groups()
        report.duplicate_groups = len(groups)
        report.duplicate_chunks = sum(len(g) for g in groups)
        report.groups = groups

        return report

    async def check_coverage(
        self,
        expected_topics: list[str],
    ) -> CoverageReport:
        """Check if expected topics are covered.

        Args:
            expected_topics: List of topics that should be covered

        Returns:
            CoverageReport with coverage analysis
        """
        report = CoverageReport(expected_topics=expected_topics)
        chunks = await self._get_all_chunks()

        # Simple coverage check - look for topic keywords in chunks
        all_text = " ".join(c.text.lower() for c in chunks)

        for topic in expected_topics:
            if topic.lower() in all_text:
                report.covered_topics.append(topic)
            else:
                report.missing_topics.append(topic)

        if expected_topics:
            report.coverage_ratio = len(report.covered_topics) / len(expected_topics)

        return report

    async def check_quality_drift(
        self,
        baseline_score: float,
    ) -> dict[str, Any]:
        """Compare current quality to historical baseline.

        Args:
            baseline_score: Historical quality score to compare against

        Returns:
            Dict with drift analysis
        """
        chunks = await self._get_all_chunks()

        current_scores = [c.quality_score for c in chunks if c.quality_score is not None]
        current_avg = sum(current_scores) / len(current_scores) if current_scores else 0

        drift = current_avg - baseline_score

        return {
            "baseline_score": baseline_score,
            "current_score": current_avg,
            "drift": drift,
            "drift_percentage": (drift / baseline_score * 100) if baseline_score else 0,
            "improved": drift > 0,
        }

    async def full_health_check(
        self,
        golden_dataset: Path | None = None,
        expected_topics: list[str] | None = None,
    ) -> HealthReport:
        """Run comprehensive health check.

        Args:
            golden_dataset: Path to golden dataset for testing
            expected_topics: List of expected topics for coverage

        Returns:
            HealthReport with all checks
        """
        report = HealthReport(
            timestamp=datetime.now(),
            collection=getattr(self.store, "collection_name", "unknown"),
        )

        try:
            chunks = await self._get_all_chunks()
            report.total_chunks = len(chunks)

            # Calculate average quality
            quality_scores = [
                c.quality_score for c in chunks
                if c.quality_score is not None
            ]
            if quality_scores:
                report.avg_quality_score = sum(quality_scores) / len(quality_scores)

            # Run checks
            report.staleness = await self.check_staleness()
            report.duplicates = await self.check_duplicates()

            if expected_topics:
                report.coverage = await self.check_coverage(expected_topics)

            # Golden dataset testing
            if golden_dataset:
                from gweta.validate.golden import GoldenDatasetRunner
                runner = GoldenDatasetRunner(self.store, golden_dataset)
                report.golden_results = await runner.run()

            # Generate recommendations
            report.recommendations = self._generate_recommendations(report)

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            report.recommendations.append(f"Health check error: {e}")

        return report

    async def _get_all_chunks(self) -> list[Chunk]:
        """Get all chunks from the store."""
        if hasattr(self.store, "get_all"):
            return await self.store.get_all()
        return []

    def _generate_recommendations(self, report: HealthReport) -> list[str]:
        """Generate actionable recommendations from report."""
        recommendations: list[str] = []

        # Staleness recommendations
        if report.staleness and report.staleness.staleness_ratio > 0.3:
            recommendations.append(
                f"Re-crawl stale sources: {', '.join(report.staleness.stale_sources[:5])}"
            )

        # Duplicate recommendations
        if report.duplicates and report.duplicates.duplicate_groups > 0:
            recommendations.append(
                f"Remove {report.duplicates.duplicate_chunks} duplicate chunks"
            )

        # Coverage recommendations
        if report.coverage and report.coverage.missing_topics:
            missing = ", ".join(report.coverage.missing_topics[:5])
            recommendations.append(f"Add content for missing topics: {missing}")

        # Quality recommendations
        if report.avg_quality_score < 0.6:
            recommendations.append(
                "Overall quality is low - review extraction and validation settings"
            )

        return recommendations
