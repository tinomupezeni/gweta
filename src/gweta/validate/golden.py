"""Golden dataset testing.

This module provides testing against golden Q&A pairs
to verify knowledge base retrieval quality.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gweta.core.exceptions import ConfigurationError
from gweta.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GoldenPair:
    """A single Q&A test case.

    Attributes:
        id: Unique identifier
        question: Test question
        expected_answer: Expected answer content
        expected_sources: Source IDs that should be retrieved
        tags: Tags for categorizing tests
    """
    id: str
    question: str
    expected_answer: str
    expected_sources: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class FailedTest:
    """Details about a failed golden test.

    Attributes:
        pair: The golden pair that failed
        retrieved_sources: Sources that were actually retrieved
        reason: Why the test failed
    """
    pair: GoldenPair
    retrieved_sources: list[str]
    reason: str


@dataclass
class GoldenTestReport:
    """Results from golden dataset testing.

    Attributes:
        total_tests: Total number of tests run
        passed: Number of tests that passed
        failed: Number of tests that failed
        retrieval_accuracy: Percentage with correct sources in top-k
        mrr: Mean Reciprocal Rank
        precision_at_k: Precision at different k values
        failed_tests: List of failed test details
        coverage_gaps: Topics with no matching chunks
    """
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    retrieval_accuracy: float = 0.0
    mrr: float = 0.0
    precision_at_k: dict[int, float] = field(default_factory=dict)
    failed_tests: list[FailedTest] = field(default_factory=list)
    coverage_gaps: list[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if self.total_tests == 0:
            return 0.0
        return self.passed / self.total_tests

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Golden Dataset Test Results",
            f"  Total Tests: {self.total_tests}",
            f"  Passed: {self.passed}",
            f"  Failed: {self.failed}",
            f"  Pass Rate: {self.pass_rate:.1%}",
            f"  Retrieval Accuracy: {self.retrieval_accuracy:.1%}",
            f"  MRR: {self.mrr:.3f}",
        ]

        if self.precision_at_k:
            lines.append("  Precision@K:")
            for k, p in sorted(self.precision_at_k.items()):
                lines.append(f"    @{k}: {p:.1%}")

        if self.coverage_gaps:
            lines.append(f"  Coverage Gaps: {len(self.coverage_gaps)}")

        return "\n".join(lines)


class GoldenDatasetRunner:
    """Test knowledge base against golden Q&A pairs.

    Runs retrieval tests to verify that:
    - Expected sources are retrieved for questions
    - Retrieval ranking is reasonable (MRR)
    - Coverage exists for all tested topics

    Example:
        >>> runner = GoldenDatasetRunner(store, "golden/business.json")
        >>> report = await runner.run()
        >>> print(report.summary())
    """

    def __init__(
        self,
        store: Any,
        dataset_path: Path | str | None = None,
    ) -> None:
        """Initialize GoldenDatasetRunner.

        Args:
            store: Vector store adapter with query() method
            dataset_path: Path to golden dataset JSON file
        """
        self.store = store
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.pairs: list[GoldenPair] = []

        if self.dataset_path:
            self.load_dataset(self.dataset_path)

    def load_dataset(self, path: Path | str) -> list[GoldenPair]:
        """Load golden dataset from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            List of GoldenPair objects
        """
        path = Path(path)
        if not path.exists():
            raise ConfigurationError(
                f"Golden dataset not found: {path}",
                setting_name="golden_dataset",
            )

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.pairs = []
        for pair_data in data.get("pairs", []):
            self.pairs.append(
                GoldenPair(
                    id=pair_data.get("id", ""),
                    question=pair_data.get("question", ""),
                    expected_answer=pair_data.get("expected_answer", ""),
                    expected_sources=pair_data.get("expected_sources", []),
                    tags=pair_data.get("tags", []),
                )
            )

        logger.info(f"Loaded {len(self.pairs)} golden pairs from {path}")
        return self.pairs

    async def run(
        self,
        k: int = 5,
        similarity_threshold: float = 0.7,
    ) -> GoldenTestReport:
        """Run all golden tests.

        Args:
            k: Number of results to retrieve
            similarity_threshold: Minimum similarity for a match

        Returns:
            GoldenTestReport with results
        """
        report = GoldenTestReport(total_tests=len(self.pairs))
        reciprocal_ranks: list[float] = []

        for pair in self.pairs:
            try:
                result = await self._test_pair(pair, k)

                if result["passed"]:
                    report.passed += 1
                else:
                    report.failed += 1
                    report.failed_tests.append(
                        FailedTest(
                            pair=pair,
                            retrieved_sources=result["retrieved_sources"],
                            reason=result["reason"],
                        )
                    )

                # Track MRR
                if result["rank"]:
                    reciprocal_ranks.append(1.0 / result["rank"])
                else:
                    reciprocal_ranks.append(0.0)

            except Exception as e:
                logger.error(f"Error testing pair {pair.id}: {e}")
                report.failed += 1

        # Calculate metrics
        if reciprocal_ranks:
            report.mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

        if report.total_tests > 0:
            report.retrieval_accuracy = report.passed / report.total_tests

        # Calculate precision@k
        for test_k in [1, 3, 5, 10]:
            if test_k <= k:
                report.precision_at_k[test_k] = await self._calculate_precision_at_k(test_k)

        return report

    async def _test_pair(
        self,
        pair: GoldenPair,
        k: int,
    ) -> dict[str, Any]:
        """Test a single golden pair.

        Args:
            pair: Golden pair to test
            k: Number of results

        Returns:
            Dict with test results
        """
        # Query the store
        results = await self.store.query(pair.question, n_results=k)

        # Extract source IDs from results
        retrieved_sources = []
        for chunk in results:
            source = chunk.source or chunk.metadata.get("source", "")
            retrieved_sources.append(source)

        # Check if expected sources are in results
        expected_set = set(pair.expected_sources)
        retrieved_set = set(retrieved_sources)

        # Find rank of first expected source
        rank = None
        for i, source in enumerate(retrieved_sources, 1):
            if source in expected_set:
                rank = i
                break

        # Determine pass/fail
        has_expected = bool(expected_set & retrieved_set)
        passed = has_expected or not pair.expected_sources

        reason = ""
        if not passed:
            reason = f"Expected sources {pair.expected_sources} not in retrieved {retrieved_sources}"

        return {
            "passed": passed,
            "rank": rank,
            "retrieved_sources": retrieved_sources,
            "reason": reason,
        }

    async def _calculate_precision_at_k(self, k: int) -> float:
        """Calculate precision at k across all pairs."""
        total_relevant = 0
        total_retrieved = 0

        for pair in self.pairs:
            try:
                results = await self.store.query(pair.question, n_results=k)
                retrieved_sources = {
                    c.source or c.metadata.get("source", "")
                    for c in results
                }
                expected_set = set(pair.expected_sources)

                relevant = len(retrieved_sources & expected_set)
                total_relevant += relevant
                total_retrieved += len(results)

            except Exception:
                continue

        if total_retrieved == 0:
            return 0.0
        return total_relevant / total_retrieved

    def to_junit_xml(self, report: GoldenTestReport) -> str:
        """Export results as JUnit XML for CI/CD.

        Args:
            report: Test report to export

        Returns:
            JUnit XML string
        """
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<testsuite name="GoldenDataset" tests="{report.total_tests}" '
            f'failures="{report.failed}" errors="0">',
        ]

        for pair in self.pairs:
            failed = any(f.pair.id == pair.id for f in report.failed_tests)
            if failed:
                failure = next(f for f in report.failed_tests if f.pair.id == pair.id)
                lines.append(
                    f'  <testcase name="{pair.id}" classname="golden">'
                )
                lines.append(
                    f'    <failure message="{failure.reason}"/>'
                )
                lines.append('  </testcase>')
            else:
                lines.append(
                    f'  <testcase name="{pair.id}" classname="golden"/>'
                )

        lines.append('</testsuite>')
        return "\n".join(lines)

    def to_json(self, report: GoldenTestReport) -> str:
        """Export results as JSON.

        Args:
            report: Test report to export

        Returns:
            JSON string
        """
        return json.dumps(
            {
                "total_tests": report.total_tests,
                "passed": report.passed,
                "failed": report.failed,
                "pass_rate": report.pass_rate,
                "retrieval_accuracy": report.retrieval_accuracy,
                "mrr": report.mrr,
                "precision_at_k": report.precision_at_k,
                "failed_tests": [
                    {
                        "id": f.pair.id,
                        "question": f.pair.question,
                        "reason": f.reason,
                    }
                    for f in report.failed_tests
                ],
            },
            indent=2,
        )
