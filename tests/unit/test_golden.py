"""Tests for Golden Dataset Testing.

These tests verify the GoldenDatasetRunner functionality including:
- Loading golden datasets from JSON
- Running retrieval tests
- Calculating MRR and precision@k
- Exporting results to JUnit XML and JSON
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

from gweta.core.types import Chunk
from gweta.validate.golden import (
    GoldenDatasetRunner,
    GoldenPair,
    GoldenTestReport,
    FailedTest,
)


class TestGoldenPairDataclass:
    """Tests for GoldenPair dataclass."""

    def test_golden_pair_creation(self):
        """Test GoldenPair creation."""
        pair = GoldenPair(
            id="test-1",
            question="What is the capital?",
            expected_answer="Harare is the capital.",
            expected_sources=["geography.txt"],
            tags=["geography", "zimbabwe"],
        )

        assert pair.id == "test-1"
        assert pair.question == "What is the capital?"
        assert len(pair.expected_sources) == 1
        assert len(pair.tags) == 2

    def test_golden_pair_defaults(self):
        """Test GoldenPair default values."""
        pair = GoldenPair(
            id="test",
            question="Question?",
            expected_answer="Answer",
        )

        assert pair.expected_sources == []
        assert pair.tags == []


class TestGoldenTestReport:
    """Tests for GoldenTestReport dataclass."""

    def test_report_defaults(self):
        """Test report default values."""
        report = GoldenTestReport()

        assert report.total_tests == 0
        assert report.passed == 0
        assert report.failed == 0
        assert report.mrr == 0.0
        assert report.precision_at_k == {}
        assert report.failed_tests == []
        assert report.coverage_gaps == []

    def test_pass_rate_calculation(self):
        """Test pass rate calculation."""
        report = GoldenTestReport(total_tests=10, passed=8, failed=2)

        assert report.pass_rate == 0.8

    def test_pass_rate_zero_tests(self):
        """Test pass rate with zero tests."""
        report = GoldenTestReport()

        assert report.pass_rate == 0.0

    def test_summary_output(self):
        """Test summary string generation."""
        report = GoldenTestReport(
            total_tests=10,
            passed=8,
            failed=2,
            retrieval_accuracy=0.8,
            mrr=0.75,
            precision_at_k={1: 0.6, 3: 0.7, 5: 0.8},
        )

        summary = report.summary()

        assert "Total Tests: 10" in summary
        assert "Passed: 8" in summary
        assert "MRR: 0.750" in summary
        assert "@1:" in summary
        assert "@3:" in summary


class TestFailedTest:
    """Tests for FailedTest dataclass."""

    def test_failed_test_creation(self):
        """Test FailedTest creation."""
        pair = GoldenPair(
            id="test-1",
            question="Question?",
            expected_answer="Answer",
            expected_sources=["source1.txt"],
        )

        failed = FailedTest(
            pair=pair,
            retrieved_sources=["wrong.txt"],
            reason="Expected source not found",
        )

        assert failed.pair.id == "test-1"
        assert "wrong.txt" in failed.retrieved_sources
        assert "Expected" in failed.reason


class TestGoldenDatasetRunnerInit:
    """Tests for GoldenDatasetRunner initialization."""

    def test_init_without_dataset(self):
        """Test initialization without dataset."""
        mock_store = MagicMock()
        runner = GoldenDatasetRunner(store=mock_store)

        assert runner.store == mock_store
        assert runner.pairs == []
        assert runner.dataset_path is None

    def test_init_with_dataset_file_not_found(self):
        """Test initialization with missing dataset file."""
        from gweta.core.exceptions import ConfigurationError

        mock_store = MagicMock()

        with pytest.raises(ConfigurationError) as exc_info:
            GoldenDatasetRunner(
                store=mock_store,
                dataset_path="/nonexistent/dataset.json",
            )

        assert "not found" in str(exc_info.value)


class TestGoldenDatasetLoading:
    """Tests for loading golden datasets."""

    def test_load_dataset(self, tmp_path):
        """Test loading dataset from JSON file."""
        dataset = {
            "pairs": [
                {
                    "id": "q1",
                    "question": "What is 2+2?",
                    "expected_answer": "4",
                    "expected_sources": ["math.txt"],
                    "tags": ["math"],
                },
                {
                    "id": "q2",
                    "question": "What color is the sky?",
                    "expected_answer": "Blue",
                    "expected_sources": ["science.txt"],
                },
            ]
        }

        dataset_file = tmp_path / "golden.json"
        dataset_file.write_text(json.dumps(dataset))

        mock_store = MagicMock()
        runner = GoldenDatasetRunner(store=mock_store)
        pairs = runner.load_dataset(dataset_file)

        assert len(pairs) == 2
        assert pairs[0].id == "q1"
        assert pairs[1].expected_sources == ["science.txt"]

    def test_load_empty_dataset(self, tmp_path):
        """Test loading empty dataset."""
        dataset = {"pairs": []}
        dataset_file = tmp_path / "empty.json"
        dataset_file.write_text(json.dumps(dataset))

        mock_store = MagicMock()
        runner = GoldenDatasetRunner(store=mock_store)
        pairs = runner.load_dataset(dataset_file)

        assert pairs == []


class TestGoldenDatasetRunning:
    """Tests for running golden dataset tests."""

    @pytest.mark.asyncio
    async def test_run_all_pass(self):
        """Test running tests where all pass."""
        # Create mock store
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(
            return_value=[
                Chunk(
                    id="1",
                    text="The answer is here",
                    source="correct.txt",
                    metadata={},
                )
            ]
        )

        # Create runner with test pairs
        runner = GoldenDatasetRunner(store=mock_store)
        runner.pairs = [
            GoldenPair(
                id="q1",
                question="Test question?",
                expected_answer="Test answer",
                expected_sources=["correct.txt"],
            ),
        ]

        report = await runner.run(k=5)

        assert report.total_tests == 1
        assert report.passed == 1
        assert report.failed == 0
        assert report.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_run_with_failures(self):
        """Test running tests with failures."""
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(
            return_value=[
                Chunk(
                    id="1",
                    text="Wrong content",
                    source="wrong.txt",
                    metadata={},
                )
            ]
        )

        runner = GoldenDatasetRunner(store=mock_store)
        runner.pairs = [
            GoldenPair(
                id="q1",
                question="Test?",
                expected_answer="Answer",
                expected_sources=["correct.txt"],
            ),
        ]

        report = await runner.run(k=5)

        assert report.total_tests == 1
        assert report.failed == 1
        assert len(report.failed_tests) == 1
        assert report.failed_tests[0].pair.id == "q1"

    @pytest.mark.asyncio
    async def test_run_mrr_calculation(self):
        """Test MRR calculation."""
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(
            return_value=[
                Chunk(id="1", text="A", source="other.txt", metadata={}),
                Chunk(id="2", text="B", source="target.txt", metadata={}),  # Rank 2
            ]
        )

        runner = GoldenDatasetRunner(store=mock_store)
        runner.pairs = [
            GoldenPair(
                id="q1",
                question="Test?",
                expected_answer="Answer",
                expected_sources=["target.txt"],
            ),
        ]

        report = await runner.run(k=5)

        # MRR should be 1/2 = 0.5 (target is at rank 2)
        assert report.mrr == 0.5

    @pytest.mark.asyncio
    async def test_run_empty_expected_sources(self):
        """Test pair with no expected sources passes."""
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(
            return_value=[
                Chunk(id="1", text="Content", source="any.txt", metadata={})
            ]
        )

        runner = GoldenDatasetRunner(store=mock_store)
        runner.pairs = [
            GoldenPair(
                id="q1",
                question="Test?",
                expected_answer="Answer",
                expected_sources=[],  # No expected sources
            ),
        ]

        report = await runner.run()

        assert report.passed == 1

    @pytest.mark.asyncio
    async def test_precision_at_k(self):
        """Test precision@k calculation."""
        mock_store = AsyncMock()

        # Return 3 chunks, 2 are relevant
        mock_store.query = AsyncMock(
            return_value=[
                Chunk(id="1", text="A", source="relevant1.txt", metadata={}),
                Chunk(id="2", text="B", source="irrelevant.txt", metadata={}),
                Chunk(id="3", text="C", source="relevant2.txt", metadata={}),
            ]
        )

        runner = GoldenDatasetRunner(store=mock_store)
        runner.pairs = [
            GoldenPair(
                id="q1",
                question="Test?",
                expected_answer="Answer",
                expected_sources=["relevant1.txt", "relevant2.txt"],
            ),
        ]

        report = await runner.run(k=5)

        # Precision@3 = 2/3 for this query
        assert 1 in report.precision_at_k or 3 in report.precision_at_k


class TestJUnitXMLExport:
    """Tests for JUnit XML export."""

    def test_junit_xml_all_pass(self):
        """Test JUnit XML export with all passing."""
        mock_store = MagicMock()
        runner = GoldenDatasetRunner(store=mock_store)
        runner.pairs = [
            GoldenPair(
                id="q1",
                question="Test?",
                expected_answer="Answer",
            ),
        ]

        report = GoldenTestReport(total_tests=1, passed=1, failed=0)

        xml = runner.to_junit_xml(report)

        assert '<?xml version="1.0"' in xml
        assert 'testsuite name="GoldenDataset"' in xml
        assert 'tests="1"' in xml
        assert 'failures="0"' in xml
        assert 'testcase name="q1"' in xml
        assert "<failure" not in xml

    def test_junit_xml_with_failures(self):
        """Test JUnit XML export with failures."""
        mock_store = MagicMock()
        runner = GoldenDatasetRunner(store=mock_store)

        pair = GoldenPair(
            id="q1",
            question="Test?",
            expected_answer="Answer",
            expected_sources=["correct.txt"],
        )
        runner.pairs = [pair]

        report = GoldenTestReport(
            total_tests=1,
            passed=0,
            failed=1,
            failed_tests=[
                FailedTest(
                    pair=pair,
                    retrieved_sources=["wrong.txt"],
                    reason="Source mismatch",
                )
            ],
        )

        xml = runner.to_junit_xml(report)

        assert 'failures="1"' in xml
        assert "<failure" in xml
        assert "Source mismatch" in xml


class TestJSONExport:
    """Tests for JSON export."""

    def test_json_export(self):
        """Test JSON export format."""
        mock_store = MagicMock()
        runner = GoldenDatasetRunner(store=mock_store)

        pair = GoldenPair(
            id="q1",
            question="Test?",
            expected_answer="Answer",
        )
        runner.pairs = [pair]

        report = GoldenTestReport(
            total_tests=2,
            passed=1,
            failed=1,
            retrieval_accuracy=0.5,
            mrr=0.75,
            precision_at_k={1: 0.5, 3: 0.6},
            failed_tests=[
                FailedTest(
                    pair=pair,
                    retrieved_sources=["wrong.txt"],
                    reason="Mismatch",
                )
            ],
        )

        json_str = runner.to_json(report)
        data = json.loads(json_str)

        assert data["total_tests"] == 2
        assert data["passed"] == 1
        assert data["pass_rate"] == 0.5
        assert data["mrr"] == 0.75
        assert data["precision_at_k"]["1"] == 0.5
        assert len(data["failed_tests"]) == 1
        assert data["failed_tests"][0]["id"] == "q1"


class TestGoldenDatasetIntegration:
    """Integration tests for golden dataset functionality."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, tmp_path):
        """Test complete golden dataset workflow."""
        # Create dataset file
        dataset = {
            "pairs": [
                {
                    "id": "geography-1",
                    "question": "What is the capital of Zimbabwe?",
                    "expected_answer": "Harare",
                    "expected_sources": ["geo.txt", "facts.txt"],
                    "tags": ["geography"],
                },
                {
                    "id": "economics-1",
                    "question": "What is the currency?",
                    "expected_answer": "ZWL",
                    "expected_sources": ["finance.txt"],
                    "tags": ["economics"],
                },
            ]
        }

        dataset_file = tmp_path / "golden.json"
        dataset_file.write_text(json.dumps(dataset))

        # Create mock store
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(
            side_effect=[
                # First query returns correct source
                [Chunk(id="1", text="Harare", source="geo.txt", metadata={})],
                # Second query returns wrong source
                [Chunk(id="2", text="ZWL", source="other.txt", metadata={})],
                # Additional calls for precision@k
                [Chunk(id="1", text="Harare", source="geo.txt", metadata={})],
                [Chunk(id="2", text="ZWL", source="other.txt", metadata={})],
                [Chunk(id="1", text="Harare", source="geo.txt", metadata={})],
                [Chunk(id="2", text="ZWL", source="other.txt", metadata={})],
            ]
        )

        # Run tests
        runner = GoldenDatasetRunner(store=mock_store, dataset_path=dataset_file)
        report = await runner.run(k=5)

        # Verify results
        assert report.total_tests == 2
        assert report.passed == 1
        assert report.failed == 1
        assert len(report.failed_tests) == 1
        assert report.failed_tests[0].pair.id == "economics-1"

        # Export and verify
        json_output = runner.to_json(report)
        xml_output = runner.to_junit_xml(report)

        assert "geography-1" in xml_output
        assert "economics-1" in xml_output
        assert '"failed": 1' in json_output
