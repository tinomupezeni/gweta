"""Tests for core types."""

import pytest

from gweta.core.types import Chunk, QualityDetails, QualityIssue, QualityReport, Source, ChunkResult


class TestChunk:
    """Tests for the Chunk class."""

    def test_chunk_creation(self):
        """Test basic chunk creation."""
        chunk = Chunk(text="Hello world", source="test.txt")
        assert chunk.text == "Hello world"
        assert chunk.source == "test.txt"
        assert chunk.id is not None
        assert chunk.metadata == {}

    def test_chunk_with_metadata(self):
        """Test chunk with metadata."""
        chunk = Chunk(
            text="Content",
            source="test.txt",
            metadata={"key": "value"},
        )
        assert chunk.metadata["key"] == "value"

    def test_chunk_with_quality_score(self):
        """Test chunk with quality score."""
        chunk = Chunk(text="Content", source="test.txt", quality_score=0.85)
        assert chunk.quality_score == 0.85

    def test_chunk_to_dict(self):
        """Test chunk serialization."""
        chunk = Chunk(
            text="Content",
            source="test.txt",
            metadata={"key": "value"},
            quality_score=0.9,
        )
        data = chunk.to_dict()
        assert data["text"] == "Content"
        assert data["source"] == "test.txt"
        assert data["metadata"]["key"] == "value"
        assert data["quality_score"] == 0.9

    def test_chunk_from_dict(self):
        """Test chunk deserialization."""
        data = {
            "id": "test-id",
            "text": "Content",
            "source": "test.txt",
            "metadata": {"key": "value"},
            "quality_score": 0.8,
        }
        chunk = Chunk.from_dict(data)
        assert chunk.id == "test-id"
        assert chunk.text == "Content"
        assert chunk.source == "test.txt"
        assert chunk.quality_score == 0.8


class TestSource:
    """Tests for the Source class."""

    def test_source_creation(self):
        """Test basic source creation."""
        source = Source(id="example", name="Example", url="https://example.com")
        assert source.url == "https://example.com"
        assert source.name == "Example"
        assert source.authority_tier == 3  # Default
        assert source.freshness_days == 90  # Default

    def test_source_with_authority(self):
        """Test source with custom authority."""
        source = Source(
            id="python-docs",
            name="Python Docs",
            url="https://docs.python.org",
            authority_tier=5,
            freshness_days=168,
        )
        assert source.authority_tier == 5
        assert source.freshness_days == 168

    def test_source_to_dict(self):
        """Test source serialization."""
        source = Source(
            id="example",
            name="Example",
            url="https://example.com",
        )
        data = source.to_dict()
        assert data["url"] == "https://example.com"
        assert data["id"] == "example"


class TestQualityReport:
    """Tests for the QualityReport class."""

    def test_empty_report(self):
        """Test empty report creation."""
        report = QualityReport(total_chunks=0, passed=0, failed=0, warnings=0, avg_quality_score=0.0)
        assert report.total_chunks == 0
        assert report.avg_quality_score == 0.0

    def test_report_with_results(self, sample_chunks):
        """Test report with chunk results."""
        report = QualityReport(
            total_chunks=3,
            passed=2,
            failed=1,
            warnings=0,
            avg_quality_score=0.75,
        )
        assert report.passed == 2
        assert report.failed == 1
        assert report.avg_quality_score == 0.75

    def test_report_to_dict(self):
        """Test report serialization."""
        report = QualityReport(
            total_chunks=10,
            passed=8,
            failed=2,
            warnings=1,
            avg_quality_score=0.82,
            issues_by_type={"LOW_DENSITY": 2},
        )
        data = report.to_dict()
        assert data["total_chunks"] == 10
        assert data["issues_by_type"]["LOW_DENSITY"] == 2


class TestQualityIssue:
    """Tests for the QualityIssue class."""

    def test_issue_creation(self):
        """Test issue creation."""
        issue = QualityIssue(
            code="LOW_DENSITY",
            message="Chunk has low information density",
            severity="warning",
        )
        assert issue.code == "LOW_DENSITY"
        assert issue.severity == "warning"

    def test_issue_with_location(self):
        """Test issue with location."""
        issue = QualityIssue(
            code="DUPLICATE",
            message="Duplicate content detected",
            severity="error",
            location="chunk_abc123",
        )
        assert issue.location == "chunk_abc123"
        assert "DUPLICATE" in str(issue)


class TestQualityDetails:
    """Tests for the QualityDetails class."""

    def test_quality_details_defaults(self):
        """Test default quality details."""
        details = QualityDetails()
        assert details.extraction_score == 1.0
        assert details.coherence_score == 1.0
        assert details.density_score == 1.0
        assert details.duplicate_score == 1.0
        assert details.aggregate_score == 1.0

    def test_aggregate_score(self):
        """Test aggregate score calculation."""
        details = QualityDetails(
            extraction_score=0.8,
            coherence_score=0.6,
            density_score=0.9,
            duplicate_score=1.0,
        )
        # (0.8 + 0.6 + 0.9 + 1.0) / 4 = 0.825
        expected = (0.8 * 0.25 + 0.6 * 0.25 + 0.9 * 0.25 + 1.0 * 0.25)
        assert abs(details.aggregate_score - expected) < 0.001

    def test_has_errors(self):
        """Test error detection."""
        details = QualityDetails(
            issues=[
                QualityIssue(code="TEST", severity="error", message="Error"),
            ]
        )
        assert details.has_errors()

        details2 = QualityDetails(
            issues=[
                QualityIssue(code="TEST", severity="warning", message="Warning"),
            ]
        )
        assert not details2.has_errors()
