"""Tests for acquisition modules."""

import pytest

from gweta.core.registry import SourceAuthorityRegistry, SourcePattern
from gweta.acquire.database import QuerySanitizer, QueryResult
from gweta.acquire.api import APIResponse
from gweta.acquire.pdf import PDFPage, PDFTable, PDFExtractionResult
from gweta.acquire.crawler import CrawlResult, PageQualityScore
from gweta.core.types import QualityIssue
from gweta.core.exceptions import DatabaseError


class TestSourceAuthorityRegistry:
    """Tests for the SourceAuthorityRegistry class."""

    def test_registry_creation(self):
        """Test creating an empty registry."""
        registry = SourceAuthorityRegistry()
        assert registry.sources == []
        assert registry.blocked_domains == []

    def test_add_source(self):
        """Test adding sources to registry."""
        registry = SourceAuthorityRegistry()
        registry.add_source(
            domain="example.com",
            name="Example",
            authority=4,
            freshness_days=30,
        )
        assert len(registry.sources) == 1
        assert registry.sources[0].pattern == "example.com"
        assert registry.sources[0].authority_tier == 4

    def test_is_allowed(self):
        """Test URL allowance checking."""
        registry = SourceAuthorityRegistry()
        registry.block_domain("blocked.com")

        assert registry.is_allowed("https://allowed.com/page")
        assert not registry.is_allowed("https://blocked.com/page")

    def test_wildcard_matching(self):
        """Test wildcard pattern matching."""
        registry = SourceAuthorityRegistry()
        registry.add_source(
            domain="*.gov.zw",
            name="Zimbabwe Gov",
            authority=5,
            freshness_days=90,
        )

        assert registry.is_trusted("https://finance.gov.zw/")
        assert registry.is_trusted("https://zimra.gov.zw/taxes")
        assert not registry.is_trusted("https://gov.zw.fake.com/")

    def test_get_authority(self):
        """Test getting authority tier for URL."""
        registry = SourceAuthorityRegistry()
        registry.add_source(
            domain="docs.python.org",
            name="Python Docs",
            authority=5,
            freshness_days=30,
        )

        # Known source
        assert registry.get_authority("https://docs.python.org/3/") == 5

        # Unknown source returns default
        authority = registry.get_authority("https://unknown.com/")
        assert 1 <= authority <= 5  # Should be default tier

    def test_get_source(self):
        """Test getting Source object for URL."""
        registry = SourceAuthorityRegistry()
        registry.add_source(
            domain="example.org",
            name="Example Org",
            authority=4,
            freshness_days=60,
        )

        source = registry.get_source("https://example.org/page")
        assert source.name == "Example Org"
        assert source.authority_tier == 4
        assert source.freshness_days == 60

    def test_from_dict(self):
        """Test creating registry from dictionary."""
        data = {
            "sources": [
                {
                    "domain": "test.com",
                    "name": "Test Site",
                    "authority": 3,
                    "freshness_days": 45,
                }
            ],
            "blocked": ["spam.com"],
        }

        registry = SourceAuthorityRegistry.from_dict(data)
        assert len(registry.sources) == 1
        assert registry.sources[0].name == "Test Site"
        assert "spam.com" in registry.blocked_domains

    def test_to_yaml(self):
        """Test exporting registry to YAML."""
        registry = SourceAuthorityRegistry()
        registry.add_source("test.com", "Test", 3, 30)
        registry.block_domain("bad.com")

        yaml_str = registry.to_yaml()
        assert "test.com" in yaml_str
        assert "bad.com" in yaml_str


class TestSourcePattern:
    """Tests for the SourcePattern class."""

    def test_exact_match(self):
        """Test exact domain matching."""
        pattern = SourcePattern(
            pattern="example.com",
            name="Example",
            authority_tier=3,
            freshness_days=30,
        )

        assert pattern.matches("https://example.com/page")
        assert not pattern.matches("https://sub.example.com/page")

    def test_wildcard_match(self):
        """Test wildcard pattern matching."""
        pattern = SourcePattern(
            pattern="*.example.com",
            name="Example Subdomains",
            authority_tier=3,
            freshness_days=30,
        )

        assert pattern.matches("https://sub.example.com/page")
        assert pattern.matches("https://api.example.com/v1")
        assert not pattern.matches("https://example.com/page")


class TestQuerySanitizer:
    """Tests for the QuerySanitizer class."""

    def test_read_only_query(self):
        """Test that SELECT queries pass validation."""
        QuerySanitizer.validate("SELECT * FROM users", read_only=True)
        QuerySanitizer.validate("SELECT id, name FROM users WHERE active = 1", read_only=True)

    def test_forbidden_keywords(self):
        """Test that write operations are rejected."""
        with pytest.raises(DatabaseError):
            QuerySanitizer.validate("INSERT INTO users (name) VALUES ('test')", read_only=True)

        with pytest.raises(DatabaseError):
            QuerySanitizer.validate("DELETE FROM users WHERE id = 1", read_only=True)

        with pytest.raises(DatabaseError):
            QuerySanitizer.validate("DROP TABLE users", read_only=True)

        with pytest.raises(DatabaseError):
            QuerySanitizer.validate("UPDATE users SET name = 'test'", read_only=True)

    def test_is_read_only(self):
        """Test is_read_only check."""
        assert QuerySanitizer.is_read_only("SELECT * FROM users")
        assert QuerySanitizer.is_read_only("SELECT COUNT(*) FROM orders")
        assert not QuerySanitizer.is_read_only("INSERT INTO users VALUES (1)")
        assert not QuerySanitizer.is_read_only("DROP DATABASE test")

    def test_read_only_disabled(self):
        """Test that validation passes when read_only is disabled."""
        # Should not raise when read_only is False
        QuerySanitizer.validate("INSERT INTO users VALUES (1)", read_only=False)


class TestQueryResult:
    """Tests for the QueryResult class."""

    def test_empty_result(self):
        """Test empty query result."""
        result = QueryResult()
        assert result.rows == []
        assert result.columns == []
        assert result.row_count == 0

    def test_result_with_data(self):
        """Test query result with data."""
        result = QueryResult(
            rows=[{"id": 1, "name": "Test"}],
            columns=["id", "name"],
            row_count=1,
            execution_time=0.05,
        )
        assert len(result.rows) == 1
        assert result.rows[0]["name"] == "Test"
        assert result.execution_time == 0.05


class TestAPIResponse:
    """Tests for the APIResponse class."""

    def test_success_response(self):
        """Test successful API response."""
        response = APIResponse(
            url="https://api.example.com/data",
            status_code=200,
            data={"items": [1, 2, 3]},
        )
        assert response.is_success
        assert response.is_json

    def test_error_response(self):
        """Test error API response."""
        response = APIResponse(
            url="https://api.example.com/data",
            status_code=404,
            data="Not Found",
        )
        assert not response.is_success
        assert not response.is_json


class TestPDFTypes:
    """Tests for PDF extraction types."""

    def test_pdf_page(self):
        """Test PDFPage creation."""
        page = PDFPage(
            number=1,
            text="Sample content from page",
            quality_score=0.9,
            is_scanned=False,
        )
        assert page.number == 1
        assert page.quality_score == 0.9

    def test_pdf_table(self):
        """Test PDFTable creation."""
        table = PDFTable(
            page=1,
            data=[["Header1", "Header2"], ["Val1", "Val2"]],
            quality_score=0.85,
            headers=["Header1", "Header2"],
        )
        assert table.page == 1
        assert len(table.data) == 2
        assert table.headers[0] == "Header1"

    def test_pdf_extraction_result(self):
        """Test PDFExtractionResult."""
        page1 = PDFPage(number=1, text="Page 1 content", quality_score=0.9)
        page2 = PDFPage(number=2, text="Page 2 content", quality_score=0.8)

        result = PDFExtractionResult(
            pages=[page1, page2],
            quality_score=0.85,
        )

        assert result.total_pages == 2
        assert "Page 1 content" in result.full_text
        assert "Page 2 content" in result.full_text


class TestCrawlTypes:
    """Tests for crawler types."""

    def test_crawl_result(self):
        """Test CrawlResult creation."""
        result = CrawlResult(url="https://example.com")
        assert result.url == "https://example.com"
        assert result.pages_crawled == 0
        assert result.chunks == []

    def test_crawl_result_summary(self):
        """Test CrawlResult summary."""
        result = CrawlResult(
            url="https://example.com",
            pages_crawled=5,
            pages_passed=4,
            pages_failed=1,
            quality_score=0.85,
        )
        summary = result.summary()
        assert "example.com" in summary
        assert "5 crawled" in summary
        assert "0.85" in summary

    def test_page_quality_score(self):
        """Test PageQualityScore."""
        score = PageQualityScore(
            url="https://example.com/page",
            extraction_score=0.9,
            content_completeness=0.8,
            boilerplate_ratio=0.2,
        )
        assert score.url == "https://example.com/page"
        assert score.passed  # No error issues
        # overall_score = 0.9*0.4 + 0.8*0.3 + (1-0.2)*0.3 = 0.36 + 0.24 + 0.24 = 0.84
        assert 0.83 <= score.overall_score <= 0.85

    def test_page_quality_with_error(self):
        """Test PageQualityScore with error issue."""
        score = PageQualityScore(
            url="https://example.com/page",
            issues=[
                QualityIssue(
                    code="CONTENT_TOO_SHORT",
                    severity="error",
                    message="Content too short",
                )
            ],
        )
        assert not score.passed  # Has error issue
