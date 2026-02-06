"""Tests for the intelligence layer."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

from gweta.core.types import Chunk
from gweta.intelligence.intent import (
    SystemIntent,
    QualityRequirements,
    GeographicFocus,
)
from gweta.intelligence.relevance import (
    RelevanceFilter,
    RelevanceResult,
    RelevanceReport,
    RelevanceDecision,
)


# ============================================================
# SystemIntent Tests
# ============================================================


class TestSystemIntent:
    """Tests for SystemIntent class."""

    def test_create_basic_intent(self):
        """Test creating a basic intent."""
        intent = SystemIntent(
            name="Test System",
            description="A test system for unit testing",
            core_questions=["What is X?", "How do I Y?"],
        )

        assert intent.name == "Test System"
        assert intent.description == "A test system for unit testing"
        assert len(intent.core_questions) == 2

    def test_intent_text_generation(self):
        """Test that intent text is generated correctly."""
        intent = SystemIntent(
            name="Career Platform",
            description="Helps graduates find careers",
            target_users=["graduates", "students"],
            core_questions=["How to register business?"],
            relevant_topics=["business", "careers"],
        )

        text = intent.intent_text
        assert "Helps graduates find careers" in text
        assert "How to register business?" in text
        assert "business" in text

    def test_from_dict(self):
        """Test creating intent from dictionary."""
        data = {
            "system": {
                "name": "My System",
                "description": "Test description",
                "core_questions": ["Q1", "Q2"],
                "relevant_topics": ["topic1"],
                "irrelevant_topics": ["bad_topic"],
                "quality_requirements": {
                    "min_relevance_score": 0.7,
                    "review_threshold": 0.5,
                },
            }
        }

        intent = SystemIntent.from_dict(data)

        assert intent.name == "My System"
        assert intent.min_relevance_score == 0.7
        assert intent.review_threshold == 0.5
        assert "bad_topic" in intent.irrelevant_topics

    def test_from_dict_flat(self):
        """Test creating intent from flat dictionary."""
        data = {
            "name": "Flat System",
            "description": "No nested system key",
        }

        intent = SystemIntent.from_dict(data)
        assert intent.name == "Flat System"

    def test_geographic_focus_string(self):
        """Test geographic focus as string."""
        data = {
            "name": "Geo Test",
            "geographic_focus": "Zimbabwe",
        }

        intent = SystemIntent.from_dict(data)
        assert intent.geographic_focus.primary == "Zimbabwe"

    def test_geographic_focus_dict(self):
        """Test geographic focus as dict."""
        data = {
            "name": "Geo Test",
            "geographic_focus": {
                "primary": "Zimbabwe",
                "secondary": ["South Africa", "Botswana"],
            },
        }

        intent = SystemIntent.from_dict(data)
        assert intent.geographic_focus.primary == "Zimbabwe"
        assert "South Africa" in intent.geographic_focus.secondary

    def test_is_irrelevant_topic(self):
        """Test irrelevant topic detection."""
        intent = SystemIntent(
            name="Test",
            description="Test",
            irrelevant_topics=["cryptocurrency", "forex trading"],
        )

        assert intent.is_irrelevant_topic("Learn about cryptocurrency investing")
        assert intent.is_irrelevant_topic("FOREX TRADING tips")
        assert not intent.is_irrelevant_topic("Business registration guide")

    def test_to_dict(self):
        """Test serialization to dict."""
        intent = SystemIntent(
            name="Test",
            description="Description",
            core_questions=["Q1"],
        )

        data = intent.to_dict()
        assert data["system"]["name"] == "Test"
        assert "Q1" in data["system"]["core_questions"]

    def test_to_yaml(self):
        """Test serialization to YAML."""
        intent = SystemIntent(
            name="Test",
            description="Description",
        )

        yaml_str = intent.to_yaml()
        assert "name: Test" in yaml_str
        assert "description: Description" in yaml_str

    def test_default_quality_requirements(self):
        """Test default quality requirements."""
        intent = SystemIntent(name="Test", description="Test")

        assert intent.min_relevance_score == 0.6
        assert intent.review_threshold == 0.4


class TestQualityRequirements:
    """Tests for QualityRequirements."""

    def test_defaults(self):
        """Test default values."""
        req = QualityRequirements()

        assert req.min_relevance_score == 0.6
        assert req.review_threshold == 0.4
        assert req.freshness_cutoff is None
        assert req.prefer_official_sources is True


class TestGeographicFocus:
    """Tests for GeographicFocus."""

    def test_defaults(self):
        """Test default values."""
        geo = GeographicFocus()

        assert geo.primary is None
        assert geo.secondary == []

    def test_with_values(self):
        """Test with values."""
        geo = GeographicFocus(
            primary="Zimbabwe",
            secondary=["Zambia", "Malawi"],
        )

        assert geo.primary == "Zimbabwe"
        assert len(geo.secondary) == 2


# ============================================================
# RelevanceResult Tests
# ============================================================


class TestRelevanceResult:
    """Tests for RelevanceResult."""

    def test_accepted_result(self):
        """Test an accepted result."""
        chunk = Chunk(text="Test", source="test")
        result = RelevanceResult(
            chunk=chunk,
            relevance_score=0.85,
            decision=RelevanceDecision.ACCEPT,
        )

        assert result.accepted
        assert not result.needs_review
        assert not result.rejected

    def test_review_result(self):
        """Test a review result."""
        chunk = Chunk(text="Test", source="test")
        result = RelevanceResult(
            chunk=chunk,
            relevance_score=0.55,
            decision=RelevanceDecision.REVIEW,
        )

        assert not result.accepted
        assert result.needs_review
        assert not result.rejected

    def test_rejected_result(self):
        """Test a rejected result."""
        chunk = Chunk(text="Test", source="test")
        result = RelevanceResult(
            chunk=chunk,
            relevance_score=0.2,
            decision=RelevanceDecision.REJECT,
            rejection_reason="Score too low",
        )

        assert not result.accepted
        assert not result.needs_review
        assert result.rejected
        assert result.rejection_reason == "Score too low"


# ============================================================
# RelevanceReport Tests
# ============================================================


class TestRelevanceReport:
    """Tests for RelevanceReport."""

    def test_empty_report(self):
        """Test empty report."""
        report = RelevanceReport(results=[], intent_name="Test")

        assert report.total_chunks == 0
        assert report.accepted_count == 0
        assert report.acceptance_rate == 0.0

    def test_report_statistics(self):
        """Test report statistics computation."""
        chunk = Chunk(text="Test", source="test")
        results = [
            RelevanceResult(chunk=chunk, relevance_score=0.9, decision=RelevanceDecision.ACCEPT),
            RelevanceResult(chunk=chunk, relevance_score=0.8, decision=RelevanceDecision.ACCEPT),
            RelevanceResult(chunk=chunk, relevance_score=0.5, decision=RelevanceDecision.REVIEW),
            RelevanceResult(chunk=chunk, relevance_score=0.2, decision=RelevanceDecision.REJECT),
        ]

        report = RelevanceReport(results=results, intent_name="Test")

        assert report.total_chunks == 4
        assert report.accepted_count == 2
        assert report.review_count == 1
        assert report.rejected_count == 1
        assert report.acceptance_rate == 0.5
        assert report.rejection_rate == 0.25
        assert report.avg_relevance_score == 0.6  # (0.9+0.8+0.5+0.2)/4

    def test_accepted_method(self):
        """Test accepted() returns chunks with metadata."""
        chunk1 = Chunk(text="Test 1", source="test", metadata={})
        chunk2 = Chunk(text="Test 2", source="test", metadata={})

        results = [
            RelevanceResult(
                chunk=chunk1,
                relevance_score=0.9,
                decision=RelevanceDecision.ACCEPT,
                matched_topics=["business"],
            ),
            RelevanceResult(
                chunk=chunk2,
                relevance_score=0.3,
                decision=RelevanceDecision.REJECT,
            ),
        ]

        report = RelevanceReport(results=results, intent_name="TestIntent")
        accepted = report.accepted()

        assert len(accepted) == 1
        assert accepted[0].metadata["relevance_score"] == 0.9
        assert accepted[0].metadata["intent"] == "TestIntent"
        assert "business" in accepted[0].metadata["matched_topics"]

    def test_for_review_method(self):
        """Test for_review() returns review chunks."""
        chunk = Chunk(text="Test", source="test")
        results = [
            RelevanceResult(chunk=chunk, relevance_score=0.5, decision=RelevanceDecision.REVIEW),
            RelevanceResult(chunk=chunk, relevance_score=0.9, decision=RelevanceDecision.ACCEPT),
        ]

        report = RelevanceReport(results=results, intent_name="Test")
        review = report.for_review()

        assert len(review) == 1


# ============================================================
# RelevanceFilter Tests (with mocked embeddings)
# ============================================================


class TestRelevanceFilter:
    """Tests for RelevanceFilter with mocked embeddings."""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock embedding engine."""
        engine = MagicMock()
        engine.model_name = "mock-model"

        # Mock embed to return a fixed vector
        engine.embed.return_value = np.array([1.0, 0.0, 0.0])

        # Mock embed_batch to return vectors
        def embed_batch_side_effect(texts, **kwargs):
            return np.array([[1.0, 0.0, 0.0]] * len(texts))

        engine.embed_batch.side_effect = embed_batch_side_effect

        # Mock cosine similarity
        engine._cosine_similarity.return_value = 0.8
        engine._batch_cosine_similarity.return_value = np.array([0.8, 0.5, 0.2])

        return engine

    @pytest.fixture
    def intent(self):
        """Create a test intent."""
        return SystemIntent(
            name="Test Intent",
            description="A test intent for filtering",
            core_questions=["What is business registration?"],
            relevant_topics=["business", "registration"],
            irrelevant_topics=["cryptocurrency"],
            quality_requirements=QualityRequirements(
                min_relevance_score=0.6,
                review_threshold=0.4,
            ),
        )

    def test_filter_accepts_relevant(self, mock_engine, intent):
        """Test that relevant content is accepted."""
        # Make similarity return high score
        mock_engine._cosine_similarity.return_value = 0.9

        filter = RelevanceFilter(intent=intent, embedding_engine=mock_engine)
        filter._intent_embedding = np.array([1.0, 0.0, 0.0])

        chunk = Chunk(text="How to register a business", source="test")
        result = filter.filter(chunk)

        assert result.accepted
        assert result.relevance_score >= 0.6

    def test_filter_rejects_irrelevant_topic(self, mock_engine, intent):
        """Test that explicitly irrelevant topics are rejected."""
        filter = RelevanceFilter(intent=intent, embedding_engine=mock_engine)
        filter._intent_embedding = np.array([1.0, 0.0, 0.0])

        chunk = Chunk(text="Invest in cryptocurrency now!", source="test")
        result = filter.filter(chunk)

        assert result.rejected
        assert result.relevance_score == 0.0
        assert "irrelevant topic" in result.rejection_reason.lower()

    def test_filter_rejects_low_score(self, mock_engine, intent):
        """Test that low scoring content is rejected."""
        # Make similarity return low score
        mock_engine._cosine_similarity.return_value = -0.5  # Will become 0.25 after normalization

        filter = RelevanceFilter(intent=intent, embedding_engine=mock_engine)
        filter._intent_embedding = np.array([1.0, 0.0, 0.0])

        chunk = Chunk(text="Unrelated content", source="test")
        result = filter.filter(chunk)

        assert result.rejected
        assert result.relevance_score < 0.4

    def test_filter_batch(self, mock_engine, intent):
        """Test batch filtering."""
        filter = RelevanceFilter(intent=intent, embedding_engine=mock_engine)
        filter._intent_embedding = np.array([1.0, 0.0, 0.0])

        chunks = [
            Chunk(text="Business registration guide", source="test"),
            Chunk(text="Medium relevance content", source="test"),
            Chunk(text="Completely unrelated", source="test"),
        ]

        # Mock returns [0.8, 0.5, 0.2] for batch similarity
        report = filter.filter_batch(chunks)

        assert report.total_chunks == 3
        # Score 0.8 -> normalized (0.8+1)/2 = 0.9 -> ACCEPT
        # Score 0.5 -> normalized (0.5+1)/2 = 0.75 -> ACCEPT
        # Score 0.2 -> normalized (0.2+1)/2 = 0.6 -> ACCEPT (just at threshold)

    def test_find_matched_topics(self, mock_engine, intent):
        """Test topic matching."""
        filter = RelevanceFilter(intent=intent, embedding_engine=mock_engine)

        matched = filter._find_matched_topics("Guide to business registration in Zimbabwe")

        assert "business" in matched
        assert "registration" in matched

    def test_score_method(self, mock_engine, intent):
        """Test scoring a single chunk."""
        mock_engine._cosine_similarity.return_value = 0.6

        filter = RelevanceFilter(intent=intent, embedding_engine=mock_engine)
        filter._intent_embedding = np.array([1.0, 0.0, 0.0])

        chunk = Chunk(text="Some content", source="test")
        score = filter.score(chunk)

        # (0.6 + 1) / 2 = 0.8
        assert score == 0.8


# ============================================================
# Integration Tests (require sentence-transformers)
# ============================================================


@pytest.mark.skipif(
    not pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed"),
    reason="sentence-transformers not installed"
)
class TestRelevanceFilterIntegration:
    """Integration tests that use real embeddings."""

    @pytest.fixture
    def simuka_intent(self):
        """Create Simuka-like intent."""
        return SystemIntent(
            name="Simuka Career Platform",
            description="Career guidance for Zimbabwean graduates",
            core_questions=[
                "How do I register a business in Zimbabwe?",
                "What freelance services can I offer?",
            ],
            relevant_topics=[
                "Zimbabwe business registration",
                "ZIMRA tax",
                "freelancing",
                "entrepreneurship",
            ],
            irrelevant_topics=[
                "US tax law",
                "cryptocurrency",
            ],
        )

    def test_real_embedding_relevance(self, simuka_intent):
        """Test with real embeddings."""
        from gweta.intelligence import RelevanceFilter

        filter = RelevanceFilter(intent=simuka_intent)

        # Highly relevant chunk
        relevant_chunk = Chunk(
            text="To register a Private Business Corporation in Zimbabwe, "
                 "you need to submit Form CR6 to the Companies Registry and "
                 "register with ZIMRA within 30 days.",
            source="test",
        )

        # Irrelevant chunk
        irrelevant_chunk = Chunk(
            text="The US Internal Revenue Service requires all LLC owners to "
                 "file Schedule C with their personal tax returns.",
            source="test",
        )

        relevant_result = filter.filter(relevant_chunk)
        irrelevant_result = filter.filter(irrelevant_chunk)

        # Relevant should score higher
        assert relevant_result.relevance_score > irrelevant_result.relevance_score
        assert relevant_result.accepted or relevant_result.needs_review
