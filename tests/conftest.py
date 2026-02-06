"""Pytest configuration and fixtures for Gweta tests."""

import pytest

from gweta.core.types import Chunk, Source


@pytest.fixture
def sample_chunk() -> Chunk:
    """Create a sample chunk for testing."""
    return Chunk(
        text="This is a sample chunk with enough content to pass basic validation. "
        "It contains multiple sentences and provides meaningful information about "
        "the topic at hand. The content is well-structured and informative.",
        source="https://example.com/page",
        metadata={"title": "Sample Page", "section": "Introduction"},
    )


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Create a list of sample chunks for batch testing."""
    return [
        Chunk(
            text="First chunk with substantial content that provides value to readers. "
            "This chunk discusses the introduction to a topic.",
            source="https://example.com/page1",
            metadata={"section": "intro"},
        ),
        Chunk(
            text="Second chunk explaining key concepts in detail. "
            "This section covers the fundamental principles.",
            source="https://example.com/page2",
            metadata={"section": "concepts"},
        ),
        Chunk(
            text="Third chunk with practical examples and use cases. "
            "Real-world applications demonstrate the concepts.",
            source="https://example.com/page3",
            metadata={"section": "examples"},
        ),
    ]


@pytest.fixture
def low_quality_chunk() -> Chunk:
    """Create a low-quality chunk for testing rejection."""
    return Chunk(
        text="bad",
        source="",
        metadata={},
    )


@pytest.fixture
def gibberish_chunk() -> Chunk:
    """Create a gibberish chunk for testing detection."""
    return Chunk(
        text="asdfghjkl qwertyuiop zxcvbnm asdf ghjkl qwer tyui opzx cvbn",
        source="https://example.com/broken",
        metadata={},
    )


@pytest.fixture
def sample_source() -> Source:
    """Create a sample source for testing."""
    return Source(
        id="example-docs",
        name="Example Documentation",
        url="https://docs.example.com",
        authority_tier=4,
        freshness_days=24,
    )


@pytest.fixture
def trusted_sources() -> list[Source]:
    """Create a list of trusted sources."""
    return [
        Source(
            id="python-docs",
            name="Python Documentation",
            url="https://docs.python.org",
            authority_tier=5,
            freshness_days=168,
        ),
        Source(
            id="mdn",
            name="MDN Web Docs",
            url="https://developer.mozilla.org",
            authority_tier=5,
            freshness_days=24,
        ),
    ]


@pytest.fixture
def golden_dataset() -> list[dict]:
    """Create a sample golden dataset for testing."""
    return [
        {
            "question": "What is Python?",
            "expected_chunks": ["Python is a programming language"],
            "min_score": 0.7,
        },
        {
            "question": "How do I install packages?",
            "expected_chunks": ["pip install", "package manager"],
            "min_score": 0.6,
        },
    ]


@pytest.fixture
def domain_rules() -> dict:
    """Create sample domain rules for testing."""
    return {
        "name": "test-domain",
        "version": "1.0",
        "rules": [
            {
                "id": "no-pii",
                "name": "No PII",
                "description": "Chunks should not contain PII",
                "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
                "action": "reject",
                "severity": "error",
            },
            {
                "id": "min-length",
                "name": "Minimum Length",
                "description": "Chunks must have minimum length",
                "condition": "len(chunk.text) >= 50",
                "action": "reject",
                "severity": "warning",
            },
        ],
        "known_facts": [
            {
                "claim": "Python was created by Guido van Rossum",
                "confidence": 1.0,
            },
        ],
    }
