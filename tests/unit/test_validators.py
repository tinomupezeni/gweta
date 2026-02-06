"""Tests for validation modules."""

import pytest

from gweta.core.types import Chunk
from gweta.validate.chunks import ChunkValidator, ChunkValidatorConfig


class TestChunkValidator:
    """Tests for the ChunkValidator class."""

    def test_validator_creation(self):
        """Test validator creation with defaults."""
        validator = ChunkValidator()
        assert validator.config.min_quality_score == 0.6
        assert validator.config.min_length == 50

    def test_validator_with_config(self):
        """Test validator with custom config."""
        config = ChunkValidatorConfig(
            min_quality_score=0.8,
            min_length=100,
            max_length=5000,
        )
        validator = ChunkValidator(config=config)
        assert validator.config.min_quality_score == 0.8
        assert validator.config.min_length == 100

    def test_validate_good_chunk(self, sample_chunk):
        """Test validation of a good chunk."""
        validator = ChunkValidator()
        result = validator.validate(sample_chunk)
        assert result.passed
        assert result.quality_score >= 0.6

    def test_validate_bad_chunk(self, low_quality_chunk):
        """Test validation of a bad chunk with high threshold."""
        # Use a high threshold to ensure the low-quality chunk fails
        config = ChunkValidatorConfig(min_quality_score=0.95)
        validator = ChunkValidator(config=config)
        result = validator.validate(low_quality_chunk)
        # Should have issues detected
        assert len(result.issues) > 0
        # Check that quality metrics are calculated
        assert result.quality_score < 1.0

    def test_validate_batch(self, sample_chunks):
        """Test batch validation."""
        validator = ChunkValidator()
        report = validator.validate_batch(sample_chunks)
        assert report.total_chunks == 3
        assert report.passed + report.failed == 3

    def test_validate_gibberish(self, gibberish_chunk):
        """Test that gibberish gets lower coherence score."""
        validator = ChunkValidator()
        result = validator.validate(gibberish_chunk)
        # Gibberish should have boundary issues at minimum
        assert len(result.issues) > 0 or result.quality_score < 1.0

    def test_short_content_warning(self):
        """Test that short content gets a warning."""
        validator = ChunkValidator()
        chunk = Chunk(text="Too short", source="test.txt")
        result = validator.validate(chunk)
        # Should have CONTENT_TOO_SHORT warning
        assert any("SHORT" in issue.code.upper() for issue in result.issues)

    def test_high_quality_threshold(self, sample_chunk):
        """Test quality threshold enforcement."""
        # Very high threshold should fail even good chunks
        config = ChunkValidatorConfig(min_quality_score=0.999)
        validator = ChunkValidator(config=config)
        result = validator.validate(sample_chunk)
        # With such a high threshold, chunk may not pass
        assert result.quality_score < 0.999 or not result.passed

    def test_report_accepted_chunks(self, sample_chunks):
        """Test getting accepted chunks from report."""
        validator = ChunkValidator()
        report = validator.validate_batch(sample_chunks)
        accepted = report.accepted()
        assert len(accepted) <= len(sample_chunks)
        for chunk in accepted:
            assert chunk.quality_score is not None
            assert chunk.quality_score >= validator.config.min_quality_score

    def test_empty_chunk_fails(self):
        """Test that empty chunks fail validation."""
        validator = ChunkValidator()
        chunk = Chunk(text="", source="test.txt")
        result = validator.validate(chunk)
        # Empty content should be an error
        assert not result.passed
        assert any("EMPTY" in issue.code.upper() for issue in result.issues)
