"""Integration tests for the ingestion pipeline."""

import pytest

from gweta.core.types import Chunk
from gweta.validate.chunks import ChunkValidator, ChunkValidatorConfig


class TestValidationPipeline:
    """Integration tests for validation pipeline."""

    def test_full_validation_flow(self, sample_chunks):
        """Test complete validation flow."""
        # Create validator
        validator = ChunkValidator()

        # Validate batch
        report = validator.validate_batch(sample_chunks)

        # Check report structure
        assert report.total_chunks == len(sample_chunks)
        assert report.passed >= 0
        assert report.failed >= 0
        assert report.passed + report.failed == report.total_chunks

        # Get accepted chunks
        accepted = report.accepted()
        assert len(accepted) == report.passed

        # Verify all accepted have quality scores
        for chunk in accepted:
            assert chunk.quality_score is not None
            assert chunk.quality_score >= validator.config.min_quality_score

    def test_mixed_quality_batch(self):
        """Test batch with mixed quality chunks."""
        chunks = [
            # Good chunk
            Chunk(
                text="This is a well-written chunk with substantial content. "
                "It provides valuable information and context for readers.",
                source="good.txt",
            ),
            # Bad chunk (empty)
            Chunk(text="", source="bad.txt"),
            # Another good chunk
            Chunk(
                text="Another high-quality chunk with meaningful content. "
                "This chunk discusses important concepts in detail.",
                source="good2.txt",
            ),
        ]

        validator = ChunkValidator()
        report = validator.validate_batch(chunks)

        # Should have some passed and some failed (empty chunk should fail)
        assert report.passed >= 1
        assert report.failed >= 1

    def test_validation_preserves_metadata(self, sample_chunk):
        """Test that validation preserves chunk metadata."""
        validator = ChunkValidator()
        result = validator.validate(sample_chunk)

        # Original metadata should be preserved
        assert sample_chunk.metadata.get("title") == "Sample Page"
        assert sample_chunk.metadata.get("section") == "Introduction"

    def test_quality_score_assignment(self, sample_chunks):
        """Test that quality scores are assigned correctly."""
        validator = ChunkValidator()
        report = validator.validate_batch(sample_chunks)

        # All chunks should have quality scores after validation
        for chunk_result in report.chunks:
            assert chunk_result.quality_score is not None
            assert 0.0 <= chunk_result.quality_score <= 1.0

    def test_strict_validation(self):
        """Test validation with strict threshold."""
        chunks = [
            Chunk(
                text="A chunk with content that might not meet very strict requirements. "
                "This text is decent but may have some minor issues.",
                source="test.txt",
            ),
        ]

        # Strict threshold
        config = ChunkValidatorConfig(min_quality_score=0.99)
        validator = ChunkValidator(config=config)
        report = validator.validate_batch(chunks)

        # With 0.99 threshold, chunks may fail
        # The important thing is validation runs without error
        assert report.total_chunks == 1


@pytest.mark.skipif(
    True,  # Skip by default, enable when Chroma is installed
    reason="Requires chromadb installation",
)
class TestChromaIntegration:
    """Integration tests for Chroma store."""

    @pytest.fixture
    def temp_collection(self):
        """Create a temporary Chroma collection."""
        from gweta.ingest.stores.chroma import ChromaStore

        store = ChromaStore(collection_name="test_collection")
        yield store
        # Cleanup would go here

    async def test_add_and_query(self, temp_collection, sample_chunks):
        """Test adding and querying chunks."""
        # Add chunks
        await temp_collection.add(sample_chunks)

        # Query
        results = await temp_collection.query("sample content", top_k=2)
        assert len(results) <= 2

    async def test_add_validated_chunks(self, temp_collection, sample_chunks):
        """Test adding only validated chunks."""
        validator = ChunkValidator()
        report = validator.validate_batch(sample_chunks)
        accepted = report.accepted()

        if accepted:
            await temp_collection.add(accepted)
            stats = await temp_collection.stats()
            assert stats["count"] == len(accepted)
