"""Chonkie adapter for Gweta.

This module provides conversion between Chonkie
Chunk objects and Gweta Chunk objects.
"""

from typing import Any

from gweta.core.types import Chunk


class ChonkieAdapter:
    """Adapter for Chonkie Chunk objects.

    Converts between Chonkie Chunks and Gweta Chunks,
    allowing Gweta validation to be used with Chonkie pipelines.

    Example:
        >>> adapter = ChonkieAdapter()
        >>> gweta_chunks = adapter.from_chonkie(chonkie_chunks)
        >>> chonkie_chunks = adapter.to_chonkie(gweta_chunks)
    """

    @staticmethod
    def from_chonkie(
        chunks: list[Any],
        source: str = "",
    ) -> list[Chunk]:
        """Convert Chonkie Chunks to Gweta Chunks.

        Args:
            chunks: List of Chonkie Chunk objects
            source: Source identifier to attach

        Returns:
            List of Gweta Chunk objects
        """
        gweta_chunks = []
        for i, chunk in enumerate(chunks):
            metadata = {}

            # Extract Chonkie-specific attributes
            if hasattr(chunk, "start_index"):
                metadata["start_index"] = chunk.start_index
            if hasattr(chunk, "end_index"):
                metadata["end_index"] = chunk.end_index
            if hasattr(chunk, "token_count"):
                metadata["token_count"] = chunk.token_count

            gweta_chunks.append(
                Chunk(
                    text=chunk.text if hasattr(chunk, "text") else str(chunk),
                    metadata=metadata,
                    source=source,
                )
            )
        return gweta_chunks

    @staticmethod
    def to_chonkie(chunks: list[Chunk]) -> list[Any]:
        """Convert Gweta Chunks back to Chonkie format.

        Args:
            chunks: List of Gweta Chunk objects

        Returns:
            List of dict representations compatible with Chonkie
        """
        # Return as dicts since we don't want to require Chonkie import
        return [
            {
                "text": chunk.text,
                "metadata": chunk.metadata,
                "quality_score": chunk.quality_score,
            }
            for chunk in chunks
        ]

    @staticmethod
    def create_validation_step(validator: Any) -> Any:
        """Create a Chonkie pipeline step for validation.

        Args:
            validator: Gweta ChunkValidator

        Returns:
            Callable that can be used as a Chonkie pipeline step
        """
        adapter = ChonkieAdapter()

        def validate_step(chunks: list[Any], **kwargs: Any) -> list[Any]:
            gweta_chunks = adapter.from_chonkie(chunks)
            report = validator.validate_batch(gweta_chunks)
            # Return original format with validation info added
            result = []
            for chunk, gweta_chunk in zip(chunks, gweta_chunks):
                if gweta_chunk.quality_score and gweta_chunk.quality_score >= 0.6:
                    result.append(chunk)
            return result

        return validate_step
