"""Chunking strategies for text splitting.

This module provides chunking implementations and
integration with external chunking libraries.
"""

import re
from dataclasses import dataclass
from typing import Any

from gweta.core.config import get_settings
from gweta.core.logging import get_logger
from gweta.core.types import Chunk

logger = get_logger(__name__)


@dataclass
class ChunkerConfig:
    """Configuration for chunking."""
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: list[str] | None = None


class RecursiveChunker:
    """Default chunker using recursive text splitting.

    Splits text recursively using a hierarchy of separators,
    trying to keep semantic units together.

    Example:
        >>> chunker = RecursiveChunker(chunk_size=500)
        >>> chunks = chunker.chunk("Long text content...", {"source": "doc.pdf"})
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ) -> None:
        """Initialize RecursiveChunker.

        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
            separators: List of separators to try (in order)
        """
        settings = get_settings()
        self.chunk_size = chunk_size or settings.default_chunk_size
        self.chunk_overlap = chunk_overlap or settings.default_chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        source: str = "",
    ) -> list[Chunk]:
        """Split text into chunks.

        Args:
            text: Text to split
            metadata: Metadata to attach to all chunks
            source: Source identifier

        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}
        splits = self._split_text(text, self.separators)

        chunks: list[Chunk] = []
        for i, split in enumerate(splits):
            if split.strip():
                chunks.append(
                    Chunk(
                        text=split.strip(),
                        source=source,
                        metadata={
                            **metadata,
                            "chunk_index": i,
                        },
                    )
                )

        return chunks

    def _split_text(
        self,
        text: str,
        separators: list[str],
    ) -> list[str]:
        """Recursively split text using separators.

        Args:
            text: Text to split
            separators: Separators to try

        Returns:
            List of text splits
        """
        if not separators:
            # No more separators, force split by character
            return self._split_by_size(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # Merge small splits back together
        merged: list[str] = []
        current = ""

        for split in splits:
            # Add separator back if not empty
            test_chunk = current + separator + split if current else split

            if len(test_chunk) <= self.chunk_size:
                current = test_chunk
            else:
                if current:
                    merged.append(current)
                if len(split) <= self.chunk_size:
                    current = split
                else:
                    # Split is too large, recurse with finer separators
                    subsplits = self._split_text(split, remaining_separators)
                    merged.extend(subsplits)
                    current = ""

        if current:
            merged.append(current)

        return merged

    def _split_by_size(self, text: str) -> list[str]:
        """Split text by size with overlap.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            if chunk.strip():
                chunks.append(chunk)

            start = end - self.chunk_overlap
            if start <= 0:
                start = end

        return chunks


class SentenceChunker:
    """Chunk text by sentences.

    Splits text at sentence boundaries, combining sentences
    to reach target chunk size.
    """

    SENTENCE_ENDINGS = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        chunk_size: int | None = None,
        min_sentences: int = 1,
    ) -> None:
        """Initialize SentenceChunker.

        Args:
            chunk_size: Target chunk size
            min_sentences: Minimum sentences per chunk
        """
        settings = get_settings()
        self.chunk_size = chunk_size or settings.default_chunk_size
        self.min_sentences = min_sentences

    def chunk(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        source: str = "",
    ) -> list[Chunk]:
        """Split text into sentence-based chunks.

        Args:
            text: Text to split
            metadata: Metadata for chunks
            source: Source identifier

        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}
        sentences = self.SENTENCE_ENDINGS.split(text)

        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if (current_length + len(sentence) > self.chunk_size
                    and len(current_sentences) >= self.min_sentences):
                # Start new chunk
                chunks.append(
                    Chunk(
                        text=" ".join(current_sentences),
                        source=source,
                        metadata={
                            **metadata,
                            "chunk_index": len(chunks),
                        },
                    )
                )
                current_sentences = []
                current_length = 0

            current_sentences.append(sentence)
            current_length += len(sentence) + 1  # +1 for space

        # Add remaining sentences
        if current_sentences:
            chunks.append(
                Chunk(
                    text=" ".join(current_sentences),
                    source=source,
                    metadata={
                        **metadata,
                        "chunk_index": len(chunks),
                    },
                )
            )

        return chunks


def get_chunker(
    strategy: str = "recursive",
    **kwargs: Any,
) -> RecursiveChunker | SentenceChunker:
    """Get a chunker by strategy name.

    Args:
        strategy: Chunking strategy (recursive, sentence, chonkie)
        **kwargs: Arguments for chunker

    Returns:
        Configured chunker instance
    """
    if strategy == "recursive":
        return RecursiveChunker(**kwargs)
    elif strategy == "sentence":
        return SentenceChunker(**kwargs)
    elif strategy == "chonkie":
        return get_chonkie_chunker(**kwargs)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


def get_chonkie_chunker(**kwargs: Any) -> Any:
    """Get Chonkie chunker if installed.

    Returns:
        Chonkie chunker wrapped as Gweta chunker

    Raises:
        ImportError: If Chonkie is not installed
    """
    try:
        from chonkie import SemanticChunker
        # Return a wrapper that provides Gweta interface
        return ChonkieWrapper(SemanticChunker(**kwargs))
    except ImportError as e:
        raise ImportError(
            "Chonkie is required for semantic chunking. "
            "Install it with: pip install gweta[chonkie]"
        ) from e


class ChonkieWrapper:
    """Wrapper for Chonkie chunkers to provide Gweta interface."""

    def __init__(self, chonkie_chunker: Any) -> None:
        self._chunker = chonkie_chunker

    def chunk(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        source: str = "",
    ) -> list[Chunk]:
        """Chunk using Chonkie and convert to Gweta Chunks."""
        metadata = metadata or {}
        chonkie_chunks = self._chunker.chunk(text)

        return [
            Chunk(
                text=c.text,
                source=source,
                metadata={
                    **metadata,
                    "chunk_index": i,
                },
            )
            for i, c in enumerate(chonkie_chunks)
        ]
