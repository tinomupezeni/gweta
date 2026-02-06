"""LangChain adapter for Gweta.

This module provides conversion between LangChain
Document objects and Gweta Chunk objects.
"""

from typing import Any

from gweta.core.types import Chunk


class LangChainAdapter:
    """Adapter for LangChain Document objects.

    Converts between LangChain Documents and Gweta Chunks,
    preserving metadata and quality information.

    Example:
        >>> adapter = LangChainAdapter()
        >>> chunks = adapter.from_documents(langchain_docs)
        >>> docs = adapter.to_documents(gweta_chunks)
    """

    @staticmethod
    def from_documents(documents: list[Any]) -> list[Chunk]:
        """Convert LangChain Documents to Gweta Chunks.

        Args:
            documents: List of LangChain Document objects

        Returns:
            List of Gweta Chunk objects
        """
        chunks = []
        for doc in documents:
            metadata = dict(doc.metadata) if hasattr(doc, "metadata") else {}
            source = metadata.pop("source", "")

            chunks.append(
                Chunk(
                    text=doc.page_content if hasattr(doc, "page_content") else str(doc),
                    metadata=metadata,
                    source=source,
                )
            )
        return chunks

    @staticmethod
    def to_documents(chunks: list[Chunk]) -> list[Any]:
        """Convert Gweta Chunks to LangChain Documents.

        Args:
            chunks: List of Gweta Chunk objects

        Returns:
            List of LangChain Document objects
        """
        try:
            from langchain_core.documents import Document
        except ImportError:
            try:
                from langchain.schema import Document
            except ImportError as e:
                raise ImportError(
                    "LangChain is required for this adapter. "
                    "Install it with: pip install langchain-core"
                ) from e

        documents = []
        for chunk in chunks:
            metadata = {
                **chunk.metadata,
                "source": chunk.source,
            }
            if chunk.quality_score is not None:
                metadata["quality_score"] = chunk.quality_score

            documents.append(
                Document(
                    page_content=chunk.text,
                    metadata=metadata,
                )
            )
        return documents

    @staticmethod
    def wrap_validator(validator: Any) -> Any:
        """Create a LangChain-compatible validator wrapper.

        Args:
            validator: Gweta ChunkValidator

        Returns:
            Callable that validates LangChain documents
        """
        adapter = LangChainAdapter()

        def validate_documents(documents: list[Any]) -> list[Any]:
            chunks = adapter.from_documents(documents)
            report = validator.validate_batch(chunks)
            return adapter.to_documents(report.accepted())

        return validate_documents
