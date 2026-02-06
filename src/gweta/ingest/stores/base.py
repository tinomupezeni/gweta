"""Abstract base class for vector store adapters.

This module defines the interface that all vector store
adapters must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from gweta.core.types import Chunk


@dataclass
class StoreStats:
    """Statistics about a vector store collection.

    Attributes:
        collection_name: Name of the collection
        chunk_count: Number of chunks stored
        dimension: Embedding dimension
        metadata: Additional store-specific metadata
    """
    collection_name: str
    chunk_count: int = 0
    dimension: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class AddResult:
    """Result of adding chunks to a store.

    Attributes:
        added: Number of chunks successfully added
        skipped: Number of chunks skipped (duplicates)
        errors: List of error messages
    """
    added: int = 0
    skipped: int = 0
    errors: list[str] | None = None


class BaseStore(ABC):
    """Abstract base class for vector store adapters.

    All vector store implementations must inherit from this
    class and implement the abstract methods.

    Example:
        >>> class MyStore(BaseStore):
        ...     async def add(self, chunks): ...
        ...     async def query(self, query, n_results): ...
        ...     async def delete(self, chunk_ids): ...
        ...     async def get_all(self): ...
        ...     def get_stats(self): ...
    """

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Get the collection/index name."""
        ...

    @abstractmethod
    async def add(self, chunks: list[Chunk]) -> AddResult:
        """Add chunks to the store.

        Args:
            chunks: List of chunks to add

        Returns:
            AddResult with add statistics
        """
        ...

    @abstractmethod
    async def query(
        self,
        query: str,
        n_results: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Query the store for similar chunks.

        Args:
            query: Query text
            n_results: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of matching Chunk objects
        """
        ...

    @abstractmethod
    async def delete(self, chunk_ids: list[str]) -> int:
        """Delete chunks by ID.

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            Number of chunks deleted
        """
        ...

    @abstractmethod
    async def get_all(self) -> list[Chunk]:
        """Get all chunks from the store.

        Returns:
            List of all Chunk objects
        """
        ...

    @abstractmethod
    def get_stats(self) -> StoreStats:
        """Get statistics about the store.

        Returns:
            StoreStats with collection info
        """
        ...

    async def update(self, chunk: Chunk) -> bool:
        """Update an existing chunk.

        Default implementation deletes and re-adds.

        Args:
            chunk: Chunk to update (must have existing ID)

        Returns:
            True if update succeeded
        """
        await self.delete([chunk.id])
        result = await self.add([chunk])
        return result.added > 0

    async def search_by_metadata(
        self,
        filter: dict[str, Any],
        limit: int = 100,
    ) -> list[Chunk]:
        """Search chunks by metadata.

        Default implementation gets all and filters.
        Override for better performance.

        Args:
            filter: Metadata key-value pairs to match
            limit: Maximum results

        Returns:
            List of matching chunks
        """
        all_chunks = await self.get_all()
        matching = []

        for chunk in all_chunks:
            if all(
                chunk.metadata.get(k) == v
                for k, v in filter.items()
            ):
                matching.append(chunk)
                if len(matching) >= limit:
                    break

        return matching
