"""Pinecone vector store adapter.

This module provides integration with Pinecone for
storing and querying validated chunks.
"""

from typing import Any, Callable
from uuid import uuid4

from gweta.core.logging import get_logger
from gweta.core.types import Chunk
from gweta.ingest.stores.base import AddResult, BaseStore, StoreStats

logger = get_logger(__name__)


def _default_embedding_function(texts: list[str]) -> list[list[float]]:
    """Default embedding function using simple hashing.

    This is a placeholder. For production use, provide
    a real embedding function.
    """
    import hashlib

    embeddings = []
    for text in texts:
        # Create a simple hash-based embedding (384 dimensions)
        hash_bytes = hashlib.sha384(text.encode()).digest()
        embedding = [float(b) / 255.0 for b in hash_bytes]
        embeddings.append(embedding)
    return embeddings


class PineconeStore(BaseStore):
    """Pinecone vector store adapter.

    Provides a Gweta interface for storing and querying
    chunks in Pinecone.

    Example:
        >>> store = PineconeStore("my-index", api_key="...")
        >>> await store.add(chunks)
        >>> results = await store.query("search text", n_results=5)
    """

    def __init__(
        self,
        index_name: str,
        api_key: str,
        environment: str | None = None,
        namespace: str = "",
        embedding_function: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        """Initialize PineconeStore.

        Args:
            index_name: Name of the Pinecone index
            api_key: Pinecone API key
            environment: Pinecone environment (deprecated in newer SDK)
            namespace: Namespace within the index
            embedding_function: Function to generate embeddings
        """
        self._index_name = index_name
        self._api_key = api_key
        self._environment = environment
        self._namespace = namespace
        self._embedding_function = embedding_function or _default_embedding_function
        self._index = None
        self._pc = None

    @property
    def collection_name(self) -> str:
        return self._index_name

    def _ensure_index(self) -> Any:
        """Ensure Pinecone index is initialized."""
        if self._index is not None:
            return self._index

        try:
            from pinecone import Pinecone
        except ImportError as e:
            raise ImportError(
                "pinecone-client is required for PineconeStore. "
                "Install it with: pip install gweta[pinecone]"
            ) from e

        self._pc = Pinecone(api_key=self._api_key)
        self._index = self._pc.Index(self._index_name)
        return self._index

    async def add(self, chunks: list[Chunk]) -> AddResult:
        """Add chunks to Pinecone.

        Args:
            chunks: List of chunks to add

        Returns:
            AddResult with operation details
        """
        if not chunks:
            return AddResult(added=0, skipped=0, errors=[])

        index = self._ensure_index()

        # Generate embeddings
        texts = [c.text for c in chunks]
        embeddings = self._embedding_function(texts)

        # Prepare vectors for upsert
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vector_id = chunk.id or str(uuid4())
            metadata = {
                "text": chunk.text[:1000],  # Pinecone has metadata size limits
                "source": chunk.source,
                "quality_score": chunk.quality_score or 0.0,
                **{k: v for k, v in chunk.metadata.items() if isinstance(v, (str, int, float, bool))},
            }
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata,
            })

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch, namespace=self._namespace)

        logger.info(f"Added {len(vectors)} chunks to Pinecone index {self._index_name}")
        return AddResult(added=len(vectors), skipped=0, errors=[])

    async def query(
        self,
        query: str,
        n_results: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Query Pinecone for similar chunks.

        Args:
            query: Query text
            n_results: Number of results to return
            filter: Optional filter conditions

        Returns:
            List of matching chunks
        """
        index = self._ensure_index()

        # Generate query embedding
        query_embedding = self._embedding_function([query])[0]

        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=n_results,
            include_metadata=True,
            namespace=self._namespace,
            filter=filter,
        )

        # Convert to chunks
        chunks = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            chunks.append(
                Chunk(
                    id=match["id"],
                    text=metadata.get("text", ""),
                    source=metadata.get("source", ""),
                    quality_score=metadata.get("quality_score"),
                    metadata={
                        k: v for k, v in metadata.items()
                        if k not in ("text", "source", "quality_score")
                    },
                )
            )

        return chunks

    async def delete(self, chunk_ids: list[str]) -> int:
        """Delete chunks by ID.

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            Number of deleted chunks
        """
        if not chunk_ids:
            return 0

        index = self._ensure_index()

        # Delete by IDs
        index.delete(ids=chunk_ids, namespace=self._namespace)

        logger.info(f"Deleted {len(chunk_ids)} chunks from {self._index_name}")
        return len(chunk_ids)

    async def get_all(self) -> list[Chunk]:
        """Get all chunks from the index.

        Note: Pinecone doesn't support listing all vectors directly.
        This is a limited implementation that may not return all vectors.

        Returns:
            List of chunks (may be incomplete)
        """
        # Pinecone requires a query to retrieve vectors
        # We'll use a zero vector to try to get some results
        logger.warning(
            "PineconeStore.get_all() is limited. "
            "Pinecone doesn't support listing all vectors."
        )

        index = self._ensure_index()

        # Get index stats
        stats = index.describe_index_stats()
        total = stats.get("total_vector_count", 0)

        if total == 0:
            return []

        # Query with zero vector to get some results
        # This won't return all vectors, just the closest to zero
        dimension = stats.get("dimension", 384)
        zero_vector = [0.0] * dimension

        results = index.query(
            vector=zero_vector,
            top_k=min(total, 10000),  # Pinecone limit
            include_metadata=True,
            namespace=self._namespace,
        )

        chunks = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            chunks.append(
                Chunk(
                    id=match["id"],
                    text=metadata.get("text", ""),
                    source=metadata.get("source", ""),
                    quality_score=metadata.get("quality_score"),
                    metadata={
                        k: v for k, v in metadata.items()
                        if k not in ("text", "source", "quality_score")
                    },
                )
            )

        return chunks

    def get_stats(self) -> StoreStats:
        """Get index statistics.

        Returns:
            StoreStats with index info
        """
        index = self._ensure_index()
        stats = index.describe_index_stats()

        return StoreStats(
            collection_name=self._index_name,
            chunk_count=stats.get("total_vector_count", 0),
            metadata={
                "dimension": stats.get("dimension"),
                "namespaces": list(stats.get("namespaces", {}).keys()),
            },
        )
