"""Qdrant vector store adapter.

This module provides integration with Qdrant for
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


class QdrantStore(BaseStore):
    """Qdrant vector store adapter.

    Provides a Gweta interface for storing and querying
    chunks in Qdrant.

    Example:
        >>> store = QdrantStore("my_collection", url="http://localhost:6333")
        >>> await store.add(chunks)
        >>> results = await store.query("search text", n_results=5)
    """

    def __init__(
        self,
        collection_name: str,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        embedding_function: Callable[[list[str]], list[list[float]]] | None = None,
        vector_size: int = 384,
    ) -> None:
        """Initialize QdrantStore.

        Args:
            collection_name: Name of the collection
            url: Qdrant server URL
            api_key: Optional API key
            embedding_function: Function to generate embeddings
            vector_size: Dimension of embedding vectors
        """
        self._collection_name = collection_name
        self._url = url
        self._api_key = api_key
        self._embedding_function = embedding_function or _default_embedding_function
        self._vector_size = vector_size
        self._client = None

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def _ensure_client(self) -> Any:
        """Ensure Qdrant client is initialized."""
        if self._client is not None:
            return self._client

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as e:
            raise ImportError(
                "qdrant-client is required for QdrantStore. "
                "Install it with: pip install gweta[qdrant]"
            ) from e

        self._client = QdrantClient(url=self._url, api_key=self._api_key)

        # Create collection if it doesn't exist
        collections = self._client.get_collections().collections
        exists = any(c.name == self._collection_name for c in collections)

        if not exists:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {self._collection_name}")

        return self._client

    async def add(self, chunks: list[Chunk]) -> AddResult:
        """Add chunks to Qdrant.

        Args:
            chunks: List of chunks to add

        Returns:
            AddResult with operation details
        """
        if not chunks:
            return AddResult(added=0, skipped=0, errors=[])

        try:
            from qdrant_client.models import PointStruct
        except ImportError as e:
            raise ImportError(
                "qdrant-client is required for QdrantStore."
            ) from e

        client = self._ensure_client()

        # Generate embeddings
        texts = [c.text for c in chunks]
        embeddings = self._embedding_function(texts)

        # Create points
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point_id = chunk.id or str(uuid4())
            payload = {
                "text": chunk.text,
                "source": chunk.source,
                "quality_score": chunk.quality_score,
                **chunk.metadata,
            }
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        # Upsert points
        client.upsert(
            collection_name=self._collection_name,
            points=points,
        )

        logger.info(f"Added {len(points)} chunks to Qdrant collection {self._collection_name}")
        return AddResult(added=len(points), skipped=0, errors=[])

    async def query(
        self,
        query: str,
        n_results: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Query Qdrant for similar chunks.

        Args:
            query: Query text
            n_results: Number of results to return
            filter: Optional filter conditions

        Returns:
            List of matching chunks
        """
        client = self._ensure_client()

        # Generate query embedding
        query_embedding = self._embedding_function([query])[0]

        # Build filter if provided
        qdrant_filter = None
        if filter:
            try:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                conditions = []
                for key, value in filter.items():
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                qdrant_filter = Filter(must=conditions)
            except ImportError:
                pass

        # Search
        results = client.search(
            collection_name=self._collection_name,
            query_vector=query_embedding,
            limit=n_results,
            query_filter=qdrant_filter,
        )

        # Convert to chunks
        chunks = []
        for result in results:
            payload = result.payload or {}
            chunks.append(
                Chunk(
                    id=str(result.id),
                    text=payload.get("text", ""),
                    source=payload.get("source", ""),
                    quality_score=payload.get("quality_score"),
                    metadata={
                        k: v for k, v in payload.items()
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

        client = self._ensure_client()

        try:
            from qdrant_client.models import PointIdsList
        except ImportError as e:
            raise ImportError(
                "qdrant-client is required for QdrantStore."
            ) from e

        client.delete(
            collection_name=self._collection_name,
            points_selector=PointIdsList(points=chunk_ids),
        )

        logger.info(f"Deleted {len(chunk_ids)} chunks from {self._collection_name}")
        return len(chunk_ids)

    async def get_all(self) -> list[Chunk]:
        """Get all chunks from the collection.

        Returns:
            List of all chunks
        """
        client = self._ensure_client()

        # Scroll through all points
        chunks = []
        offset = None

        while True:
            results, offset = client.scroll(
                collection_name=self._collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in results:
                payload = point.payload or {}
                chunks.append(
                    Chunk(
                        id=str(point.id),
                        text=payload.get("text", ""),
                        source=payload.get("source", ""),
                        quality_score=payload.get("quality_score"),
                        metadata={
                            k: v for k, v in payload.items()
                            if k not in ("text", "source", "quality_score")
                        },
                    )
                )

            if offset is None:
                break

        return chunks

    def get_stats(self) -> StoreStats:
        """Get collection statistics.

        Returns:
            StoreStats with collection info
        """
        client = self._ensure_client()
        info = client.get_collection(self._collection_name)

        return StoreStats(
            collection_name=self._collection_name,
            chunk_count=info.points_count,
            metadata={
                "vectors_count": info.vectors_count,
                "status": info.status.value if info.status else "unknown",
            },
        )
