"""Weaviate vector store adapter.

This module provides integration with Weaviate for
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


class WeaviateStore(BaseStore):
    """Weaviate vector store adapter.

    Provides a Gweta interface for storing and querying
    chunks in Weaviate.

    Example:
        >>> store = WeaviateStore("Document", url="http://localhost:8080")
        >>> await store.add(chunks)
        >>> results = await store.query("search text", n_results=5)
    """

    def __init__(
        self,
        class_name: str,
        url: str = "http://localhost:8080",
        api_key: str | None = None,
        embedding_function: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        """Initialize WeaviateStore.

        Args:
            class_name: Weaviate class name
            url: Weaviate server URL
            api_key: Optional API key
            embedding_function: Function to generate embeddings
        """
        self._class_name = class_name
        self._url = url
        self._api_key = api_key
        self._embedding_function = embedding_function or _default_embedding_function
        self._client = None

    @property
    def collection_name(self) -> str:
        return self._class_name

    def _ensure_client(self) -> Any:
        """Ensure Weaviate client is initialized."""
        if self._client is not None:
            return self._client

        try:
            import weaviate
            from weaviate.classes.init import Auth
        except ImportError as e:
            raise ImportError(
                "weaviate-client is required for WeaviateStore. "
                "Install it with: pip install gweta[weaviate]"
            ) from e

        # Parse URL
        host = self._url.replace("http://", "").replace("https://", "")
        if ":" in host:
            host, port = host.split(":")
            port = int(port)
        else:
            port = 8080

        # Connect to Weaviate
        if self._api_key:
            self._client = weaviate.connect_to_local(
                host=host,
                port=port,
                auth_credentials=Auth.api_key(self._api_key),
            )
        else:
            self._client = weaviate.connect_to_local(
                host=host,
                port=port,
            )

        # Ensure collection exists
        try:
            if not self._client.collections.exists(self._class_name):
                self._client.collections.create(
                    name=self._class_name,
                    properties=[
                        {"name": "text", "dataType": ["text"]},
                        {"name": "source", "dataType": ["text"]},
                        {"name": "quality_score", "dataType": ["number"]},
                        {"name": "chunk_id", "dataType": ["text"]},
                    ],
                )
                logger.info(f"Created Weaviate class: {self._class_name}")
        except Exception as e:
            logger.warning(f"Could not create collection: {e}")

        return self._client

    async def add(self, chunks: list[Chunk]) -> AddResult:
        """Add chunks to Weaviate.

        Args:
            chunks: List of chunks to add

        Returns:
            AddResult with operation details
        """
        if not chunks:
            return AddResult(added=0, skipped=0, errors=[])

        client = self._ensure_client()
        collection = client.collections.get(self._class_name)

        # Generate embeddings
        texts = [c.text for c in chunks]
        embeddings = self._embedding_function(texts)

        # Add objects
        added = 0
        errors = []

        for chunk, embedding in zip(chunks, embeddings):
            try:
                chunk_id = chunk.id or str(uuid4())
                properties = {
                    "text": chunk.text,
                    "source": chunk.source,
                    "quality_score": chunk.quality_score or 0.0,
                    "chunk_id": chunk_id,
                }

                collection.data.insert(
                    properties=properties,
                    vector=embedding,
                )
                added += 1

            except Exception as e:
                errors.append(str(e))
                logger.error(f"Error adding chunk: {e}")

        logger.info(f"Added {added} chunks to Weaviate class {self._class_name}")
        return AddResult(added=added, skipped=0, errors=errors)

    async def query(
        self,
        query: str,
        n_results: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Query Weaviate for similar chunks.

        Args:
            query: Query text
            n_results: Number of results to return
            filter: Optional filter conditions

        Returns:
            List of matching chunks
        """
        client = self._ensure_client()
        collection = client.collections.get(self._class_name)

        # Generate query embedding
        query_embedding = self._embedding_function([query])[0]

        # Query Weaviate
        try:
            results = collection.query.near_vector(
                near_vector=query_embedding,
                limit=n_results,
                return_properties=["text", "source", "quality_score", "chunk_id"],
            )

            # Convert to chunks
            chunks = []
            for obj in results.objects:
                props = obj.properties
                chunks.append(
                    Chunk(
                        id=props.get("chunk_id", str(obj.uuid)),
                        text=props.get("text", ""),
                        source=props.get("source", ""),
                        quality_score=props.get("quality_score"),
                        metadata={},
                    )
                )

            return chunks

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

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
        collection = client.collections.get(self._class_name)

        deleted = 0
        for chunk_id in chunk_ids:
            try:
                # Find object by chunk_id property
                results = collection.query.fetch_objects(
                    filters=collection.filter.by_property("chunk_id").equal(chunk_id),
                    limit=1,
                )
                if results.objects:
                    collection.data.delete_by_id(results.objects[0].uuid)
                    deleted += 1
            except Exception as e:
                logger.error(f"Error deleting chunk {chunk_id}: {e}")

        logger.info(f"Deleted {deleted} chunks from {self._class_name}")
        return deleted

    async def get_all(self) -> list[Chunk]:
        """Get all chunks from the class.

        Returns:
            List of all chunks
        """
        client = self._ensure_client()
        collection = client.collections.get(self._class_name)

        chunks = []

        try:
            # Iterate through all objects
            for obj in collection.iterator(
                include_vector=False,
                return_properties=["text", "source", "quality_score", "chunk_id"],
            ):
                props = obj.properties
                chunks.append(
                    Chunk(
                        id=props.get("chunk_id", str(obj.uuid)),
                        text=props.get("text", ""),
                        source=props.get("source", ""),
                        quality_score=props.get("quality_score"),
                        metadata={},
                    )
                )

        except Exception as e:
            logger.error(f"Error getting all chunks: {e}")

        return chunks

    def get_stats(self) -> StoreStats:
        """Get class statistics.

        Returns:
            StoreStats with class info
        """
        try:
            client = self._ensure_client()
            collection = client.collections.get(self._class_name)
            count = collection.aggregate.over_all(total_count=True).total_count
        except Exception:
            count = 0

        return StoreStats(
            collection_name=self._class_name,
            chunk_count=count,
            metadata={"url": self._url},
        )

    def close(self) -> None:
        """Close the Weaviate client connection."""
        if self._client:
            self._client.close()
            self._client = None
