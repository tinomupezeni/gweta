"""Chroma vector store adapter.

This module provides integration with ChromaDB for
storing and querying validated chunks.
"""

from typing import Any, Callable

from gweta.core.exceptions import IngestionError
from gweta.core.logging import get_logger
from gweta.core.types import Chunk
from gweta.ingest.stores.base import AddResult, BaseStore, StoreStats

logger = get_logger(__name__)


def _get_default_embedding_function() -> Any:
    """Get the default embedding function.

    Uses ChromaDB's default SentenceTransformer embedding function.
    Falls back to None if sentence-transformers is not installed.

    Returns:
        Embedding function or None
    """
    try:
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )
        return SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. "
            "Install with: pip install sentence-transformers "
            "or provide a custom embedding_function."
        )
        return None


class ChromaStore(BaseStore):
    """ChromaDB vector store adapter.

    Provides a Gweta interface for storing and querying
    chunks in ChromaDB.

    Example:
        Basic usage (uses default all-MiniLM-L6-v2 embeddings):

        >>> store = ChromaStore("my_collection")
        >>> await store.add(chunks)
        >>> results = await store.query("What is...", n_results=5)

        With custom embedding function:

        >>> from chromadb.utils.embedding_functions import (
        ...     SentenceTransformerEmbeddingFunction
        ... )
        >>> embed_fn = SentenceTransformerEmbeddingFunction(
        ...     model_name="all-mpnet-base-v2"
        ... )
        >>> store = ChromaStore("my_collection", embedding_function=embed_fn)

        With OpenAI embeddings:

        >>> from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        >>> embed_fn = OpenAIEmbeddingFunction(api_key="sk-...")
        >>> store = ChromaStore("my_collection", embedding_function=embed_fn)

        With persistence:

        >>> store = ChromaStore(
        ...     "my_collection",
        ...     persist_directory="./chroma_data"
        ... )
    """

    def __init__(
        self,
        collection_name: str,
        client: Any = None,
        embedding_function: Any | None = None,
        persist_directory: str | None = None,
        use_default_embeddings: bool = True,
    ) -> None:
        """Initialize ChromaStore.

        Args:
            collection_name: Name of the collection
            client: Existing ChromaDB client (creates new if not provided)
            embedding_function: Custom embedding function. If not provided and
                use_default_embeddings is True, uses SentenceTransformer
                with "all-MiniLM-L6-v2" model.
            persist_directory: Directory for persistence (None for in-memory)
            use_default_embeddings: If True and no embedding_function provided,
                uses the default SentenceTransformer embeddings. Set to False
                to disable and use ChromaDB's internal default. Default: True.
        """
        self._collection_name = collection_name
        self._client = client
        self._persist_directory = persist_directory
        self._collection = None

        # Set up embedding function
        if embedding_function is not None:
            self._embedding_function = embedding_function
        elif use_default_embeddings:
            self._embedding_function = _get_default_embedding_function()
        else:
            self._embedding_function = None

    @property
    def collection_name(self) -> str:
        """Get collection name."""
        return self._collection_name

    def _ensure_collection(self) -> Any:
        """Ensure collection is initialized."""
        if self._collection is not None:
            return self._collection

        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "ChromaDB is required for ChromaStore. "
                "Install it with: pip install gweta[chroma]"
            ) from e

        if self._client is None:
            if self._persist_directory:
                self._client = chromadb.PersistentClient(
                    path=self._persist_directory
                )
            else:
                self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_function,
        )

        return self._collection

    async def add(self, chunks: list[Chunk]) -> AddResult:
        """Add chunks to Chroma.

        Args:
            chunks: Chunks to add

        Returns:
            AddResult with statistics
        """
        if not chunks:
            return AddResult(added=0)

        collection = self._ensure_collection()

        try:
            # Prepare data for Chroma
            ids = [chunk.id for chunk in chunks]
            documents = [chunk.text for chunk in chunks]
            metadatas = []

            for chunk in chunks:
                # Chroma metadata must be str, int, float, or bool
                metadata = {}
                for key, value in chunk.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    elif value is not None:
                        metadata[key] = str(value)

                # Add quality metadata
                if chunk.quality_score is not None:
                    metadata["quality_score"] = chunk.quality_score
                metadata["source"] = chunk.source

                metadatas.append(metadata)

            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

            logger.info(f"Added {len(chunks)} chunks to {self._collection_name}")
            return AddResult(added=len(chunks))

        except Exception as e:
            logger.error(f"Failed to add chunks: {e}")
            raise IngestionError(
                f"Failed to add chunks to Chroma: {e}",
                store_type="chroma",
                collection=self._collection_name,
                chunk_count=len(chunks),
            ) from e

    async def query(
        self,
        query: str,
        n_results: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Query Chroma for similar chunks.

        Args:
            query: Query text
            n_results: Number of results
            filter: Metadata filter

        Returns:
            List of matching Chunk objects
        """
        collection = self._ensure_collection()

        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter,
            )

            chunks: list[Chunk] = []
            if results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            text=results["documents"][0][i] if results["documents"] else "",
                            metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                            source=results["metadatas"][0][i].get("source", "") if results["metadatas"] else "",
                        )
                    )

            return chunks

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

    async def delete(self, chunk_ids: list[str]) -> int:
        """Delete chunks by ID.

        Args:
            chunk_ids: IDs to delete

        Returns:
            Number deleted
        """
        if not chunk_ids:
            return 0

        collection = self._ensure_collection()

        try:
            collection.delete(ids=chunk_ids)
            return len(chunk_ids)
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return 0

    async def get_all(self) -> list[Chunk]:
        """Get all chunks from collection.

        Returns:
            All chunks
        """
        collection = self._ensure_collection()

        try:
            results = collection.get()

            chunks: list[Chunk] = []
            if results["ids"]:
                for i, chunk_id in enumerate(results["ids"]):
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            text=results["documents"][i] if results["documents"] else "",
                            metadata=results["metadatas"][i] if results["metadatas"] else {},
                            source=results["metadatas"][i].get("source", "") if results["metadatas"] else "",
                        )
                    )

            return chunks

        except Exception as e:
            logger.error(f"Get all failed: {e}")
            return []

    def get_stats(self) -> StoreStats:
        """Get collection statistics.

        Returns:
            StoreStats
        """
        collection = self._ensure_collection()

        return StoreStats(
            collection_name=self._collection_name,
            chunk_count=collection.count(),
        )
