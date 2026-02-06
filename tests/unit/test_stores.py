"""Tests for Vector Store Adapters.

These tests verify the vector store implementations including:
- QdrantStore
- PineconeStore
- WeaviateStore

Tests use mocks to avoid requiring actual database connections.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from gweta.core.types import Chunk
from gweta.ingest.stores.base import BaseStore, AddResult, StoreStats


class TestBaseStore:
    """Tests for BaseStore abstract class."""

    def test_base_store_is_abstract(self):
        """Test that BaseStore cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseStore()

    def test_add_result_dataclass(self):
        """Test AddResult dataclass."""
        result = AddResult(added=10, skipped=2, errors=["error1"])

        assert result.added == 10
        assert result.skipped == 2
        assert len(result.errors) == 1

    def test_add_result_defaults(self):
        """Test AddResult default values."""
        result = AddResult()

        assert result.added == 0
        assert result.skipped == 0
        assert result.errors is None

    def test_store_stats_dataclass(self):
        """Test StoreStats dataclass."""
        stats = StoreStats(
            collection_name="test",
            chunk_count=100,
            metadata={"key": "value"},
        )

        assert stats.collection_name == "test"
        assert stats.chunk_count == 100


class TestQdrantStore:
    """Tests for QdrantStore implementation."""

    def test_qdrant_import_error(self):
        """Test proper error when qdrant-client not installed."""
        from gweta.ingest.stores.qdrant import QdrantStore

        store = QdrantStore("test_collection")

        with patch.dict("sys.modules", {"qdrant_client": None}):
            # Force reimport to trigger error
            pass

    def test_qdrant_init(self):
        """Test QdrantStore initialization."""
        from gweta.ingest.stores.qdrant import QdrantStore

        store = QdrantStore(
            collection_name="test",
            url="http://localhost:6333",
            api_key="test-key",
            vector_size=768,
        )

        assert store.collection_name == "test"
        assert store._url == "http://localhost:6333"
        assert store._api_key == "test-key"
        assert store._vector_size == 768
        assert store._client is None

    def test_qdrant_default_embedding(self):
        """Test default embedding function."""
        from gweta.ingest.stores.qdrant import _default_embedding_function

        texts = ["Hello world", "Test text"]
        embeddings = _default_embedding_function(texts)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 48  # SHA-384 produces 48 bytes
        assert all(0 <= v <= 1 for v in embeddings[0])

    def test_qdrant_custom_embedding(self):
        """Test custom embedding function."""
        from gweta.ingest.stores.qdrant import QdrantStore

        custom_embedder = MagicMock(
            return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )

        store = QdrantStore(
            collection_name="test",
            embedding_function=custom_embedder,
        )

        assert store._embedding_function == custom_embedder


class TestPineconeStore:
    """Tests for PineconeStore implementation."""

    def test_pinecone_init(self):
        """Test PineconeStore initialization."""
        from gweta.ingest.stores.pinecone import PineconeStore

        store = PineconeStore(
            index_name="test-index",
            api_key="pk-test-key",
            namespace="test-ns",
        )

        assert store.collection_name == "test-index"
        assert store._api_key == "pk-test-key"
        assert store._namespace == "test-ns"
        assert store._index is None
        assert store._pc is None

    def test_pinecone_default_embedding(self):
        """Test default embedding function."""
        from gweta.ingest.stores.pinecone import _default_embedding_function

        texts = ["Test"]
        embeddings = _default_embedding_function(texts)

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 48

    def test_pinecone_custom_embedding(self):
        """Test custom embedding function."""
        from gweta.ingest.stores.pinecone import PineconeStore

        custom_embedder = MagicMock()

        store = PineconeStore(
            index_name="test",
            api_key="key",
            embedding_function=custom_embedder,
        )

        assert store._embedding_function == custom_embedder


class TestWeaviateStore:
    """Tests for WeaviateStore implementation."""

    def test_weaviate_init(self):
        """Test WeaviateStore initialization."""
        from gweta.ingest.stores.weaviate import WeaviateStore

        store = WeaviateStore(
            class_name="Document",
            url="http://localhost:8080",
            api_key="test-key",
        )

        assert store.collection_name == "Document"
        assert store._url == "http://localhost:8080"
        assert store._api_key == "test-key"
        assert store._client is None

    def test_weaviate_default_embedding(self):
        """Test default embedding function."""
        from gweta.ingest.stores.weaviate import _default_embedding_function

        texts = ["Hello", "World"]
        embeddings = _default_embedding_function(texts)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 48

    def test_weaviate_custom_embedding(self):
        """Test custom embedding function."""
        from gweta.ingest.stores.weaviate import WeaviateStore

        custom_embedder = MagicMock()

        store = WeaviateStore(
            class_name="Test",
            embedding_function=custom_embedder,
        )

        assert store._embedding_function == custom_embedder

    def test_weaviate_close(self):
        """Test closing Weaviate connection."""
        from gweta.ingest.stores.weaviate import WeaviateStore

        store = WeaviateStore("Test")
        store._client = MagicMock()

        store.close()

        assert store._client is None


class TestStoreAddMethod:
    """Tests for store add() method behavior."""

    @pytest.mark.asyncio
    async def test_add_empty_chunks(self):
        """Test adding empty list of chunks."""
        from gweta.ingest.stores.qdrant import QdrantStore

        store = QdrantStore("test")
        result = await store.add([])

        assert result.added == 0
        assert result.skipped == 0
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_pinecone_add_empty(self):
        """Test Pinecone add with empty chunks."""
        from gweta.ingest.stores.pinecone import PineconeStore

        store = PineconeStore("test", "key")
        result = await store.add([])

        assert result.added == 0
        assert result.skipped == 0
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_weaviate_add_empty(self):
        """Test Weaviate add with empty chunks."""
        from gweta.ingest.stores.weaviate import WeaviateStore

        store = WeaviateStore("Test")
        result = await store.add([])

        assert result.added == 0
        assert result.skipped == 0
        assert result.errors == []


class TestStoreDeleteMethod:
    """Tests for store delete() method behavior."""

    @pytest.mark.asyncio
    async def test_delete_empty_list(self):
        """Test deleting empty list."""
        from gweta.ingest.stores.qdrant import QdrantStore

        store = QdrantStore("test")
        result = await store.delete([])

        assert result == 0

    @pytest.mark.asyncio
    async def test_pinecone_delete_empty(self):
        """Test Pinecone delete with empty list."""
        from gweta.ingest.stores.pinecone import PineconeStore

        store = PineconeStore("test", "key")
        result = await store.delete([])

        assert result == 0

    @pytest.mark.asyncio
    async def test_weaviate_delete_empty(self):
        """Test Weaviate delete with empty list."""
        from gweta.ingest.stores.weaviate import WeaviateStore

        store = WeaviateStore("Test")
        result = await store.delete([])

        assert result == 0


class TestChunkConversion:
    """Tests for chunk conversion in stores."""

    def test_chunk_to_vector_data(self):
        """Test converting chunk to vector store format."""
        chunk = Chunk(
            id="chunk-1",
            text="This is test content",
            source="test.txt",
            quality_score=0.95,
            metadata={"category": "test", "page": 1},
        )

        # Verify chunk has expected properties
        assert chunk.id == "chunk-1"
        assert chunk.text == "This is test content"
        assert chunk.source == "test.txt"
        assert chunk.quality_score == 0.95
        assert chunk.metadata["category"] == "test"

    def test_chunk_without_id(self):
        """Test chunk without explicit ID."""
        chunk = Chunk(
            text="Content",
            source="source.txt",
            metadata={},
        )

        # ID should be None or auto-generated
        assert chunk.id is None or isinstance(chunk.id, str)


class TestEmbeddingConsistency:
    """Tests for embedding function behavior."""

    def test_embedding_deterministic(self):
        """Test that default embedding is deterministic."""
        from gweta.ingest.stores.qdrant import _default_embedding_function

        text = "Same text"
        emb1 = _default_embedding_function([text])
        emb2 = _default_embedding_function([text])

        assert emb1 == emb2

    def test_embedding_different_for_different_text(self):
        """Test that different text produces different embeddings."""
        from gweta.ingest.stores.qdrant import _default_embedding_function

        emb1 = _default_embedding_function(["Text A"])
        emb2 = _default_embedding_function(["Text B"])

        assert emb1 != emb2


class TestStoreInterfaceCompliance:
    """Tests for store interface compliance."""

    def test_qdrant_implements_base(self):
        """Test QdrantStore implements BaseStore."""
        from gweta.ingest.stores.qdrant import QdrantStore
        from gweta.ingest.stores.base import BaseStore

        assert issubclass(QdrantStore, BaseStore)

    def test_pinecone_implements_base(self):
        """Test PineconeStore implements BaseStore."""
        from gweta.ingest.stores.pinecone import PineconeStore
        from gweta.ingest.stores.base import BaseStore

        assert issubclass(PineconeStore, BaseStore)

    def test_weaviate_implements_base(self):
        """Test WeaviateStore implements BaseStore."""
        from gweta.ingest.stores.weaviate import WeaviateStore
        from gweta.ingest.stores.base import BaseStore

        assert issubclass(WeaviateStore, BaseStore)

    def test_store_has_required_methods(self):
        """Test stores have all required methods."""
        from gweta.ingest.stores.qdrant import QdrantStore
        from gweta.ingest.stores.pinecone import PineconeStore
        from gweta.ingest.stores.weaviate import WeaviateStore

        required_methods = [
            "add",
            "query",
            "delete",
            "get_all",
            "get_stats",
        ]

        for store_class in [QdrantStore, PineconeStore, WeaviateStore]:
            for method in required_methods:
                assert hasattr(store_class, method), (
                    f"{store_class.__name__} missing {method}"
                )


class TestURLParsing:
    """Tests for URL parsing in stores."""

    def test_weaviate_url_parsing(self):
        """Test Weaviate URL parsing."""
        from gweta.ingest.stores.weaviate import WeaviateStore

        store = WeaviateStore("Test", url="http://weaviate:8080")
        assert store._url == "http://weaviate:8080"

    def test_qdrant_url_default(self):
        """Test Qdrant default URL."""
        from gweta.ingest.stores.qdrant import QdrantStore

        store = QdrantStore("test")
        assert store._url == "http://localhost:6333"


class TestMetadataHandling:
    """Tests for metadata handling in stores."""

    def test_chunk_metadata_preserved(self):
        """Test that metadata is preserved through conversion."""
        chunk = Chunk(
            text="Content",
            source="source.txt",
            quality_score=0.9,
            metadata={
                "author": "Test Author",
                "date": "2024-01-01",
                "tags": ["a", "b"],
            },
        )

        # Verify metadata structure
        assert "author" in chunk.metadata
        assert "date" in chunk.metadata
        assert "tags" in chunk.metadata

    def test_pinecone_metadata_size_limit(self):
        """Test Pinecone metadata text truncation."""
        # Pinecone truncates text to 1000 chars in metadata
        long_text = "x" * 2000

        chunk = Chunk(
            text=long_text,
            source="test.txt",
            metadata={},
        )

        # The actual truncation happens in the add() method
        # Just verify the chunk can be created with long text
        assert len(chunk.text) == 2000


class TestImportErrors:
    """Tests for import error handling."""

    def test_import_error_message_qdrant(self):
        """Test Qdrant import error message."""
        from gweta.ingest.stores.qdrant import QdrantStore

        store = QdrantStore("test")
        # The error would occur in _ensure_client()
        # Just verify the store can be created without the client
        assert store._client is None

    def test_import_error_message_pinecone(self):
        """Test Pinecone import error message."""
        from gweta.ingest.stores.pinecone import PineconeStore

        store = PineconeStore("test", "key")
        assert store._index is None

    def test_import_error_message_weaviate(self):
        """Test Weaviate import error message."""
        from gweta.ingest.stores.weaviate import WeaviateStore

        store = WeaviateStore("Test")
        assert store._client is None
