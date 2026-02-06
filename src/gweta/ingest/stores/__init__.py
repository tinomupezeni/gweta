"""Vector store adapters.

This module provides adapters for loading validated
chunks into various vector databases.
"""

from gweta.ingest.stores.base import BaseStore, StoreStats, AddResult

__all__ = [
    "BaseStore",
    "StoreStats",
    "AddResult",
]

# Lazy imports for optional stores
def get_chroma_store(*args, **kwargs):
    """Get ChromaStore (requires chromadb)."""
    from gweta.ingest.stores.chroma import ChromaStore
    return ChromaStore(*args, **kwargs)

def get_qdrant_store(*args, **kwargs):
    """Get QdrantStore (requires qdrant-client)."""
    from gweta.ingest.stores.qdrant import QdrantStore
    return QdrantStore(*args, **kwargs)

def get_pinecone_store(*args, **kwargs):
    """Get PineconeStore (requires pinecone-client)."""
    from gweta.ingest.stores.pinecone import PineconeStore
    return PineconeStore(*args, **kwargs)

def get_weaviate_store(*args, **kwargs):
    """Get WeaviateStore (requires weaviate-client)."""
    from gweta.ingest.stores.weaviate import WeaviateStore
    return WeaviateStore(*args, **kwargs)
