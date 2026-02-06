"""Ingestion layer for Gweta.

This module provides chunking strategies and vector store
adapters for loading validated data.
"""

from gweta.ingest.chunkers import RecursiveChunker, get_chunker
from gweta.ingest.pipeline import IngestionPipeline
from gweta.ingest.stores.base import BaseStore, StoreStats, AddResult

__all__ = [
    "RecursiveChunker",
    "get_chunker",
    "IngestionPipeline",
    "BaseStore",
    "StoreStats",
    "AddResult",
]
