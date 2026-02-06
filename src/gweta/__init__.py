"""Gweta - RAG data quality and ingestion framework.

Acquire. Validate. Ingest.

Gweta provides a comprehensive pipeline for quality-aware data ingestion
into vector stores for RAG applications. It handles the full lifecycle:
acquisition from various sources, multi-stage validation, and ingestion
with quality tracking.

Example:
    >>> from gweta import Chunk, ChunkValidator, ChromaStore
    >>>
    >>> # Validate chunks
    >>> validator = ChunkValidator()
    >>> report = validator.validate_batch(chunks)
    >>>
    >>> # Ingest validated chunks
    >>> store = ChromaStore(collection_name="my-kb")
    >>> await store.add(report.accepted())
"""

from gweta._version import __version__
from gweta.core.config import GwetaSettings, get_settings
from gweta.core.exceptions import (
    AcquisitionError,
    ConfigurationError,
    GwetaError,
    IngestionError,
    ValidationError,
)
from gweta.core.logging import get_logger, setup_logging
from gweta.core.registry import SourceAuthorityRegistry
from gweta.core.types import (
    Chunk,
    ChunkResult,
    QualityDetails,
    QualityIssue,
    QualityReport,
    Source,
)

__all__ = [
    # Version
    "__version__",
    # Core types
    "Chunk",
    "Source",
    "QualityReport",
    "QualityDetails",
    "QualityIssue",
    "ChunkResult",
    # Config
    "GwetaSettings",
    "get_settings",
    # Exceptions
    "GwetaError",
    "ValidationError",
    "AcquisitionError",
    "IngestionError",
    "ConfigurationError",
    # Logging
    "get_logger",
    "setup_logging",
    # Registry
    "SourceAuthorityRegistry",
    # Validation (lazy)
    "ChunkValidator",
    "ExtractionValidator",
    "DomainRuleEngine",
    "HealthChecker",
    # Acquisition (lazy)
    "GwetaCrawler",
    "PDFExtractor",
    "APIClient",
    "DatabaseSource",
    # Ingestion (lazy)
    "ChromaStore",
    "IngestionPipeline",
    # Adapters (lazy)
    "LangChainAdapter",
    "LlamaIndexAdapter",
    "ChonkieAdapter",
]


def __getattr__(name: str):
    """Lazy import for optional modules."""
    if name == "ChunkValidator":
        from gweta.validate.chunks import ChunkValidator
        return ChunkValidator

    if name == "ExtractionValidator":
        from gweta.validate.extraction import ExtractionValidator
        return ExtractionValidator

    if name == "DomainRuleEngine":
        from gweta.validate.rules import DomainRuleEngine
        return DomainRuleEngine

    if name == "HealthChecker":
        from gweta.validate.health import HealthChecker
        return HealthChecker

    if name == "GwetaCrawler":
        from gweta.acquire.crawler import GwetaCrawler
        return GwetaCrawler

    if name == "PDFExtractor":
        from gweta.acquire.pdf import PDFExtractor
        return PDFExtractor

    if name == "APIClient":
        from gweta.acquire.api import APIClient
        return APIClient

    if name == "DatabaseSource":
        from gweta.acquire.database import DatabaseSource
        return DatabaseSource

    if name == "ChromaStore":
        from gweta.ingest.stores.chroma import ChromaStore
        return ChromaStore

    if name == "IngestionPipeline":
        from gweta.ingest.pipeline import IngestionPipeline
        return IngestionPipeline

    if name == "LangChainAdapter":
        from gweta.adapters.langchain import LangChainAdapter
        return LangChainAdapter

    if name == "LlamaIndexAdapter":
        from gweta.adapters.llamaindex import LlamaIndexAdapter
        return LlamaIndexAdapter

    if name == "ChonkieAdapter":
        from gweta.adapters.chonkie import ChonkieAdapter
        return ChonkieAdapter

    raise AttributeError(f"module 'gweta' has no attribute {name!r}")
