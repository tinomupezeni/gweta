"""Core module for Gweta - types, configuration, and utilities."""

from gweta.core.types import (
    Chunk,
    ChunkResult,
    QualityDetails,
    QualityIssue,
    QualityReport,
    Source,
)
from gweta.core.config import GwetaSettings
from gweta.core.exceptions import (
    GwetaError,
    ValidationError,
    ConfigurationError,
    AcquisitionError,
    IngestionError,
)
from gweta.core.registry import SourceAuthorityRegistry

__all__ = [
    # Types
    "Chunk",
    "ChunkResult",
    "QualityDetails",
    "QualityIssue",
    "QualityReport",
    "Source",
    # Config
    "GwetaSettings",
    # Registry
    "SourceAuthorityRegistry",
    # Exceptions
    "GwetaError",
    "ValidationError",
    "ConfigurationError",
    "AcquisitionError",
    "IngestionError",
]
