"""Validation layer for Gweta.

This module provides multi-layer validation:
- Layer 1: Extraction quality scoring
- Layer 2: Chunk quality validation
- Layer 3: Domain rule engine
- Layer 4: Knowledge base health monitoring
"""

from gweta.validate.extraction import ExtractionValidator, ExtractionResult
from gweta.validate.chunks import ChunkValidator
from gweta.validate.rules import DomainRuleEngine, Rule, KnownFact
from gweta.validate.health import HealthChecker, HealthReport
from gweta.validate.golden import GoldenDatasetRunner, GoldenPair, GoldenTestReport

__all__ = [
    # Extraction
    "ExtractionValidator",
    "ExtractionResult",
    # Chunks
    "ChunkValidator",
    # Rules
    "DomainRuleEngine",
    "Rule",
    "KnownFact",
    # Health
    "HealthChecker",
    "HealthReport",
    # Golden
    "GoldenDatasetRunner",
    "GoldenPair",
    "GoldenTestReport",
]
