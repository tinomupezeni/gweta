"""Gweta Intelligence Layer.

This module provides intent-aware relevance filtering for RAG systems.
It uses embedding-based semantic similarity to score content against
a system's defined purpose.

Example:
    >>> from gweta.intelligence import SystemIntent, RelevanceFilter
    >>>
    >>> # Load system intent
    >>> intent = SystemIntent.from_yaml("simuka_intent.yaml")
    >>>
    >>> # Create filter
    >>> filter = RelevanceFilter(intent)
    >>>
    >>> # Score chunks for relevance
    >>> results = filter.filter_batch(chunks)
    >>> relevant_chunks = results.accepted()

Or use the unified Pipeline:

    >>> from gweta.intelligence import Pipeline
    >>>
    >>> pipeline = Pipeline(
    ...     intent="simuka_intent.yaml",
    ...     store=ChromaStore("my_kb")
    ... )
    >>> result = await pipeline.ingest(chunks)
"""

from gweta.intelligence.intent import SystemIntent, QualityRequirements, GeographicFocus
from gweta.intelligence.embeddings import EmbeddingEngine
from gweta.intelligence.relevance import (
    RelevanceFilter,
    RelevanceResult,
    RelevanceReport,
    RelevanceDecision,
)
from gweta.intelligence.pipeline import Pipeline, PipelineResult

__all__ = [
    # Intent
    "SystemIntent",
    "QualityRequirements",
    "GeographicFocus",
    # Embeddings
    "EmbeddingEngine",
    # Relevance
    "RelevanceFilter",
    "RelevanceResult",
    "RelevanceReport",
    "RelevanceDecision",
    # Pipeline
    "Pipeline",
    "PipelineResult",
]
