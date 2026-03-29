"""Gweta Intelligence Layer.

This module provides intent-aware relevance filtering and AI-driven
data acquisition for RAG systems.

Components:
- SystemIntent: Define the purpose and goal of your RAG system.
- RelevanceFilter: Semantic filtering using embedding similarity.
- IntelligenceScout: Goal-driven web discovery, navigation, and extraction.
- Pipeline: Unified ingestion pipeline combining quality and relevance.

Example:
    >>> from gweta.intelligence import IntelligenceScout
    >>>
    >>> scout = IntelligenceScout(model="gpt-4o")
    >>> result = await scout.scout(
    ...     goal="Find business registration fees in Zimbabwe",
    ...     max_pages=3
    ... )
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
from gweta.intelligence.llm import LLMClient
from gweta.intelligence.scout import IntelligenceScout, ScoutResult

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
    # LLM & Scouting
    "LLMClient",
    "IntelligenceScout",
    "ScoutResult",
]
