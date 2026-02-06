"""Unified pipeline for intent-aware ingestion.

This module provides a high-level API that combines acquisition,
validation, relevance filtering, and ingestion into a single pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gweta.core.logging import get_logger
from gweta.core.types import Chunk
from gweta.ingest.stores.base import BaseStore
from gweta.intelligence.embeddings import EmbeddingEngine
from gweta.intelligence.intent import SystemIntent
from gweta.intelligence.relevance import RelevanceFilter, RelevanceReport
from gweta.validate.chunks import ChunkValidator

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Result of running the ingestion pipeline."""

    # Counts
    total_chunks: int = 0
    quality_passed: int = 0
    relevance_passed: int = 0
    ingested: int = 0

    # Rates
    quality_pass_rate: float = 0.0
    relevance_pass_rate: float = 0.0
    overall_pass_rate: float = 0.0

    # Scores
    avg_quality_score: float = 0.0
    avg_relevance_score: float = 0.0

    # Details
    rejected_chunks: list[Chunk] = field(default_factory=list)
    review_chunks: list[Chunk] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Compute rates."""
        if self.total_chunks > 0:
            self.quality_pass_rate = self.quality_passed / self.total_chunks
            if self.quality_passed > 0:
                self.relevance_pass_rate = self.relevance_passed / self.quality_passed
            self.overall_pass_rate = self.ingested / self.total_chunks


class Pipeline:
    """Unified pipeline for intent-aware ingestion.

    Combines:
    - Quality validation (ChunkValidator)
    - Relevance filtering (RelevanceFilter)
    - Vector store ingestion

    Example:
        >>> pipeline = Pipeline(
        ...     intent="simuka_intent.yaml",
        ...     store=ChromaStore("simuka_kb")
        ... )
        >>> result = await pipeline.ingest(chunks)
        >>> print(f"Ingested {result.ingested} of {result.total_chunks}")
    """

    def __init__(
        self,
        intent: SystemIntent | str | Path,
        store: BaseStore | None = None,
        embedding_model: str | None = None,
        min_quality_score: float = 0.5,
        validate_quality: bool = True,
    ) -> None:
        """Initialize the pipeline.

        Args:
            intent: SystemIntent instance or path to intent YAML file
            store: Vector store for ingestion. If None, pipeline only
                filters without storing.
            embedding_model: Sentence-transformers model name
            min_quality_score: Minimum quality score for chunks
            validate_quality: Whether to run quality validation
        """
        # Load intent if path provided
        if isinstance(intent, (str, Path)):
            self.intent = SystemIntent.from_yaml(intent)
        else:
            self.intent = intent

        self.store = store
        self.validate_quality = validate_quality
        self.min_quality_score = min_quality_score

        # Initialize components
        self.embedding_engine = EmbeddingEngine(model_name=embedding_model)
        self.relevance_filter = RelevanceFilter(
            intent=self.intent,
            embedding_engine=self.embedding_engine,
        )

        if validate_quality:
            self.quality_validator = ChunkValidator(min_length=50)
        else:
            self.quality_validator = None

        logger.info(f"Pipeline initialized for intent: {self.intent.name}")

    async def ingest(
        self,
        chunks: list[Chunk],
        skip_existing: bool = True,
    ) -> PipelineResult:
        """Run the full ingestion pipeline.

        Steps:
        1. Quality validation (optional)
        2. Relevance filtering
        3. Store ingestion (if store configured)

        Args:
            chunks: Chunks to process
            skip_existing: Skip chunks already in store

        Returns:
            PipelineResult with statistics
        """
        result = PipelineResult(total_chunks=len(chunks))

        if not chunks:
            logger.info("No chunks to process")
            return result

        logger.info(f"Processing {len(chunks)} chunks through pipeline...")

        # Step 1: Quality validation
        if self.validate_quality and self.quality_validator:
            logger.info("Step 1: Quality validation...")
            quality_report = self.quality_validator.validate_batch(chunks)

            result.quality_passed = quality_report.passed
            result.avg_quality_score = quality_report.avg_quality_score

            # Filter to quality-passed chunks
            quality_chunks = [
                r.chunk
                for r in quality_report.chunks
                if r.passed and r.quality_score >= self.min_quality_score
            ]

            logger.info(
                f"Quality validation: {len(quality_chunks)}/{len(chunks)} passed"
            )
        else:
            quality_chunks = chunks
            result.quality_passed = len(chunks)

        # Step 2: Relevance filtering
        logger.info("Step 2: Relevance filtering...")
        relevance_report = self.relevance_filter.filter_batch(quality_chunks)

        result.relevance_passed = relevance_report.accepted_count
        result.avg_relevance_score = relevance_report.avg_relevance_score
        result.review_chunks = relevance_report.for_review()

        # Collect rejected with reasons
        for r in relevance_report.rejected():
            result.rejected_chunks.append(r.chunk)

        relevant_chunks = relevance_report.accepted()

        logger.info(
            f"Relevance filtering: {len(relevant_chunks)}/{len(quality_chunks)} accepted"
        )

        # Step 3: Store ingestion
        if self.store and relevant_chunks:
            logger.info("Step 3: Ingesting to store...")
            try:
                add_result = await self.store.add(relevant_chunks)
                result.ingested = add_result.added

                if add_result.errors:
                    result.errors.extend(add_result.errors)

                logger.info(f"Ingested {add_result.added} chunks to store")
            except Exception as e:
                logger.error(f"Store ingestion failed: {e}")
                result.errors.append(str(e))
        else:
            result.ingested = len(relevant_chunks)

        # Compute final rates
        result.__post_init__()

        logger.info(
            f"Pipeline complete: {result.ingested}/{result.total_chunks} ingested "
            f"({result.overall_pass_rate:.1%})"
        )

        return result

    def filter_only(self, chunks: list[Chunk]) -> RelevanceReport:
        """Filter chunks without ingesting.

        Useful for previewing what would be accepted/rejected.

        Args:
            chunks: Chunks to filter

        Returns:
            RelevanceReport with decisions
        """
        # Quality filter first
        if self.validate_quality and self.quality_validator:
            quality_report = self.quality_validator.validate_batch(chunks)
            quality_chunks = [
                r.chunk
                for r in quality_report.chunks
                if r.passed and r.quality_score >= self.min_quality_score
            ]
        else:
            quality_chunks = chunks

        # Relevance filter
        return self.relevance_filter.filter_batch(quality_chunks)

    def score_chunk(self, chunk: Chunk) -> dict[str, Any]:
        """Score a single chunk for quality and relevance.

        Args:
            chunk: Chunk to score

        Returns:
            Dictionary with scores and decisions
        """
        result: dict[str, Any] = {
            "text_preview": chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
            "source": chunk.source,
        }

        # Quality score
        if self.validate_quality and self.quality_validator:
            quality_result = self.quality_validator.validate(chunk)
            result["quality_score"] = quality_result.quality_score
            result["quality_passed"] = quality_result.passed
            result["quality_issues"] = [
                {"code": i.code, "message": i.message}
                for i in quality_result.issues
            ]
        else:
            result["quality_score"] = None
            result["quality_passed"] = True

        # Relevance score
        relevance_result = self.relevance_filter.filter(chunk)
        result["relevance_score"] = relevance_result.relevance_score
        result["relevance_decision"] = relevance_result.decision.value
        result["matched_topics"] = relevance_result.matched_topics

        # Overall decision
        result["would_ingest"] = (
            result["quality_passed"]
            and relevance_result.decision.value == "accept"
        )

        return result

    def __repr__(self) -> str:
        store_info = self.store.collection_name if self.store else "no store"
        return f"Pipeline(intent={self.intent.name!r}, store={store_info})"
