"""Relevance filtering for intent-aware curation.

This module provides the core relevance scoring and filtering
functionality that makes Gweta intent-aware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from gweta.core.logging import get_logger
from gweta.core.types import Chunk
from gweta.intelligence.embeddings import EmbeddingEngine
from gweta.intelligence.intent import SystemIntent

logger = get_logger(__name__)


class RelevanceDecision(Enum):
    """Decision for a chunk based on relevance score."""

    ACCEPT = "accept"  # Score >= min_relevance_score
    REVIEW = "review"  # Score between review_threshold and min_relevance_score
    REJECT = "reject"  # Score < review_threshold


@dataclass
class RelevanceResult:
    """Result of relevance scoring for a single chunk."""

    chunk: Chunk
    relevance_score: float
    decision: RelevanceDecision
    matched_topics: list[str] = field(default_factory=list)
    rejection_reason: str | None = None

    @property
    def accepted(self) -> bool:
        """Whether the chunk was accepted."""
        return self.decision == RelevanceDecision.ACCEPT

    @property
    def needs_review(self) -> bool:
        """Whether the chunk needs manual review."""
        return self.decision == RelevanceDecision.REVIEW

    @property
    def rejected(self) -> bool:
        """Whether the chunk was rejected."""
        return self.decision == RelevanceDecision.REJECT


@dataclass
class RelevanceReport:
    """Report from filtering a batch of chunks."""

    results: list[RelevanceResult]
    intent_name: str
    total_chunks: int = 0
    accepted_count: int = 0
    review_count: int = 0
    rejected_count: int = 0
    avg_relevance_score: float = 0.0

    def __post_init__(self) -> None:
        """Compute statistics from results."""
        self.total_chunks = len(self.results)

        if self.total_chunks == 0:
            return

        self.accepted_count = sum(1 for r in self.results if r.accepted)
        self.review_count = sum(1 for r in self.results if r.needs_review)
        self.rejected_count = sum(1 for r in self.results if r.rejected)

        scores = [r.relevance_score for r in self.results]
        self.avg_relevance_score = sum(scores) / len(scores)

    @property
    def acceptance_rate(self) -> float:
        """Percentage of chunks accepted."""
        if self.total_chunks == 0:
            return 0.0
        return self.accepted_count / self.total_chunks

    @property
    def rejection_rate(self) -> float:
        """Percentage of chunks rejected."""
        if self.total_chunks == 0:
            return 0.0
        return self.rejected_count / self.total_chunks

    def accepted(self) -> list[Chunk]:
        """Get all accepted chunks with enriched metadata."""
        chunks = []
        for result in self.results:
            if result.accepted:
                # Enrich chunk with relevance metadata
                chunk = result.chunk
                chunk.metadata["relevance_score"] = result.relevance_score
                chunk.metadata["matched_topics"] = result.matched_topics
                chunk.metadata["intent"] = self.intent_name
                chunks.append(chunk)
        return chunks

    def for_review(self) -> list[Chunk]:
        """Get chunks that need manual review."""
        return [r.chunk for r in self.results if r.needs_review]

    def rejected(self) -> list[RelevanceResult]:
        """Get rejected results with reasons."""
        return [r for r in self.results if r.rejected]

    def __repr__(self) -> str:
        return (
            f"RelevanceReport("
            f"accepted={self.accepted_count}, "
            f"review={self.review_count}, "
            f"rejected={self.rejected_count}, "
            f"avg_score={self.avg_relevance_score:.2f})"
        )


class RelevanceFilter:
    """Filters chunks based on relevance to system intent.

    This is the core component that makes Gweta intent-aware.
    It uses embedding similarity to score how relevant each chunk
    is to the system's defined purpose.

    Example:
        >>> intent = SystemIntent.from_yaml("simuka_intent.yaml")
        >>> filter = RelevanceFilter(intent)
        >>> report = filter.filter_batch(chunks)
        >>> relevant_chunks = report.accepted()
    """

    def __init__(
        self,
        intent: SystemIntent,
        embedding_engine: EmbeddingEngine | None = None,
        model_name: str | None = None,
    ) -> None:
        """Initialize relevance filter.

        Args:
            intent: System intent defining what's relevant
            embedding_engine: Pre-configured embedding engine.
                If None, creates one with model_name or default.
            model_name: Model name if creating new engine.
        """
        self.intent = intent

        # Initialize or use provided embedding engine
        if embedding_engine is not None:
            self.engine = embedding_engine
        else:
            self.engine = EmbeddingEngine(model_name=model_name)

        # Pre-compute intent embedding
        self._intent_embedding: np.ndarray | None = None

    @property
    def intent_embedding(self) -> np.ndarray:
        """Get or compute the intent embedding."""
        if self._intent_embedding is None:
            logger.info("Computing intent embedding...")
            self._intent_embedding = self.engine.embed(self.intent.intent_text)
            logger.info("Intent embedding computed.")
        return self._intent_embedding

    def score(self, chunk: Chunk) -> float:
        """Score a single chunk for relevance.

        Args:
            chunk: Chunk to score

        Returns:
            Relevance score between 0 and 1
        """
        # Quick rejection for explicitly irrelevant topics
        if self.intent.is_irrelevant_topic(chunk.text):
            return 0.0

        # Compute embedding similarity
        chunk_embedding = self.engine.embed(chunk.text)
        similarity = float(
            self.engine._cosine_similarity(chunk_embedding, self.intent_embedding)
        )

        # Normalize to 0-1 range (cosine similarity can be negative)
        score = (similarity + 1) / 2

        return score

    def filter(self, chunk: Chunk) -> RelevanceResult:
        """Filter a single chunk.

        Args:
            chunk: Chunk to evaluate

        Returns:
            RelevanceResult with score and decision
        """
        # Quick rejection for explicitly irrelevant topics
        if self.intent.is_irrelevant_topic(chunk.text):
            return RelevanceResult(
                chunk=chunk,
                relevance_score=0.0,
                decision=RelevanceDecision.REJECT,
                rejection_reason="Contains explicitly irrelevant topic",
            )

        # Score the chunk
        score = self.score(chunk)

        # Make decision based on thresholds
        if score >= self.intent.min_relevance_score:
            decision = RelevanceDecision.ACCEPT
        elif score >= self.intent.review_threshold:
            decision = RelevanceDecision.REVIEW
        else:
            decision = RelevanceDecision.REJECT

        # Find matched topics (simple keyword matching)
        matched_topics = self._find_matched_topics(chunk.text)

        # Determine rejection reason if rejected
        rejection_reason = None
        if decision == RelevanceDecision.REJECT:
            rejection_reason = f"Relevance score {score:.2f} below threshold {self.intent.review_threshold}"

        return RelevanceResult(
            chunk=chunk,
            relevance_score=score,
            decision=decision,
            matched_topics=matched_topics,
            rejection_reason=rejection_reason,
        )

    def filter_batch(
        self,
        chunks: list[Chunk],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> RelevanceReport:
        """Filter a batch of chunks.

        Args:
            chunks: Chunks to evaluate
            batch_size: Batch size for embedding computation
            show_progress: Show progress bar

        Returns:
            RelevanceReport with all results and statistics
        """
        if not chunks:
            return RelevanceReport(results=[], intent_name=self.intent.name)

        logger.info(f"Filtering {len(chunks)} chunks for relevance...")

        results = []

        # Pre-filter for explicitly irrelevant topics
        chunks_to_embed = []
        chunk_indices = []

        for i, chunk in enumerate(chunks):
            if self.intent.is_irrelevant_topic(chunk.text):
                results.append(
                    RelevanceResult(
                        chunk=chunk,
                        relevance_score=0.0,
                        decision=RelevanceDecision.REJECT,
                        rejection_reason="Contains explicitly irrelevant topic",
                    )
                )
            else:
                chunks_to_embed.append(chunk)
                chunk_indices.append(i)

        # Batch embed remaining chunks
        if chunks_to_embed:
            texts = [c.text for c in chunks_to_embed]
            embeddings = self.engine.embed_batch(
                texts,
                batch_size=batch_size,
                show_progress=show_progress,
            )

            # Compute similarities
            similarities = self.engine._batch_cosine_similarity(
                embeddings, self.intent_embedding
            )

            # Process results
            for chunk, similarity in zip(chunks_to_embed, similarities):
                # Normalize to 0-1
                score = float((similarity + 1) / 2)

                # Decision
                if score >= self.intent.min_relevance_score:
                    decision = RelevanceDecision.ACCEPT
                elif score >= self.intent.review_threshold:
                    decision = RelevanceDecision.REVIEW
                else:
                    decision = RelevanceDecision.REJECT

                # Matched topics
                matched_topics = self._find_matched_topics(chunk.text)

                # Rejection reason
                rejection_reason = None
                if decision == RelevanceDecision.REJECT:
                    rejection_reason = (
                        f"Relevance score {score:.2f} below threshold "
                        f"{self.intent.review_threshold}"
                    )

                results.append(
                    RelevanceResult(
                        chunk=chunk,
                        relevance_score=score,
                        decision=decision,
                        matched_topics=matched_topics,
                        rejection_reason=rejection_reason,
                    )
                )

        # Sort results to match original order
        # (pre-filtered rejections were added first)
        results_by_chunk = {id(r.chunk): r for r in results}
        ordered_results = [results_by_chunk[id(c)] for c in chunks]

        report = RelevanceReport(
            results=ordered_results,
            intent_name=self.intent.name,
        )

        logger.info(
            f"Relevance filtering complete: "
            f"{report.accepted_count} accepted, "
            f"{report.review_count} review, "
            f"{report.rejected_count} rejected"
        )

        return report

    def _find_matched_topics(self, text: str) -> list[str]:
        """Find which relevant topics appear in the text."""
        text_lower = text.lower()
        matched = []

        for topic in self.intent.relevant_topics:
            # Simple keyword matching
            topic_words = topic.lower().split()
            if all(word in text_lower for word in topic_words):
                matched.append(topic)

        return matched[:5]  # Limit to top 5 matches

    def __repr__(self) -> str:
        return (
            f"RelevanceFilter(intent={self.intent.name!r}, "
            f"model={self.engine.model_name!r})"
        )
