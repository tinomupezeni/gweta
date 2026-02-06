"""Layer 2: Chunk quality validation.

This module validates chunks BEFORE they enter any vector store,
checking for coherence, information density, and metadata completeness.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from gweta.core.config import get_settings
from gweta.core.logging import get_logger
from gweta.core.types import Chunk, ChunkResult, QualityDetails, QualityIssue, QualityReport
from gweta.validate.detectors.density import calculate_density
from gweta.validate.detectors.duplicates import DuplicateDetector
from gweta.validate.detectors.gibberish import detect_gibberish

logger = get_logger(__name__)


@dataclass
class ChunkValidatorConfig:
    """Configuration for ChunkValidator.

    Attributes:
        min_quality_score: Minimum quality score to pass validation
        min_length: Minimum text length
        max_length: Maximum text length
        min_density_score: Minimum information density score
        max_duplicate_similarity: Max similarity before flagging as duplicate
        required_metadata: List of required metadata fields
    """
    min_quality_score: float = 0.6
    min_length: int = 50
    max_length: int = 10000
    min_density_score: float = 0.3
    max_duplicate_similarity: float = 0.9
    required_metadata: list[str] = field(default_factory=list)


class ChunkValidator:
    """Layer 2: Validates chunks before vector store loading.

    Performs validation checks for:
    - Coherence scoring (does chunk make sense standalone?)
    - Information density (signal vs noise ratio)
    - Metadata completeness (required fields present?)
    - Duplicate/near-duplicate detection
    - Boundary quality (proper sentence boundaries?)
    - Minimum/maximum length

    Example:
        >>> validator = ChunkValidator(required_metadata=["source", "date"])
        >>> result = validator.validate(chunk)
        >>> if result.passed:
        ...     # Load chunk
        ...     pass
    """

    def __init__(
        self,
        config: ChunkValidatorConfig | None = None,
        required_metadata: list[str] | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
    ) -> None:
        """Initialize ChunkValidator.

        Args:
            config: Validator configuration
            required_metadata: List of required metadata field names
            min_length: Minimum chunk text length
            max_length: Maximum chunk text length
        """
        self.config = config or ChunkValidatorConfig()

        # Override config with explicit params if provided
        if required_metadata is not None:
            self.config.required_metadata = required_metadata
        if min_length is not None:
            self.config.min_length = min_length
        if max_length is not None:
            self.config.max_length = max_length

        # Initialize duplicate detector
        self._duplicate_detector = DuplicateDetector(
            threshold=self.config.max_duplicate_similarity
        )

    def validate(self, chunk: Chunk) -> ChunkResult:
        """Validate a single chunk.

        Args:
            chunk: Chunk to validate

        Returns:
            ChunkResult with validation details
        """
        issues: list[QualityIssue] = []
        details = QualityDetails()

        # Check text length
        self._check_length(chunk, issues, details)

        # Check for gibberish
        self._check_gibberish(chunk, issues, details)

        # Calculate information density
        self._check_density(chunk, issues, details)

        # Check metadata completeness
        self._check_metadata(chunk, issues)

        # Check for duplicates
        self._check_duplicates(chunk, issues, details)

        # Check boundary quality
        self._check_boundaries(chunk, issues, details)

        # Calculate overall score
        details.issues = issues
        quality_score = details.aggregate_score

        # Update chunk with quality info
        chunk.quality_score = quality_score
        chunk.quality_details = details

        # Determine pass/fail
        passed = (
            quality_score >= self.config.min_quality_score
            and not details.has_errors()
        )

        return ChunkResult(
            chunk=chunk,
            passed=passed,
            quality_score=quality_score,
            issues=issues,
        )

    def validate_batch(
        self,
        chunks: list[Chunk],
        parallel: bool = True,
        max_workers: int = 4,
    ) -> QualityReport:
        """Validate a batch of chunks.

        Args:
            chunks: List of chunks to validate
            parallel: Whether to validate in parallel
            max_workers: Maximum parallel workers

        Returns:
            QualityReport with aggregate results
        """
        if not chunks:
            return QualityReport(
                total_chunks=0,
                passed=0,
                failed=0,
                warnings=0,
                avg_quality_score=0.0,
            )

        # Reset duplicate detector for batch
        self._duplicate_detector = DuplicateDetector(
            threshold=self.config.max_duplicate_similarity
        )

        results: list[ChunkResult] = []

        if parallel and len(chunks) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(self.validate, chunks))
        else:
            results = [self.validate(chunk) for chunk in chunks]

        # Aggregate results
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        warnings = sum(1 for r in results if r.has_warnings and r.passed)
        avg_score = sum(r.quality_score for r in results) / len(results)

        # Count issues by type
        issues_by_type: dict[str, int] = {}
        for r in results:
            for issue in r.issues:
                issues_by_type[issue.code] = issues_by_type.get(issue.code, 0) + 1

        return QualityReport(
            total_chunks=len(chunks),
            passed=passed,
            failed=failed,
            warnings=warnings,
            avg_quality_score=avg_score,
            issues_by_type=issues_by_type,
            chunks=results,
        )

    async def async_validate_batch(
        self,
        chunks: list[Chunk],
    ) -> QualityReport:
        """Async batch validation.

        Args:
            chunks: List of chunks to validate

        Returns:
            QualityReport
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.validate_batch, chunks, True, 4
        )

    def _check_length(
        self,
        chunk: Chunk,
        issues: list[QualityIssue],
        details: QualityDetails,
    ) -> None:
        """Check chunk text length."""
        text_len = len(chunk.text.strip())

        if text_len == 0:
            issues.append(
                QualityIssue(
                    code="EMPTY_CONTENT",
                    severity="error",
                    message="Chunk text is empty",
                )
            )
            details.extraction_score = 0.0
        elif text_len < self.config.min_length:
            issues.append(
                QualityIssue(
                    code="CONTENT_TOO_SHORT",
                    severity="warning",
                    message=f"Chunk length ({text_len}) below minimum ({self.config.min_length})",
                )
            )
            details.extraction_score *= 0.7
        elif text_len > self.config.max_length:
            issues.append(
                QualityIssue(
                    code="CONTENT_TOO_LONG",
                    severity="warning",
                    message=f"Chunk length ({text_len}) exceeds maximum ({self.config.max_length})",
                )
            )

    def _check_gibberish(
        self,
        chunk: Chunk,
        issues: list[QualityIssue],
        details: QualityDetails,
    ) -> None:
        """Check for garbled/gibberish content."""
        if not chunk.text:
            return

        gibberish_result = detect_gibberish(chunk.text)

        if gibberish_result.is_gibberish:
            issues.append(
                QualityIssue(
                    code="GIBBERISH_DETECTED",
                    severity="error" if gibberish_result.confidence > 0.8 else "warning",
                    message=f"Content appears to be gibberish (confidence: {gibberish_result.confidence:.1%})",
                )
            )
            details.extraction_score *= (1 - gibberish_result.confidence)

    def _check_density(
        self,
        chunk: Chunk,
        issues: list[QualityIssue],
        details: QualityDetails,
    ) -> None:
        """Calculate and check information density."""
        if not chunk.text:
            details.density_score = 0.0
            return

        density_result = calculate_density(chunk.text)
        details.density_score = density_result.score

        if density_result.score < self.config.min_density_score:
            issues.append(
                QualityIssue(
                    code="LOW_DENSITY",
                    severity="warning",
                    message=f"Low information density ({density_result.score:.2f})",
                )
            )

    def _check_metadata(
        self,
        chunk: Chunk,
        issues: list[QualityIssue],
    ) -> None:
        """Check for required metadata fields."""
        missing = [
            field for field in self.config.required_metadata
            if field not in chunk.metadata or chunk.metadata[field] is None
        ]

        if missing:
            issues.append(
                QualityIssue(
                    code="MISSING_METADATA",
                    severity="warning",
                    message=f"Missing required metadata: {', '.join(missing)}",
                )
            )

    def _check_duplicates(
        self,
        chunk: Chunk,
        issues: list[QualityIssue],
        details: QualityDetails,
    ) -> None:
        """Check for near-duplicate content."""
        if not chunk.text:
            return

        # Check against existing chunks in this batch
        duplicates = self._duplicate_detector.find_duplicates(chunk.text)

        if duplicates:
            best_match = max(duplicates, key=lambda d: d.similarity)
            details.duplicate_score = 1 - best_match.similarity

            issues.append(
                QualityIssue(
                    code="NEAR_DUPLICATE",
                    severity="warning",
                    message=f"Similar to chunk {best_match.chunk_id} ({best_match.similarity:.1%})",
                )
            )
        else:
            details.duplicate_score = 1.0

        # Add this chunk to the detector
        self._duplicate_detector.add(chunk.id, chunk.text)

    def _check_boundaries(
        self,
        chunk: Chunk,
        issues: list[QualityIssue],
        details: QualityDetails,
    ) -> None:
        """Check for proper sentence boundaries."""
        text = chunk.text.strip()
        if not text:
            return

        # Check if starts with lowercase (likely mid-sentence)
        if text[0].islower():
            issues.append(
                QualityIssue(
                    code="BAD_START_BOUNDARY",
                    severity="info",
                    message="Chunk may start mid-sentence",
                )
            )
            details.coherence_score *= 0.9

        # Check if ends without punctuation
        if text[-1] not in ".!?:;\"'":
            issues.append(
                QualityIssue(
                    code="BAD_END_BOUNDARY",
                    severity="info",
                    message="Chunk may end mid-sentence",
                )
            )
            details.coherence_score *= 0.9
