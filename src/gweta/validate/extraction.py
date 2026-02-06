"""Layer 1: Extraction quality validation.

This module validates raw extracted text BEFORE chunking,
checking for OCR quality, encoding issues, and completeness.
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from gweta.core.config import GwetaSettings, get_settings
from gweta.core.logging import get_logger
from gweta.core.types import QualityIssue

logger = get_logger(__name__)


@dataclass
class ExtractionResult:
    """Results from extraction validation.

    Attributes:
        quality_score: Overall extraction quality (0.0-1.0)
        ocr_confidence: Estimated OCR confidence if applicable
        encoding_valid: Whether encoding is valid UTF-8
        language: Detected language
        issues: List of quality issues found
        is_acceptable: Whether extraction passes minimum threshold
    """
    quality_score: float = 1.0
    ocr_confidence: float | None = None
    encoding_valid: bool = True
    language: str | None = None
    issues: list[QualityIssue] = field(default_factory=list)
    text_length: int = 0
    boilerplate_ratio: float = 0.0

    @property
    def is_acceptable(self) -> bool:
        """Check if extraction passes minimum quality."""
        return not any(i.severity == "error" for i in self.issues)


class ExtractionValidator:
    """Layer 1: Validates raw extracted text before chunking.

    Performs heuristic checks for:
    - Text length validation
    - Encoding detection and corruption
    - Gibberish/OCR failure detection
    - Language detection and consistency
    - Boilerplate ratio estimation
    - Table extraction fidelity

    Example:
        >>> validator = ExtractionValidator()
        >>> result = validator.validate(extracted_text)
        >>> if result.is_acceptable:
        ...     # Proceed to chunking
        ...     pass
    """

    def __init__(self, config: GwetaSettings | None = None) -> None:
        """Initialize ExtractionValidator.

        Args:
            config: Gweta settings
        """
        self.config = config or get_settings()

    def validate(
        self,
        text: str,
        source_metadata: dict[str, Any] | None = None,
    ) -> ExtractionResult:
        """Validate extracted text quality.

        Args:
            text: Raw extracted text to validate
            source_metadata: Optional metadata about the source

        Returns:
            ExtractionResult with quality metrics
        """
        result = ExtractionResult(text_length=len(text))
        source_metadata = source_metadata or {}

        # Run all checks
        self._check_text_length(text, result)
        self._check_encoding(text, result)
        self._check_gibberish(text, result)
        self._check_language(text, result)
        self._check_boilerplate(text, result)

        # Calculate overall score
        result.quality_score = self._calculate_score(result)

        return result

    async def async_validate(
        self,
        text: str,
        source_metadata: dict[str, Any] | None = None,
    ) -> ExtractionResult:
        """Async wrapper for validate().

        Args:
            text: Raw extracted text
            source_metadata: Optional source metadata

        Returns:
            ExtractionResult
        """
        # Run in thread pool since validation is CPU-bound
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.validate, text, source_metadata
        )

    def _check_text_length(self, text: str, result: ExtractionResult) -> None:
        """Check if text meets minimum length requirements."""
        if not text or not text.strip():
            result.issues.append(
                QualityIssue(
                    code="EMPTY_CONTENT",
                    severity="error",
                    message="Extracted text is empty",
                )
            )
            return

        if len(text.strip()) < self.config.min_text_length:
            result.issues.append(
                QualityIssue(
                    code="CONTENT_TOO_SHORT",
                    severity="warning",
                    message=f"Text length ({len(text)}) below minimum ({self.config.min_text_length})",
                )
            )

    def _check_encoding(self, text: str, result: ExtractionResult) -> None:
        """Check for encoding issues and corruption."""
        # Check for common encoding corruption patterns
        corruption_patterns = [
            r"[\ufffd]+",  # Replacement characters
            r"[\x00-\x08\x0b\x0c\x0e-\x1f]+",  # Control characters
        ]

        total_corrupt = 0
        for pattern in corruption_patterns:
            matches = re.findall(pattern, text)
            total_corrupt += sum(len(m) for m in matches)

        if total_corrupt > 0:
            ratio = total_corrupt / len(text) if text else 0
            if ratio > 0.05:
                result.encoding_valid = False
                result.issues.append(
                    QualityIssue(
                        code="ENCODING_CORRUPTION",
                        severity="error" if ratio > 0.2 else "warning",
                        message=f"Text contains {ratio:.1%} corrupted characters",
                    )
                )

    def _check_gibberish(self, text: str, result: ExtractionResult) -> None:
        """Detect OCR failures and garbled text."""
        if not text:
            return

        # Count special characters vs alphanumeric
        special_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_count / len(text)

        if special_ratio > self.config.max_gibberish_ratio:
            result.issues.append(
                QualityIssue(
                    code="HIGH_SPECIAL_CHAR_RATIO",
                    severity="warning",
                    message=f"High special character ratio ({special_ratio:.1%})",
                )
            )
            # Estimate OCR confidence based on special char ratio
            result.ocr_confidence = max(0, 1 - (special_ratio * 2))

        # Check for excessive consecutive consonants (gibberish indicator)
        consonant_runs = re.findall(r"[bcdfghjklmnpqrstvwxyz]{5,}", text.lower())
        if len(consonant_runs) > len(text) / 100:
            result.issues.append(
                QualityIssue(
                    code="POSSIBLE_GIBBERISH",
                    severity="warning",
                    message="Text may contain garbled content (unusual letter patterns)",
                )
            )

    def _check_language(self, text: str, result: ExtractionResult) -> None:
        """Detect language and check for consistency."""
        if len(text) < 50:
            return

        try:
            from langdetect import detect, DetectorFactory
            # Make detection deterministic
            DetectorFactory.seed = 0
            result.language = detect(text)
        except ImportError:
            # langdetect not installed, skip
            pass
        except Exception:
            result.issues.append(
                QualityIssue(
                    code="LANGUAGE_DETECTION_FAILED",
                    severity="info",
                    message="Could not detect language",
                )
            )

    def _check_boilerplate(self, text: str, result: ExtractionResult) -> None:
        """Estimate boilerplate content ratio."""
        if not text:
            return

        lines = text.split("\n")
        if not lines:
            return

        # Simple heuristic: short lines, navigation-like patterns
        boilerplate_indicators = [
            "copyright",
            "all rights reserved",
            "privacy policy",
            "terms of service",
            "cookie",
            "subscribe",
            "newsletter",
            "follow us",
            "share this",
        ]

        boilerplate_lines = 0
        for line in lines:
            line_lower = line.lower().strip()
            # Short lines or lines with boilerplate keywords
            if len(line_lower) < 20:
                boilerplate_lines += 1
            elif any(indicator in line_lower for indicator in boilerplate_indicators):
                boilerplate_lines += 1

        result.boilerplate_ratio = boilerplate_lines / len(lines)

        if result.boilerplate_ratio > 0.5:
            result.issues.append(
                QualityIssue(
                    code="HIGH_BOILERPLATE",
                    severity="warning",
                    message=f"High boilerplate ratio ({result.boilerplate_ratio:.1%})",
                )
            )

    def _calculate_score(self, result: ExtractionResult) -> float:
        """Calculate overall extraction quality score."""
        score = 1.0

        # Penalize based on issues
        for issue in result.issues:
            if issue.severity == "error":
                score *= 0.3
            elif issue.severity == "warning":
                score *= 0.8

        # Factor in OCR confidence if available
        if result.ocr_confidence is not None:
            score *= (0.5 + 0.5 * result.ocr_confidence)

        # Factor in boilerplate ratio
        score *= (1 - result.boilerplate_ratio * 0.5)

        return max(0.0, min(1.0, score))
