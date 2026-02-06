"""Coherence scoring for chunks.

This module estimates how well a chunk stands alone
and makes sense without additional context.
"""

import re
from dataclasses import dataclass


# Pronouns that typically require antecedents
DANGLING_PRONOUNS = {"it", "this", "that", "these", "those", "they", "them", "he", "she"}


@dataclass
class CoherenceResult:
    """Result of coherence analysis.

    Attributes:
        score: Overall coherence score (0.0-1.0)
        has_complete_sentences: Whether text has complete sentences
        has_dangling_references: Whether text has unresolved references
        starts_properly: Whether text starts with a capital letter
        ends_properly: Whether text ends with punctuation
    """
    score: float
    has_complete_sentences: bool = True
    has_dangling_references: bool = False
    starts_properly: bool = True
    ends_properly: bool = True


def calculate_coherence(text: str) -> CoherenceResult:
    """Score how well a chunk stands alone.

    Checks:
    - Complete sentences (starts capital, ends punctuation)
    - No dangling references ("it", "this" without antecedent)
    - Proper opening and closing

    Args:
        text: Text to analyze

    Returns:
        CoherenceResult with coherence metrics
    """
    if not text or not text.strip():
        return CoherenceResult(score=0.0)

    text = text.strip()
    score = 1.0

    # Check if starts properly
    starts_properly = text[0].isupper() or text[0].isdigit()
    if not starts_properly:
        score *= 0.85

    # Check if ends properly
    ends_properly = text[-1] in '.!?;:"' + "'"
    if not ends_properly:
        score *= 0.9

    # Check for complete sentences
    sentences = re.split(r"[.!?]+", text)
    complete_sentences = sum(
        1 for s in sentences
        if s.strip() and len(s.strip()) > 10
    )
    has_complete_sentences = complete_sentences > 0
    if not has_complete_sentences:
        score *= 0.7

    # Check for dangling references at the start
    first_words = text.lower().split()[:3]
    has_dangling = any(w in DANGLING_PRONOUNS for w in first_words)

    # "It" or "This" at the start often indicates missing context
    if first_words and first_words[0] in {"it", "this", "that"}:
        has_dangling = True
        score *= 0.85

    return CoherenceResult(
        score=max(0.0, min(1.0, score)),
        has_complete_sentences=has_complete_sentences,
        has_dangling_references=has_dangling,
        starts_properly=starts_properly,
        ends_properly=ends_properly,
    )


def is_coherent(text: str, threshold: float = 0.7) -> bool:
    """Quick check if text is coherent.

    Args:
        text: Text to check
        threshold: Minimum coherence score

    Returns:
        True if coherence is above threshold
    """
    result = calculate_coherence(text)
    return result.score >= threshold
