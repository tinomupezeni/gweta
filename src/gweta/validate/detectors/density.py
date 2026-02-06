"""Information density scoring.

This module calculates how much useful information
is contained in a piece of text vs noise/boilerplate.
"""

import re
from dataclasses import dataclass


# Common English stop words
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "it", "its", "this", "that", "these", "those", "i", "you", "he",
    "she", "we", "they", "what", "which", "who", "when", "where", "why",
    "how", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "also", "now", "here", "there",
}


@dataclass
class DensityResult:
    """Result of information density calculation.

    Attributes:
        score: Overall density score (0.0-1.0)
        unique_word_ratio: Ratio of unique words to total words
        stop_word_ratio: Ratio of stop words
        avg_word_length: Average word length
        sentence_count: Number of sentences
    """
    score: float
    unique_word_ratio: float = 0.0
    stop_word_ratio: float = 0.0
    avg_word_length: float = 0.0
    sentence_count: int = 0


def calculate_density(text: str) -> DensityResult:
    """Calculate information density score.

    Metrics used:
    1. Unique word ratio (vocabulary richness)
    2. Stop word ratio (low = more content words)
    3. Average word length (longer = more specific)
    4. Sentence complexity

    Args:
        text: Text to analyze

    Returns:
        DensityResult with density metrics
    """
    if not text or not text.strip():
        return DensityResult(score=0.0)

    # Extract words
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

    if not words:
        return DensityResult(score=0.0)

    # Calculate metrics
    unique_words = set(words)
    unique_ratio = len(unique_words) / len(words)

    stop_words = [w for w in words if w in STOP_WORDS]
    stop_ratio = len(stop_words) / len(words)

    avg_length = sum(len(w) for w in words) / len(words)

    # Count sentences
    sentences = re.split(r"[.!?]+", text)
    sentence_count = len([s for s in sentences if s.strip()])

    # Calculate overall score
    # Higher unique ratio = better (vocabulary richness)
    # Lower stop word ratio = better (more content words)
    # Higher average word length = better (more specific terms)

    score = 0.0

    # Unique word ratio (weight: 0.3)
    # Normalize: 0.3-0.7 is typical range
    score += min(1.0, unique_ratio / 0.5) * 0.3

    # Stop word ratio (weight: 0.3)
    # Lower is better, typical range 0.3-0.6
    score += max(0.0, 1 - stop_ratio) * 0.3

    # Average word length (weight: 0.2)
    # Normalize: 4-6 is typical, higher = more technical
    score += min(1.0, avg_length / 6) * 0.2

    # Sentence presence (weight: 0.2)
    # At least some sentences indicate structure
    if sentence_count > 0:
        score += 0.2

    return DensityResult(
        score=min(1.0, score),
        unique_word_ratio=unique_ratio,
        stop_word_ratio=stop_ratio,
        avg_word_length=avg_length,
        sentence_count=sentence_count,
    )


def is_low_density(text: str, threshold: float = 0.3) -> bool:
    """Quick check if text has low information density.

    Args:
        text: Text to check
        threshold: Minimum density score

    Returns:
        True if density is below threshold
    """
    result = calculate_density(text)
    return result.score < threshold
