"""Gibberish and OCR failure detection.

This module provides heuristics to detect garbled text,
OCR failures, and encoding corruption.
"""

import re
from dataclasses import dataclass


@dataclass
class GibberishResult:
    """Result of gibberish detection.

    Attributes:
        is_gibberish: Whether text appears to be gibberish
        confidence: Confidence level (0.0-1.0)
        entropy: Character entropy score
        word_ratio: Ratio of dictionary-like words
        special_ratio: Ratio of special characters
    """
    is_gibberish: bool
    confidence: float
    entropy: float = 0.0
    word_ratio: float = 0.0
    special_ratio: float = 0.0


def detect_gibberish(text: str, threshold: float = 0.5) -> GibberishResult:
    """Detect OCR failures and encoding corruption.

    Uses multiple heuristics:
    1. Character entropy (random chars = high entropy)
    2. Consecutive consonant ratio
    3. Dictionary word ratio
    4. Special character density
    5. Repeated character sequences

    Args:
        text: Text to analyze
        threshold: Confidence threshold for gibberish detection

    Returns:
        GibberishResult with detection details
    """
    if not text or len(text) < 10:
        return GibberishResult(is_gibberish=False, confidence=0.0)

    # Calculate metrics
    entropy = _calculate_entropy(text)
    special_ratio = _calculate_special_ratio(text)
    word_ratio = _estimate_word_ratio(text)
    consonant_score = _check_consonant_runs(text)

    # Combine signals
    gibberish_score = 0.0

    # High entropy is suspicious
    if entropy > 4.5:
        gibberish_score += 0.2

    # High special character ratio
    if special_ratio > 0.3:
        gibberish_score += 0.3 * (special_ratio / 0.5)

    # Low word ratio
    if word_ratio < 0.5:
        gibberish_score += 0.3 * (1 - word_ratio)

    # Unusual consonant patterns
    gibberish_score += consonant_score * 0.2

    confidence = min(1.0, gibberish_score)

    return GibberishResult(
        is_gibberish=confidence >= threshold,
        confidence=confidence,
        entropy=entropy,
        word_ratio=word_ratio,
        special_ratio=special_ratio,
    )


def estimate_ocr_confidence(text: str) -> float:
    """Estimate OCR quality from text characteristics.

    Signals for good OCR:
    - Clean word boundaries
    - Proper punctuation
    - Recognizable sentence structure
    - Low special character ratio

    Args:
        text: Text to analyze

    Returns:
        Estimated OCR confidence (0.0-1.0)
    """
    if not text:
        return 0.0

    score = 1.0

    # Check special character ratio
    special_ratio = _calculate_special_ratio(text)
    score -= special_ratio * 0.5

    # Check for proper sentences
    sentences = re.split(r"[.!?]+", text)
    proper_sentences = sum(
        1 for s in sentences
        if s.strip() and s.strip()[0].isupper()
    )
    if sentences:
        sentence_score = proper_sentences / len(sentences)
        score *= (0.5 + 0.5 * sentence_score)

    # Check for dictionary words
    word_ratio = _estimate_word_ratio(text)
    score *= (0.5 + 0.5 * word_ratio)

    return max(0.0, min(1.0, score))


def _calculate_entropy(text: str) -> float:
    """Calculate Shannon entropy of text."""
    import math
    from collections import Counter

    if not text:
        return 0.0

    counter = Counter(text.lower())
    total = len(text)

    entropy = 0.0
    for count in counter.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def _calculate_special_ratio(text: str) -> float:
    """Calculate ratio of special characters."""
    if not text:
        return 0.0

    special = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return special / len(text)


def _estimate_word_ratio(text: str) -> float:
    """Estimate ratio of dictionary-like words.

    Uses simple heuristics since we don't want to load
    a full dictionary. Checks for:
    - Reasonable word length
    - Contains vowels
    - Not all consonants
    """
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    if not words:
        return 0.0

    vowels = set("aeiouAEIOU")
    good_words = 0

    for word in words:
        # Skip very short or very long words
        if len(word) < 2 or len(word) > 20:
            continue

        # Check for vowels
        has_vowel = any(c in vowels for c in word)
        if not has_vowel and len(word) > 3:
            continue

        good_words += 1

    return good_words / len(words) if words else 0.0


def _check_consonant_runs(text: str) -> float:
    """Check for unusual consonant runs (gibberish indicator)."""
    # Find runs of 5+ consonants
    consonant_runs = re.findall(r"[bcdfghjklmnpqrstvwxyz]{5,}", text.lower())

    if not text:
        return 0.0

    # Calculate how much of the text is in consonant runs
    run_chars = sum(len(run) for run in consonant_runs)
    return min(1.0, run_chars / len(text) * 10)
