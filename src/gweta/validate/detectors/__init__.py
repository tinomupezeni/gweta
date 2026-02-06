"""Quality detectors for validation.

This module provides heuristic detectors for:
- Gibberish/OCR garbage detection
- Near-duplicate detection
- Information density scoring
- Coherence estimation
- Staleness tracking
"""

from gweta.validate.detectors.gibberish import detect_gibberish, GibberishResult
from gweta.validate.detectors.duplicates import DuplicateDetector, DuplicateMatch
from gweta.validate.detectors.density import calculate_density, DensityResult
from gweta.validate.detectors.coherence import calculate_coherence, CoherenceResult
from gweta.validate.detectors.staleness import StalenessChecker

__all__ = [
    "detect_gibberish",
    "GibberishResult",
    "DuplicateDetector",
    "DuplicateMatch",
    "calculate_density",
    "DensityResult",
    "calculate_coherence",
    "CoherenceResult",
    "StalenessChecker",
]
