"""Near-duplicate detection using MinHash LSH.

This module provides efficient near-duplicate detection
for chunks using the datasketch library.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class DuplicateMatch:
    """A detected near-duplicate match.

    Attributes:
        chunk_id: ID of the matching chunk
        similarity: Jaccard similarity (0.0-1.0)
    """
    chunk_id: str
    similarity: float


class DuplicateDetector:
    """Near-duplicate detection using MinHash LSH.

    Uses datasketch library for efficient approximate
    nearest neighbor detection based on Jaccard similarity.

    Example:
        >>> detector = DuplicateDetector(threshold=0.9)
        >>> detector.add("chunk1", "This is some text content")
        >>> detector.add("chunk2", "This is some text content!")
        >>> matches = detector.find_duplicates("This is some text content")
        >>> print(matches[0].similarity)  # ~0.95
    """

    def __init__(
        self,
        threshold: float = 0.92,
        num_perm: int = 128,
    ) -> None:
        """Initialize DuplicateDetector.

        Args:
            threshold: Similarity threshold for duplicate detection
            num_perm: Number of permutations for MinHash (higher = more accurate)
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self._lsh: Any = None
        self._minhashes: dict[str, Any] = {}
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of LSH index."""
        if self._initialized:
            return

        try:
            from datasketch import MinHashLSH
            self._lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
            self._initialized = True
        except ImportError:
            # Fallback to simple string matching if datasketch not available
            self._lsh = None
            self._initialized = True

    def _create_minhash(self, text: str) -> Any:
        """Create MinHash for text."""
        try:
            from datasketch import MinHash

            mh = MinHash(num_perm=self.num_perm)
            # Create shingles (3-grams)
            words = text.lower().split()
            for i in range(len(words) - 2):
                shingle = " ".join(words[i : i + 3])
                mh.update(shingle.encode("utf-8"))
            return mh
        except ImportError:
            return None

    def add(self, chunk_id: str, text: str) -> None:
        """Add a chunk to the index.

        Args:
            chunk_id: Unique identifier for the chunk
            text: Text content of the chunk
        """
        self._ensure_initialized()

        if self._lsh is None:
            # Fallback: store raw text
            self._minhashes[chunk_id] = text
            return

        mh = self._create_minhash(text)
        if mh is not None:
            self._minhashes[chunk_id] = mh
            self._lsh.insert(chunk_id, mh)

    def find_duplicates(self, text: str) -> list[DuplicateMatch]:
        """Find near-duplicates of the given text.

        Args:
            text: Text to find duplicates for

        Returns:
            List of DuplicateMatch objects
        """
        self._ensure_initialized()
        matches: list[DuplicateMatch] = []

        if self._lsh is None:
            # Fallback: simple string similarity
            return self._find_duplicates_simple(text)

        mh = self._create_minhash(text)
        if mh is None:
            return []

        # Query LSH for candidates
        candidates = self._lsh.query(mh)

        for candidate_id in candidates:
            if candidate_id in self._minhashes:
                candidate_mh = self._minhashes[candidate_id]
                similarity = mh.jaccard(candidate_mh)
                if similarity >= self.threshold:
                    matches.append(
                        DuplicateMatch(chunk_id=candidate_id, similarity=similarity)
                    )

        return sorted(matches, key=lambda m: m.similarity, reverse=True)

    def _find_duplicates_simple(self, text: str) -> list[DuplicateMatch]:
        """Fallback duplicate detection using simple string similarity."""
        matches: list[DuplicateMatch] = []

        text_lower = text.lower()
        text_words = set(text_lower.split())

        for chunk_id, stored_text in self._minhashes.items():
            stored_lower = stored_text.lower()
            stored_words = set(stored_lower.split())

            # Jaccard similarity
            if text_words or stored_words:
                intersection = len(text_words & stored_words)
                union = len(text_words | stored_words)
                similarity = intersection / union if union > 0 else 0

                if similarity >= self.threshold:
                    matches.append(
                        DuplicateMatch(chunk_id=chunk_id, similarity=similarity)
                    )

        return sorted(matches, key=lambda m: m.similarity, reverse=True)

    def get_duplicate_groups(self) -> list[list[str]]:
        """Get all groups of duplicate chunks.

        Returns:
            List of groups, where each group is a list of chunk IDs
            that are duplicates of each other.
        """
        self._ensure_initialized()
        groups: list[list[str]] = []
        processed: set[str] = set()

        for chunk_id, mh in self._minhashes.items():
            if chunk_id in processed:
                continue

            # Find all duplicates of this chunk
            if self._lsh is not None and hasattr(mh, "jaccard"):
                candidates = self._lsh.query(mh)
            else:
                candidates = []

            group = [chunk_id]
            for candidate_id in candidates:
                if candidate_id != chunk_id and candidate_id not in processed:
                    group.append(candidate_id)
                    processed.add(candidate_id)

            if len(group) > 1:
                groups.append(group)
            processed.add(chunk_id)

        return groups

    def clear(self) -> None:
        """Clear the index."""
        self._minhashes.clear()
        self._initialized = False
        self._lsh = None
