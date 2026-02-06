"""Embedding engine for semantic similarity.

This module wraps sentence-transformers to provide embedding
functionality for the intelligence layer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gweta.core.logging import get_logger

logger = get_logger(__name__)

# Default model - good balance of speed and accuracy
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Alternative models
MODELS = {
    "fast": "all-MiniLM-L6-v2",  # 80MB, fastest
    "balanced": "all-MiniLM-L12-v2",  # 120MB, better accuracy
    "accurate": "all-mpnet-base-v2",  # 420MB, best accuracy
}


class EmbeddingEngine:
    """Wrapper for sentence-transformers embedding models.

    Provides a simple interface for generating embeddings and
    computing semantic similarity.

    Example:
        >>> engine = EmbeddingEngine()
        >>> embedding = engine.embed("Some text")
        >>> similarity = engine.similarity("Text A", "Text B")
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        """Initialize embedding engine.

        Args:
            model_name: Name of sentence-transformers model.
                Can be a model name from MODELS dict ("fast", "balanced", "accurate")
                or a full model name from HuggingFace.
            device: Device to use ("cpu", "cuda", "mps"). Auto-detected if None.
            cache_dir: Directory to cache downloaded models.
        """
        # Resolve model name
        if model_name is None:
            model_name = DEFAULT_MODEL
        elif model_name in MODELS:
            model_name = MODELS[model_name]

        self._model_name = model_name
        self._device = device
        self._cache_dir = cache_dir
        self._model: Any = None
        self._dimension: int | None = None

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        self._ensure_model()
        return self._dimension  # type: ignore

    def _ensure_model(self) -> None:
        """Lazy load the model on first use."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for the intelligence layer. "
                "Install it with: pip install sentence-transformers"
            ) from e

        logger.info(f"Loading embedding model: {self._model_name}")

        self._model = SentenceTransformer(
            self._model_name,
            device=self._device,
            cache_folder=self._cache_dir,
        )
        self._dimension = self._model.get_sentence_embedding_dimension()

        logger.info(f"Model loaded. Dimension: {self._dimension}")

    def embed(self, text: str | list[str]) -> np.ndarray:
        """Generate embedding(s) for text.

        Args:
            text: Single text or list of texts

        Returns:
            Embedding vector(s) as numpy array.
            Shape: (dimension,) for single text, (n, dimension) for list.
        """
        self._ensure_model()

        # Handle single text
        if isinstance(text, str):
            return self._model.encode(text, convert_to_numpy=True)

        # Handle list
        return self._model.encode(text, convert_to_numpy=True)

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Embeddings as numpy array of shape (n, dimension)
        """
        self._ensure_model()

        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts.

        Args:
            text_a: First text
            text_b: Second text

        Returns:
            Cosine similarity score between 0 and 1
        """
        emb_a = self.embed(text_a)
        emb_b = self.embed(text_b)
        return float(self._cosine_similarity(emb_a, emb_b))

    def similarity_to_reference(
        self,
        texts: list[str],
        reference: str | np.ndarray,
    ) -> np.ndarray:
        """Compute similarity of multiple texts to a reference.

        Args:
            texts: List of texts to compare
            reference: Reference text or pre-computed embedding

        Returns:
            Array of similarity scores
        """
        # Get reference embedding
        if isinstance(reference, str):
            ref_emb = self.embed(reference)
        else:
            ref_emb = reference

        # Get text embeddings
        text_embs = self.embed_batch(texts)

        # Compute similarities
        similarities = np.array([
            self._cosine_similarity(emb, ref_emb)
            for emb in text_embs
        ])

        return similarities

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    @staticmethod
    def _batch_cosine_similarity(
        embeddings: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity of batch to reference vector.

        Args:
            embeddings: Matrix of shape (n, dimension)
            reference: Vector of shape (dimension,)

        Returns:
            Array of similarities of shape (n,)
        """
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normalized = embeddings / norms

        # Normalize reference
        ref_norm = np.linalg.norm(reference)
        if ref_norm == 0:
            return np.zeros(len(embeddings))
        normalized_ref = reference / ref_norm

        # Dot product gives cosine similarity for normalized vectors
        return np.dot(normalized, normalized_ref)

    def __repr__(self) -> str:
        loaded = "loaded" if self._model is not None else "not loaded"
        return f"EmbeddingEngine(model={self._model_name!r}, {loaded})"
