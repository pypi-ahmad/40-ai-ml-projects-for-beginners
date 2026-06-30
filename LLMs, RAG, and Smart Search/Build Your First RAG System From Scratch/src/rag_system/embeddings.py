"""Embedding utilities for local Ollama-based RAG."""

from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np
import ollama

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Generate embeddings using local Ollama embedding models."""

    def __init__(
        self,
        model_name: str = "qwen3-embedding:4b",
        host: str = "http://127.0.0.1:11434",
        request_timeout_s: float = 300.0,
        max_retries: int = 3,
    ) -> None:
        self.model_name = model_name
        self.host = host
        self.client = ollama.Client(host=host, timeout=request_timeout_s)
        self.max_retries = max_retries
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        """Resolve embedding vector dimension lazily."""
        if self._dimension is None:
            self._dimension = len(self.embed("dimension probe"))
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """Embed one text string."""
        if not text.strip():
            raise ValueError("Cannot embed empty text")

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.embed(model=self.model_name, input=text)
                vector = response.embeddings[0]
                if self._dimension is None:
                    self._dimension = len(vector)
                return vector
            except Exception as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(f"Embedding request failed after {attempt} attempts: {exc}") from exc
                sleep_s = min(2.0**attempt, 8.0)
                logger.warning("Embedding attempt %d failed, retrying in %.1fs: %s", attempt, sleep_s, exc)
                time.sleep(sleep_s)

        raise RuntimeError("Unreachable embedding retry loop")

    def embed_batch(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """Embed many texts in batches.

        Batch mode significantly reduces overhead for large corpora.
        """
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            return []

        vectors: list[list[float]] = []
        total = len(valid_texts)
        for start in range(0, total, batch_size):
            batch = valid_texts[start : start + batch_size]
            vectors.extend(self._embed_batch_with_fallback(batch=batch, start_index=start))

            if show_progress:
                logger.info("Embedded %d/%d texts", min(start + batch_size, total), total)

        return vectors

    def _embed_batch_with_fallback(self, batch: Sequence[str], start_index: int) -> list[list[float]]:
        """Embed a batch, recursively splitting when large batch requests time out."""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.embed(model=self.model_name, input=list(batch))
                batch_vectors = response.embeddings
                if len(batch_vectors) != len(batch):
                    raise RuntimeError(
                        f"Embedding count mismatch: expected {len(batch)} vectors, got {len(batch_vectors)}"
                    )
                if self._dimension is None and batch_vectors:
                    self._dimension = len(batch_vectors[0])
                return batch_vectors
            except Exception as exc:
                if attempt == self.max_retries:
                    if len(batch) == 1:
                        raise RuntimeError(
                            f"Batch embedding failed for batch starting at {start_index}: {exc}"
                        ) from exc

                    # Timeout-prone larger batches are recursively split to improve robustness.
                    split = max(1, len(batch) // 2)
                    logger.warning(
                        "Batch embedding failed at index %d for size %d after %d attempts. "
                        "Splitting into %d and %d.",
                        start_index,
                        len(batch),
                        attempt,
                        split,
                        len(batch) - split,
                    )
                    left = self._embed_batch_with_fallback(batch=batch[:split], start_index=start_index)
                    right = self._embed_batch_with_fallback(
                        batch=batch[split:],
                        start_index=start_index + split,
                    )
                    return [*left, *right]

                sleep_s = min(2.0**attempt, 8.0)
                logger.warning(
                    "Batch embedding attempt %d failed at index %d, retrying in %.1fs: %s",
                    attempt,
                    start_index,
                    sleep_s,
                    exc,
                )
                time.sleep(sleep_s)

        raise RuntimeError("Unreachable batch embedding retry loop")

    @staticmethod
    def cosine_similarity(
        vector_a: Sequence[float] | np.ndarray,
        vector_b: Sequence[float] | np.ndarray,
    ) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.asarray(vector_a, dtype=np.float64)
        b = np.asarray(vector_b, dtype=np.float64)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def euclidean_distance(
        vector_a: Sequence[float] | np.ndarray,
        vector_b: Sequence[float] | np.ndarray,
    ) -> float:
        """Compute Euclidean distance between two vectors."""
        a = np.asarray(vector_a, dtype=np.float64)
        b = np.asarray(vector_b, dtype=np.float64)
        return float(np.linalg.norm(a - b))

    def similarity_matrix(self, vectors: list[list[float]]) -> np.ndarray:
        """Compute pairwise cosine-similarity matrix for visualization."""
        matrix = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        safe_matrix = matrix / np.clip(norms, 1e-12, None)
        return safe_matrix @ safe_matrix.T
