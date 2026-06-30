"""Ollama embedding client utilities."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
from loguru import logger
from ollama import Client


class OllamaEmbeddingClient:
    """Wrapper around Ollama embeddings endpoint."""

    def __init__(
        self,
        model: str,
        host: str,
        normalize: bool = True,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self.model = model
        self.host = host
        self.normalize = normalize
        self.max_retries = max(1, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.client = Client(host=host)
        self._cached_dimension: int | None = None

    def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        """Embed texts and optionally normalize vectors."""

        if not texts:
            return []
        all_embeddings: list[list[float]] = []

        for offset in range(0, len(texts), batch_size):
            batch = list(texts[offset : offset + batch_size])
            response = self._embed_batch_with_retries(batch=batch)
            embeddings = response["embeddings"]
            if self.normalize:
                embeddings = self._normalize(embeddings)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def embedding_dimension(self) -> int:
        """Infer embedding dimensionality from probe text."""

        if self._cached_dimension is not None:
            return self._cached_dimension
        embedding = self.embed_texts(["dimension probe"])[0]
        self._cached_dimension = len(embedding)
        return self._cached_dimension

    def timed_embed(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> tuple[list[list[float]], float]:
        """Run embedding with elapsed milliseconds."""

        started = time.perf_counter()
        vectors = self.embed_texts(texts=texts, batch_size=batch_size)
        elapsed = (time.perf_counter() - started) * 1000
        logger.info("Embedded {} texts in {:.2f} ms", len(texts), elapsed)
        return vectors, elapsed

    def _embed_batch_with_retries(self, batch: list[str]) -> dict[str, list[list[float]]]:
        """Embed one batch with retry for transient local model errors."""

        for attempt in range(1, self.max_retries + 1):
            try:
                return self.client.embed(model=self.model, input=batch)
            except Exception as exc:  # noqa: BLE001
                if attempt >= self.max_retries:
                    raise
                delay_seconds = self.retry_backoff_seconds * attempt
                logger.warning(
                    "Embed batch retry {}/{} after error: {}",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
        raise RuntimeError("Unreachable embed retry state")

    @staticmethod
    def _normalize(embeddings: Sequence[Sequence[float]]) -> list[list[float]]:
        arr = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (arr / norms).tolist()
