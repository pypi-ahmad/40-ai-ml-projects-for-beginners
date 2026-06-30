"""Ollama embedding client utilities."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
from loguru import logger
from ollama import Client


class OllamaEmbeddingClient:
    """Wrapper around Ollama embeddings endpoint."""

    def __init__(self, model: str, host: str, normalize: bool = True) -> None:
        self.model = model
        self.host = host
        self.normalize = normalize
        self.client = Client(host=host)

    def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        """Embed texts and optionally normalize vectors."""

        all_embeddings: list[list[float]] = []

        for offset in range(0, len(texts), batch_size):
            batch = list(texts[offset : offset + batch_size])
            response = self.client.embed(model=self.model, input=batch)
            embeddings = response["embeddings"]
            if self.normalize:
                embeddings = self._normalize(embeddings)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def embedding_dimension(self) -> int:
        """Infer embedding dimensionality from probe text."""

        embedding = self.embed_texts(["dimension probe"])[0]
        return len(embedding)

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

    @staticmethod
    def _normalize(embeddings: Sequence[Sequence[float]]) -> list[list[float]]:
        arr = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (arr / norms).tolist()
