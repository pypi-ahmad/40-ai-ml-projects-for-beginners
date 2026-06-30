"""Embedding adapters and comparison utilities."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from ollama import Client
from sentence_transformers import SentenceTransformer


class EmbeddingProvider(Protocol):
    """Embedding interface used by retrieval and indexing layers."""

    @property
    def model_name(self) -> str:
        """Provider model name."""

    def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        """Embed text inputs."""

    def embedding_dimension(self) -> int:
        """Get embedding vector dimension."""


class SentenceTransformerEmbedder:
    """Hugging Face/SentenceTransformer embedding adapter."""

    def __init__(self, model_name: str, normalize: bool = True) -> None:
        self._model_name = model_name
        self.normalize = normalize
        self.model = SentenceTransformer(model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()

    def embedding_dimension(self) -> int:
        if hasattr(self.model, "get_embedding_dimension"):
            return int(self.model.get_embedding_dimension())
        return int(self.model.get_sentence_embedding_dimension())


class OllamaEmbedder:
    """Ollama embedding adapter for local embedding models."""

    def __init__(self, model_name: str, host: str, normalize: bool = True) -> None:
        self._model_name = model_name
        self.host = host
        self.normalize = normalize
        self.client = Client(host=host)
        self._cached_dimension: int | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        all_embeddings: list[list[float]] = []
        for offset in range(0, len(texts), batch_size):
            batch = list(texts[offset : offset + batch_size])
            response = self.client.embed(model=self._model_name, input=batch)
            embeddings = response["embeddings"]
            if self.normalize:
                embeddings = self._normalize(embeddings)
            all_embeddings.extend(embeddings)
        return all_embeddings

    def embedding_dimension(self) -> int:
        if self._cached_dimension is not None:
            return self._cached_dimension
        sample = self.embed_texts(["dimension probe"])
        self._cached_dimension = len(sample[0]) if sample else 0
        return self._cached_dimension

    @staticmethod
    def _normalize(embeddings: Sequence[Sequence[float]]) -> list[list[float]]:
        arr = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (arr / norms).tolist()


class HashingEmbedder:
    """Offline-safe deterministic embedder used as last-resort fallback."""

    def __init__(self, model_name: str = "hashing-embedder", dimension: int = 384) -> None:
        self._model_name = model_name
        self.dimension = dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            for token in text.lower().split():
                slot = hash(token) % self.dimension
                vector[slot] += 1.0
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector /= norm
            vectors.append(vector.tolist())
        return vectors

    def embedding_dimension(self) -> int:
        return self.dimension


@dataclass(slots=True)
class EmbeddingBenchmark:
    """Embedding benchmark output."""

    model_name: str
    latency_ms: float
    dimension: int
    memory_bytes: int


def build_embedding_provider(model_name: str, ollama_host: str, normalize: bool = True) -> EmbeddingProvider:
    """Factory for supported embedding providers."""

    if model_name == "nomic-embed-text" or ":" in model_name:
        try:
            return OllamaEmbedder(model_name=model_name, host=ollama_host, normalize=normalize)
        except Exception:  # noqa: BLE001
            return HashingEmbedder(model_name=f"hash::{model_name}")

    try:
        return SentenceTransformerEmbedder(model_name=model_name, normalize=normalize)
    except Exception:  # noqa: BLE001
        return HashingEmbedder(model_name=f"hash::{model_name}")


def benchmark_embedding_model(
    provider: EmbeddingProvider,
    texts: Sequence[str],
    batch_size: int = 32,
) -> EmbeddingBenchmark:
    """Benchmark one embedding provider on sample texts."""

    started = time.perf_counter()
    vectors = provider.embed_texts(texts, batch_size=batch_size)
    latency_ms = (time.perf_counter() - started) * 1000
    dimension = provider.embedding_dimension()
    memory_bytes = int(len(vectors) * max(1, dimension) * 4)
    return EmbeddingBenchmark(
        model_name=provider.model_name,
        latency_ms=latency_ms,
        dimension=dimension,
        memory_bytes=memory_bytes,
    )
