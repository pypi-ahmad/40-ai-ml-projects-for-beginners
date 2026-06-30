"""Embedding backends and helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import requests

from semantic_search.config import AppConfig, EmbeddingModelConfig
from semantic_search.logging_utils import get_logger

logger = get_logger()


class EmbeddingBackend(ABC):
    """Common interface for embedding providers."""

    model_name: str

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return embeddings for texts."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""


class SentenceTransformerBackend(EmbeddingBackend):
    """SentenceTransformers backend."""

    def __init__(self, model_name: str, normalize: bool = True, local_files_only: bool = False):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.normalize = normalize
        if local_files_only:
            self._model = SentenceTransformer(model_name, local_files_only=True, device="cpu")
        else:
            try:
                self._model = SentenceTransformer(model_name, local_files_only=True, device="cpu")
            except Exception:  # noqa: BLE001
                self._model = SentenceTransformer(model_name, device="cpu")
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension


class OllamaEmbeddingBackend(EmbeddingBackend):
    """Ollama `/api/embed` backend."""

    def __init__(self, model_name: str, base_url: str = "http://127.0.0.1:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._dimension = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        payload = {
            "model": self.model_name,
            "input": texts,
        }
        response = requests.post(f"{self.base_url}/api/embed", json=payload, timeout=90)
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings")
        if not embeddings:
            raise RuntimeError("Ollama embed response missing `embeddings`")
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        self._dimension = int(matrix.shape[1])
        return matrix

    @property
    def dimension(self) -> int:
        return self._dimension


def build_embedding_backend(model_cfg: EmbeddingModelConfig, config: AppConfig) -> EmbeddingBackend:
    """Factory for embedding backend."""
    provider = model_cfg.provider.lower()
    if provider == "sentence_transformers":
        return SentenceTransformerBackend(model_name=model_cfg.model_name, normalize=model_cfg.normalize)
    if provider == "ollama":
        base_url = "http://127.0.0.1:11434"
        return OllamaEmbeddingBackend(model_name=model_cfg.model_name, base_url=base_url)
    raise ValueError(f"Unsupported embedding provider: {model_cfg.provider}")


def embed_text_batches(
    texts: list[str],
    backend: EmbeddingBackend,
    batch_size: int,
) -> np.ndarray:
    """Embed texts in batches for memory safety."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    output: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        output.append(backend.encode(batch))
    vectors = np.vstack(output)
    logger.info(
        "embedding_complete",
        model=backend.model_name,
        text_count=len(texts),
        dimension=int(vectors.shape[1]),
    )
    return vectors.astype(np.float32)
