"""Embedding backend with SentenceTransformer primary and Ollama fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import httpx
import numpy as np

from resume_ai.config.loader import AppConfig

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional import in constrained env
    SentenceTransformer = None  # type: ignore[assignment]


@dataclass(slots=True)
class EmbeddingService:
    config: AppConfig
    _model: object | None = field(default=None, init=False, repr=False)
    _sentence_model_failed: bool = field(default=False, init=False, repr=False)

    def _load_sentence_model(self) -> None:
        if self._model is not None:
            return
        if self._sentence_model_failed:
            return
        if SentenceTransformer is None:
            return
        try:
            self._model = SentenceTransformer(
                self.config.embeddings.sentence_transformer_model,
                local_files_only=True,
            )
        except Exception:
            self._sentence_model_failed = True
            self._model = None

    def embed_text(self, text: str) -> np.ndarray:
        clean = text.strip()
        if not clean:
            return np.zeros(self.config.embeddings.output_dimension, dtype=float)

        vector: np.ndarray
        if self.config.embeddings.backend == "sentence_transformers":
            self._load_sentence_model()
            if self._model is not None:
                vector = np.asarray(self._model.encode(clean), dtype=float)
                return self._normalize_dimension(vector)

        try:
            vector = self._embed_with_ollama(clean)
        except Exception:
            vector = self._hash_embedding(clean)
        return self._normalize_dimension(vector)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed_text(text) for text in texts]

    def _embed_with_ollama(self, text: str) -> np.ndarray:
        timeout_s = float(min(self.config.retries.timeout_seconds, 5))
        payload = {
            "model": self.config.embeddings.ollama_embedding_model,
            "input": text,
        }
        with httpx.Client(timeout=httpx.Timeout(timeout_s, connect=1.0)) as client:
            response = client.post(
                f"{self.config.models.ollama_base_url}/api/embed",
                json=payload,
            )
            response.raise_for_status()
            vector = response.json().get("embeddings", [[0.0]])[0]
        return np.asarray(vector, dtype=float)

    @staticmethod
    def _hash_embedding(text: str, dim: int = 384) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = np.frombuffer(digest * (dim // len(digest) + 1), dtype=np.uint8)[:dim]
        vector = (values.astype(float) / 255.0) * 2 - 1
        return vector

    def _normalize_dimension(self, vector: np.ndarray) -> np.ndarray:
        target = self.config.embeddings.output_dimension
        if target <= 0:
            return vector.astype(float)
        if vector.shape[0] == target:
            return vector.astype(float)
        if vector.shape[0] > target:
            return vector[:target].astype(float)
        padded = np.zeros(target, dtype=float)
        padded[: vector.shape[0]] = vector
        return padded


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
