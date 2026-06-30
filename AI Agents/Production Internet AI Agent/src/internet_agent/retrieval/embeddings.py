"""Embedding utilities based on Sentence Transformers."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer


class EmbeddingEngine:
    """Sentence transformer wrapper with normalized embeddings."""

    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        arr = self._model.encode(texts, normalize_embeddings=True)
        return arr.tolist()
