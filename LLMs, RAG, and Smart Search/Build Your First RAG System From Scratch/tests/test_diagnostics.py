from __future__ import annotations

import pytest

from rag_system.diagnostics import embedding_integrity_report


class _FakeEmbeddingEngine:
    def embed(self, text: str) -> list[float]:
        base = float(len(text))
        return [base, base + 1.0, base + 2.0]

    def embed_batch(self, texts, batch_size=32):
        return [self.embed(text) for text in texts]


def test_embedding_integrity_report_exposes_batch_cosine() -> None:
    engine = _FakeEmbeddingEngine()
    report = embedding_integrity_report(engine, texts=["alpha", "beta", "gamma"], batch_size=2)

    assert report["num_texts"] == 3
    assert report["dimension"] == 3
    assert report["nan_vectors"] == 0
    assert report["inf_vectors"] == 0
    assert report["batch_consistent"] is True
    assert report["batch_min_cosine_similarity"] == pytest.approx(1.0, abs=1e-12)
