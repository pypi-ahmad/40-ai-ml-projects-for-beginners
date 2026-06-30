from pathlib import Path

import numpy as np

from semantic_search.config import load_config
from semantic_search.schemas import DocumentRecord, SearchRequest
from semantic_search.service import SemanticSearchService
from semantic_search.utils import hash_text


class FakeEmbeddingBackend:
    model_name = "fake-backend"

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            token_count = len(text.split())
            vectors.append([float(token_count), float(len(text) % 11), 1.0])
        return np.asarray(vectors, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return 3


def test_service_search_pipeline_with_fake_embeddings(tmp_path: Path, monkeypatch):
    cfg = load_config("config/default.yaml")
    cfg.paths["processed_data_dir"] = str(tmp_path / "processed")
    cfg.paths["cache_dir"] = str(tmp_path / "cache")
    cfg.paths["chroma_dir"] = str(tmp_path / "chroma")
    cfg.paths["faiss_dir"] = str(tmp_path / "faiss")
    cfg.paths["logs_dir"] = str(tmp_path / "logs")
    cfg.reranker.enabled = False

    service = SemanticSearchService(cfg)
    service.documents = [
        DocumentRecord(
            doc_id="doc-1",
            source="test",
            text="machine learning improves retrieval quality",
            title="ML retrieval",
            category="TECH",
            document_hash=hash_text("machine learning improves retrieval quality"),
        ),
        DocumentRecord(
            doc_id="doc-2",
            source="test",
            text="sports analytics and football news",
            title="sports",
            category="SPORTS",
            document_hash=hash_text("sports analytics and football news"),
        ),
    ]
    service.chunk_documents(strategy="recursive", chunk_size=64, chunk_overlap=0)

    monkeypatch.setattr("semantic_search.service.build_embedding_backend", lambda *args, **kwargs: FakeEmbeddingBackend())
    service.build_indexes()

    response = service.search(
        SearchRequest(
            query="machine learning retrieval",
            mode="hybrid",
            top_k=3,
            rerank=False,
            similarity_threshold=-1.0,
        )
    )
    assert response.hits
    assert any(hit.document_id == "doc-1" for hit in response.hits)
