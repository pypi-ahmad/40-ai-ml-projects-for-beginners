from __future__ import annotations

from pathlib import Path

from local_rag.config import AppSettings
from local_rag.index_manager import IndexManifestManager
from local_rag.indexing import IndexingService
from local_rag.types import LoadedDocument


class FakeLoader:
    def __init__(self, docs):
        self.docs = docs

    def load_directory(self, _=None):
        return self.docs


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def timed_embed(self, texts, batch_size=32):
        del batch_size
        self.calls += 1
        vectors = [[0.1, 0.2, 0.3] for _ in texts]
        return vectors, 1.0


class FakeStore:
    def __init__(self) -> None:
        self._count = 0
        self.reset_calls = 0
        self._chunks = []

    def reset(self):
        self.reset_calls += 1
        self._count = 0
        self._chunks = []

    def delete_by_doc_ids(self, _):
        return None

    def upsert_chunks(self, chunks, _):
        self._count += len(chunks)
        self._chunks.extend(chunks)

    def export_chunks(self):
        return list(self._chunks)

    def count(self):
        return self._count


class FakeLexical:
    def __init__(self) -> None:
        self.calls = 0

    def build(self, chunks):
        self.calls += 1

        class _Stats:
            chunk_count = len(chunks)

        return _Stats()


def _docs() -> list[LoadedDocument]:
    return [
        LoadedDocument(
            doc_id="doc1",
            text=" ".join(["alpha"] * 100),
            metadata={
                "doc_id": "doc1",
                "hash": "h1",
                "version_id": "v1",
                "source_path": "a.txt",
            },
        )
    ]


def test_second_index_run_skips_reembedding(tmp_path: Path) -> None:
    settings = AppSettings(
        documents_dir=tmp_path / "docs",
        index_manifest_path=tmp_path / "manifest.json",
        embedding_model="qwen3-embedding:4b",
    )
    settings.ensure_directories()

    loader = FakeLoader(_docs())
    embedder = FakeEmbedder()
    store = FakeStore()
    lexical = FakeLexical()
    manifest = IndexManifestManager(tmp_path / "manifest.json")

    service = IndexingService(settings, loader, embedder, store, manifest, lexical)

    first = service.build_or_update(chunk_size=128, chunk_overlap=32, force_rebuild=False)
    second = service.build_or_update(chunk_size=128, chunk_overlap=32, force_rebuild=False)

    assert first.embedded_chunks > 0
    assert second.embedded_chunks == 0
    assert embedder.calls == 1
    assert lexical.calls == 2
