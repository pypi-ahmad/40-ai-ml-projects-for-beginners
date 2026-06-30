from __future__ import annotations

from pathlib import Path

from hybrid_research_assistant.loaders import DocumentLoader


def test_loader_reads_markdown_and_text(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "a.md").write_text("# Title\n\nSome markdown content.", encoding="utf-8")
    (docs_dir / "b.txt").write_text("plain text", encoding="utf-8")

    loader = DocumentLoader(base_dir=docs_dir, namespace="test")
    docs = loader.load_directory()

    assert len(docs) == 2
    for doc in docs:
        assert doc.metadata["namespace"] == "test"
        assert doc.metadata["source"]
        assert doc.metadata["document_hash"]
        assert doc.metadata["doc_id"] == doc.doc_id
