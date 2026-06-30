from __future__ import annotations

from pathlib import Path

from local_rag.document_manager import DocumentManager


def test_document_manager_upsert_list_delete(tmp_path: Path) -> None:
    base = tmp_path / "docs"
    manager = DocumentManager(base, tmp_path / "catalog.json")

    manager.upsert_document("policies/security.md", b"# Security Policy")
    rows = manager.list_documents()
    assert len(rows) == 1
    assert rows[0].source_path == "policies/security.md"

    assert manager.delete_document("policies/security.md") is True
    assert manager.list_documents() == []


def test_write_catalog(tmp_path: Path) -> None:
    base = tmp_path / "docs"
    manager = DocumentManager(base, tmp_path / "catalog.json")
    manager.upsert_document("a.txt", b"hello")
    path = manager.write_catalog()
    assert path.exists()
