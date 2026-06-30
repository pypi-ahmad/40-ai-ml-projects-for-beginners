from __future__ import annotations

from pathlib import Path

from local_rag.audit import scan_forbidden_patterns, validate_citations
from local_rag.types import RetrievalResult


def _hit() -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk-1",
        doc_id="doc-1",
        text="example",
        score=0.9,
        metadata={"source_path": "docs/a.txt"},
    )


def test_validate_citations_flags_unknown_entries() -> None:
    result = validate_citations(
        citations=[
            {"source_path": "docs/a.txt", "chunk_id": "chunk-1"},
            {"source_path": "docs/missing.txt", "chunk_id": "chunk-x"},
        ],
        retrieved=[_hit()],
    )
    assert result.valid is False
    assert result.invalid_citations == 1


def test_scan_forbidden_patterns_detects_cloud_sdk_terms(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.py"
    file_path.write_text("import openai\n", encoding="utf-8")
    matches = scan_forbidden_patterns([file_path])
    assert "openai" in matches
