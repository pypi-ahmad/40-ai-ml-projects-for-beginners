from __future__ import annotations

from local_rag.prompts import SYSTEM_PROMPT, build_context_block, build_messages
from local_rag.types import RetrievalResult


def _hit() -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk_1234567890",
        doc_id="doc1",
        text="ACPI controls power and configuration interfaces.",
        score=0.91,
        metadata={"source_path": "linux/acpi.txt", "doc_id": "doc1"},
    )


def test_system_prompt_contains_unavailable_policy() -> None:
    assert "Information unavailable in provided context." in SYSTEM_PROMPT


def test_context_block_contains_source_and_score() -> None:
    block = build_context_block([_hit()])
    assert "source_path: linux/acpi.txt" in block
    assert "similarity_score:" in block


def test_build_messages_shape() -> None:
    messages = build_messages("What is ACPI?", [_hit()])
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "Question:" in messages[1]["content"]
