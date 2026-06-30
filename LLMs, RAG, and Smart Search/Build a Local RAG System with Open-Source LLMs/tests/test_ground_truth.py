from __future__ import annotations

from local_rag.ground_truth import CandidateQA, generate_candidate_qa, to_eval_examples
from local_rag.types import LoadedDocument


def _doc(text: str, source_path: str, doc_id: str) -> LoadedDocument:
    return LoadedDocument(
        doc_id=doc_id,
        text=text,
        metadata={"source_path": source_path},
    )


def test_generate_candidate_qa_uses_lexical_anchor_query() -> None:
    docs = [
        _doc(
            text=(
                "Advanced Configuration and Power Interface defines power states and "
                "hardware configuration interfaces for operating systems."
            ),
            source_path="docs/acpi.txt",
            doc_id="doc-1",
        )
    ]
    rows = generate_candidate_qa(docs, max_examples=1, random_seed=1)
    assert len(rows) == 1
    assert rows[0].query.endswith("?")
    assert "Advanced Configuration and Power Interface" in rows[0].query
    assert rows[0].doc_id == "doc-1"


def test_to_eval_examples_requires_verified_rows() -> None:
    rows = [
        CandidateQA(
            query="sample question?",
            answer_hint="sample answer",
            doc_id="doc-1",
            source_path="docs/a.txt",
            verified=False,
        ),
        CandidateQA(
            query="verified question?",
            answer_hint="verified answer",
            doc_id="doc-2",
            source_path="docs/b.txt",
            verified=True,
        ),
    ]
    examples = to_eval_examples(rows)
    assert len(examples) == 1
    assert examples[0].relevant_doc_ids == ["doc-2"]

