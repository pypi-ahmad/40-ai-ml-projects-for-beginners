"""Hybrid auto+manual ground-truth utilities for retrieval evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass

from local_rag.types import EvalExample, LoadedDocument


@dataclass(slots=True)
class CandidateQA:
    """Auto-generated candidate QA row for manual curation."""

    query: str
    answer_hint: str
    doc_id: str
    source_path: str
    verified: bool = False


def generate_candidate_qa(
    docs: list[LoadedDocument],
    *,
    max_examples: int = 200,
    random_seed: int = 42,
) -> list[CandidateQA]:
    """Generate candidate QA pairs from document snippets."""

    random.seed(random_seed)
    candidates: list[CandidateQA] = []

    sampled_docs = docs.copy()
    random.shuffle(sampled_docs)

    for doc in sampled_docs:
        if len(candidates) >= max_examples:
            break

        words = doc.text.split()
        snippet = " ".join(words[:40]).strip()
        if len(snippet) < 40:
            continue

        source_path = str(doc.metadata.get("source_path", "unknown"))
        # Lexical anchor query is intentionally built from source text to make
        # auto-generated eval rows retrievable before manual curation.
        anchor = " ".join(words[:20]).strip().rstrip(".,;:")
        query = f"{anchor}?"
        candidates.append(
            CandidateQA(
                query=query,
                answer_hint=snippet,
                doc_id=doc.doc_id,
                source_path=source_path,
            )
        )

    return candidates


def to_eval_examples(rows: list[CandidateQA]) -> list[EvalExample]:
    """Convert manually verified candidate rows into EvalExample list."""

    examples: list[EvalExample] = []
    for row in rows:
        if not row.verified:
            continue
        examples.append(
            EvalExample(
                query=row.query,
                relevant_doc_ids=[row.doc_id],
                relevant_chunk_ids=[],
                answer=row.answer_hint,
            )
        )
    return examples
