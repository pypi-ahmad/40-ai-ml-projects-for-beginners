"""Generate weakly labeled evaluation queries with category coverage."""

from __future__ import annotations

import json
import random
from pathlib import Path

from semantic_search.config import load_config
from semantic_search.logging_utils import configure_logging, get_logger
from semantic_search.schemas import EvaluationCase
from semantic_search.service import SemanticSearchService


CATEGORIES = ["technical", "general", "reasoning", "comparison", "multi_document"]


def make_query(doc_title: str, category: str) -> str:
    if category == "technical":
        return f"Technical details about {doc_title}"
    if category == "general":
        return f"Summary of {doc_title}"
    if category == "reasoning":
        return f"Why does {doc_title} matter?"
    if category == "comparison":
        return f"Compare viewpoints related to {doc_title}"
    return f"Find multiple related pieces about {doc_title}"


def main() -> None:
    config = load_config()
    configure_logging(config)
    logger = get_logger()
    service = SemanticSearchService(config)
    docs = service.load_documents()

    rng = random.Random(config.dataset.seed)
    sample_docs = rng.sample(docs, min(config.evaluation.query_count, len(docs)))

    cases: list[EvaluationCase] = []
    for idx, doc in enumerate(sample_docs):
        category = CATEGORIES[idx % len(CATEGORIES)]
        query_text = make_query(doc.title or doc.doc_id, category)
        case = EvaluationCase(
            query_id=f"q_{idx:04d}",
            query=query_text,
            category=category,
            relevant_doc_ids=[doc.doc_id],
            notes="Weak label generated from source title; requires manual audit subset.",
        )
        cases.append(case)

    output_path = Path(config.paths["processed_data_dir"]) / "evaluation_queries.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(case.model_dump_json() + "\n")

    manifest_path = Path(config.paths["reports_dir"]) / "evaluation_generation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "query_count": len(cases),
                "method": "weak_labels_plus_manual_audit_required",
                "seed": config.dataset.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("evaluation_queries_generated", count=len(cases), path=str(output_path))


if __name__ == "__main__":
    main()
