"""Run failure-mode analysis scenarios and save report."""

from __future__ import annotations

import json
from pathlib import Path

from semantic_search.config import load_config
from semantic_search.logging_utils import configure_logging, get_logger
from semantic_search.schemas import SearchRequest
from semantic_search.service import SemanticSearchService


SCENARIOS = [
    {"name": "misspelled_query", "query": "machien lernng retrievel", "mode": "hybrid"},
    {"name": "very_short_query", "query": "ai", "mode": "hybrid"},
    {"name": "very_long_query", "query": " ".join(["semantic"] * 120), "mode": "hybrid"},
    {"name": "no_relevant_documents", "query": "mars colony legal framework for interplanetary courts", "mode": "hybrid"},
    {"name": "mixed_language", "query": "inteligencia artificial noticias and machine learning", "mode": "hybrid"},
]


def main() -> None:
    config = load_config()
    configure_logging(config)
    logger = get_logger()

    service = SemanticSearchService(config)
    service.load_documents()
    service.load_chunks()
    service.build_indexes(config.embedding.primary)

    report: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        response = service.search(
            SearchRequest(
                query=str(scenario["query"]),
                mode=str(scenario["mode"]),
                top_k=10,
                rerank=True,
            )
        )
        report.append(
            {
                "scenario": scenario["name"],
                "query": scenario["query"],
                "latency_ms": response.latency_ms,
                "hit_count": len(response.hits),
                "top_document_ids": [hit.document_id for hit in response.hits[:3]],
                "notes": "Review low-hit scenarios for query rewrite and hybrid weighting updates.",
            }
        )

    output_path = Path(config.paths["reports_dir"]) / "failure_analysis_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("failure_analysis_complete", path=str(output_path), scenarios=len(report))


if __name__ == "__main__":
    main()
