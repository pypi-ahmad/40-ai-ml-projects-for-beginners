#!/usr/bin/env python3
"""Generate 100-question benchmark dataset for hybrid assistant evaluation."""

from __future__ import annotations

import json
import random
from pathlib import Path


def main() -> None:
    random.seed(42)

    categories = [
        ("factual", 20),
        ("reasoning", 20),
        ("comparison", 20),
        ("summarization", 15),
        ("multi_document", 15),
        ("fresh_knowledge", 10),
    ]

    templates = {
        "factual": [
            "What does {topic} define as {detail}?",
            "According to docs, what is {detail} in {topic}?",
        ],
        "reasoning": [
            "Why is {detail} important in {topic}?",
            "Explain tradeoffs of {detail} in {topic}.",
        ],
        "comparison": [
            "Compare {topic} and {detail} for production use.",
            "What are differences between {topic} and {detail}?",
        ],
        "summarization": [
            "Summarize key points from {topic} about {detail}.",
            "Give concise summary of {topic} on {detail}.",
        ],
        "multi_document": [
            "Synthesize how {topic} and {detail} align across multiple sources.",
            "Combine evidence from docs on {topic} and {detail}.",
        ],
        "fresh_knowledge": [
            "What happened in AI recently about {topic} and {detail}?",
            "Latest updates on {topic} regarding {detail}.",
        ],
    }

    topics = [
        "LangGraph",
        "FastAPI",
        "Python documentation",
        "Scikit-learn",
        "CUDA",
        "Linux documentation",
        "RAG architecture",
        "vector databases",
        "Ollama",
        "retrieval pipelines",
        "enterprise policy",
        "LLM evaluation",
    ]

    details = [
        "routing",
        "chunking",
        "memory",
        "latency",
        "indexing",
        "grounding",
        "citation quality",
        "security",
        "failure recovery",
        "benchmarking",
        "metadata filtering",
        "reranking",
    ]

    rows = []
    idx = 1
    for category, count in categories:
        for _ in range(count):
            template = random.choice(templates[category])
            topic = random.choice(topics)
            detail = random.choice(details)
            expected_mode = "web" if category == "fresh_knowledge" else "local"
            if category in {"comparison", "multi_document"}:
                expected_mode = "hybrid"

            rows.append(
                {
                    "id": f"q{idx:03d}",
                    "category": category,
                    "question": template.format(topic=topic, detail=detail),
                    "expected_mode": expected_mode,
                    "expected_keywords": [topic.lower(), detail.lower()],
                    "expected_sources": [],
                    "reference_answer": None,
                }
            )
            idx += 1

    output = Path("data/eval/benchmark_questions.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
