"""CLI entrypoints for hybrid research assistant."""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Annotated

import typer
from loguru import logger

from hybrid_research_assistant.app import build_runtime, load_settings
from hybrid_research_assistant.benchmark import run_full_benchmark
from hybrid_research_assistant.evaluation import evaluate_benchmark, load_eval_samples, save_eval_results
from hybrid_research_assistant.failure_analysis import run_failure_analysis
from hybrid_research_assistant.schemas import ChunkingStrategy, RetrievalMode
from hybrid_research_assistant.utils import json_dump, write_jsonl

app = typer.Typer(help="Hybrid Real-Time AI Research Assistant CLI")


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


@contextmanager
def runtime_session() -> Iterator:
    """Build runtime and guarantee cleanup."""

    runtime = build_runtime()
    try:
        yield runtime
    finally:
        runtime.close()


@app.command("bootstrap-corpus")
def bootstrap_corpus() -> None:
    """Create starter corpus directories and notes placeholder."""

    settings = load_settings()
    settings.ensure_directories()

    placeholder = settings.paths.documents_dir / "README_CORPUS.md"
    if not placeholder.exists():
        placeholder.write_text(
            "# Corpus Bootstrap\n\n"
            "Place high-quality PDFs, markdown, text, HTML, and DOCX documents in this directory.\n"
            "Use scripts/download_corpus.sh for automated bootstrap.\n",
            encoding="utf-8",
        )
    logger.info("Corpus bootstrap complete at {}", settings.paths.documents_dir)


@app.command("ingest")
def ingest(
    chunk_size: Annotated[int | None, typer.Option(help="Chunk size override")] = None,
    chunk_overlap: Annotated[int | None, typer.Option(help="Chunk overlap override")] = None,
    strategy: Annotated[str, typer.Option(help="Chunking strategy: recursive|token|semantic")] = "recursive",
    rebuild: Annotated[bool, typer.Option(help="Force full rebuild")] = False,
) -> None:
    """Build/update persistent Chroma index."""

    with runtime_session() as runtime:
        settings = runtime.settings

        selected_chunk_size = chunk_size or settings.chunking.chunk_size_default
        selected_chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunking.chunk_overlap_default
        selected_strategy = ChunkingStrategy(strategy)

        report = runtime.indexer.build_or_update(
            chunk_size=selected_chunk_size,
            chunk_overlap=selected_chunk_overlap,
            strategy=selected_strategy,
            force_rebuild=rebuild,
        )

        output_path = settings.paths.outputs_dir / "benchmarks" / f"indexing_{_ts()}.json"
        json_dump(output_path, asdict(report))
        logger.info("Indexing report saved: {}", output_path)


@app.command("query")
def query(
    question: Annotated[str, typer.Argument(help="Question to ask")],
    mode: Annotated[str, typer.Option(help="auto|local|web|hybrid")] = "auto",
    prompt: Annotated[str, typer.Option(help="strict_qa|research_assistant|teacher|technical_mentor|summarizer")] = "research_assistant",
    provider: Annotated[str | None, typer.Option(help="duckduckgo|tavily|brave")] = None,
) -> None:
    """Run one assistant query and print JSON response."""

    with runtime_session() as runtime:
        cache_hit = runtime.cache.get(question)
        if cache_hit is not None:
            payload = {
                "query": question,
                "mode": cache_hit.mode,
                "answer": cache_hit.response,
                "cached": True,
            }
            typer.echo(json.dumps(payload, indent=2, ensure_ascii=True))
            return

        response = runtime.workflow.ask(
            question,
            mode=RetrievalMode(mode),
            prompt_name=prompt,
            provider=provider,
        )

        runtime.cache.put(question, response.answer, response.mode.value)
        runtime.memory.add("user", question)
        runtime.memory.add("assistant", response.answer)

        payload = {
            "query": response.query,
            "mode": response.mode.value,
            "route_reason": response.route_reason,
            "answer": response.answer,
            "citations": [asdict(citation) for citation in response.citations],
            "judge": response.judge,
            "timings_ms": asdict(response.timings),
            "cached": False,
        }
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=True))


@app.command("evaluate")
def evaluate(
    max_samples: Annotated[int, typer.Option(help="Maximum samples to evaluate")] = 100,
    force_mode: Annotated[str, typer.Option(help="Force mode: auto|local|web|hybrid")] = "auto",
) -> None:
    """Run benchmark evaluation and write reports."""

    with runtime_session() as runtime:
        settings = runtime.settings

        samples = load_eval_samples(settings.evaluation.benchmark_path)[:max_samples]
        forced = None if force_mode == "auto" else RetrievalMode(force_mode)
        rows, report = evaluate_benchmark(runtime.workflow, samples, force_mode=forced)

        output_dir = settings.evaluation.reports_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        rows_path = output_dir / f"evaluation_rows_{_ts()}.jsonl"
        report_path = output_dir / f"evaluation_report_{_ts()}.json"

        save_eval_results(rows_path, rows)
        json_dump(report_path, report.model_dump())
        logger.info("Evaluation rows: {}", rows_path)
        logger.info("Evaluation report: {}", report_path)


@app.command("benchmark")
def benchmark() -> None:
    """Run chunking and embedding benchmarks."""

    with runtime_session() as runtime:
        settings = runtime.settings

        artifacts = run_full_benchmark(
            loader=runtime.loader,
            chunk_sizes=settings.chunking.chunk_sizes,
            overlaps=settings.chunking.chunk_overlaps,
            embedding_models=settings.models.embedding_candidates,
            ollama_host=settings.ollama_host,
            output_dir=settings.paths.outputs_dir / "benchmarks",
        )
        logger.info("Benchmark artifacts: {}", artifacts)


@app.command("run-failure-analysis")
def failure_analysis(
    include_web: Annotated[bool, typer.Option(help="Include web/hybrid failure scenarios")] = False,
) -> None:
    """Run failure-mode checks and write report."""

    with runtime_session() as runtime:
        rows = run_failure_analysis(runtime.workflow, include_web=include_web)
        report_path = runtime.settings.paths.outputs_dir / "reports" / f"failure_analysis_{_ts()}.jsonl"
        write_jsonl(report_path, [asdict(row) for row in rows])
        logger.info("Failure analysis report: {}", report_path)


@app.command("generate-eval-set")
def generate_eval_set() -> None:
    """Generate a 100-question benchmark dataset with category balance."""

    settings = load_settings()
    random.seed(settings.runtime.seed)

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
            "According to documentation, what is {detail} in {topic}?",
        ],
        "reasoning": [
            "Why would {topic} choose {detail} under constrained conditions?",
            "Explain tradeoffs of {detail} in {topic}.",
        ],
        "comparison": [
            "Compare {topic} and {detail} for production usage.",
            "What is difference between {topic} and {detail}?",
        ],
        "summarization": [
            "Summarize key points from {topic} about {detail}.",
            "Provide concise summary of {topic} section on {detail}.",
        ],
        "multi_document": [
            "Synthesize how {topic} and {detail} align across multiple sources.",
            "Combine evidence from docs about {topic} and {detail}.",
        ],
        "fresh_knowledge": [
            "What happened recently with {topic} related to {detail}?",
            "Latest announcement about {topic} and {detail}.",
        ],
    }

    topics = [
        "LangGraph", "FastAPI", "Python docs", "Scikit-learn", "CUDA", "Linux kernel", "RAG systems",
        "vector databases", "prompt engineering", "retrieval pipelines", "enterprise policy", "LLM evaluation",
    ]
    details = [
        "routing", "chunking", "memory", "latency", "indexing", "grounding", "citations", "security",
        "failure recovery", "benchmarking", "metadata filtering", "reranking",
    ]

    rows = []
    cursor = 1
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
                    "id": f"q{cursor:03d}",
                    "category": category,
                    "question": template.format(topic=topic, detail=detail),
                    "expected_mode": expected_mode,
                    "expected_keywords": [topic.lower(), detail.lower()],
                    "expected_sources": [],
                    "reference_answer": None,
                }
            )
            cursor += 1

    write_jsonl(settings.evaluation.benchmark_path, rows)
    logger.info("Generated evaluation dataset with {} rows at {}", len(rows), settings.evaluation.benchmark_path)


@app.command("export-workflow")
def export_workflow() -> None:
    """Export workflow diagram artifact."""

    with runtime_session() as runtime:
        output = runtime.workflow.export_diagram(
            runtime.settings.paths.outputs_dir / "diagrams" / "langgraph_workflow.mmd"
        )
        logger.info("Workflow diagram exported to {}", output)


if __name__ == "__main__":
    app()
