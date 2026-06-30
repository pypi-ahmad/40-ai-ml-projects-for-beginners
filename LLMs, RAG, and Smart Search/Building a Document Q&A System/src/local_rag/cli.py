"""CLI entrypoints for enterprise document Q&A project."""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from local_rag.app import AppRuntime, ensure_eval_file, load_settings
from local_rag.benchmarking import BenchmarkRunner
from local_rag.corpus import (
    build_quickstart_corpus,
    corpus_stats,
    ensure_mixed_formats,
    write_corpus_manifest,
)
from local_rag.corpus_bootstrap import CorpusBootstrapper
from local_rag.evaluator import (
    GenerationEvaluator,
    ResponseEvaluator,
    RetrievalEvaluator,
    dump_generation_metrics,
    dump_retrieval_metrics,
)
from local_rag.failures import run_failure_case_analysis
from local_rag.ground_truth import CandidateQA, generate_candidate_qa, to_eval_examples
from local_rag.llm_judge import LLMJudge, dump_judge_scores
from local_rag.logging_utils import configure_logging
from local_rag.performance import capture_resource_snapshot
from local_rag.retriever import RetrievalStrategy
from local_rag.types import EvalExample
from local_rag.utils import json_dump, read_jsonl, write_jsonl

cli = typer.Typer(help="Production-grade local Document Q&A CLI")

DEFAULT_EVAL_PATH = Path("data/eval/retrieval_eval.jsonl")
DEFAULT_CANDIDATES_PATH = Path("data/eval/retrieval_candidates.jsonl")


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _filters(
    source_type: str | None,
    section: str | None,
    domain: str | None,
) -> dict[str, str] | None:
    payload: dict[str, str] = {}
    if source_type and source_type != "all":
        payload["source_type"] = source_type
    if section:
        payload["section"] = section
    if domain:
        payload["domain"] = domain
    return payload or None


def _load_eval_examples(eval_path: Path) -> list[EvalExample]:
    ensure_eval_file(eval_path)
    rows = read_jsonl(eval_path)
    return [
        EvalExample(
            query=row["query"],
            relevant_doc_ids=list(row.get("relevant_doc_ids", [])),
            relevant_chunk_ids=list(row.get("relevant_chunk_ids", [])),
            answer=row.get("answer"),
        )
        for row in rows
    ]


def _resolve_profile(profile: str) -> str:
    return "quickstart" if profile == "quickstart" else "full"


@cli.command("bootstrap")
def bootstrap_corpus(
    build_quickstart: Annotated[
        bool,
        typer.Option(help="Build quickstart sample set after corpus bootstrap."),
    ] = True,
) -> None:
    """Download mixed-format corpus to local documents directory."""

    configure_logging()
    settings = load_settings()
    settings.ensure_directories()

    bootstrapper = CorpusBootstrapper(settings.documents_dir)
    asyncio.run(bootstrapper.bootstrap())

    full_stats = ensure_mixed_formats(
        source_dir=settings.documents_dir,
        fallback_dir=settings.quickstart_seed_documents_dir,
    )
    write_corpus_manifest(settings.corpus_manifest_path, full_stats)
    logger.info(
        "Corpus files: {} (pdf={}, markdown={}, text={})",
        full_stats.total_files,
        full_stats.pdf_files,
        full_stats.markdown_files,
        full_stats.text_files,
    )

    if build_quickstart:
        quickstart_stats = build_quickstart_corpus(
            source_dir=settings.documents_dir,
            target_dir=settings.quickstart_documents_dir,
        )
        quickstart_manifest = settings.reports_dir / "corpus_manifest_quickstart.json"
        write_corpus_manifest(quickstart_manifest, quickstart_stats)

    logger.info("Corpus bootstrap completed")


@cli.command("corpus-report")
def corpus_report(
    profile: Annotated[
        str,
        typer.Option(help="Corpus profile to report: full or quickstart"),
    ] = "full",
) -> None:
    """Compute and persist corpus stats."""

    configure_logging()
    settings = load_settings()
    settings.corpus_profile = _resolve_profile(profile)
    stats = corpus_stats(settings.active_documents_dir)
    manifest_path = settings.corpus_manifest_path
    if settings.corpus_profile == "quickstart":
        manifest_path = settings.reports_dir / "corpus_manifest_quickstart.json"
    write_corpus_manifest(manifest_path, stats)
    logger.info("Saved corpus report: {}", manifest_path)


@cli.command("list-docs")
def list_docs(
    profile: Annotated[str, typer.Option(help="Corpus profile: full or quickstart")] = "full",
) -> None:
    """List managed documents with version metadata."""

    configure_logging()
    settings = load_settings()
    settings.corpus_profile = _resolve_profile(profile)
    runtime = AppRuntime(settings)
    rows = [asdict(row) for row in runtime.document_manager.list_documents()]
    out_path = settings.reports_dir / f"document_catalog_{_timestamp()}.json"
    json_dump(out_path, {"rows": rows, "count": len(rows)})
    logger.info("Saved document catalog: {}", out_path)


@cli.command("delete-doc")
def delete_doc(
    source_path: Annotated[str, typer.Argument(help="Source-relative document path to delete")],
    profile: Annotated[str, typer.Option(help="Corpus profile: full or quickstart")] = "full",
) -> None:
    """Delete one document from managed corpus."""

    configure_logging()
    settings = load_settings()
    settings.corpus_profile = _resolve_profile(profile)
    runtime = AppRuntime(settings)
    deleted = runtime.document_manager.delete_document(source_path)
    if not deleted:
        raise typer.BadParameter(f"Document not found: {source_path}")
    logger.info("Deleted document: {}", source_path)


@cli.command("update-doc")
def update_doc(
    source_file: Annotated[Path, typer.Argument(help="Local source file to copy into corpus")],
    target_rel_path: Annotated[str, typer.Argument(help="Target relative path under corpus")],
    profile: Annotated[str, typer.Option(help="Corpus profile: full or quickstart")] = "full",
) -> None:
    """Insert or replace one corpus document."""

    configure_logging()
    settings = load_settings()
    settings.corpus_profile = _resolve_profile(profile)
    runtime = AppRuntime(settings)
    if not source_file.exists():
        raise typer.BadParameter(f"Source file missing: {source_file}")
    payload = source_file.read_bytes()
    target = runtime.document_manager.upsert_document(target_rel_path, payload)
    logger.info("Upserted document: {}", target)


@cli.command("ingest")
def ingest(
    chunk_size: Annotated[int | None, typer.Option(help="Chunk size override")] = None,
    chunk_overlap: Annotated[int | None, typer.Option(help="Chunk overlap override")] = None,
    rebuild: Annotated[bool, typer.Option(help="Force full rebuild")] = False,
    profile: Annotated[
        str,
        typer.Option(help="Corpus profile to ingest: full or quickstart"),
    ] = "full",
) -> None:
    """Build or update persistent Chroma + BM25 indexes."""

    configure_logging()
    settings = load_settings()
    settings.corpus_profile = _resolve_profile(profile)
    if settings.corpus_profile == "full":
        ensure_mixed_formats(
            source_dir=settings.documents_dir,
            fallback_dir=settings.quickstart_seed_documents_dir,
        )
    runtime = AppRuntime(settings)

    selected_chunk_size = chunk_size or settings.chunking.default_chunk_size
    selected_chunk_overlap = chunk_overlap or settings.chunking.default_chunk_overlap

    report = runtime.indexer.build_or_update(
        chunk_size=selected_chunk_size,
        chunk_overlap=selected_chunk_overlap,
        force_rebuild=rebuild,
    )

    snapshot = capture_resource_snapshot(settings.vector_db_path)
    payload = asdict(report) | asdict(snapshot) | {
        "corpus_profile": settings.corpus_profile,
        "collection_name": settings.active_collection_name,
    }
    report_path = settings.benchmarks_dir / f"indexing_{_timestamp()}.json"
    json_dump(report_path, payload)

    logger.info("Index mode: {}", report.mode)
    logger.info("Saved benchmark: {}", report_path)


@cli.command("query")
def query(
    question: Annotated[str, typer.Argument(help="Question to ask")],
    top_k: Annotated[int | None, typer.Option(help="Top-k retrieval")] = None,
    model: Annotated[str | None, typer.Option(help="Generation model override")] = None,
    source_type: Annotated[
        str | None,
        typer.Option(help="Optional metadata filter, e.g. pdf/markdown/txt"),
    ] = None,
    section: Annotated[str | None, typer.Option(help="Optional metadata section filter")] = None,
    domain: Annotated[str | None, typer.Option(help="Optional metadata domain filter")] = None,
    strategy: Annotated[
        RetrievalStrategy,
        typer.Option(help="Retrieval strategy: vector|keyword|hybrid"),
    ] = "hybrid",
    prompt_template: Annotated[
        str,
        typer.Option(help="Prompt template name"),
    ] = "enterprise_qa",
    stream: Annotated[bool, typer.Option(help="Stream answer tokens")] = False,
    temperature: Annotated[
        float | None,
        typer.Option(help="Generation temperature override"),
    ] = None,
    max_tokens: Annotated[
        int | None,
        typer.Option(help="Generation max tokens override"),
    ] = None,
    profile: Annotated[
        str,
        typer.Option(help="Corpus profile to query: full or quickstart"),
    ] = "full",
) -> None:
    """Run RAG question answering query."""

    configure_logging()
    settings = load_settings()
    settings.corpus_profile = _resolve_profile(profile)
    runtime = AppRuntime(settings, generation_model=model)
    filters = _filters(source_type=source_type, section=section, domain=domain)
    target_k = top_k or settings.retrieval.default_k

    if stream:
        stream_iter, session = runtime.pipeline.ask_stream(
            query=question,
            top_k=target_k,
            filters=filters,
            strategy=strategy,
            prompt_template=prompt_template,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        started_generation = time.perf_counter()
        tokens: list[str] = []
        for token in stream_iter:
            tokens.append(token)
            typer.echo(token, nl=False)
        typer.echo()
        generation_ms = (time.perf_counter() - started_generation) * 1000
        result = runtime.pipeline.finalize_stream(
            session=session,
            answer="".join(tokens),
            generation_ms=generation_ms,
        )
    else:
        result = runtime.pipeline.ask(
            query=question,
            top_k=target_k,
            filters=filters,
            strategy=strategy,
            prompt_template=prompt_template,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )

    run_path = settings.reports_dir / f"query_{_timestamp()}.json"
    json_dump(
        run_path,
        {
            "query": result.query,
            "answer": result.answer,
            "top_k": result.top_k,
            "model": result.model,
            "strategy": result.retrieval_strategy,
            "timings": asdict(result.timings),
            "citations": result.citations,
            "answer_length": len(result.answer.split()),
            "citation_count": len(result.citations),
            "filters": filters or {},
        },
    )

    logger.info("Answer: {}", result.answer)
    logger.info("Citations: {}", result.citations)
    logger.info("Saved query output: {}", run_path)


@cli.command("evaluate")
def evaluate(
    eval_path: Annotated[Path, typer.Option(help="Evaluation JSONL path")] = DEFAULT_EVAL_PATH,
    strategy: Annotated[
        RetrievalStrategy,
        typer.Option(help="Primary strategy for generation metrics"),
    ] = "hybrid",
    profile: Annotated[
        str,
        typer.Option(help="Corpus profile to evaluate: full or quickstart"),
    ] = "full",
) -> None:
    """Run retrieval and generation evaluation over labeled examples."""

    configure_logging()
    settings = load_settings()
    settings.corpus_profile = _resolve_profile(profile)
    runtime = AppRuntime(settings)

    examples = _load_eval_examples(eval_path)
    retrieval_evaluator = RetrievalEvaluator(runtime.retriever)
    retrieval_metrics = retrieval_evaluator.evaluate(
        examples,
        ks=tuple(settings.retrieval.candidate_ks),
    )

    response_evaluator = ResponseEvaluator(runtime.pipeline)
    response_metrics = response_evaluator.evaluate(
        (row.query for row in examples),
        top_k=settings.retrieval.default_k,
        strategy=strategy,
    )

    generation_evaluator = GenerationEvaluator(runtime.pipeline)
    generation_metrics = generation_evaluator.evaluate(
        examples,
        top_k=settings.retrieval.default_k,
        strategy=strategy,
    )

    retrieval_path = settings.reports_dir / f"retrieval_metrics_{_timestamp()}.jsonl"
    dump_retrieval_metrics(retrieval_path, retrieval_metrics)
    response_path = settings.reports_dir / f"response_metrics_{_timestamp()}.json"
    json_dump(response_path, asdict(response_metrics))
    generation_path = settings.reports_dir / f"generation_metrics_{_timestamp()}.jsonl"
    dump_generation_metrics(generation_path, generation_metrics)

    logger.info("Saved retrieval metrics: {}", retrieval_path)
    logger.info("Saved response metrics: {}", response_path)
    logger.info("Saved generation metrics: {}", generation_path)


@cli.command("generate-eval-set")
def generate_eval_set(
    output_path: Annotated[
        Path,
        typer.Option(help="Output candidate set for manual verification"),
    ] = DEFAULT_CANDIDATES_PATH,
    max_examples: Annotated[int, typer.Option(help="Maximum generated candidate rows")] = 250,
    profile: Annotated[
        str,
        typer.Option(help="Corpus profile for candidate generation"),
    ] = "full",
) -> None:
    """Generate hybrid auto+manual retrieval evaluation candidates."""

    configure_logging()
    settings = load_settings()
    settings.corpus_profile = _resolve_profile(profile)
    runtime = AppRuntime(settings)

    docs = runtime.loader.load_directory(settings.active_documents_dir)
    candidates = generate_candidate_qa(docs, max_examples=max_examples)

    write_jsonl(
        output_path,
        [
            {
                "query": row.query,
                "answer_hint": row.answer_hint,
                "doc_id": row.doc_id,
                "source_path": row.source_path,
                "verified": row.verified,
            }
            for row in candidates
        ],
    )
    logger.info("Saved candidate eval set: {} (rows={})", output_path, len(candidates))


@cli.command("compile-eval-set")
def compile_eval_set(
    input_path: Annotated[
        Path,
        typer.Option(help="Manually curated candidate file"),
    ] = DEFAULT_CANDIDATES_PATH,
    output_path: Annotated[
        Path,
        typer.Option(help="Compiled evaluation file"),
    ] = DEFAULT_EVAL_PATH,
) -> None:
    """Compile verified candidate rows into retrieval evaluation set."""

    configure_logging()
    rows = read_jsonl(input_path)
    candidates = [
        CandidateQA(
            query=row["query"],
            answer_hint=row["answer_hint"],
            doc_id=row["doc_id"],
            source_path=row.get("source_path", "unknown"),
            verified=bool(row.get("verified", False)),
        )
        for row in rows
    ]
    examples = to_eval_examples(candidates)

    write_jsonl(
        output_path,
        [
            {
                "query": row.query,
                "relevant_doc_ids": row.relevant_doc_ids,
                "relevant_chunk_ids": row.relevant_chunk_ids,
                "answer": row.answer,
            }
            for row in examples
        ],
    )
    logger.info("Saved compiled eval set: {} (rows={})", output_path, len(examples))


@cli.command("judge")
def judge(
    query: Annotated[str, typer.Option(help="Question")],
    answer: Annotated[str, typer.Option(help="Answer to judge")],
    context: Annotated[str, typer.Option(help="Retrieved context text")],
    reference: Annotated[str | None, typer.Option(help="Optional reference answer")] = None,
) -> None:
    """Run LLM-as-a-judge rubric scoring with local model."""

    configure_logging()
    settings = load_settings()

    judge_model = LLMJudge(host=settings.ollama_host, model=settings.judge_model)
    score = judge_model.evaluate(
        query=query,
        answer=answer,
        context=context,
        reference_answer=reference,
    )

    out_path = settings.reports_dir / f"judge_scores_{_timestamp()}.jsonl"
    dump_judge_scores(out_path, [score])
    logger.info("Judge score: {}", asdict(score))
    logger.info("Saved judge scores: {}", out_path)


@cli.command("judge-batch")
def judge_batch(
    input_path: Annotated[
        Path,
        typer.Option(help="JSONL rows: query, answer, context, reference_answer"),
    ] = Path("outputs/reports/judge_input.jsonl"),
) -> None:
    """Run local LLM-as-a-judge over many rows and persist aggregate."""

    configure_logging()
    settings = load_settings()
    rows = read_jsonl(input_path)
    judge_model = LLMJudge(host=settings.ollama_host, model=settings.judge_model)
    scores = judge_model.evaluate_batch(rows)

    out_path = settings.reports_dir / f"judge_scores_{_timestamp()}.jsonl"
    dump_judge_scores(out_path, scores)
    aggregate = asdict(judge_model.aggregate(scores))
    summary_path = settings.reports_dir / f"judge_summary_{_timestamp()}.json"
    json_dump(summary_path, aggregate)
    logger.info("Saved judge batch scores: {}", out_path)
    logger.info("Saved judge aggregate: {}", summary_path)


@cli.command("failures")
def failures() -> None:
    """Run canned failure-case analysis."""

    configure_logging()
    settings = load_settings()
    runtime = AppRuntime(settings)

    results = run_failure_case_analysis(runtime.pipeline)
    out_path = settings.reports_dir / f"failure_cases_{_timestamp()}.json"
    json_dump(out_path, {"rows": [asdict(row) for row in results]})
    logger.info("Saved failure analysis: {}", out_path)


@cli.command("benchmark")
def benchmark(
    profile: Annotated[
        str,
        typer.Option(help="Corpus profile to benchmark: full or quickstart"),
    ] = "full",
    query_file: Annotated[
        Path,
        typer.Option(help="Text file with one query per line"),
    ] = Path("data/eval/benchmark_queries.txt"),
) -> None:
    """Run retrieval+generation latency benchmarks across strategies."""

    configure_logging()
    settings = load_settings()
    settings.corpus_profile = _resolve_profile(profile)
    runtime = AppRuntime(settings)

    if not query_file.exists():
        query_file.parent.mkdir(parents=True, exist_ok=True)
        query_file.write_text(
            "\n".join(
                [
                    "Which document discusses encryption requirements?",
                    "Compare policy and technical documentation guidance on access control.",
                    "What changed between two manuals on logging behavior?",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    queries = [
        line.strip()
        for line in query_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    runner = BenchmarkRunner(runtime)
    rows = [
        runner.run_queries(queries, top_k=settings.retrieval.default_k, strategy="vector"),
        runner.run_queries(queries, top_k=settings.retrieval.default_k, strategy="keyword"),
        runner.run_queries(queries, top_k=settings.retrieval.default_k, strategy="hybrid"),
    ]
    out_path = settings.benchmarks_dir / f"benchmark_{_timestamp()}.json"
    BenchmarkRunner.save(out_path, rows)
    logger.info("Saved benchmark report: {}", out_path)


@cli.command("run-experiments")
def run_experiments(
    eval_path: Annotated[Path, typer.Option(help="Evaluation JSONL path")] = DEFAULT_EVAL_PATH,
    profile: Annotated[
        str,
        typer.Option(help="Corpus profile to evaluate: full or quickstart"),
    ] = "full",
) -> None:
    """Run chunk-size/overlap and top-k retrieval experiments."""

    configure_logging()
    settings = load_settings()
    settings.corpus_profile = _resolve_profile(profile)
    runtime = AppRuntime(settings)
    examples = _load_eval_examples(eval_path)

    rows: list[dict[str, float | int | str]] = []
    evaluator = RetrievalEvaluator(runtime.retriever)

    for chunk_size in settings.chunking.chunk_sizes:
        for chunk_overlap in settings.chunking.chunk_overlaps:
            report = runtime.indexer.build_or_update(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                force_rebuild=True,
            )
            metrics = evaluator.evaluate(examples, ks=tuple(settings.retrieval.candidate_ks))
            for metric in metrics:
                rows.append(
                    {
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "strategy": metric.strategy,
                        "k": metric.k,
                        "precision_at_k": metric.precision_at_k,
                        "recall_at_k": metric.recall_at_k,
                        "mrr": metric.mrr,
                        "ndcg": metric.ndcg,
                        "avg_retrieval_latency_ms": metric.avg_retrieval_latency_ms,
                        "vector_count": report.vector_count,
                        "lexical_chunk_count": report.lexical_chunk_count,
                    }
                )

    out_path = settings.reports_dir / f"chunk_topk_experiments_{_timestamp()}.jsonl"
    write_jsonl(out_path, rows)
    logger.info("Saved experiment report: {}", out_path)


@cli.command("diagram")
def diagram() -> None:
    """Generate architecture diagram artifact."""

    configure_logging()
    settings = load_settings()
    settings.ensure_directories()

    from local_rag.visualization import save_pipeline_architecture_diagram

    path = settings.diagrams_dir / "pipeline_architecture.png"
    save_pipeline_architecture_diagram(path)
    logger.info("Saved diagram: {}", path)


@cli.command("run-all")
def run_all(
    profile: Annotated[
        str,
        typer.Option(help="Corpus profile: full or quickstart"),
    ] = "full",
    eval_path: Annotated[
        Path,
        typer.Option(help="Evaluation JSONL path"),
    ] = DEFAULT_EVAL_PATH,
) -> None:
    """Run ingest, evaluation, benchmark, failures, and diagram generation."""

    configure_logging()
    settings = load_settings()
    settings.corpus_profile = _resolve_profile(profile)
    runtime = AppRuntime(settings)

    runtime.indexer.build_or_update(
        chunk_size=settings.chunking.default_chunk_size,
        chunk_overlap=settings.chunking.default_chunk_overlap,
        force_rebuild=False,
    )

    typer.echo("Running evaluation...")
    evaluate(eval_path=eval_path, strategy="hybrid", profile=profile)
    typer.echo("Running benchmark...")
    benchmark(profile=profile)
    typer.echo("Running failure analysis...")
    failures()
    typer.echo("Generating diagram...")
    diagram()
    typer.echo("Run-all completed.")



def main() -> None:
    """Run CLI."""

    cli()


if __name__ == "__main__":
    main()
