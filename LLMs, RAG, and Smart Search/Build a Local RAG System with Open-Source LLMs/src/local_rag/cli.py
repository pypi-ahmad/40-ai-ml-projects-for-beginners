"""CLI entrypoints for local RAG project."""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from ollama import Client as OllamaClient

from local_rag.app import AppRuntime, ensure_eval_file, load_settings
from local_rag.audit import scan_forbidden_patterns, validate_citations
from local_rag.config import AppSettings
from local_rag.corpus import (
    build_quickstart_corpus,
    validate_corpus,
    write_corpus_manifest,
)
from local_rag.corpus_bootstrap import CorpusBootstrapper
from local_rag.evaluator import (
    ResponseEvaluator,
    RetrievalEvaluator,
    dump_retrieval_metrics,
)
from local_rag.failures import run_failure_case_analysis
from local_rag.ground_truth import CandidateQA, generate_candidate_qa, to_eval_examples
from local_rag.llm_judge import LLMJudge, dump_judge_scores
from local_rag.logging_utils import configure_logging
from local_rag.performance import capture_resource_snapshot
from local_rag.types import EvalExample
from local_rag.utils import json_dump, read_jsonl, write_jsonl

cli = typer.Typer(help="Local production-style RAG CLI")

DEFAULT_EVAL_PATH = Path("data/eval/retrieval_eval.jsonl")
DEFAULT_CANDIDATES_PATH = Path("data/eval/retrieval_candidates.jsonl")


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _filters(source_type: str | None, section: str | None) -> dict[str, str] | None:
    payload: dict[str, str] = {}
    if source_type:
        payload["source_type"] = source_type
    if section:
        payload["section"] = section
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


def _collect_source_files() -> list[Path]:
    roots = [Path("src"), Path("tests"), Path("streamlit_app"), Path("scripts")]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(path for path in root.rglob("*") if path.is_file())
    return files


def _join_errors(errors: list[str]) -> str:
    return "; ".join(errors)


def _doctor_hints(failed_checks: list[str], settings: AppSettings) -> list[str]:
    failed = set(failed_checks)
    hints: list[str] = []
    missing_models: list[str] = []

    if "ollama_connected" in failed:
        hints.append(
            f"Ollama is unreachable at {settings.ollama_host}. Start it with `ollama serve`."
        )
    if "embedding_model_available" in failed:
        missing_models.append(settings.embedding_model)
    if "generation_model_available" in failed:
        missing_models.append(settings.generation_model)
    if "judge_model_available" in failed:
        missing_models.append(settings.judge_model)
    if missing_models:
        model_cmds = ", ".join(f"`ollama pull {model}`" for model in dict.fromkeys(missing_models))
        hints.append(f"Install missing model(s): {model_cmds}.")
    if "full_corpus_non_empty" in failed:
        hints.append(
            "Full corpus is empty. Run `uv run python -m local_rag bootstrap --build-quickstart`."
        )
    if "quickstart_non_empty" in failed:
        hints.append(
            "Quickstart corpus is empty. Run `uv run python -m local_rag build-quickstart`."
        )
    if "python_version_3_12" in failed:
        hints.append("Use Python 3.12.x (`uv python install 3.12`).")

    return hints


def _exception_hints(exc: Exception, settings: AppSettings) -> list[str]:
    message = str(exc).lower()
    hints: list[str] = []

    if any(token in message for token in ("connection refused", "connect", "11434", "timeout")):
        hints.append(
            f"Cannot reach Ollama at {settings.ollama_host}. Ensure `ollama serve` is running."
        )
    if "model" in message and any(token in message for token in ("not found", "404")):
        hints.append(
            "Model not available locally. Pull required models: "
            + ", ".join(f"`ollama pull {model}`" for model in settings.required_models)
            + "."
        )
    if any(token in message for token in ("no such file", "file not found")):
        hints.append(
            "Missing local files detected. Bootstrap corpus with "
            "`uv run python -m local_rag bootstrap --build-quickstart`."
        )
    collection_missing = "collection" in message and any(
        token in message for token in ("not found", "does not exist")
    )
    if collection_missing:
        hints.append(
            "Index collection missing. Build it with "
            "`uv run python -m local_rag ingest --profile full`."
        )

    return hints


def _fail_with_hints(action: str, exc: Exception, settings: AppSettings) -> None:
    logger.error("{} failed: {}", action, exc)
    for hint in _exception_hints(exc, settings):
        logger.error("Hint: {}", hint)
    raise typer.Exit(code=1) from exc


@contextmanager
def _command_guard(action: str, settings: AppSettings) -> Iterator[None]:
    try:
        yield
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail_with_hints(action, exc, settings)


@cli.command("doctor")
def doctor() -> None:
    """Run local environment preflight checks."""

    configure_logging()
    settings = load_settings()
    settings.ensure_directories()

    checks: dict[str, bool] = {}
    checks["documents_dir_exists"] = settings.documents_dir.exists()
    checks["quickstart_dir_exists"] = settings.quickstart_documents_dir.exists()
    checks["vector_db_path_exists"] = settings.vector_db_path.exists()
    checks["reports_dir_exists"] = settings.reports_dir.exists()
    checks["python_version_3_12"] = sys.version_info[:2] == (3, 12)

    corpus_validation = validate_corpus(
        settings.documents_dir,
        min_total_files=1,
        min_pdf_files=0,
        min_markdown_files=0,
        min_text_files=0,
    )
    checks["full_corpus_non_empty"] = corpus_validation.stats.total_files > 0

    quickstart_validation = validate_corpus(
        settings.quickstart_documents_dir,
        min_total_files=1,
        min_pdf_files=0,
        min_markdown_files=0,
        min_text_files=0,
    )
    checks["quickstart_non_empty"] = quickstart_validation.stats.total_files > 0

    try:
        ollama_client = OllamaClient(host=settings.ollama_host)
        response = ollama_client.list()
        available_models = {
            row.get("model", "") for row in response.get("models", [])
        }
        checks["ollama_connected"] = True
        checks["embedding_model_available"] = settings.embedding_model in available_models
        checks["generation_model_available"] = settings.generation_model in available_models
        checks["judge_model_available"] = settings.judge_model in available_models
    except Exception:  # noqa: BLE001
        checks["ollama_connected"] = False
        checks["embedding_model_available"] = False
        checks["generation_model_available"] = False
        checks["judge_model_available"] = False

    payload = {
        "checks": checks,
        "full_corpus_stats": asdict(corpus_validation.stats),
        "quickstart_corpus_stats": asdict(quickstart_validation.stats),
    }
    out_path = settings.reports_dir / f"doctor_{_timestamp()}.json"
    json_dump(out_path, payload)
    logger.info("Doctor report saved: {}", out_path)
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        logger.warning("Doctor failed checks: {}", failed)
        for hint in _doctor_hints(failed, settings):
            logger.warning("Doctor hint: {}", hint)
        raise typer.Exit(code=1)
    logger.info("Doctor checks passed")


@cli.command("validate-local")
def validate_local() -> None:
    """Fail if forbidden cloud-LLM patterns are detected in codebase."""

    configure_logging()
    settings = load_settings()
    matches = scan_forbidden_patterns(_collect_source_files())
    out_path = settings.reports_dir / f"local_only_validation_{_timestamp()}.json"
    json_dump(out_path, {"matches": matches})
    logger.info("Saved local-only validation report: {}", out_path)
    if matches:
        raise typer.Exit(code=1)


@cli.command("validate-index")
def validate_index(
    profile: Annotated[
        str,
        typer.Option(help="Corpus profile to validate: full or quickstart"),
    ] = "full",
) -> None:
    """Validate Chroma collection and manifest integrity."""

    configure_logging()
    settings = load_settings()
    settings.corpus_profile = "quickstart" if profile == "quickstart" else "full"
    with _command_guard("validate-index", settings):
        runtime = AppRuntime(settings)
        report = runtime.vector_store.integrity_report()
        manifest = runtime.manifest.load()
        report["manifest_exists"] = bool(manifest)
        report["manifest_documents"] = int(len(manifest.get("documents", {})))
        report["manifest_collection_name_matches"] = (
            manifest.get("collection_name") == settings.active_collection_name
        )
        report["vector_count_matches_collection"] = (
            int(report["total_vectors"]) == runtime.vector_store.count()
        )

        out_path = settings.reports_dir / (
            f"index_integrity_{settings.corpus_profile}_{_timestamp()}.json"
        )
        json_dump(out_path, report)
        logger.info("Saved index integrity report: {}", out_path)
        if (
            int(report["duplicate_chunk_ids"]) > 0
            or int(report["missing_required_metadata"]) > 0
            or not report["manifest_collection_name_matches"]
        ):
            raise typer.Exit(code=1)


@cli.command("bootstrap")
def bootstrap_corpus(
    build_quickstart: Annotated[
        bool,
        typer.Option(help="Build quickstart sample set after corpus bootstrap."),
    ] = True,
) -> None:
    """Download mixed-format Linux corpus to local documents directory."""

    configure_logging()
    settings = load_settings()
    settings.ensure_directories()
    with _command_guard("bootstrap", settings):
        bootstrapper = CorpusBootstrapper(settings.documents_dir)
        asyncio.run(bootstrapper.bootstrap())

        full_validation = validate_corpus(
            settings.documents_dir,
            min_total_files=100,
            min_pdf_files=1,
            min_markdown_files=1,
            min_text_files=1,
        )
        full_stats = full_validation.stats
        if not full_validation.ok:
            raise ValueError(
                "Corpus validation failed after bootstrap: "
                f"{_join_errors(full_validation.errors)}"
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
            logger.info(
                "Quickstart files: {} (pdf={}, markdown={}, text={})",
                quickstart_stats.total_files,
                quickstart_stats.pdf_files,
                quickstart_stats.markdown_files,
                quickstart_stats.text_files,
            )

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
    settings.corpus_profile = "quickstart" if profile == "quickstart" else "full"
    validation = validate_corpus(
        settings.active_documents_dir,
        min_total_files=3 if settings.corpus_profile == "quickstart" else 100,
        min_pdf_files=1,
        min_markdown_files=1,
        min_text_files=1,
    )
    stats = validation.stats
    manifest_path = settings.corpus_manifest_path
    if settings.corpus_profile == "quickstart":
        manifest_path = settings.reports_dir / "corpus_manifest_quickstart.json"
    write_corpus_manifest(manifest_path, stats)
    logger.info("Saved corpus report: {}", manifest_path)
    logger.info(
        "Corpus profile={} files={} mixed_formats={}",
        settings.corpus_profile,
        stats.total_files,
        stats.has_mixed_formats,
    )
    if not validation.ok:
        logger.warning("Corpus validation errors: {}", validation.errors)
        raise typer.Exit(code=1)


@cli.command("build-quickstart")
def build_quickstart(
    max_pdf: Annotated[int, typer.Option(help="Maximum quickstart PDFs")] = 20,
    max_markdown: Annotated[int, typer.Option(help="Maximum quickstart markdown files")] = 40,
    max_text: Annotated[int, typer.Option(help="Maximum quickstart text files")] = 120,
) -> None:
    """Build deterministic quickstart corpus from full corpus directory."""

    configure_logging()
    settings = load_settings()
    settings.ensure_directories()
    with _command_guard("build-quickstart", settings):
        validation = validate_corpus(
            settings.documents_dir,
            min_total_files=100,
            min_pdf_files=1,
            min_markdown_files=1,
            min_text_files=1,
        )
        if not validation.ok:
            raise ValueError(
                "Source corpus invalid for quickstart build: "
                f"{_join_errors(validation.errors)}"
            )

        stats = build_quickstart_corpus(
            source_dir=settings.documents_dir,
            target_dir=settings.quickstart_documents_dir,
            max_pdf=max_pdf,
            max_markdown=max_markdown,
            max_text=max_text,
        )
        manifest_path = settings.reports_dir / "corpus_manifest_quickstart.json"
        write_corpus_manifest(manifest_path, stats)
        logger.info("Quickstart corpus refreshed at {}", settings.quickstart_documents_dir)


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
    """Build or update persistent Chroma index."""

    configure_logging()
    settings = load_settings()
    settings.corpus_profile = "quickstart" if profile == "quickstart" else "full"
    with _command_guard("ingest", settings):
        validation = validate_corpus(
            settings.active_documents_dir,
            min_total_files=3 if settings.corpus_profile == "quickstart" else 100,
            min_pdf_files=1,
            min_markdown_files=1,
            min_text_files=1,
        )
        if not validation.ok:
            raise ValueError(
                "Corpus validation failed: "
                f"{_join_errors(validation.errors)}"
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
        logger.info(
            "Documents: {} | Chunks: {} | Embedded: {}",
            report.total_documents,
            report.total_chunks,
            report.embedded_chunks,
        )
        logger.info(
            "Vector count: {} | Indexing ms: {:.2f}",
            report.vector_count,
            report.indexing_ms,
        )
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
    settings.corpus_profile = "quickstart" if profile == "quickstart" else "full"
    with _command_guard("query", settings):
        runtime = AppRuntime(settings, generation_model=model)
        filters = _filters(source_type=source_type, section=section)
        target_k = top_k or settings.retrieval.default_k
        if target_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if stream:
            stream_iter, session = runtime.pipeline.ask_stream(
                query=question,
                top_k=target_k,
                filters=filters,
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
                temperature=temperature,
                max_tokens=max_tokens,
            )

        run_path = settings.reports_dir / f"query_{_timestamp()}.json"
        citation_audit = validate_citations(result.citations, result.retrieved)
        json_dump(
            run_path,
            {
                "query": result.query,
                "answer": result.answer,
                "top_k": result.top_k,
                "model": result.model,
                "timings": asdict(result.timings),
                "citations": result.citations,
                "answer_length": len(result.answer.split()),
                "citation_count": len(result.citations),
                "filters": filters or {},
                "citation_audit": asdict(citation_audit),
            },
        )

        logger.info("Answer: {}", result.answer)
        logger.info("Citations: {}", result.citations)
        logger.info("Citation audit valid: {}", citation_audit.valid)
        logger.info("Timings: {}", asdict(result.timings))
        logger.info("Saved query output: {}", run_path)


@cli.command("evaluate")
def evaluate(
    eval_path: Annotated[Path, typer.Option(help="Evaluation JSONL path")] = DEFAULT_EVAL_PATH,
    profile: Annotated[
        str,
        typer.Option(help="Corpus profile to evaluate: full or quickstart"),
    ] = "full",
) -> None:
    """Run retrieval and response evaluation over labeled examples."""

    configure_logging()
    settings = load_settings()
    settings.corpus_profile = "quickstart" if profile == "quickstart" else "full"
    with _command_guard("evaluate", settings):
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
        )

        out_path = settings.reports_dir / f"retrieval_metrics_{_timestamp()}.jsonl"
        dump_retrieval_metrics(out_path, retrieval_metrics)
        summary_path = settings.reports_dir / f"response_metrics_{_timestamp()}.json"
        json_dump(summary_path, asdict(response_metrics))
        logger.info("Saved retrieval metrics: {}", out_path)
        logger.info("Saved response metrics: {}", summary_path)


@cli.command("generate-eval-set")
def generate_eval_set(
    output_path: Annotated[
        Path,
        typer.Option(help="Output candidate set for manual verification"),
    ] = DEFAULT_CANDIDATES_PATH,
    max_examples: Annotated[int, typer.Option(help="Maximum generated candidate rows")] = 200,
    profile: Annotated[
        str,
        typer.Option(help="Corpus profile for candidate generation"),
    ] = "full",
) -> None:
    """Generate hybrid auto+manual retrieval evaluation candidates."""

    configure_logging()
    settings = load_settings()
    settings.corpus_profile = "quickstart" if profile == "quickstart" else "full"
    with _command_guard("generate-eval-set", settings):
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
    include_unverified: Annotated[
        bool,
        typer.Option(
            help=(
                "Treat unverified candidate rows as verified. "
                "Useful for automated local pipeline checks."
            )
        ),
    ] = False,
) -> None:
    """Compile verified candidate rows into retrieval evaluation set."""

    configure_logging()
    settings = load_settings()
    with _command_guard("compile-eval-set", settings):
        rows = read_jsonl(input_path)
        verified_count = sum(bool(row.get("verified", False)) for row in rows)
        if include_unverified and verified_count == 0:
            logger.warning(
                "No manually verified rows found; compiling with unverified rows for automation."
            )
        candidates = [
            CandidateQA(
                query=row["query"],
                answer_hint=row["answer_hint"],
                doc_id=row["doc_id"],
                source_path=row.get("source_path", "unknown"),
                verified=bool(row.get("verified", False) or include_unverified),
            )
            for row in rows
        ]
        examples = to_eval_examples(candidates)
        if not examples:
            raise ValueError(
                "No verified evaluation rows available. "
                "Mark rows with `verified=true` or rerun with `--include-unverified`."
            )

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
    with _command_guard("judge", settings):
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
    with _command_guard("judge-batch", settings):
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
    with _command_guard("failures", settings):
        runtime = AppRuntime(settings)

        results = run_failure_case_analysis(runtime.pipeline)
        out_path = settings.reports_dir / f"failure_cases_{_timestamp()}.json"
        json_dump(out_path, {"rows": [asdict(row) for row in results]})
        logger.info("Saved failure analysis: {}", out_path)


@cli.command("run-experiments")
def run_experiments(
    eval_path: Annotated[Path, typer.Option(help="Evaluation JSONL path")] = DEFAULT_EVAL_PATH,
    chunk_sizes: Annotated[
        str | None,
        typer.Option(
            help=(
                "Optional comma-separated chunk sizes override, "
                "for example: 512,768,1024."
            )
        ),
    ] = None,
    profile: Annotated[
        str,
        typer.Option(help="Corpus profile for experiments: full or quickstart"),
    ] = "full",
) -> None:
    """Run chunk-size and top-k retrieval experiments and save report."""

    configure_logging()
    settings = load_settings()
    settings.corpus_profile = "quickstart" if profile == "quickstart" else "full"
    with _command_guard("run-experiments", settings):
        exp_settings = AppSettings(**settings.model_dump())
        exp_settings.vector_db_path = Path("vectordb/chroma_experiments")
        exp_settings.index_manifest_path = Path("vectordb/index_manifest_experiments.json")
        exp_settings.collection_name = f"{settings.collection_name}_experiments"
        runtime = AppRuntime(exp_settings)
        examples = _load_eval_examples(eval_path)
        selected_chunk_sizes = settings.chunking.chunk_sizes
        if chunk_sizes:
            selected_chunk_sizes = [
                int(token.strip()) for token in chunk_sizes.split(",") if token.strip()
            ]
            if not selected_chunk_sizes:
                raise ValueError("chunk_sizes override must include at least one integer.")
            if any(value <= 0 for value in selected_chunk_sizes):
                raise ValueError("chunk_sizes override values must be positive integers.")

        rows: list[dict[str, float | int]] = []
        evaluator = RetrievalEvaluator(runtime.retriever)

        for chunk_size in selected_chunk_sizes:
            report = runtime.indexer.build_or_update(
                chunk_size=chunk_size,
                chunk_overlap=settings.chunking.default_chunk_overlap,
                force_rebuild=True,
            )
            metrics = evaluator.evaluate(examples, ks=tuple(settings.retrieval.candidate_ks))
            for metric in metrics:
                rows.append(
                    {
                        "corpus_profile": settings.corpus_profile,
                        "chunk_size": chunk_size,
                        "chunk_overlap": settings.chunking.default_chunk_overlap,
                        "k": metric.k,
                        "precision_at_k": metric.precision_at_k,
                        "recall_at_k": metric.recall_at_k,
                        "mrr": metric.mrr,
                        "ndcg": metric.ndcg,
                        "avg_retrieval_latency_ms": metric.avg_retrieval_latency_ms,
                        "vector_count": report.vector_count,
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
    with _command_guard("diagram", settings):
        from local_rag.visualization import save_pipeline_architecture_diagram

        path = settings.diagrams_dir / "pipeline_architecture.png"
        save_pipeline_architecture_diagram(path)
        logger.info("Saved diagram: {}", path)


def main() -> None:
    """Run CLI."""

    cli()


if __name__ == "__main__":
    main()
