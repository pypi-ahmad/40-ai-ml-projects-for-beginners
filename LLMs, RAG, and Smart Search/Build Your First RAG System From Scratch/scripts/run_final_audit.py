"""Run end-to-end real RAG audit and persist publication-ready artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from rag_system import ChunkingStrategy, ProjectRunner, RAGConfig
from rag_system.diagnostics import embedding_integrity_report, index_integrity_report, retrieval_diagnostics
from rag_system.types import DocumentRecord, QueryRecord
from rag_system.utils import assert_ollama_available

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full verification/audit suite (real runs)")
    parser.add_argument("--profile", choices=["fast", "balanced", "max_depth"], default="max_depth")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--chunking-docs", type=int, default=None)
    parser.add_argument("--chunking-queries", type=int, default=None)
    parser.add_argument("--advanced-queries", type=int, default=None)
    parser.add_argument("--retrieval-limit", type=int, default=None)
    parser.add_argument("--generation-limit", type=int, default=None)
    parser.add_argument("--judge-limit", type=int, default=None)
    parser.add_argument("--hallucination-limit", type=int, default=None)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--output", default="data/artifacts/final_audit/report.json")
    parser.add_argument("--strict-gates", action="store_true", default=True)
    parser.add_argument("--no-strict-gates", dest="strict_gates", action="store_false")
    return parser.parse_args()


def _first_hit_rank(doc_ids: list[str], gold_doc_ids: list[str]) -> int | None:
    gold = set(gold_doc_ids)
    for idx, doc_id in enumerate(doc_ids, start=1):
        if doc_id in gold:
            return idx
    return None


def _hit_at_k(doc_ids: list[str], gold_doc_ids: list[str]) -> int:
    return int(_first_hit_rank(doc_ids, gold_doc_ids) is not None)


def _filter_queries_for_indexed_docs(
    queries: list[QueryRecord],
    indexed_doc_ids: set[str],
    limit: int,
) -> list[QueryRecord]:
    eligible = [query for query in queries if any(doc_id in indexed_doc_ids for doc_id in query.gold_doc_ids)]
    return eligible[:limit]


def _select_docs_and_queries_for_budget(
    documents: list[DocumentRecord],
    queries: list[QueryRecord],
    doc_limit: int,
    query_limit: int,
) -> tuple[list[DocumentRecord], list[QueryRecord]]:
    """Build a doc/query subset where each selected query is answerable from selected docs."""
    doc_map = {doc.doc_id: doc for doc in documents}
    selected_docs: dict[str, DocumentRecord] = {}
    selected_queries: list[QueryRecord] = []

    for query in queries:
        gold_ids = [doc_id for doc_id in query.gold_doc_ids if doc_id in doc_map]
        if not gold_ids:
            continue
        gold_doc_id = gold_ids[0]

        if gold_doc_id not in selected_docs and len(selected_docs) >= doc_limit:
            continue

        selected_docs.setdefault(gold_doc_id, doc_map[gold_doc_id])
        selected_queries.append(query)
        if len(selected_queries) >= query_limit:
            break

    return list(selected_docs.values()), selected_queries


def _assert_publication_gates(report: dict[str, Any], required_output_files: list[Path]) -> None:
    leakage = report.get("leakage_audit", {})
    if not leakage.get("leakage_pass", False):
        raise RuntimeError(f"Publication gate failed: leakage audit did not pass: {leakage}")

    index_integrity = report.get("index_integrity", {})
    if not index_integrity.get("count_matches_expected", False):
        raise RuntimeError(f"Publication gate failed: index integrity mismatch: {index_integrity}")

    diagnostics_count = int(report.get("retrieval_diagnostics_count", 0))
    if diagnostics_count <= 0:
        raise RuntimeError("Publication gate failed: retrieval diagnostics table is empty")

    missing_outputs = [str(path) for path in required_output_files if not path.exists()]
    if missing_outputs:
        raise RuntimeError(f"Publication gate failed: missing required output files: {missing_outputs}")


def validate_report_payload(report: dict[str, Any]) -> None:
    """Validate required report keys for downstream tooling."""
    required_keys = {
        "run_type",
        "timestamp_utc",
        "profile_used",
        "effective_eval_limits",
        "models_used",
        "retrieval_summary",
        "generation_summary",
        "judge_summary",
        "leakage_audit",
        "index_integrity",
        "embedding_integrity",
        "required_outputs",
    }
    missing = sorted(required_keys - set(report.keys()))
    if missing:
        raise RuntimeError(f"Report payload missing required keys: {missing}")


def main() -> None:
    args = parse_args()
    cfg = RAGConfig(
        project_root=Path(args.project_root),
        profile=args.profile,
        top_k=args.top_k,
        enable_judge_eval=not args.skip_judge,
    )
    assert_ollama_available(cfg.ollama_host)
    runner = ProjectRunner(cfg)

    t0 = time.perf_counter()
    bundle = runner.ingest_dataset(force_rebuild=args.rebuild_data)
    ingestion_s = time.perf_counter() - t0

    documents = bundle["documents"]
    queries = bundle["queries"]
    sampled_queries = list(queries)
    random.Random(runner.config.sampling_seed).shuffle(sampled_queries)

    chunking_docs_limit = min(args.chunking_docs or runner.config.chunking_benchmark_docs, len(documents))
    chunking_queries_limit = min(args.chunking_queries or runner.config.retrieval_eval_queries, len(queries))
    advanced_queries_limit = min(args.advanced_queries or runner.config.generation_eval_queries, len(queries))
    retrieval_limit = min(args.retrieval_limit or runner.config.retrieval_eval_queries, len(queries))
    generation_limit = min(args.generation_limit or runner.config.generation_eval_queries, len(queries))
    judge_limit = min(args.judge_limit or runner.config.judge_eval_queries, len(queries))
    hallucination_limit = min(
        args.hallucination_limit or min(judge_limit, 300),
        len(queries),
    )

    chunking_docs, chunking_queries = _select_docs_and_queries_for_budget(
        documents=documents,
        queries=sampled_queries,
        doc_limit=chunking_docs_limit,
        query_limit=chunking_queries_limit,
    )
    if not chunking_queries:
        raise RuntimeError("No eligible chunking benchmark queries map to the indexed document subset")

    chunking_rows: list[dict[str, Any]] = []
    for strategy in [
        ChunkingStrategy.FIXED,
        ChunkingStrategy.RECURSIVE,
        ChunkingStrategy.SEMANTIC,
        ChunkingStrategy.PARENT_CHILD,
    ]:
        t_chunk = time.perf_counter()
        chunking = runner.run_chunking(documents=chunking_docs, strategy=strategy)
        chunk_time = time.perf_counter() - t_chunk

        t_index = time.perf_counter()
        runner.index_chunks(chunking.chunks, reset=True)
        index_time = time.perf_counter() - t_index

        retrieval_summary, _, _ = runner.evaluator.evaluate_retrieval(
            queries=chunking_queries,
            top_k=args.top_k,
            max_queries=len(chunking_queries),
        )

        chunking_rows.append(
            {
                "strategy": strategy.value,
                "num_chunks": len(chunking.chunks),
                "avg_chunk_length": chunking.avg_chunk_length,
                "chunking_time_s": chunk_time,
                "indexing_time_s": index_time,
                "num_eval_queries": len(chunking_queries),
                **asdict(retrieval_summary),
            }
        )

    chunking_df = pd.DataFrame(chunking_rows)
    best_strategy = str(chunking_df.sort_values("mrr", ascending=False).iloc[0]["strategy"])

    strategy_map = {
        "fixed": ChunkingStrategy.FIXED,
        "recursive": ChunkingStrategy.RECURSIVE,
        "semantic": ChunkingStrategy.SEMANTIC,
        "parent_child": ChunkingStrategy.PARENT_CHILD,
    }

    evaluation_queries_limit = max(retrieval_limit, generation_limit, judge_limit, hallucination_limit, advanced_queries_limit)
    final_docs = documents
    evaluation_queries: list[QueryRecord]
    if runner.config.profile != "max_depth":
        final_docs, evaluation_queries = _select_docs_and_queries_for_budget(
            documents=documents,
            queries=sampled_queries,
            doc_limit=runner.config.chunking_benchmark_docs,
            query_limit=evaluation_queries_limit,
        )
    else:
        final_docs = documents
        final_doc_ids = {doc.doc_id for doc in final_docs}
        evaluation_queries = _filter_queries_for_indexed_docs(
            queries=sampled_queries,
            indexed_doc_ids=final_doc_ids,
            limit=evaluation_queries_limit,
        )
    if not evaluation_queries:
        raise RuntimeError("No eligible evaluation queries map to the final indexed document subset")

    t_chunk_final = time.perf_counter()
    final_chunking = runner.run_chunking(final_docs, strategy=strategy_map[best_strategy])
    final_chunking_time_s = time.perf_counter() - t_chunk_final

    t_index_final = time.perf_counter()
    runner.index_chunks(final_chunking.chunks, reset=True)
    final_indexing_time_s = time.perf_counter() - t_index_final

    embedding_report = embedding_integrity_report(
        embedding_engine=runner.embedding_engine,
        texts=[chunk.text for chunk in final_chunking.chunks[:128]],
        batch_size=16,
    )
    index_report = index_integrity_report(
        retrieval_engine=runner.retrieval_engine,
        expected_chunks=final_chunking.chunks,
    )

    t_eval = time.perf_counter()
    prior_retrieval_limit = runner.config.retrieval_eval_queries
    prior_generation_limit = runner.config.generation_eval_queries
    prior_judge_limit = runner.config.judge_eval_queries
    try:
        runner.config.retrieval_eval_queries = retrieval_limit
        runner.config.generation_eval_queries = generation_limit
        runner.config.judge_eval_queries = judge_limit
        evaluation = runner.run_evaluation(queries=evaluation_queries, hallucination_limit=hallucination_limit)
    finally:
        runner.config.retrieval_eval_queries = prior_retrieval_limit
        runner.config.generation_eval_queries = prior_generation_limit
        runner.config.judge_eval_queries = prior_judge_limit
    evaluation_time_s = time.perf_counter() - t_eval

    # Fair advanced retrieval comparison on same query set and same K.
    advanced_subset = evaluation_queries[:advanced_queries_limit]
    advanced_rows: list[dict[str, Any]] = []
    for query_record in advanced_subset:
        base_hits = runner.retrieval_engine.query(query_record.query, top_k=args.top_k, dedupe_by_doc=True)
        adv_out = runner.advanced_retriever.retrieve(query_record.query, top_k=args.top_k)

        base_doc_ids = [hit.doc_id for hit in base_hits]
        adv_doc_ids = [hit.doc_id for hit in adv_out.chunks]

        base_rank = _first_hit_rank(base_doc_ids, query_record.gold_doc_ids)
        adv_rank = _first_hit_rank(adv_doc_ids, query_record.gold_doc_ids)

        advanced_rows.append(
            {
                "query_id": query_record.query_id,
                "base_hit": _hit_at_k(base_doc_ids, query_record.gold_doc_ids),
                "advanced_hit": _hit_at_k(adv_doc_ids, query_record.gold_doc_ids),
                "base_rr": (1.0 / base_rank) if base_rank else 0.0,
                "advanced_rr": (1.0 / adv_rank) if adv_rank else 0.0,
                "base_top_score": base_hits[0].score if base_hits else 0.0,
                "advanced_top_hybrid": float(adv_out.chunks[0].metadata.get("hybrid_score", 0.0)) if adv_out.chunks else 0.0,
            }
        )
    advanced_df = pd.DataFrame(advanced_rows)

    retrieval_diag_rows = retrieval_diagnostics(
        retrieval_engine=runner.retrieval_engine,
        queries=evaluation_queries,
        top_k=args.top_k,
        min_relevance_score=runner.config.min_relevance_score,
        max_queries=min(runner.config.retrieval_eval_queries, len(evaluation_queries)),
    )
    retrieval_diag_df = pd.DataFrame(retrieval_diag_rows)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunking_csv = out_path.parent / "chunking_benchmark.csv"
    advanced_csv = out_path.parent / "advanced_retrieval.csv"
    diagnostics_csv = out_path.parent / "retrieval_diagnostics.csv"

    chunking_df.to_csv(chunking_csv, index=False)
    advanced_df.to_csv(advanced_csv, index=False)
    retrieval_diag_df.to_csv(diagnostics_csv, index=False)

    eval_bundle = evaluation["bundle"]

    advanced_summary = {
        "num_queries": int(len(advanced_df)),
        "base_hit_rate": float(advanced_df["base_hit"].mean()) if not advanced_df.empty else 0.0,
        "advanced_hit_rate": float(advanced_df["advanced_hit"].mean()) if not advanced_df.empty else 0.0,
        "base_mrr": float(advanced_df["base_rr"].mean()) if not advanced_df.empty else 0.0,
        "advanced_mrr": float(advanced_df["advanced_rr"].mean()) if not advanced_df.empty else 0.0,
    }

    report = {
        "run_type": "real",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "profile_used": runner.config.profile,
        "top_k": args.top_k,
        "models_used": {
            "embedding": runner.config.embedding_model,
            "generator": runner.config.generator_model,
            "judge": runner.config.judge_model,
        },
        "effective_eval_limits": {
            "chunking_docs_requested": chunking_docs_limit,
            "chunking_docs_used": len(chunking_docs),
            "chunking_queries_requested": chunking_queries_limit,
            "chunking_queries_used": len(chunking_queries),
            "retrieval_eval_queries_requested": retrieval_limit,
            "generation_eval_queries_requested": generation_limit,
            "judge_eval_queries_requested": judge_limit if runner.config.enable_judge_eval else 0,
            "hallucination_eval_queries_requested": hallucination_limit,
            "advanced_queries_requested": advanced_queries_limit,
            "evaluation_queries_pool_used": len(evaluation_queries),
            "final_index_docs_used": len(final_docs),
            "retrieval_eval_queries_used": int(eval_bundle.retrieval.num_queries),
            "generation_eval_queries_used": int(eval_bundle.generation.num_examples),
            "judge_eval_queries_used": int(eval_bundle.judge.num_examples),
            "advanced_queries_used": int(len(advanced_subset)),
        },
        "ingestion_time_s": ingestion_s,
        "final_chunking_time_s": final_chunking_time_s,
        "final_indexing_time_s": final_indexing_time_s,
        "evaluation_time_s": evaluation_time_s,
        "best_chunking_strategy": best_strategy,
        "chunking_benchmark": chunking_rows,
        "leakage_audit": bundle.get("leakage_audit", {}),
        "embedding_integrity": embedding_report,
        "index_integrity": index_report,
        "retrieval_summary": asdict(eval_bundle.retrieval),
        "generation_summary": asdict(eval_bundle.generation),
        "judge_summary": asdict(eval_bundle.judge),
        "advanced_retrieval": advanced_summary,
        "retrieval_diagnostics_count": int(len(retrieval_diag_df)),
        "retrieval_failure_buckets": retrieval_diag_df["failure_bucket"].value_counts(dropna=False).to_dict()
        if "failure_bucket" in retrieval_diag_df.columns
        else {},
        "artifact_summary_path": str(evaluation["summary_path"]),
        "visual_paths": {key: str(path) for key, path in evaluation["visual_paths"].items()},
        "required_outputs": [
            str(out_path),
            str(chunking_csv),
            str(advanced_csv),
            str(diagnostics_csv),
            str(Path(evaluation["summary_path"])),
        ],
    }

    validate_report_payload(report)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if args.strict_gates:
        required_paths = [Path(path) for path in report["required_outputs"]]
        _assert_publication_gates(report, required_paths)

    logger.info("Final audit report saved to %s", out_path)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
