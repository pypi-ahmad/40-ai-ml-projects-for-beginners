"""CLI entrypoint for full local RAG workflow with audit-friendly controls."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from rag_system import ChunkingStrategy, ProjectRunner, RAGConfig
from rag_system.diagnostics import embedding_integrity_report, index_integrity_report
from rag_system.utils import assert_ollama_available

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full local RAG project pipeline")
    parser.add_argument("--profile", choices=["fast", "balanced", "max_depth"], default="balanced")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--strict-audit", action="store_true")
    parser.add_argument("--reuse-artifacts", action="store_true", default=True)
    parser.add_argument("--no-reuse-artifacts", dest="reuse_artifacts", action="store_false")
    return parser.parse_args()


def _assert_strict_audit(result: dict) -> None:
    leakage = result.get("leakage_audit", {})
    if not leakage.get("leakage_pass", False):
        raise RuntimeError(f"Leakage audit failed: {leakage}")

    index_integrity = result.get("index_integrity", {})
    if not index_integrity.get("count_matches_expected", False):
        raise RuntimeError(f"Index integrity failed: {index_integrity}")


def main() -> None:
    args = parse_args()
    cfg = RAGConfig(
        project_root=Path(args.project_root),
        profile=args.profile,
        top_k=args.top_k,
        enable_judge_eval=not args.skip_judge,
        reuse_processed_artifacts=args.reuse_artifacts,
    )
    assert_ollama_available(cfg.ollama_host)
    runner = ProjectRunner(cfg)

    logger.info("Ingesting dataset artifacts")
    ingestion = runner.ingest_dataset(force_rebuild=args.rebuild_data)
    documents = ingestion["documents"]
    if runner.config.profile != "max_depth":
        documents = documents[: runner.config.chunking_benchmark_docs]
        logger.info("Using %d documents for %s profile run", len(documents), runner.config.profile)

    needs_reindex = args.rebuild_index or runner.retrieval_engine.collection.count() == 0
    if needs_reindex:
        logger.info("Chunking and indexing documents")
        chunking = runner.run_chunking(documents=documents, strategy=ChunkingStrategy.RECURSIVE)
        indexed = runner.index_chunks(chunking.chunks, reset=True)
        chunk_count = len(chunking.chunks)
        embedding_integrity = embedding_integrity_report(
            embedding_engine=runner.embedding_engine,
            texts=[chunk.text for chunk in chunking.chunks[:128]],
            batch_size=16,
        )
        embedding_integrity["status"] = "computed"
        index_integrity = index_integrity_report(
            retrieval_engine=runner.retrieval_engine,
            expected_chunks=chunking.chunks,
        )
        index_integrity["status"] = "computed"
    else:
        logger.info("Reusing existing index; pass --rebuild-index to force reindex")
        indexed = runner.retrieval_engine.collection.count()
        chunk_count = indexed
        embedding_integrity = {"status": "skipped_reused_index"}
        index_integrity = {"status": "skipped_reused_index", "collection_count": indexed}

    result = {
        "documents_total": len(ingestion["documents"]),
        "documents_used": len(documents),
        "queries": len(ingestion["queries"]),
        "chunks": chunk_count,
        "indexed": indexed,
        "leakage_audit": ingestion.get("leakage_audit", {}),
        "embedding_integrity": embedding_integrity,
        "index_integrity": index_integrity,
    }

    if not args.skip_eval:
        logger.info("Running evaluation suite")
        eval_out = runner.run_evaluation(queries=ingestion["queries"])
        bundle = eval_out["bundle"]
        result["evaluation"] = {
            "retrieval": asdict(bundle.retrieval),
            "generation": asdict(bundle.generation),
            "judge": asdict(bundle.judge),
            "summary_path": str(eval_out["summary_path"]),
        }

    if args.strict_audit:
        _assert_strict_audit(result)

    summary_path = runner.config.artifacts_dir / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.info("Run complete. Summary saved to %s", summary_path)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
