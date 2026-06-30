"""Run a practical local smoke evaluation and persist real metrics/artifacts."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from rag_system import ChunkingStrategy, ProjectRunner, RAGConfig
from rag_system.utils import assert_ollama_available

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    cfg = RAGConfig(project_root=Path('.'), profile='balanced')
    assert_ollama_available(cfg.ollama_host)
    runner = ProjectRunner(cfg)

    logger.info('Loading and preprocessing dataset')
    bundle = runner.ingest_dataset()

    # Keep smoke runtime manageable while still using real pipeline.
    candidate_queries = bundle['queries'][:24]
    doc_by_id = {doc.doc_id: doc for doc in bundle['documents']}
    required_doc_ids = {doc_id for query in candidate_queries for doc_id in query.gold_doc_ids if doc_id in doc_by_id}

    docs = [doc_by_id[doc_id] for doc_id in sorted(required_doc_ids)]
    if len(docs) < 220:
        remaining = [doc for doc in bundle['documents'] if doc.doc_id not in required_doc_ids]
        docs.extend(remaining[: 220 - len(docs)])

    indexed_doc_ids = {doc.doc_id for doc in docs}
    queries = [query for query in candidate_queries if any(doc_id in indexed_doc_ids for doc_id in query.gold_doc_ids)]

    if not queries:
        raise RuntimeError(
            'Smoke query pool is empty after filtering by indexed document ids. '
            'Increase smoke document subset size.'
        )

    logger.info('Chunking + indexing')
    chunking = runner.run_chunking(docs, strategy=ChunkingStrategy.RECURSIVE)
    runner.index_chunks(chunking.chunks, reset=True)

    logger.info('Running retrieval/gen/judge evaluation')
    eval_bundle, frames = runner.evaluator.run_full_evaluation(
        queries=queries,
        top_k=6,
        retrieval_limit=20,
        generation_limit=4,
        judge_limit=4,
    )

    hallucination_df = runner.evaluator.evaluate_hallucination_reduction(queries=queries[:4], max_queries=4)
    frames['hallucination'] = hallucination_df

    out_dir = Path('data/artifacts/smoke')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'tables').mkdir(exist_ok=True)

    for name, frame in frames.items():
        frame.to_parquet(out_dir / 'tables' / f'{name}.parquet', index=False)
        frame.to_csv(out_dir / 'tables' / f'{name}.csv', index=False)

    summary = {
        'retrieval': asdict(eval_bundle.retrieval),
        'generation': asdict(eval_bundle.generation),
        'judge': asdict(eval_bundle.judge),
        'leakage_audit': bundle.get('leakage_audit', {}),
        'documents_indexed': len(docs),
        'chunks_indexed': len(chunking.chunks),
        'query_pool': len(queries),
    }

    with (out_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    logger.info('Smoke metrics saved to %s', out_dir / 'summary.json')


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
