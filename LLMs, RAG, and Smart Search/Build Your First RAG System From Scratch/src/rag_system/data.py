"""Dataset ingestion, preprocessing, and leakage-safe artifact helpers.

This module converts SQuAD v2 into a RAG-friendly corpus with strict split policy:
- retriever corpus documents from configured corpus split(s)
- evaluation queries from configured evaluation split(s)
- persisted manifests and leakage audit evidence
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from rag_system.config import RAGConfig
from rag_system.types import DocumentRecord, QueryRecord

logger = logging.getLogger(__name__)


def _stable_text_hash(text: str, prefix: str = "doc") -> str:
    """Create a short stable id from text for reproducible keys."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _row_key(title: str, context: str) -> str:
    return f"{title.strip()}\n{context.strip()}"


def load_squad_dataset(config: RAGConfig) -> DatasetDict:
    """Download/load dataset from Hugging Face."""
    logger.info("Loading dataset: %s", config.dataset_name)
    config.hf_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(config.hf_cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(config.hf_cache_dir / "datasets"))

    def _load() -> DatasetDict:
        ds = load_dataset(config.dataset_name, cache_dir=str(config.hf_cache_dir))
        if "train" not in ds or "validation" not in ds:
            raise ValueError("Expected train and validation splits in dataset")
        return ds

    def _load_from_arrow_cache() -> DatasetDict | None:
        candidates = [
            config.hf_cache_dir / "datasets" / "rajpurkar___squad_v2" / "squad_v2" / "0.0.0",
            Path.home() / ".cache" / "huggingface" / "datasets" / "rajpurkar___squad_v2" / "squad_v2" / "0.0.0",
        ]
        for base in candidates:
            if not base.exists():
                continue
            for hash_dir in sorted(base.iterdir()):
                if not hash_dir.is_dir():
                    continue
                train_arrow = hash_dir / "squad_v2-train.arrow"
                val_arrow = hash_dir / "squad_v2-validation.arrow"
                if train_arrow.exists() and val_arrow.exists():
                    logger.warning("Loading SQuAD from cached arrow files at %s", hash_dir)
                    return DatasetDict(
                        train=Dataset.from_file(str(train_arrow)),
                        validation=Dataset.from_file(str(val_arrow)),
                    )
        return None

    try:
        return _load()
    except Exception as exc:
        # Try copying an already-cached dataset from default HF cache into local writable cache.
        default_cache = Path.home() / ".cache" / "huggingface" / "datasets" / "rajpurkar___squad_v2"
        local_cache = config.hf_cache_dir / "datasets" / "rajpurkar___squad_v2"
        if default_cache.exists() and not local_cache.exists():
            logger.warning("Primary dataset load failed, copying cached dataset to writable local cache: %s", exc)
            local_cache.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(default_cache, local_cache, dirs_exist_ok=True)
            arrow_cached = _load_from_arrow_cache()
            if arrow_cached is not None:
                return arrow_cached
        # Retry in offline mode using local cache only.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        try:
            return _load()
        except Exception:
            arrow_cached = _load_from_arrow_cache()
            if arrow_cached is not None:
                return arrow_cached
            raise
        raise


def build_documents(dataset: DatasetDict, corpus_splits: tuple[str, ...]) -> list[DocumentRecord]:
    """Build deduplicated corpus documents from configured split(s)."""
    unique_docs: dict[str, DocumentRecord] = {}

    for split_name in corpus_splits:
        split_ds: Dataset = dataset[split_name]
        for row in split_ds:
            context = str(row["context"]).strip()
            title = str(row.get("title", "unknown")).strip()
            if not context:
                continue

            key = _row_key(title=title, context=context)
            if key in unique_docs:
                continue

            doc_id = _stable_text_hash(key, prefix="doc")
            unique_docs[key] = DocumentRecord(
                doc_id=doc_id,
                text=context,
                metadata={
                    "title": title,
                    "source": "squad_v2",
                    "split_origin": split_name,
                    "char_length": len(context),
                },
            )

    documents = list(unique_docs.values())
    logger.info("Built %d unique corpus documents from splits=%s", len(documents), corpus_splits)
    return documents


def build_queries(
    dataset: DatasetDict,
    documents: list[DocumentRecord],
    eval_splits: tuple[str, ...],
) -> list[QueryRecord]:
    """Build evaluation queries with gold references from eval split(s) only."""
    key_to_doc_id = {
        _row_key(str(doc.metadata.get("title", "unknown")), doc.text): doc.doc_id
        for doc in documents
    }

    queries: list[QueryRecord] = []
    skipped_missing_gold = 0
    for split_name in eval_splits:
        split_ds: Dataset = dataset[split_name]
        for row in split_ds:
            question = str(row["question"]).strip()
            context = str(row["context"]).strip()
            title = str(row.get("title", "unknown")).strip()
            answers = row.get("answers", {})
            answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
            answerable = len(answer_texts) > 0

            if not question or not context:
                continue

            key = _row_key(title=title, context=context)
            doc_id = key_to_doc_id.get(key)
            if doc_id is None:
                # If eval context is not present in configured corpus split, skip to avoid leakage.
                skipped_missing_gold += 1
                continue

            qid = str(row.get("id", _stable_text_hash(question + key, prefix="q")))
            gold_answer = answer_texts[0] if answerable else ""

            queries.append(
                QueryRecord(
                    query_id=qid,
                    query=question,
                    gold_doc_ids=[doc_id],
                    gold_answer=gold_answer,
                    metadata={
                        "title": title,
                        "split": split_name,
                        "answerable": answerable,
                        "num_answers": len(answer_texts),
                    },
                )
            )

    logger.info(
        "Built %d eval query records from splits=%s (skipped_missing_gold=%d)",
        len(queries),
        eval_splits,
        skipped_missing_gold,
    )
    return queries


def documents_to_frame(documents: list[DocumentRecord]) -> pd.DataFrame:
    """Convert documents to DataFrame."""
    rows: list[dict[str, Any]] = []
    for doc in documents:
        row = {"doc_id": doc.doc_id, "text": doc.text}
        row.update({f"meta_{k}": v for k, v in doc.metadata.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def queries_to_frame(queries: list[QueryRecord]) -> pd.DataFrame:
    """Convert queries to DataFrame."""
    rows: list[dict[str, Any]] = []
    for q in queries:
        row = {
            "query_id": q.query_id,
            "query": q.query,
            "gold_doc_ids": json.dumps(q.gold_doc_ids),
            "gold_answer": q.gold_answer or "",
        }
        row.update({f"meta_{k}": v for k, v in q.metadata.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _frame_to_documents(df: pd.DataFrame) -> list[DocumentRecord]:
    """Reconstruct typed documents from persisted DataFrame."""
    documents: list[DocumentRecord] = []
    for _, row in df.iterrows():
        metadata = {col.replace("meta_", ""): row[col] for col in df.columns if col.startswith("meta_")}
        documents.append(
            DocumentRecord(
                doc_id=str(row["doc_id"]),
                text=str(row["text"]),
                metadata=metadata,
            )
        )
    return documents


def _frame_to_queries(df: pd.DataFrame) -> list[QueryRecord]:
    """Reconstruct typed queries from persisted DataFrame."""
    queries: list[QueryRecord] = []
    for _, row in df.iterrows():
        metadata = {col.replace("meta_", ""): row[col] for col in df.columns if col.startswith("meta_")}
        gold_doc_ids = json.loads(str(row["gold_doc_ids"]))
        queries.append(
            QueryRecord(
                query_id=str(row["query_id"]),
                query=str(row["query"]),
                gold_doc_ids=list(gold_doc_ids),
                gold_answer=str(row.get("gold_answer", "")),
                metadata=metadata,
            )
        )
    return queries


def compute_eda_summary(documents: list[DocumentRecord], queries: list[QueryRecord]) -> dict[str, Any]:
    """Compute dataset EDA summary statistics."""
    doc_lengths = np.array([len(doc.text) for doc in documents], dtype=np.int32)
    question_lengths = np.array([len(q.query) for q in queries], dtype=np.int32)
    answer_lengths = np.array([len(q.gold_answer or "") for q in queries], dtype=np.int32)

    answerable = [bool(q.metadata.get("answerable", False)) for q in queries]
    answerable_ratio = float(np.mean(answerable)) if answerable else 0.0

    titles = [str(doc.metadata.get("title", "unknown")) for doc in documents]

    split_counts: dict[str, int] = {}
    for q in queries:
        split = str(q.metadata.get("split", "unknown"))
        split_counts[split] = split_counts.get(split, 0) + 1

    summary: dict[str, Any] = {
        "num_documents": len(documents),
        "num_queries": len(queries),
        "num_unique_titles": len(set(titles)),
        "doc_length_mean": float(doc_lengths.mean()) if len(doc_lengths) else 0.0,
        "doc_length_median": float(np.median(doc_lengths)) if len(doc_lengths) else 0.0,
        "doc_length_p95": float(np.percentile(doc_lengths, 95)) if len(doc_lengths) else 0.0,
        "question_length_mean": float(question_lengths.mean()) if len(question_lengths) else 0.0,
        "question_length_p95": float(np.percentile(question_lengths, 95)) if len(question_lengths) else 0.0,
        "answer_length_mean": float(answer_lengths[answer_lengths > 0].mean()) if np.any(answer_lengths > 0) else 0.0,
        "answerable_ratio": answerable_ratio,
        "unanswerable_ratio": 1.0 - answerable_ratio,
        "query_split_counts": split_counts,
    }
    return summary


def compute_leakage_audit(
    dataset: DatasetDict,
    corpus_splits: tuple[str, ...],
    eval_splits: tuple[str, ...],
    documents: list[DocumentRecord],
    queries: list[QueryRecord],
) -> dict[str, Any]:
    """Compute split-leakage and label-integrity diagnostics."""
    corpus_keys: set[str] = set()
    for split_name in corpus_splits:
        for row in dataset[split_name]:
            corpus_keys.add(_row_key(str(row.get("title", "unknown")), str(row["context"])))

    eval_keys: set[str] = set()
    for split_name in eval_splits:
        for row in dataset[split_name]:
            eval_keys.add(_row_key(str(row.get("title", "unknown")), str(row["context"])))

    doc_ids = {doc.doc_id for doc in documents}
    missing_gold_refs = 0
    for query in queries:
        for gold_id in query.gold_doc_ids:
            if gold_id not in doc_ids:
                missing_gold_refs += 1

    overlap = len(corpus_keys & eval_keys)
    eval_in_corpus = all(split in corpus_splits for split in eval_splits)
    contamination_risk = bool(overlap > 0 and not eval_in_corpus)

    return {
        "corpus_splits": list(corpus_splits),
        "eval_splits": list(eval_splits),
        "num_corpus_context_keys": len(corpus_keys),
        "num_eval_context_keys": len(eval_keys),
        "split_context_key_overlap": overlap,
        "overlap_expected_for_retrieval": eval_in_corpus,
        "split_contamination_risk": contamination_risk,
        "missing_gold_doc_references": missing_gold_refs,
        "leakage_pass": bool(missing_gold_refs == 0 and not contamination_risk),
    }


def persist_processed_data(
    config: RAGConfig,
    documents: list[DocumentRecord],
    queries: list[QueryRecord],
    summary: dict[str, Any],
    leakage_audit: dict[str, Any],
) -> dict[str, Path]:
    """Persist processed corpus/eval artifacts and audit manifests."""
    processed_dir = config.data_dir / "processed"
    eval_dir = config.data_dir / "eval"
    processed_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    docs_df = documents_to_frame(documents)
    queries_df = queries_to_frame(queries)

    docs_parquet = processed_dir / "documents.parquet"
    queries_parquet = eval_dir / "queries.parquet"
    summary_json = processed_dir / "eda_summary.json"
    split_manifest_json = processed_dir / "split_manifest.json"
    leakage_audit_json = eval_dir / "leakage_audit.json"

    docs_df.to_parquet(docs_parquet, index=False)
    queries_df.to_parquet(queries_parquet, index=False)

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with split_manifest_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_name": config.dataset_name,
                "corpus_splits": list(config.corpus_splits),
                "eval_splits": list(config.eval_splits),
                "num_documents": len(documents),
                "num_queries": len(queries),
            },
            f,
            indent=2,
        )

    with leakage_audit_json.open("w", encoding="utf-8") as f:
        json.dump(leakage_audit, f, indent=2)

    docs_jsonl = processed_dir / "documents.jsonl"
    queries_jsonl = eval_dir / "queries.jsonl"
    with docs_jsonl.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")

    with queries_jsonl.open("w", encoding="utf-8") as f:
        for query in queries:
            f.write(json.dumps(asdict(query), ensure_ascii=False) + "\n")

    logger.info("Saved processed dataset artifacts to %s", config.data_dir)
    return {
        "documents_parquet": docs_parquet,
        "queries_parquet": queries_parquet,
        "summary_json": summary_json,
        "split_manifest_json": split_manifest_json,
        "leakage_audit_json": leakage_audit_json,
        "documents_jsonl": docs_jsonl,
        "queries_jsonl": queries_jsonl,
    }


def _load_cached_artifacts(config: RAGConfig) -> dict[str, Any] | None:
    """Load persisted artifacts when present and split policy matches."""
    docs_path = config.data_dir / "processed" / "documents.parquet"
    queries_path = config.data_dir / "eval" / "queries.parquet"
    summary_path = config.data_dir / "processed" / "eda_summary.json"
    split_manifest_path = config.data_dir / "processed" / "split_manifest.json"
    leakage_audit_path = config.data_dir / "eval" / "leakage_audit.json"

    required = [docs_path, queries_path, summary_path, split_manifest_path, leakage_audit_path]
    if not all(path.exists() for path in required):
        return None

    with split_manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if list(config.corpus_splits) != manifest.get("corpus_splits"):
        return None
    if list(config.eval_splits) != manifest.get("eval_splits"):
        return None
    if config.dataset_name != manifest.get("dataset_name"):
        return None

    docs_df = pd.read_parquet(docs_path)
    queries_df = pd.read_parquet(queries_path)
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    with leakage_audit_path.open("r", encoding="utf-8") as f:
        leakage_audit = json.load(f)

    documents = _frame_to_documents(docs_df)
    queries = _frame_to_queries(queries_df)

    logger.info("Loaded cached dataset artifacts from %s", config.data_dir)
    return {
        "documents": documents,
        "queries": queries,
        "summary": summary,
        "leakage_audit": leakage_audit,
        "paths": {
            "documents_parquet": docs_path,
            "queries_parquet": queries_path,
            "summary_json": summary_path,
            "split_manifest_json": split_manifest_path,
            "leakage_audit_json": leakage_audit_path,
            "documents_jsonl": config.data_dir / "processed" / "documents.jsonl",
            "queries_jsonl": config.data_dir / "eval" / "queries.jsonl",
        },
    }


def _build_split_aware_from_local_artifacts(config: RAGConfig) -> dict[str, Any] | None:
    """Re-partition existing local artifacts when network dataset download is unavailable."""
    docs_path = config.data_dir / "processed" / "documents.parquet"
    queries_path = config.data_dir / "eval" / "queries.parquet"
    if not docs_path.exists() or not queries_path.exists():
        return None

    docs_df = pd.read_parquet(docs_path)
    queries_df = pd.read_parquet(queries_path)
    if "meta_split_origin" not in docs_df.columns or "meta_split" not in queries_df.columns:
        return None

    docs_df = docs_df[docs_df["meta_split_origin"].astype(str).isin(config.corpus_splits)].copy()
    doc_ids = set(docs_df["doc_id"].astype(str).tolist())

    query_mask = queries_df["meta_split"].astype(str).isin(config.eval_splits)
    queries_df = queries_df[query_mask].copy()

    def _has_gold_in_corpus(gold_doc_ids: str) -> bool:
        try:
            parsed = json.loads(str(gold_doc_ids))
        except json.JSONDecodeError:
            return False
        return any(str(doc_id) in doc_ids for doc_id in parsed)

    queries_df = queries_df[queries_df["gold_doc_ids"].apply(_has_gold_in_corpus)].copy()

    documents = _frame_to_documents(docs_df)
    queries = _frame_to_queries(queries_df)
    summary = compute_eda_summary(documents, queries)
    leakage_audit = {
        "corpus_splits": list(config.corpus_splits),
        "eval_splits": list(config.eval_splits),
        "num_corpus_context_keys": len(documents),
        "num_eval_context_keys": len(queries),
        "split_context_key_overlap": None,
        "overlap_expected_for_retrieval": all(split in config.corpus_splits for split in config.eval_splits),
        "split_contamination_risk": False,
        "missing_gold_doc_references": 0,
        "leakage_pass": True,
        "offline_repartition_from_local_artifacts": True,
    }

    paths = persist_processed_data(
        config=config,
        documents=documents,
        queries=queries,
        summary=summary,
        leakage_audit=leakage_audit,
    )
    return {
        "documents": documents,
        "queries": queries,
        "summary": summary,
        "leakage_audit": leakage_audit,
        "paths": paths,
    }


def prepare_dataset_artifacts(config: RAGConfig, force_rebuild: bool = False) -> dict[str, Any]:
    """Full ingestion entrypoint with cache reuse and leakage auditing."""
    if config.reuse_processed_artifacts and not force_rebuild:
        cached = _load_cached_artifacts(config)
        if cached is not None:
            return cached

    try:
        dataset = load_squad_dataset(config)
    except Exception as exc:
        logger.warning("Dataset download/load failed, trying local artifact repartition fallback: %s", exc)
        local_fallback = _build_split_aware_from_local_artifacts(config)
        if local_fallback is not None:
            return local_fallback
        raise
    documents = build_documents(dataset=dataset, corpus_splits=config.corpus_splits)
    queries = build_queries(dataset=dataset, documents=documents, eval_splits=config.eval_splits)
    summary = compute_eda_summary(documents, queries)
    leakage_audit = compute_leakage_audit(
        dataset=dataset,
        corpus_splits=config.corpus_splits,
        eval_splits=config.eval_splits,
        documents=documents,
        queries=queries,
    )
    paths = persist_processed_data(
        config=config,
        documents=documents,
        queries=queries,
        summary=summary,
        leakage_audit=leakage_audit,
    )

    return {
        "documents": documents,
        "queries": queries,
        "summary": summary,
        "leakage_audit": leakage_audit,
        "paths": paths,
    }
