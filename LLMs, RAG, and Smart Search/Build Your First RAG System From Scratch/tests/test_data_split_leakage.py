from datasets import Dataset, DatasetDict

from rag_system.data import build_documents, build_queries, compute_leakage_audit


def _toy_dataset() -> DatasetDict:
    train = Dataset.from_dict(
        {
            "id": ["t1", "t2"],
            "title": ["A", "B"],
            "context": ["alpha context", "beta context"],
            "question": ["q1", "q2"],
            "answers": [{"text": ["alpha"], "answer_start": [0]}, {"text": ["beta"], "answer_start": [0]}],
        }
    )
    validation = Dataset.from_dict(
        {
            "id": ["v1", "v2"],
            "title": ["A", "Z"],
            "context": ["alpha context", "zeta context"],
            "question": ["What is alpha?", "What is zeta?"],
            "answers": [{"text": ["alpha"], "answer_start": [0]}, {"text": ["zeta"], "answer_start": [0]}],
        }
    )
    return DatasetDict(train=train, validation=validation)


def test_split_aware_query_building_skips_missing_gold_docs() -> None:
    ds = _toy_dataset()
    docs = build_documents(ds, corpus_splits=("train",))
    queries = build_queries(ds, documents=docs, eval_splits=("validation",))

    # v1 maps to train context, v2 should be skipped because its context is not in corpus split.
    assert len(queries) == 1
    assert queries[0].query_id == "v1"


def test_leakage_audit_reports_missing_gold_refs_as_zero_for_kept_queries() -> None:
    ds = _toy_dataset()
    docs = build_documents(ds, corpus_splits=("train",))
    queries = build_queries(ds, documents=docs, eval_splits=("validation",))
    audit = compute_leakage_audit(
        dataset=ds,
        corpus_splits=("train",),
        eval_splits=("validation",),
        documents=docs,
        queries=queries,
    )

    assert audit["missing_gold_doc_references"] == 0
    assert audit["split_contamination_risk"] is True
    assert audit["leakage_pass"] is False
