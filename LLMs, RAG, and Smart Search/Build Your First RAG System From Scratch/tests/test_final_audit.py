import pytest

from scripts import run_final_audit
from rag_system.types import DocumentRecord, QueryRecord


def test_parse_args_defaults_to_real_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_final_audit.py"])
    args = run_final_audit.parse_args()

    assert args.profile == "max_depth"
    assert args.chunking_docs is None
    assert args.chunking_queries is None
    assert args.advanced_queries is None
    assert args.retrieval_limit is None
    assert args.generation_limit is None
    assert args.judge_limit is None
    assert args.hallucination_limit is None
    assert args.strict_gates is True


def test_validate_report_payload_requires_core_keys() -> None:
    with pytest.raises(RuntimeError):
        run_final_audit.validate_report_payload({"run_type": "real"})


def test_validate_report_payload_accepts_complete_payload() -> None:
    payload = {
        "run_type": "real",
        "timestamp_utc": "2026-06-25T00:00:00+00:00",
        "profile_used": "max_depth",
        "effective_eval_limits": {},
        "models_used": {},
        "retrieval_summary": {},
        "generation_summary": {},
        "judge_summary": {},
        "leakage_audit": {},
        "index_integrity": {},
        "embedding_integrity": {},
        "required_outputs": [],
    }

    run_final_audit.validate_report_payload(payload)


def test_filter_queries_for_indexed_docs_keeps_only_eligible() -> None:
    queries = [
        QueryRecord(query_id="q1", query="a", gold_doc_ids=["d1"]),
        QueryRecord(query_id="q2", query="b", gold_doc_ids=["d2"]),
        QueryRecord(query_id="q3", query="c", gold_doc_ids=["d3", "d9"]),
    ]

    filtered = run_final_audit._filter_queries_for_indexed_docs(
        queries=queries,
        indexed_doc_ids={"d2", "d3"},
        limit=10,
    )

    assert [q.query_id for q in filtered] == ["q2", "q3"]


def test_select_docs_and_queries_for_budget_guarantees_coverage() -> None:
    documents = [
        DocumentRecord(doc_id="d1", text="one"),
        DocumentRecord(doc_id="d2", text="two"),
        DocumentRecord(doc_id="d3", text="three"),
    ]
    queries = [
        QueryRecord(query_id="q1", query="a", gold_doc_ids=["d1"]),
        QueryRecord(query_id="q2", query="b", gold_doc_ids=["d2"]),
        QueryRecord(query_id="q3", query="c", gold_doc_ids=["d3"]),
    ]

    docs, qrows = run_final_audit._select_docs_and_queries_for_budget(
        documents=documents,
        queries=queries,
        doc_limit=2,
        query_limit=3,
    )

    doc_ids = {doc.doc_id for doc in docs}
    assert doc_ids == {"d1", "d2"}
    assert [q.query_id for q in qrows] == ["q1", "q2"]
