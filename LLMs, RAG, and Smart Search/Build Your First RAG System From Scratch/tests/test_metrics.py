from rag_system.metrics import compute_retrieval_metrics


def test_retrieval_metrics_expected_values() -> None:
    query_ids = ["q1", "q2"]
    retrieved = [["d1", "d2", "d3"], ["d9", "d8", "d7"]]
    gold = [["d2"], ["d8", "d10"]]

    summary, rows = compute_retrieval_metrics(
        query_ids=query_ids,
        retrieved_doc_ids=retrieved,
        gold_doc_ids=gold,
        top_k=3,
    )

    assert len(rows) == 2
    assert abs(summary.precision_at_k - 0.3333) < 1e-3
    assert abs(summary.recall_at_k - 0.75) < 1e-6
    assert abs(summary.f1_at_k - 0.45) < 1e-6
    assert abs(summary.mrr - 0.5) < 1e-6
    assert abs(summary.ndcg - 0.50889) < 1e-3
