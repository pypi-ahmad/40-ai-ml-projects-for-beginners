from semantic_search.retrieval import reciprocal_rank_fusion


def test_rrf_prefers_items_present_in_multiple_lists():
    scores = reciprocal_rank_fusion([
        ["a", "b", "c"],
        ["b", "a", "d"],
    ], k=60)
    assert scores["b"] > scores["c"]
