from semantic_search.query_processing import QueryProcessor


def test_query_processor_expands_abbreviation_and_synonyms():
    qp = QueryProcessor()
    qp.build_vocabulary(["application programming interface", "artificial intelligence"])
    processed = qp.process("API ai")
    assert "application" in processed
    assert "artificial" in processed


def test_query_processor_spell_correction():
    qp = QueryProcessor(spell_correction=True, synonym_expansion=False, abbreviation_expansion=False)
    qp.build_vocabulary(["embedding", "semantic", "retrieval"])
    processed = qp.process("embeddng retrieval")
    assert "embedding" in processed
