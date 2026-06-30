from rag_system.prompts import PromptLibrary


def test_rag_prompt_contains_grounding_instruction() -> None:
    messages = PromptLibrary.rag_answer("What is RAG?", "[1] RAG retrieves docs")
    joined = "\n".join(message["content"] for message in messages)
    assert "provided context" in joined.lower() or "context" in joined.lower()
    assert "cite" in joined.lower()


def test_query_expansion_prompt_mentions_json_output() -> None:
    messages = PromptLibrary.query_expansion("How does retrieval work?", n=4)
    assert "json" in messages[0]["content"].lower()
