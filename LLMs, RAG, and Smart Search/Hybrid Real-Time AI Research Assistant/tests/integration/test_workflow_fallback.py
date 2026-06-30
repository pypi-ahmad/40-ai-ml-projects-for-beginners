from __future__ import annotations

from hybrid_research_assistant.graph import FallbackWorkflow, GraphComponents


class _Router:
    def route(self, query, requested_mode):  # noqa: ANN001, ANN201
        class Decision:
            mode = requested_mode
            reason = "test"
            confidence = 1.0

        return Decision()


class _Retrieval:
    def retrieve_local(self, query, top_k, metadata_filter=None):  # noqa: ANN001, ANN201
        from hybrid_research_assistant.schemas import RetrievedContext

        return [
            RetrievedContext(
                chunk_id="chunk-1",
                doc_id="doc-1",
                text="LangGraph is graph orchestration.",
                score=0.9,
                metadata={"source": "doc.md", "page_number": None, "url": None, "document_title": "Doc"},
                source="local",
            )
        ], 1.0

    async def retrieve_web(self, query, top_k, provider=None):  # noqa: ANN001, ANN201
        return [], 1.0

    async def retrieve_hybrid(self, query, local_k, web_k, metadata_filter=None, provider=None):  # noqa: ANN001, ANN201
        return [], 1.0


class _Reranker:
    def rerank(self, query, rows, top_k):  # noqa: ANN001, ANN201
        class Report:
            before_scores = [0.9]
            after_scores = [0.9]
            latency_ms = 1.0

        return rows[:top_k], Report()


class _LLM:
    def generate(self, messages):  # noqa: ANN001, ANN201
        return "LangGraph is orchestration. [doc.md|chunk-1]", 1.0


class _Judge:
    def evaluate(self, query, answer, context):  # noqa: ANN001, ANN201
        return {"grounding": 5, "correctness": 5, "completeness": 5, "clarity": 5, "citation_quality": 5}, 1.0


def test_fallback_workflow_end_to_end() -> None:
    workflow = FallbackWorkflow(
        GraphComponents(
            intent_router=_Router(),
            retrieval=_Retrieval(),
            reranker=_Reranker(),
            llm=_LLM(),
            judge=_Judge(),
            fallback_text="I don't know based on the retrieved information.",
            retrieval_top_k=5,
            candidate_k=5,
        )
    )

    state = workflow.invoke(
        {
            "query": "What is LangGraph?",
            "requested_mode": "local",
            "prompt_name": "research_assistant",
        }
    )

    assert "answer" in state
    assert "citations" in state
    assert state["timings"]["total_ms"] >= 0.0
