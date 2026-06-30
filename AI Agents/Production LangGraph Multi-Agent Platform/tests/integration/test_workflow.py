from __future__ import annotations

from typing import Any

from langgraph_platform.config.settings import AppConfig
from langgraph_platform.engine.llm import LLMResponse
from langgraph_platform.engine.workflow import LangGraphWorkflowEngine
from langgraph_platform.tools.base import ToolResult


class FakeLLM:
    def close(self) -> None:
        return None

    def json_with_fallback(
        self, prompt: str, model_chain: list[str], system: str | None = None
    ) -> tuple[dict[str, Any], LLMResponse]:
        if "Planner Agent" in (system or ""):
            return (
                {
                    "plan": "Test plan",
                    "subtasks": ["collect", "write", "verify"],
                    "routing": {
                        "web": True,
                        "rag": False,
                        "memory": False,
                        "code": False,
                        "verification": True,
                    },
                    "confidence": 0.85,
                },
                LLMResponse(
                    text="{}", model=model_chain[0], prompt_tokens=10, completion_tokens=20
                ),
            )
        return (
            {"improvements": ["tighten intro"], "confidence": 0.82},
            LLMResponse(text="{}", model=model_chain[0], prompt_tokens=8, completion_tokens=11),
        )

    def generate_with_fallback(
        self, prompt: str, model_chain: list[str], system: str | None = None
    ) -> LLMResponse:
        report = "# Executive Summary\n\nTest report with http://example.com citation and assumptions section."
        return LLMResponse(
            text=report, model=model_chain[0], prompt_tokens=14, completion_tokens=55
        )


class FakeToolRegistry:
    def run(self, name: str, args: dict[str, Any], context: Any = None) -> ToolResult:
        if name == "duckduckgo_search":
            return ToolResult(
                ok=True,
                output=[{"title": "Result", "href": "http://example.com", "body": "Body"}],
                source=name,
            )
        if name == "github_search":
            return ToolResult(
                ok=True,
                output=[
                    {"title": "Repo", "url": "http://github.com/example/repo", "body": "Repo body"}
                ],
                source=name,
            )
        if name == "documentation_search":
            return ToolResult(
                ok=True,
                output=[
                    {"title": "Doc", "url": "http://docs.example.com", "snippet": "Doc snippet"}
                ],
                source=name,
            )
        if name == "memory_search":
            return ToolResult(ok=True, output=[], source=name)
        return ToolResult(ok=True, output=[], source=name)


class FakeRAG:
    def retrieve(self, query: str, top_k: int = 5) -> list[Any]:
        return []

    def ingest_paths(self, paths: list[str], source_type: str = "file") -> Any:
        return {"documents_loaded": len(paths), "chunks_created": len(paths)}

    def ingest_urls(self, urls: list[str]) -> Any:
        return {"documents_loaded": len(urls), "chunks_created": len(urls)}


def test_workflow_run_end_to_end(tmp_path: Any) -> None:
    config = AppConfig()
    config.memory.sqlite_path = str(tmp_path / "platform.db")
    config.memory.chroma_path = str(tmp_path / "chroma")
    engine = LangGraphWorkflowEngine(config)

    engine.runtime.llm_client = FakeLLM()
    engine.runtime.tool_registry = FakeToolRegistry()  # type: ignore[assignment]
    engine.rag_pipeline = FakeRAG()  # type: ignore[assignment]
    engine.runtime.rag_pipeline = engine.rag_pipeline  # type: ignore[assignment]

    result = engine.run("Generate enterprise research report")

    assert result.workflow_id.startswith("wf_")
    assert "Executive Summary" in result.final_report
    assert result.confidence > 0
    engine.close()
