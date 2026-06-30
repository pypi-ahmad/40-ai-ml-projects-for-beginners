"""Local RAG tool using workspace docs + semantic memory."""

from __future__ import annotations

from pydantic import BaseModel, Field

from reasoning_agent.memory import MemoryEvent, MemoryScope, MemoryService
from reasoning_agent.tooling.base import ToolContext, ToolSpec


class LocalRAGInput(BaseModel):
    """Local RAG input payload."""

    query: str
    directory: str = "."
    top_k: int = Field(default=5, ge=1, le=20)
    chunk_size: int = Field(default=800, ge=200, le=4000)


class LocalRAGOutput(BaseModel):
    """Local RAG output payload."""

    query: str
    chunks: list[dict[str, object]]


def _chunk_text(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def make_handler(memory: MemoryService):
    """Create local RAG handler."""

    def handler(payload: LocalRAGInput, context: ToolContext) -> LocalRAGOutput:
        base = (context.workspace_root / payload.directory).resolve()
        root = context.workspace_root.resolve()
        if root not in base.parents and base != root:
            raise ValueError("Directory outside workspace root")

        for file_path in base.rglob("*"):
            if not file_path.is_file() or file_path.suffix.lower() not in {".md", ".txt", ".py", ".json"}:
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for chunk in _chunk_text(text, payload.chunk_size):
                if not chunk.strip():
                    continue
                memory.write(
                    MemoryEvent(
                        session_id=context.session_id,
                        run_id=context.run_id,
                        scope=MemoryScope.SEMANTIC,
                        text=chunk,
                        metadata={"path": str(file_path.relative_to(root))},
                    )
                )

        hits = memory.retrieve(payload.query, k=payload.top_k, scope=MemoryScope.SEMANTIC)
        return LocalRAGOutput(
            query=payload.query,
            chunks=[
                {
                    "text": hit.text,
                    "score": hit.score,
                    "metadata": hit.metadata,
                }
                for hit in hits
            ],
        )

    return handler


spec = ToolSpec(
    name="local_rag",
    description="Index local documents and retrieve relevant semantic chunks",
    input_model=LocalRAGInput,
    output_model=LocalRAGOutput,
    tags=["rag", "memory", "filesystem"],
)
