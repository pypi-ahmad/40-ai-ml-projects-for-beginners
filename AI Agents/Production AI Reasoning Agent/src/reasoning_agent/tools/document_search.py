"""Document keyword search over local markdown/txt files."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from reasoning_agent.tools.base import BaseTool


class DocumentSearchInput(BaseModel):
    query: str
    glob_pattern: str = "**/*.*"
    max_hits: int = 5


class DocumentSearchOutput(BaseModel):
    hits: list[dict[str, str]]


class DocumentSearchTool(BaseTool[DocumentSearchInput, DocumentSearchOutput]):
    name = "document_search"
    description = "Keyword search in local documents"
    input_model = DocumentSearchInput
    output_model = DocumentSearchOutput

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.resolve()

    async def run(self, payload: DocumentSearchInput) -> DocumentSearchOutput:
        query = payload.query.lower()
        hits: list[dict[str, str]] = []
        for path in self.workspace_root.glob(payload.glob_pattern):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".txt", ".md", ".py", ".json", ".yaml", ".yml"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            idx = text.lower().find(query)
            if idx == -1:
                continue
            start = max(0, idx - 120)
            end = min(len(text), idx + 120)
            hits.append({"path": str(path.relative_to(self.workspace_root)), "snippet": text[start:end]})
            if len(hits) >= payload.max_hits:
                break
        return DocumentSearchOutput(hits=hits)
