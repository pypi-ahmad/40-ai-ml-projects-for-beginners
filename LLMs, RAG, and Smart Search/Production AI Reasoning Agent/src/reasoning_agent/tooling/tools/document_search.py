"""Keyword document search tool over local text/markdown files."""

from __future__ import annotations

from pydantic import BaseModel, Field

from reasoning_agent.tooling.base import ToolContext, ToolSpec


class DocumentSearchInput(BaseModel):
    """Document search input payload."""

    query: str
    directory: str = "."
    extensions: list[str] = Field(default_factory=lambda: [".txt", ".md", ".py", ".json"])
    max_results: int = Field(default=10, ge=1, le=50)


class DocumentSearchOutput(BaseModel):
    """Document search output payload."""

    query: str
    matches: list[dict[str, str | int]]


def search_documents(payload: DocumentSearchInput, context: ToolContext) -> DocumentSearchOutput:
    """Simple lexical search in workspace files."""

    base = (context.workspace_root / payload.directory).resolve()
    root = context.workspace_root.resolve()
    if root not in base.parents and base != root:
        raise ValueError("Directory outside workspace root is not allowed")

    query = payload.query.lower()
    results: list[dict[str, str | int]] = []

    for path in base.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {ext.lower() for ext in payload.extensions}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for idx, line in enumerate(text.splitlines(), start=1):
            if query in line.lower():
                results.append(
                    {
                        "path": str(path.relative_to(root)),
                        "line": idx,
                        "snippet": line.strip()[:200],
                    }
                )
                if len(results) >= payload.max_results:
                    return DocumentSearchOutput(query=payload.query, matches=results)

    return DocumentSearchOutput(query=payload.query, matches=results)


spec = ToolSpec(
    name="document_search",
    description="Search workspace documents by keyword",
    input_model=DocumentSearchInput,
    output_model=DocumentSearchOutput,
    tags=["filesystem", "search"],
)
