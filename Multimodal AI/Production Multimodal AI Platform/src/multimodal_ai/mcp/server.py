"""MCP server exposing multimodal platform tools."""

from __future__ import annotations

import json

from multimodal_ai.domain import InputPayload, RequestEnvelope, TraceContext
from multimodal_ai.services.bootstrap import build_platform_service


def build_mcp_server():
    """Build MCP server instance lazily to avoid import hard-fail."""

    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"MCP server dependency not available: {exc}") from exc

    service = build_platform_service()
    mcp = FastMCP("multimodal-ai-platform")

    @mcp.tool()
    def ocr(path: str) -> str:
        req = RequestEnvelope(
            input=InputPayload(document_path=path),
            trace=TraceContext(source="mcp"),
        )
        return json.dumps(service.ocr(req).model_dump(mode="json"))

    @mcp.tool()
    def caption(image_path: str, style: str = "detailed") -> str:
        req = RequestEnvelope(
            input=InputPayload(image_path=image_path),
            options={"style": style},
            trace=TraceContext(source="mcp"),
        )
        return json.dumps(service.caption(req).model_dump(mode="json"))

    @mcp.tool()
    def search(query: str, modality: str = "image", top_k: int = 5) -> str:
        req = RequestEnvelope(
            input=InputPayload(query=query),
            options={"modality": modality, "top_k": top_k},
            trace=TraceContext(source="mcp"),
        )
        return json.dumps(service.search(req).model_dump(mode="json"))

    @mcp.tool()
    def vqa(image_path: str, question: str) -> str:
        req = RequestEnvelope(
            input=InputPayload(image_path=image_path, question=question),
            trace=TraceContext(source="mcp"),
        )
        return json.dumps(service.vqa(req).model_dump(mode="json"))

    @mcp.tool()
    def embeddings(text: str) -> str:
        req = RequestEnvelope(
            input=InputPayload(text=text),
            trace=TraceContext(source="mcp"),
        )
        return json.dumps(service.embeddings(req).model_dump(mode="json"))

    @mcp.tool()
    def retrieve(query: str, modality: str = "document", top_k: int = 5) -> str:
        req = RequestEnvelope(
            input=InputPayload(query=query),
            options={"modality": modality, "top_k": top_k},
            trace=TraceContext(source="mcp"),
        )
        return json.dumps(service.retrieve(req).model_dump(mode="json"))

    return mcp


def run() -> None:
    """Run MCP stdio server."""

    server = build_mcp_server()
    server.run()


if __name__ == "__main__":
    run()
