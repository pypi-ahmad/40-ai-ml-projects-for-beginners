"""Typer CLI for internet-agent runtime operations."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from internet_agent.config import get_settings
from internet_agent.services.agent_service import InternetAgentService

app = typer.Typer(help="Production Internet AI Agent CLI")
console = Console()


def _service() -> InternetAgentService:
    return InternetAgentService(settings=get_settings())


@app.command()
def chat(message: str | None = typer.Argument(default=None), session_id: str = "cli") -> None:
    """Send chat query through LangGraph workflow."""

    query = message or typer.prompt("Query")
    payload = asyncio.run(_service().chat(session_id=session_id, message=query))

    console.print("\n[bold]Answer[/bold]")
    console.print(payload["answer"])
    console.print(f"\nConfidence: {payload['confidence']:.3f}")

    if payload.get("citations"):
        table = Table(title="Citations")
        table.add_column("Title")
        table.add_column("URL")
        for row in payload["citations"]:
            table.add_row(row.get("title", ""), row.get("url", ""))
        console.print(table)


@app.command()
def search(query: str, session_id: str = "cli", providers: str = "") -> None:
    """Run hybrid search/retrieval pipeline."""

    selected = [p.strip() for p in providers.split(",") if p.strip()] or None
    payload = asyncio.run(_service().search(session_id=session_id, query=query, providers=selected))

    table = Table(title=f"Results for: {query}")
    table.add_column("Score")
    table.add_column("Source")
    table.add_column("Title")
    table.add_column("URL")

    for row in payload.get("results", [])[:10]:
        table.add_row(
            f"{row.get('rank_score', 0):.3f}",
            row.get("source", ""),
            row.get("title", "")[:80],
            row.get("url", "")[:120],
        )
    console.print(table)
    console.print(f"Documents: {len(payload.get('documents', []))} | Chunks: {len(payload.get('chunks', []))}")


@app.command()
def summarize(url: str, session_id: str = "cli") -> None:
    """Fetch URL and summarize content."""

    service = _service()
    summary = asyncio.run(_summarize_url(service, session_id=session_id, url=url))
    console.print(summary)


@app.command()
def report(
    session_id: str = "cli",
    format: str = "json",
    payload_path: str = "",
) -> None:
    """Export report from JSON payload or latest chat output file."""

    service = _service()

    if payload_path:
        payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    else:
        history = service.history(session_id)
        messages = history.get("messages", [])
        if not messages:
            raise typer.BadParameter("No history found. Provide --payload-path.")
        payload = {
            "session_id": session_id,
            "query": messages[-2]["content"] if len(messages) >= 2 else "",
            "answer": messages[-1]["content"],
            "citations": [],
            "reasoning_trace": [],
            "confidence": 0.0,
            "hallucination_risk": "unknown",
            "missing_info": [],
            "conflicts": [],
            "timestamp": "",
        }

    result = service.export_report(session_id=session_id, payload=payload, fmt=format)
    console.print(f"Report saved: {result['path']}")


@app.command()
def memory(query: str = "", top_k: int = 5, session_id: str = "cli") -> None:
    """Query semantic memory or print conversation history."""

    service = _service()
    if query:
        hits = service.memory_search(query=query, top_k=top_k)
        table = Table(title=f"Semantic Memory Hits: {query}")
        table.add_column("Distance")
        table.add_column("Content")
        for row in hits.get("hits", []):
            table.add_row(f"{row.get('distance', 0):.3f}", row.get("content", "")[:140])
        console.print(table)
    else:
        history = service.history(session_id)
        for msg in history.get("messages", []):
            console.print(f"[{msg['role']}] {msg['content']}")


@app.command()
def doctor() -> None:
    """Run environment and runtime diagnostics."""

    service = _service()
    settings = service.settings

    console.print("[bold]Internet Agent Doctor[/bold]")
    console.print(f"Config loaded: {settings.app.name}")
    console.print(f"Ollama URL: {settings.llm.base_url}")
    console.print(f"Models: {settings.llm.planning_model}, {settings.llm.reasoning_model}")

    monitor = service.monitor()
    console.print(f"CPU: {monitor['cpu_percent']:.1f}%")
    console.print(f"RAM: {monitor['memory']['percent']:.1f}%")

    if monitor["gpu"].get("available"):
        for gpu in monitor["gpu"].get("gpus", []):
            console.print(
                f"GPU {gpu['name']}: util={gpu['utilization_gpu_percent']:.1f}% "
                f"VRAM={gpu['memory_used_mb']:.0f}/{gpu['memory_total_mb']:.0f}MB"
            )
    else:
        console.print("GPU stats unavailable (nvidia-smi not found)")


async def _summarize_url(service: InternetAgentService, session_id: str, url: str) -> str:
    browse_payload = await service.browse(session_id=session_id, url=url)
    content = browse_payload.get("content", "")[:12000]
    return await service.llm.ask(
        task_model=service.settings.llm.summarization_model,
        system_prompt="Summarize technical web content into concise bullet points.",
        user_prompt=content,
    )


if __name__ == "__main__":
    app()
