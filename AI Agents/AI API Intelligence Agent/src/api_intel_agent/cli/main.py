"""CLI for AI API Intelligence Agent."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from api_intel_agent.agents import AgentRunner
from api_intel_agent.connectors import ConnectorRegistry
from api_intel_agent.core.schemas import AnalyzeRequest, MemorySearchRequest
from api_intel_agent.memory import MemoryManager

app = typer.Typer(help="Production AI API Intelligence Agent CLI")
search_app = typer.Typer(help="Search commands")
app.add_typer(search_app, name="search")
console = Console()
runner = AgentRunner()
connectors = ConnectorRegistry()
memory = MemoryManager()


@app.command()
def query(text: str, model: str | None = None):
    """Run full multi-agent analysis for user query."""
    response = runner.query_sync(text, model=model)
    console.print(f"[bold]Run ID:[/bold] {response.run_id}")
    console.print(response.summary)
    if response.recommendations:
        console.print("\nRecommendations:")
        for item in response.recommendations:
            console.print(f"- {item}")


@app.command()
def github(q: str = "open source llm"):
    """Fetch GitHub repositories."""
    import asyncio

    result = asyncio.run(connectors.get("github").execute(q))
    table = Table(title="GitHub Results")
    table.add_column("Name")
    table.add_column("Stars", justify="right")
    for row in result.records[:10]:
        table.add_row(str(row.get("name", "-")), str(row.get("stargazers_count", "0")))
    console.print(table)


@app.command()
def weather(city: str = "London", lat: float = 51.5, lon: float = -0.12):
    """Fetch weather (city used as label; open-meteo uses coordinates)."""
    import asyncio

    result = asyncio.run(connectors.get("weather").execute(city, {"latitude": lat, "longitude": lon}))
    console.print_json(data=result.model_dump(mode="json"))


@app.command()
def news(topic: str = "AI"):
    """Fetch news entries."""
    import asyncio

    result = asyncio.run(connectors.get("news").execute(topic))
    console.print_json(data=result.model_dump(mode="json"))


@app.command()
def analyze(path: str):
    """Analyze existing JSON report payload."""
    payload = json.loads(Path(path).read_text())
    query_text = payload.get("query") or payload.get("summary") or "analyze report"
    response = runner.query_sync(query_text)
    console.print_json(data=response.model_dump(mode="json"))


@app.command("search-memory")
def search_memory(text: str, top_k: int = 5):
    """Semantic search over persistent memory."""
    hits = memory.search(MemorySearchRequest(query=text, top_k=top_k))
    console.print_json(data={"hits": [hit.model_dump(mode="json") for hit in hits]})


@search_app.command("memory")
def search_memory_nested(text: str, top_k: int = 5):
    """Compatibility command: agent search memory \"text\"."""
    search_memory(text=text, top_k=top_k)


@app.command("validate-models")
def validate_models(no_smoke: bool = True):
    """Print supported local model families without running live smoke checks."""
    _ = no_smoke
    from api_intel_agent.config import load_settings

    settings = load_settings()
    console.print_json(data={"supported_models": settings.llm.supported_models})


if __name__ == "__main__":
    app()
