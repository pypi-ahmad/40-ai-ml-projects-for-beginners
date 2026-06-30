from __future__ import annotations

import asyncio
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from api.app import create_api_app
from server.platform import Platform

app = typer.Typer(help="Production MCP Server CLI")
console = Console()


def _platform(config: str) -> Platform:
    return Platform.from_config(config)


@app.command()
def run(
    config: str = typer.Option("configs/default.yaml", help="Path to YAML config"),
    mode: str = typer.Option(
        "mcp",
        help="Run mode: mcp | api | all",
    ),
) -> None:
    platform = _platform(config)

    if mode == "mcp":
        platform.run_mcp_server()
        return

    if mode == "api":
        api_app = create_api_app(platform)
        uvicorn.run(api_app, host=platform.settings.transport.host, port=platform.settings.transport.port)
        return

    if mode == "all":
        api_app = create_api_app(platform)
        uvicorn.run(api_app, host=platform.settings.transport.host, port=platform.settings.transport.port)
        return

    raise typer.BadParameter("mode must be one of: mcp, api, all")


@app.command()
def tools(config: str = typer.Option("configs/default.yaml")) -> None:
    platform = _platform(config)
    table = Table(title="Available Tools")
    table.add_column("Name")
    table.add_column("Read-Only")
    table.add_column("Description")

    for item in platform.tools.list():
        table.add_row(item["name"], str(item["annotations"]["readOnlyHint"]), item["description"])
    console.print(table)


@app.command()
def resources(config: str = typer.Option("configs/default.yaml")) -> None:
    platform = _platform(config)
    table = Table(title="Available Resources")
    table.add_column("URI")
    table.add_column("MIME")
    table.add_column("Description")
    for resource in platform.resources.list():
        table.add_row(resource["uri"], resource["mime_type"], resource["description"])
    console.print(table)


@app.command()
def prompts(config: str = typer.Option("configs/default.yaml")) -> None:
    platform = _platform(config)
    table = Table(title="Prompt Library")
    table.add_column("Name")
    table.add_column("Role")
    table.add_column("Objective")
    for prompt in platform.prompts.list():
        table.add_row(prompt["name"], prompt["role"], prompt["objective"])
    console.print(table)


@app.command()
def memory(
    config: str = typer.Option("configs/default.yaml"),
    query: str = typer.Option("server", help="Semantic query"),
    top_k: int = typer.Option(5),
) -> None:
    platform = _platform(config)
    results = platform.memory.semantic_search(query, top_k=top_k)
    console.print(results)


@app.command()
def report(
    config: str = typer.Option("configs/default.yaml"),
    query: str = typer.Argument(..., help="Workflow query"),
) -> None:
    platform = _platform(config)
    result = asyncio.run(platform.run_workflow(query))
    console.print(result)


@app.command()
def doctor(config: str = typer.Option("configs/default.yaml")) -> None:
    platform = _platform(config)
    table = Table(title="Platform Diagnostics")
    table.add_column("Check")
    table.add_column("Value")

    table.add_row("Config path", str(Path(config).resolve()))
    table.add_row("Runtime", platform.settings.transport.runtime)
    table.add_row("Transport", platform.settings.transport.mode)
    table.add_row("FastMCP available", str(platform.fastmcp.available()))
    table.add_row("MCP SDK available", str(platform.mcp_sdk.available()))
    table.add_row("Tools", str(len(platform.tools.names())))
    table.add_row("Resources", str(len(platform.resources.list())))
    table.add_row("Prompts", str(len(platform.prompts.names())))
    table.add_row("SQLite", platform.settings.memory.sqlite_path)
    table.add_row("Chroma", platform.settings.memory.chroma_path)
    console.print(table)


@app.command()
def monitor(
    config: str = typer.Option("configs/default.yaml"),
    limit: int = typer.Option(30),
) -> None:
    platform = _platform(config)
    metrics = platform.metrics.recent(limit=limit)
    table = Table(title="Recent Metrics")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_column("Labels")
    table.add_column("Timestamp")

    for metric in metrics:
        table.add_row(
            metric["metric_name"],
            str(metric["metric_value"]),
            str(metric["labels"]),
            metric["created_at"],
        )
    console.print(table)


if __name__ == "__main__":
    app()
