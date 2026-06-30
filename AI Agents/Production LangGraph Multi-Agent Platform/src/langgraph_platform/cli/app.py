"""Typer CLI for LangGraph platform."""

from __future__ import annotations

import json
import subprocess

import typer
from rich import print

from langgraph_platform.config.loader import load_config
from langgraph_platform.engine.workflow import LangGraphWorkflowEngine
from langgraph_platform.ui.graph_visualizer import export_graph_files

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command("run")
def run_workflow(request: str = typer.Argument(..., help="User request")) -> None:
    """Run workflow once and print final report."""

    config = load_config()
    engine = LangGraphWorkflowEngine(config)
    try:
        result = engine.run(request)
        print(f"[bold green]workflow_id:[/bold green] {result.workflow_id}")
        print(f"[bold green]confidence:[/bold green] {result.confidence:.2f}")
        print(result.final_report)
    finally:
        engine.close()


@app.command("graph")
def graph_info() -> None:
    """Show graph topology and export visual artifacts."""

    config = load_config()
    engine = LangGraphWorkflowEngine(config)
    try:
        info = engine.inspect_graph()
        exports = export_graph_files(info)
        print(json.dumps(info, indent=2))
        print(f"[bold]exports:[/bold] {exports}")
    finally:
        engine.close()


@app.command("state")
def state(limit: int = 10) -> None:
    """Show recent workflow states."""

    config = load_config()
    engine = LangGraphWorkflowEngine(config)
    try:
        print(json.dumps(engine.sqlite_store.list_recent_runs(limit=limit), indent=2))
    finally:
        engine.close()


@app.command("memory")
def memory(query: str = typer.Argument("", help="Memory query"), limit: int = 10) -> None:
    """Search persistent memory."""

    config = load_config()
    engine = LangGraphWorkflowEngine(config)
    try:
        result = engine.runtime.tool_registry.run("memory_search", {"query": query, "limit": limit})
        print(json.dumps(result.output, indent=2, default=str))
    finally:
        engine.close()


@app.command("report")
def report(workflow_id: str, markdown_path: str) -> None:
    """Export report via API endpoint helper."""

    from pathlib import Path

    from langgraph_platform.exporters.report_exporter import ReportExporter

    markdown_report = Path(markdown_path).read_text(encoding="utf-8")
    exporter = ReportExporter()
    paths = exporter.export_all(workflow_id, markdown_report, {"workflow_id": workflow_id})
    print(paths)


@app.command("doctor")
def doctor() -> None:
    """Run basic environment diagnostics."""

    checks: dict[str, str] = {}
    checks["python"] = subprocess.getoutput("python3 --version")
    checks["ollama"] = subprocess.getoutput("ollama --version")
    checks["uv"] = subprocess.getoutput("uv --version")
    checks["gpu"] = subprocess.getoutput(
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
    )
    print(json.dumps(checks, indent=2))


@app.command("dashboard")
def dashboard() -> None:
    """Launch Streamlit dashboard."""

    subprocess.run(
        [
            "streamlit",
            "run",
            "src/langgraph_platform/ui/dashboard.py",
            "--server.fileWatcherType",
            "none",
        ],
        check=False,
    )


if __name__ == "__main__":
    app()
