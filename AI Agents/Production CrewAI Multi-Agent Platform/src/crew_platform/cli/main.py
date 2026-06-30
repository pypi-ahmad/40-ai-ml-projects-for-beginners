"""Typer CLI for crew platform."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich import print

from crew_platform.config import load_settings
from crew_platform.orchestration import CrewRunRequest, PlanApproval, create_service

app = typer.Typer(help="Production CrewAI Multi-Agent Platform CLI")


def _pretty(data) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


async def _local_service_call(query: str, execute: bool, force_consensus: bool) -> dict:
    service = create_service(load_settings())
    planned = await service.create_plan(CrewRunRequest(query=query, auto_execute=False))
    if service.settings.orchestration.plan_approval_required:
        await service.apply_approval(planned.run_id, PlanApproval(approved=True, reviewer="cli"))
    if execute:
        result = await service.execute_run(planned.run_id, force_consensus=force_consensus)
        return result.model_dump(mode="json")
    return planned.model_dump(mode="json")


@app.command("run")
def run_command(
    query: str = typer.Argument(..., help="Objective to solve"),
    execute: bool = typer.Option(True, help="Execute after planning"),
    force_consensus: bool = typer.Option(False, help="Force triad consensus"),
    api_url: Optional[str] = typer.Option(None, help="FastAPI base URL (if provided, CLI uses API mode)"),
) -> None:
    """Plan and execute collaboration workflow."""

    if api_url:
        with httpx.Client(timeout=120) as client:
            planned = client.post(f"{api_url.rstrip('/')}/crew", json={"query": query, "auto_execute": False})
            planned.raise_for_status()
            data = planned.json()
            run_id = data["run_id"]
            client.post(f"{api_url.rstrip('/')}/crew/{run_id}/approve", json={"approved": True, "reviewer": "cli"})
            if execute:
                response = client.post(
                    f"{api_url.rstrip('/')}/crew/{run_id}/execute",
                    params={"force_consensus": str(force_consensus).lower()},
                )
                response.raise_for_status()
                _pretty(response.json())
                return
            _pretty(data)
            return

    output = asyncio.run(_local_service_call(query, execute, force_consensus))
    _pretty(output)


@app.command("agents")
def agents_command(api_url: Optional[str] = typer.Option(None, help="FastAPI base URL")) -> None:
    """List registered agents."""

    if api_url:
        response = httpx.get(f"{api_url.rstrip('/')}/agents", timeout=30)
        response.raise_for_status()
        _pretty(response.json())
        return

    service = create_service(load_settings())
    _pretty({"agents": service.list_agents()})


@app.command("task")
def task_command(
    run_id: str = typer.Argument(..., help="Run identifier"),
    api_url: str = typer.Option("http://127.0.0.1:8000", help="FastAPI base URL"),
) -> None:
    """Show tasks for run."""

    response = httpx.get(f"{api_url.rstrip('/')}/tasks", params={"run_id": run_id}, timeout=60)
    response.raise_for_status()
    _pretty(response.json())


@app.command("report")
def report_command(
    run_id: str = typer.Argument(..., help="Run identifier"),
    format: str = typer.Option("markdown", help="Export format: markdown|json|html|pdf"),
    api_url: str = typer.Option("http://127.0.0.1:8000", help="FastAPI base URL"),
) -> None:
    """Get or export run report."""

    response = httpx.post(
        f"{api_url.rstrip('/')}/reports/{run_id}/export",
        json={"format": format},
        timeout=90,
    )
    response.raise_for_status()
    _pretty(response.json())


@app.command("memory")
def memory_command(
    query: Optional[str] = typer.Option(None, help="Optional semantic query"),
    limit: int = typer.Option(30, help="Rows limit"),
    api_url: str = typer.Option("http://127.0.0.1:8000", help="FastAPI base URL"),
) -> None:
    """Inspect persistent and semantic memory."""

    response = httpx.get(f"{api_url.rstrip('/')}/memory", params={"limit": limit, "query": query}, timeout=60)
    response.raise_for_status()
    _pretty(response.json())


@app.command("doctor")
def doctor_command(api_url: str = typer.Option("http://127.0.0.1:8000", help="FastAPI base URL")) -> None:
    """Run health and metrics diagnostics."""

    health = httpx.get(f"{api_url.rstrip('/')}/health", timeout=20)
    metrics = httpx.get(f"{api_url.rstrip('/')}/metrics", timeout=20)
    health.raise_for_status()
    metrics.raise_for_status()
    _pretty({"health": health.json(), "metrics": metrics.json()})


@app.command("dashboard")
def dashboard_command(host: str = "127.0.0.1", port: int = 8501) -> None:
    """Launch Streamlit dashboard."""

    dashboard_path = Path("streamlit_app/Home.py")
    if not dashboard_path.exists():
        raise typer.BadParameter("streamlit_app/Home.py not found")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    app()
