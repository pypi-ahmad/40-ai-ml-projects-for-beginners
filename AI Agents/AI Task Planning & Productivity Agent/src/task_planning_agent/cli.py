"""Typer CLI for task planning agent."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from task_planning_agent.agent.service import PlanningService
from task_planning_agent.api.app import run as run_api
from task_planning_agent.config import load_config
from task_planning_agent.schemas import PriorityStrategy


app = typer.Typer(help="AI Task Planning & Productivity Agent CLI")
console = Console()


def _service() -> PlanningService:
    return PlanningService(load_config())


@app.command("plan")
def plan_cmd(
    user_id: str = typer.Option(...),
    input_text: str = typer.Option(...),
    strategy: PriorityStrategy = typer.Option(PriorityStrategy.WSJF),
    timezone: str = typer.Option("Asia/Kolkata"),
) -> None:
    """Generate plan from raw text input."""

    report = _service().plan(user_id=user_id, raw_input=input_text, strategy=strategy, timezone=timezone)
    console.print_json(data=report.model_dump(mode="json"))


@app.command("replan")
def replan_cmd(
    user_id: str = typer.Option(...),
    reason: str = typer.Option(...),
    additional_input: str = typer.Option(""),
) -> None:
    """Re-plan from latest run context."""

    report = _service().replan(user_id=user_id, reason=reason, additional_input=additional_input)
    console.print_json(data=report.model_dump(mode="json"))


@app.command("search")
def search_cmd(user_id: str = typer.Option(...), query: str = typer.Option(...)) -> None:
    """Search task and semantic history."""

    service = _service()
    tasks = service.memory.search_tasks(user_id=user_id, query=query)
    semantic = service.memory.semantic_search(query)

    table = Table(title="Search Results")
    table.add_column("Type")
    table.add_column("Value")
    for task in tasks:
        table.add_row("task", f"{task.name} ({task.priority_score})")
    for item in semantic[:5]:
        table.add_row("semantic", str(item))
    console.print(table)


@app.command("report")
def report_cmd(user_id: str = typer.Option(...), output: str = typer.Option("artifacts/reports/latest.json")) -> None:
    """Export latest plan session report."""

    service = _service()
    history = service.memory.history(user_id=user_id, limit=1)
    if not history:
        raise typer.Exit("No history found for user")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(history[0].model_dump(mode="json"), indent=2), encoding="utf-8")
    console.print(f"Saved report to {output}")


@app.command("serve-api")
def serve_api() -> None:
    """Run FastAPI service."""

    run_api()


@app.command("serve-ui")
def serve_ui() -> None:
    """Run Streamlit dashboard."""

    cfg = load_config()
    ui_cfg = cfg.streamlit
    subprocess.run(
        [
            "streamlit",
            "run",
            "streamlit_app.py",
            "--server.address",
            str(ui_cfg.get("host", "0.0.0.0")),
            "--server.port",
            str(ui_cfg.get("port", 8501)),
        ],
        check=True,
    )


@app.command("verify")
def verify() -> None:
    """Run focused verification checks."""

    subprocess.run(["uv", "run", "pytest", "-q"], check=True)


if __name__ == "__main__":
    app()
