"""Command-line interface for AI SQL Assistant workflows."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ai_sql_assistant.benchmarking.benchmark import BenchmarkRunner
from ai_sql_assistant.config import get_settings
from ai_sql_assistant.constants import REPORTS_DIR
from ai_sql_assistant.data.northwind import build_northwind_databases, sqlite_md5
from ai_sql_assistant.logging_utils import configure_logging, logger
from ai_sql_assistant.pipeline.assistant import AISQLAssistant
from ai_sql_assistant.schema.introspector import generate_erd, inspect_database, save_schema_report
from ai_sql_assistant.types import QueryRequest
from ai_sql_assistant.utils.runtime import assert_runtime_or_raise

app = typer.Typer(help="AI SQL Analytics Assistant CLI")
console = Console()


def _init() -> None:
    configure_logging()


@app.command("runtime-check")
def runtime_check() -> None:
    """Validate runtime requirements."""
    _init()
    assert_runtime_or_raise()
    console.print("Python runtime check passed.")


@app.command("data-build")
def data_build(scale_factor: int = typer.Option(8, min=1, max=20)) -> None:
    """Build raw and scaled Northwind SQLite databases."""
    _init()
    settings = get_settings()
    result = build_northwind_databases(
        raw_db_path=settings.database.raw_db_path,
        scaled_db_path=settings.database.scaled_db_path,
        scale_factor=scale_factor,
    )
    payload = {
        "source": result.source,
        "raw_db": str(result.raw_db_path),
        "scaled_db": str(result.scaled_db_path),
        "raw_orders": result.raw_orders,
        "scaled_orders": result.scaled_orders,
        "raw_md5": sqlite_md5(result.raw_db_path),
        "scaled_md5": sqlite_md5(result.scaled_db_path),
    }
    console.print_json(json.dumps(payload))


@app.command("schema-report")
def schema_report(use_raw: bool = False) -> None:
    """Inspect schema and generate markdown/json report + ERD image."""
    _init()
    settings = get_settings()
    db_path = settings.database.raw_db_path if use_raw else settings.database.scaled_db_path
    report = inspect_database(db_path)
    md_path, json_path = save_schema_report(report)
    erd_path = generate_erd(report)

    console.print(
        f"Schema report generated.\n- markdown: {md_path}\n- json: {json_path}\n- erd: {erd_path}"
    )


@app.command("ask")
def ask(
    question: str = typer.Argument(...),
    approach: str = typer.Option("langchain", help="langchain|direct"),
    persona: str = typer.Option("Business Analyst"),
    model: str | None = typer.Option(None),
    conversation_id: str = typer.Option("cli-session"),
) -> None:
    """Ask one business question and execute full SQL pipeline."""
    _init()
    settings = get_settings()
    assistant = AISQLAssistant(settings)
    try:
        req = QueryRequest(
            question=question,
            persona=persona,
            conversation_id=conversation_id,
            user_id="cli",
        )
        response = assistant.ask(req, approach=approach, model=model)

        console.print("\n[bold]Generated SQL[/bold]")
        console.print(response.execution.sql)

        if response.validation.issues:
            console.print("\n[bold]Validation Issues[/bold]")
            for issue in response.validation.issues:
                console.print(f"- {issue.code}: {issue.message}")

        console.print("\n[bold]Execution[/bold]")
        console.print(
            f"status={response.execution.status}, rows={response.execution.row_count}, "
            f"time_ms={response.execution.execution_time_ms:.2f}"
        )

        if response.execution.rows:
            table = Table(show_header=True)
            for col in response.execution.columns:
                table.add_column(col)
            for row in response.execution.rows[:20]:
                table.add_row(*[str(row.get(col, "")) for col in response.execution.columns])
            console.print(table)

        report_path = REPORTS_DIR / "last_query_report.json"
        assistant.export_sql_report(response, report_path)
        console.print(f"Saved query report: {report_path}")
    finally:
        assistant.close()


@app.command("benchmark-run")
def benchmark_run(cases_file: Path = typer.Option(Path("benchmarks/benchmark_cases.json"))) -> None:
    """Run full benchmark matrix and save report."""
    _init()
    settings = get_settings()
    assistant = AISQLAssistant(settings)
    runner = BenchmarkRunner(assistant=assistant, settings=settings)

    try:
        cases = runner.load_cases(cases_file)
        run = runner.run(cases)
        case_map = {case.case_id: case for case in cases}
        judge_report = runner.evaluate_with_judge(case_map, run)
        report_path = runner.save_run(run, judge_report)

        console.print(f"Benchmark complete: {report_path}")
        console.print_json(json.dumps(run.metrics))
    finally:
        runner.close()
        assistant.close()


@app.command("streamlit-run")
def streamlit_run() -> None:
    """Launch Streamlit UI."""
    _init()
    subprocess.run(["uv", "run", "streamlit", "run", "streamlit_app/app.py"], check=True)


@app.command("why-temperature-zero")
def why_temperature_zero() -> None:
    """Explain deterministic decoding choice for SQL generation."""
    message = {
        "reasoning": [
            "SQL generation must be reproducible for safety validation and regression tests.",
            "Lower randomness reduces syntax drift and accidental unsafe tokens.",
            "Benchmark comparisons need stable outputs across repeated runs.",
            "Deterministic decoding simplifies debugging and human review in enterprise analytics flows.",
        ]
    }
    console.print_json(json.dumps(message))


if __name__ == "__main__":
    app()
