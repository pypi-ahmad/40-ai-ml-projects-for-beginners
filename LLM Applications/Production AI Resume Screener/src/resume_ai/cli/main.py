"""Typer CLI for resume intelligence platform."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from resume_ai.observability.logging import configure_logging
from resume_ai.service import ResumeAIService

app = typer.Typer(help="AI Resume Screener CLI")
console = Console()
service: ResumeAIService | None = None


def get_service() -> ResumeAIService:
    global service
    if service is None:
        service = ResumeAIService()
    return service


@app.callback()
def _bootstrap() -> None:
    configure_logging("INFO")


@app.command("ingest")
def ingest(path: str, blind_mode: bool = True) -> None:
    """Ingest one resume file or folder."""
    source = Path(path)
    svc = get_service()
    if source.is_dir():
        queued = svc.enqueue_folder(path)
        ran = svc.run_queue(blind_mode=blind_mode)
        console.print_json(json.dumps({**queued, **ran}))
        return

    result = svc.upload_resume(path, blind_mode=blind_mode)
    console.print_json(json.dumps(result))


@app.command("score")
def score(candidate_id: int, job_id: int) -> None:
    """Score candidate for job."""
    result = get_service().score(candidate_id=candidate_id, job_id=job_id)
    console.print_json(result.model_dump_json())


@app.command("compare")
def compare(job_id: int, candidate_ids: list[int]) -> None:
    """Compare candidates for same job."""
    result = get_service().compare(job_id=job_id, candidate_ids=candidate_ids)
    console.print_json(json.dumps({"results": result}))


@app.command("interview")
def interview(candidate_id: int, job_id: int) -> None:
    """Generate interview question set."""
    result = get_service().generate_interview(candidate_id=candidate_id, job_id=job_id)
    console.print_json(json.dumps(result))


@app.command("search")
def search(query: str, top_k: int = 10) -> None:
    """Semantic recruiter search."""
    result = get_service().search(query=query, top_k=top_k)
    console.print_json(json.dumps(result))


@app.command("report")
def report(candidate_id: int, job_id: int, output_dir: str = "outputs/reports") -> None:
    """Export hiring report artifacts."""
    result = get_service().generate_report(candidate_id=candidate_id, job_id=job_id, output_dir=output_dir)
    console.print_json(json.dumps(result))


@app.command("dashboard")
def dashboard() -> None:
    """Launch Streamlit dashboard."""
    typer.echo("Run: streamlit run src/resume_ai/ui/app.py")


if __name__ == "__main__":
    app()
