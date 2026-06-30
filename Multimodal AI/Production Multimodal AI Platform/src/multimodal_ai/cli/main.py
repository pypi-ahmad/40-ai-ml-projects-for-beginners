"""Typer CLI entrypoint for multimodal-ai."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from multimodal_ai.domain import InputPayload, RequestEnvelope, TraceContext
from multimodal_ai.services.bootstrap import build_platform_service
from multimodal_ai.services.platform_service import PlatformService

app = typer.Typer(help="Multimodal AI platform CLI")
console = Console()


def _service() -> PlatformService:
    return build_platform_service()


def _print_json(payload: dict) -> None:
    console.print_json(json.dumps(payload, default=str))


@app.command()
def caption(image: Path, style: str = typer.Option("detailed", help="Caption style")) -> None:
    """Generate image caption."""

    service = _service()
    request = RequestEnvelope(
        input=InputPayload(image_path=str(image)),
        options={"style": style},
        trace=TraceContext(source="cli"),
    )
    response = service.caption(request).model_dump()
    _print_json(response)


@app.command()
def search(
    query: str,
    modality: str = typer.Option("image", help="Search modality"),
    top_k: int = typer.Option(5, help="Top K hits"),
) -> None:
    """Run semantic search."""

    service = _service()
    request = RequestEnvelope(
        input=InputPayload(query=query),
        options={"modality": modality, "top_k": top_k},
        trace=TraceContext(source="cli"),
    )
    response = service.search(request).model_dump()
    _print_json(response)


@app.command()
def ocr(file: Path) -> None:
    """Run OCR for image/document."""

    service = _service()
    request = RequestEnvelope(
        input=InputPayload(document_path=str(file)),
        trace=TraceContext(source="cli"),
    )
    response = service.ocr(request).model_dump()
    _print_json(response)


@app.command()
def compare(images: list[Path]) -> None:
    """Compare two or more images."""

    service = _service()
    request = RequestEnvelope(
        input=InputPayload(image_paths=[str(path) for path in images]),
        trace=TraceContext(source="cli"),
    )
    response = service.compare(request).model_dump()
    _print_json(response)


@app.command()
def analyze(
    image: Path | None = typer.Option(None),
    document: Path | None = typer.Option(None),
    question: str = typer.Option("Summarize key insights"),
) -> None:
    """Run comprehensive multimodal analysis."""

    service = _service()
    request = RequestEnvelope(
        input=InputPayload(
            image_path=str(image) if image else None,
            document_path=str(document) if document else None,
            question=question,
        ),
        trace=TraceContext(source="cli"),
    )
    response = service.analyze(request).model_dump()
    _print_json(response)


@app.command()
def dashboard() -> None:
    """Launch Streamlit dashboard."""

    subprocess.run(
        [
            "streamlit",
            "run",
            "src/multimodal_ai/ui/streamlit_app.py",
            "--server.address",
            "0.0.0.0",
            "--server.port",
            "8501",
        ],
        check=False,
    )


@app.command()
def doctor() -> None:
    """Show runtime health and model readiness."""

    service = _service()
    health = service.health().model_dump()

    table = Table(title="Platform Doctor")
    table.add_column("Key")
    table.add_column("Value")
    table.add_row("Status", health.get("status", "unknown"))
    table.add_row("Trace", health.get("trace_id", ""))
    table.add_row("Adapters", json.dumps(health.get("result", {}).get("adapters", {}), indent=2))
    table.add_row("System", json.dumps(health.get("result", {}).get("system", {}), indent=2))
    console.print(table)


if __name__ == "__main__":
    app()
