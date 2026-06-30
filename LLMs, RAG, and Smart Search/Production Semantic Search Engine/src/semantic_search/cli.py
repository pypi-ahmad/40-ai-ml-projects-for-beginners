"""Typer CLI for semantic search pipeline."""

from __future__ import annotations

from dataclasses import asdict
import json
import subprocess
from pathlib import Path

import typer

from semantic_search.benchmark import BenchmarkRunner, estimate_dir_size_bytes
from semantic_search.config import load_config
from semantic_search.evaluation import load_evaluation_cases
from semantic_search.logging_utils import configure_logging
from semantic_search.schemas import SearchRequest
from semantic_search.schemas import DocumentRecord
from semantic_search.service import SemanticSearchService

app = typer.Typer(help="Production Semantic Search Engine CLI")


def _service(config_path: str | None = None) -> SemanticSearchService:
    config = load_config(config_path)
    configure_logging(config)
    return SemanticSearchService(config)


@app.command("check-models")
def check_models(config_path: str | None = typer.Option(default=None, help="Path to YAML config")) -> None:
    """Ensure required Ollama models are installed."""
    service = _service(config_path)
    service.ensure_ollama_models(include_optional_qwen=True)
    typer.echo("Ollama models ready.")


@app.command("ingest")
def ingest(
    source: str = typer.Option("huggingface", help="huggingface|folder"),
    folder_path: str = typer.Option("data/raw", help="Folder path for local ingestion"),
    config_path: str | None = typer.Option(default=None, help="Path to YAML config"),
) -> None:
    """Ingest documents into canonical JSONL."""
    service = _service(config_path)
    service.ensure_ollama_models(include_optional_qwen=False)
    if source == "huggingface":
        docs = service.ingest_huggingface()
    else:
        docs = service.ingest_folder(folder_path)
    typer.echo(f"Ingested {len(docs)} documents.")


@app.command("chunk")
def chunk(
    strategy: str | None = typer.Option(default=None),
    chunk_size: int | None = typer.Option(default=None),
    chunk_overlap: int | None = typer.Option(default=None),
    config_path: str | None = typer.Option(default=None),
) -> None:
    """Generate chunks from ingested documents."""
    service = _service(config_path)
    chunks = service.chunk_documents(strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    typer.echo(f"Generated {len(chunks)} chunks.")


@app.command("index")
def index(
    model: str = typer.Option("primary", help="primary|all-MiniLM-L6-v2|nomic-embed-text"),
    config_path: str | None = typer.Option(default=None),
) -> None:
    """Build Chroma and FAISS indexes."""
    service = _service(config_path)
    cfg = service.config

    selected = cfg.embedding.primary
    if model != "primary":
        lookup = {m.model_name: m for m in cfg.embedding.comparisons + [cfg.embedding.primary]}
        if model not in lookup:
            raise typer.BadParameter(f"Unknown model: {model}")
        selected = lookup[model]

    if selected.provider == "ollama":
        service.ensure_ollama_models(include_optional_qwen=False)

    service.build_indexes(selected)
    typer.echo(f"Indexes built using {selected.model_name}.")


@app.command("incremental-index")
def incremental_index(
    input_jsonl: str = typer.Option(..., help="Path to JSONL of new DocumentRecord rows"),
    config_path: str | None = typer.Option(default=None),
) -> None:
    """Incrementally index new documents with deduplication."""
    service = _service(config_path)
    if not service.documents:
        docs_path = Path(service.config.paths["processed_data_dir"]) / "documents.jsonl"
        if docs_path.exists():
            service.load_documents(docs_path)
    if not service.chunks:
        chunks_path = Path(service.config.paths["processed_data_dir"]) / "chunks.jsonl"
        if chunks_path.exists():
            service.load_chunks(chunks_path)
    if service.embedding_backend is None:
        service.build_indexes(service.config.embedding.primary)

    new_docs: list[DocumentRecord] = []
    with Path(input_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            new_docs.append(DocumentRecord.model_validate_json(line))
    added = service.incremental_index_documents(new_docs)
    typer.echo(f"Incremental indexing complete. Added documents: {added}")


@app.command("delete-docs")
def delete_docs(
    document_ids: str = typer.Option(..., help="Comma separated document IDs"),
    config_path: str | None = typer.Option(default=None),
) -> None:
    """Delete documents and rebuild retrieval indexes."""
    service = _service(config_path)
    if not service.documents:
        docs_path = Path(service.config.paths["processed_data_dir"]) / "documents.jsonl"
        if docs_path.exists():
            service.load_documents(docs_path)
    if not service.chunks:
        chunks_path = Path(service.config.paths["processed_data_dir"]) / "chunks.jsonl"
        if chunks_path.exists():
            service.load_chunks(chunks_path)
    if service.embedding_backend is None:
        service.build_indexes(service.config.embedding.primary)
    removed = service.delete_documents([doc_id.strip() for doc_id in document_ids.split(",") if doc_id.strip()])
    typer.echo(f"Removed documents: {removed}")


@app.command("search")
def search(
    query: str = typer.Argument(..., help="Search query text"),
    mode: str = typer.Option("hybrid", help="semantic|lexical|hybrid"),
    top_k: int = typer.Option(10, help="Top K results"),
    rerank: bool = typer.Option(True, help="Apply reranking"),
    filters: str | None = typer.Option(None, help="JSON filters, e.g. '{\"category\": \"TECH\"}'"),
    config_path: str | None = typer.Option(default=None),
) -> None:
    """Execute search query."""
    service = _service(config_path)
    request = SearchRequest(
        query=query,
        mode=mode,
        top_k=top_k,
        rerank=rerank,
        filters=json.loads(filters) if filters else {},
    )
    response = service.search(request)
    typer.echo(response.model_dump_json(indent=2))


@app.command("evaluate")
def evaluate(
    cases_path: str = typer.Option("data/processed/evaluation_queries.jsonl"),
    output_path: str = typer.Option("artifacts/reports/evaluation_results.json"),
    mode: str = typer.Option("hybrid"),
    config_path: str | None = typer.Option(default=None),
) -> None:
    """Run offline retrieval evaluation."""
    service = _service(config_path)
    output = service.evaluate(cases_path=cases_path, output_path=output_path, mode=mode)
    typer.echo(output.summary.model_dump_json(indent=2))


@app.command("benchmark")
def benchmark(
    cases_path: str = typer.Option("data/processed/evaluation_queries.jsonl"),
    config_path: str | None = typer.Option(default=None),
) -> None:
    """Run embedding and retrieval benchmarks."""
    service = _service(config_path)
    if not service.documents:
        service.load_documents()
    if not service.chunks:
        service.load_chunks()
    service._ensure_runtime()

    benchmarker = BenchmarkRunner(service.config)
    sample_texts = [doc.text for doc in service.documents[:200]]
    emb_rows = benchmarker.benchmark_embeddings(sample_texts)

    cases = load_evaluation_cases(cases_path)
    run = benchmarker.benchmark_retrieval(
        system_name="hybrid-rerank",
        evaluation_cases=cases,
        search_fn=lambda q: service.search(
            SearchRequest(query=q, mode="hybrid", top_k=10, rerank=True, use_cache=False)
        ),
        index_size_bytes=estimate_dir_size_bytes(service.config.paths["chroma_dir"]),
        embedding_dim=service.embedding_backend.dimension if service.embedding_backend else 0,
        chunk_strategy=service.config.chunking.strategy,
        chunk_size=service.config.chunking.chunk_size,
        chunk_overlap=service.config.chunking.chunk_overlap,
        retriever_mode="hybrid",
        reranker_enabled=True,
        embedding_model=service.config.embedding.primary.model_name,
    )

    report = {
        "embedding_benchmark": [asdict(row) for row in emb_rows],
        "retrieval_benchmark": run.model_dump(),
    }
    output_path = Path(service.config.paths["reports_dir"]) / "benchmark_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


@app.command("app")
def run_app() -> None:
    """Launch Streamlit UI."""
    subprocess.run(["streamlit", "run", "app.py"], check=True)


if __name__ == "__main__":
    app()
