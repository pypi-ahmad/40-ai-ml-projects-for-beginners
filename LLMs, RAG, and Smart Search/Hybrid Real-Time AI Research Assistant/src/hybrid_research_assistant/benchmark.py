"""Chunking and embedding benchmark utilities."""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from hybrid_research_assistant.chunking import Chunker
from hybrid_research_assistant.embeddings import (
    EmbeddingBenchmark,
    HashingEmbedder,
    benchmark_embedding_model,
    build_embedding_provider,
)
from hybrid_research_assistant.loaders import DocumentLoader
from hybrid_research_assistant.schemas import ChunkingStrategy, DocumentRecord
from hybrid_research_assistant.utils import json_dump


@dataclass(slots=True)
class ChunkingBenchmarkRow:
    strategy: str
    chunk_size: int
    chunk_overlap: int
    total_chunks: int
    avg_chunk_length: float
    latency_ms: float


def benchmark_chunking(
    docs: list[DocumentRecord],
    chunk_sizes: list[int],
    overlaps: list[int],
) -> list[ChunkingBenchmarkRow]:
    """Run chunking grid benchmarks across configured strategies."""

    rows: list[ChunkingBenchmarkRow] = []
    for strategy in [ChunkingStrategy.RECURSIVE, ChunkingStrategy.TOKEN, ChunkingStrategy.SEMANTIC]:
        for chunk_size in chunk_sizes:
            for overlap in overlaps:
                if strategy == ChunkingStrategy.SEMANTIC and overlap != 0:
                    continue
                started = time.perf_counter()
                chunker = Chunker(chunk_size=chunk_size, chunk_overlap=overlap, strategy=strategy)
                chunks = chunker.split_documents(docs)
                elapsed_ms = (time.perf_counter() - started) * 1000
                avg_len = statistics.mean(len(chunk.text) for chunk in chunks) if chunks else 0.0
                rows.append(
                    ChunkingBenchmarkRow(
                        strategy=strategy.value,
                        chunk_size=chunk_size,
                        chunk_overlap=overlap,
                        total_chunks=len(chunks),
                        avg_chunk_length=avg_len,
                        latency_ms=elapsed_ms,
                    )
                )
    return rows


def run_embedding_benchmark(
    model_names: list[str],
    sample_texts: list[str],
    *,
    ollama_host: str,
) -> list[EmbeddingBenchmark]:
    """Run embedding model benchmark table."""

    rows: list[EmbeddingBenchmark] = []
    for model_name in model_names:
        provider = build_embedding_provider(model_name=model_name, ollama_host=ollama_host)
        try:
            rows.append(benchmark_embedding_model(provider=provider, texts=sample_texts))
        except Exception:  # noqa: BLE001
            fallback = HashingEmbedder(model_name=f"hash::{model_name}")
            rows.append(benchmark_embedding_model(provider=fallback, texts=sample_texts))
    return rows


def run_full_benchmark(
    *,
    loader: DocumentLoader,
    chunk_sizes: list[int],
    overlaps: list[int],
    embedding_models: list[str],
    ollama_host: str,
    output_dir: Path,
) -> dict[str, Path]:
    """Run chunking + embedding benchmark and persist artifacts."""

    docs = loader.load_directory()
    if not docs:
        raise ValueError("No documents available for benchmark")

    chunk_rows = benchmark_chunking(docs=docs, chunk_sizes=chunk_sizes, overlaps=overlaps)
    text_samples = [doc.text[:500] for doc in docs[:50]]
    embed_rows = run_embedding_benchmark(
        model_names=embedding_models,
        sample_texts=text_samples,
        ollama_host=ollama_host,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = output_dir / "chunking_benchmark.json"
    embed_path = output_dir / "embedding_benchmark.json"
    json_dump(chunk_path, {"rows": [asdict(row) for row in chunk_rows]})
    json_dump(embed_path, {"rows": [asdict(row) for row in embed_rows]})

    return {"chunking": chunk_path, "embedding": embed_path}
