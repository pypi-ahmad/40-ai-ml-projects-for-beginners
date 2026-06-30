"""Orchestration layer for full local RAG project workflow."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from rag_system.advanced_retrieval import AdvancedRetriever
from rag_system.chunking import ChunkingResult, ChunkingStrategy, TextChunker
from rag_system.config import RAGConfig
from rag_system.data import prepare_dataset_artifacts
from rag_system.diagnostics import embedding_integrity_report, index_integrity_report, retrieval_diagnostics
from rag_system.embeddings import EmbeddingEngine
from rag_system.evaluation import RAGEvaluator
from rag_system.generation import GenerationEngine, RAGPipeline
from rag_system.retrieval import RetrievalEngine
from rag_system.types import ChunkRecord, DocumentRecord, QueryRecord
from rag_system.visualization import generate_all_core_visuals

logger = logging.getLogger(__name__)


class ProjectRunner:
    """End-to-end orchestrator used by scripts/notebooks/UI."""

    def __init__(self, config: RAGConfig) -> None:
        self.config = config.apply_profile()

        self.embedding_engine = EmbeddingEngine(
            model_name=self.config.embedding_model,
            host=self.config.ollama_host,
        )
        self.retrieval_engine = RetrievalEngine(
            collection_name="rag_documents_v1",
            persist_directory=str(self.config.chroma_dir),
            embedding_engine=self.embedding_engine,
            default_top_k=self.config.top_k,
        )
        self.generation_engine = GenerationEngine(
            model_name=self.config.generator_model,
            host=self.config.ollama_host,
        )
        self.judge_engine = GenerationEngine(
            model_name=self.config.judge_model,
            host=self.config.ollama_host,
            temperature=0.0,
            max_tokens=300,
        )

        self.pipeline = RAGPipeline(
            retrieval_engine=self.retrieval_engine,
            generation_engine=self.generation_engine,
            min_relevance_score=self.config.min_relevance_score,
            abstain_threshold=self.config.abstain_threshold,
        )
        self.advanced_retriever = AdvancedRetriever(
            retrieval_engine=self.retrieval_engine,
            generation_engine=self.generation_engine,
        )
        self.evaluator = RAGEvaluator(
            pipeline=self.pipeline,
            judge_engine=self.judge_engine,
        )

    def ingest_dataset(self, force_rebuild: bool = False) -> dict[str, Any]:
        """Download and preprocess dataset artifacts."""
        bundle = prepare_dataset_artifacts(self.config, force_rebuild=force_rebuild)
        return bundle

    def run_chunking(
        self,
        documents: list[DocumentRecord],
        strategy: ChunkingStrategy,
        max_documents: int | None = None,
    ) -> ChunkingResult:
        """Chunk documents using selected strategy."""
        subset = documents[:max_documents] if max_documents else documents
        chunker = TextChunker(
            strategy=strategy,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            parent_chunk_size=self.config.parent_chunk_size,
            parent_chunk_overlap=self.config.parent_chunk_overlap,
            embedding_engine=self.embedding_engine,
        )
        return chunker.chunk_documents(subset)

    def index_chunks(self, chunks: list[ChunkRecord], reset: bool = True) -> int:
        """Index chunk set into ChromaDB."""
        if reset:
            self.retrieval_engine.clear()
        return self.retrieval_engine.index_chunks(chunks)

    def run_evaluation(
        self,
        queries: list[QueryRecord],
        output_dir: Path | None = None,
        hallucination_limit: int | None = None,
    ) -> dict[str, Any]:
        """Run retrieval + generation + judge + hallucination suite."""
        output_dir = output_dir or (self.config.data_dir / "artifacts")
        output_dir.mkdir(parents=True, exist_ok=True)

        bundle, frames = self.evaluator.run_full_evaluation(
            queries=queries,
            top_k=self.config.top_k,
            retrieval_limit=self.config.retrieval_eval_queries,
            generation_limit=self.config.generation_eval_queries,
            judge_limit=self.config.judge_eval_queries,
            run_judge=self.config.enable_judge_eval,
        )

        resolved_hallucination_limit = hallucination_limit
        if resolved_hallucination_limit is None:
            resolved_hallucination_limit = min(self.config.judge_eval_queries, 300)

        hallucination_df = self.evaluator.evaluate_hallucination_reduction(
            queries=queries,
            max_queries=resolved_hallucination_limit,
        )
        frames["hallucination"] = hallucination_df

        retrieval_diag = pd.DataFrame(
            retrieval_diagnostics(
                retrieval_engine=self.retrieval_engine,
                queries=queries,
                top_k=self.config.top_k,
                min_relevance_score=self.config.min_relevance_score,
                max_queries=min(500, len(queries)),
            )
        )
        frames["retrieval_diagnostics"] = retrieval_diag

        # Persist summary and row-level outputs for reproducibility.
        summary_path = output_dir / "evaluation_summary.json"
        frames_dir = output_dir / "tables"
        frames_dir.mkdir(parents=True, exist_ok=True)

        summary_payload = {
            "retrieval": asdict(bundle.retrieval),
            "generation": asdict(bundle.generation),
            "judge": asdict(bundle.judge),
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        for name, frame in frames.items():
            frame.to_parquet(frames_dir / f"{name}.parquet", index=False)
            frame.to_csv(frames_dir / f"{name}.csv", index=False)

        visuals = generate_all_core_visuals(
            doc_df=pd.read_parquet(self.config.data_dir / "processed" / "documents.parquet"),
            retrieval_summary=asdict(bundle.retrieval),
            generation_summary=asdict(bundle.generation),
            hallucination_df=hallucination_df,
            output_dir=output_dir / "figures",
        )

        return {
            "bundle": bundle,
            "frames": frames,
            "summary_path": summary_path,
            "visual_paths": visuals,
        }

    def run_full(self) -> dict[str, Any]:
        """Execute complete pipeline from dataset ingestion to evaluation."""
        bundle = self.ingest_dataset()
        documents: list[DocumentRecord] = bundle["documents"]
        queries: list[QueryRecord] = bundle["queries"]

        chunking = self.run_chunking(
            documents=documents,
            strategy=ChunkingStrategy.RECURSIVE,
            max_documents=None,
        )
        indexed = self.index_chunks(chunking.chunks, reset=True)
        evaluation = self.run_evaluation(queries=queries)
        embedding_report = embedding_integrity_report(
            embedding_engine=self.embedding_engine,
            texts=[chunk.text for chunk in chunking.chunks[:128]],
            batch_size=16,
        )
        index_report = index_integrity_report(
            retrieval_engine=self.retrieval_engine,
            expected_chunks=chunking.chunks,
        )

        return {
            "documents": len(documents),
            "queries": len(queries),
            "leakage_audit": bundle.get("leakage_audit", {}),
            "chunks": len(chunking.chunks),
            "indexed": indexed,
            "embedding_integrity": embedding_report,
            "index_integrity": index_report,
            "evaluation": evaluation,
        }
