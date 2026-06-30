"""Gradio interface for transparent local RAG system."""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr
import pandas as pd

from rag_system import ChunkingStrategy, ProjectRunner, RAGConfig
from rag_system.utils import assert_ollama_available

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGAppState:
    """Mutable app state so indexing runs once per session."""

    def __init__(self) -> None:
        self.runner = ProjectRunner(RAGConfig(project_root=Path("."), profile="balanced"))
        self.initialized = False

    def initialize(self) -> str:
        """Prepare dataset and vector index for app queries."""
        if self.initialized:
            stats = self.runner.retrieval_engine.get_collection_stats()
            return f"Already initialized. Indexed chunks: {stats['count']}"

        try:
            assert_ollama_available(self.runner.config.ollama_host)
        except RuntimeError as exc:
            return f"Initialization failed: {exc}"

        bundle = self.runner.ingest_dataset()
        documents = bundle["documents"]
        chunking = self.runner.run_chunking(
            documents=documents,
            strategy=ChunkingStrategy.RECURSIVE,
            max_documents=5000,
        )
        self.runner.index_chunks(chunking.chunks, reset=True)
        self.initialized = True
        return f"Initialized with {len(chunking.chunks)} chunks from {len(documents)} documents."


state = RAGAppState()


def _query_rag(query: str, top_k: int) -> tuple[str, pd.DataFrame, str, str, str]:
    if not query.strip():
        return "Enter a query.", pd.DataFrame(), "", "", ""

    if not state.initialized:
        msg = "Initialize index first using the setup button."
        return msg, pd.DataFrame(), "", "", ""

    rag_result = state.runner.pipeline.answer(query=query, top_k=top_k)
    baseline = state.runner.generation_engine.generate_baseline_answer(query=query)

    rows = []
    for idx, chunk in enumerate(rag_result.retrieved_chunks, start=1):
        rows.append(
            {
                "rank": idx,
                "doc_id": chunk.doc_id,
                "score": round(chunk.score, 4),
                "distance": round(chunk.distance, 4),
                "title": chunk.metadata.get("title", "unknown"),
                "chunk_preview": chunk.text[:220].replace("\n", " "),
            }
        )

    retrieval_df = pd.DataFrame(rows)
    context_view = rag_result.context
    top_score = rag_result.retrieved_chunks[0].score if rag_result.retrieved_chunks else 0.0
    diag = (
        "### Retrieval Diagnostics\n"
        f"- retrieved_chunks: {len(rag_result.retrieved_chunks)}\n"
        f"- top_score: {top_score:.3f}\n"
        f"- abstained: {rag_result.abstained}\n"
        f"- abstain_reason: {rag_result.abstain_reason or 'n/a'}"
    )
    comparison = (
        "### RAG vs No-RAG\n"
        f"**RAG latency:** {rag_result.total_latency_s:.2f}s\n\n"
        f"**No-RAG latency:** {baseline['latency_s']:.2f}s\n\n"
        "**No-RAG answer:**\n"
        f"{baseline['text']}"
    )
    return rag_result.answer, retrieval_df, context_view, comparison, diag


def build_app() -> gr.Blocks:
    """Create full Gradio interface."""
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # Build Your First RAG System - Transparent Demo
            Local models: `qwen3-embedding:4b` + `qwen3.5:4b`.
            Shows retrieved chunks, similarity scores, and answer grounding.
            """
        )

        with gr.Row():
            init_btn = gr.Button("Initialize Dataset + Index", variant="primary")
            init_status = gr.Textbox(label="Setup Status")

        query_input = gr.Textbox(label="User Query", placeholder="Ask something about SQuAD contexts...")
        top_k = gr.Slider(label="Top-K Retrieval", minimum=2, maximum=12, value=6, step=1)
        submit_btn = gr.Button("Run RAG", variant="primary")

        answer_box = gr.Markdown(label="Generated Answer")
        retrieval_table = gr.Dataframe(label="Retrieved Chunks + Scores")
        context_box = gr.Textbox(label="Retrieved Context (for prompt grounding)", lines=10)
        compare_box = gr.Markdown(label="RAG vs No-RAG")
        diagnostics_box = gr.Markdown(label="Diagnostics")

        init_btn.click(fn=state.initialize, outputs=[init_status])
        submit_btn.click(
            fn=_query_rag,
            inputs=[query_input, top_k],
            outputs=[answer_box, retrieval_table, context_box, compare_box, diagnostics_box],
        )

    return demo


def main() -> None:
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
