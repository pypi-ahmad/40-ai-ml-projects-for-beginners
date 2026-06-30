"""Streamlit UI for local persistent RAG application."""

from __future__ import annotations

import asyncio
import time

import pandas as pd
import streamlit as st

from local_rag.app import AppRuntime, load_settings
from local_rag.corpus import corpus_stats, write_corpus_manifest
from local_rag.corpus_bootstrap import CorpusBootstrapper
from local_rag.indexing import report_to_dict

st.set_page_config(
    page_title="Local RAG (Ollama + Chroma)",
    page_icon="🧠",
    layout="wide",
)

settings = load_settings()
settings.ensure_directories()


@st.cache_resource
def get_runtime(model: str, profile: str) -> AppRuntime:
    scoped = load_settings()
    scoped.corpus_profile = "quickstart" if profile == "quickstart" else "full"
    return AppRuntime(scoped, generation_model=model)


def save_uploaded_files(files: list[st.runtime.uploaded_file_manager.UploadedFile]) -> int:
    saved = 0
    upload_dir = settings.documents_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    for uploaded in files:
        target = upload_dir / uploaded.name
        target.write_bytes(uploaded.getvalue())
        saved += 1
    return saved


st.title("Local Enterprise RAG Assistant")
st.caption(
    "Fully local RAG with Ollama (`qwen3-embedding:4b`, `qwen3.5:4b`) + ChromaDB persistence"
)

with st.sidebar:
    st.header("Configuration")
    corpus_profile = st.radio(
        "Corpus profile",
        options=["full", "quickstart"],
        index=0,
        horizontal=True,
    )
    generation_model = st.selectbox(
        "Generation model",
        options=[settings.generation_model, settings.judge_model],
        index=0,
    )
    top_k = st.select_slider("Top-K", options=[3, 5, 10], value=settings.retrieval.default_k)
    stream_output = st.toggle("Stream output", value=settings.generation_streaming_default)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    max_tokens = st.number_input("Max tokens", min_value=64, max_value=2048, value=512, step=64)

    st.divider()
    st.subheader("Index Controls")
    chunk_size = st.selectbox("Chunk size", settings.chunking.chunk_sizes, index=2)
    chunk_overlap = st.selectbox("Chunk overlap", settings.chunking.chunk_overlaps, index=2)

    st.divider()
    st.subheader("Retrieval Filters")
    source_type = st.selectbox("Source type", options=["all", "pdf", "markdown", "txt"], index=0)
    section = st.text_input("Section (optional)", value="")
    show_health = st.button("Refresh Health Panel", use_container_width=True)

runtime = get_runtime(generation_model, corpus_profile)

health_report = runtime.vector_store.integrity_report()
if show_health:
    st.sidebar.json(
        {
            "collection": runtime.settings.active_collection_name,
            "manifest": str(runtime.settings.active_index_manifest_path),
            "integrity": health_report,
        }
    )

col_a, col_b, col_c = st.columns(3)
with col_a:
    if st.button("Bootstrap Corpus", use_container_width=True):
        with st.spinner("Downloading corpus..."):
            bootstrapper = CorpusBootstrapper(settings.documents_dir)
            asyncio.run(bootstrapper.bootstrap())
            stats = corpus_stats(settings.documents_dir)
            write_corpus_manifest(settings.corpus_manifest_path, stats)
        st.success("Corpus bootstrap completed.")

with col_b:
    if st.button("Build / Update Index", use_container_width=True):
        try:
            with st.spinner("Indexing documents..."):
                report = runtime.indexer.build_or_update(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    force_rebuild=False,
                )
            st.success("Index update finished.")
            st.json(report_to_dict(report))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Index update failed: {exc}")

with col_c:
    if st.button("Rebuild Index", use_container_width=True):
        try:
            with st.spinner("Rebuilding index..."):
                report = runtime.indexer.build_or_update(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    force_rebuild=True,
                )
            st.success("Full rebuild finished.")
            st.json(report_to_dict(report))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Index rebuild failed: {exc}")

st.divider()

upload_files = st.file_uploader(
    "Upload PDF/TXT/MD files",
    type=["pdf", "txt", "md", "markdown"],
    accept_multiple_files=True,
)
if upload_files and st.button("Save Uploaded Documents"):
    count = save_uploaded_files(upload_files)
    st.success(f"Saved {count} files to data/documents/uploads")

st.divider()

col_d, col_e = st.columns([2, 1])
with col_d:
    st.subheader("Ask Questions")
    question = st.text_input("Query", placeholder="Ask about Linux docs in local corpus")
    run_query = st.button("Run RAG Query", type="primary")

with col_e:
    st.subheader("Collection")
    st.metric("Vectors", runtime.vector_store.count())
    if st.button("Clear Database"):
        runtime.vector_store.reset()
        manifest = runtime.settings.active_index_manifest_path
        if manifest.exists():
            manifest.unlink()
        st.warning("Vector database cleared. Profile manifest removed.")

response = None
if run_query and question.strip():
    filters: dict[str, str] | None = None
    if source_type != "all" or section.strip():
        filters = {}
        if source_type != "all":
            filters["source_type"] = source_type
        if section.strip():
            filters["section"] = section.strip()

    try:
        if stream_output:
            with st.spinner("Retrieving and streaming..."):
                stream_iter, session_state = runtime.pipeline.ask_stream(
                    question,
                    top_k=top_k,
                    filters=filters,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                )
                placeholder = st.empty()
                tokens: list[str] = []
                started_generation = time.perf_counter()
                for token in stream_iter:
                    tokens.append(token)
                    placeholder.markdown("### Answer\n" + "".join(tokens))
                generation_ms = (time.perf_counter() - started_generation) * 1000
                response = runtime.pipeline.finalize_stream(
                    session=session_state,
                    answer="".join(tokens),
                    generation_ms=generation_ms,
                )
        else:
            with st.spinner("Retrieving and generating..."):
                response = runtime.pipeline.ask(
                    question,
                    top_k=top_k,
                    filters=filters,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                )
            st.markdown("### Answer")
            st.write(response.answer)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Query failed: {exc}")
        response = None

if run_query and question.strip() and response is not None:

    st.markdown("### Latency Breakdown (ms)")
    st.json(
        {
            "embedding_ms": round(response.timings.embedding_ms, 2),
            "retrieval_ms": round(response.timings.retrieval_ms, 2),
            "prompt_ms": round(response.timings.prompt_ms, 2),
            "generation_ms": round(response.timings.generation_ms, 2),
            "total_ms": round(response.timings.total_ms, 2),
        }
    )

    st.markdown("### Retrieval Stats")
    st.json(
        {
            "answer_length": len(response.answer.split()),
            "citation_count": len(response.citations),
            "filters": filters or {},
        }
    )

    st.markdown("### Citations")
    if response.citations:
        st.dataframe(pd.DataFrame(response.citations), use_container_width=True)
    else:
        st.info("No citations found.")

    st.markdown("### Retrieved Chunks")
    for idx, hit in enumerate(response.retrieved, start=1):
        source_path = hit.metadata.get("source_path", "unknown")
        label = f"Hit {idx} | score={hit.score:.4f} | {source_path}"
        with st.expander(label):
            st.markdown("**Metadata**")
            st.json(hit.metadata)
            st.markdown("**Text**")
            st.write(hit.text)

st.divider()
st.subheader("Indexed Documents")
indexed_docs = runtime.vector_store.list_indexed_documents()
if indexed_docs:
    st.dataframe(pd.DataFrame(indexed_docs), use_container_width=True)
else:
    st.info("No indexed documents yet.")

st.caption("All inference and retrieval run locally. No external LLM API calls.")
