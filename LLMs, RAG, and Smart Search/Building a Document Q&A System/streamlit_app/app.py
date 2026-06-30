"""Streamlit UI for enterprise document Q&A application."""

from __future__ import annotations

import asyncio
import time

import pandas as pd
import streamlit as st

from local_rag.app import AppRuntime, load_settings
from local_rag.corpus import corpus_stats, write_corpus_manifest
from local_rag.corpus_bootstrap import CorpusBootstrapper
from local_rag.indexing import report_to_dict
from local_rag.retriever import RetrievalStrategy

st.set_page_config(
    page_title="Enterprise Document Q&A",
    page_icon="📚",
    layout="wide",
)

settings = load_settings()
settings.ensure_directories()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "last_response" not in st.session_state:
    st.session_state["last_response"] = None


@st.cache_resource
def get_runtime(model: str, profile: str) -> AppRuntime:
    scoped = load_settings()
    scoped.corpus_profile = "quickstart" if profile == "quickstart" else "full"
    return AppRuntime(scoped, generation_model=model)


def save_uploaded_files(
    runtime: AppRuntime,
    files: list[st.runtime.uploaded_file_manager.UploadedFile],
) -> int:
    saved = 0
    upload_dir = runtime.settings.active_documents_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    for uploaded in files:
        target = upload_dir / uploaded.name
        target.write_bytes(uploaded.getvalue())
        saved += 1
    return saved


st.title("Enterprise Document Intelligence Assistant")
st.caption(
    "Local-only RAG stack: qwen3-embedding:4b + qwen3.5:4b + granite4.1:3b + ChromaDB"
)

with st.sidebar:
    st.header("Runtime")
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

    st.subheader("Retrieval")
    strategy: RetrievalStrategy = st.selectbox(
        "Strategy",
        options=["hybrid", "vector", "keyword"],
        index=0,
    )
    top_k = st.select_slider("Top-K", options=[1, 3, 5, 10], value=settings.retrieval.default_k)
    source_type = st.selectbox("Source type", options=["all", "pdf", "markdown", "txt"], index=0)
    domain_filter = st.selectbox(
        "Domain",
        options=["all", "technical", "policy", "research", "finance"],
        index=0,
    )
    section_filter = st.text_input("Section filter", value="")

    st.subheader("Generation")
    prompt_template = st.selectbox(
        "Prompt template",
        options=[
            "enterprise_qa",
            "strict_grounding",
            "citation_focus",
            "legal_qa",
            "technical_qa",
            "unknown_safe",
        ],
        index=0,
    )
    stream_output = st.toggle("Stream output", value=settings.generation_streaming_default)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    max_tokens = st.number_input("Max tokens", min_value=64, max_value=2048, value=700, step=64)

runtime = get_runtime(generation_model, corpus_profile)


def _build_filters() -> dict[str, str] | None:
    payload: dict[str, str] = {}
    if source_type != "all":
        payload["source_type"] = source_type
    if domain_filter != "all":
        payload["domain"] = domain_filter
    if section_filter.strip():
        payload["section"] = section_filter.strip()
    return payload or None


def _as_chat_messages(history: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in history[-20:]:
        rows.append({"role": "user", "content": row["question"]})
        rows.append({"role": "assistant", "content": row["answer"]})
    return rows


doc_tab, index_tab, qa_tab, analytics_tab = st.tabs([
    "Document Management",
    "Indexing",
    "Q&A",
    "Analytics",
])

with doc_tab:
    st.subheader("Upload and Manage Documents")
    upload_files = st.file_uploader(
        "Upload PDF/TXT/MD files",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
    )
    col_u1, col_u2, col_u3 = st.columns(3)
    with col_u1:
        if upload_files and st.button("Save Uploads", use_container_width=True):
            count = save_uploaded_files(runtime, upload_files)
            runtime.document_manager.write_catalog()
            st.success(f"Saved {count} files")
    with col_u2:
        if st.button("Refresh Catalog", use_container_width=True):
            runtime.document_manager.write_catalog()
            st.success("Catalog refreshed")
    with col_u3:
        if st.button("Bootstrap Public Corpus", use_container_width=True):
            with st.spinner("Downloading corpus sources..."):
                bootstrapper = CorpusBootstrapper(runtime.settings.documents_dir)
                asyncio.run(bootstrapper.bootstrap())
                stats = corpus_stats(runtime.settings.documents_dir)
                write_corpus_manifest(runtime.settings.corpus_manifest_path, stats)
            st.success("Corpus bootstrap done")

    docs = runtime.document_manager.list_documents()
    if docs:
        doc_df = pd.DataFrame([row.__dict__ for row in docs])
        st.dataframe(doc_df, use_container_width=True)

        st.markdown("Delete document by `source_path`:")
        target_delete = st.text_input("Delete source_path")
        if st.button("Delete Document") and target_delete.strip():
            if runtime.document_manager.delete_document(target_delete.strip()):
                st.success("Document deleted")
            else:
                st.warning("Document not found")
    else:
        st.info("No managed documents found.")

with index_tab:
    st.subheader("Index Controls")
    chunk_size = st.selectbox("Chunk size", settings.chunking.chunk_sizes, index=2)
    chunk_overlap = st.selectbox("Chunk overlap", settings.chunking.chunk_overlaps, index=2)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Build / Update Index", use_container_width=True):
            with st.spinner("Indexing..."):
                report = runtime.indexer.build_or_update(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    force_rebuild=False,
                )
            st.success("Incremental indexing finished")
            st.json(report_to_dict(report))
    with c2:
        if st.button("Force Rebuild", use_container_width=True):
            with st.spinner("Rebuilding index..."):
                report = runtime.indexer.build_or_update(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    force_rebuild=True,
                )
            st.success("Full rebuild finished")
            st.json(report_to_dict(report))
    with c3:
        if st.button("Clear Collection", use_container_width=True):
            runtime.vector_store.reset()
            manifest = runtime.settings.active_index_manifest_path
            if manifest.exists():
                manifest.unlink()
            st.warning("Collection cleared")

    st.metric("Vector Count", runtime.vector_store.count())
    indexed_docs = runtime.vector_store.list_indexed_documents()
    if indexed_docs:
        st.dataframe(pd.DataFrame(indexed_docs), use_container_width=True)

with qa_tab:
    st.subheader("Ask Questions Across Documents")
    question = st.text_area("Question", height=100)

    q1, q2 = st.columns([1, 1])
    with q1:
        run_query = st.button("Run Query", type="primary", use_container_width=True)
    with q2:
        if st.button("Reset Conversation", use_container_width=True):
            st.session_state["chat_history"] = []
            st.success("Conversation reset")

    if run_query and question.strip():
        filters = _build_filters()
        history = _as_chat_messages(st.session_state["chat_history"])

        if stream_output:
            with st.spinner("Retrieving and streaming answer..."):
                stream_iter, session_state = runtime.pipeline.ask_stream(
                    query=question,
                    top_k=top_k,
                    filters=filters,
                    strategy=strategy,
                    prompt_template=prompt_template,  # type: ignore[arg-type]
                    conversation_history=history,
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
                    query=question,
                    top_k=top_k,
                    filters=filters,
                    strategy=strategy,
                    prompt_template=prompt_template,  # type: ignore[arg-type]
                    conversation_history=history,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                )
            st.markdown("### Answer")
            st.write(response.answer)

        st.session_state["chat_history"].append(
            {
                "question": question,
                "answer": response.answer,
                "strategy": response.retrieval_strategy,
            }
        )
        st.session_state["last_response"] = response

    response = st.session_state.get("last_response")
    if response is not None:
        st.markdown("### Latency")
        latency_payload = {
            "embedding_ms": round(response.timings.embedding_ms, 2),
            "retrieval_ms": round(response.timings.retrieval_ms, 2),
            "prompt_ms": round(response.timings.prompt_ms, 2),
            "generation_ms": round(response.timings.generation_ms, 2),
            "total_ms": round(response.timings.total_ms, 2),
            "strategy": response.retrieval_strategy,
        }
        st.json(latency_payload)

        st.markdown("### Citations")
        if response.citations:
            st.dataframe(pd.DataFrame(response.citations), use_container_width=True)
        else:
            st.info("No citations available.")

        st.markdown("### Retrieved Chunks")
        for idx, hit in enumerate(response.retrieved, start=1):
            doc_name = hit.metadata.get("document_name", "unknown")
            label = f"Hit {idx} | score={hit.score:.4f} | {doc_name}"
            with st.expander(label):
                st.json(hit.metadata)
                st.write(hit.text)

with analytics_tab:
    st.subheader("Collection and Query Analytics")
    docs = runtime.document_manager.list_documents()
    st.metric("Documents", len(docs))
    st.metric("Vectors", runtime.vector_store.count())

    if docs:
        frame = pd.DataFrame([row.__dict__ for row in docs])
        by_suffix = frame.groupby("file_suffix")["document_name"].count().reset_index()
        by_suffix.columns = ["file_suffix", "count"]
        st.bar_chart(by_suffix.set_index("file_suffix"))

    history = st.session_state["chat_history"]
    if history:
        st.markdown("### Conversation History")
        st.dataframe(pd.DataFrame(history), use_container_width=True)

st.caption("All indexing, retrieval, generation, and evaluation run locally.")
