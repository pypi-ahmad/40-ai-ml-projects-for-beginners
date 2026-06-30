"""Streamlit application for semantic search engine."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from semantic_search.config import load_config
from semantic_search.logging_utils import configure_logging
from semantic_search.schemas import SearchRequest
from semantic_search.service import SemanticSearchService

CONFIG_PATH = Path("config/default.yaml")


@st.cache_resource
def get_service() -> SemanticSearchService:
    config = load_config(CONFIG_PATH)
    configure_logging(config)
    return SemanticSearchService(config)


def _enforce_rate_limit(max_per_minute: int) -> bool:
    now = time.time()
    bucket = st.session_state.setdefault("request_times", [])
    bucket[:] = [ts for ts in bucket if now - ts < 60]
    if len(bucket) >= max_per_minute:
        return False
    bucket.append(now)
    return True


def _highlight(text: str, query: str) -> str:
    if not query.strip():
        return text
    escaped = re.escape(query.strip())
    return re.sub(escaped, lambda m: f"**{m.group(0)}**", text, flags=re.IGNORECASE)


def _init_state() -> None:
    st.session_state.setdefault("recent_searches", [])
    st.session_state.setdefault("search_history", [])
    st.session_state.setdefault("dark_mode", False)


def page_home(service: SemanticSearchService) -> None:
    st.title("Production Semantic Search Engine")
    st.write(
        "Semantic + BM25 hybrid retrieval with ChromaDB, FAISS comparison, reranking, evaluation, and analytics."
    )

    st.subheader("Quick Status")
    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", len(service.documents))
    col2.metric("Chunks", len(service.chunks))
    col3.metric("Embedding Model", service.embedding_model_cfg.model_name)


def page_index_builder(service: SemanticSearchService) -> None:
    st.header("Index Builder")
    config = service.config

    ingest_source = st.selectbox("Ingestion source", ["huggingface", "folder"])
    folder_path = st.text_input("Folder path", value="data/raw")

    if st.button("1) Ingest Documents"):
        if ingest_source == "huggingface":
            docs = service.ingest_huggingface()
        else:
            docs = service.ingest_folder(folder_path)
        st.success(f"Ingested {len(docs)} documents")

    c1, c2, c3 = st.columns(3)
    strategy = c1.selectbox("Chunk strategy", ["recursive", "token", "semantic"], index=0)
    chunk_size = c2.selectbox("Chunk size", [256, 512, 768, 1024], index=1)
    chunk_overlap = c3.selectbox("Chunk overlap", [0, 50, 100, 200], index=1)

    if st.button("2) Chunk Documents"):
        chunks = service.chunk_documents(strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        st.success(f"Generated {len(chunks)} chunks")

    model_options = [config.embedding.primary.model_name] + [m.model_name for m in config.embedding.comparisons]
    selected_model = st.selectbox("Embedding model", model_options)

    if st.button("3) Build Indexes"):
        model_cfg = next(
            m for m in [config.embedding.primary, *config.embedding.comparisons] if m.model_name == selected_model
        )
        if model_cfg.provider == "ollama":
            service.ensure_ollama_models(include_optional_qwen=False)
        service.build_indexes(model_cfg)
        st.success("Indexes built (Chroma + FAISS)")


def page_documents(service: SemanticSearchService) -> None:
    st.header("Documents")
    if not service.documents:
        doc_path = Path(service.config.paths["processed_data_dir"]) / "documents.jsonl"
        if doc_path.exists():
            service.load_documents(doc_path)

    if not service.documents:
        st.info("No documents loaded yet.")
        return

    manifest = service.list_collection_versions()
    if manifest:
        st.subheader("Collection Explorer")
        st.json(manifest)

    df = pd.DataFrame([doc.model_dump() for doc in service.documents])
    st.dataframe(df[["doc_id", "title", "category", "author", "published_date", "language"]], use_container_width=True)


def page_search(service: SemanticSearchService) -> None:
    st.header("Search")
    _init_state()

    q = st.text_input("Search query", value="")
    suggestions = st.session_state["recent_searches"][-5:]
    if suggestions:
        st.caption("Recent: " + " | ".join(suggestions[::-1]))
    if q.strip():
        auto = [item for item in suggestions if item.lower().startswith(q.lower())]
        if auto:
            st.caption("Autocomplete: " + " | ".join(auto))

    col1, col2, col3, col4, col5 = st.columns(5)
    mode = col1.selectbox("Mode", ["hybrid", "semantic", "lexical"], index=0)
    top_k = col2.slider("Top K", 1, 50, 10)
    rerank = col3.checkbox("Rerank", value=True)
    threshold = col4.slider("Similarity threshold", 0.0, 1.0, 0.05, step=0.01)
    sort_by = col5.selectbox("Sort", ["rank", "score_desc", "date_desc"], index=0)

    f1, f2, f3, f4 = st.columns(4)
    category_filter = f1.text_input("Category filter")
    author_filter = f2.text_input("Author filter")
    date_from = f3.text_input("Date from (YYYY-MM-DD)")
    date_to = f4.text_input("Date to (YYYY-MM-DD)")

    if st.button("Run Search"):
        if not _enforce_rate_limit(service.config.streamlit.rate_limit_per_minute):
            st.error("Rate limit reached. Wait and retry.")
            return
        if not q.strip():
            st.warning("Enter query")
            return

        filters = {}
        if category_filter.strip():
            filters["category"] = category_filter.strip()
        if author_filter.strip():
            filters["author"] = author_filter.strip()
        if date_from.strip():
            filters["date_from"] = date_from.strip()
        if date_to.strip():
            filters["date_to"] = date_to.strip()

        response = service.search(
            SearchRequest(
                query=q,
                mode=mode,
                top_k=top_k,
                rerank=rerank,
                filters=filters,
                similarity_threshold=threshold,
            )
        )

        st.session_state["recent_searches"].append(q)
        st.session_state["recent_searches"] = st.session_state["recent_searches"][
            -service.config.streamlit.max_recent_searches :
        ]
        st.session_state["search_history"].append(
            {
                "query": q,
                "mode": mode,
                "filters": filters,
                "latency_ms": round(response.latency_ms, 3),
                "hits": len(response.hits),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        st.write(f"Latency: {response.latency_ms:.2f} ms")
        st.write(f"Hits: {len(response.hits)}")

        sorted_hits = list(response.hits)
        if sort_by == "score_desc":
            sorted_hits.sort(key=lambda h: h.score, reverse=True)
        elif sort_by == "date_desc":
            sorted_hits.sort(key=lambda h: str(h.metadata.get("date") or ""), reverse=True)

        for hit in sorted_hits:
            with st.container(border=True):
                st.markdown(f"**Rank {hit.rank}** | score={hit.score:.4f} | source={hit.retrieval_source}")
                st.markdown(_highlight(hit.text[:600], q))
                st.json(hit.metadata)

        download_payload = [hit.model_dump() for hit in sorted_hits]
        st.download_button(
            label="Download results (JSON)",
            data=json.dumps(download_payload, indent=2),
            file_name="search_results.json",
            mime="application/json",
        )

    if st.session_state["search_history"]:
        st.subheader("Search History")
        st.dataframe(pd.DataFrame(st.session_state["search_history"][-25:]), use_container_width=True)


def page_analytics(service: SemanticSearchService) -> None:
    st.header("Analytics")
    if not service.documents:
        docs_path = Path(service.config.paths["processed_data_dir"]) / "documents.jsonl"
        if docs_path.exists():
            service.load_documents(docs_path)
    if not service.chunks:
        chunks_path = Path(service.config.paths["processed_data_dir"]) / "chunks.jsonl"
        if chunks_path.exists():
            service.load_chunks(chunks_path)

    data = service.analytics()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", int(data.get("num_documents", 0)))
    c2.metric("Chunks", int(data.get("num_chunks", 0)))
    c3.metric("Avg chunk length", f"{data.get('average_chunk_length', 0):.1f}")
    c4.metric("Success rate", f"{data.get('success_rate', 0.0) * 100:.1f}%")

    st.subheader("Top Search Terms")
    st.json(data.get("top_terms", {}))

    category_dist = data.get("category_distribution", {})
    if category_dist:
        st.subheader("Category Distribution")
        cat_df = pd.DataFrame(
            [{"category": key, "count": value} for key, value in category_dist.items()]
        )
        st.plotly_chart(px.bar(cat_df, x="category", y="count", title="Documents by Category"), use_container_width=True)

    top_terms = data.get("top_terms", {})
    if top_terms:
        st.subheader("Top Search Terms Chart")
        term_df = pd.DataFrame([{"term": key, "count": value} for key, value in top_terms.items()])
        st.plotly_chart(
            px.bar(term_df.head(15), x="term", y="count", title="Most Frequent Search Terms"),
            use_container_width=True,
        )


def page_benchmarks(service: SemanticSearchService) -> None:
    st.header("Benchmarks")
    report_path = Path(service.config.paths["reports_dir"]) / "benchmark_report.json"
    if report_path.exists():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        st.json(payload)
    else:
        st.info("No benchmark report yet. Run `semantic-search benchmark`.")


def page_settings(service: SemanticSearchService) -> None:
    st.header("Settings")
    st.session_state["dark_mode"] = st.toggle("Dark mode", value=st.session_state.get("dark_mode", False))

    config_json = json.dumps(service.config.model_dump(mode="json"), indent=2)
    st.download_button(
        label="Download current config",
        data=config_json,
        file_name="active_config.json",
        mime="application/json",
    )
    st.code(config_json, language="json")


def apply_theme() -> None:
    if st.session_state.get("dark_mode"):
        st.markdown(
            """
            <style>
            .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #e2e8f0; }
            [data-testid="stSidebar"] { background-color: #111827; }
            </style>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    service = get_service()
    st.set_page_config(page_title=service.config.streamlit.page_title, layout="wide")
    _init_state()
    apply_theme()

    pages = {
        "Home": page_home,
        "Index Builder": page_index_builder,
        "Documents": page_documents,
        "Search": page_search,
        "Analytics": page_analytics,
        "Benchmarks": page_benchmarks,
        "Settings": page_settings,
    }
    selected = st.sidebar.radio("Pages", list(pages.keys()))
    pages[selected](service)


if __name__ == "__main__":
    main()
