"""Streamlit dashboard for multimodal-ai platform."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from multimodal_ai.domain import InputPayload, RequestEnvelope, TraceContext
from multimodal_ai.services.bootstrap import build_platform_service

st.set_page_config(page_title="Multimodal AI Platform", layout="wide")


@st.cache_resource(show_spinner=False)
def get_service():
    return build_platform_service()


service = get_service()

st.title("Production-Grade Multimodal AI Platform")
st.caption("FastAPI + Streamlit + CLI + MCP + ChromaDB + SQLite")

page = st.sidebar.radio(
    "Page",
    [
        "Dashboard",
        "Image Upload",
        "Caption Generator",
        "OCR",
        "Semantic Search",
        "Image Similarity",
        "VQA",
        "Chart Analyzer",
        "Document Analyzer",
        "Analytics",
        "Settings",
    ],
)

if page == "Dashboard":
    health = service.health().model_dump()
    st.subheader("Platform Health")
    st.json(health)

elif page == "Image Upload":
    uploaded = st.file_uploader(
        "Upload image or document",
        type=["png", "jpg", "jpeg", "webp", "tiff", "pdf", "docx", "pptx"],
    )
    if uploaded:
        save_path = Path("data/uploads") / uploaded.name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(uploaded.getbuffer())
        modality = (
            "image"
            if save_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".tiff"}
            else "document"
        )
        result = service.index_asset(str(save_path), modality=modality)
        st.success(f"Indexed {uploaded.name}")
        st.json(result)

elif page == "Caption Generator":
    image_path = st.text_input("Image path")
    style = st.selectbox("Style", ["short", "detailed", "social", "technical", "alt_text"])
    if st.button("Generate Caption") and image_path:
        req = RequestEnvelope(
            input=InputPayload(image_path=image_path),
            options={"style": style},
            trace=TraceContext(source="streamlit"),
        )
        st.json(service.caption(req).model_dump())

elif page == "OCR":
    file_path = st.text_input("Document/Image path")
    if st.button("Run OCR") and file_path:
        req = RequestEnvelope(
            input=InputPayload(document_path=file_path),
            trace=TraceContext(source="streamlit"),
        )
        result = service.ocr(req).model_dump()
        st.json(result)
        text = result.get("result", {}).get("text", "")
        if text:
            st.text_area("OCR Text", text, height=240)

elif page == "Semantic Search":
    query = st.text_input("Search query")
    modality = st.selectbox("Modality", ["image", "document", "ocr", "screenshot", "chart"])
    top_k = st.slider("Top K", min_value=1, max_value=20, value=5)
    if st.button("Search") and query:
        req = RequestEnvelope(
            input=InputPayload(query=query),
            options={"modality": modality, "top_k": top_k},
            trace=TraceContext(source="streamlit"),
        )
        st.json(service.search(req).model_dump())

elif page == "Image Similarity":
    first = st.text_input("Image path A")
    second = st.text_input("Image path B")
    if st.button("Compare") and first and second:
        req = RequestEnvelope(
            input=InputPayload(image_paths=[first, second]),
            trace=TraceContext(source="streamlit"),
        )
        st.json(service.compare(req).model_dump())

elif page == "VQA":
    image_path = st.text_input("Image path")
    question = st.text_input("Question", value="What is shown in this image?")
    if st.button("Ask") and image_path:
        req = RequestEnvelope(
            input=InputPayload(image_path=image_path, question=question),
            trace=TraceContext(source="streamlit"),
        )
        st.json(service.vqa(req).model_dump())

elif page == "Chart Analyzer":
    chart_path = st.text_input("Chart image path")
    if st.button("Analyze Chart") and chart_path:
        req = RequestEnvelope(
            input=InputPayload(
                image_path=chart_path, question="Summarize chart trends and anomalies"
            ),
            trace=TraceContext(source="streamlit"),
        )
        st.json(service.analyze(req).model_dump())

elif page == "Document Analyzer":
    doc_path = st.text_input("Document path")
    question = st.text_input("Question", value="Summarize this document")
    if st.button("Analyze Document") and doc_path:
        req = RequestEnvelope(
            input=InputPayload(document_path=doc_path, question=question),
            trace=TraceContext(source="streamlit"),
        )
        st.json(service.analyze(req).model_dump())

elif page == "Analytics":
    metrics = service.analytics().model_dump()
    st.json(metrics)
    model_usage = metrics.get("result", {}).get("model_usage", [])
    if model_usage:
        st.dataframe(pd.DataFrame(model_usage))

elif page == "Settings":
    st.write("Current defaults")
    st.json(
        {
            "vision_model": service._config.default_vision_model,  # noqa: SLF001
            "llm_backend": service._config.default_llm_backend,  # noqa: SLF001
            "embedding_model": service._config.default_embedding_model,  # noqa: SLF001
            "ocr_engine": service._config.ocr_primary_engine,  # noqa: SLF001
        }
    )
