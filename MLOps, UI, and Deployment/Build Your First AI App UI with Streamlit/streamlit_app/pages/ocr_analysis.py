"""OCR and document analysis page."""

from __future__ import annotations

import streamlit as st

from streamlit_app.components.ui_components import render_result_box
from streamlit_app.utils.caching import cached_ocr


SUPPORTED_TYPES = ["pdf", "docx", "png", "jpg", "jpeg", "bmp", "tiff"]


def render() -> None:
    st.title("PDF / OCR Analysis")

    with st.expander("Learning Module: OCR + LLM Pipeline", expanded=False):
        st.markdown(
            """
**Definition**: OCR (Optical Character Recognition) extracts text from visual documents.

**Theory**: We separate extraction from reasoning for modular reliability.

**Motivation**: Enterprise workflows require parsing contracts, invoices, and scanned forms.

**Real-world example**: Finance automation extracts invoice fields and summarizes exceptions.

**Visual explanation**: file upload -> text extraction -> primary OCR model -> fallback OCR model.

**Code explanation**: This page routes by extension, validates upload, and handles model failures.

**Best practices**: Keep extraction logs and cap text length before LLM analysis.

**Common mistakes**: Sending binary files directly to text-only models without preprocessing.
            """
        )

    uploaded = st.file_uploader(
        "Upload document",
        type=SUPPORTED_TYPES,
        help="Accepted: PDF, DOCX, PNG, JPG, JPEG, BMP, TIFF",
    )

    if not uploaded:
        st.info("Upload a document to begin analysis.")
        return

    file_bytes = uploaded.read()
    if not file_bytes:
        st.error("Uploaded file is empty.")
        return

    st.success(f"Loaded file: {uploaded.name} ({len(file_bytes):,} bytes)")

    if uploaded.type and "image" in uploaded.type:
        st.image(file_bytes, caption="Uploaded image", use_container_width=True)

    run = st.button("Run OCR + analysis", type="primary", use_container_width=True)
    if not run:
        return

    with st.spinner("Running extraction and analysis..."):
        primary_model = st.session_state["selected_models"].get("ocr_primary", "glm-ocr:latest")
        fallback_model = st.session_state["selected_models"].get("ocr_fallback", "deepseek-ocr:latest")
        result = cached_ocr(file_bytes, uploaded.name, primary_model, fallback_model)

    render_result_box(result, title="Document Insights")

    if result.startswith("Primary and fallback OCR analysis models failed"):
        st.error("Both local OCR models failed. Verify models are pulled and Ollama is running.")
