from __future__ import annotations

import streamlit as st

from streamlit_app.utils import init_state, save_uploaded_file

init_state()

st.title("Upload Documents")
st.caption("Upload PDF, DOCX, TXT, Markdown, HTML files for local indexing")

files = st.file_uploader(
    "Choose files",
    type=["pdf", "docx", "txt", "md", "markdown", "html", "htm", "png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)
if files and st.button("Save Files", type="primary"):
    saved_paths = [save_uploaded_file(uploaded) for uploaded in files]
    st.success(f"Saved {len(saved_paths)} files")
    for path in saved_paths:
        st.code(str(path))
