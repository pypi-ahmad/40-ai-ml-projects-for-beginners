from __future__ import annotations

from pathlib import Path

REQUIRED_PAGES = [
    "streamlit_app/Home.py",
    "streamlit_app/pages/1_Knowledge_Base.py",
    "streamlit_app/pages/2_Upload_Documents.py",
    "streamlit_app/pages/3_Index_Builder.py",
    "streamlit_app/pages/4_Search.py",
    "streamlit_app/pages/5_Chat.py",
    "streamlit_app/pages/6_Retrieved_Chunks.py",
    "streamlit_app/pages/7_Sources.py",
    "streamlit_app/pages/8_Evaluation.py",
    "streamlit_app/pages/9_Settings.py",
    "streamlit_app/pages/10_Analytics.py",
]


def test_streamlit_pages_exist() -> None:
    for rel in REQUIRED_PAGES:
        assert Path(rel).exists(), f"Missing page: {rel}"
