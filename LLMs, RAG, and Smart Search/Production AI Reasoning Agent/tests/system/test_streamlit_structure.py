from __future__ import annotations

from pathlib import Path


def test_streamlit_pages_exist() -> None:
    base = Path("streamlit_app")
    assert (base / "Home.py").exists()
    assert (base / "pages" / "1_Execution_Trace.py").exists()
    assert (base / "pages" / "2_Tool_Calls.py").exists()
    assert (base / "pages" / "3_Memory.py").exists()
    assert (base / "pages" / "4_Analytics.py").exists()
    assert (base / "pages" / "5_Settings.py").exists()
