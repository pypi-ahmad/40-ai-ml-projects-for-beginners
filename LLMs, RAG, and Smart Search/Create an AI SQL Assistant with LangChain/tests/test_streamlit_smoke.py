from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest


def test_streamlit_app_importable() -> None:
    app_path = Path("streamlit_app/app.py")
    spec = importlib.util.spec_from_file_location("streamlit_app.app", app_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main")


@pytest.mark.skipif(
    importlib.util.find_spec("streamlit.testing.v1") is None,
    reason="streamlit testing module unavailable",
)
def test_streamlit_testing_api_available() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: F401

    assert AppTest is not None
