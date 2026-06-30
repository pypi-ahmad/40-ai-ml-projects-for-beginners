from __future__ import annotations

import pytest
from ollama import Client


@pytest.mark.integration
def test_ollama_optional_health() -> None:
    client = Client(host="http://127.0.0.1:11434")
    try:
        client.list()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Ollama not available: {exc}")
