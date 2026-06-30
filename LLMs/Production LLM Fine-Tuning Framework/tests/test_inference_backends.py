import asyncio

import pytest

from llmft.config.schemas import InferenceConfig
from llmft.inference import InferenceRouter


def test_transformers_backend_fallback_returns_text() -> None:
    cfg = InferenceConfig(backend="transformers")
    router = InferenceRouter(cfg)
    out = asyncio.run(router.generate("hello"))
    assert isinstance(out, str)
    assert out


def test_vllm_backend_fallback_without_server() -> None:
    cfg = InferenceConfig(backend="vllm", vllm_host="http://127.0.0.1:9")
    router = InferenceRouter(cfg)
    with pytest.raises(RuntimeError):
        asyncio.run(router.generate("hello"))


def test_ollama_backend_fallback_without_server() -> None:
    cfg = InferenceConfig(backend="ollama", ollama_host="http://127.0.0.1:9")
    router = InferenceRouter(cfg)
    with pytest.raises(RuntimeError):
        asyncio.run(router.generate("hello"))
