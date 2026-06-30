import pytest

from src.ollama_client import OllamaClient


@pytest.fixture
def client() -> OllamaClient:
    c = OllamaClient()
    yield c
    c.close()


def test_generate_basic(client: OllamaClient) -> None:
    result = client.generate("qwen3.5:2b", "Say hello in one word.", temperature=0.0)
    assert "response" in result
    assert len(result["response"]) > 0


def test_chat_basic(client: OllamaClient) -> None:
    messages = [{"role": "user", "content": "Say hi"}]
    result = client.chat("qwen3.5:4b", messages, temperature=0.0)
    assert "message" in result
    assert "content" in result["message"]


def test_embed_basic(client: OllamaClient) -> None:
    vec = client.embed("qwen3-embedding:4b", "hello world")
    assert isinstance(vec, list)
    assert len(vec) > 0
    assert isinstance(vec[0], float)


def test_measure_inference_time(client: OllamaClient) -> None:
    m = client.measure_inference_time("qwen3.5:2b", "test", temperature=0.0)
    assert "latency_s" in m
    assert "tokens" in m
    assert m["latency_s"] > 0
    assert m["tokens"] > 0
