from __future__ import annotations

from pathlib import Path

from reasoning_agent.memory.base import MemoryHit
from reasoning_agent.tooling.base import ToolContext
from reasoning_agent.tooling.tools import currency_converter, weather_tool
from reasoning_agent.tooling.tools.duckduckgo_search import SearchInput, search
from reasoning_agent.tooling.tools.local_rag import LocalRAGInput
from reasoning_agent.tooling.tools.local_rag import make_handler as rag_handler
from reasoning_agent.tooling.tools.semantic_search import SemanticSearchInput
from reasoning_agent.tooling.tools.semantic_search import make_handler as sem_handler
from reasoning_agent.tooling.tools.vector_search import VectorSearchInput
from reasoning_agent.tooling.tools.vector_search import make_handler as vec_handler
from reasoning_agent.tooling.tools.webpage_reader import WebpageReaderInput, read_webpage
from reasoning_agent.tooling.tools.wikipedia_tool import WikipediaInput, wikipedia_lookup


class _FakeDDGS:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def text(self, query: str, max_results: int = 5):
        for i in range(max_results):
            yield {"title": f"{query}-{i}", "href": f"https://example.com/{i}", "body": "snippet"}


class _FakeResponse:
    def __init__(self, payload: dict[str, object] | str):
        self._payload = payload
        self.text = payload if isinstance(payload, str) else ""
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        if isinstance(self._payload, dict):
            return self._payload
        return {}


class _FakeClient:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url: str, params=None):
        if "geocoding-api" in url:
            return _FakeResponse({"results": [{"latitude": 1.0, "longitude": 2.0}]})
        if "open-meteo.com/v1/forecast" in url:
            return _FakeResponse({"current": {"temperature_2m": 30.5, "wind_speed_10m": 11.0}})
        if "frankfurter" in url:
            return _FakeResponse({"rates": {"EUR": 0.92}})
        return _FakeResponse("<html><body><h1>hello</h1><p>world</p></body></html>")


class _FakeMemory:
    def __init__(self):
        self.texts: list[str] = []

    def write(self, event):
        self.texts.append(event.text)

    def retrieve(self, query: str, k: int = 5, scope=None):
        return [MemoryHit(text=t, score=1.0, metadata={"scope": str(scope)}) for t in self.texts[:k]]


def test_network_tools_with_monkeypatched_clients(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("reasoning_agent.tooling.tools.duckduckgo_search.DDGS", _FakeDDGS)
    monkeypatch.setattr("reasoning_agent.tooling.tools.webpage_reader.httpx.Client", _FakeClient)
    monkeypatch.setattr("reasoning_agent.tooling.tools.weather_tool.httpx.Client", _FakeClient)
    monkeypatch.setattr("reasoning_agent.tooling.tools.currency_converter.httpx.Client", _FakeClient)
    monkeypatch.setattr("reasoning_agent.tooling.tools.wikipedia_tool.wikipedia.search", lambda *args, **kwargs: ["Agent"])

    class _Page:
        title = "Agent"
        url = "https://en.wikipedia.org/wiki/Agent"

    monkeypatch.setattr("reasoning_agent.tooling.tools.wikipedia_tool.wikipedia.page", lambda *args, **kwargs: _Page())
    monkeypatch.setattr("reasoning_agent.tooling.tools.wikipedia_tool.wikipedia.summary", lambda *args, **kwargs: "Agent summary")

    ctx = ToolContext("s", "r", tmp_path)

    s_out = search(SearchInput(query="llm", max_results=3), ctx)
    assert len(s_out.results) == 3

    w_out = read_webpage(WebpageReaderInput(url="https://example.com"), ctx)
    assert "hello" in w_out.content.lower()

    weather = weather_tool.make_handler(weather_tool.WeatherProvider())(weather_tool.WeatherInput(location="Delhi"), ctx)
    assert weather.available is True

    currency = currency_converter.make_handler(currency_converter.CurrencyProvider())(
        currency_converter.CurrencyInput(amount=1.0, from_currency="USD", to_currency="EUR"),
        ctx,
    )
    assert currency.available is True

    wiki = wikipedia_lookup(WikipediaInput(query="agent"), ctx)
    assert "Agent" in wiki.title


def test_semantic_vector_and_local_rag_tools(tmp_path: Path) -> None:
    memory = _FakeMemory()
    ctx = ToolContext("s", "r", tmp_path)
    (tmp_path / "doc.md").write_text("LangGraph agent memory retrieval", encoding="utf-8")

    rag = rag_handler(memory)(LocalRAGInput(query="memory", directory="."), ctx)
    assert rag.chunks

    sem = sem_handler(memory)(SemanticSearchInput(query="agent", top_k=3), ctx)
    vec = vec_handler(memory)(VectorSearchInput(query="agent", scope="semantic", top_k=3), ctx)
    assert sem.hits
    assert vec.hits
