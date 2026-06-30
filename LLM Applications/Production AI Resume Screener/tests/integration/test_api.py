from resume_ai.api.main import health, search
from resume_ai.api.schemas import SearchRequest


class FakeService:
    def health(self):
        return {"status": "ok"}

    def search(self, query: str, top_k: int = 10):
        return {"answer": f"q={query}", "citations": [], "top_k": top_k}

    def analytics(self):
        return {"snapshot": {"total_candidates": 0, "avg_match_score": 0.0, "top_skills": {}}, "scores": []}

def test_health_endpoint(monkeypatch):
    monkeypatch.setattr("resume_ai.api.main.service", FakeService())
    payload = health().model_dump(mode="json")
    assert payload["status"] == "ok"


def test_search_endpoint(monkeypatch):
    monkeypatch.setattr("resume_ai.api.main.service", FakeService())
    payload = search(SearchRequest(query="langgraph", top_k=3)).model_dump(mode="json")
    assert payload["data"]["answer"] == "q=langgraph"
