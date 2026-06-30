"""Source ranking by relevance, freshness, and authority signals."""

from __future__ import annotations

from datetime import datetime
from urllib.parse import urlparse


def _authority_boost(url: str) -> float:
    domain = urlparse(url).netloc.lower()
    if any(host in domain for host in ("python.org", "wikipedia.org", "github.com", "arxiv.org")):
        return 0.2
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return 0.15
    return 0.05


def _freshness_score(published: str) -> float:
    if not published:
        return 0.0
    try:
        ts = datetime.fromisoformat(published.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    age_days = max((datetime.now(ts.tzinfo) - ts).days, 0)
    if age_days <= 1:
        return 0.2
    if age_days <= 7:
        return 0.12
    if age_days <= 30:
        return 0.07
    return 0.02


def rank_sources(results: list[dict], query: str) -> list[dict]:
    """Rank result rows with heuristic blend score."""

    query_tokens = {token.lower() for token in query.split() if len(token) > 2}
    ranked: list[dict] = []

    for row in results:
        title = row.get("title", "")
        snippet = row.get("snippet", "")
        haystack = f"{title} {snippet}".lower()
        overlap = sum(1 for token in query_tokens if token in haystack)
        relevance = overlap / max(len(query_tokens), 1)
        score = relevance * 0.65
        score += _authority_boost(row.get("url", ""))
        score += _freshness_score(row.get("published", ""))
        row = dict(row)
        row["rank_score"] = round(score, 4)
        ranked.append(row)

    ranked.sort(key=lambda x: x.get("rank_score", 0.0), reverse=True)
    return ranked
