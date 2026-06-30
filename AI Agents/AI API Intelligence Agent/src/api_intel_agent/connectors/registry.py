"""Connector registry and planner-facing router."""

from __future__ import annotations

from typing import Iterable

from api_intel_agent.connectors.base import BaseConnector
from api_intel_agent.connectors.providers import (
    CoinGeckoConnector,
    ExchangeRateConnector,
    GitHubConnector,
    GitLabConnector,
    GmailConnector,
    GoogleSheetsConnector,
    HuggingFaceConnector,
    JSONPlaceholderConnector,
    JiraConnector,
    NASAConnector,
    NewsConnector,
    NotionConnector,
    OpenLibraryConnector,
    PublicHolidayConnector,
    RESTCountriesConnector,
    RedditConnector,
    SlackConnector,
    StackOverflowConnector,
    WeatherConnector,
)


class ConnectorRegistry:
    def __init__(self) -> None:
        self._connectors: dict[str, BaseConnector] = {
            "github": GitHubConnector(),
            "huggingface": HuggingFaceConnector(),
            "stackoverflow": StackOverflowConnector(),
            "reddit": RedditConnector(),
            "news": NewsConnector(),
            "openlibrary": OpenLibraryConnector(),
            "weather": WeatherConnector(),
            "exchange_rate": ExchangeRateConnector(),
            "coingecko": CoinGeckoConnector(),
            "nasa": NASAConnector(),
            "restcountries": RESTCountriesConnector(),
            "holidays": PublicHolidayConnector(),
            "jsonplaceholder": JSONPlaceholderConnector(),
            "jira": JiraConnector(),
            "gitlab": GitLabConnector(),
            "notion": NotionConnector(),
            "google_sheets": GoogleSheetsConnector(),
            "slack": SlackConnector(),
            "gmail": GmailConnector(),
        }

    def get(self, name: str) -> BaseConnector:
        return self._connectors[name]

    def list_names(self) -> list[str]:
        return sorted(self._connectors)

    def select(self, names: Iterable[str]) -> list[BaseConnector]:
        selected = []
        for name in names:
            connector = self._connectors.get(name)
            if connector:
                selected.append(connector)
        return selected

    def infer_from_query(self, query: str) -> list[str]:
        lowered = query.lower()
        matches: set[str] = set()
        mapping = {
            "github": ["github", "repository", "repo", "stars"],
            "news": ["news", "headline"],
            "weather": ["weather", "temperature"],
            "coingecko": ["coin", "crypto", "bitcoin", "ethereum"],
            "exchange_rate": ["exchange", "currency", "fx", "usd"],
            "nasa": ["space", "nasa", "mars"],
            "reddit": ["reddit", "subreddit"],
            "openlibrary": ["book", "author", "isbn"],
            "huggingface": ["huggingface", "model", "dataset"],
            "stackoverflow": ["stack", "stackoverflow", "error", "bug"],
            "restcountries": ["country", "population", "capital"],
            "holidays": ["holiday", "public holiday"],
            "jsonplaceholder": ["jsonplaceholder", "test api", "dummy"],
        }

        for provider, keywords in mapping.items():
            if any(keyword in lowered for keyword in keywords):
                matches.add(provider)

        return sorted(matches or {"github", "news"})
