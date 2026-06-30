"""Provider-specific API connector implementations."""

from __future__ import annotations

from typing import Any

from api_intel_agent.connectors.base import APIRequest, BaseConnector


class GitHubConnector(BaseConnector):
    def __init__(self) -> None:
        super().__init__("github")

    def build_request(self, query: str, params: dict[str, Any] | None = None) -> APIRequest:
        merged = {"q": query, "per_page": 20, **(params or {})}
        return APIRequest(endpoint="/search/repositories", params=merged)


class HuggingFaceConnector(BaseConnector):
    def __init__(self) -> None:
        super().__init__("huggingface")

    def build_request(self, query: str, params: dict[str, Any] | None = None) -> APIRequest:
        return APIRequest(endpoint="/models", params={"search": query, **(params or {})})


class StackOverflowConnector(BaseConnector):
    def __init__(self) -> None:
        super().__init__("stackoverflow")

    def build_request(self, query: str, params: dict[str, Any] | None = None) -> APIRequest:
        merged = {
            "order": "desc",
            "sort": "relevance",
            "intitle": query,
            "site": "stackoverflow",
            **(params or {}),
        }
        return APIRequest(endpoint="/search", params=merged)


class RedditConnector(BaseConnector):
    def __init__(self) -> None:
        super().__init__("reddit")

    def build_request(self, query: str, params: dict[str, Any] | None = None) -> APIRequest:
        merged = {"q": query, "limit": 25, **(params or {})}
        return APIRequest(endpoint="/search.json", params=merged)

    def normalize(self, payload: Any) -> list[dict[str, Any]]:
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        children = data.get("children", []) if isinstance(data, dict) else []
        return [item.get("data", {}) for item in children if isinstance(item, dict)]


class NewsConnector(BaseConnector):
    def __init__(self) -> None:
        super().__init__("news")

    def build_request(self, query: str, params: dict[str, Any] | None = None) -> APIRequest:
        merged = {"q": query, "pageSize": 20, **(params or {})}
        return APIRequest(endpoint="/everything", params=merged)


class OpenLibraryConnector(BaseConnector):
    def __init__(self) -> None:
        super().__init__("openlibrary")

    def build_request(self, query: str, params: dict[str, Any] | None = None) -> APIRequest:
        return APIRequest(endpoint="/search.json", params={"q": query, **(params or {})})

    def normalize(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            docs = payload.get("docs", [])
            if isinstance(docs, list):
                return [doc for doc in docs if isinstance(doc, dict)]
        return []


class WeatherConnector(BaseConnector):
    def __init__(self) -> None:
        super().__init__("weather")

    def build_request(self, query: str, params: dict[str, Any] | None = None) -> APIRequest:
        merged = {"latitude": 51.5, "longitude": -0.12, "current_weather": True, **(params or {})}
        return APIRequest(endpoint="/forecast", params=merged)


class ExchangeRateConnector(BaseConnector):
    def __init__(self) -> None:
        super().__init__("exchange_rate")

    def build_request(self, query: str, params: dict[str, Any] | None = None) -> APIRequest:
        base = (params or {}).get("base", query.split()[0] if query else "USD")
        return APIRequest(endpoint="/latest", params={"base": base, **(params or {})})


class CoinGeckoConnector(BaseConnector):
    def __init__(self) -> None:
        super().__init__("coingecko")

    def build_request(self, query: str, params: dict[str, Any] | None = None) -> APIRequest:
        return APIRequest(endpoint="/search", params={"query": query, **(params or {})})


class NASAConnector(BaseConnector):
    def __init__(self) -> None:
        super().__init__("nasa")

    def build_request(self, query: str, params: dict[str, Any] | None = None) -> APIRequest:
        merged = {"q": query, **(params or {})}
        return APIRequest(endpoint="/search", params=merged)


class RESTCountriesConnector(BaseConnector):
    def __init__(self) -> None:
        super().__init__("restcountries")

    def build_request(self, query: str, params: dict[str, Any] | None = None) -> APIRequest:
        name = (params or {}).get("name", query.split()[0] if query else "India")
        return APIRequest(endpoint=f"/name/{name}", params={k: v for k, v in (params or {}).items() if k != "name"})


class PublicHolidayConnector(BaseConnector):
    def __init__(self) -> None:
        super().__init__("holidays")

    def build_request(self, query: str, params: dict[str, Any] | None = None) -> APIRequest:
        year = (params or {}).get("year", 2026)
        country = (params or {}).get("country", "US")
        return APIRequest(endpoint=f"/PublicHolidays/{year}/{country}", params=None)


class JSONPlaceholderConnector(BaseConnector):
    def __init__(self) -> None:
        super().__init__("jsonplaceholder")

    def build_request(self, query: str, params: dict[str, Any] | None = None) -> APIRequest:
        endpoint = (params or {}).get("endpoint", "/posts")
        query_param = (params or {}).get("q")
        request_params = {k: v for k, v in (params or {}).items() if k != "endpoint"}
        if query_param:
            request_params["q"] = query_param
        return APIRequest(endpoint=endpoint, params=request_params or None)


class JiraConnector(BaseConnector):
    def __init__(self) -> None:
        super().__init__("jira")

    def build_request(self, query: str, params: dict[str, Any] | None = None) -> APIRequest:
        merged = {"jql": f'text ~ "{query}"', "maxResults": 20, **(params or {})}
        return APIRequest(endpoint="/rest/api/3/search", params=merged)


class GitLabConnector(BaseConnector):
    def __init__(self) -> None:
        super().__init__("gitlab")

    def build_request(self, query: str, params: dict[str, Any] | None = None) -> APIRequest:
        return APIRequest(endpoint="/projects", params={"search": query, **(params or {})})


class NotionConnector(BaseConnector):
    def __init__(self) -> None:
        super().__init__("notion")

    def build_request(self, query: str, params: dict[str, Any] | None = None) -> APIRequest:
        payload = {"query": query, **(params or {})}
        return APIRequest(endpoint="/search", method="POST", json=payload)


class GoogleSheetsConnector(BaseConnector):
    def __init__(self) -> None:
        super().__init__("google_sheets")

    def build_request(self, query: str, params: dict[str, Any] | None = None) -> APIRequest:
        sheet_id = (params or {}).get("sheet_id", "")
        range_name = (params or {}).get("range", "A1:C20")
        return APIRequest(endpoint=f"/spreadsheets/{sheet_id}/values/{range_name}", params={})


class SlackConnector(BaseConnector):
    def __init__(self) -> None:
        super().__init__("slack")

    def build_request(self, query: str, params: dict[str, Any] | None = None) -> APIRequest:
        return APIRequest(endpoint="/search.messages", params={"query": query, **(params or {})})


class GmailConnector(BaseConnector):
    def __init__(self) -> None:
        super().__init__("gmail")

    def build_request(self, query: str, params: dict[str, Any] | None = None) -> APIRequest:
        user_id = (params or {}).get("user_id", "me")
        return APIRequest(endpoint=f"/users/{user_id}/messages", params={"q": query, **(params or {})})
