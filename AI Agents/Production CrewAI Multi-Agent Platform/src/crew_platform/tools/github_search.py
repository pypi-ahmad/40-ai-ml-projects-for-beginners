"""Optional GitHub repository search tool."""

from __future__ import annotations

import httpx
from pydantic import BaseModel

from crew_platform.tools.base import BaseTool
from crew_platform.utils.network import offline_mode


class GitHubSearchInput(BaseModel):
    query: str
    max_results: int = 5


class GitHubSearchOutput(BaseModel):
    items: list[dict[str, str | int]]


class GitHubSearchTool(BaseTool[GitHubSearchInput, GitHubSearchOutput]):
    name = "github_search"
    description = "Searches GitHub repositories"
    input_model = GitHubSearchInput
    output_model = GitHubSearchOutput

    async def run(self, payload: GitHubSearchInput) -> GitHubSearchOutput:
        items = []
        if offline_mode():
            return GitHubSearchOutput(
                items=[
                    {
                        "full_name": "offline-fallback",
                        "html_url": "",
                        "description": "github disabled by AGENT_OFFLINE_MODE",
                        "stargazers_count": 0,
                    }
                ]
            )
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    "https://api.github.com/search/repositories",
                    params={"q": payload.query, "per_page": payload.max_results},
                    headers={"Accept": "application/vnd.github+json"},
                )
                response.raise_for_status()
                data = response.json()

            for item in data.get("items", []):
                items.append(
                    {
                        "full_name": item.get("full_name", ""),
                        "html_url": item.get("html_url", ""),
                        "description": item.get("description", "") or "",
                        "stargazers_count": int(item.get("stargazers_count", 0)),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            items.append(
                {
                    "full_name": "offline-fallback",
                    "html_url": "",
                    "description": f"github unavailable: {exc}",
                    "stargazers_count": 0,
                }
            )
        return GitHubSearchOutput(items=items)
