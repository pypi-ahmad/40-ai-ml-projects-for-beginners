from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from config.settings import Settings
from memory.service import MemoryService


@dataclass(slots=True)
class ResourceDefinition:
    uri: str
    name: str
    description: str
    mime_type: str
    loader: Callable[[], str]


class ResourceLibrary:
    def __init__(self, settings: Settings, memory: MemoryService) -> None:
        self.settings = settings
        self.memory = memory
        self.root = Path(settings.plugins.directory).parents[0]
        self._resources: dict[str, ResourceDefinition] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        readme_path = self.root / "README.md"
        config_path = self.root / "configs" / "default.yaml"
        logs_path = self.root / "logs" / "server.log"

        if readme_path.exists():
            self.register(
                ResourceDefinition(
                    uri="file://README.md",
                    name="Project README",
                    description="Primary project documentation",
                    mime_type="text/markdown",
                    loader=lambda: readme_path.read_text(encoding="utf-8"),
                )
            )

        if config_path.exists():
            self.register(
                ResourceDefinition(
                    uri="config://default",
                    name="Default Configuration",
                    description="Primary YAML configuration",
                    mime_type="application/yaml",
                    loader=lambda: config_path.read_text(encoding="utf-8"),
                )
            )

        self.register(
            ResourceDefinition(
                uri="memory://recent-conversations",
                name="Recent Conversations",
                description="Latest recorded conversation events",
                mime_type="application/json",
                loader=lambda: json.dumps(
                    self.memory.fetch_recent_conversations(session_id="default", limit=20),
                    indent=2,
                ),
            )
        )

        self.register(
            ResourceDefinition(
                uri="metrics://recent",
                name="Recent Metrics",
                description="Recent monitoring metrics",
                mime_type="application/json",
                loader=lambda: json.dumps(self.memory.recent_metrics(limit=200), indent=2),
            )
        )

        if logs_path.exists():
            self.register(
                ResourceDefinition(
                    uri="logs://server",
                    name="Server Logs",
                    description="Structured run logs",
                    mime_type="application/json",
                    loader=lambda: logs_path.read_text(encoding="utf-8"),
                )
            )

    def register(self, resource: ResourceDefinition) -> None:
        self._resources[resource.uri] = resource

    def names(self) -> list[str]:
        return sorted(resource.name for resource in self._resources.values())

    def list(self) -> list[dict[str, Any]]:
        return [
            {
                "uri": res.uri,
                "name": res.name,
                "description": res.description,
                "mime_type": res.mime_type,
            }
            for res in self._resources.values()
        ]

    def read(self, uri: str) -> dict[str, Any]:
        if uri in self._resources:
            resource = self._resources[uri]
            return {
                "uri": resource.uri,
                "mime_type": resource.mime_type,
                "content": resource.loader(),
            }

        if uri.startswith("csv://"):
            rel = uri.removeprefix("csv://")
            path = (self.root / rel).resolve()
            rows: list[dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for idx, row in enumerate(reader):
                    rows.append(dict(row))
                    if idx >= 99:
                        break
            return {"uri": uri, "mime_type": "application/json", "content": json.dumps(rows, indent=2)}

        if uri.startswith("sqlite://"):
            query = uri.removeprefix("sqlite://")
            return {
                "uri": uri,
                "mime_type": "application/json",
                "content": json.dumps({"note": "Use sqlite_query tool", "query": query}, indent=2),
            }

        path = (self.root / uri).resolve()
        if path.exists() and path.is_file():
            return {"uri": uri, "mime_type": "text/plain", "content": path.read_text(encoding="utf-8")}

        raise KeyError(f"Unknown resource URI: {uri}")
