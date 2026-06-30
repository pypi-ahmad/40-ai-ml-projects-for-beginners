"""Plugin loader for YAML-defined extensions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class PluginLoader:
    """Loads plugin manifests without modifying core code."""

    def __init__(self, plugin_dir: str = "plugins") -> None:
        self.plugin_dir = Path(plugin_dir)

    def manifests(self) -> list[dict[str, Any]]:
        if not self.plugin_dir.exists():
            return []
        manifests: list[dict[str, Any]] = []
        for file in sorted(self.plugin_dir.glob("*.yaml")):
            data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict):
                data["_file"] = str(file)
                manifests.append(data)
        return manifests
