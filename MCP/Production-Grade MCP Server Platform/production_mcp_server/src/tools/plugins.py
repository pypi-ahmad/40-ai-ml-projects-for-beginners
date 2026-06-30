from __future__ import annotations

import hashlib
import importlib.util
import logging
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from config.settings import Settings
from tools.base import ToolDefinition
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class PluginManager:
    def __init__(self, settings: Settings, registry: ToolRegistry) -> None:
        self.settings = settings
        self.registry = registry
        self.plugin_dir = Path(settings.plugins.directory)
        self.allowlist_manifest = Path(settings.plugins.allowlist_manifest)

    def _load_allowlist(self) -> dict[str, str]:
        if not self.allowlist_manifest.exists():
            return {}
        payload = yaml.safe_load(self.allowlist_manifest.read_text(encoding="utf-8")) or {}
        plugins = payload.get("plugins", {})
        return {str(k): str(v) for k, v in plugins.items()}

    def _digest(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _import_module(self, path: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location(path.stem, str(path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to import plugin: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def load_plugins(self) -> list[str]:
        if not self.settings.plugins.enabled:
            return []

        allowlist = self._load_allowlist()
        loaded: list[str] = []
        self.plugin_dir.mkdir(parents=True, exist_ok=True)

        for file_path in sorted(self.plugin_dir.glob("*.py")):
            expected = allowlist.get(file_path.name)
            actual = self._digest(file_path)
            if expected is None:
                logger.warning("Plugin rejected (not allowlisted): %s", file_path.name)
                continue
            if expected != actual:
                logger.warning("Plugin rejected (hash mismatch): %s", file_path.name)
                continue

            try:
                module = self._import_module(file_path)
                register = getattr(module, "register_plugin")
                tools = register()
            except Exception as exc:
                logger.exception("Plugin load failed: %s", file_path.name)
                logger.error("Plugin error: %s", exc)
                continue

            if not isinstance(tools, list):
                logger.warning("Plugin register_plugin must return list: %s", file_path.name)
                continue

            valid = 0
            for tool in tools:
                if isinstance(tool, ToolDefinition):
                    self.registry.register(tool)
                    valid += 1
                elif isinstance(tool, dict):
                    self.registry.register(ToolDefinition(**tool))
                    valid += 1

            if valid:
                loaded.append(file_path.name)

        return loaded

    def generate_allowlist_template(self) -> dict[str, Any]:
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        plugins: dict[str, str] = {}
        for file_path in sorted(self.plugin_dir.glob("*.py")):
            plugins[file_path.name] = self._digest(file_path)
        payload = {"plugins": plugins}
        self.allowlist_manifest.parent.mkdir(parents=True, exist_ok=True)
        self.allowlist_manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return payload
