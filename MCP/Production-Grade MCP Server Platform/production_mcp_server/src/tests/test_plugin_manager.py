from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from config.settings import load_settings
from memory.service import MemoryService
from prompts.library import PromptLibrary
from resources.library import ResourceLibrary
from tools.builtin import build_builtin_tools
from tools.registry import ToolRegistry
from tools.plugins import PluginManager


def test_plugin_allowlist_hash(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir(parents=True)
    plugin_file = plugin_dir / "sample_plugin.py"
    plugin_file.write_text(
        """
from tools.base import ToolDefinition

async def ping_tool() -> dict[str, str]:
    return {"pong": "ok"}

def register_plugin() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="plugin_ping",
            description="Plugin ping",
            input_schema={"type": "object", "properties": {}},
            handler=ping_tool,
            read_only=True,
        )
    ]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    digest = hashlib.sha256(plugin_file.read_bytes()).hexdigest()
    allowlist = tmp_path / "plugins_allowlist.yaml"
    allowlist.write_text(yaml.safe_dump({"plugins": {"sample_plugin.py": digest}}), encoding="utf-8")

    settings = load_settings("configs/default.yaml")
    settings.plugins.directory = str(plugin_dir)
    settings.plugins.allowlist_manifest = str(allowlist)

    memory = MemoryService(settings)
    resources = ResourceLibrary(settings, memory)
    prompts = PromptLibrary()

    registry = ToolRegistry()
    for tool in build_builtin_tools(settings=settings, memory=memory, resources=resources, prompts=prompts):
        registry.register(tool)

    manager = PluginManager(settings=settings, registry=registry)
    loaded = manager.load_plugins()

    assert loaded == ["sample_plugin.py"]
    assert "plugin_ping" in registry.names()
