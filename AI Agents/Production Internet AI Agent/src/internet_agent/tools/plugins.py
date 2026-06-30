"""Plugin loading for custom tools/search providers."""

from __future__ import annotations

import importlib
from collections.abc import Iterable

from internet_agent.tools.base import BaseTool
from internet_agent.tools.registry import ToolRegistry


class ToolPluginLoader:
    """Load tools from dotted-path factories.

    Factory format: `package.module:function_name` and must return iterable of tool instances.
    """

    @staticmethod
    def load_into_registry(registry: ToolRegistry, factories: Iterable[str]) -> list[str]:
        loaded: list[str] = []
        for factory in factories:
            module_name, func_name = factory.split(":", maxsplit=1)
            module = importlib.import_module(module_name)
            factory_func = getattr(module, func_name)
            tools = factory_func()
            for tool in tools:
                if not isinstance(tool, BaseTool):
                    raise TypeError(f"Plugin tool must extend BaseTool: {tool}")
                registry.register(tool)
                loaded.append(tool.name)
        return loaded
