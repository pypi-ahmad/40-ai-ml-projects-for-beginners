"""Plugin discovery and registration."""

from __future__ import annotations

from importlib import metadata

from langgraph_platform.plugins.base import PluginBundle


class PluginRegistry:
    """Registry for external plugins via entry points."""

    def __init__(self) -> None:
        self.bundles: dict[str, PluginBundle] = {}

    def discover(self, group: str = "langgraph_platform.plugins") -> None:
        """Load plugin bundles from package entry points."""

        for entry_point in metadata.entry_points(group=group):
            bundle = entry_point.load()()
            if not isinstance(bundle, PluginBundle):
                raise TypeError(f"Invalid plugin bundle from {entry_point.name}")
            self.bundles[bundle.name] = bundle

    def list_plugins(self) -> list[str]:
        return sorted(self.bundles.keys())
