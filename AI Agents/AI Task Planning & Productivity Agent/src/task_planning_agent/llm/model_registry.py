"""Model registry and fallback resolution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ModelRegistry:
    families: dict[str, list[str]]
    default_family: str

    def candidates(self, family: str | None = None) -> list[str]:
        key = family or self.default_family
        return self.families.get(key, [])
