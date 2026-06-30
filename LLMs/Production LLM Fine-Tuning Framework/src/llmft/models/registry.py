"""Model alias registry with deterministic fallback behavior."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ModelSpec:
    """Canonical model target and fallback candidates."""

    alias: str
    primary_id: str
    fallback_ids: list[str] = field(default_factory=list)
    family: str = "general"


@dataclass(slots=True)
class ModelResolution:
    """Resolved model outcome."""

    alias: str
    selected_id: str
    used_fallback: bool
    reason: str


class ModelRegistry:
    """Resolve model aliases to model IDs with fallback logic."""

    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(
            ModelSpec(
                alias="llama3_8b",
                primary_id="meta-llama/Meta-Llama-3-8B-Instruct",
                fallback_ids=["meta-llama/Llama-3.1-8B-Instruct", "NousResearch/Meta-Llama-3-8B-Instruct"],
                family="llama",
            )
        )
        self.register(
            ModelSpec(
                alias="qwen3_8b",
                primary_id="Qwen/Qwen3-8B-Instruct",
                fallback_ids=["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-3B-Instruct"],
                family="qwen",
            )
        )
        self.register(
            ModelSpec(
                alias="mistral_7b",
                primary_id="mistralai/Mistral-7B-Instruct-v0.3",
                fallback_ids=["mistralai/Mistral-7B-Instruct-v0.2"],
                family="mistral",
            )
        )
        self.register(
            ModelSpec(
                alias="gemma3",
                primary_id="google/gemma-3-4b-it",
                fallback_ids=["google/gemma-2-9b-it", "google/gemma-2-2b-it"],
                family="gemma",
            )
        )
        self.register(
            ModelSpec(
                alias="phi4_mini",
                primary_id="microsoft/Phi-4-mini-instruct",
                fallback_ids=["microsoft/Phi-3.5-mini-instruct"],
                family="phi",
            )
        )
        self.register(
            ModelSpec(
                alias="granite41",
                primary_id="ibm-granite/granite-4.1-8b-instruct",
                fallback_ids=["ibm-granite/granite-3.2-8b-instruct"],
                family="granite",
            )
        )

    def register(self, spec: ModelSpec) -> None:
        """Register or replace model spec."""
        self._specs[spec.alias] = spec

    def list_aliases(self) -> list[str]:
        """List available aliases."""
        return sorted(self._specs)

    def resolve(
        self,
        alias: str,
        available_model_ids: set[str] | None = None,
        allow_fallback: bool = True,
    ) -> ModelResolution:
        """Resolve model alias to selected model ID.

        Args:
            alias: Alias key.
            available_model_ids: Optional available IDs for strict selection.
            allow_fallback: Allow fallback candidates.

        Returns:
            Model resolution with fallback metadata.
        """
        if alias not in self._specs:
            raise KeyError(f"Unknown model alias: {alias}")

        spec = self._specs[alias]
        if not available_model_ids:
            return ModelResolution(alias=alias, selected_id=spec.primary_id, used_fallback=False, reason="primary")

        if spec.primary_id in available_model_ids:
            return ModelResolution(alias=alias, selected_id=spec.primary_id, used_fallback=False, reason="primary_available")

        if allow_fallback:
            for candidate in spec.fallback_ids:
                if candidate in available_model_ids:
                    return ModelResolution(alias=alias, selected_id=candidate, used_fallback=True, reason=f"fallback:{candidate}")

        return ModelResolution(alias=alias, selected_id=spec.primary_id, used_fallback=False, reason="unavailable_no_fallback")
