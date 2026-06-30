"""Base tool interface with schema contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

InputModelT = TypeVar("InputModelT", bound=BaseModel)
OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class ToolDescriptor(BaseModel):
    """Tool metadata for discovery."""

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]


class BaseTool(ABC, Generic[InputModelT, OutputModelT]):
    """Base class for all tools."""

    name: str
    description: str
    input_model: type[InputModelT]
    output_model: type[OutputModelT]

    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name,
            description=self.description,
            input_schema=self.input_model.model_json_schema(),
            output_schema=self.output_model.model_json_schema(),
        )

    def validate_input(self, payload: dict[str, Any]) -> InputModelT:
        return self.input_model.model_validate(payload)

    def validate_output(self, payload: Any) -> OutputModelT:
        if isinstance(payload, self.output_model):
            return payload
        return self.output_model.model_validate(payload)

    def is_safe(self) -> bool:
        return True

    @abstractmethod
    async def run(self, payload: InputModelT) -> OutputModelT:
        """Execute tool."""
