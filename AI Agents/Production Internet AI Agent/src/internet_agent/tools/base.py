"""Tool abstractions and typed descriptors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ToolDescriptor(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)


class BaseTool[TIn: BaseModel, TOut: BaseModel](ABC):
    """Base class for all agent tools."""

    name: str
    description: str
    input_model: type[TIn]
    output_model: type[TOut]

    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name,
            description=self.description,
            input_schema=self.input_model.model_json_schema(),
            output_schema=self.output_model.model_json_schema(),
        )

    def validate_input(self, payload: dict[str, Any]) -> TIn:
        return self.input_model.model_validate(payload)

    def validate_output(self, payload: dict[str, Any]) -> TOut:
        return self.output_model.model_validate(payload)

    @abstractmethod
    async def run(self, payload: TIn) -> TOut:
        raise NotImplementedError
