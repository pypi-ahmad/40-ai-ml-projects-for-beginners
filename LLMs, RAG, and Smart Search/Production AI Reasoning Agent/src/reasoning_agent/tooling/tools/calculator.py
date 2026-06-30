"""Safe calculator tool with arithmetic and scientific support."""

from __future__ import annotations

import ast
import math
import re
import statistics
from typing import Any

from pydantic import BaseModel, Field

from reasoning_agent.tooling.base import ToolContext, ToolSpec


class CalculatorInput(BaseModel):
    """Calculator input payload."""

    expression: str = Field(description="Math expression, e.g. sin(0.5) + mean([1,2,3])")


class CalculatorOutput(BaseModel):
    """Calculator output payload."""

    result: float
    expression: str


_ALLOWED_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "pow": pow,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "pi": math.pi,
    "e": math.e,
    "mean": statistics.mean,
    "median": statistics.median,
    "stdev": statistics.stdev,
}

_ALLOWED_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Load,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.List,
    ast.Tuple,
}


def _validate_ast(node: ast.AST) -> None:
    for child in ast.walk(node):
        if type(child) not in _ALLOWED_NODES:
            raise ValueError(f"Unsupported syntax: {type(child).__name__}")
        if isinstance(child, ast.Call):
            if not isinstance(child.func, ast.Name):
                raise ValueError("Only direct function calls allowed")
            if child.func.id not in _ALLOWED_FUNCTIONS:
                raise ValueError(f"Function not allowed: {child.func.id}")
        if isinstance(child, ast.Name) and child.id not in _ALLOWED_FUNCTIONS:
            raise ValueError(f"Name not allowed: {child.id}")


def calculate(payload: CalculatorInput, _: ToolContext) -> CalculatorOutput:
    """Evaluate safe math expression."""

    expression = payload.expression.strip()
    if not expression:
        raise ValueError("Expression cannot be empty")

    parsed_expression = expression
    try:
        tree = ast.parse(parsed_expression, mode="eval")
    except SyntaxError:
        match = re.search(r"([0-9][0-9\s\+\-\*\/\(\)\.\%]+)", expression)
        if not match:
            raise
        parsed_expression = match.group(1).strip()
        tree = ast.parse(parsed_expression, mode="eval")

    _validate_ast(tree)
    code = compile(tree, "<calculator>", "eval")
    result = eval(code, {"__builtins__": {}}, _ALLOWED_FUNCTIONS)  # noqa: S307 - guarded AST
    return CalculatorOutput(result=float(result), expression=parsed_expression)


spec = ToolSpec(
    name="calculator",
    description="Safe scientific calculator with arithmetic, trig, logs, and basic statistics",
    input_model=CalculatorInput,
    output_model=CalculatorOutput,
    tags=["math", "reasoning"],
)
