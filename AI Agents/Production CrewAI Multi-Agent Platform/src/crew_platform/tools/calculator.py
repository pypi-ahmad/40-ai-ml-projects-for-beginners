"""Safe scientific calculator tool."""

from __future__ import annotations

import ast
import math
import statistics
from typing import Any

from pydantic import BaseModel, Field

from crew_platform.tools.base import BaseTool


class CalculatorInput(BaseModel):
    """Calculator input payload."""

    expression: str = Field(min_length=1)


class CalculatorOutput(BaseModel):
    """Calculator output payload."""

    value: float
    normalized_expression: str


_ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "pow": pow,
    "floor": math.floor,
    "ceil": math.ceil,
    "mean": statistics.mean,
    "median": statistics.median,
    "stdev": statistics.stdev,
    "variance": statistics.variance,
}
_SEQUENCE_FUNCTIONS = {"mean", "median", "stdev", "variance"}

_ALLOWED_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Tuple,
    ast.List,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.FloorDiv,
    ast.USub,
    ast.UAdd,
)


class _SafeEvaluator:
    def eval_expression(self, expression: str) -> float:
        tree = ast.parse(expression, mode="eval")
        self._validate_tree(tree)
        value = self._eval_node(tree.body)
        return float(value)

    def _validate_tree(self, node: ast.AST) -> None:
        for child in ast.walk(node):
            if not isinstance(child, _ALLOWED_NODES):
                raise ValueError(f"Unsupported expression element: {type(child).__name__}")

    def _eval_node(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants allowed")

        if isinstance(node, ast.Name):
            if node.id in _ALLOWED_CONSTANTS:
                return _ALLOWED_CONSTANTS[node.id]
            raise ValueError(f"Unknown symbol: {node.id}")

        if isinstance(node, ast.UnaryOp):
            value = self._eval_node(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return +value
            raise ValueError("Unsupported unary operator")

        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left**right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            raise ValueError("Unsupported binary operator")

        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._eval_node(element) for element in node.elts]

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function names allowed")
            fn_name = node.func.id
            if fn_name not in _ALLOWED_FUNCTIONS:
                raise ValueError(f"Function not allowed: {fn_name}")
            args = [self._eval_node(arg) for arg in node.args]
            if fn_name in _SEQUENCE_FUNCTIONS and args and not isinstance(args[0], list):
                args = [args]
            return _ALLOWED_FUNCTIONS[fn_name](*args)

        raise ValueError(f"Unsupported node: {type(node).__name__}")


class CalculatorTool(BaseTool[CalculatorInput, CalculatorOutput]):
    """Safe arithmetic/scientific calculator."""

    name = "calculator"
    description = "Evaluate safe arithmetic and scientific expressions"
    input_model = CalculatorInput
    output_model = CalculatorOutput

    def __init__(self) -> None:
        self._evaluator = _SafeEvaluator()

    async def run(self, payload: CalculatorInput) -> CalculatorOutput:
        expr = payload.expression.strip()
        value = self._evaluator.eval_expression(expr)
        return CalculatorOutput(value=value, normalized_expression=expr)
