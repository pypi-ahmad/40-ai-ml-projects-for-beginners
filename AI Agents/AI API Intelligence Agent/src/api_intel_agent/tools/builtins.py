"""Built-in tools required by API intelligence agent."""

from __future__ import annotations

import ast
import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from api_intel_agent.tools.base import ToolResult

SAFE_ROOT = Path.cwd()


class SafeEval(ast.NodeVisitor):
    allowed_nodes = {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.FloorDiv,
        ast.Constant,
        ast.USub,
        ast.UAdd,
        ast.Call,
        ast.Name,
    }
    allowed_names = {k: getattr(math, k) for k in ("sqrt", "log", "sin", "cos", "tan", "pi", "e")}

    def visit(self, node: ast.AST) -> Any:
        if type(node) not in self.allowed_nodes:
            raise ValueError(f"unsupported expression node: {type(node).__name__}")
        return super().visit(node)

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numeric constants allowed")

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id not in self.allowed_names:
            raise ValueError("name not allowed")
        return self.allowed_names[node.id]

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        value = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
        raise ValueError("unsupported unary op")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        raise ValueError("unsupported operator")

    def visit_Call(self, node: ast.Call) -> Any:
        fn = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]
        return fn(*args)


def _safe_path(path: str) -> Path:
    target = (SAFE_ROOT / path).resolve()
    if not str(target).startswith(str(SAFE_ROOT.resolve())):
        raise ValueError("path outside workspace")
    return target


async def http_client_tool(url: str, method: str = "GET", params: dict[str, Any] | None = None) -> ToolResult:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.request(method, url, params=params)
            response.raise_for_status()
            payload = response.json() if "json" in response.headers.get("content-type", "") else response.text
            return ToolResult(name="http_client", success=True, payload={"response": payload})
    except Exception as exc:
        return ToolResult(name="http_client", success=False, payload={}, error=str(exc))


async def calculator_tool(expression: str) -> ToolResult:
    try:
        tree = ast.parse(expression, mode="eval")
        result = SafeEval().visit(tree)
        return ToolResult(name="calculator", success=True, payload={"result": result})
    except Exception as exc:
        return ToolResult(name="calculator", success=False, payload={}, error=str(exc))


async def file_reader_tool(path: str) -> ToolResult:
    try:
        content = _safe_path(path).read_text()
        return ToolResult(name="file_reader", success=True, payload={"content": content})
    except Exception as exc:
        return ToolResult(name="file_reader", success=False, payload={}, error=str(exc))


async def markdown_generator_tool(title: str, sections: list[dict[str, str]]) -> ToolResult:
    md = [f"# {title}"]
    for section in sections:
        md.append(f"\n## {section.get('heading', 'Section')}\n")
        md.append(section.get("body", ""))
    return ToolResult(name="markdown_generator", success=True, payload={"markdown": "\n".join(md)})


async def csv_export_tool(path: str, rows: list[dict[str, Any]]) -> ToolResult:
    try:
        target = _safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        headers = sorted({key for row in rows for key in row}) if rows else []
        with target.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        return ToolResult(name="csv_export", success=True, payload={"path": str(target), "rows": len(rows)})
    except Exception as exc:
        return ToolResult(name="csv_export", success=False, payload={}, error=str(exc))


async def json_export_tool(path: str, payload: dict[str, Any]) -> ToolResult:
    try:
        target = _safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, default=str))
        return ToolResult(name="json_export", success=True, payload={"path": str(target)})
    except Exception as exc:
        return ToolResult(name="json_export", success=False, payload={}, error=str(exc))


async def pdf_report_tool(path: str, title: str, lines: list[str]) -> ToolResult:
    try:
        target = _safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        pdf = canvas.Canvas(str(target), pagesize=letter)
        y = 760
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(72, y, title)
        pdf.setFont("Helvetica", 10)
        y -= 24
        for line in lines:
            if y < 72:
                pdf.showPage()
                y = 760
            pdf.drawString(72, y, line[:110])
            y -= 14
        pdf.save()
        return ToolResult(name="pdf_report", success=True, payload={"path": str(target)})
    except Exception as exc:
        return ToolResult(name="pdf_report", success=False, payload={}, error=str(exc))


async def chart_generator_tool(path: str, title: str, rows: list[dict[str, Any]], x: str, y: str) -> ToolResult:
    try:
        target = _safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig = px.line(rows, x=x, y=y, title=title) if rows else px.line(title=title)
        fig.write_html(target)
        return ToolResult(name="chart_generator", success=True, payload={"path": str(target)})
    except Exception as exc:
        return ToolResult(name="chart_generator", success=False, payload={}, error=str(exc))


async def web_search_tool(query: str) -> ToolResult:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "utf8": "",
        "format": "json",
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        results = data.get("query", {}).get("search", [])
        return ToolResult(name="web_search", success=True, payload={"results": results[:5]})
    except Exception as exc:
        return ToolResult(name="web_search", success=False, payload={}, error=str(exc))


async def datetime_tool(timezone: str = "UTC") -> ToolResult:
    _ = timezone
    now = datetime.now(UTC).isoformat()
    return ToolResult(name="datetime", success=True, payload={"utc_now": now})


async def unit_conversion_tool(value: float, from_unit: str, to_unit: str) -> ToolResult:
    conversion = {
        ("km", "m"): 1000,
        ("m", "km"): 0.001,
        ("c", "f"): None,
        ("f", "c"): None,
    }
    key = (from_unit.lower(), to_unit.lower())
    try:
        if key == ("c", "f"):
            converted = (value * 9 / 5) + 32
        elif key == ("f", "c"):
            converted = (value - 32) * 5 / 9
        elif key in conversion:
            converted = value * float(conversion[key])
        else:
            return ToolResult(
                name="unit_conversion",
                success=False,
                payload={},
                error=f"unsupported conversion: {from_unit}->{to_unit}",
            )
        return ToolResult(name="unit_conversion", success=True, payload={"value": converted})
    except Exception as exc:
        return ToolResult(name="unit_conversion", success=False, payload={}, error=str(exc))
