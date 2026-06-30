from __future__ import annotations

import ast
import asyncio
import csv
import json
import os
import platform
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import psutil
from duckduckgo_search import DDGS

from config.settings import Settings
from memory.service import MemoryService
from prompts.library import PromptLibrary
from resources.library import ResourceLibrary
from tools.base import ToolDefinition
from tools.sandbox import execute_python_sandboxed, run_shell_whitelisted


def _project_root(settings: Settings) -> Path:
    return Path(settings.plugins.directory).parents[0]


def _resolve_safe_path(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    if not str(path).startswith(str(root.resolve())):
        raise ValueError("Path escapes project root")
    return path


def _safe_eval(expr: str) -> float | int:
    tree = ast.parse(expr, mode="eval")
    allowed_nodes = {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.FloorDiv,
        ast.USub,
        ast.UAdd,
        ast.Load,
    }
    for node in ast.walk(tree):
        if type(node) not in allowed_nodes:
            raise ValueError("Expression contains unsupported operations")
    value = eval(compile(tree, "<calculator>", "eval"), {"__builtins__": {}}, {})
    return value


def _report_path(root: Path, basename: str, suffix: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in basename).strip("_")
    if not safe:
        safe = "report"
    target = root / "reports" / f"{safe}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def build_builtin_tools(
    settings: Settings,
    memory: MemoryService,
    resources: ResourceLibrary,
    prompts: PromptLibrary,
) -> list[ToolDefinition]:
    root = _project_root(settings)

    async def calculator(expression: str) -> dict[str, Any]:
        value = _safe_eval(expression)
        return {"ok": True, "result": value}

    async def weather(location: str) -> dict[str, Any]:
        # Open-Meteo geocode + forecast with no API key.
        async with httpx.AsyncClient(timeout=15) as client:
            geocode = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1},
            )
            geocode.raise_for_status()
            payload = geocode.json()
            results = payload.get("results") or []
            if not results:
                return {"ok": False, "error": f"Location not found: {location}"}
            first = results[0]
            forecast = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": first["latitude"],
                    "longitude": first["longitude"],
                    "current": "temperature_2m,wind_speed_10m",
                },
            )
            forecast.raise_for_status()
            data = forecast.json()

        return {
            "ok": True,
            "location": {
                "name": first.get("name"),
                "country": first.get("country"),
                "latitude": first.get("latitude"),
                "longitude": first.get("longitude"),
            },
            "current": data.get("current", {}),
        }

    async def file_reader(path: str) -> dict[str, Any]:
        target = _resolve_safe_path(root, path)
        content = target.read_text(encoding="utf-8")
        return {"ok": True, "path": str(target), "content": content}

    async def file_writer(path: str, content: str) -> dict[str, Any]:
        target = _resolve_safe_path(root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"ok": True, "path": str(target), "bytes": len(content.encode("utf-8"))}

    async def csv_reader(path: str, limit: int = 100) -> dict[str, Any]:
        target = _resolve_safe_path(root, path)
        with target.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = []
            for idx, row in enumerate(reader):
                rows.append(dict(row))
                if idx + 1 >= limit:
                    break
        return {"ok": True, "path": str(target), "rows": rows}

    async def json_reader(path: str) -> dict[str, Any]:
        target = _resolve_safe_path(root, path)
        data = json.loads(target.read_text(encoding="utf-8"))
        return {"ok": True, "path": str(target), "data": data}

    async def sqlite_query(query: str, db_path: str | None = None) -> dict[str, Any]:
        database = Path(db_path) if db_path else Path(settings.memory.sqlite_path)
        conn = sqlite3.connect(database)
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description or []]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            return {"ok": True, "rows": rows, "row_count": len(rows)}
        finally:
            conn.close()

    async def chroma_search(query: str, top_k: int = 5) -> dict[str, Any]:
        results = memory.semantic_search(query, top_k=top_k)
        return {"ok": True, "results": results}

    async def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
        def _search() -> list[dict[str, Any]]:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        try:
            results = await asyncio.to_thread(_search)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "degraded": True}
        return {"ok": True, "results": results}

    async def github_search(query: str, max_results: int = 5) -> dict[str, Any]:
        headers = {"Accept": "application/vnd.github+json"}
        token = os.environ.get(settings.external.github_token_env)
        if token:
            headers["Authorization"] = f"Bearer {token}"

        async with httpx.AsyncClient(timeout=20, headers=headers) as client:
            response = await client.get(
                "https://api.github.com/search/repositories",
                params={"q": query, "per_page": max_results},
            )

        if response.status_code >= 400:
            return {"ok": False, "error": response.text, "degraded": True}

        payload = response.json()
        items = payload.get("items", [])
        return {
            "ok": True,
            "results": [
                {
                    "full_name": item.get("full_name"),
                    "description": item.get("description"),
                    "url": item.get("html_url"),
                    "stars": item.get("stargazers_count"),
                }
                for item in items
            ],
        }

    async def news_search(query: str, max_results: int = 5) -> dict[str, Any]:
        key = os.environ.get(settings.external.news_api_key_env)
        if not key:
            return {
                "ok": False,
                "degraded": True,
                "error": f"Missing {settings.external.news_api_key_env}",
                "results": [],
            }

        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(
                "https://newsapi.org/v2/everything",
                params={"q": query, "pageSize": max_results, "sortBy": "publishedAt", "apiKey": key},
            )

        if response.status_code >= 400:
            return {"ok": False, "error": response.text, "degraded": True}

        payload = response.json()
        return {
            "ok": True,
            "results": [
                {
                    "title": item.get("title"),
                    "source": item.get("source", {}).get("name"),
                    "url": item.get("url"),
                    "published_at": item.get("publishedAt"),
                }
                for item in payload.get("articles", [])
            ],
        }

    async def python_executor(code: str, timeout_seconds: int = 5) -> dict[str, Any]:
        return await execute_python_sandboxed(code=code, timeout_seconds=timeout_seconds, memory_limit_mb=128)

    async def markdown_generator(title: str, sections: list[dict[str, str]]) -> dict[str, Any]:
        lines = [f"# {title}", ""]
        for section in sections:
            lines.append(f"## {section.get('heading', 'Section')}")
            lines.append(section.get("content", ""))
            lines.append("")
        return {"ok": True, "markdown": "\n".join(lines).strip()}

    async def report_generator(title: str, body: str, save: bool = True) -> dict[str, Any]:
        report = f"# {title}\n\nGenerated at: {datetime.now(UTC).isoformat()}\n\n{body}\n"
        if not save:
            return {"ok": True, "report": report}

        path = _report_path(root, title, "md")
        path.write_text(report, encoding="utf-8")
        memory.store_semantic_memory(report, metadata={"type": "report", "path": str(path)})
        return {"ok": True, "path": str(path), "report": report}

    async def pdf_generator(title: str, body: str) -> dict[str, Any]:
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
        except Exception as exc:
            return {"ok": False, "error": str(exc), "degraded": True}

        path = _report_path(root, title, "pdf")
        c = canvas.Canvas(str(path), pagesize=letter)
        text = c.beginText(40, 760)
        text.textLine(title)
        text.textLine("")
        for line in body.splitlines():
            text.textLine(line)
        c.drawText(text)
        c.save()
        return {"ok": True, "path": str(path)}

    async def directory_search(path: str = ".", pattern: str = "*") -> dict[str, Any]:
        base = _resolve_safe_path(root, path)
        matches = [str(p.relative_to(root)) for p in base.rglob(pattern) if p.is_file()]
        return {"ok": True, "matches": matches[:500], "count": len(matches)}

    async def code_search(pattern: str, path: str = "src") -> dict[str, Any]:
        target = _resolve_safe_path(root, path)
        rg_bin = subprocess.run(["which", "rg"], capture_output=True, text=True, check=False).stdout.strip()
        if rg_bin:
            cmd = ["rg", "-n", pattern, str(target)]
            completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if completed.returncode not in {0, 1}:
                return {"ok": False, "error": completed.stderr.strip()}
            matches = completed.stdout.splitlines()
            return {"ok": True, "matches": matches[:500], "count": len(matches)}

        matches: list[str] = []
        for file_path in target.rglob("*.py"):
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for idx, line in enumerate(content.splitlines(), start=1):
                if pattern in line:
                    matches.append(f"{file_path}:{idx}:{line.strip()}")
        return {"ok": True, "matches": matches[:500], "count": len(matches)}

    async def shell_command(command: str) -> dict[str, Any]:
        allowed = set(settings.shell.whitelist)
        result = run_shell_whitelisted(command=command, allowed=allowed, cwd=str(root), timeout=10)
        return result

    async def system_information() -> dict[str, Any]:
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage(str(root))

        gpu_info: list[dict[str, Any]] = []
        nvidia = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if nvidia.returncode == 0:
            for row in nvidia.stdout.splitlines():
                parts = [part.strip() for part in row.split(",")]
                if len(parts) == 4:
                    gpu_info.append(
                        {
                            "name": parts[0],
                            "memory_total_mb": float(parts[1]),
                            "memory_used_mb": float(parts[2]),
                            "utilization_gpu_pct": float(parts[3]),
                        }
                    )

        return {
            "ok": True,
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_percent": cpu,
            "memory_total": mem.total,
            "memory_used": mem.used,
            "memory_percent": mem.percent,
            "disk_total": disk.total,
            "disk_used": disk.used,
            "disk_percent": disk.percent,
            "gpu": gpu_info,
        }

    return [
        ToolDefinition(
            name="calculator",
            description="Safely evaluate arithmetic expression",
            input_schema={"type": "object", "required": ["expression"], "properties": {"expression": {"type": "string"}}},
            examples=[{"expression": "2 + 2 * 10"}],
            handler=calculator,
            read_only=True,
            open_world=False,
        ),
        ToolDefinition(
            name="weather",
            description="Fetch current weather using Open-Meteo",
            input_schema={"type": "object", "required": ["location"], "properties": {"location": {"type": "string"}}},
            examples=[{"location": "Bengaluru"}],
            handler=weather,
            open_world=True,
        ),
        ToolDefinition(
            name="file_reader",
            description="Read file from project workspace",
            input_schema={"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}},
            handler=file_reader,
        ),
        ToolDefinition(
            name="file_writer",
            description="Write file to project workspace",
            input_schema={
                "type": "object",
                "required": ["path", "content"],
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            },
            handler=file_writer,
            read_only=False,
            destructive=False,
            idempotent=False,
        ),
        ToolDefinition(
            name="csv_reader",
            description="Read CSV file as JSON rows",
            input_schema={
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}, "limit": {"type": "integer", "default": 100}},
            },
            handler=csv_reader,
        ),
        ToolDefinition(
            name="json_reader",
            description="Read JSON file",
            input_schema={"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}},
            handler=json_reader,
        ),
        ToolDefinition(
            name="sqlite_query",
            description="Run SQL query against SQLite",
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string"}, "db_path": {"type": "string"}},
            },
            handler=sqlite_query,
            read_only=False,
            idempotent=False,
        ),
        ToolDefinition(
            name="chroma_search",
            description="Semantic search against Chroma memory",
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 5}},
            },
            handler=chroma_search,
        ),
        ToolDefinition(
            name="web_search",
            description="Web search using DuckDuckGo",
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                },
            },
            handler=web_search,
            open_world=True,
        ),
        ToolDefinition(
            name="github_search",
            description="Search GitHub repositories",
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                },
            },
            handler=github_search,
            open_world=True,
        ),
        ToolDefinition(
            name="news_search",
            description="Search latest news articles",
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                },
            },
            handler=news_search,
            open_world=True,
        ),
        ToolDefinition(
            name="python_executor",
            description="Execute Python in hardened sandbox",
            input_schema={
                "type": "object",
                "required": ["code"],
                "properties": {
                    "code": {"type": "string"},
                    "timeout_seconds": {"type": "integer", "default": 5},
                },
            },
            handler=python_executor,
            read_only=False,
            idempotent=False,
        ),
        ToolDefinition(
            name="markdown_generator",
            description="Generate markdown from title and sections",
            input_schema={
                "type": "object",
                "required": ["title", "sections"],
                "properties": {
                    "title": {"type": "string"},
                    "sections": {"type": "array", "items": {"type": "object"}},
                },
            },
            handler=markdown_generator,
            read_only=False,
        ),
        ToolDefinition(
            name="report_generator",
            description="Generate and optionally persist markdown report",
            input_schema={
                "type": "object",
                "required": ["title", "body"],
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "save": {"type": "boolean", "default": True},
                },
            },
            handler=report_generator,
            read_only=False,
            idempotent=False,
        ),
        ToolDefinition(
            name="pdf_generator",
            description="Generate PDF report",
            input_schema={
                "type": "object",
                "required": ["title", "body"],
                "properties": {"title": {"type": "string"}, "body": {"type": "string"}},
            },
            handler=pdf_generator,
            read_only=False,
            idempotent=False,
        ),
        ToolDefinition(
            name="directory_search",
            description="Search directory with glob pattern",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "pattern": {"type": "string", "default": "*"},
                },
            },
            handler=directory_search,
        ),
        ToolDefinition(
            name="code_search",
            description="Search code with ripgrep fallback",
            input_schema={
                "type": "object",
                "required": ["pattern"],
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "src"},
                },
            },
            handler=code_search,
        ),
        ToolDefinition(
            name="shell_command",
            description="Execute command from safe whitelist",
            input_schema={"type": "object", "required": ["command"], "properties": {"command": {"type": "string"}}},
            handler=shell_command,
            read_only=False,
            idempotent=False,
            destructive=True,
        ),
        ToolDefinition(
            name="system_information",
            description="Collect system CPU, memory, disk, and GPU info",
            input_schema={"type": "object", "properties": {}},
            handler=system_information,
        ),
    ]
