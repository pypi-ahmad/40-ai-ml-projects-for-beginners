"""Built-in local-first tools."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from task_planning_agent.schemas import ToolResult
from task_planning_agent.tools.base import Tool


class DateTimeTool(Tool):
    name = "datetime"
    description = "Return current UTC and local date/time"

    def run(self, **kwargs: Any) -> ToolResult:
        now = datetime.now()
        return ToolResult(tool_name=self.name, success=True, output={"now": now.isoformat()})


class FileReaderTool(Tool):
    name = "file_reader"
    description = "Read UTF-8 local file"

    def run(self, **kwargs: Any) -> ToolResult:
        path = kwargs.get("path")
        if not path:
            return ToolResult(tool_name=self.name, success=False, output=None, error="path is required")
        try:
            text = Path(path).read_text(encoding="utf-8")
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, output=None, error=str(exc))
        return ToolResult(tool_name=self.name, success=True, output=text)


class CalculatorTool(Tool):
    name = "calculator"
    description = "Safe arithmetic evaluator"

    def run(self, **kwargs: Any) -> ToolResult:
        expr = kwargs.get("expression", "")
        safe_globals = {"__builtins__": {}}
        safe_locals = {"sqrt": math.sqrt, "pow": pow, "abs": abs, "round": round}
        try:
            value = eval(expr, safe_globals, safe_locals)
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, output=None, error=str(exc))
        return ToolResult(tool_name=self.name, success=True, output={"result": value})


class NotesTool(Tool):
    name = "notes"
    description = "Create lightweight local note"

    def run(self, **kwargs: Any) -> ToolResult:
        text = str(kwargs.get("text", "")).strip()
        if not text:
            return ToolResult(tool_name=self.name, success=False, output=None, error="text is required")
        notes_dir = Path("data/processed/notes")
        notes_dir.mkdir(parents=True, exist_ok=True)
        note_path = notes_dir / f"note-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.md"
        note_path.write_text(text, encoding="utf-8")
        return ToolResult(tool_name=self.name, success=True, output={"path": str(note_path)})


class ReminderTool(Tool):
    name = "reminder"
    description = "Create local reminder entry"

    def run(self, **kwargs: Any) -> ToolResult:
        payload = {
            "task": kwargs.get("task", ""),
            "remind_at": kwargs.get("remind_at", datetime.now(timezone.utc).isoformat()),
        }
        reminders_dir = Path("data/processed/reminders")
        reminders_dir.mkdir(parents=True, exist_ok=True)
        output_path = reminders_dir / "reminders.jsonl"
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{payload}\n")
        return ToolResult(tool_name=self.name, success=True, output=payload)


class TimerTool(Tool):
    name = "timer"
    description = "Generate timer metadata"

    def run(self, **kwargs: Any) -> ToolResult:
        minutes = int(kwargs.get("minutes", 25))
        return ToolResult(
            tool_name=self.name,
            success=True,
            output={"minutes": minutes, "message": f"Set focus timer for {minutes} minutes"},
        )


class WeatherTool(Tool):
    name = "weather"
    description = "Local-first weather stub"

    def run(self, **kwargs: Any) -> ToolResult:
        location = kwargs.get("location", "unknown")
        payload = {
            "location": location,
            "status": "offline_stub",
            "temperature_c": None,
            "note": "Configure online provider API key for live weather",
        }
        return ToolResult(tool_name=self.name, success=True, output=payload)


class WebSearchTool(Tool):
    name = "web_search"
    description = "Local-first web search stub"

    def run(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query", "")
        return ToolResult(
            tool_name=self.name,
            success=True,
            output={
                "query": query,
                "results": [],
                "note": "Online search provider not configured; returning empty local-safe result",
            },
        )
