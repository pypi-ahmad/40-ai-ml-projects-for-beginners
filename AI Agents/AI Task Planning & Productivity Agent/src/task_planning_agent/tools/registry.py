"""Tool registry for local-first execution."""

from __future__ import annotations

from task_planning_agent.schemas import ToolResult
from task_planning_agent.tools.base import Tool
from task_planning_agent.tools.builtins import (
    CalculatorTool,
    DateTimeTool,
    FileReaderTool,
    NotesTool,
    ReminderTool,
    TimerTool,
    WeatherTool,
    WebSearchTool,
)


class ToolRegistry:
    """In-memory registry of available tools."""

    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {
            tool.name: tool
            for tool in [
                DateTimeTool(),
                FileReaderTool(),
                CalculatorTool(),
                WeatherTool(),
                WebSearchTool(),
                NotesTool(),
                ReminderTool(),
                TimerTool(),
            ]
        }

    def list_tools(self) -> list[str]:
        return sorted(self.tools.keys())

    def run(self, name: str, **kwargs: object) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(tool_name=name, success=False, output=None, error="tool not found")
        return tool.run(**kwargs)
