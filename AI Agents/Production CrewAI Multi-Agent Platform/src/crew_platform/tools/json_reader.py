"""JSON reader alias."""

from __future__ import annotations

from crew_platform.tools.json_explorer import JSONExplorerTool


class JSONReaderTool(JSONExplorerTool):
    name = "json_reader"
    description = "Read JSON structure"
