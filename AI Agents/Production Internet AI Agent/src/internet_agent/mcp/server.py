"""Minimal MCP-compatible stdio server exposing internet-agent tools."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from internet_agent.config import get_settings
from internet_agent.services.agent_service import InternetAgentService


class MCPServer:
    """JSON-RPC server over stdio for tool listing and invocation."""

    def __init__(self) -> None:
        self.service = InternetAgentService(settings=get_settings())

    async def handle(self, req: dict[str, Any]) -> dict[str, Any]:
        method = req.get("method", "")
        params = req.get("params", {})
        req_id = req.get("id")

        try:
            if method == "initialize":
                result = {
                    "protocolVersion": "2025-06-18",
                    "serverInfo": {"name": "internet-agent-mcp", "version": "0.1.0"},
                    "capabilities": {"tools": {}},
                }
            elif method in {"tools/list", "list_tools"}:
                tools = [tool.model_dump(mode="json") for tool in self.service.tool_registry.discover()]
                result = {"tools": tools}
            elif method in {"tools/call", "call_tool"}:
                name = params.get("name", "")
                arguments = params.get("arguments", {})
                session_id = params.get("session_id", "mcp")
                output = await self.service.tool_registry.invoke(
                    session_id=session_id,
                    name=name,
                    payload=arguments,
                )
                result = {"content": [{"type": "json", "json": output}]}
            else:
                raise ValueError(f"Unsupported method: {method}")

            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        except Exception as exc:  # noqa: BLE001
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(exc)},
            }

    async def run(self) -> None:
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue
            response = await self.handle(req)
            sys.stdout.write(json.dumps(response, ensure_ascii=True) + "\n")
            sys.stdout.flush()


def run_stdio_server() -> None:
    asyncio.run(MCPServer().run())


if __name__ == "__main__":
    run_stdio_server()
