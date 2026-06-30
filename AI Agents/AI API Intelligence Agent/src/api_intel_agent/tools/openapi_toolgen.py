"""OpenAPI/Swagger ingestion and generated tool registration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import yaml

from api_intel_agent.tools.base import ToolResult
from api_intel_agent.tools.registry import ToolRegistry, ToolSpec


def _load_spec(path_or_url: str) -> dict[str, Any]:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        response = httpx.get(path_or_url, timeout=30)
        response.raise_for_status()
        raw = response.text
    else:
        raw = Path(path_or_url).read_text()

    if path_or_url.endswith(".json"):
        return json.loads(raw)
    return yaml.safe_load(raw)


async def generated_openapi_tool(
    base_url: str,
    path: str,
    method: str,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> ToolResult:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.request(
                method.upper(),
                f"{base_url.rstrip('/')}/{path.lstrip('/')}",
                params=params,
                json=json_body,
            )
            response.raise_for_status()
            return ToolResult(
                name=f"openapi_{method}_{path.replace('/', '_')}",
                success=True,
                payload={"response": response.json()},
            )
    except Exception as exc:
        return ToolResult(name="openapi_generated", success=False, payload={}, error=str(exc))


def register_openapi_tools(registry: ToolRegistry, path_or_url: str, service_name: str) -> int:
    spec = _load_spec(path_or_url)
    paths = spec.get("paths", {}) if isinstance(spec, dict) else {}
    servers = spec.get("servers", []) if isinstance(spec, dict) else []
    base_url = servers[0].get("url") if servers and isinstance(servers[0], dict) else ""

    count = 0
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, op in methods.items():
            if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                continue
            operation_id = op.get("operationId") if isinstance(op, dict) else None
            tool_name = operation_id or f"{service_name}_{method}_{path.strip('/').replace('/', '_')}"

            async def _handler(
                _path: str = path,
                _method: str = method,
                params: dict[str, Any] | None = None,
                json_body: dict[str, Any] | None = None,
            ) -> ToolResult:
                return await generated_openapi_tool(
                    base_url=base_url,
                    path=_path,
                    method=_method,
                    params=params,
                    json_body=json_body,
                )

            registry.register(
                ToolSpec(
                    name=tool_name,
                    description=f"Generated from {service_name} OpenAPI: {method.upper()} {path}",
                    handler=_handler,
                )
            )
            count += 1
    return count
