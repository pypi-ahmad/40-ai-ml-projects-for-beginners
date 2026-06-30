from __future__ import annotations

import json
import logging
import re
from typing import Any

from prompts.library import PromptLibrary
from resources.library import ResourceLibrary
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)
_RESERVED_PROMPT_FIELDS = {"role", "objective", "constraints", "expected_output"}


class MCPSDKAdapter:
    def __init__(
        self,
        server_name: str,
        tools: ToolRegistry,
        resources: ResourceLibrary,
        prompts: PromptLibrary,
    ) -> None:
        self.server_name = server_name
        self.tools = tools
        self.resources = resources
        self.prompts = prompts
        self.server = None
        self.lowlevel_server = None
        self._lowlevel_registered = False

        try:
            from mcp.server import MCPServer

            self.server = MCPServer(server_name)
        except Exception:
            try:
                from mcp.server.mcpserver import MCPServer

                self.server = MCPServer(server_name)
            except Exception as high_level_exc:
                try:
                    from mcp.server import Server

                    self.lowlevel_server = Server(server_name)
                    logger.info(
                        "Using low-level official mcp server fallback because high-level runtime is unavailable: %s",
                        high_level_exc,
                    )
                except Exception as low_level_exc:
                    logger.info(
                        "Official mcp runtime unavailable (high-level and low-level): %s / %s",
                        high_level_exc,
                        low_level_exc,
                    )

    def available(self) -> bool:
        return self.server is not None or self.lowlevel_server is not None

    def _build_tool_handler(self, tool_name: str, input_schema: dict[str, Any]):
        properties = input_schema.get("properties", {})
        required = set(input_schema.get("required", []))
        field_names = [name for name in properties if name.isidentifier()]

        namespace: dict[str, Any] = {"registry": self.tools}

        if not field_names:
            source = (
                "async def handler():\n"
                f"    return await registry.call('{tool_name}', {{}})\n"
            )
        else:
            params: list[str] = []
            payload_entries: list[str] = []
            for field in field_names:
                if field in required:
                    params.append(field)
                else:
                    params.append(f"{field}=None")
                payload_entries.append(f'\"{field}\": {field}')

            source = (
                f"async def handler({', '.join(params)}):\n"
                f"    payload = {{{', '.join(payload_entries)}}}\n"
                "    payload = {k: v for k, v in payload.items() if v is not None}\n"
                f"    return await registry.call('{tool_name}', payload)\n"
            )

        exec(source, namespace)
        fn = namespace["handler"]
        fn.__name__ = f"tool_{tool_name}"
        return fn

    def _extract_prompt_variables(self, prompt_name: str) -> list[str]:
        prompt_def = self.prompts.get(prompt_name)
        fields: list[str] = []
        for match in re.findall(r"{([a-zA-Z_][a-zA-Z0-9_]*)}", prompt_def.template):
            if match in _RESERVED_PROMPT_FIELDS or match in fields:
                continue
            fields.append(match)
        return fields

    def _build_prompt_handler(self, prompt_name: str):
        fields = self._extract_prompt_variables(prompt_name)

        namespace: dict[str, Any] = {"prompts": self.prompts}
        if not fields:
            source = (
                "def handler():\n"
                f"    return prompts.render('{prompt_name}', {{}})\n"
            )
        else:
            params = ", ".join(f"{field}: str" for field in fields)
            payload_entries = ", ".join(f'\"{field}\": {field}' for field in fields)
            source = (
                f"def handler({params}):\n"
                f"    payload = {{{payload_entries}}}\n"
                f"    return prompts.render('{prompt_name}', payload)\n"
            )

        exec(source, namespace)
        fn = namespace["handler"]
        fn.__name__ = f"prompt_{prompt_name}"
        return fn

    def _register_lowlevel_capabilities(self) -> None:
        if self.lowlevel_server is None or self._lowlevel_registered:
            return

        import mcp.types as types

        @self.lowlevel_server.list_tools()
        async def _list_tools() -> list[types.Tool]:
            result: list[types.Tool] = []
            for tool in self.tools.list():
                ann = tool.get("annotations", {})
                annotations = types.ToolAnnotations(
                    readOnlyHint=ann.get("readOnlyHint"),
                    destructiveHint=ann.get("destructiveHint"),
                    idempotentHint=ann.get("idempotentHint"),
                    openWorldHint=ann.get("openWorldHint"),
                )
                result.append(
                    types.Tool(
                        name=tool["name"],
                        description=tool["description"],
                        inputSchema=tool["input_schema"],
                        annotations=annotations,
                    )
                )
            return result

        @self.lowlevel_server.call_tool(validate_input=False)
        async def _call_tool(name: str, arguments: dict[str, Any] | None = None):
            payload = arguments or {}
            result = await self.tools.call(name, payload)
            return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

        @self.lowlevel_server.list_resources()
        async def _list_resources() -> list[types.Resource]:
            result: list[types.Resource] = []
            for item in self.resources.list():
                result.append(
                    types.Resource(
                        name=item["name"],
                        uri=item["uri"],
                        description=item["description"],
                        mimeType=item["mime_type"],
                    )
                )
            return result

        @self.lowlevel_server.read_resource()
        async def _read_resource(uri):
            payload = self.resources.read(str(uri))
            return payload["content"]

        @self.lowlevel_server.list_prompts()
        async def _list_prompts() -> list[types.Prompt]:
            prompts: list[types.Prompt] = []
            for name in self.prompts.names():
                definition = self.prompts.get(name)
                fields = self._extract_prompt_variables(name)
                arguments = [types.PromptArgument(name=field, required=True) for field in fields] or None
                prompts.append(
                    types.Prompt(
                        name=name,
                        description=definition.objective,
                        arguments=arguments,
                    )
                )
            return prompts

        @self.lowlevel_server.get_prompt()
        async def _get_prompt(name: str, arguments: dict[str, str] | None = None):
            if name not in self.prompts.names():
                raise ValueError(f"Unknown prompt: {name}")
            definition = self.prompts.get(name)
            rendered = self.prompts.render(name, dict(arguments or {}))
            return types.GetPromptResult(
                description=definition.objective,
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(type="text", text=rendered),
                    )
                ],
            )

        self._lowlevel_registered = True

    def register_capabilities(self) -> None:
        if self.server is not None:
            for tool in self.tools.list():
                name = tool["name"]
                description = tool["description"]
                fn = self._build_tool_handler(name, tool["input_schema"])
                fn.__doc__ = description
                try:
                    self.server.tool(name=name, description=description)(fn)
                except Exception as exc:
                    logger.warning("Failed registering MCP SDK tool %s: %s", name, exc)

            for item in self.resources.list():
                uri = item["uri"]
                description = item["description"]

                def _resource_loader(__uri: str = uri) -> str:
                    return self.resources.read(__uri)["content"]

                _resource_loader.__name__ = f"resource_{uri.replace(':', '_').replace('/', '_')}"
                _resource_loader.__doc__ = description
                try:
                    self.server.resource(uri)(_resource_loader)
                except Exception as exc:
                    logger.warning("Failed registering MCP SDK resource %s: %s", uri, exc)

            for name in self.prompts.names():
                prompt_def = self.prompts.get(name)
                fn = self._build_prompt_handler(name)
                fn.__doc__ = prompt_def.objective
                try:
                    self.server.prompt(name=name)(fn)
                except Exception as exc:
                    logger.warning("Failed registering MCP SDK prompt %s: %s", name, exc)
            return

        self._register_lowlevel_capabilities()

    def run(self, transport: str, host: str, port: int, sse_path: str, http_path: str) -> None:
        if self.server is not None:
            kwargs: dict[str, Any] = {"transport": transport}
            if transport in {"http", "sse", "streamable-http"}:
                kwargs.update({"host": host, "port": port})
            if transport == "sse":
                kwargs["sse_path"] = sse_path
            if transport in {"http", "streamable-http"}:
                kwargs["streamable_http_path"] = http_path

            self.server.run(**kwargs)
            return

        if self.lowlevel_server is None:
            raise RuntimeError("Official mcp runtime not available")

        self._register_lowlevel_capabilities()

        if transport not in {"stdio"}:
            raise RuntimeError(
                "Low-level official mcp fallback currently supports only stdio transport in this environment"
            )

        from mcp.server.stdio import stdio_server
        import anyio

        async def _run_stdio() -> None:
            async with stdio_server() as streams:
                await self.lowlevel_server.run(
                    streams[0],
                    streams[1],
                    self.lowlevel_server.create_initialization_options(),
                )

        anyio.run(_run_stdio)
