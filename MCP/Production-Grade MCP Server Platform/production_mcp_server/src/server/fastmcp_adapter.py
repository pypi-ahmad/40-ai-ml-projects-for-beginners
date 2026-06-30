from __future__ import annotations

import logging
import re
from typing import Any

from prompts.library import PromptLibrary
from resources.library import ResourceLibrary
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)
_RESERVED_PROMPT_FIELDS = {"role", "objective", "constraints", "expected_output"}


class FastMCPAdapter:
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

        try:
            from fastmcp import FastMCP

            self.server = FastMCP(server_name)
        except Exception as exc:
            logger.warning("FastMCP unavailable: %s", exc)
            self.server = None

    def available(self) -> bool:
        return self.server is not None

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

    def _build_prompt_handler(self, prompt_name: str):
        prompt_def = self.prompts.get(prompt_name)
        fields = []
        for match in re.findall(r"{([a-zA-Z_][a-zA-Z0-9_]*)}", prompt_def.template):
            if match in _RESERVED_PROMPT_FIELDS or match in fields:
                continue
            fields.append(match)

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

    def register_capabilities(self) -> None:
        if self.server is None:
            return

        for tool in self.tools.list():
            name = tool["name"]
            description = tool["description"]
            fn = self._build_tool_handler(name, tool["input_schema"])
            fn.__doc__ = description

            try:
                from fastmcp.tools import Tool

                self.server.add_tool(Tool.from_function(fn, name=name, description=description))
            except Exception:
                try:
                    self.server.tool(name=name, description=description)(fn)
                except Exception as exc:
                    logger.warning("Failed to register FastMCP tool %s: %s", name, exc)

        for item in self.resources.list():
            uri = item["uri"]
            name = item["name"]
            description = item["description"]

            # FastMCP resource decorator expects URI template with params.
            template_uri = uri if "{" in uri else f"{uri}?k={{k}}"

            def _resource_loader(k: str = "default", __uri: str = uri) -> str:
                _ = k
                payload = self.resources.read(__uri)
                return payload["content"]

            _resource_loader.__name__ = f"resource_{name.lower().replace(' ', '_')}"
            _resource_loader.__doc__ = description

            try:
                self.server.resource(uri=template_uri, name=name, description=description)(_resource_loader)
            except Exception:
                try:
                    self.server.resource(template_uri)(_resource_loader)
                except Exception as exc:
                    logger.warning("Failed to register FastMCP resource %s: %s", uri, exc)

        for name in self.prompts.names():
            prompt_def = self.prompts.get(name)
            fn = self._build_prompt_handler(name)
            fn.__doc__ = prompt_def.objective

            try:
                self.server.prompt(name=name, description=prompt_def.objective)(fn)
            except Exception as exc:
                logger.warning("Failed to register FastMCP prompt %s: %s", name, exc)

    def run(self, transport: str, host: str, port: int, sse_path: str, http_path: str) -> None:
        if self.server is None:
            raise RuntimeError("FastMCP runtime not available")

        kwargs: dict[str, Any] = {"transport": transport, "show_banner": False}
        if transport in {"http", "sse", "streamable-http"}:
            kwargs.update({"host": host, "port": port})
        if transport == "sse":
            kwargs["sse_path"] = sse_path
        if transport in {"http", "streamable-http"}:
            kwargs["http_path"] = http_path

        self.server.run(**kwargs)
