from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from auth.security import AuthService
from config.settings import Settings, load_settings
from memory.service import MemoryService
from monitoring.metrics import MetricsCollector
from monitoring.scheduler import JobScheduler
from prompts.library import PromptLibrary
from resources.library import ResourceLibrary
from server.fastmcp_adapter import FastMCPAdapter
from server.mcp_sdk_adapter import MCPSDKAdapter
from tools.builtin import build_builtin_tools
from tools.plugins import PluginManager
from tools.registry import ToolRegistry
from utils.logging import configure_logging
from workflows.graph import WorkflowEngine

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Platform:
    settings: Settings
    auth: AuthService
    memory: MemoryService
    prompts: PromptLibrary
    resources: ResourceLibrary
    tools: ToolRegistry
    plugins: PluginManager
    metrics: MetricsCollector
    scheduler: JobScheduler
    workflow_engine: WorkflowEngine
    fastmcp: FastMCPAdapter
    mcp_sdk: MCPSDKAdapter

    @classmethod
    def from_config(cls, config_path: str | Path = "configs/default.yaml") -> "Platform":
        settings = load_settings(config_path)
        configure_logging(settings.logging)

        memory = MemoryService(settings)
        prompts = PromptLibrary()
        resources = ResourceLibrary(settings, memory)
        tools = ToolRegistry()

        for tool in build_builtin_tools(
            settings=settings,
            memory=memory,
            resources=resources,
            prompts=prompts,
        ):
            tools.register(tool)

        plugins = PluginManager(settings=settings, registry=tools)
        loaded_plugins = plugins.load_plugins()
        if loaded_plugins:
            logger.info("Loaded plugins: %s", loaded_plugins)

        auth = AuthService(settings.auth)
        metrics = MetricsCollector(config=settings.monitoring, memory=memory)
        scheduler = JobScheduler(settings=settings, memory=memory)
        workflow_engine = WorkflowEngine(tools=tools, memory=memory, prompts=prompts)

        fastmcp = FastMCPAdapter(
            server_name=settings.app.name,
            tools=tools,
            resources=resources,
            prompts=prompts,
        )
        mcp_sdk = MCPSDKAdapter(
            server_name=settings.app.name,
            tools=tools,
            resources=resources,
            prompts=prompts,
        )

        return cls(
            settings=settings,
            auth=auth,
            memory=memory,
            prompts=prompts,
            resources=resources,
            tools=tools,
            plugins=plugins,
            metrics=metrics,
            scheduler=scheduler,
            workflow_engine=workflow_engine,
            fastmcp=fastmcp,
            mcp_sdk=mcp_sdk,
        )

    async def call_tool(self, name: str, payload: dict[str, Any], session_id: str = "api") -> dict[str, Any]:
        started = time.perf_counter()
        result = await self.tools.call(name, payload)
        latency_ms = int((time.perf_counter() - started) * 1000)

        self.memory.log_tool_call(
            session_id=session_id,
            tool_name=name,
            request_payload=payload,
            response_payload=result,
            latency_ms=latency_ms,
        )
        self.metrics.record_tool_latency(name, latency_ms)
        return result

    async def run_workflow(self, query: str) -> dict[str, Any]:
        started = time.perf_counter()
        result = await self.workflow_engine.run(query)
        latency_ms = int((time.perf_counter() - started) * 1000)
        self.metrics.record_mcp_latency(self.settings.transport.runtime, latency_ms)
        return result

    def startup_background_services(self) -> None:
        self.scheduler.start()
        # metrics collector requires running event loop.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if loop.is_running():
            self.metrics.start()

    async def shutdown_background_services(self) -> None:
        self.scheduler.stop()
        await self.metrics.stop()

    def run_mcp_server(self) -> None:
        runtime = self.settings.transport.runtime
        mode = self.settings.transport.mode
        host = self.settings.transport.host
        port = self.settings.transport.port
        sse_path = self.settings.transport.sse_path
        http_path = self.settings.transport.http_path

        if runtime == "fastmcp":
            if not self.fastmcp.available():
                raise RuntimeError("FastMCP runtime requested but package unavailable")
            self.fastmcp.register_capabilities()
            self.fastmcp.run(transport=mode, host=host, port=port, sse_path=sse_path, http_path=http_path)
            return

        if runtime == "mcp":
            if not self.mcp_sdk.available():
                raise RuntimeError("Official mcp runtime requested but package unavailable")
            self.mcp_sdk.register_capabilities()
            self.mcp_sdk.run(transport=mode, host=host, port=port, sse_path=sse_path, http_path=http_path)
            return

        raise ValueError(f"Unknown runtime: {runtime}")
