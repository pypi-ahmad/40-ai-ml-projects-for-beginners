from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, Header

from api.schemas import PromptRenderRequest, ResourceReadRequest, ToolExecutionRequest, WorkflowRequest
from auth.security import Identity, Role
from server.platform import Platform


def create_api_app(platform: Platform) -> FastAPI:
    app = FastAPI(title="Production MCP Server API", version="0.1.0")

    async def identity_dependency(x_api_key: str | None = Header(default=None)) -> Identity:
        return platform.auth.authenticate(x_api_key)

    async def require_user(identity: Identity = Depends(identity_dependency)) -> Identity:
        platform.auth.authorize(identity, Role.USER)
        return identity

    async def require_admin(identity: Identity = Depends(identity_dependency)) -> Identity:
        platform.auth.authorize(identity, Role.ADMIN)
        return identity

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "runtime": platform.settings.transport.runtime,
            "transport": platform.settings.transport.mode,
        }

    @app.get("/tools")
    async def list_tools(identity: Identity = Depends(require_user)) -> dict[str, Any]:
        platform.memory.log_audit(identity.api_key, "list_tools", {})
        return {"tools": platform.tools.list()}

    @app.post("/tools")
    async def execute_tool(
        payload: ToolExecutionRequest,
        identity: Identity = Depends(require_user),
    ) -> dict[str, Any]:
        if platform.settings.auth.read_only_mode:
            tool = platform.tools.get(payload.tool_name)
            if not tool.read_only:
                platform.auth.ensure_not_read_only_mode()

        result = await platform.call_tool(
            name=payload.tool_name,
            payload=payload.arguments,
            session_id=payload.session_id,
        )
        platform.memory.log_audit(identity.api_key, "execute_tool", {"tool": payload.tool_name})
        return result

    @app.get("/resources")
    async def list_resources(identity: Identity = Depends(require_user)) -> dict[str, Any]:
        platform.memory.log_audit(identity.api_key, "list_resources", {})
        return {"resources": platform.resources.list()}

    @app.post("/resources")
    async def read_resource(
        payload: ResourceReadRequest,
        identity: Identity = Depends(require_user),
    ) -> dict[str, Any]:
        platform.memory.log_audit(identity.api_key, "read_resource", {"uri": payload.uri})
        return platform.resources.read(payload.uri)

    @app.get("/prompts")
    async def list_prompts(identity: Identity = Depends(require_user)) -> dict[str, Any]:
        platform.memory.log_audit(identity.api_key, "list_prompts", {})
        return {"prompts": platform.prompts.list()}

    @app.post("/prompts")
    async def render_prompt(
        payload: PromptRenderRequest,
        identity: Identity = Depends(require_user),
    ) -> dict[str, Any]:
        rendered = platform.prompts.render(payload.prompt_name, payload.variables)
        platform.memory.log_prompt(payload.prompt_name, payload.variables, rendered)
        platform.memory.log_audit(identity.api_key, "render_prompt", {"prompt": payload.prompt_name})
        return {"prompt": payload.prompt_name, "rendered": rendered}

    @app.get("/memory")
    async def memory_search(
        query: str,
        top_k: int = 5,
        identity: Identity = Depends(require_user),
    ) -> dict[str, Any]:
        results = platform.memory.semantic_search(query=query, top_k=top_k)
        platform.memory.log_audit(identity.api_key, "memory_search", {"query": query})
        return {"results": results}

    @app.get("/search")
    async def search(
        query: str,
        identity: Identity = Depends(require_user),
    ) -> dict[str, Any]:
        web = await platform.call_tool("web_search", {"query": query}, session_id="search")
        docs = platform.memory.semantic_search(query=query, top_k=3)
        platform.memory.log_audit(identity.api_key, "search", {"query": query})
        return {"web": web, "memory": docs}

    @app.post("/reports")
    async def create_report(
        payload: WorkflowRequest,
        identity: Identity = Depends(require_user),
    ) -> dict[str, Any]:
        result = await platform.run_workflow(payload.query)
        platform.memory.log_audit(identity.api_key, "create_report", {"query": payload.query})
        return result

    @app.get("/metrics")
    async def metrics(identity: Identity = Depends(require_admin)) -> dict[str, Any]:
        platform.memory.log_audit(identity.api_key, "view_metrics", {})
        return {"metrics": platform.metrics.recent(limit=500)}

    return app
