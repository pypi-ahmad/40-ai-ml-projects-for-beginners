"""FastAPI dependency wiring."""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from task_planning_agent.agent.service import PlanningService
from task_planning_agent.api.auth import decode_access_token
from task_planning_agent.calendar.service import CalendarService
from task_planning_agent.config import get_runtime_settings, load_config
from task_planning_agent.observability.monitor import RuntimeMonitor
from task_planning_agent.tools.registry import ToolRegistry


security = HTTPBearer()


@lru_cache(maxsize=1)
def get_service() -> PlanningService:
    return PlanningService(load_config())


@lru_cache(maxsize=1)
def get_calendar_service() -> CalendarService:
    return CalendarService()


@lru_cache(maxsize=1)
def get_tool_registry() -> ToolRegistry:
    return ToolRegistry()


@lru_cache(maxsize=1)
def get_monitor() -> RuntimeMonitor:
    return RuntimeMonitor()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict[str, str]:
    settings = get_runtime_settings()
    try:
        payload = decode_access_token(
            credentials.credentials,
            secret=settings.jwt_secret,
            algorithm=load_config().raw.get("security", {}).get("jwt_algorithm", "HS256"),
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    user_id = str(payload.get("sub", ""))
    role = str(payload.get("role", "user"))
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return {"user_id": user_id, "role": role}
