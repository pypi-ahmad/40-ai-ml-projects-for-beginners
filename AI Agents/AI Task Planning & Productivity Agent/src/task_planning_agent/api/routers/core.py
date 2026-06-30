"""Main API routes."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from task_planning_agent.api.auth import create_access_token, hash_password, verify_password
from task_planning_agent.api.deps import (
    get_calendar_service,
    get_current_user,
    get_monitor,
    get_service,
    get_tool_registry,
)
from task_planning_agent.config import get_runtime_settings, load_config
from task_planning_agent.schemas import PlanRequest, PriorityStrategy, ReplanRequest, UserPreference


router = APIRouter()


class AuthRequest(BaseModel):
    user_id: str
    password: str = Field(min_length=8)
    role: str = "user"


@router.post("/auth/register")
def register(payload: AuthRequest, service=Depends(get_service)) -> dict[str, str]:
    password_hash = hash_password(payload.password)
    service.memory.sqlite.upsert_user(payload.user_id, password_hash, role=payload.role)
    return {"status": "registered", "user_id": payload.user_id}


@router.post("/auth/login")
def login(payload: AuthRequest, service=Depends(get_service)) -> dict[str, str]:
    user = service.memory.sqlite.get_user(payload.user_id)
    if user is None or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    cfg = load_config().raw.get("security", {})
    settings = get_runtime_settings()
    token = create_access_token(
        subject=payload.user_id,
        role=user["role"],
        secret=settings.jwt_secret,
        algorithm=cfg.get("jwt_algorithm", "HS256"),
        expires_minutes=int(cfg.get("token_expiry_minutes", 1440)),
    )
    return {"access_token": token, "token_type": "bearer"}


@router.post("/plan")
def plan(
    payload: PlanRequest,
    service=Depends(get_service),
    current_user: dict[str, str] = Depends(get_current_user),
):
    if payload.user_id != current_user["user_id"] and current_user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    return service.plan(
        user_id=payload.user_id,
        raw_input=payload.raw_input,
        strategy=payload.strategy,
        timezone=payload.timezone,
    )


@router.post("/replan")
def replan(
    payload: ReplanRequest,
    service=Depends(get_service),
    current_user: dict[str, str] = Depends(get_current_user),
):
    if payload.user_id != current_user["user_id"] and current_user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    return service.replan(
        user_id=payload.user_id,
        reason=payload.reason,
        additional_input=payload.additional_input,
    )


@router.get("/tasks")
def tasks(
    user_id: str,
    service=Depends(get_service),
    current_user: dict[str, str] = Depends(get_current_user),
):
    if user_id != current_user["user_id"] and current_user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    return service.memory.sqlite.list_tasks(user_id=user_id)


@router.get("/history")
def history(
    user_id: str,
    limit: int = 20,
    service=Depends(get_service),
    current_user: dict[str, str] = Depends(get_current_user),
):
    if user_id != current_user["user_id"] and current_user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    return service.memory.history(user_id=user_id, limit=limit)


@router.get("/search")
def search(
    user_id: str,
    query: str,
    service=Depends(get_service),
    current_user: dict[str, str] = Depends(get_current_user),
):
    if user_id != current_user["user_id"] and current_user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    return {
        "tasks": [task.model_dump() for task in service.memory.search_tasks(user_id=user_id, query=query)],
        "semantic": service.memory.semantic_search(query),
    }


@router.get("/calendar")
def calendar_info():
    cfg = load_config()
    return {
        "mode": cfg.calendar.get("mode", "local_ics"),
        "google": get_calendar_service().google.status().model_dump(),
    }


@router.post("/calendar/export")
def calendar_export(
    user_id: str,
    output_path: str,
    service=Depends(get_service),
    current_user: dict[str, str] = Depends(get_current_user),
):
    if user_id != current_user["user_id"] and current_user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    history = service.memory.history(user_id=user_id, limit=1)
    if not history:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No plan history")
    path = get_calendar_service().export_ics(output_path, history[0].schedule_blocks)
    return {"path": path}


@router.get("/preferences")
def get_preferences(
    user_id: str,
    service=Depends(get_service),
    current_user: dict[str, str] = Depends(get_current_user),
):
    if user_id != current_user["user_id"] and current_user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    prefs = service.memory.sqlite.get_preferences(user_id)
    if prefs is None:
        prefs = UserPreference(user_id=user_id)
        service.memory.sqlite.upsert_preferences(prefs)
    return prefs


@router.post("/preferences")
def set_preferences(
    payload: UserPreference,
    service=Depends(get_service),
    current_user: dict[str, str] = Depends(get_current_user),
):
    if payload.user_id != current_user["user_id"] and current_user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    service.memory.sqlite.upsert_preferences(payload)
    return {"status": "updated"}


@router.get("/report")
def report(
    user_id: str,
    service=Depends(get_service),
    current_user: dict[str, str] = Depends(get_current_user),
):
    if user_id != current_user["user_id"] and current_user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    history = service.memory.history(user_id=user_id, limit=1)
    if not history:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No reports available")
    return history[0]


@router.get("/health")
def health(
    service=Depends(get_service),
    monitor=Depends(get_monitor),
    tools=Depends(get_tool_registry),
):
    runtime = monitor.collect()
    return {
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat(),
        "tools": tools.list_tools(),
        "runtime": asdict(runtime),
        "db": str(service.memory.sqlite.db_path),
    }


@router.get("/strategies")
def strategies() -> dict[str, list[str]]:
    return {"strategies": [strategy.value for strategy in PriorityStrategy]}


@router.post("/tools/{tool_name}")
def run_tool(tool_name: str, payload: dict[str, object], tools=Depends(get_tool_registry)):
    return tools.run(tool_name, **payload)
