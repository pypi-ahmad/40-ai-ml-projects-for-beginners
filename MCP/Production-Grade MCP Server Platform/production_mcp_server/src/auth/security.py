from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from fastapi import HTTPException, status

from config.settings import AuthConfig


class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"
    READ_ONLY = "read_only"


@dataclass(slots=True)
class Identity:
    api_key: str
    role: Role


class AuthService:
    def __init__(self, config: AuthConfig) -> None:
        self.config = config

    def authenticate(self, api_key: str | None) -> Identity:
        if not self.config.enabled:
            return Identity(api_key="anonymous", role=Role.ADMIN)

        if not api_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key")

        role_raw = self.config.api_keys.get(api_key)
        if role_raw is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

        try:
            role = Role(role_raw)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Invalid configured role: {role_raw}",
            ) from exc

        return Identity(api_key=api_key, role=role)

    def authorize(self, identity: Identity, required: Role) -> None:
        hierarchy = {Role.READ_ONLY: 1, Role.USER: 2, Role.ADMIN: 3}
        if hierarchy[identity.role] < hierarchy[required]:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")

    def ensure_not_read_only_mode(self) -> None:
        if self.config.read_only_mode:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Mutation blocked by read-only mode",
            )
