"""Authentication utilities for API connectors and app auth."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from passlib.context import CryptContext

from api_intel_agent.config import get_secret, load_settings

# bcrypt backend can be inconsistent across environments; pbkdf2_sha256 is stable and secure.
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


@dataclass(slots=True)
class AuthHeaders:
    headers: dict[str, str]
    status: str


class AuthManager:
    def __init__(self) -> None:
        self.settings = load_settings()
        self._optional_auth_providers = {
            "github",
            "huggingface",
            "gitlab",
            "reddit",
        }

    def auth_headers(self, provider: str) -> AuthHeaders:
        api_config = self.settings.apis.get(provider, {})
        env_name = api_config.get("auth_env", "")
        if not env_name:
            return AuthHeaders(headers={}, status="no_auth_required")

        token = os.getenv(env_name, "").strip()
        if not token:
            if provider in self._optional_auth_providers:
                return AuthHeaders(headers={}, status="no_auth_fallback")
            return AuthHeaders(headers={}, status="skipped_missing_credentials")

        if "key" in env_name.lower():
            if provider == "news":
                return AuthHeaders(headers={"X-Api-Key": token}, status="ok")
            if provider == "nasa":
                return AuthHeaders(headers={"x-api-key": token}, status="ok")
            return AuthHeaders(headers={"Authorization": f"Bearer {token}"}, status="ok")

        return AuthHeaders(headers={"Authorization": f"Bearer {token}"}, status="ok")

    @staticmethod
    def hash_password(password: str) -> str:
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        return pwd_context.verify(password, hashed)

    def create_access_token(self, subject: str, role: str) -> str:
        expires = datetime.now(UTC) + timedelta(minutes=self.settings.auth.access_token_minutes)
        payload: dict[str, Any] = {"sub": subject, "role": role, "type": "access", "exp": expires}
        secret = get_secret(self.settings)
        return jwt.encode(payload, secret, algorithm=self.settings.auth.jwt_algorithm)

    def create_refresh_token(self, subject: str) -> str:
        expires = datetime.now(UTC) + timedelta(minutes=self.settings.auth.refresh_token_minutes)
        payload: dict[str, Any] = {"sub": subject, "type": "refresh", "exp": expires}
        secret = get_secret(self.settings)
        return jwt.encode(payload, secret, algorithm=self.settings.auth.jwt_algorithm)

    def decode_token(self, token: str) -> dict[str, Any]:
        secret = get_secret(self.settings)
        return jwt.decode(token, secret, algorithms=[self.settings.auth.jwt_algorithm])
