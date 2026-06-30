"""JWT auth helpers and password utilities."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt_bytes = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 200_000)
    return f"{base64.b64encode(salt_bytes).decode()}${base64.b64encode(digest).decode()}"


def verify_password(password: str, hashed: str) -> bool:
    try:
        salt_b64, digest_b64 = hashed.split("$", maxsplit=1)
        salt = base64.b64decode(salt_b64.encode())
        expected = base64.b64decode(digest_b64.encode())
    except Exception:
        return False

    computed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return hmac.compare_digest(computed, expected)


def create_access_token(
    *, subject: str, role: str, secret: str, algorithm: str, expires_minutes: int
) -> str:
    now = datetime.now(tz=timezone.utc)
    payload: dict[str, Any] = {
        "sub": subject,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expires_minutes)).timestamp()),
    }
    return jwt.encode(payload, secret, algorithm=algorithm)


def decode_access_token(token: str, secret: str, algorithm: str) -> dict[str, Any]:
    return jwt.decode(token, secret, algorithms=[algorithm])
