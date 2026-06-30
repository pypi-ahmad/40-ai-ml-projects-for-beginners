"""FastAPI auth dependencies."""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from api_intel_agent.api.db import SessionLocal, User
from api_intel_agent.auth import AuthManager

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
auth_manager = AuthManager()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    try:
        payload = auth_manager.decode_token(token)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc

    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token subject")

    user = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def require_role(required_role: str):
    hierarchy = {"viewer": 1, "analyst": 2, "admin": 3}

    def dependency(user: User = Depends(get_current_user)) -> User:
        if hierarchy.get(user.role, 0) < hierarchy.get(required_role, 0):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return user

    return dependency
