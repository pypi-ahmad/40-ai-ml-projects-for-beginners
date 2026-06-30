"""SQLite user database for multi-user auth."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(30), nullable=False, default="viewer")


DB_PATH = Path("artifacts/auth/users.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
