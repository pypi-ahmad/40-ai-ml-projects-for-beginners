"""Database engine and session management."""

from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from internet_agent.config import Settings

Base = declarative_base()


class Database:
    """SQLite database wrapper for memory and cache persistence."""

    def __init__(self, settings: Settings) -> None:
        self._engine = create_engine(settings.memory.sqlite_url, future=True)
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)

    def create_all(self) -> None:
        Base.metadata.create_all(self._engine)

    @contextmanager
    def session(self) -> Session:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
