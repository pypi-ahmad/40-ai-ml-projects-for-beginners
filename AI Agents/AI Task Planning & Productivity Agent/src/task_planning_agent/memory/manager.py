"""High-level memory manager combining SQLite and Chroma stores."""

from __future__ import annotations

from task_planning_agent.memory.chroma_store import ChromaSemanticStore
from task_planning_agent.memory.sqlite_store import SQLiteMemoryStore
from task_planning_agent.schemas import PlanSession, Task


class MemoryManager:
    """Persist operational + semantic memory for each planning run."""

    def __init__(self, sqlite_path: str, chroma_dir: str, collection_name: str = "plan_memory") -> None:
        self.sqlite = SQLiteMemoryStore(sqlite_path)
        self.chroma = ChromaSemanticStore(chroma_dir, collection_name=collection_name)

    def persist_plan(self, session: PlanSession) -> None:
        self.sqlite.save_tasks(session.user_id, session.tasks)
        self.sqlite.save_plan(session)
        for task in session.tasks:
            self.chroma.upsert(
                item_id=task.id,
                text=f"{task.name}\n{task.description}",
                metadata={"kind": "task", "user_id": session.user_id, "plan_id": session.plan_id},
            )
        self.chroma.upsert(
            item_id=session.plan_id,
            text=session.raw_input,
            metadata={"kind": "plan", "user_id": session.user_id},
        )

    def semantic_search(self, query: str, top_k: int = 8) -> list[dict[str, object]]:
        return self.chroma.query(query, n_results=top_k)

    def history(self, user_id: str, limit: int = 30) -> list[PlanSession]:
        return self.sqlite.list_plan_sessions(user_id=user_id, limit=limit)

    def search_tasks(self, user_id: str, query: str, limit: int = 20) -> list[Task]:
        return self.sqlite.search_tasks_text(user_id=user_id, query=query, limit=limit)
