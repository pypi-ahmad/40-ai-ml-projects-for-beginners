from pathlib import Path

from task_planning_agent.memory.manager import MemoryManager
from task_planning_agent.schemas import PlanSession, Task


def test_memory_persist_and_search(tmp_path: Path) -> None:
    manager = MemoryManager(
        sqlite_path=str(tmp_path / "memory.db"),
        chroma_dir=str(tmp_path / "chroma"),
    )
    session = PlanSession(user_id="u1", raw_input="task input", tasks=[Task(name="Write report")])
    manager.persist_plan(session)

    history = manager.history("u1")
    assert history
    semantic = manager.semantic_search("report")
    assert semantic
