from api_intel_agent.core.schemas import AnalyzeResponse, RunStatus
from api_intel_agent.memory.sqlite_store import SQLiteMemoryStore


def test_memory_history(tmp_path):
    store = SQLiteMemoryStore(path=str(tmp_path / 'memory.db'))
    response = AnalyzeResponse(run_id='run1', status=RunStatus.SUCCESS, summary='ok')
    store.save_response('query', response)
    history = store.history(limit=5)
    assert history
    assert history[0].run_id == 'run1'
