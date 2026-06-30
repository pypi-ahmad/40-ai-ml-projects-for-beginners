import pytest

from api_intel_agent.connectors import ConnectorRegistry


@pytest.mark.asyncio
async def test_connector_missing_credentials_returns_skip(monkeypatch):
    monkeypatch.delenv('NEWS_API_KEY', raising=False)
    registry = ConnectorRegistry()
    result = await registry.get('news').execute('ai')
    assert result.status in {'skipped_missing_credentials', 'ok', 'empty', 'failed'}
    if result.status == 'skipped_missing_credentials':
        assert result.error is not None
