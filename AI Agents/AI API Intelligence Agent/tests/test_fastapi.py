from api_intel_agent.api.app import health


def test_health_function():
    payload = health()
    assert payload['status'] == 'ok'
    assert 'connectors' in payload
