from api_intel_agent.auth import AuthManager


def test_password_hash_and_verify():
    manager = AuthManager()
    hashed = manager.hash_password('secret')
    assert manager.verify_password('secret', hashed)
    assert not manager.verify_password('bad', hashed)


def test_jwt_roundtrip():
    manager = AuthManager()
    token = manager.create_access_token('alice', 'admin')
    payload = manager.decode_token(token)
    assert payload['sub'] == 'alice'
    assert payload['role'] == 'admin'
