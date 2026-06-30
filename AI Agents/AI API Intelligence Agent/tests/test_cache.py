from api_intel_agent.cache.backends import SQLiteCache


def test_sqlite_cache_set_get(tmp_path):
    cache = SQLiteCache(str(tmp_path / 'cache.db'))
    cache.set('ns', 'k', {'value': 123}, ttl_seconds=60)
    assert cache.get('ns', 'k') == {'value': 123}
