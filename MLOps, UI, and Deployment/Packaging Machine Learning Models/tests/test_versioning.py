import json
import tempfile

import pytest

from ml_package.versioning import ModelVersion, VersionRegistry


class TestModelVersion:
    def test_create_version(self):
        mv = ModelVersion("v1", "models/model.pkl", {"accuracy": 0.95})
        assert mv.version_id == "v1"
        assert mv.status == "created"

    def test_mark_active(self):
        mv = ModelVersion("v1", "models/model.pkl")
        mv.mark_active()
        assert mv.status == "active"

    def test_mark_archived(self):
        mv = ModelVersion("v1", "models/model.pkl")
        mv.mark_archived()
        assert mv.status == "archived"

    def test_mark_failed(self):
        mv = ModelVersion("v1", "models/model.pkl")
        mv.mark_failed()
        assert mv.status == "failed"

    def test_to_dict(self):
        mv = ModelVersion("v1", "models/model.pkl", {"accuracy": 0.95})
        d = mv.to_dict()
        assert d["version_id"] == "v1"
        assert d["metrics"]["accuracy"] == 0.95


class TestVersionRegistry:
    @pytest.fixture
    def registry(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"active_version": None, "versions": {}}, f)
            temp_path = f.name
        reg = VersionRegistry(temp_path)
        yield reg

    def test_register_and_activate(self, registry):
        registry.register("v1", "models/model.pkl", {"accuracy": 0.95})
        registry.activate("v1")
        active = registry.get_active()
        assert active is not None
        assert active.version_id == "v1"

    def test_get_active_returns_none_when_empty(self, registry):
        assert registry.get_active() is None

    def test_activate_nonexistent_raises(self, registry):
        with pytest.raises(KeyError):
            registry.activate("nonexistent")

    def test_get_nonexistent_raises(self, registry):
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_list_versions_empty(self, registry):
        assert registry.list_versions() == []

    def test_rollback(self, registry):
        registry.register("v1", "models/v1.pkl")
        registry.register("v2", "models/v2.pkl")
        registry.activate("v2")

        result = registry.rollback_to("v1")
        assert result["previous_active"] == "v2"
        assert result["current_active"] == "v1"
        assert registry.get_active().version_id == "v1"

    def test_register_with_missing_parent_raises(self, registry):
        with pytest.raises(ValueError, match="Parent version"):
            registry.register(
                "v2",
                "models/v2.pkl",
                parent_version="v1",
            )
