import pytest

from mfp_agent.config import _resolve_env, load_mcp_connections


def test_resolve_env(monkeypatch):
    monkeypatch.setenv("TEST_AGENT_VALUE", "resolved")
    assert _resolve_env({"key": "prefix-${TEST_AGENT_VALUE}"}) == {"key": "prefix-resolved"}


def test_missing_env_raises(monkeypatch):
    monkeypatch.delenv("TEST_AGENT_MISSING", raising=False)
    with pytest.raises(ValueError, match="TEST_AGENT_MISSING"):
        _resolve_env("${TEST_AGENT_MISSING}")


def test_project_mcp_config_loads():
    connections = load_mcp_connections()
    assert connections["myfitnesspal"]["transport"] == "stdio"
