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


def test_agent_limits_unconstrained_by_default(monkeypatch):
    monkeypatch.delenv("OPENAI_MAX_TOKENS", raising=False)
    monkeypatch.delenv("MFP_AGENT_RECURSION_LIMIT", raising=False)
    import asyncio
    from langchain_core.messages import AIMessage
    from mfp_agent.agent import MyFitnessPalAgent

    class FakeGraph:
        def __init__(self):
            self.passed_config = None

        async def ainvoke(self, state, config):
            self.passed_config = config
            return {"messages": [*state["messages"], AIMessage(content="OK")]}

    async def exercise():
        agent = MyFitnessPalAgent()
        agent._agent = FakeGraph()

        answer, _ = await agent.ask("hello")
        assert answer == "OK"
        assert "recursion_limit" not in agent._agent.passed_config

    asyncio.run(exercise())


def test_agent_limits_explicit_env_override(monkeypatch):
    monkeypatch.setenv("OPENAI_MAX_TOKENS", "4096")
    monkeypatch.setenv("MFP_AGENT_RECURSION_LIMIT", "50")
    import asyncio
    from langchain_core.messages import AIMessage
    from mfp_agent.agent import MyFitnessPalAgent

    class FakeGraph:
        def __init__(self):
            self.passed_config = None

        async def ainvoke(self, state, config):
            self.passed_config = config
            return {"messages": [*state["messages"], AIMessage(content="OK")]}

    async def exercise():
        agent = MyFitnessPalAgent()
        agent._agent = FakeGraph()

        answer, _ = await agent.ask("hello")
        assert answer == "OK"
        assert agent._agent.passed_config["recursion_limit"] == 50

    asyncio.run(exercise())

