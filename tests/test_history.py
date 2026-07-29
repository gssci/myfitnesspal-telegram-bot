import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mfp_agent.history import get_history_turn_limit, trim_history


def test_trim_history_keeps_complete_recent_turns():
    messages = [
        HumanMessage(content="first"),
        AIMessage(content="", tool_calls=[{"name": "tool", "args": {}, "id": "1"}]),
        ToolMessage(content="result", tool_call_id="1"),
        AIMessage(content="first answer"),
        HumanMessage(content="second"),
        AIMessage(content="second answer"),
        HumanMessage(content="third"),
        AIMessage(content="third answer"),
    ]

    trimmed = trim_history(messages, max_turns=2)

    assert [message.content for message in trimmed] == [
        "second",
        "second answer",
        "third",
        "third answer",
    ]


def test_history_turn_limit_from_environment(monkeypatch):
    monkeypatch.setenv("MFP_CHAT_HISTORY_TURNS", "4")
    assert get_history_turn_limit() == 4


@pytest.mark.parametrize("value", ["0", "-1", "many"])
def test_history_turn_limit_must_be_positive(monkeypatch, value):
    monkeypatch.setenv("MFP_CHAT_HISTORY_TURNS", value)
    with pytest.raises(ValueError, match="positive integer"):
        get_history_turn_limit()
