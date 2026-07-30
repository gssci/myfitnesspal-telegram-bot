from __future__ import annotations

import os
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

DEFAULT_HISTORY_TURNS = 6


def get_history_turn_limit() -> int:
    """Return the number of recent user/assistant turns retained per chat."""
    raw = os.getenv("MFP_CHAT_HISTORY_TURNS", str(DEFAULT_HISTORY_TURNS))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("MFP_CHAT_HISTORY_TURNS must be a positive integer") from exc
    if value < 1:
        raise ValueError("MFP_CHAT_HISTORY_TURNS must be a positive integer")
    return value


def trim_history(
    messages: Sequence[BaseMessage], max_turns: int | None = None
) -> list[BaseMessage]:
    """Keep complete message/tool-call sequences for the most recent user turns."""
    limit = max_turns if max_turns is not None else get_history_turn_limit()
    if limit < 1:
        raise ValueError("max_turns must be a positive integer")

    user_message_indexes = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, HumanMessage)
    ]
    if len(user_message_indexes) <= limit:
        return list(messages)
    return list(messages[user_message_indexes[-limit] :])


def compact_history(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Keep user requests and final answers, dropping completed tool traces."""
    compacted: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, HumanMessage) or (
            isinstance(message, AIMessage)
            and message.content
            and not message.tool_calls
        ):
            compacted.append(message)
    return compacted
