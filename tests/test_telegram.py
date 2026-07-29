import asyncio

import pytest

from mfp_agent.telegram import (
    TelegramAgentBot,
    parse_allowed_user_ids,
    split_telegram_text,
)


def test_parse_allowed_user_ids():
    assert parse_allowed_user_ids("123, 456") == {123, 456}
    assert parse_allowed_user_ids("") == set()


def test_parse_allowed_user_ids_rejects_names():
    with pytest.raises(ValueError, match="numeric IDs"):
        parse_allowed_user_ids("username")


def test_split_telegram_text_respects_limit_and_preserves_text():
    chunks = split_telegram_text("one two three four", limit=9)

    assert chunks == ["one two", "three", "four"]
    assert all(len(chunk) <= 9 for chunk in chunks)


def test_agent_worker_owns_start_requests_and_close_in_one_task():
    class FakeAgent:
        tools = [object()]

        async def start(self):
            self.owner = asyncio.current_task()

        async def ask(self, text, history, *, session_id):
            assert asyncio.current_task() is self.owner
            return f"answer: {text}", []

        async def close(self):
            assert asyncio.current_task() is self.owner

    async def exercise():
        service = TelegramAgentBot(agent=FakeAgent(), allowed_user_ids={123})
        await service.startup(None)
        assert await service.ask_agent("hello", None, "session") == (
            "answer: hello",
            [],
        )
        await service.shutdown(None)

    asyncio.run(exercise())
