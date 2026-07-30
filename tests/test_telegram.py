import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from telegram.constants import ParseMode
from telegram.error import BadRequest

from mfp_agent.telegram import (
    TelegramAgentBot,
    parse_allowed_user_ids,
    send_agent_response,
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


def test_agent_response_uses_legacy_markdown_parse_mode():
    class FakeMessage:
        def __init__(self):
            self.calls = []

        async def reply_text(self, text, **kwargs):
            self.calls.append((text, kwargs))

    async def exercise():
        message = FakeMessage()
        await send_agent_response(message, "*Added* _yogurt_")
        assert message.calls == [
            (
                "*Added* _yogurt_",
                {"parse_mode": ParseMode.MARKDOWN},
            )
        ]

    asyncio.run(exercise())


def test_agent_response_falls_back_to_escaped_legacy_markdown():
    class FakeMessage:
        def __init__(self):
            self.calls = []

        async def reply_text(self, text, **kwargs):
            self.calls.append((text, kwargs))
            if len(self.calls) == 1:
                raise BadRequest("Can't parse entities")

    async def exercise():
        message = FakeMessage()
        await send_agent_response(message, "*unclosed")
        assert message.calls == [
            ("*unclosed", {"parse_mode": ParseMode.MARKDOWN}),
            (r"\*unclosed", {"parse_mode": ParseMode.MARKDOWN}),
        ]

    asyncio.run(exercise())


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


def test_agent_worker_processes_audio_requests_one_at_a_time():
    class FakeAgent:
        tools = [object()]

        def __init__(self):
            self.active = 0
            self.max_active = 0

        async def start(self):
            pass

        async def ask_audio(self, audio, history, *, session_id):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0)
            self.active -= 1
            return f"heard {len(audio)} bytes", []

        async def close(self):
            pass

    async def exercise():
        agent = FakeAgent()
        service = TelegramAgentBot(agent=agent, allowed_user_ids={123})
        await service.startup(None)
        results = await asyncio.gather(
            service.ask_agent_audio(b"first", None, "one"),
            service.ask_agent_audio(b"second", None, "two"),
        )
        await service.shutdown(None)

        assert agent.max_active == 1
        assert results == [("heard 5 bytes", []), ("heard 6 bytes", [])]

    asyncio.run(exercise())


def test_audio_handler_downloads_converts_and_updates_text_history(monkeypatch):
    converted = []

    async def fake_convert(source, destination):
        converted.append(Path(source).read_bytes())
        Path(destination).write_bytes(b"converted wav")

    monkeypatch.setattr("mfp_agent.telegram.convert_to_model_wav", fake_convert)

    class FakeTelegramFile:
        async def download_to_drive(self, custom_path):
            Path(custom_path).write_bytes(b"telegram voice")

    class FakeBot:
        def __init__(self):
            self.actions = []

        async def send_chat_action(self, **kwargs):
            self.actions.append(kwargs)

        async def get_file(self, file_id):
            assert file_id == "voice-file"
            return FakeTelegramFile()

    class FakeMessage:
        audio = None
        voice = SimpleNamespace(file_id="voice-file")

        def __init__(self):
            self.replies = []

        async def reply_text(self, text, **kwargs):
            self.replies.append((text, kwargs))

    async def exercise():
        service = TelegramAgentBot(agent=object(), allowed_user_ids={123})
        message = FakeMessage()
        update = SimpleNamespace(
            effective_user=SimpleNamespace(id=123),
            effective_chat=SimpleNamespace(id=456),
            effective_message=message,
        )
        context = SimpleNamespace(bot=FakeBot())

        async def fake_ask_agent_audio(wav_audio, history, session_id):
            assert wav_audio == b"converted wav"
            assert history is None
            assert session_id == "telegram:456:123"
            return "*Done*", [HumanMessage(content="Voice summary")]

        service.ask_agent_audio = fake_ask_agent_audio
        await service.handle_audio(update, context)

        assert converted == [b"telegram voice"]
        assert service.histories["telegram:456:123"][0].content == "Voice summary"
        assert message.replies == [
            ("*Done*", {"parse_mode": ParseMode.MARKDOWN})
        ]

    asyncio.run(exercise())
