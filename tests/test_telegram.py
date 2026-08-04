import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from telegram.constants import ParseMode
from telegram.error import BadRequest

from mfp_agent import telegram as telegram_module
from mfp_agent.telegram import (
    TelegramAgentBot,
    _parse_args,
    harden_markdown_v2,
    parse_allowed_user_ids,
    run_forever_with_restart,
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


def test_harden_markdown_v2_keeps_bold_and_escapes_punctuation():
    # A realistic confirmation line: the bold pair survives, and the parentheses,
    # dot and percent-adjacent punctuation that would break MarkdownV2 do not.
    assert harden_markdown_v2("*Yogurt 0.1%* (170 g) — 96 kcal!") == (
        "*Yogurt 0\\.1%* \\(170 g\\) — 96 kcal\\!"
    )


def test_harden_markdown_v2_escapes_an_unbalanced_asterisk():
    # One stray asterisk would make Telegram reject the whole message, so bold
    # is given up for that chunk rather than risking the send.
    assert harden_markdown_v2("*Added 50 g") == "\\*Added 50 g"


def test_harden_markdown_v2_leaves_existing_escapes_alone():
    assert harden_markdown_v2(r"2\.5 g") == r"2\.5 g"
    assert harden_markdown_v2("back\\slash") == "back\\\\slash"


def test_agent_response_escapes_and_uses_markdown_v2_parse_mode():
    class FakeMessage:
        def __init__(self):
            self.calls = []

        async def reply_text(self, text, **kwargs):
            self.calls.append((text, kwargs))

    async def exercise():
        message = FakeMessage()
        await send_agent_response(message, "*Added to Lunch* (2026-08-04)")
        assert message.calls == [
            (
                "*Added to Lunch* \\(2026\\-08\\-04\\)",
                {"parse_mode": ParseMode.MARKDOWN_V2},
            )
        ]

    asyncio.run(exercise())


def test_agent_response_falls_back_to_fully_escaped_markdown_v2():
    class FakeMessage:
        def __init__(self):
            self.calls = []

        async def reply_text(self, text, **kwargs):
            self.calls.append((text, kwargs))
            if len(self.calls) == 1:
                raise BadRequest("Can't parse entities")

    async def exercise():
        message = FakeMessage()
        await send_agent_response(message, "*Added.*")
        assert message.calls == [
            ("*Added\\.*", {"parse_mode": ParseMode.MARKDOWN_V2}),
            ("\\*Added\\.\\*", {"parse_mode": ParseMode.MARKDOWN_V2}),
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


def test_parse_args_defaults_to_llama_server(monkeypatch):
    monkeypatch.delenv("MFP_MODEL_BACKEND", raising=False)
    assert _parse_args([]).backend == "llama-server"


def test_parse_args_accepts_ollama_positional():
    assert _parse_args(["ollama"]).backend == "ollama"


def test_parse_args_rejects_unknown_backend():
    with pytest.raises(SystemExit):
        _parse_args(["bogus"])


def test_parse_args_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("MFP_MODEL_BACKEND", "ollama")
    assert _parse_args([]).backend == "ollama"


def test_parse_args_explicit_arg_overrides_env(monkeypatch):
    monkeypatch.setenv("MFP_MODEL_BACKEND", "ollama")
    assert _parse_args(["llama-server"]).backend == "llama-server"


def test_run_forever_with_restart_retries_after_crash(monkeypatch):
    calls = []

    class FakeApp:
        def run_polling(self, **kwargs):
            calls.append(kwargs)
            if len(calls) < 3:
                raise RuntimeError("boom")
            return None  # clean stop on the third attempt

    captured_backends = []

    def fake_build_application(token, backend):
        captured_backends.append(backend)
        assert token == "token"
        return FakeApp()

    sleeps = []
    monkeypatch.setattr(telegram_module, "build_application", fake_build_application)
    monkeypatch.setattr(telegram_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    run_forever_with_restart("token", "ollama")

    assert len(calls) == 3
    assert captured_backends == ["ollama", "ollama", "ollama"]
    assert sleeps == [
        telegram_module.DEFAULT_RESTART_DELAY,
        telegram_module.DEFAULT_RESTART_DELAY * 2,
    ]


def test_run_forever_with_restart_propagates_keyboard_interrupt(monkeypatch):
    class FakeApp:
        def run_polling(self, **kwargs):
            raise KeyboardInterrupt

    monkeypatch.setattr(
        telegram_module, "build_application", lambda token, backend: FakeApp()
    )

    with pytest.raises(KeyboardInterrupt):
        run_forever_with_restart("token", "llama-server")


def test_run_forever_with_restart_stops_cleanly_without_restarting(monkeypatch):
    calls = []

    class FakeApp:
        def run_polling(self, **kwargs):
            calls.append(1)
            return None

    monkeypatch.setattr(
        telegram_module, "build_application", lambda token, backend: FakeApp()
    )
    monkeypatch.setattr(
        telegram_module.time, "sleep", lambda seconds: pytest.fail("should not sleep")
    )

    run_forever_with_restart("token", "llama-server")

    assert calls == [1]


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
            ("*Done*", {"parse_mode": ParseMode.MARKDOWN_V2})
        ]

    asyncio.run(exercise())
