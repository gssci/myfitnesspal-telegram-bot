from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from telegram import Message, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.helpers import escape_markdown

from .agent import DEFAULT_BACKEND, SUPPORTED_BACKENDS, MyFitnessPalAgent
from .audio import convert_to_model_wav
from .config import PROJECT_ROOT
from .logging_setup import configure_logging

LOGGER = logging.getLogger(__name__)
TELEGRAM_MESSAGE_LIMIT = 4096
TELEGRAM_MARKDOWN_CHUNK_LIMIT = 3500
TELEGRAM_FALLBACK_CHUNK_LIMIT = 1800

# Auto-restart tuning: if the bot process crashes, relaunch it with
# exponential backoff instead of exiting. The backoff resets once a run has
# stayed healthy for RESTART_HEALTHY_UPTIME seconds, so one bad crash doesn't
# slow down recovery from a later, unrelated one.
DEFAULT_RESTART_DELAY = 5.0
MAX_RESTART_DELAY = 300.0
RESTART_HEALTHY_UPTIME = 60.0


@dataclass(slots=True)
class _AgentRequest:
    content: str | bytes
    history: list[BaseMessage] | None
    session_id: str
    result: asyncio.Future[tuple[str, list[BaseMessage]]]


def parse_allowed_user_ids(value: str | None) -> set[int]:
    """Parse a comma-separated Telegram user ID allowlist, closed by default."""
    if value is None or not value.strip():
        return set()
    try:
        return {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError(
            "TELEGRAM_ALLOWED_USER_IDS must contain comma-separated numeric IDs"
        ) from exc


def split_telegram_text(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Split a response into Telegram-safe messages, preferring line boundaries."""
    if limit < 1:
        raise ValueError("limit must be positive")
    chunks: list[str] = []
    remaining = text or "The agent returned no text response."
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit + 1)
        if split_at <= 0:
            split_at = remaining.rfind(" ", 0, limit + 1)
        if split_at <= 0:
            split_at = limit
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


async def send_agent_response(message: Message, text: str) -> None:
    """Send legacy Markdown, falling back to safely escaped text if parsing fails."""
    for chunk in split_telegram_text(text, limit=TELEGRAM_MARKDOWN_CHUNK_LIMIT):
        try:
            await message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
        except BadRequest:
            LOGGER.warning(
                "Invalid legacy Markdown from agent; sending escaped fallback"
            )
            for fallback_chunk in split_telegram_text(
                chunk, limit=TELEGRAM_FALLBACK_CHUNK_LIMIT
            ):
                await message.reply_text(
                    escape_markdown(fallback_chunk, version=1),
                    parse_mode=ParseMode.MARKDOWN,
                )


class TelegramAgentBot:
    def __init__(
        self,
        agent: MyFitnessPalAgent | None = None,
        allowed_user_ids: set[int] | None = None,
        backend: str = DEFAULT_BACKEND,
    ) -> None:
        self.agent = agent or MyFitnessPalAgent(backend=backend)
        self.allowed_user_ids = allowed_user_ids or set()
        self.histories: dict[str, list[BaseMessage]] = {}
        self._session_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._requests: asyncio.Queue[_AgentRequest | None] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._worker_ready: asyncio.Future[None] | None = None

    async def startup(self, application: Application) -> None:
        loop = asyncio.get_running_loop()
        self._worker_ready = loop.create_future()
        self._worker_task = loop.create_task(
            self._run_agent_worker(), name="mfp-telegram-agent"
        )
        await self._worker_ready
        LOGGER.info("Telegram bot started with %d agent tools", len(self.agent.tools))

    async def shutdown(self, application: Application) -> None:
        if self._worker_task is None:
            return
        if not self._worker_task.done():
            await self._requests.put(None)
        await self._worker_task
        self._worker_task = None

    async def _run_agent_worker(self) -> None:
        try:
            await self.agent.start()
        except BaseException as exc:
            if self._worker_ready and not self._worker_ready.done():
                self._worker_ready.set_exception(exc)
            return

        if self._worker_ready and not self._worker_ready.done():
            self._worker_ready.set_result(None)
        try:
            while request := await self._requests.get():
                try:
                    if isinstance(request.content, bytes):
                        response = await self.agent.ask_audio(
                            request.content,
                            request.history,
                            session_id=request.session_id,
                        )
                    else:
                        response = await self.agent.ask(
                            request.content,
                            request.history,
                            session_id=request.session_id,
                        )
                except BaseException as exc:
                    if not request.result.done():
                        request.result.set_exception(exc)
                else:
                    if not request.result.done():
                        request.result.set_result(response)
        finally:
            await self.agent.close()

    async def ask_agent(
        self,
        text: str,
        history: list[BaseMessage] | None,
        session_id: str,
    ) -> tuple[str, list[BaseMessage]]:
        if self._worker_task is None or self._worker_task.done():
            raise RuntimeError("Telegram agent worker is not running")
        result = asyncio.get_running_loop().create_future()
        await self._requests.put(
            _AgentRequest(
                content=text,
                history=history,
                session_id=session_id,
                result=result,
            )
        )
        return await result

    async def ask_agent_audio(
        self,
        wav_audio: bytes,
        history: list[BaseMessage] | None,
        session_id: str,
    ) -> tuple[str, list[BaseMessage]]:
        if self._worker_task is None or self._worker_task.done():
            raise RuntimeError("Telegram agent worker is not running")
        result = asyncio.get_running_loop().create_future()
        await self._requests.put(
            _AgentRequest(
                content=wav_audio,
                history=history,
                session_id=session_id,
                result=result,
            )
        )
        return await result

    def _is_allowed(self, update: Update) -> bool:
        user = update.effective_user
        return bool(user and user.id in self.allowed_user_ids)

    async def _reject_if_unauthorized(self, update: Update) -> bool:
        if self._is_allowed(update):
            return False
        if update.effective_message:
            await update.effective_message.reply_text("This bot is private.")
        return True

    @staticmethod
    def _session_id(update: Update) -> str:
        chat = update.effective_chat
        user = update.effective_user
        if chat is None or user is None:
            raise ValueError("Telegram update has no chat or user")
        return f"telegram:{chat.id}:{user.id}"

    async def start_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        message = update.effective_message
        user = update.effective_user
        if message is None or user is None:
            return
        if not self._is_allowed(update):
            await message.reply_text(
                "This bot is private and your account is not authorized. "
                f"Your Telegram user ID is {user.id}. Add this number to "
                "TELEGRAM_ALLOWED_USER_IDS and restart the bot."
            )
            return
        await message.reply_text(
            "Send me a MyFitnessPal request in text, as a voice message, or as "
            "an audio file. I remember the most recent conversation turns for "
            "follow-up messages. Use /reset to clear that context."
        )

    async def reset_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if await self._reject_if_unauthorized(update):
            return
        session_id = self._session_id(update)
        async with self._session_locks[session_id]:
            self.histories.pop(session_id, None)
        await update.effective_message.reply_text("Chat context cleared.")

    async def handle_text(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if await self._reject_if_unauthorized(update):
            return
        message = update.effective_message
        chat = update.effective_chat
        if message is None or chat is None or not message.text:
            return

        session_id = self._session_id(update)
        async with self._session_locks[session_id]:
            try:
                await context.bot.send_chat_action(
                    chat_id=chat.id, action=ChatAction.TYPING
                )
                answer, updated = await self.ask_agent(
                    message.text,
                    self.histories.get(session_id),
                    session_id,
                )
                self.histories[session_id] = updated
            except Exception:
                LOGGER.exception("Telegram agent request failed for %s", session_id)
                await message.reply_text(
                    "I couldn't process that request. Check the bot logs and try again."
                )
                return

        await send_agent_response(message, answer)

    async def handle_audio(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if await self._reject_if_unauthorized(update):
            return
        message = update.effective_message
        chat = update.effective_chat
        if message is None or chat is None:
            return
        media = message.audio or message.voice
        if media is None:
            return

        session_id = self._session_id(update)
        async with self._session_locks[session_id]:
            try:
                await context.bot.send_chat_action(
                    chat_id=chat.id, action=ChatAction.TYPING
                )
                with TemporaryDirectory(prefix="mfp-audio-") as temporary_dir:
                    source = Path(temporary_dir) / "telegram-audio"
                    wav = Path(temporary_dir) / "model-input.wav"
                    telegram_file = await context.bot.get_file(media.file_id)
                    await telegram_file.download_to_drive(custom_path=source)
                    await convert_to_model_wav(source, wav)
                    wav_audio = wav.read_bytes()

                answer, updated = await self.ask_agent_audio(
                    wav_audio,
                    self.histories.get(session_id),
                    session_id,
                )
                self.histories[session_id] = updated
            except Exception:
                LOGGER.exception("Telegram audio request failed for %s", session_id)
                await message.reply_text(
                    "I couldn't process that audio message. Check that ffmpeg and "
                    "the llama-server multimodal projector are available, then try again."
                )
                return

        await send_agent_response(message, answer)


def build_application(token: str, backend: str = DEFAULT_BACKEND) -> Application:
    allowed_user_ids = parse_allowed_user_ids(
        os.getenv("TELEGRAM_ALLOWED_USER_IDS")
    )
    service = TelegramAgentBot(allowed_user_ids=allowed_user_ids, backend=backend)
    application = (
        ApplicationBuilder()
        .token(token)
        .post_init(service.startup)
        .post_shutdown(service.shutdown)
        .build()
    )
    application.bot_data["telegram_agent_service"] = service
    application.add_handler(CommandHandler("start", service.start_command))
    application.add_handler(CommandHandler("reset", service.reset_command))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, service.handle_text)
    )
    application.add_handler(
        MessageHandler(filters.AUDIO | filters.VOICE, service.handle_audio)
    )
    return application


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mfp-telegram",
        description="Run the MyFitnessPal Telegram bot.",
    )
    parser.add_argument(
        "backend",
        nargs="?",
        choices=SUPPORTED_BACKENDS,
        default=os.getenv("MFP_MODEL_BACKEND", DEFAULT_BACKEND),
        help=(
            "LLM backend to run the agent against: 'llama-server' (default) for a "
            "local llama-server OpenAI-compatible endpoint, or 'ollama' for a "
            "local Ollama server. Both use the gemma-4-e4b model. Defaults to the "
            "MFP_MODEL_BACKEND env var, or 'llama-server'."
        ),
    )
    return parser.parse_args(argv)


def run_forever_with_restart(token: str, backend: str) -> None:
    """Run the bot, automatically relaunching it if it crashes.

    Application.run_polling() only returns without raising on a clean shutdown
    (SIGINT/SIGTERM); any exception escaping it means the bot crashed
    unexpectedly. On a crash we rebuild a fresh Application and keep polling,
    backing off exponentially between attempts (capped at MAX_RESTART_DELAY).
    """
    delay = DEFAULT_RESTART_DELAY
    attempt = 0
    while True:
        attempt += 1
        started = time.monotonic()
        try:
            build_application(token, backend=backend).run_polling(
                allowed_updates=Update.ALL_TYPES
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            uptime = time.monotonic() - started
            if uptime >= RESTART_HEALTHY_UPTIME:
                delay = DEFAULT_RESTART_DELAY
            LOGGER.exception(
                "Telegram bot crashed after %.1fs (attempt %d); restarting in %.0fs",
                uptime,
                attempt,
                delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, MAX_RESTART_DELAY)
        else:
            LOGGER.info("Telegram bot stopped cleanly; not restarting")
            return


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    configure_logging()
    args = _parse_args(sys.argv[1:])
    run_forever_with_restart(token, args.backend)


if __name__ == "__main__":
    main()
