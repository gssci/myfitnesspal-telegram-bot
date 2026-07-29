from __future__ import annotations

import asyncio
import logging
import os
from collections import defaultdict
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .agent import MyFitnessPalAgent
from .config import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)
TELEGRAM_MESSAGE_LIMIT = 4096


@dataclass(slots=True)
class _AgentRequest:
    text: str
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


class TelegramAgentBot:
    def __init__(
        self,
        agent: MyFitnessPalAgent | None = None,
        allowed_user_ids: set[int] | None = None,
    ) -> None:
        self.agent = agent or MyFitnessPalAgent()
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
                    response = await self.agent.ask(
                        request.text,
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
                text=text,
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
            "Send me a MyFitnessPal request in plain text. I remember the most "
            "recent conversation turns for follow-up messages. Use /reset to "
            "clear that context."
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

        for chunk in split_telegram_text(answer):
            await message.reply_text(chunk)


def build_application(token: str) -> Application:
    allowed_user_ids = parse_allowed_user_ids(
        os.getenv("TELEGRAM_ALLOWED_USER_IDS")
    )
    service = TelegramAgentBot(allowed_user_ids=allowed_user_ids)
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
    return application


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # httpx logs Telegram Bot API URLs, which contain the secret token.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    build_application(token).run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
