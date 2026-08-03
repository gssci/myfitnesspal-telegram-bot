from __future__ import annotations

import logging
import os


def configure_logging() -> None:
    """Configure process-wide logging once, honoring LOG_LEVEL.

    Every entry point (CLI, web, Telegram) calls this before it can log
    anything, so `LOG_LEVEL` and the message format stay consistent no matter
    how the agent is run. Safe to call more than once: `logging.basicConfig`
    is a no-op after the first call unless a handler is already configured.
    """
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # httpx logs full request URLs, which can include the Telegram bot token
    # or MCP session cookies. Keep it quiet unless something goes wrong.
    logging.getLogger("httpx").setLevel(logging.WARNING)
