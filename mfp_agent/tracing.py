from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage

SENSITIVE_KEYS = {"password", "token", "authorization", "cookie", "secret", "api_key"}
TOOL_LOGGER = logging.getLogger("mfp_agent.tools")
_ARG_PREVIEW_LIMIT = 80
_RESULT_PREVIEW_LIMIT = 200


def _safe(value: Any) -> Any:
    if isinstance(value, BaseMessage):
        return {
            "type": value.type,
            "content": value.content,
            "tool_calls": getattr(value, "tool_calls", None),
            "tool_call_id": getattr(value, "tool_call_id", None),
            "status": getattr(value, "status", None),
        }
    if isinstance(value, dict):
        safe_value = {
            str(key): "[REDACTED]" if str(key).lower() in SENSITIVE_KEYS else _safe(item)
            for key, item in value.items()
        }
        if value.get("type") == "input_audio" and isinstance(
            safe_value.get("input_audio"), dict
        ):
            safe_value["input_audio"]["data"] = "[REDACTED AUDIO]"
        return safe_value
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "model_dump"):
        return _safe(value.model_dump())
    return str(value)


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _flatten_tool_args(inputs: Any) -> dict[str, Any]:
    """Unwrap the MCP adapter's {"params": {...}} envelope for readable logs."""
    if isinstance(inputs, dict) and set(inputs) == {"params"} and isinstance(inputs["params"], dict):
        return inputs["params"]
    return inputs if isinstance(inputs, dict) else {"input": inputs}


def _format_tool_args(inputs: Any) -> str:
    safe_args = _safe(_flatten_tool_args(inputs))
    if not isinstance(safe_args, dict) or not safe_args:
        return ""
    return ", ".join(f"{key}={_truncate(str(value), _ARG_PREVIEW_LIMIT)}" for key, value in safe_args.items())


def _format_tool_result(output: Any) -> str:
    content = getattr(output, "content", output)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        content = " | ".join(parts)
    return _truncate(str(content), _RESULT_PREVIEW_LIMIT)


class ToolCallLogger(BaseCallbackHandler):
    """Always-on, human-readable log line for every tool call.

    AgentTraceCallback below writes a complete JSONL trace, but it is opt-in
    (MFP_AGENT_DEBUG) and meant for after-the-fact debugging. This callback
    goes through the standard `logging` module instead, so a one-line summary
    of every tool call — name, arguments, result preview, and timing — shows
    up live wherever the process already logs (terminal, systemd journal,
    Telegram bot logs), regardless of debug mode.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or TOOL_LOGGER
        self._started: dict[UUID, tuple[str, float]] = {}

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", "unknown_tool")
        self._started[run_id] = (name, monotonic())
        args = _format_tool_args(inputs if inputs is not None else input_str)
        self._logger.info("-> %s(%s)", name, args)

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        name, started = self._started.pop(run_id, ("unknown_tool", monotonic()))
        elapsed = monotonic() - started
        self._logger.info("<- %s (%.2fs): %s", name, elapsed, _format_tool_result(output))

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        name, started = self._started.pop(run_id, ("unknown_tool", monotonic()))
        elapsed = monotonic() - started
        self._logger.error(
            "x %s (%.2fs) raised %s: %s", name, elapsed, type(error).__name__, error
        )


class AgentTraceCallback(BaseCallbackHandler):
    """Write model decisions and MCP tool calls as local JSON Lines."""

    def __init__(self, path: Path, *, include_prompts: bool = False) -> None:
        self.path = path
        self.include_prompts = include_prompts
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, event: str, **payload: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **_safe(payload),
        }
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        payload: dict[str, Any] = {
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "metadata": metadata,
        }
        if self.include_prompts:
            payload["messages"] = messages
        self._write("model_start", **payload)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        generations = []
        for batch in getattr(response, "generations", []) or []:
            for generation in batch:
                message = getattr(generation, "message", None)
                generations.append(message if message is not None else getattr(generation, "text", ""))
        self._write(
            "model_end",
            run_id=run_id,
            parent_run_id=parent_run_id,
            generations=generations,
        )

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        inputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._write(
            "tool_start",
            run_id=run_id,
            parent_run_id=parent_run_id,
            tool=serialized.get("name"),
            inputs=inputs if inputs is not None else input_str,
            metadata=metadata,
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._write("tool_end", run_id=run_id, parent_run_id=parent_run_id, output=output)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._write(
            "tool_error",
            run_id=run_id,
            parent_run_id=parent_run_id,
            error=f"{type(error).__name__}: {error}",
        )
