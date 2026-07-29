from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage

SENSITIVE_KEYS = {"password", "token", "authorization", "cookie", "secret", "api_key"}


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
        return {
            str(key): "[REDACTED]" if str(key).lower() in SENSITIVE_KEYS else _safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "model_dump"):
        return _safe(value.model_dump())
    return str(value)


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
