from __future__ import annotations

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain.agents.middleware import ModelRequest, dynamic_prompt


def configured_timezone() -> ZoneInfo:
    name = os.getenv("MFP_AGENT_TIMEZONE", "Europe/Rome")
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown MFP_AGENT_TIMEZONE: {name}") from exc


def build_temporal_context(
    now: datetime | None = None,
    *,
    timezone: ZoneInfo | None = None,
) -> str:
    """Build explicit relative-date anchors for the current model call."""
    local_timezone = timezone or configured_timezone()
    if now is None:
        local_now = datetime.now(local_timezone)
    elif now.tzinfo is None:
        local_now = now.replace(tzinfo=local_timezone)
    else:
        local_now = now.astimezone(local_timezone)

    today = local_now.date()
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)
    return f"""CURRENT LOCAL DATE AND TIME
- Timezone: {local_timezone.key}
- Current local datetime: {local_now.isoformat(timespec="seconds")}
- Today: {today.isoformat()}
- Yesterday: {yesterday.isoformat()}
- Tomorrow: {tomorrow.isoformat()}

TEMPORAL RULES
- Resolve relative expressions such as today, yesterday, tomorrow, tonight,
  this morning, and last night from the anchors above.
- When calling a tool for a resolved relative date, pass the explicit YYYY-MM-DD
  date. Example: a request about "yesterday" must use date="{yesterday.isoformat()}".
- Never infer the current date from training knowledge or an earlier chat turn.
"""


def create_datetime_prompt(base_prompt: str):
    """Create LangChain middleware that refreshes time before every model call."""

    @dynamic_prompt
    def datetime_prompt(request: ModelRequest) -> str:
        del request
        return f"{base_prompt.rstrip()}\n\n{build_temporal_context()}"

    return datetime_prompt
