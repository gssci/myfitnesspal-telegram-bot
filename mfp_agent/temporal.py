from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain.agents.middleware import ModelRequest, dynamic_prompt


def configured_timezone() -> ZoneInfo:
    name = os.getenv("MFP_AGENT_TIMEZONE", "Europe/Rome")
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown MFP_AGENT_TIMEZONE: {name}") from exc


def today_in_timezone(
    now: datetime | None = None,
    *,
    timezone: ZoneInfo | None = None,
) -> date:
    """Return the local calendar date the agent should resolve dates against."""
    local_timezone = timezone or configured_timezone()
    if now is None:
        return datetime.now(local_timezone).date()
    if now.tzinfo is None:
        return now.replace(tzinfo=local_timezone).date()
    return now.astimezone(local_timezone).date()


def build_temporal_context(
    now: datetime | None = None,
    *,
    timezone: ZoneInfo | None = None,
) -> str:
    """Build explicit relative-date anchors for one request.

    Deliberately carries no clock time. This string sits at the end of the
    system prompt, ahead of the tool schemas and the whole conversation, so any
    token of it that changes invalidates the server's prompt cache for
    everything after it. With a second-precision timestamp refreshed before
    every model call, that meant ~1,700 tokens of prefix reused and ~23,000
    reprocessed on every turn — measured at 148k tokens of prompt processing
    for a single three-food request, over 95% of its wall time.

    A calendar date is all the agent ever needed, and it holds the cache steady
    for a whole day, across requests as well as within them.
    """
    today = today_in_timezone(now, timezone=timezone)
    return (
        f"Today={today.isoformat()}, "
        f"yesterday={(today - timedelta(days=1)).isoformat()}, "
        f"tomorrow={(today + timedelta(days=1)).isoformat()}. Resolve relative "
        "dates from these anchors and pass explicit YYYY-MM-DD dates to tools."
    )


def create_datetime_prompt(base_prompt: str):
    """Create middleware that anchors relative dates for every model call.

    The anchor is rebuilt per call, but only its calendar date can differ, so
    the prompt is byte-identical for every call made on the same day.
    """

    @dynamic_prompt
    def datetime_prompt(request: ModelRequest) -> str:
        del request
        return f"{base_prompt.rstrip()}\n\n{build_temporal_context()}"

    return datetime_prompt
