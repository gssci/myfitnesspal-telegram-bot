from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from mfp_agent.temporal import build_temporal_context, configured_timezone


def test_temporal_context_includes_relative_date_anchors():
    context = build_temporal_context(
        datetime(2026, 7, 29, 0, 5, 30),
        timezone=ZoneInfo("Europe/Rome"),
    )
    assert "Today=2026-07-29" in context
    assert "yesterday=2026-07-28" in context
    assert "tomorrow=2026-07-30" in context
    assert "pass explicit YYYY-MM-DD dates" in context


def test_aware_datetime_is_converted_to_configured_timezone():
    context = build_temporal_context(
        datetime(2026, 7, 28, 22, 30, tzinfo=timezone.utc),
        timezone=ZoneInfo("Europe/Rome"),
    )
    assert "Today=2026-07-29" in context
    assert "yesterday=2026-07-28" in context


def test_the_anchor_is_stable_across_calls_on_the_same_day():
    """The prompt sits ahead of the tools and the whole conversation.

    A clock time here changed on every model call, invalidating the server's
    prompt cache from that token onward and forcing the entire rest of the
    prompt to be reprocessed every turn.
    """
    morning = build_temporal_context(
        datetime(2026, 7, 29, 8, 15, 42), timezone=ZoneInfo("Europe/Rome")
    )
    evening = build_temporal_context(
        datetime(2026, 7, 29, 21, 59, 3), timezone=ZoneInfo("Europe/Rome")
    )

    assert morning == evening


def test_timezone_is_configurable(monkeypatch):
    monkeypatch.setenv("MFP_AGENT_TIMEZONE", "America/New_York")
    assert configured_timezone().key == "America/New_York"


def test_invalid_timezone_has_clear_error(monkeypatch):
    monkeypatch.setenv("MFP_AGENT_TIMEZONE", "Not/A_Timezone")
    with pytest.raises(ValueError, match="Unknown MFP_AGENT_TIMEZONE"):
        configured_timezone()
