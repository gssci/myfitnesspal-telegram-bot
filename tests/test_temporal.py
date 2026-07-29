from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from mfp_agent.temporal import build_temporal_context, configured_timezone


def test_temporal_context_includes_relative_date_anchors():
    context = build_temporal_context(
        datetime(2026, 7, 29, 0, 5, 30),
        timezone=ZoneInfo("Europe/Rome"),
    )
    assert "Current local datetime: 2026-07-29T00:05:30+02:00" in context
    assert "Today: 2026-07-29" in context
    assert "Yesterday: 2026-07-28" in context
    assert "Tomorrow: 2026-07-30" in context
    assert 'date="2026-07-28"' in context


def test_aware_datetime_is_converted_to_configured_timezone():
    context = build_temporal_context(
        datetime(2026, 7, 28, 22, 30, tzinfo=timezone.utc),
        timezone=ZoneInfo("Europe/Rome"),
    )
    assert "Current local datetime: 2026-07-29T00:30:00+02:00" in context
    assert "Yesterday: 2026-07-28" in context


def test_timezone_is_configurable(monkeypatch):
    monkeypatch.setenv("MFP_AGENT_TIMEZONE", "America/New_York")
    assert configured_timezone().key == "America/New_York"


def test_invalid_timezone_has_clear_error(monkeypatch):
    monkeypatch.setenv("MFP_AGENT_TIMEZONE", "Not/A_Timezone")
    with pytest.raises(ValueError, match="Unknown MFP_AGENT_TIMEZONE"):
        configured_timezone()
