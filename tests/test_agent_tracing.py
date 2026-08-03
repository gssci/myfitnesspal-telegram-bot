import json
import logging
from uuid import uuid4

from langchain_core.messages import ToolMessage

from mfp_agent.tracing import AgentTraceCallback, ToolCallLogger, _safe


def test_audio_data_is_always_redacted_from_traces():
    value = {
        "type": "input_audio",
        "input_audio": {"data": "base64 audio", "format": "wav"},
    }

    assert _safe(value) == {
        "type": "input_audio",
        "input_audio": {"data": "[REDACTED AUDIO]", "format": "wav"},
    }


def test_tool_calls_are_written_as_json_lines(tmp_path):
    path = tmp_path / "trace.jsonl"
    callback = AgentTraceCallback(path)
    run_id = uuid4()
    callback.on_tool_start(
        {"name": "mfp_add_food_to_diary"},
        "",
        run_id=run_id,
        inputs={"params": {"amount": 250, "unit": "g"}},
    )
    callback.on_tool_end(
        ToolMessage(content='{"success": true}', tool_call_id="call-1"),
        run_id=run_id,
    )

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["event"] for record in records] == ["tool_start", "tool_end"]
    assert records[0]["inputs"]["params"] == {"amount": 250, "unit": "g"}
    assert records[1]["output"]["content"] == '{"success": true}'


def test_sensitive_fields_are_redacted(tmp_path):
    path = tmp_path / "trace.jsonl"
    callback = AgentTraceCallback(path)
    callback.on_tool_start(
        {"name": "example"},
        "",
        run_id=uuid4(),
        inputs={"password": "secret", "value": "safe"},
    )
    record = json.loads(path.read_text())
    assert record["inputs"] == {"password": "[REDACTED]", "value": "safe"}


def test_tool_call_logger_logs_name_args_and_result(caplog):
    caplog.set_level(logging.INFO, logger="mfp_agent.tools")
    callback = ToolCallLogger()
    run_id = uuid4()

    callback.on_tool_start(
        {"name": "mfp_add_food_to_diary"},
        "",
        run_id=run_id,
        inputs={"params": {"mfp_id": "abc123", "meal": "Breakfast", "amount": 250, "unit": "g"}},
    )
    callback.on_tool_end(
        ToolMessage(content="Successfully added 250 g of Banana to Breakfast", tool_call_id="call-1"),
        run_id=run_id,
    )

    messages = [record.message for record in caplog.records]
    assert any(
        "mfp_add_food_to_diary(mfp_id=abc123, meal=Breakfast, amount=250, unit=g)" in msg
        for msg in messages
    )
    assert any("Successfully added 250 g of Banana to Breakfast" in msg for msg in messages)


def test_tool_call_logger_redacts_and_reports_errors(caplog):
    caplog.set_level(logging.INFO, logger="mfp_agent.tools")
    callback = ToolCallLogger()
    run_id = uuid4()

    callback.on_tool_start(
        {"name": "refresh_browser_cookies"},
        "",
        run_id=run_id,
        inputs={"params": {"token": "super-secret"}},
    )
    callback.on_tool_error(RuntimeError("session expired"), run_id=run_id)

    messages = [record.message for record in caplog.records]
    assert not any("super-secret" in msg for msg in messages)
    assert any("[REDACTED]" in msg for msg in messages)
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("refresh_browser_cookies" in r.message and "session expired" in r.message for r in error_records)
