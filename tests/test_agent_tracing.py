import json
from uuid import uuid4

from langchain_core.messages import ToolMessage

from mfp_agent.tracing import AgentTraceCallback


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
