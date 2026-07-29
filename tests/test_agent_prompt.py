from mfp_agent.agent import SYSTEM_PROMPT


def test_prompt_supports_multiple_food_entries():
    assert "one add call for EACH distinct food entry" in SYSTEM_PROMPT
    assert "three searches and then three add calls" in SYSTEM_PROMPT
    assert "reconcile results against the original checklist" in SYSTEM_PROMPT


def test_prompt_does_not_forbid_multiple_tool_calls():
    assert "Multiple calls are allowed whenever the task genuinely needs them" in SYSTEM_PROMPT
    assert "Multiple writes are expected for\n  multiple foods" in SYSTEM_PROMPT
