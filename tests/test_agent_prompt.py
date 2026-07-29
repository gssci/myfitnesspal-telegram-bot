from mfp_agent.agent import SYSTEM_PROMPT


def test_prompt_supports_multiple_food_entries():
    assert "one add call for EACH distinct food entry" in SYSTEM_PROMPT
    assert "resolving three foods and then three add calls" in SYSTEM_PROMPT
    assert "reconcile results against the original checklist" in SYSTEM_PROMPT


def test_prompt_does_not_forbid_multiple_tool_calls():
    assert "Multiple calls are allowed whenever the task genuinely needs them" in SYSTEM_PROMPT
    assert "Multiple writes are expected for\n  multiple foods" in SYSTEM_PROMPT


def test_prompt_uses_meal_history_before_global_search():
    assert "call mfp_get_meal_foods once for each requested meal" in SYSTEM_PROMPT
    assert "0=Breakfast, 1=Lunch, 2=Dinner, 3=Snacks" in SYSTEM_PROMPT
    assert "call\n  mfp_resolve_meal_food" in SYSTEM_PROMPT
    assert "Fall back to mfp_search_food" in SYSTEM_PROMPT


def test_prompt_rejects_implausible_nutrition():
    assert "800 kcal for 1 g of olive oil" in SYSTEM_PROMPT
    assert "Reject entries with\n  impossible energy density" in SYSTEM_PROMPT
