from mfp_agent.agent import ESSENTIAL_MCP_TOOLS, SYSTEM_PROMPT


def test_only_requested_mcp_tools_are_enabled():
    assert ESSENTIAL_MCP_TOOLS == {
        "refresh_browser_cookies",
        "mfp_get_diary",
        "mfp_add_food_to_diary",
        "mfp_remove_food_from_diary",
        "mfp_get_meal_foods",
        "mfp_resolve_meal_food",
        "mfp_search_food",
        "mfp_get_food_details",
        "mfp_get_report",
    }


def test_prompt_supports_multiple_food_entries_and_physical_amounts():
    assert "Add each distinct food once" in SYSTEM_PROMPT
    assert '50 g means amount=50/unit="g"' in SYSTEM_PROMPT
    assert "report partial failures" in SYSTEM_PROMPT


def test_prompt_prefers_recent_foods_before_search():
    assert "Before global search, get recent/frequent foods" in SYSTEM_PROMPT
    assert "its history_id. Search only when no good history match exists" in SYSTEM_PROMPT


def test_prompt_preserves_weight_and_count_units():
    assert 'amount=50/unit="g"' in SYSTEM_PROMPT
    assert 'amount=2/unit="count"' in SYSTEM_PROMPT
    assert "do not ask\n  for grams" in SYSTEM_PROMPT
    assert "never for\n  grams or item counts" in SYSTEM_PROMPT


def test_prompt_rejects_implausible_nutrition():
    assert "nutrition_plausibility=implausible" in SYSTEM_PROMPT


def test_prompt_never_trades_food_identity_for_unit_support():
    assert "Food identity outranks unit support" in SYSTEM_PROMPT
    assert "never choose a\n  loosely related result" in SYSTEM_PROMPT


def test_prompt_requires_telegram_legacy_markdown():
    assert "Telegram uses legacy Markdown" in SYSTEM_PROMPT
    assert "Stay below 3,500 characters" in SYSTEM_PROMPT


def test_prompt_stays_compact():
    assert len(SYSTEM_PROMPT) < 2_600
