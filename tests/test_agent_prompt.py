from mfp_agent.agent import ESSENTIAL_MCP_TOOLS, SYSTEM_PROMPT

# Assert against whitespace-normalized text so these tests only break when the
# prompt's meaning changes, not when a sentence gets re-wrapped onto different
# lines.
NORMALIZED_PROMPT = " ".join(SYSTEM_PROMPT.split())


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


def test_prompt_distinguishes_meal_argument_formats():
    # mfp_get_meal_foods/mfp_resolve_meal_food take an integer meal index, but
    # mfp_add_food_to_diary/mfp_remove_food_from_diary take a meal name string.
    # Mixing these up is a real, easy-to-make bug.
    assert "take meal as a NUMBER" in NORMALIZED_PROMPT
    assert "take meal as a NAME" in NORMALIZED_PROMPT
    assert "0=Breakfast, 1=Lunch, 2=Dinner, 3=Snacks" in NORMALIZED_PROMPT


def test_prompt_never_passes_history_id_to_add():
    assert "mfp_add_food_to_diary only accepts an mfp_id" in NORMALIZED_PROMPT
    assert "Never pass a history_id to it" in NORMALIZED_PROMPT


def test_prompt_prefers_recent_foods_before_search():
    assert "Call mfp_get_meal_foods once for that food's meal" in NORMALIZED_PROMPT
    assert "Call mfp_search_food instead when" in NORMALIZED_PROMPT


def test_prompt_never_trades_food_identity_for_unit_support():
    assert "identity matters more than unit support" in NORMALIZED_PROMPT
    assert (
        "Do not pick a different, loosely-related food just because its units fit"
        in NORMALIZED_PROMPT
    )


def test_prompt_rejects_implausible_nutrition():
    assert NORMALIZED_PROMPT.count('nutrition_plausibility.status is "implausible"') >= 2


def test_prompt_preserves_weight_and_count_units():
    assert 'amount=50, unit="g"' in NORMALIZED_PROMPT
    assert 'amount=2, unit="count"' in NORMALIZED_PROMPT
    assert "Never ask the user for a gram amount instead" in NORMALIZED_PROMPT
    assert (
        'Never use unit="serving" for a gram amount or an item count' in NORMALIZED_PROMPT
    )


def test_prompt_forbids_serving_multiplier_math():
    assert "Never calculate or apply a database serving multiplier yourself" in NORMALIZED_PROMPT


def test_prompt_supports_multiple_food_entries_and_reports_partial_failures():
    assert (
        "resolve and add each distinct food independently, once each" in NORMALIZED_PROMPT
    )
    assert "report which ones failed and why" in NORMALIZED_PROMPT


def test_prompt_checks_write_result_before_claiming_success():
    assert "Only say it succeeded if success is true AND" in NORMALIZED_PROMPT
    assert "requested_amount/requested_unit match what you sent" in NORMALIZED_PROMPT
    assert 'Do not retry the same food with unit="serving" instead' in NORMALIZED_PROMPT


def test_prompt_refreshes_cookies_only_after_auth_error():
    assert (
        "Call refresh_browser_cookies only after a tool returns an authentication or "
        "session error" in NORMALIZED_PROMPT
    )
    assert "never before, and never as a first step" in NORMALIZED_PROMPT


def test_prompt_mentions_effective_date():
    assert "Always state which date the entries were logged under" in NORMALIZED_PROMPT


def test_prompt_stays_within_reasonable_length():
    # Generous ceiling to catch runaway bloat; the local model has a limited
    # context budget, so the prompt shouldn't grow unboundedly over time.
    assert len(SYSTEM_PROMPT) < 4_500
