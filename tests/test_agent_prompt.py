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


def test_prompt_says_every_meal_argument_accepts_either_format():
    # The tools once split into number-taking (mfp_get_meal_foods,
    # mfp_resolve_meal_food) and name-taking (mfp_add_food_to_diary,
    # mfp_remove_food_from_diary), so the prompt had to spell the split out.
    # Every meal field now runs through normalize_meal_number/normalize_meal_name
    # via the MealNumber/MealName annotated types, so both spellings validate on
    # every tool and the model only has to know the number-to-meal mapping.
    assert "every meal argument accepts either the number" in NORMALIZED_PROMPT
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


def test_prompt_chooses_from_the_units_the_food_itself_supports():
    # The food's serving_options/count_units are the menu; the job is choosing
    # the closest fit, not deciding a unit and then hoping the food takes it.
    assert (
        "the food's own serving_options and count_units ARE the units it supports"
        in NORMALIZED_PROMPT
    )
    assert "the one that fits the user's words most closely" in NORMALIZED_PROMPT
    assert '"1 kiwi" [fruit, g] -> amount=1, unit="fruit"' in NORMALIZED_PROMPT
    assert "A kiwi IS one fruit" in NORMALIZED_PROMPT
    assert '"2 slices of bread" [slice, g] -> amount=2, unit="slice"' in NORMALIZED_PROMPT


def test_prompt_picks_the_right_item_size_rather_than_the_first():
    # unit="count" takes the FIRST item serving, so on [small, medium, large] it
    # would log a small banana for "1 medium banana". Naming the unit is the
    # only way to say which item was meant.
    assert (
        '"1 medium banana" [small, medium, large] -> unit="medium", never "small"'
        in NORMALIZED_PROMPT
    )
    assert 'Use the generic unit="count" only when no listed unit fits' in NORMALIZED_PROMPT
    assert "it silently takes the FIRST item unit, so a name is always better" in (
        NORMALIZED_PROMPT
    )


def test_prompt_still_forbids_logging_a_countable_item_as_grams():
    # The exact failure seen in production: "2 kiwis" logged as 2 g, ~1 kcal.
    assert (
        'Never send a whole item as a weight: amount=2, unit="g" logs two grams'
        in NORMALIZED_PROMPT
    )
    assert "never ask the user for a gram amount instead" in NORMALIZED_PROMPT.lower()


def test_prompt_distrusts_supports_grams_as_a_signal_of_intent():
    # Nearly every MFP food carries a generic "1 g" serving, so the flag says
    # nothing about what the user asked for.
    assert "It never tells you the user meant grams" in NORMALIZED_PROMPT


def test_prompt_reserves_serving_for_an_explicit_portion_count():
    assert (
        '"2 servings" / "2 portions", said out loud -> unit="serving", never otherwise'
        in NORMALIZED_PROMPT
    )


def test_prompt_recovers_from_a_wrong_unit_without_changing_food():
    assert "resend the SAME food with the right unit" in NORMALIZED_PROMPT
    assert "Only a rejection about the food itself" in NORMALIZED_PROMPT


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
    assert 'never fall back to unit="serving"' in NORMALIZED_PROMPT


def test_prompt_refreshes_cookies_only_after_auth_error():
    assert (
        "Call refresh_browser_cookies only after a tool returns an authentication or "
        "session error" in NORMALIZED_PROMPT
    )
    assert "never before, and never as a first step" in NORMALIZED_PROMPT


def test_prompt_mentions_effective_date():
    assert "Always state which date the entries were logged under" in NORMALIZED_PROMPT


def test_prompt_confirms_adds_without_a_second_diary_read():
    assert "Never call mfp_get_diary to confirm" in NORMALIZED_PROMPT
    assert "never total the numbers yourself" in NORMALIZED_PROMPT
    assert (
        "the LAST add's meal_totals and day_totals are already the running totals"
        in NORMALIZED_PROMPT
    )


def test_prompt_reports_meal_totals_as_well_as_day_totals():
    assert "meal_totals (its meal) and day_totals (the whole day)" in NORMALIZED_PROMPT
    assert "then that meal's totals, then the day's" in NORMALIZED_PROMPT
    assert "*Snacks so far* — 374.5 kcal" in NORMALIZED_PROMPT


def test_prompt_asks_for_one_food_emoji_with_a_category_fallback():
    assert "Start every food line with exactly one emoji" in NORMALIZED_PROMPT
    assert "Prefer an exact match (🍶 yogurt, 🥝 kiwi" in NORMALIZED_PROMPT
    # Falling back to a category emoji beats guessing or omitting one.
    assert "otherwise use its category (🍞 bread and crackers" in NORMALIZED_PROMPT
    assert "🍽 anything else" in NORMALIZED_PROMPT


def test_prompt_requests_telegram_markdown_v2_output():
    assert "sent to Telegram as MarkdownV2" in NORMALIZED_PROMPT
    # harden_markdown_v2() does the escaping, so the model must not also try.
    assert "never write a backslash" in NORMALIZED_PROMPT
    assert "use *bold* only, always as a closed pair" in NORMALIZED_PROMPT


def test_prompt_shows_a_worked_confirmation_example():
    # The default model is small, so the format is taught by example as well as
    # by rule: emoji, bolded name, amount, calories and all three macros.
    assert (
        "🥝 *Kiwi* — 2 count, 92.0 kcal, P 1.7 g, C 22.0 g, F 0.8 g" in NORMALIZED_PROMPT
    )
    assert "*Today so far* — 1469.0 kcal, P 160.0 g, C 121.0 g, F 29.0 g" in NORMALIZED_PROMPT


def test_prompt_stays_within_reasonable_length():
    # Generous ceiling to catch runaway bloat; the local model has a limited
    # context budget, so the prompt shouldn't grow unboundedly over time.
    # Raised deliberately as the answer format and the unit rules were added.
    # Prefer tightening the wording over raising this again.
    assert len(SYSTEM_PROMPT) < 6_000
