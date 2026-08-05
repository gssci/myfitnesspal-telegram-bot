from mfp_agent.agent import ESSENTIAL_MCP_TOOLS, SYSTEM_PROMPT

# Assert against whitespace-normalized text so these tests only break when the
# prompt's meaning changes, not when a sentence gets re-wrapped onto different
# lines.
NORMALIZED_PROMPT = " ".join(SYSTEM_PROMPT.split())


def test_only_requested_mcp_tools_are_enabled():
    assert ESSENTIAL_MCP_TOOLS == {
        "refresh_browser_cookies",
        "mfp_get_diary",
        "mfp_log_food",
        "mfp_add_food_to_diary",
        "mfp_remove_food_from_diary",
        "mfp_get_meal_foods",
        "mfp_search_food",
        "mfp_get_report",
    }


def test_the_steps_mfp_log_food_absorbed_are_no_longer_advertised():
    # Resolving a history_id and reading a food's details were only ever stages
    # of a lookup the server now runs itself, and every advertised tool costs
    # its whole JSON schema on every model call.
    assert "mfp_resolve_meal_food" not in ESSENTIAL_MCP_TOOLS
    assert "mfp_get_food_details" not in ESSENTIAL_MCP_TOOLS


def test_prompt_says_every_meal_argument_accepts_either_format():
    assert "every meal argument accepts either the number" in NORMALIZED_PROMPT
    assert "0=Breakfast, 1=Lunch, 2=Dinner, 3=Snacks" in NORMALIZED_PROMPT


def test_prompt_logs_through_one_call_per_food():
    # The four-turn loop (history, resolve, search, add) is what made a
    # three-food request cost eight model round trips.
    assert "call mfp_log_food once per distinct food, and nothing else" in NORMALIZED_PROMPT
    assert "Do not look the food up first" in NORMALIZED_PROMPT


def test_prompt_keeps_the_lookup_tools_out_of_the_logging_path():
    assert (
        "mfp_search_food and mfp_get_meal_foods are for answering questions about "
        "the diary, never a step on the way to logging" in NORMALIZED_PROMPT
    )


def test_prompt_reads_the_rejection_reasons_before_retrying():
    # "considered" is what lets a failed call be fixed rather than repeated.
    assert '"considered" names the foods it turned down and why' in NORMALIZED_PROMPT
    assert "If every why_not is about the unit" in NORMALIZED_PROMPT
    assert "call it again for the SAME food" in NORMALIZED_PROMPT


def test_prompt_chooses_the_unit_from_the_users_own_words():
    # The caller no longer sees a food's serving table before naming a unit, so
    # the rule is about the request, not about the candidate.
    assert "send the user's number unchanged, with the unit their own words point at" in (
        NORMALIZED_PROMPT
    )
    assert '"50 g of oats" -> amount=50, unit="g"' in NORMALIZED_PROMPT
    assert '"2 slices of bread" -> amount=2, unit="slice"' in NORMALIZED_PROMPT


def test_prompt_counts_a_whole_item_and_keeps_a_named_size():
    # unit="count" takes the food's first item serving, which is right for a
    # bare "1 kiwi" and wrong for "1 medium banana" on [small, medium, large].
    assert '"1 kiwi" -> amount=1, unit="count"' in NORMALIZED_PROMPT
    assert "A whole item with no size word is a count" in NORMALIZED_PROMPT
    assert '"1 medium banana" -> amount=1, unit="medium"' in NORMALIZED_PROMPT
    assert "A size word IS the unit" in NORMALIZED_PROMPT


def test_prompt_still_forbids_logging_a_countable_item_as_grams():
    # The exact failure seen in production: "2 kiwis" logged as 2 g, ~1 kcal.
    assert (
        'Never send a whole item as a weight: amount=2, unit="g" logs two grams'
        in NORMALIZED_PROMPT
    )
    assert "never ask the user for a gram amount instead" in NORMALIZED_PROMPT.lower()


def test_prompt_reserves_serving_for_an_explicit_portion_count():
    assert (
        '"2 servings" / "2 portions", said out loud -> unit="serving", never otherwise'
        in NORMALIZED_PROMPT
    )


def test_prompt_forbids_serving_multiplier_math():
    assert "Never calculate or apply a database serving multiplier yourself" in NORMALIZED_PROMPT


def test_prompt_supports_multiple_food_entries_and_reports_partial_failures():
    assert "one mfp_log_food call per distinct food, once each" in NORMALIZED_PROMPT
    assert "report which ones failed and why" in NORMALIZED_PROMPT


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
        "the LAST result's meal_totals and day_totals are already the running totals"
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
    # This sits ahead of the tool schemas and the whole conversation in every
    # single model call, so it is the most expensive text in the system.
    # Lowered once mfp_log_food absorbed the ID plumbing and the candidate
    # ranking. Prefer tightening the wording over raising it again.
    assert len(SYSTEM_PROMPT) < 4_500
