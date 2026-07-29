# Nutrition Summary PDF

This repository also includes a local LangChain chat agent that connects Ollama
to the MyFitnessPal MCP server. It also exposes a safe local calculator tool for
quantity, serving, percentage, calorie, macro, and unit-conversion arithmetic.

## MyFitnessPal chat agent

Prerequisites:

- Ollama running locally with the `gemma4` model available.
- The MyFitnessPal MCP project and Python environment at the path in `mcp.json`.

Install everything and copy the environment template if needed:

```bash
uv sync
cp .env.example .env  # skip this if .env already exists
```

Set `MFP_USERNAME` and `MFP_PASSWORD` in `.env`. The checked-in `mcp.json` uses
`${...}` placeholders, so credentials are not stored in source control.

Telegram is the primary agent interface. Create a bot with BotFather and add its
token to the untracked `.env` file:

```dotenv
TELEGRAM_BOT_TOKEN=replace-with-your-token
# Required for access (comma-separated numeric IDs):
TELEGRAM_ALLOWED_USER_IDS=123456789
MFP_CHAT_HISTORY_TURNS=6
```

Start the Telegram long-polling interface with:

```bash
uv run mfp-telegram
```

The bot is closed by default. With no allowed IDs configured, `/start` reports
the sender's own numeric Telegram ID but all agent requests are denied. Send
`/start`, add your ID to `TELEGRAM_ALLOWED_USER_IDS`, and restart the process.
Only listed users can then reach the agent. Normal text is forwarded to the same
LangChain agent used by the browser and terminal interfaces. Recent context is
isolated by Telegram chat and user; `/reset` clears it. History is held in
process memory, so restarting the bot also clears it.
`MFP_CHAT_HISTORY_TURNS` controls the number of recent turns retained for all
interfaces (default: 6).

The browser and terminal interfaces remain available as development aids:

```bash
uv run mfp-chat  # browser chat at http://127.0.0.1:8000
uv run mfp-agent # terminal chat
```

Configuration can be changed in `.env` (`OLLAMA_MODEL`, `OLLAMA_BASE_URL`) and
`mcp.json` (MCP command, arguments, environment, and transport).

`MFP_AGENT_TIMEZONE` controls the local time used to resolve relative dates and
defaults to `Europe/Rome`. The current local timestamp plus explicit today,
yesterday, and tomorrow dates are regenerated before every model call, so a
long-running server remains correct across midnight and daylight-saving changes.

### Debug agent decisions and tool calls

Set `MFP_AGENT_DEBUG=true` in `.env` (enabled in the local setup). Each model
decision, exact MCP tool arguments, result, run ID, and error is written as JSON
Lines to `logs/agent-debug.jsonl`. Credentials and common secret fields are
redacted. Follow the trace while chatting with:

```bash
tail -f logs/agent-debug.jsonl
```

Complete prompts are excluded by default because they may contain private diary
data. Set `MFP_AGENT_LOG_PROMPTS=true` only for a short, focused debugging run.

### Log several foods at once

The agent can parse a meal containing multiple entries, select the best database
match for each food, and issue one add call per entry. For example:

```text
For dinner I had 200 g chicken breast, 70 g rice, and 250 g broccoli.
```

This produces three independently checked diary entries for Dinner. Multiple
tool calls are allowed; a single food amount is kept intact unless splitting it
is genuinely required.

When the user supplies a weight in grams, search results exposing a gram serving
are preferred and the original amount is passed to the MCP tool with `unit="g"`.
The tool handles MyFitnessPal's internal serving multiplier automatically.

Before global search, the agent now checks MyFitnessPal's meal-specific recent
and frequent lists (0 Breakfast, 1 Lunch, 2 Dinner, 3 Snacks). A matching
history item is resolved to the modern food ID and nutrition-checked; global
search remains the fallback when no match can be resolved.

Database results are treated as untrusted user-contributed data. Search and
history resolution report energy-density plausibility, and the MCP write path
refuses physically impossible records such as an olive-oil entry claiming 800
kcal per gram.

Create a compact, one-page landscape A4 PDF from a MyFitnessPal **Nutrition Summary** CSV export. The report has seven day columns, each with total consumed calories, a protein/carbohydrate/fat energy pie chart, macro grams and percentages, fibre and sodium totals, and a meal-level calorie/macro summary.

## Setup

This project uses [uv](https://docs.astral.sh/uv/). From this directory:

```bash
uv sync
```

## Create a report

```bash
uv run python nutrition_summary_pdf.py \
  /path/to/Nutrition-Summary-2026-07-13-to-2026-07-20.csv \
  --output nutrition-summary.pdf
```

By default, the report uses the seven calendar days ending on the most recent logged day in the file, retaining unlogged days in their correct positions. To choose the first date explicitly, use:

```bash
uv run python nutrition_summary_pdf.py input.csv -o weekly-report.pdf --start-date 2026-07-14
```

Generate an Italian version with `--language italian`:

```bash
uv run python nutrition_summary_pdf.py input.csv -o riepilogo-settimanale.pdf --language italian
```

Add a weekly Apple Health-style CSV to show each day's steps, estimated total energy expenditure, sleep, weight, and body fat in a compact health panel:

```bash
uv run python nutrition_summary_pdf.py nutrition.csv \
  --health-csv health_export_summary.csv \
  --language italian \
  --output riepilogo-settimanale.pdf
```

The report matches health records to nutrition records by their `Date` field. Health data is optional; days without a matching health record display a neutral unavailable state.

To omit a day’s nutrition values but preserve the seven-column weekly layout, pass `--exclude-date`. The report will show that date as a grey **Excluded day** / **Giorno escluso** column. Repeat the option to exclude multiple dates:

```bash
uv run python nutrition_summary_pdf.py input.csv --language italian \
  --exclude-date 2026-07-14 --output riepilogo-settimanale.pdf
```

The pie-chart percentages are estimated shares of macro energy (protein/carbohydrate: 4 kcal/g; fat: 9 kcal/g), rather than shares of macro grams. Source calories are displayed separately because food-label calories and macro-derived energy can legitimately differ through rounding, alcohol, fibre, and database-entry variation.
