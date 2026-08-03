Esempio: uv run python nutrition_summary_pdf.py /Users/savioscibetta/Downloads/File-Export-2026-07-27-to-2026-08-02/Nutrition-Summary-2026-07-27-to-2026-08-02.csv --health-csv /Users/savioscibetta/Downloads/health_export_summary_20260802.csv --language italian --output riepilogo-nutrizione-e-salute-2026-07-27-al-2026-08-02.pdf

# Nutrition Summary PDF

This repository also includes a local LangChain chat agent that connects an
OpenAI-compatible llama-server to the MyFitnessPal MCP server. It exposes only
the essential diary, food-search, reporting, and authentication tools to the
model.

## MyFitnessPal chat agent

Prerequisites:

- llama-server exposing its OpenAI-compatible API at `http://127.0.0.1:8081`,
  with the Gemma 4 multimodal projector loaded.
- `ffmpeg` available on `PATH` for Telegram audio conversion.
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
uv run mfp-telegram                # llama-server backend (default)
uv run mfp-telegram llama-server   # same, explicit
uv run mfp-telegram ollama         # local Ollama server instead
```

The backend picks which local LLM server the agent talks to; both use the
`gemma-4-e4b` model. `llama-server` (the default) talks to a local llama-server
OpenAI-compatible endpoint (`OPENAI_BASE_URL`, `OPENAI_MODEL`,
`OPENAI_API_KEY`). `ollama` talks to a local Ollama server instead
(`OLLAMA_BASE_URL`, default `http://127.0.0.1:11434`; `OLLAMA_MODEL`). The
choice can also be set with `MFP_MODEL_BACKEND=ollama` in `.env` instead of a
command-line argument; an explicit argument wins. Selecting `ollama` requires
`langchain-ollama`, already listed in `pyproject.toml` — run `uv sync` after
pulling this change.

If the bot process crashes (unhandled exception, MCP/LLM server unreachable at
startup, etc.), it relaunches itself automatically with exponential backoff
(5s, 10s, 20s, ... capped at 5 minutes; the backoff resets after a run stays up
for at least a minute). A clean stop (Ctrl+C / SIGTERM) exits normally without
restarting.

The bot is closed by default. With no allowed IDs configured, `/start` reports
the sender's own numeric Telegram ID but all agent requests are denied. Send
`/start`, add your ID to `TELEGRAM_ALLOWED_USER_IDS`, and restart the process.
Only listed users can then reach the agent. Text, Telegram voice messages, and
audio files are forwarded to the LangChain agent. Audio is converted to 16 kHz
mono 16-bit PCM WAV and Gemma 4 produces a compact, faithful transcription,
which the tool-enabled agent handles as the user request. Requests are processed
one at a time. After an audio response, only that textual transcription is
retained in history; the encoded audio is discarded.
Recent context is isolated by Telegram chat and user; `/reset` clears it.
History is held in process memory, so restarting the bot also clears it.
`MFP_CHAT_HISTORY_TURNS` controls the number of recent turns retained for all
interfaces (default: 6).

The browser and terminal interfaces remain available as development aids:

```bash
uv run mfp-chat  # browser chat at http://127.0.0.1:8000
uv run mfp-agent # terminal chat
```

Configuration can be changed in `.env` (`OPENAI_MODEL`, `OPENAI_BASE_URL`, and
`OPENAI_API_KEY`) and `mcp.json` (MCP command, arguments, environment, and
transport). `OPENAI_BASE_URL` must include the `/v1` suffix. The default model
name is `gemma-4-e4b`; llama-server accepts it when serving a single model.
`OPENAI_MAX_TOKENS` and `MFP_AGENT_RECURSION_LIMIT` can optionally be set if response token limits or tool recursion limits are needed.

`MFP_AGENT_TIMEZONE` controls the local time used to resolve relative dates and
defaults to `Europe/Rome`. The current local timestamp plus explicit today,
yesterday, and tomorrow dates are regenerated before every model call, so a
long-running server remains correct across midnight and daylight-saving changes.

### See what the agent and its tools are doing

Every tool call is logged live, regardless of debug mode. Each entry point
(`mfp-agent`, `mfp-chat`, the Telegram bot) configures logging the same way via
`LOG_LEVEL` (defaults to `INFO`), so a normal run already prints one line per
tool call and one per request:

```text
2026-08-01 12:00:01 INFO mfp_agent: session=telegram:1:1 request started
2026-08-01 12:00:02 INFO mfp_agent.tools: -> mfp_search_food(query=banana)
2026-08-01 12:00:03 INFO mfp_agent.tools: <- mfp_search_food (0.84s): {"query": "banana", "count": 5, ...}
2026-08-01 12:00:04 INFO mfp_agent.tools: -> mfp_add_food_to_diary(mfp_id=abc123, meal=Breakfast, amount=120, unit=g)
2026-08-01 12:00:04 INFO mfp_agent.tools: <- mfp_add_food_to_diary (0.31s): Successfully added 120 g of Banana to Breakfast
2026-08-01 12:00:04 INFO mfp_agent: session=telegram:1:1 request finished in 3.10s
```

Sensitive fields (passwords, tokens, cookies, secrets) are always redacted, and
long arguments/results are truncated to keep each line readable. The
`myfitnesspal` MCP server logs the same way on its own side (tool name,
arguments, timing, and a result preview), including a WARNING for any tool that
returns a caught error string instead of raising — those lines show up
alongside the agent's, since the server runs as a subprocess of the agent.

For deeper, structured debugging (full JSON per model decision and tool call,
including run IDs) set `MFP_AGENT_DEBUG=true` in `.env` (enabled in the local
setup). That trace is written as JSON Lines to `logs/agent-debug.jsonl`:

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
Whole-item quantities use `unit="count"`, which maps to discrete database units
such as fruit, piece, egg size, or another named item serving without requiring
the user to provide grams.

The agent checks meal-specific recent and frequent entries before global search.
A matching history item is resolved to a current food ID; the server caches that
lookup briefly so resolution does not repeat the recent/frequent HTTP requests.

Completed internal tool calls and their potentially large results are removed
from retained chat history. User messages and final answers remain available for
conversation continuity without resending old tool payloads.

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
