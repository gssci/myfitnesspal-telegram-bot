# MyFitnessPal Telegram Bot

A LangChain agent, exposed via a Telegram bot, that logs food to MyFitnessPal
on your behalf. Runs entirely on your local machine; tested with a local
Gemma 4 E4B model.

**Example:** send the bot a text or voice message such as *"For dinner I had
200 g chicken breast, 70 g rice, and 250 g broccoli"* and it logs each item to
your MyFitnessPal diary under the right meal — no manual food search or
data entry required.

The tools the agent calls are implemented by
[myfitnesspal-mcp-python](https://github.com/gssci/myfitnesspal-mcp-python), a
fork of AdamWalt's original repo.

## Prerequisites

- llama-server exposing its OpenAI-compatible API at `http://127.0.0.1:8081`,
  with the Gemma 4 multimodal projector loaded.
- `ffmpeg` on `PATH`, for Telegram audio conversion.
- The [myfitnesspal-mcp-python](https://github.com/gssci/myfitnesspal-mcp-python)
  project, with its Python environment set up at the path referenced in `mcp.json`.

## Setup

```bash
uv sync
cp .env.example .env  # skip if .env already exists
```

Set `MFP_USERNAME` and `MFP_PASSWORD` in `.env`. The checked-in `mcp.json` uses
`${...}` placeholders, so credentials never live in source control.

## Telegram interface

The primary way to use the bot. Create a bot with BotFather, add its token to
`.env`, and start the service:

```dotenv
TELEGRAM_BOT_TOKEN=replace-with-your-token
# Required for access (comma-separated numeric IDs):
TELEGRAM_ALLOWED_USER_IDS=123456789
```

```bash
uv run mfp-telegram                # llama-server backend (default)
uv run mfp-telegram ollama         # local Ollama server instead
```

Access is denied by default. Send `/start` to retrieve your Telegram user ID,
add it to `TELEGRAM_ALLOWED_USER_IDS`, and restart the process. Both text and
voice messages are supported — audio is transcribed locally before being
passed to the agent. `/reset` clears the conversation; otherwise a handful of
recent turns are retained per chat, in memory only, so a restart also clears
history. On an unhandled crash, the process restarts automatically.

## Development interfaces

The browser and terminal interfaces remain available as development aids:

```bash
uv run mfp-chat  # browser chat at http://127.0.0.1:8000
uv run mfp-agent # terminal chat
```

## Configuration

The LLM connection (`OPENAI_MODEL`, `OPENAI_BASE_URL` — must include the `/v1`
suffix, `OPENAI_API_KEY`) and optional limits (`OPENAI_MAX_TOKENS`,
`MFP_AGENT_RECURSION_LIMIT`) are set in `.env`; the MCP server command,
arguments, environment, and transport are configured in `mcp.json`.
`MFP_AGENT_TIMEZONE` (default `Europe/Rome`) controls the local time used to
resolve relative dates, recomputed before every model call so a long-running
server stays correct across midnight and daylight-saving changes.
