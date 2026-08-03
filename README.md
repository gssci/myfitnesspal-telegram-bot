# MyFitnessPal Telegram Bot

An AI Agent developed using Langchain connected to a Telegram Bot interface to quickly log
food to the MyFitnessPal app.
Completely runnable on your local machine and tested with local Gemma 4 E4B model.

**Example usage:** send the bot a text or voice message like *"For dinner I had 200 g
chicken breast, 70 g rice, and 250 g broccoli"* and it logs each item to your
MyFitnessPal diary under the right meal — no manual food search or entering
measurements by hand.

The tools the agent calls are implemented by
[myfitnesspal-mcp-python](https://github.com/gssci/myfitnesspal-mcp-python),
a fork of AdamWalt's original repo.

## Prerequisites

- llama-server exposing its OpenAI-compatible API at `http://127.0.0.1:8081`,
  with the Gemma 4 multimodal projector loaded.
- `ffmpeg` available on `PATH` for Telegram audio conversion.
- The [myfitnesspal-mcp-python](https://github.com/gssci/myfitnesspal-mcp-python)
  project and its Python environment set up at the path referenced in `mcp.json`.

## Setup

Install everything and copy the environment template if needed:

```bash
uv sync
cp .env.example .env  # skip this if .env already exists
```

Set `MFP_USERNAME` and `MFP_PASSWORD` in `.env`. The checked-in `mcp.json` uses
`${...}` placeholders, so credentials are not stored in source control.

## Telegram interface

This is the main way you'll actually use the bot day to day. Grab a token from
BotFather, drop it into `.env`, and start it up:

```dotenv
TELEGRAM_BOT_TOKEN=replace-with-your-token
# Required for access (comma-separated numeric IDs):
TELEGRAM_ALLOWED_USER_IDS=123456789
```

```bash
uv run mfp-telegram                # llama-server backend (default)
uv run mfp-telegram ollama         # or point it at a local Ollama server instead
```

The bot is locked down by default — nobody can talk to it until you add your
own Telegram user ID to `TELEGRAM_ALLOWED_USER_IDS`. Send `/start` first to get
your ID, then add it and restart. From there you can just talk to it: text or
voice notes both work, since incoming audio gets transcribed locally before
being handed to the agent. `/reset` wipes the conversation if you want a clean
slate; otherwise it remembers a handful of recent turns per chat (in memory
only, so a restart clears it too). If the process ever dies unexpectedly it
restarts itself automatically rather than needing a babysitter.

## Development interfaces

The browser and terminal interfaces remain available as development aids:

```bash
uv run mfp-chat  # browser chat at http://127.0.0.1:8000
uv run mfp-agent # terminal chat
```

## Configuration

Set the LLM connection (`OPENAI_MODEL`, `OPENAI_BASE_URL` — must include the
`/v1` suffix, `OPENAI_API_KEY`) and optional limits (`OPENAI_MAX_TOKENS`,
`MFP_AGENT_RECURSION_LIMIT`) in `.env`; the MCP server command, arguments,
environment, and transport are configured in `mcp.json`. `MFP_AGENT_TIMEZONE`
(default `Europe/Rome`) controls the local time used to resolve relative
dates, regenerated before every model call so a long-running server stays
correct across midnight and daylight-saving changes.
