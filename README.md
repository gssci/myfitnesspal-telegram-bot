# MyFitnessPal Telegram Agent

A local LangChain chat agent that connects an OpenAI-compatible llama-server
(or Ollama) to the MyFitnessPal MCP server, exposed primarily through a
Telegram bot. It exposes only the essential diary, food-search, reporting,
and authentication tools to the model.

The tools the agent calls are implemented by
[myfitnesspal-mcp-python](https://github.com/gssci/myfitnesspal-mcp-python),
a fork of [AdamWalt/myfitnesspal-mcp-python](https://github.com/AdamWalt/myfitnesspal-mcp-python).
That project must be cloned and set up separately; this repo only talks to it
over MCP per the configuration in `mcp.json`.

**Example:** send the bot a text or voice message like *"For dinner I had 200 g
chicken breast, 70 g rice, and 250 g broccoli"* and it logs each item to your
MyFitnessPal diary under the right meal — no manual food search or entering
measurements by hand.

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
`langchain-ollama`, already listed in `pyproject.toml`.

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
