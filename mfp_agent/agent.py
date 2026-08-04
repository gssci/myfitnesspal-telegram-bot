from __future__ import annotations

import base64
import logging
import os
from contextlib import AsyncExitStack
from pathlib import Path
from time import monotonic
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI

from .config import load_mcp_connections
from .history import compact_history, trim_history
from .logging_setup import configure_logging
from .temporal import create_datetime_prompt
from .tracing import AgentTraceCallback, ToolCallLogger

LOGGER = logging.getLogger("mfp_agent")

BACKEND_LLAMA_SERVER = "llama-server"
BACKEND_OLLAMA = "ollama"
SUPPORTED_BACKENDS = (BACKEND_LLAMA_SERVER, BACKEND_OLLAMA)
DEFAULT_BACKEND = BACKEND_LLAMA_SERVER
DEFAULT_MODEL = "gemma-4-e4b"

ESSENTIAL_MCP_TOOLS = frozenset(
    {
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
)

SYSTEM_PROMPT = r"""You are a MyFitnessPal diary assistant. You only know what the
tools tell you. Never invent food IDs, nutrition numbers, or tool results.

MEAL ARGUMENT: every meal argument accepts either the number (0=Breakfast,
1=Lunch, 2=Dinner, 3=Snacks) or the name ("Breakfast", "Lunch", "Dinner",
"Snacks"). Use whichever form that tool's own description asks for.

ID RULES:
- mfp_get_meal_foods returns a history_id. It cannot be added directly.
- Pass that history_id to mfp_resolve_meal_food to get an add-ready mfp_id.
- mfp_search_food also returns an add-ready mfp_id directly.
- mfp_add_food_to_diary only accepts an mfp_id. Never pass a history_id to it.

STEPS TO LOG ONE FOOD:
1. Call mfp_get_meal_foods once for that food's meal. If you already called it
   for this meal for an earlier food in this same request, reuse that result —
   do not call it again.
2. Look for an exact match in that result: same food, same brand, same
   preparation as the user described. If found, call mfp_resolve_meal_food with
   its history_id.
3. Call mfp_search_food instead when: no exact match exists in step 2, step 2's
   resolve failed, or its nutrition_plausibility.status is "implausible".
4. From search results, first pick the result with the correct food identity and
   preparation — identity matters more than unit support. Only then check units.
   - If the correct-identity result does not support the unit you need, search
     again with a more specific query (add brand, preparation, or unit words).
     Do not pick a different, loosely-related food just because its units fit.
   - Never pick a result whose nutrition_plausibility.status is "implausible".
5. Call mfp_get_food_details only if you still need more information to decide
   between results.
6. Call mfp_add_food_to_diary once with the chosen mfp_id.
7. Check the result. Only say it succeeded if success is true AND
   requested_amount/requested_unit match what you sent. If the tool rejected the
   amount/unit as a calorie or unit-safety problem, that is a failure — go back
   to step 4 and pick a different food result. Do not retry the same food with
   unit="serving" instead.
8. Confirm it from that same result, in the ANSWER FORMAT below. Every add
   returns nutrition (that entry) and day_totals (the whole day, including it).
   Never call mfp_get_diary to confirm, and never total the numbers yourself:
   after several adds, the LAST add's day_totals is the day's running total.

QUANTITY RULES — copy the user's number and unit, never convert it yourself:
- "50 g of X" -> amount=50, unit="g". The chosen food must have
  supports_grams=true.
- "2 kiwis" / "2 eggs" / "2 bananas" -> amount=2, unit="count". The chosen food
  must have supports_count=true. Never ask the user for a gram amount instead.
- Use a named unit like "slice" or "cup" only if the user said that word AND it
  appears in the food's count_units or serving_options.
- Use unit="serving" only if the user explicitly said "serving(s)" or
  "portion(s)". Never use unit="serving" for a gram amount or an item count.
- Never calculate or apply a database serving multiplier yourself. Send the
  amount and unit as given; the server converts it.
- Assume the food is eaten/prepared (cooked) unless the user said raw/uncooked.

MULTIPLE FOODS: resolve and add each distinct food independently, once each. If
one fails, still add the others, then report which ones failed and why.

OTHER RULES:
- Use tools for all account data and changes. Batch independent read calls
  together instead of calling them one at a time.
- Call refresh_browser_cookies only after a tool returns an authentication or
  session error — never before, and never as a first step. Then retry the exact
  same call once. An authentication error never means the food is missing, so
  never respond to one by changing the query, picking another food, or giving up
  on that food.
- mfp_get_meal_foods reporting recent_count 0 and frequent_count 0 is a real
  answer, not a failure: go straight to mfp_search_food for that meal.
- Ask the user a question only when two candidate foods are genuinely different
  foods (not just different brands or serving sizes of the same food).
- Always state which date the entries were logged under in your final answer.

ANSWER FORMAT: answers are sent to Telegram as MarkdownV2. Write plain text and
use *bold* only, always as a closed pair. No _italics_, no `code`, no headings,
no bullet or dash lists, and never write a backslash — punctuation is escaped
for you. Keep it short: one line per food, then the day's totals, then any food
that failed and why. Example:

*Added to Lunch* (2026-08-04)
*Oat flakes* — 50 g, 228 kcal, P 8.4 g, C 33.5 g, F 4.1 g
*Kiwi* — 2 count, 92 kcal, P 1.7 g, C 22 g, F 0.8 g

*Today so far* — 1487 kcal, P 96.2 g, C 150.4 g, F 48.9 g
"""


class MyFitnessPalAgent:
    def __init__(self, backend: str | None = None) -> None:
        """Create the agent, picking the LLM backend to run against.

        `backend` selects the model provider: "llama-server" (default) talks
        to a local llama-server via its OpenAI-compatible API; "ollama" talks
        to a local Ollama server instead. Both default to the gemma-4-e4b
        model. Falls back to the MFP_MODEL_BACKEND env var, then
        DEFAULT_BACKEND, when not passed explicitly.
        """
        self.backend = backend or os.getenv("MFP_MODEL_BACKEND", DEFAULT_BACKEND)
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown backend {self.backend!r}; expected one of {SUPPORTED_BACKENDS}"
            )
        self._mcp_client: MultiServerMCPClient | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._agent: Any = None
        self._model: BaseChatModel | None = None
        self.tools: list[Any] = []
        self.tool_names: list[str] = []
        self.trace_path: Path | None = None

    async def start(self) -> None:
        configure_logging()
        connections = load_mcp_connections()
        self._mcp_client = MultiServerMCPClient(connections)
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()
        tools = []
        try:
            for server_name in connections:
                session = await self._exit_stack.enter_async_context(
                    self._mcp_client.session(server_name)
                )
                server_tools = await load_mcp_tools(session)
                tools.extend(tool for tool in server_tools if tool.name in ESSENTIAL_MCP_TOOLS)
        except BaseException:
            await self.close()
            raise
        if not tools:
            raise RuntimeError("The MCP server connected but returned no tools")
        self.tools = tools
        self.tool_names = [tool.name for tool in tools]

        self._model = self._build_model()
        self._agent = create_agent(
            model=self._model,
            tools=tools,
            middleware=[create_datetime_prompt(SYSTEM_PROMPT)],
        )

    async def close(self) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None

    def _build_model(self) -> BaseChatModel:
        """Build the chat model for whichever backend was selected at launch."""
        model_name = os.getenv("MFP_MODEL", DEFAULT_MODEL)
        if self.backend == BACKEND_OLLAMA:
            try:
                from langchain_ollama import ChatOllama
            except ImportError as exc:
                raise RuntimeError(
                    "The 'ollama' backend requires the langchain-ollama package. "
                    "Run `uv sync` (it's already listed in pyproject.toml) and try again."
                ) from exc
            return ChatOllama(
                model=os.getenv("OLLAMA_MODEL", model_name),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                temperature=0,
            )

        model_kwargs: dict[str, Any] = {
            "model": os.getenv("OPENAI_MODEL", model_name),
            "base_url": os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8081/v1"),
            "api_key": os.getenv("OPENAI_API_KEY", "not-needed"),
            "temperature": 0,
            "use_responses_api": False,
        }
        if max_tokens := os.getenv("OPENAI_MAX_TOKENS"):
            model_kwargs["max_tokens"] = int(max_tokens)
        return ChatOpenAI(**model_kwargs)

    async def ask(
        self,
        text: str,
        history: list[BaseMessage] | None = None,
        *,
        session_id: str | None = None,
    ) -> tuple[str, list[BaseMessage]]:
        return await self._invoke(
            HumanMessage(content=text),
            history,
            session_id=session_id,
        )

    async def ask_audio(
        self,
        wav_audio: bytes,
        history: list[BaseMessage] | None = None,
        *,
        session_id: str | None = None,
    ) -> tuple[str, list[BaseMessage]]:
        """Transcribe one WAV request, then handle and retain it as text."""
        if not wav_audio:
            raise ValueError("wav_audio must not be empty")
        encoded = base64.b64encode(wav_audio).decode("ascii")
        summary = await self._summarize_audio(encoded)
        return await self._invoke(
            HumanMessage(content=summary),
            history,
            session_id=session_id,
        )

    async def _summarize_audio(self, encoded_wav: str) -> str:
        if self._model is None:
            raise RuntimeError("Agent has not been started")
        response = await self._model.ainvoke(
            [
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": (
                                "Transcribe and compactly summarize this spoken user "
                                "request for conversation history. Preserve every food, "
                                "amount, unit, meal, and date. Return only the user's "
                                "request in plain text, without commentary."
                            ),
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": encoded_wav,
                                "format": "wav",
                            },
                        },
                    ]
                )
            ]
        )
        summary = self._message_text(response.content).strip()
        if not summary:
            raise RuntimeError("Model returned an empty audio transcription")
        return summary

    async def _invoke(
        self,
        user_message: HumanMessage,
        history: list[BaseMessage] | None,
        *,
        session_id: str | None,
    ) -> tuple[str, list[BaseMessage]]:
        if self._agent is None:
            raise RuntimeError("Agent has not been started")
        messages = [*trim_history(compact_history(history or [])), user_message]
        callbacks: list[Any] = [ToolCallLogger()]
        if os.getenv("MFP_AGENT_DEBUG", "false").lower() in {"1", "true", "yes", "on"}:
            self.trace_path = Path(os.getenv("MFP_AGENT_LOG", "logs/agent-debug.jsonl")).resolve()
            callbacks.append(
                AgentTraceCallback(
                    self.trace_path,
                    include_prompts=os.getenv("MFP_AGENT_LOG_PROMPTS", "false").lower()
                    in {"1", "true", "yes", "on"},
                )
            )
        config: dict[str, Any] = {
            "callbacks": callbacks,
            "metadata": {"chat_session_id": session_id},
        }
        if recursion_limit := os.getenv("MFP_AGENT_RECURSION_LIMIT"):
            config["recursion_limit"] = int(recursion_limit)
        started = monotonic()
        LOGGER.info("session=%s request started", session_id)
        try:
            result = await self._agent.ainvoke(
                {"messages": messages},
                config=config,
            )
        except Exception:
            LOGGER.exception(
                "session=%s request failed after %.2fs", session_id, monotonic() - started
            )
            raise
        updated = trim_history(compact_history(result["messages"]))
        answer = next(
            (message.content for message in reversed(updated) if isinstance(message, AIMessage) and message.content),
            "The agent returned no text response.",
        )
        LOGGER.info(
            "session=%s request finished in %.2fs", session_id, monotonic() - started
        )
        return self._message_text(answer), updated

    @staticmethod
    def _message_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [
                block["text"]
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ]
            if text_parts:
                return "\n".join(text_parts)
        return str(content)
