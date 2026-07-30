from __future__ import annotations

import base64
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI

from .config import load_mcp_connections
from .history import compact_history, trim_history
from .temporal import create_datetime_prompt
from .tracing import AgentTraceCallback

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

SYSTEM_PROMPT = r"""You are a fast, accurate MyFitnessPal diary assistant.

- Use tools for account data or changes; never invent results. Prefer the fewest
  calls needed and batch independent calls. Refresh browser cookies only after an
  authentication/session error.
- Before global search, get recent/frequent foods once for each requested meal and
  reuse that list for all its foods. Meals: 0 Breakfast, 1 Lunch, 2 Dinner, 3 Snacks.
  Prefer an exact semantic history match, including preparation/brand, and resolve
  its history_id. Search only when no good history match exists, resolution fails,
  or nutrition is implausible. Never pass a history_id to the add tool.
- For search results, choose the highest-ranked correct identity and preparation.
  Food identity outranks unit support. If the correct result lacks the requested
  unit, search again with a more specific food/brand/unit query; never choose a
  loosely related result merely because its units fit. Reject
  nutrition_plausibility=implausible. Get details only when needed; ask only when
  choices are materially different.
- Preserve the user's quantity exactly. For weights, 50 g means amount=50/unit="g".
  Pass the original amount and require supports_grams=true. For whole items such as
  2 kiwis,
  eggs, or bananas, pass amount=2/unit="count" when supports_count=true; do not ask
  for grams. Use a named serving unit only when the user specified it. Use
  unit="serving" only when the user explicitly said servings/portions—never for
  grams or item counts. Do not calculate database serving multipliers.
- Add each distinct food once. Pass the user's physical amount unchanged in its
  unit; the server selects the database portion and converts it. For several foods,
  resolve independently, add once each, and report partial failures. Interpret food
  as eaten/prepared unless explicitly raw or uncooked.
- Check each write result. Claim success only when success=true and verify its
  requested_amount/requested_unit. Treat any calorie/unit safety rejection as a
  failure and choose a better food result instead of retrying as servings. Mention
  the effective date.
- Keep replies brief and practical. Telegram uses legacy Markdown: only *bold*,
  _italic_, `code`, links, and plain bullets; no headings or nested formatting.
  Escape literal underscores, asterisks, backticks, and opening brackets in dynamic
  names. Stay below 3,500 characters.
"""


class MyFitnessPalAgent:
    def __init__(self) -> None:
        self._mcp_client: MultiServerMCPClient | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._agent: Any = None
        self._model: ChatOpenAI | None = None
        self.tools: list[Any] = []
        self.tool_names: list[str] = []
        self.trace_path: Path | None = None

    async def start(self) -> None:
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

        model_kwargs: dict[str, Any] = {
            "model": os.getenv("OPENAI_MODEL", "gemma-4-e4b"),
            "base_url": os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8081/v1"),
            "api_key": os.getenv("OPENAI_API_KEY", "not-needed"),
            "temperature": 0,
            "use_responses_api": False,
        }
        if max_tokens := os.getenv("OPENAI_MAX_TOKENS"):
            model_kwargs["max_tokens"] = int(max_tokens)
        self._model = ChatOpenAI(**model_kwargs)
        self._agent = create_agent(
            model=self._model,
            tools=tools,
            middleware=[create_datetime_prompt(SYSTEM_PROMPT)],
        )

    async def close(self) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None

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
        callbacks = []
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
        result = await self._agent.ainvoke(
            {"messages": messages},
            config=config,
        )
        updated = trim_history(compact_history(result["messages"]))
        answer = next(
            (message.content for message in reversed(updated) if isinstance(message, AIMessage) and message.content),
            "The agent returned no text response.",
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
