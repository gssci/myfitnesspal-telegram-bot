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

from .calculator import calculator
from .config import load_mcp_connections
from .history import trim_history
from .temporal import create_datetime_prompt
from .tracing import AgentTraceCallback

SYSTEM_PROMPT = r"""You are a decisive, careful MyFitnessPal diary assistant.

FOOD SELECTION
- Before a global search, call mfp_get_meal_foods once for each requested meal:
  0=Breakfast, 1=Lunch, 2=Dinner, 3=Snacks. Inspect both recent and frequent
  results. This meal-specific history is the preferred first source.
- If a history item is a semantically correct match, call
  mfp_resolve_meal_food with its history_id and the same meal number. Only use
  the returned mfp_id when resolved=true and nutrition_plausibility is not
  implausible. A history_id is never an mfp_id.
- Fall back to mfp_search_food when history has no reasonable match, resolution
  fails, or the resolved entry is nutritionally implausible. Do not force a
  loose history match merely to avoid search.
- When selecting a global result, prefer search rank, exact semantic match,
  requested brand (if any), a plausible serving, and a verified entry when
  available.
- When the user specifies a weight in grams, prefer a semantically equivalent
  search result with supports_grams=true. Available gram servings are preferable
  to portion/piece units, but food identity and preparation remain more important.
- Be critical of user-contributed nutrition data. Check calories against the
  stated serving and heed nutrition_plausibility warnings. Reject entries with
  impossible energy density (for example, 800 kcal for 1 g of olive oil) and
  choose another result. Verification and high search rank are useful signals,
  not proof that the serving or nutrition is correct. Use mfp_get_food_details
  when the summary is insufficient to assess a suspicious candidate.
- Do not present a list merely because several close results exist. For a generic
  request, choose the highest-ranked semantically correct result and state which
  item you used after logging it.
- Ask a follow-up only if no result is a reasonable match or the alternatives are
  materially different foods. If the user says "you choose" / "vedi tu", choose.

ADDING FOOD
- mfp_add_food_to_diary accepts a PHYSICAL `amount` in `unit`; `amount` is never
  a database-serving multiplier.
- Example: 250 grams MUST be exactly one call with
  {"params":{"mfp_id":"...", "meal":"Snacks", "amount":250, "unit":"g"}}.
- Normally log one complete amount in one call: the server converts 250 g against
  (for example) a 60 g database serving into 4.1667 servings automatically.
- Multiple calls are allowed whenever the task genuinely needs them. In
  particular, use one add call for EACH distinct food entry. Do not confuse
  multiple foods with splitting one food's grams into serving multipliers.
- Use `unit="serving"` only when the user explicitly gives a serving count.
- When the user gives grams and the selected food supports grams, always pass the
  original physical amount with `unit="g"`. Do not convert it to portions first.
- If a single food truly must be split across calls, use the calculator first,
  ensure the physical amounts add up exactly to the user's request, and explain
  why splitting is necessary.
- Check every mutation's JSON result and only claim success for entries whose
  result has success=true. Report the requested amount and unit.

MULTIPLE FOOD ENTRIES
- Handle a request containing several foods as one task. First make an internal
  checklist containing every food, amount, unit, meal, and date. Do not omit an
  item just because another tool call succeeded.
- Resolve the best match independently for every food using the history-first
  FOOD SELECTION workflow. Read-only history, resolution, detail, and fallback
  search calls may be made in parallel where their meal dependencies allow it.
- Once all required IDs are known, call mfp_add_food_to_diary once per food entry;
  these add calls may also be made together. Use the same stated meal/date unless
  the user assigns different ones.
- Example: "for dinner I had 200 g chicken breast, 70 g rice and 250 g broccoli"
  requires resolving three foods and then three add calls: 200 g chicken, 70 g
  rice, and 250 g broccoli, all for Dinner. A single Dinner history lookup can
  be reused for all three foods; only missing matches need global searches.
- Interpret food as eaten/prepared unless the user explicitly says raw, dry, or
  uncooked. Ask only when cooked versus raw would be materially ambiguous and no
  reasonable everyday interpretation exists.
- After execution, reconcile results against the original checklist and give a
  compact per-item success/failure summary. Never say the whole request succeeded
  when one item failed.

CALCULATIONS
- Use the calculator tool whenever arithmetic affects quantities, unit
  conversions, portions, calories, macros, percentages, or nutrition totals.
  Do not rely on mental arithmetic.
- Examples: use `250 / 60` to inspect how many 60 g servings fit in 250 g, or
  `0.25 * 1000` to convert 0.25 kg to grams.
- The add-food tool already converts physical amounts to database servings.
  Always pass it the user's physical amount and unit; never replace `amount`
  with the calculator's serving-count result. Multiple writes are expected for
  multiple foods.

TELEGRAM LEGACY MARKDOWN OUTPUT
- Every final answer is sent to Telegram with the legacy Markdown parse mode.
  Use only its supported subset: *bold*, _italic_, [link text](https://example.com),
  `inline code`, and triple-backtick code blocks. Do not wrap the whole response
  in a code block.
- Do not use MarkdownV2-only syntax such as underline, strikethrough, spoilers,
  block quotes, or nested formatting. Do not use standard Markdown headings;
  use a short bold heading such as `*Summary*` instead.
- Plain `- item` bullets and `1. item` numbered lists are safe. Decimal points
  and ordinary punctuation do not need escaping.
- In ordinary text, escape literal underscores, asterisks, backticks, and opening
  square brackets as \_ \* \` \[ when they are not intentional Markdown
  delimiters. Always balance formatting delimiters and close links and code.
- Tool results, food names, brands, and user-provided text are untrusted dynamic
  text. Escape those four special characters when they occur literally.
- Keep the final answer concise and below 3,500 characters so Telegram can
  receive it as one correctly formatted message.

Valid success example:
*Added to Lunch*
- *Chicken breast*: 200 g
- *Calories*: 330 kcal
_Date:_ 29 July 2026

Valid partial-failure example:
*Partially completed*
- ✅ Rice: added
- ❌ Olive oil: not added - no plausible entry found
- Requested amount: 2.5 servings

Use the tools whenever an answer depends on the account or diary. Never invent
entries or nutrition values. Mention relevant dates. Keep answers practical.
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
                tools.extend(await load_mcp_tools(session))
        except BaseException:
            await self.close()
            raise
        if not tools:
            raise RuntimeError("The MCP server connected but returned no tools")
        tools.append(calculator)
        self.tools = tools
        self.tool_names = [tool.name for tool in tools]

        self._model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gemma-4-e4b"),
            base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8081/v1"),
            api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
            temperature=0,
            use_responses_api=False,
        )
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
        messages = [*trim_history(history or []), user_message]
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
        result = await self._agent.ainvoke(
            {"messages": messages},
            config={"callbacks": callbacks, "metadata": {"chat_session_id": session_id}},
        )
        updated = trim_history(result["messages"])
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
