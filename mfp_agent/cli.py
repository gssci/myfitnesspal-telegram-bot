from __future__ import annotations

import asyncio
from uuid import uuid4

from .agent import MyFitnessPalAgent


async def run() -> None:
    agent = MyFitnessPalAgent()
    await agent.start()
    print(f"Connected. Loaded tools: {', '.join(agent.tool_names)}")
    print("Type 'quit' to exit.\n")
    history = []
    session_id = str(uuid4())
    try:
        while True:
            try:
                prompt = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt.lower() in {"quit", "exit"}:
                break
            if not prompt:
                continue
            try:
                answer, history = await agent.ask(prompt, history, session_id=session_id)
                print(f"Agent: {answer}\n")
                if agent.trace_path:
                    print(f"Debug trace: {agent.trace_path}\n")
            except Exception as exc:
                print(f"Error: {exc}\n")
    finally:
        await agent.close()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
