import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from mfp_agent.agent import MyFitnessPalAgent


def test_audio_history_replaces_encoded_audio_with_summary():
    class FakeGraph:
        async def ainvoke(self, state, config):
            assert state["messages"][-1].content == (
                "Add 200 g chicken to lunch today"
            )
            return {"messages": [*state["messages"], AIMessage(content="Done")]}

    class FakeModel:
        async def ainvoke(self, messages):
            return AIMessage(content="Add 200 g chicken to lunch today")

    async def exercise():
        agent = MyFitnessPalAgent()
        agent._agent = FakeGraph()
        agent._model = FakeModel()

        answer, history = await agent.ask_audio(
            b"RIFF fake wav",
            [HumanMessage(content="Earlier request")],
            session_id="test",
        )

        assert answer == "Done"
        assert [message.content for message in history] == [
            "Earlier request",
            "Add 200 g chicken to lunch today",
            "Done",
        ]
        assert "UklGRiBmYWtlIHdhdg==" not in repr(history)

    asyncio.run(exercise())
