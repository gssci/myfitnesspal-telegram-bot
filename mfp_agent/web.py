from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent import MyFitnessPalAgent

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    agent = MyFitnessPalAgent()
    await agent.start()
    app.state.agent = agent
    app.state.sessions = {}
    try:
        yield
    finally:
        await agent.close()


app = FastAPI(title="MyFitnessPal Agent", lifespan=lifespan)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, object]:
    return {"status": "ok", "tools": app.state.agent.tool_names}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or str(uuid4())
    history = app.state.sessions.get(session_id, [])
    try:
        answer, updated = await app.state.agent.ask(
            request.message, history, session_id=session_id
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    app.state.sessions[session_id] = updated
    return ChatResponse(answer=answer, session_id=session_id)


def main() -> None:
    uvicorn.run(
        "mfp_agent.web:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
