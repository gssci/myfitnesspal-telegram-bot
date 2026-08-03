import sys

import pytest
from langchain_openai import ChatOpenAI

from mfp_agent.agent import (
    BACKEND_LLAMA_SERVER,
    BACKEND_OLLAMA,
    MyFitnessPalAgent,
)


def test_default_backend_is_llama_server(monkeypatch):
    monkeypatch.delenv("MFP_MODEL_BACKEND", raising=False)
    agent = MyFitnessPalAgent()
    assert agent.backend == BACKEND_LLAMA_SERVER


def test_backend_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("MFP_MODEL_BACKEND", "ollama")
    agent = MyFitnessPalAgent()
    assert agent.backend == BACKEND_OLLAMA


def test_explicit_backend_overrides_env(monkeypatch):
    monkeypatch.setenv("MFP_MODEL_BACKEND", "ollama")
    agent = MyFitnessPalAgent(backend="llama-server")
    assert agent.backend == BACKEND_LLAMA_SERVER


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown backend"):
        MyFitnessPalAgent(backend="bogus")


def test_build_model_llama_server_uses_chat_openai(monkeypatch):
    monkeypatch.delenv("MFP_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    agent = MyFitnessPalAgent(backend="llama-server")

    model = agent._build_model()

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gemma-4-e4b"


def test_build_model_llama_server_honors_env_overrides(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "custom-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://example.invalid/v1")
    agent = MyFitnessPalAgent(backend="llama-server")

    model = agent._build_model()

    assert model.model_name == "custom-model"


def test_build_model_ollama_without_dependency_raises_clear_error(monkeypatch):
    # langchain_ollama may or may not be installed in this environment; force
    # the import to fail either way so the error-message path is exercised.
    monkeypatch.setitem(sys.modules, "langchain_ollama", None)
    agent = MyFitnessPalAgent(backend="ollama")

    with pytest.raises(RuntimeError, match="langchain-ollama"):
        agent._build_model()


def test_build_model_ollama_uses_chat_ollama(monkeypatch):
    langchain_ollama = pytest.importorskip("langchain_ollama")
    monkeypatch.delenv("MFP_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    agent = MyFitnessPalAgent(backend="ollama")

    model = agent._build_model()

    assert isinstance(model, langchain_ollama.ChatOllama)
    assert model.model == "gemma-4-e4b"
