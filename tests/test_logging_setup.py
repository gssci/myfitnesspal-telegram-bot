import logging

from mfp_agent.logging_setup import configure_logging


def test_configure_logging_honors_log_level_env(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level
    try:
        root.handlers = []
        configure_logging()
        assert root.level == logging.DEBUG
    finally:
        root.handlers = previous_handlers
        root.setLevel(previous_level)


def test_configure_logging_quiets_httpx():
    configure_logging()
    assert logging.getLogger("httpx").level == logging.WARNING
