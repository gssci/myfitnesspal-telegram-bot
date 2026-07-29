from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def load_mcp_connections(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load a standard mcp.json and resolve ${ENV_VAR} placeholders."""
    load_dotenv(PROJECT_ROOT / ".env")
    config_path = path or PROJECT_ROOT / "mcp.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    servers = raw.get("mcpServers")
    if not isinstance(servers, dict) or not servers:
        raise ValueError(f"No mcpServers found in {config_path}")

    resolved = _resolve_env(servers)
    for name, connection in resolved.items():
        connection.setdefault("transport", "stdio")
        command = connection.get("command")
        if command and not Path(command).is_file():
            raise FileNotFoundError(f"MCP server '{name}' command not found: {command}")
    return resolved


def _resolve_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        result = os.getenv(name)
        if result is None:
            raise ValueError(f"Required environment variable {name} is not set")
        return result

    return ENV_PATTERN.sub(replace, value)
