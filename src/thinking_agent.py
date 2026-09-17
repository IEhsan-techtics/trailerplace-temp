from __future__ import annotations

from typing import Any


def thinking_agent_enabled() -> bool:
    return False


def thinking_agent_background() -> bool:
    return False


def generate_thinking_flow(payload: dict[str, Any]) -> dict[str, str]:
    return {"status": "ok", "thinking_markdown": ""}


def log_thinking_flow(sid: str, payload: dict[str, Any], result: dict[str, Any]) -> None:
    return None
