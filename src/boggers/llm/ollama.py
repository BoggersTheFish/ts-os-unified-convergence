"""
Chat completion for **Wave Cycle** step 10 (TENSION DETECT & BREAK/EVOLVE).

**Tension** is a scalar field: when high, the system asks the local LLM to
propose new nodes that resolve conflicting constraints.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from boggers.core.config import DEFAULT_CONFIG, TSOSConfig


def chat(
    messages: list[dict[str, str]],
    *,
    config: TSOSConfig | None = None,
    timeout: float = 120.0,
) -> str:
    """
    Call Ollama `/api/chat`. Returns empty string on failure (callers may stub).
    """
    cfg = config or DEFAULT_CONFIG
    url = f"{cfg.ollama_host.rstrip('/')}/api/chat"
    payload: dict[str, Any] = {"model": cfg.chat_model, "messages": messages, "stream": False}
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            msg = data.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content
    except (httpx.HTTPError, json.JSONDecodeError, KeyError, ValueError):
        pass
    return ""
