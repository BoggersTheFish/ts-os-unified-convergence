"""LLM client smoke tests."""

from __future__ import annotations

from boggers.llm.ollama import chat


def test_chat_graceful_offline() -> None:
    # With no Ollama, chat returns empty string — must not raise.
    out = chat([{"role": "user", "content": "ping"}])
    assert isinstance(out, str)
