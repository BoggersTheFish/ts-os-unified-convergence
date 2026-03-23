"""Configuration tests."""

from __future__ import annotations

import os

import pytest

from boggers.core.config import TSOSConfig


def test_default_config_has_models() -> None:
    c = TSOSConfig()
    assert c.embed_model
    assert c.chat_model


@pytest.mark.parametrize(
    "field,val",
    [
        ("ollama_host", "http://127.0.0.1:11434"),
        ("embedding_dim", 768),
        ("convergence_merge_threshold", 0.82),
    ],
)
def test_config_fields(field: str, val: object) -> None:
    c = TSOSConfig()
    assert getattr(c, field) == val


def test_use_rust_env(monkeypatch) -> None:
    monkeypatch.setenv("TSOS_USE_RUST", "0")
    c = TSOSConfig()
    assert c.use_rust_wave is False
