"""Embedding client tests."""

from __future__ import annotations

import pytest

from boggers.core.config import TSOSConfig
from boggers.embeddings.ollama import embed_text, embed_many


def test_embed_text_returns_vector() -> None:
    v = embed_text("hello world", config=TSOSConfig(embedding_dim=8))
    assert len(v) == 8
    norm = sum(x * x for x in v) ** 0.5
    assert norm > 0.01


def test_embed_many_lengths() -> None:
    vs = embed_many(["a", "b", "c"], config=TSOSConfig(embedding_dim=16))
    assert len(vs) == 3
    assert all(len(v) == 16 for v in vs)


@pytest.mark.parametrize("text", ["x", "longer phrase about graphs", "TS-OS wave"])
def test_embed_stable_dim(text: str) -> None:
    v = embed_text(text, config=TSOSConfig(embedding_dim=32))
    assert len(v) == 32
