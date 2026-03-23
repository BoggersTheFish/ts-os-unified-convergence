"""Shared fixtures for TS-OS tests."""

from __future__ import annotations

import os

# Ensure deterministic NumPy fallbacks in tests (import before boggers loads DEFAULT_CONFIG).
os.environ.setdefault("TSOS_USE_RUST", "0")

import pytest

from boggers.graph.store import GraphStore


@pytest.fixture()
def tmp_store(tmp_path) -> GraphStore:
    db = tmp_path / "t.db"
    return GraphStore(db)
