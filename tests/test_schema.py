"""Schema smoke tests."""

from __future__ import annotations

from boggers.graph.schema import SCHEMA_SQL


def test_schema_contains_nodes_table() -> None:
    assert "CREATE TABLE IF NOT EXISTS nodes" in SCHEMA_SQL


def test_schema_contains_edges_table() -> None:
    assert "CREATE TABLE IF NOT EXISTS edges" in SCHEMA_SQL


def test_schema_contains_wave_log() -> None:
    assert "wave_log" in SCHEMA_SQL


def test_schema_mentions_embedding_column() -> None:
    assert "embedding" in SCHEMA_SQL
