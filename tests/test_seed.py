"""Tests for historical repository seed data."""

from __future__ import annotations

import pytest

from boggers.graph.seed import HISTORICAL_REPOS, seed_historical_nodes


def test_historical_repo_count_is_24() -> None:
    assert len(HISTORICAL_REPOS) == 24


@pytest.mark.parametrize("idx", range(24))
def test_each_repo_has_three_tuple(idx: int) -> None:
    row = HISTORICAL_REPOS[idx]
    assert len(row) == 3
    repo_id, url, sentence = row
    assert repo_id
    assert url.startswith("https://github.com/BoggersTheFish/")
    assert "http" in url
    assert len(sentence) > 10


def test_seed_inserts_all_once(tmp_store) -> None:
    n = seed_historical_nodes(tmp_store)
    assert n == 24
    assert tmp_store.count_nodes() == 24
    n2 = seed_historical_nodes(tmp_store)
    assert n2 == 0
