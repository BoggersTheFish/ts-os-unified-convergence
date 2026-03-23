"""GraphStore tests."""

from __future__ import annotations

import pytest

from boggers.graph.store import GraphStore


def test_upsert_historical_idempotent(tmp_store: GraphStore) -> None:
    assert tmp_store.upsert_historical(
        repo_id="A",
        url="https://github.com/BoggersTheFish/A",
        content="hello",
        topics=["t"],
        meta={"k": 1},
    )
    assert not tmp_store.upsert_historical(
        repo_id="A",
        url="https://github.com/BoggersTheFish/A",
        content="hello",
        topics=["t"],
        meta={"k": 1},
    )


def test_update_embedding(tmp_store: GraphStore) -> None:
    tmp_store.upsert_historical(
        repo_id="A",
        url="u",
        content="c",
        topics=[],
        meta={},
    )
    n = tmp_store.list_nodes()[0]
    tmp_store.update_node_embedding(n.id, [0.1, 0.2])
    again = tmp_store.list_nodes()[0]
    assert again.embedding == [0.1, 0.2]


def test_edges_roundtrip(tmp_store: GraphStore) -> None:
    tmp_store.upsert_historical(repo_id="A", url="u", content="c", topics=[], meta={})
    tmp_store.upsert_historical(repo_id="B", url="u", content="c", topics=[], meta={})
    a = tmp_store.list_nodes()[0].id
    b = tmp_store.list_nodes()[1].id
    tmp_store.insert_edge(a, b, 0.5)
    es = tmp_store.list_edges()
    assert len(es) >= 1


def test_prune_edges(tmp_store: GraphStore) -> None:
    tmp_store.upsert_historical(repo_id="A", url="u", content="c", topics=[], meta={})
    tmp_store.upsert_historical(repo_id="B", url="u", content="c", topics=[], meta={})
    a = tmp_store.list_nodes()[0].id
    b = tmp_store.list_nodes()[1].id
    tmp_store.insert_edge(a, b, 0.01)
    pruned = tmp_store.prune_edges_below(0.05)
    assert pruned >= 1


def test_merge_nodes_concatenates(tmp_store: GraphStore) -> None:
    tmp_store.upsert_historical(repo_id="A", url="u", content="alpha", topics=[], meta={})
    tmp_store.upsert_historical(repo_id="B", url="u", content="beta", topics=[], meta={})
    ids = sorted(n.id for n in tmp_store.list_nodes())
    keep, drop = ids[0], ids[1]
    tmp_store.merge_nodes(keep, drop)
    nodes = tmp_store.list_nodes()
    assert len(nodes) == 1
    assert "alpha" in nodes[0].content and "beta" in nodes[0].content


def test_insert_spawned_node(tmp_store: GraphStore) -> None:
    nid = tmp_store.insert_spawned_node(
        content="spawn",
        topics=["x"],
        embedding=[0.0, 1.0],
        meta={"m": 1},
    )
    assert nid > 0
    assert tmp_store.count_nodes() == 1


def test_wave_log(tmp_store: GraphStore) -> None:
    tmp_store.log_wave_step("X", "d", 0.2, convergence_mode=True)
    logs = tmp_store.recent_logs(5)
    assert logs[0]["step_name"] == "X"
    assert logs[0]["convergence_mode"] is True


@pytest.mark.parametrize("k", range(5))
def test_bulk_activation_map(tmp_store: GraphStore, k: int) -> None:
    tmp_store.upsert_historical(repo_id=f"R{k}", url="u", content="c", topics=[], meta={})
    nodes = tmp_store.list_nodes()
    m = {n.id: float(i) * 0.1 for i, n in enumerate(nodes)}
    tmp_store.bulk_set_activations_map(m)
    for n in tmp_store.list_nodes():
        assert n.activation == pytest.approx(m[n.id])
