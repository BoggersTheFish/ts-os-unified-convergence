"""Tests for Rust/NumPy bridge fallbacks."""

from __future__ import annotations

import pytest

from boggers.wave import rust_bridge as rb


def test_normalize_rows_py_identity_square() -> None:
    m = [[1.0, 0.0], [0.0, 1.0]]
    out = rb.normalize_rows_py(m)
    assert len(out) == 2
    assert abs(out[0][0] - 1.0) < 1e-6


def test_pairwise_cosine_parallel() -> None:
    m = [[1.0, 0.0], [1.0, 0.0]]
    sim = rb.pairwise_cosine_py(m)
    assert sim[0][1] > 0.99


def test_propagate_py_mixes() -> None:
    a = [1.0, 0.0]
    adj = [[0.5, 0.5], [0.5, 0.5]]
    out = rb.propagate_py(a, adj, 0.5)
    assert len(out) == 2


def test_relax_py() -> None:
    out = rb.relax_py([1.0, 1.0], [0.5, 0.5], [1.0, 1.0], 0.1)
    assert len(out) == 2


def test_norm_l2_py() -> None:
    out = rb.norm_l2_py([3.0, 4.0])
    assert abs(out[0] - 0.6) < 1e-6
    assert abs(out[1] - 0.8) < 1e-6


def test_merge_pairs_py() -> None:
    sim = [[1.0, 0.9], [0.9, 1.0]]
    pairs = rb.merge_pairs_py(sim, 0.85)
    assert (0, 1) in pairs


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_propagate_alpha_sweep(alpha: float) -> None:
    a = [0.2, 0.8]
    adj = [[1.0, 0.0], [0.0, 1.0]]
    out = rb.propagate_py(a, adj, alpha)
    assert len(out) == 2
