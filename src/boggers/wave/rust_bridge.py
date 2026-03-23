"""
Bridge to **rust_wave** (PyO3). When `TSOS_USE_RUST=0` or the extension is
absent, these functions fall back to NumPy with identical semantics.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from boggers.core.config import DEFAULT_CONFIG

try:
    import rust_wave as _rust_wave

    _HAS_RUST = bool(getattr(_rust_wave, "USE_RUST_EXTENSION", False))
    merge_candidate_pairs = _rust_wave.merge_candidate_pairs
    normalize_activations_l2 = _rust_wave.normalize_activations_l2
    normalize_rows = _rust_wave.normalize_rows
    pairwise_cosine_similarity = _rust_wave.pairwise_cosine_similarity
    propagate_dense = _rust_wave.propagate_dense
    relax_activations = _rust_wave.relax_activations
except Exception:  # pragma: no cover - until maturin build
    _HAS_RUST = False
    merge_candidate_pairs = None  # type: ignore[assignment]
    normalize_activations_l2 = None
    normalize_rows = None
    pairwise_cosine_similarity = None
    propagate_dense = None
    relax_activations = None


def use_rust() -> bool:
    return bool(DEFAULT_CONFIG.use_rust_wave and _HAS_RUST)


def np_normalize_rows(embeddings: list[list[float]]) -> list[list[float]]:
    a = np.asarray(embeddings, dtype=np.float64)
    if a.size == 0:
        return []
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (a / norms).tolist()


def np_pairwise_cosine(embeddings: list[list[float]]) -> list[list[float]]:
    a = np.asarray(embeddings, dtype=np.float64)
    if a.size == 0:
        return []
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    a = a / norms
    return (a @ a.T).tolist()


def np_propagate(activations: list[float], adjacency: list[list[float]], alpha: float) -> list[float]:
    a = np.asarray(activations, dtype=np.float64)
    m = np.asarray(adjacency, dtype=np.float64)
    p = m @ a
    return ((1.0 - alpha) * a + alpha * p).tolist()


def np_relax(
    activations: list[float],
    stabilities: list[float],
    base_strengths: list[float],
    rate: float,
) -> list[float]:
    a = np.asarray(activations)
    s = np.asarray(stabilities)
    b = np.asarray(base_strengths)
    return (a * (1.0 - rate) + s * rate * b).tolist()


def np_norm_l2(activations: list[float]) -> list[float]:
    a = np.asarray(activations, dtype=np.float64)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return [0.0] * len(a)
    return (a / n).tolist()


def np_merge_pairs(sim: list[list[float]], threshold: float) -> list[tuple[int, int]]:
    n = len(sim)
    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i][j] >= threshold:
                pairs.append((i, j))
    return pairs


def normalize_rows_py(embeddings: list[list[float]]) -> list[list[float]]:
    if use_rust() and normalize_rows is not None:
        return normalize_rows(embeddings)  # type: ignore[misc]
    return np_normalize_rows(embeddings)


def pairwise_cosine_py(embeddings: list[list[float]]) -> list[list[float]]:
    if use_rust() and pairwise_cosine_similarity is not None:
        return pairwise_cosine_similarity(embeddings)  # type: ignore[misc]
    return np_pairwise_cosine(embeddings)


def propagate_py(activations: list[float], adjacency: list[list[float]], alpha: float) -> list[float]:
    if use_rust() and propagate_dense is not None:
        return propagate_dense(activations, adjacency, alpha)  # type: ignore[misc]
    return np_propagate(activations, adjacency, alpha)


def relax_py(
    activations: list[float],
    stabilities: Sequence[float],
    base_strengths: Sequence[float],
    rate: float,
) -> list[float]:
    if use_rust() and relax_activations is not None:
        return relax_activations(  # type: ignore[misc]
            activations,
            list(stabilities),
            list(base_strengths),
            rate,
        )
    return np_relax(activations, list(stabilities), list(base_strengths), rate)


def norm_l2_py(activations: list[float]) -> list[float]:
    if use_rust() and normalize_activations_l2 is not None:
        return normalize_activations_l2(activations)  # type: ignore[misc]
    return np_norm_l2(activations)


def merge_pairs_py(sim: list[list[float]], threshold: float) -> list[tuple[int, int]]:
    if use_rust() and merge_candidate_pairs is not None:
        return merge_candidate_pairs(sim, threshold)  # type: ignore[misc]
    return np_merge_pairs(sim, threshold)
