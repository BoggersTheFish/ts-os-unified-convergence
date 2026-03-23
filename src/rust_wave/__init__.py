"""
rust_wave — compiled TS-OS hot path (PyO3).

TS-OS = Thinking System / Thinking Wave Operating System.

If the native extension is missing, callers should use `boggers.wave.rust_bridge`
fallbacks (pure NumPy). Install with: `maturin develop` from the repo root.
"""

USE_RUST_EXTENSION = False
try:
    from .rust_wave import (  # type: ignore[attr-defined, import-not-found]
        add_vectors,
        merge_candidate_pairs,
        normalize_activations_l2,
        normalize_rows,
        pairwise_cosine_similarity,
        propagate_dense,
        relax_activations,
        sum_squares,
    )

    USE_RUST_EXTENSION = True
except ImportError:  # pragma: no cover - extension optional until built
    pass

__all__ = [
    "USE_RUST_EXTENSION",
    "add_vectors",
    "merge_candidate_pairs",
    "normalize_activations_l2",
    "normalize_rows",
    "pairwise_cosine_similarity",
    "propagate_dense",
    "relax_activations",
    "sum_squares",
]
