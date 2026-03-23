"""
TS-OS runtime configuration.

**Tension** is a scalar field that drives evolution: when conceptual strain rises,
the system may spawn new nodes or break unstable merges.

Local-first defaults target **Ollama** with **nomic-embed-text** for embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class TSOSConfig:
    """Immutable settings for graph I/O and model endpoints."""

    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    embed_model: str = os.environ.get("TSOS_EMBED_MODEL", "nomic-embed-text")
    chat_model: str = os.environ.get("TSOS_CHAT_MODEL", "llama3.2")
    embedding_dim: int = 768  # nomic-embed-text default
    use_rust_wave: bool = os.environ.get("TSOS_USE_RUST", "1") not in ("0", "false", "False")
    convergence_merge_threshold: float = 0.82
    convergence_tension_boost: float = 1.35
    wave_prune_edge_threshold: float = 0.08
    wave_split_activation: float = 2.5


DEFAULT_CONFIG = TSOSConfig()
