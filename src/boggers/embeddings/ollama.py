"""
Embedding via **Ollama** using **nomic-embed-text** (local-first).

The **UniversalLivingGraph** stores vectors as JSON for cosine similarity
during **Wave Cycle** step 2 (PROPAGATE) and step 6 (MERGE SIMILAR).
"""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

import httpx

from boggers.core.config import DEFAULT_CONFIG, TSOSConfig


def _hash_embed(text: str, dim: int) -> list[float]:
    """Deterministic pseudo-embedding for offline tests when Ollama is down."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    out: list[float] = []
    for i in range(dim):
        # stretch hash into dim floats in [-1,1]
        b = h[i % len(h)]
        out.append((b / 127.5) - 1.0)
    # crude normalize
    s = sum(x * x for x in out) ** 0.5 or 1.0
    return [x / s for x in out]


def embed_text(
    text: str,
    *,
    config: TSOSConfig | None = None,
    timeout: float = 60.0,
) -> list[float]:
    """
    Return an embedding vector. On HTTP failure, falls back to deterministic hash
    so CI and tests stay green without a live Ollama.
    """
    cfg = config or DEFAULT_CONFIG
    url = f"{cfg.ollama_host.rstrip('/')}/api/embeddings"
    payload = {"model": cfg.embed_model, "prompt": text}
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            emb = data.get("embedding")
            if isinstance(emb, list) and emb:
                return [float(x) for x in emb]
    except (httpx.HTTPError, json.JSONDecodeError, KeyError, ValueError):
        pass
    return _hash_embed(text, cfg.embedding_dim)


def embed_many(texts: Sequence[str], *, config: TSOSConfig | None = None) -> list[list[float]]:
    return [embed_text(t, config=config) for t in texts]
