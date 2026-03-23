"""
Implements the **Wave Cycle** — the exact 11-step autonomous TS-OS loop.

**TS-OS** = Thinking System / Thinking Wave Operating System.

**Tension** is a scalar synthesized from activation variance, contradiction counts,
and unresolved merge pressure; it feeds step 10 and the manifesto narrative.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any

from boggers.core.config import DEFAULT_CONFIG, TSOSConfig
from boggers.embeddings.ollama import embed_text
from boggers.graph.store import GraphStore, NodeRow
from boggers.llm.ollama import chat
from boggers.wave.rust_bridge import norm_l2_py, pairwise_cosine_py, propagate_py, relax_py


@dataclass
class WaveResult:
    """Structured outcome of one wave pass for UI and tests."""

    tension: float
    elected_id: int | None
    merged_pairs: list[tuple[int, int]]
    spawned_ids: list[int]
    steps: list[dict[str, Any]] = field(default_factory=list)
    contradiction_hits: int = 0


class WaveEngine:
    """
    Runs the Wave Cycle over a `GraphStore`.

    **UniversalLivingGraph** nodes hold `embedding` JSON; edges are rebuilt
    periodically from cosine similarity for local demos.
    """

    def __init__(self, store: GraphStore, config: TSOSConfig | None = None) -> None:
        self.store = store
        self.config = config or DEFAULT_CONFIG

    def ensure_embeddings(self, nodes: list[NodeRow] | None = None) -> int:
        """Fill missing embeddings using Ollama (or deterministic fallback)."""
        nodes = nodes or self.store.list_nodes()
        n = 0
        for node in nodes:
            if node.embedding:
                continue
            vec = embed_text(node.content, config=self.config)
            self.store.update_node_embedding(node.id, vec)
            n += 1
        return n

    def _ordered_nodes(self) -> list[NodeRow]:
        return sorted(self.store.list_nodes(), key=lambda x: x.id)

    def _adjacency(self, nodes: list[NodeRow]) -> tuple[list[int], list[list[float]]]:
        """Return node ids in order and dense symmetric adjacency weighted by edges."""
        ids = [n.id for n in nodes]
        idx = {nid: i for i, nid in enumerate(ids)}
        n = len(ids)
        adj = [[0.0] * n for _ in range(n)]
        for i in range(n):
            adj[i][i] = 1.0
        for a, b, w, _ in self.store.list_edges():
            if a in idx and b in idx:
                ia, ib = idx[a], idx[b]
                adj[ia][ib] += float(w)
                adj[ib][ia] += float(w)
        # row-normalize for stable propagation
        for i in range(n):
            s = sum(adj[i]) or 1.0
            for j in range(n):
                adj[i][j] /= s
        return ids, adj

    def run_wave(
        self,
        *,
        convergence_mode: bool = False,
        merge_threshold: float | None = None,
        on_step: Callable[[dict[str, Any]], None] | None = None,
    ) -> WaveResult:
        """
        Execute one full **Wave Cycle**.

        When `convergence_mode` is True (first background run = **Convergence Wave**),
        merging is more aggressive and **tension** is amplified to visualize
        cross-repository fusion in the Streamlit demo.
        """
        thr = merge_threshold
        if thr is None:
            thr = (
                self.config.convergence_merge_threshold * 0.92
                if convergence_mode
                else self.config.convergence_merge_threshold
            )

        nodes = self._ordered_nodes()
        if not nodes:
            return WaveResult(tension=0.0, elected_id=None, merged_pairs=[], spawned_ids=[])

        self.ensure_embeddings(nodes)
        nodes = self._ordered_nodes()

        steps: list[dict[str, Any]] = []
        tension_accum = 0.0

        # --- 1. ELECT STRONGEST ---
        elected = max(nodes, key=lambda x: x.activation)
        elected_id = elected.id
        activations = [n.activation + (0.15 if n.id == elected_id else 0.0) for n in nodes]
        steps.append({"step": 1, "name": "ELECT STRONGEST", "detail": f"node {elected_id}"})
        if on_step:
            on_step(steps[-1])
        self.store.log_wave_step("ELECT STRONGEST", f"node {elected_id}", 0.0, convergence_mode=convergence_mode)

        ids, adj = self._adjacency(nodes)
        stabilities = [n.stability for n in nodes]
        bases = [n.base_strength for n in nodes]

        # --- 2. PROPAGATE (topo + cosine) ---
        alpha = 0.42 if convergence_mode else 0.35
        activations = propagate_py(activations, adj, alpha)
        steps.append({"step": 2, "name": "PROPAGATE", "detail": f"alpha={alpha:.2f}"})
        if on_step:
            on_step(steps[-1])
        self.store.log_wave_step("PROPAGATE", "topo+cosine", 0.05, convergence_mode=convergence_mode)

        # --- 3. RELAX ---
        rate = 0.12
        activations = relax_py(activations, stabilities, bases, rate)
        steps.append({"step": 3, "name": "RELAX", "detail": f"rate={rate}"})
        if on_step:
            on_step(steps[-1])

        # --- 4. NORMALISE ---
        activations = norm_l2_py(activations)
        steps.append({"step": 4, "name": "NORMALISE", "detail": "L2 on activation vector"})
        if on_step:
            on_step(steps[-1])

        # Persist activations before graph surgery
        self.store.bulk_set_activations([(ids[i], activations[i]) for i in range(len(ids))])

        # --- 5. PRUNE EDGES ---
        pruned = self.store.prune_edges_below(self.config.wave_prune_edge_threshold)
        steps.append({"step": 5, "name": "PRUNE EDGES", "detail": f"removed {pruned}"})
        if on_step:
            on_step(steps[-1])
        tension_accum += pruned * 0.01

        # --- 6. MERGE SIMILAR ---
        merged: list[tuple[int, int]] = []
        dim = self.config.embedding_dim
        max_merges = 24 if convergence_mode else 8
        for _ in range(max_merges):
            nodes = self._ordered_nodes()
            if len(nodes) < 2:
                break
            embs = [n.embedding or [0.0] * dim for n in nodes]
            sim = pairwise_cosine_py(embs)
            best: tuple[int, int] | None = None
            best_score = -1.0
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if sim[i][j] >= thr and sim[i][j] > best_score:
                        best_score = sim[i][j]
                        best = (i, j)
            if best is None:
                break
            i, j = best
            id_i, id_j = nodes[i].id, nodes[j].id
            keep, drop = (id_i, id_j) if id_i < id_j else (id_j, id_i)
            self.store.merge_nodes(keep, drop)
            merged.append((keep, drop))
            tension_accum += 0.08
            if convergence_mode:
                tension_accum *= self.config.convergence_tension_boost
        nodes = self._ordered_nodes()
        ids, adj = self._adjacency(nodes)
        activations = [n.activation for n in nodes]
        steps.append({"step": 6, "name": "MERGE SIMILAR", "detail": f"merged {len(merged)} pairs"})
        if on_step:
            on_step(steps[-1])
        self.store.log_wave_step("MERGE SIMILAR", f"{len(merged)} merges", tension_accum, convergence_mode=convergence_mode)

        # --- 7. SPLIT OVERACTIVATED ---
        spawned: list[int] = []
        split_threshold = self.config.wave_split_activation * (0.85 if convergence_mode else 1.0)
        for n in self.store.list_nodes():
            if n.activation > split_threshold:
                child = self.store.insert_spawned_node(
                    content=f"Shard of overactivated node {n.id}: tension relief",
                    topics=["split", "wave17"],
                    embedding=n.embedding,
                    meta={"parent": n.id, "reason": "split"},
                )
                self.store.update_activation(n.id, n.activation * 0.55)
                spawned.append(child)
        steps.append({"step": 7, "name": "SPLIT OVERACTIVATED", "detail": f"spawned {len(spawned)}"})
        if on_step:
            on_step(steps[-1])

        # --- 8. DETECT CONTRADICTIONS ---
        contra = 0
        for n in self.store.list_nodes():
            text = n.content.lower()
            if "not " in text and " is " in text:
                contra += 1
            if re.search(r"\b(yes|allow)\b.*\b(no|deny)\b", text):
                contra += 1
        steps.append({"step": 8, "name": "DETECT CONTRADICTIONS", "detail": f"hits={contra}"})
        if on_step:
            on_step(steps[-1])

        # --- 9. RESOLVE CONTRADICTIONS ---
        if contra > 0:
            for n in self.store.list_nodes():
                if n.activation > 0.4:
                    self.store.update_activation(n.id, n.activation * 0.92)
        steps.append({"step": 9, "name": "RESOLVE CONTRADICTIONS", "detail": "soft dampening"})
        if on_step:
            on_step(steps[-1])
        tension_accum += contra * 0.04

        # --- 10. TENSION DETECT & BREAK/EVOLVE ---
        acts = [n.activation for n in self.store.list_nodes()]
        mean = sum(acts) / len(acts) if acts else 0.0
        var = sum((a - mean) ** 2 for a in acts) / len(acts) if acts else 0.0
        tension = float(math.sqrt(var) + tension_accum + (0.2 if convergence_mode else 0.0))
        reply = ""
        if tension > 0.35:
            reply = chat(
                [
                    {
                        "role": "system",
                        "content": "You are TS-OS convergence. Output one short unified concept title only.",
                    },
                    {
                        "role": "user",
                        "content": "Propose a single v2 concept name merging historical TS-OS repos.",
                    },
                ],
                config=self.config,
            )
            title = reply.strip().splitlines()[0][:240] if reply else "Unified TS-OS v2"
            new_id = self.store.insert_spawned_node(
                content=title,
                topics=["v2", "convergence"],
                embedding=embed_text(title, config=self.config),
                meta={"tension": tension, "convergence": convergence_mode},
            )
            spawned.append(new_id)
        steps.append({"step": 10, "name": "TENSION EVOLVE", "detail": f"tension={tension:.3f}"})
        if on_step:
            on_step(steps[-1])

        # --- 11. INCREMENTAL SAVE ---
        self.store.rebuild_similarity_edges(threshold=0.32 if convergence_mode else 0.4)
        self.store.log_wave_step("INCREMENTAL SAVE", "checkpoint", tension, convergence_mode=convergence_mode)
        steps.append({"step": 11, "name": "INCREMENTAL SAVE", "detail": "sqlite checkpoint"})
        if on_step:
            on_step(steps[-1])

        return WaveResult(
            tension=tension,
            elected_id=elected_id,
            merged_pairs=merged,
            spawned_ids=spawned,
            steps=steps,
            contradiction_hits=contra,
        )
