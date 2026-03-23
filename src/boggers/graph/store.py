"""
SQLite persistence for the **UniversalLivingGraph**.

Nodes store JSON-serialized embeddings; edges carry associative weights used in
**Wave Cycle** step 2 (PROPAGATE: topo + cosine).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .schema import SCHEMA_SQL


@dataclass
class NodeRow:
    """In-memory mirror of a graph node."""

    id: int
    content: str
    topics: list[str]
    activation: float
    stability: float
    base_strength: float
    embedding: list[float] | None
    collapsed: bool
    source_url: str | None
    repo_id: str | None
    node_type: str
    meta: dict[str, Any]


class GraphStore:
    """Thin DAO over SQLite for TS-OS graphs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def upsert_historical(
        self,
        *,
        repo_id: str,
        url: str,
        content: str,
        topics: list[str],
        meta: dict[str, Any],
    ) -> bool:
        """Insert historical node if repo_id not present. Returns True if inserted."""
        cur = self._conn.execute("SELECT id FROM nodes WHERE repo_id = ?", (repo_id,))
        if cur.fetchone():
            return False
        self._conn.execute(
            """
            INSERT INTO nodes (content, topics, activation, stability, base_strength,
                embedding, collapsed, source_url, repo_id, node_type, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'historical', ?)
            """,
            (
                content,
                json.dumps(topics),
                0.1,
                1.0,
                1.0,
                None,
                0,
                url,
                repo_id,
                json.dumps(meta),
            ),
        )
        self._conn.commit()
        return True

    def count_nodes(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS c FROM nodes").fetchone()
        return int(row["c"]) if row else 0

    def list_nodes(self) -> list[NodeRow]:
        rows = self._conn.execute("SELECT * FROM nodes ORDER BY id").fetchall()
        return [self._row_to_node(r) for r in rows]

    def _row_to_node(self, r: sqlite3.Row) -> NodeRow:
        topics_raw = r["topics"]
        topics = json.loads(topics_raw) if topics_raw else []
        emb_raw = r["embedding"]
        emb = json.loads(emb_raw) if emb_raw else None
        meta_raw = r["meta"]
        meta = json.loads(meta_raw) if meta_raw else {}
        return NodeRow(
            id=int(r["id"]),
            content=str(r["content"]),
            topics=topics,
            activation=float(r["activation"]),
            stability=float(r["stability"]),
            base_strength=float(r["base_strength"]),
            embedding=emb,
            collapsed=bool(r["collapsed"]),
            source_url=r["source_url"],
            repo_id=r["repo_id"],
            node_type=str(r["node_type"]),
            meta=meta,
        )

    def update_node_embedding(self, node_id: int, embedding: Sequence[float]) -> None:
        self._conn.execute(
            "UPDATE nodes SET embedding = ? WHERE id = ?",
            (json.dumps(list(embedding)), node_id),
        )
        self._conn.commit()

    def update_activation(self, node_id: int, activation: float) -> None:
        self._conn.execute("UPDATE nodes SET activation = ? WHERE id = ?", (activation, node_id))
        self._conn.commit()

    def bulk_set_activations(self, pairs: Iterable[tuple[int, float]]) -> None:
        self._conn.executemany(
            "UPDATE nodes SET activation = ? WHERE id = ?",
            [(activation, nid) for (nid, activation) in pairs],
        )
        self._conn.commit()

    def bulk_set_activations_map(self, m: dict[int, float]) -> None:
        self.bulk_set_activations(m.items())

    def set_collapsed(self, node_id: int, collapsed: bool = True) -> None:
        self._conn.execute("UPDATE nodes SET collapsed = ? WHERE id = ?", (1 if collapsed else 0, node_id))
        self._conn.commit()

    def insert_edge(self, source_id: int, target_id: int, weight: float, edge_type: str = "assoc") -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO edges (source_id, target_id, weight, edge_type)
            VALUES (?, ?, ?, ?)
            """,
            (source_id, target_id, weight, edge_type),
        )
        self._conn.commit()

    def prune_edges_below(self, threshold: float) -> int:
        cur = self._conn.execute("DELETE FROM edges WHERE ABS(weight) < ?", (threshold,))
        self._conn.commit()
        return cur.rowcount

    def list_edges(self) -> list[tuple[int, int, float, str]]:
        rows = self._conn.execute("SELECT source_id, target_id, weight, edge_type FROM edges").fetchall()
        return [(int(r[0]), int(r[1]), float(r[2]), str(r[3])) for r in rows]

    def delete_edges_for_node(self, node_id: int) -> None:
        self._conn.execute(
            "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
            (node_id, node_id),
        )
        self._conn.commit()

    def delete_node(self, node_id: int) -> None:
        self.delete_edges_for_node(node_id)
        self._conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        self._conn.commit()

    def insert_spawned_node(
        self,
        *,
        content: str,
        topics: list[str],
        embedding: Sequence[float] | None,
        meta: dict[str, Any],
        node_type: str = "synthesized",
    ) -> int:
        self._conn.execute(
            """
            INSERT INTO nodes (content, topics, activation, stability, base_strength,
                embedding, collapsed, source_url, repo_id, node_type, meta)
            VALUES (?, ?, ?, ?, ?, ?, 0, NULL, NULL, ?, ?)
            """,
            (
                content,
                json.dumps(topics),
                0.5,
                0.9,
                1.0,
                json.dumps(list(embedding)) if embedding is not None else None,
                node_type,
                json.dumps(meta),
            ),
        )
        self._conn.commit()
        return int(self._conn.execute("SELECT last_insert_rowid()").fetchone()[0])

    def log_wave_step(
        self,
        step_name: str,
        detail: str,
        tension: float,
        *,
        convergence_mode: bool = False,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO wave_log (step_name, detail, tension, convergence_mode)
            VALUES (?, ?, ?, ?)
            """,
            (step_name, detail, tension, 1 if convergence_mode else 0),
        )
        self._conn.commit()

    def recent_logs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM wave_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "step_name": r["step_name"],
                    "detail": r["detail"],
                    "tension": r["tension"],
                    "convergence_mode": bool(r["convergence_mode"]),
                    "created_at": r["created_at"],
                }
            )
        return out

    def rebuild_similarity_edges(self, threshold: float = 0.35) -> int:
        """
        Connect nodes whose embedding cosine similarity exceeds `threshold`.
        Used after embeddings exist to give PROPAGATE a topo+cosine substrate.
        """
        nodes = self.list_nodes()
        embs: list[tuple[int, list[float]]] = [(n.id, e) for n in nodes if (e := n.embedding)]
        if len(embs) < 2:
            return 0
        import math

        def norm(v: list[float]) -> list[float]:
            s = math.sqrt(sum(x * x for x in v)) or 1.0
            return [x / s for x in v]

        normed = [(nid, norm(v)) for nid, v in embs]
        added = 0
        for i, (ia, va) in enumerate(normed):
            for ib, vb in normed[i + 1 :]:
                sim = sum(x * y for x, y in zip(va, vb))
                if sim >= threshold:
                    self.insert_edge(ia, ib, sim, "cosine")
                    self.insert_edge(ib, ia, sim, "cosine")
                    added += 2
        return added

    def merge_nodes(self, keep_id: int, drop_id: int) -> None:
        """Merge drop_id into keep_id: concatenate content, rewire edges, delete drop."""
        drop = self._conn.execute("SELECT content FROM nodes WHERE id = ?", (drop_id,)).fetchone()
        keep = self._conn.execute("SELECT content FROM nodes WHERE id = ?", (keep_id,)).fetchone()
        if not drop or not keep:
            return
        new_content = f"{keep['content']}\n\n--- merged from node {drop_id} ---\n{drop['content']}"
        self._conn.execute("UPDATE nodes SET content = ?, collapsed = 1 WHERE id = ?", (new_content, keep_id))
        # Rewire edges that pointed to/from drop onto keep, then remove drop.
        self._conn.execute("UPDATE edges SET source_id = ? WHERE source_id = ?", (keep_id, drop_id))
        self._conn.execute("UPDATE edges SET target_id = ? WHERE target_id = ?", (keep_id, drop_id))
        self._conn.execute("DELETE FROM edges WHERE source_id = target_id")
        self._conn.execute("DELETE FROM nodes WHERE id = ?", (drop_id,))
        self._conn.commit()
