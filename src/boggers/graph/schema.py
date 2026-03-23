"""
Schema for the **UniversalLivingGraph**.

Each row is a *constraint cluster* in TS-OS: `content` holds natural language,
`topics` JSON lists thematic hooks, `activation` is transient energy, `stability`
resists drift, `base_strength` anchors long-term identity, `embedding` stores
the **nomic-embed-text** vector (local via Ollama), and `collapsed` marks merged
historical shards.
"""

from __future__ import annotations

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    topics TEXT,
    activation REAL NOT NULL DEFAULT 0.0,
    stability REAL NOT NULL DEFAULT 1.0,
    base_strength REAL NOT NULL DEFAULT 1.0,
    embedding TEXT,
    collapsed INTEGER NOT NULL DEFAULT 0,
    source_url TEXT,
    repo_id TEXT UNIQUE,
    node_type TEXT NOT NULL DEFAULT 'concept',
    meta TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_nodes_collapsed ON nodes(collapsed);

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    edge_type TEXT NOT NULL DEFAULT 'assoc',
    meta TEXT,
    UNIQUE(source_id, target_id, edge_type)
);

CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_tgt ON edges(target_id);

CREATE TABLE IF NOT EXISTS wave_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    step_name TEXT NOT NULL,
    detail TEXT,
    tension REAL NOT NULL DEFAULT 0.0,
    convergence_mode INTEGER NOT NULL DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);
"""
