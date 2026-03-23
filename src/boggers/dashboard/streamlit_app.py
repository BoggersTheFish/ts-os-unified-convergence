"""
**TS-OS Unified Convergence** — Streamlit playground (Wave 17).

**TS-OS** = Thinking System / Thinking Wave Operating System.

This UI visualizes the **UniversalLivingGraph**, runs the **Convergence Wave**
(first background-style merge of 24 historical repositories), and streams **tension**
(the scalar that drives evolution) alongside the **Wave Cycle** (11 steps).

Launch:
    streamlit run src/boggers/dashboard/streamlit_app.py
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    # The PyPI package exports the *function* `streamlit_cytoscape` (not `st_cytoscape`,
    # which is a submodule — calling it caused TypeError: 'module' object is not callable).
    from streamlit_cytoscape import EdgeStyle, NodeStyle, streamlit_cytoscape
except ImportError:  # pragma: no cover - optional until deps installed
    EdgeStyle = None  # type: ignore[misc, assignment]
    NodeStyle = None  # type: ignore[misc, assignment]
    streamlit_cytoscape = None

from boggers.core.config import DEFAULT_CONFIG
from boggers.graph.seed import HISTORICAL_REPOS, seed_historical_nodes
from boggers.graph.store import GraphStore
from boggers.llm.ollama import chat
from boggers.wave.convergence import ConvergenceWave
from boggers.wave.cycle import WaveEngine


def _default_db_path() -> Path:
    root = Path.cwd() / ".tsos"
    root.mkdir(parents=True, exist_ok=True)
    return root / "universal_living_graph.db"


def _graph_elements(
    store: GraphStore,
    *,
    edge_tension_cutoff: float = 0.45,
) -> dict[str, Any]:
    """
    Build the `elements` dict expected by streamlit-cytoscape: separate `nodes` and `edges`
    arrays (Cytoscape.js JSON), each item shaped as `{"data": {...}}`.

    High-similarity edges use `data.label == "tension"` so **EdgeStyle** can render them red
    (visual “tension” paths during Convergence).
    """
    rows = store.list_nodes()
    edge_rows = store.list_edges()
    cy_nodes: list[dict[str, Any]] = []
    for n in rows:
        display = (n.repo_id or f"node-{n.id}")[:42]
        # `label` in data is the style group key (see NodeStyle(label=...)).
        cy_nodes.append(
            {
                "data": {
                    "id": str(n.id),
                    "label": n.node_type,
                    "display_name": display,
                    "category": n.node_type,
                    "activation": n.activation,
                }
            }
        )
    cy_edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for s, t, w, et in edge_rows:
        key = (str(s), str(t), et)
        if key in seen:
            continue
        seen.add(key)
        # Split styling: strong cosine / assoc → "tension" (red); weaker → "assoc" (grey).
        edge_group = "tension" if float(w) >= edge_tension_cutoff else "assoc"
        cy_edges.append(
            {
                "data": {
                    "id": f"e-{s}-{t}-{et}",
                    "source": str(s),
                    "target": str(t),
                    "weight": w,
                    "label": edge_group,
                    "edge_type": et,
                }
            }
        )
    return {"nodes": cy_nodes, "edges": cy_edges}


def _graph_node_styles() -> list[Any]:
    """NodeStyle groups match the `label` field on each node's data (node_type)."""
    if NodeStyle is None:
        return []
    return [
        NodeStyle(label="historical", color="#00c896", caption="display_name"),
        NodeStyle(label="synthesized", color="#ff7b54", caption="display_name"),
        NodeStyle(label="concept", color="#5b8cff", caption="display_name"),
    ]


def _graph_edge_styles() -> list[Any]:
    """Red “tension” edges vs neutral links — matches `data.label` on edges."""
    if EdgeStyle is None:
        return []
    return [
        EdgeStyle(label="tension", color="#e53935", directed=True, curve_style="bezier"),
        EdgeStyle(label="assoc", color="#9aa7c7", directed=True, curve_style="bezier"),
    ]


def _events_dataframe(store: GraphStore, limit: int = 20) -> pd.DataFrame:
    """Recent wave log as a dashboard-friendly events table."""
    logs = store.recent_logs(limit)

    def _action(name: str) -> str:
        u = name.upper()
        if "MERGE" in u:
            return "MERGED"
        if "PROPAGATE" in u or "PRUNE" in u:
            return "LINKED"
        if "SAVE" in u or "INCREMENTAL" in u:
            return "SYNCED"
        if "SPLIT" in u or "TENSION" in u or "EVOLVE" in u or "SPAWN" in u:
            return "CREATED"
        return "WAVE"

    rows: list[dict[str, Any]] = []
    for r in logs:
        rows.append(
            {
                "ID": r["id"],
                "Node Name": "-",
                "Action": _action(r["step_name"]),
                "Details": f"{r['step_name']}: {r['detail']}",
                "Timestamp": r.get("created_at") or "",
            }
        )
    return pd.DataFrame(rows)


def _ensure_state() -> None:
    if "db_path" not in st.session_state:
        st.session_state.db_path = str(_default_db_path())
    if "tension_series" not in st.session_state:
        st.session_state.tension_series = []
    if "spawn_log" not in st.session_state:
        st.session_state.spawn_log = []
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []


def main() -> None:
    st.set_page_config(
        page_title="TS-OS Unified Convergence",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _ensure_state()

    st.title("TS-OS Unified Convergence — Wave 17")
    st.caption(
        "TS-OS (Thinking System / Thinking Wave Operating System) is the underlying theory — "
        "Convergence is the act of merging all past nodes into one. "
        "Local runtime: UniversalLivingGraph + Ollama (`nomic-embed-text` + chat model)."
    )

    with st.sidebar:
        st.header("Graph storage")
        st.session_state.db_path = st.text_input("SQLite path", value=st.session_state.db_path)
        st.caption(f"Embed model: `{DEFAULT_CONFIG.embed_model}` · Chat: `{DEFAULT_CONFIG.chat_model}`")
        with st.expander("Global Reference Tuning", expanded=False):
            merge_thr = st.slider(
                "Merge similarity threshold",
                min_value=0.65,
                max_value=0.95,
                value=DEFAULT_CONFIG.convergence_merge_threshold,
                step=0.01,
                help="Used on the next Convergence / autonomous wave (MERGE SIMILAR step).",
            )
            tension_boost = st.slider(
                "Convergence tension boost",
                min_value=1.0,
                max_value=1.6,
                value=DEFAULT_CONFIG.convergence_tension_boost,
                step=0.05,
                help="Amplifies tension accumulation during Convergence mode.",
            )
            edge_cutoff = st.slider(
                "Highlight tension edges (weight ≥)",
                min_value=0.2,
                max_value=0.85,
                value=0.45,
                step=0.01,
                help="Edges at or above this weight render as red “tension” paths in Cytoscape.",
            )

    store = GraphStore(st.session_state.db_path)

    tuned = replace(
        DEFAULT_CONFIG,
        convergence_merge_threshold=merge_thr,
        convergence_tension_boost=tension_boost,
    )

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        if st.button("Seed Historical Nodes", type="primary"):
            n = seed_historical_nodes(store)
            eng = WaveEngine(store, config=tuned)
            emb_n = eng.ensure_embeddings()
            added_edges = store.rebuild_similarity_edges(threshold=0.28)
            st.success(
                f"Seeded {n} new historical rows; refreshed embeddings on {emb_n} nodes; "
                f"cosine edges touched: {added_edges}."
            )
    with col_b:
        if st.button("Run Convergence Wave"):
            prog = st.progress(0.0, text="Convergence Wave — Wave Cycle steps 1–11")
            status = st.empty()
            graph_live = st.empty()
            cw = ConvergenceWave(store, config=tuned)

            def _on_step(step: dict) -> None:
                idx = int(step.get("step", 1))
                prog.progress(idx / 11.0, text=f"{step.get('name')} ({idx}/11)")
                status.markdown(f"**{step.get('name')}** — {step.get('detail')}")
                graph_live.caption(f"Live nodes: **{store.count_nodes()}** (graph refreshes after the wave)")
                time.sleep(0.04)

            result = cw.run(on_step=_on_step)
            st.session_state.tension_series.append(result.tension)
            st.session_state.spawn_log.extend([f"spawned id {i}" for i in result.spawned_ids])
            st.success(
                f"Convergence complete for this pass — tension={result.tension:.4f}, "
                f"merges={len(result.merged_pairs)}, spawns={len(result.spawned_ids)}"
            )
    with col_c:
        if st.button("Watch Autonomous Thinking"):
            eng = WaveEngine(store, config=tuned)
            for _k in range(3):
                r = eng.run_wave(convergence_mode=False)
                st.session_state.tension_series.append(r.tension)
                time.sleep(0.05)
            st.info("Completed 3 standard wave cycles.")

    st.subheader("Living graph (Cytoscape)")
    elems = _graph_elements(store, edge_tension_cutoff=edge_cutoff)
    n_nodes = len(elems["nodes"])
    layout_name = "circle" if n_nodes < 80 else "cose"
    if streamlit_cytoscape is not None:
        streamlit_cytoscape(
            elems,
            layout=layout_name,
            node_styles=_graph_node_styles(),
            edge_styles=_graph_edge_styles(),
            height=520,
            key="tsos-universal-graph",
        )
    else:
        st.warning("Install `streamlit-cytoscape` for interactive graph rendering.")
        st.json(
            {
                "nodes_preview": elems["nodes"][:20],
                "edges_preview": elems["edges"][:20],
                "truncated": n_nodes > 20,
            }
        )

    st.subheader("Tension evolution")
    c1, c2 = st.columns(2)
    with c1:
        series = st.session_state.tension_series
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=series, mode="lines+markers", name="tension", line=dict(color="#ff4ecd")))
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            title="Tension over waves",
            xaxis_title="run",
            yaxis_title="tension",
            template="plotly_dark",
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.write("**Recent events**")
        st.dataframe(_events_dataframe(store, limit=20), use_container_width=True, height=320)

    st.subheader("Chat with the converged runtime")
    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    prompt = st.chat_input("Ask TS-OS about the convergence…")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        ctx_nodes = store.list_nodes()[:12]
        context = "\n".join(n.content[:400] for n in ctx_nodes)
        reply = chat(
            [
                {
                    "role": "system",
                    "content": "You are TS-OS Unified Convergence. Explain wave reasoning briefly.",
                },
                {"role": "user", "content": f"Context from graph:\n{context}\n\nQuestion: {prompt}"},
            ]
        )
        if not reply:
            reply = (
                "Ollama chat unavailable — start `ollama serve` and pull a chat model. "
                "The graph still converges locally."
            )
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        st.rerun()

    ex1, ex2 = st.columns(2)
    with ex1:
        _payload = {
            "nodes": [asdict(n) for n in store.list_nodes()],
            "edges": store.list_edges(),
            "historical_repo_count": len(HISTORICAL_REPOS),
        }
        st.download_button(
            "Export Converged Graph (JSON)",
            data=json.dumps(_payload, indent=2, default=str),
            file_name="tsos_converged_graph.json",
            mime="application/json",
        )
    with ex2:
        hist_n = sum(1 for n in store.list_nodes() if n.node_type == "historical")
        st.metric("Number of historical nodes integrated", hist_n)
        st.caption(f"Total nodes in graph: {store.count_nodes()}")

    st.divider()
    st.markdown(
        "**Wave Cycle (11 steps):** "
        "ELECT STRONGEST → PROPAGATE → RELAX → NORMALISE → PRUNE EDGES → MERGE SIMILAR → "
        "SPLIT OVERACTIVATED → DETECT CONTRADICTIONS → RESOLVE → TENSION EVOLVE → INCREMENTAL SAVE."
    )


if __name__ == "__main__":
    main()
