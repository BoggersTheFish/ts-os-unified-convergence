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
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    # The PyPI package exports the *function* `streamlit_cytoscape` (not `st_cytoscape`,
    # which is a submodule — calling it caused TypeError: 'module' object is not callable).
    from streamlit_cytoscape import NodeStyle, streamlit_cytoscape
except ImportError:  # pragma: no cover - optional until deps installed
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


def _graph_elements(store: GraphStore) -> dict[str, Any]:
    """
    Build the `elements` dict expected by streamlit-cytoscape: separate `nodes` and `edges`
    arrays (Cytoscape.js JSON), each item shaped as `{"data": {...}}`.
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
        cy_edges.append(
            {
                "data": {
                    "id": f"e-{s}-{t}-{et}",
                    "source": str(s),
                    "target": str(t),
                    "weight": w,
                    "label": et,
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
        "TS-OS = Thinking System / Thinking Wave Operating System — "
        "UniversalLivingGraph + Convergence Wave + local Ollama (nomic-embed-text)."
    )

    with st.sidebar:
        st.header("Graph storage")
        st.session_state.db_path = st.text_input("SQLite path", value=st.session_state.db_path)
        st.caption(f"Configured embed model: `{DEFAULT_CONFIG.embed_model}`")
        st.caption(f"Chat model: `{DEFAULT_CONFIG.chat_model}`")

    store = GraphStore(st.session_state.db_path)

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        if st.button("Load Historical Nodes", type="primary"):
            n = seed_historical_nodes(store)
            eng = WaveEngine(store)
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
            cw = ConvergenceWave(store)

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
            eng = WaveEngine(store)
            for k in range(3):
                r = eng.run_wave(convergence_mode=False)
                st.session_state.tension_series.append(r.tension)
                time.sleep(0.05)
            st.info("Completed 3 standard wave cycles.")

    st.subheader("Living graph (Cytoscape)")
    elems = _graph_elements(store)
    n_nodes = len(elems["nodes"])
    layout_name = "circle" if n_nodes < 80 else "cose"
    if streamlit_cytoscape is not None:
        streamlit_cytoscape(
            elems,
            layout=layout_name,
            node_styles=_graph_node_styles(),
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

    st.subheader("Tension + evolution")
    c1, c2 = st.columns(2)
    with c1:
        series = st.session_state.tension_series
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=series, mode="lines+markers", name="tension", line=dict(color="#ff4ecd")))
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            title="Tension over wave runs",
            xaxis_title="run",
            yaxis_title="tension",
            template="plotly_dark",
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.write("**Recent wave log**")
        logs = store.recent_logs(12)
        st.dataframe(pd.DataFrame(logs), use_container_width=True, height=320)

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
        st.metric("Nodes in UniversalLivingGraph", store.count_nodes())

    st.divider()
    st.markdown(
        "**Wave Cycle (11 steps):** "
        "ELECT STRONGEST → PROPAGATE → RELAX → NORMALISE → PRUNE EDGES → MERGE SIMILAR → "
        "SPLIT OVERACTIVATED → DETECT CONTRADICTIONS → RESOLVE → TENSION EVOLVE → INCREMENTAL SAVE."
    )


if __name__ == "__main__":
    main()
