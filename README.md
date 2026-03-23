<image-card alt="Python" src="https://img.shields.io/badge/Python-3.10+-blue" ></image-card>
<image-card alt="Rust" src="https://img.shields.io/badge/Rust-PyO3-orange" ></image-card>
<image-card alt="Streamlit" src="https://img.shields.io/badge/Streamlit-Demo-green" ></image-card>


# TS-OS Unified Convergence

**This is the Convergence Wave — every past repo has now evolved into one.**

**TS-OS** (Thinking System / Thinking Wave Operating System) treats cognition as waves over a **UniversalLivingGraph** (SQLite + JSON + `nomic-embed-text` embeddings). **Tension** is the scalar that drives evolution. The **Wave Cycle** is the exact 11-step autonomous loop that reshapes the graph.

Wave 17 imports twenty-four historical GitHub repositories as seed nodes, runs a **Convergence Wave** (aggressive cross-repo merge + tension amplification), and exposes a Streamlit playground for live visualization.

## Requirements

- Python 3.10+
- [Rust](https://rustup.rs/) (for the `rust_wave` PyO3 extension — optional but default-on)
- [Ollama](https://ollama.com/) running locally with:
  - `nomic-embed-text` for embeddings
  - A chat model such as `llama3.2` (override via `TSOS_CHAT_MODEL`)

## Install

Use a virtual environment so **maturin** can bind the `rust_wave` extension to that interpreter:

```powershell
cd ts-os-unified-convergence
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip maturin pytest pytest-cov ruff
maturin develop --release
```

`maturin develop` installs Python dependencies from `pyproject.toml`, compiles `rust_wave`, and registers the `boggers` + `rust_wave` packages in editable mode.

Without Rust, export `TSOS_USE_RUST=0` and install dependencies with `pip install numpy httpx streamlit streamlit-cytoscape pandas plotly httpx pytest` (the NumPy wave path is used automatically).

## One-command dashboard

```bash
streamlit run src/boggers/dashboard/streamlit_app.py
```

Use **Load Historical Nodes** to seed all 24 repositories, then **Run Convergence Wave** to watch the Wave Cycle merge similar concepts in real time (with Cytoscape + tension chart).

## Tests & completion banner

```bash
python -m pytest
python scripts/print_convergence.py
```

Expected final line:

`Convergence complete. All 24 historical nodes merged. Wave 17 is alive.`

## Project layout

- `src/boggers/` — graph store, embeddings, LLM, wave engine, Streamlit UI
- `rust_wave/` — PyO3 kernels (normalize, cosine, propagate, relax)
- `MANIFESTO.md` — Wave 17 narrative, Mermaid diagram, comparisons, roadmap, PDF export notes

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama HTTP API |
| `TSOS_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `TSOS_CHAT_MODEL` | `llama3.2` | Chat model for tension evolution |
| `TSOS_USE_RUST` | `1` | Enable Rust hot path when built |
