"""
Seed data: every prior BoggersTheFish repository becomes one **historical node**
in the **UniversalLivingGraph**. The Convergence Wave then treats these as
merge candidates and evolves unified v2 concepts.

**TS-OS** = Thinking System / Thinking Wave Operating System.
"""

from __future__ import annotations

from typing import Any

# One entry per repository: (repo_id, url, one_sentence_contribution)
HISTORICAL_REPOS: list[tuple[str, str, str]] = [
    (
        "BoggersTheAI",
        "https://github.com/BoggersTheFish/BoggersTheAI",
        "Flagship live TS-OS runtime that demonstrates the full wave-autonomous thinking loop.",
    ),
    (
        "BoggersTheAI-Dev",
        "https://github.com/BoggersTheFish/BoggersTheAI-Dev",
        "Development branch of the flagship runtime where experimental wave policies land first.",
    ),
    (
        "TS-Core",
        "https://github.com/BoggersTheFish/TS-Core",
        "Canonical core graph engine shared across TS-OS deployments for storage and traversal.",
    ),
    (
        "GOAT-TS",
        "https://github.com/BoggersTheFish/GOAT-TS",
        "Foundational research and knowledge-graph substrate pinned as semantic ground truth.",
    ),
    (
        "BoggersTheMind",
        "https://github.com/BoggersTheFish/BoggersTheMind",
        "Personal cognitive TUI assistant exploring human-in-the-loop steering of thought waves.",
    ),
    (
        "BLM",
        "https://github.com/BoggersTheFish/BLM",
        "WaveTS-LLM experiments introducing tension terms directly into the loss landscape.",
    ),
    (
        "BoggersTheCIG",
        "https://github.com/BoggersTheFish/BoggersTheCIG",
        "Cognitive Intelligence Graph v1 encoding context as a living relational structure.",
    ),
    (
        "BoggersTheCIG_v2",
        "https://github.com/BoggersTheFish/BoggersTheCIG_v2",
        "CIG v2 prototypes pushing richer node typing and dynamic edge reweighting.",
    ),
    (
        "CIG-APP-V1",
        "https://github.com/BoggersTheFish/CIG-APP-V1",
        "Contextual Information Generator application v1 packaging CIG for interactive use.",
    ),
    (
        "CIG-APP-V2",
        "https://github.com/BoggersTheFish/CIG-APP-V2",
        "CIG app v2 refining UX flows and tighter coupling to retrieval-augmented reasoning.",
    ),
    (
        "BoggersTheOS-Alpha",
        "https://github.com/BoggersTheFish/BoggersTheOS-Alpha",
        "Rust OS layer alpha exploring low-level scheduling metaphors for wave timing.",
    ),
    (
        "BoggersTheOS-Beta",
        "https://github.com/BoggersTheFish/BoggersTheOS-Beta",
        "Rust OS layer beta stabilizing syscall surfaces for future TS-OS kernel experiments.",
    ),
    (
        "GOAT-OS",
        "https://github.com/BoggersTheFish/GOAT-OS",
        "C-language OS skeleton tracing a minimal boot path for embedded cognition demos.",
    ),
    (
        "BoggersThePulse",
        "https://github.com/BoggersTheFish/BoggersThePulse",
        "Pulse-based timing experiments aligning activation rhythms with external telemetry.",
    ),
    (
        "BoggersTheEngine",
        "https://github.com/BoggersTheFish/BoggersTheEngine",
        "Game-engine-style abstractions for scene graphs that mirror cognitive subgraphs.",
    ),
    (
        "BAGI",
        "https://github.com/BoggersTheFish/BAGI",
        "Additional agent intelligence experiments probing multi-agent tension and coordination.",
    ),
    (
        "BOG-TS",
        "https://github.com/BoggersTheFish/BOG-TS",
        "Proof-of-concept linking layer bridging external tools into TS-OS thought streams.",
    ),
    (
        "boggersthefish-site",
        "https://github.com/BoggersTheFish/boggersthefish-site",
        "Official Next.js/TypeScript website narrating the TS-OS story for the public web.",
    ),
    (
        "GOAT-TS-LITE",
        "https://github.com/BoggersTheFish/GOAT-TS-LITE",
        "Lite GOAT-TS distribution optimized for constrained devices and quick experiments.",
    ),
    (
        "GOAT-TS-SUPERLITE",
        "https://github.com/BoggersTheFish/GOAT-TS-SUPERLITE",
        "Superlite GOAT-TS variant stripping features to bare retrieval and logging.",
    ),
    (
        "GOAT-TS-DEVELOPMENT",
        "https://github.com/BoggersTheFish/GOAT-TS-DEVELOPMENT",
        "Development branch of GOAT-TS integrating bleeding-edge graph operations.",
    ),
    (
        "GOAT-SIMPLE",
        "https://github.com/BoggersTheFish/GOAT-SIMPLE",
        "Simple GOAT prototype validating the smallest viable knowledge-graph loop.",
    ),
    (
        "GOAT-PUBLIC_TEST",
        "https://github.com/BoggersTheFish/GOAT-PUBLIC_TEST",
        "Public test repository used for CI smoke tests and reproducible benchmarks.",
    ),
    (
        "hehe",
        "https://github.com/BoggersTheFish/hehe",
        "Rust experimentation playground for unsafe ideas that might graduate into TS-Core.",
    ),
]


def seed_historical_nodes(store: Any) -> int:
    """
    Insert all 24 repositories as `node_type='historical'`.

    Returns the number of rows inserted (skips duplicates by repo_id).
    """
    inserted = 0
    for repo_id, url, sentence in HISTORICAL_REPOS:
        content = f"{sentence} Source: {url}"
        topics = ["historical", "wave17", repo_id.lower()]
        meta = {"wave": 17, "role": "seed_node"}
        if store.upsert_historical(repo_id=repo_id, url=url, content=content, topics=topics, meta=meta):
            inserted += 1
    return inserted
