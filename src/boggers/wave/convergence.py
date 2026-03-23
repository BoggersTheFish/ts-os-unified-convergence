"""
**Convergence Wave** — special mode for Wave 17.

The first background run should set `convergence_mode=True` so duplicate
concepts across the 24 historical repositories merge aggressively, tensions
inflate for visualization, and unified v2 nodes appear.

**TS-OS** = Thinking System / Thinking Wave Operating System.
**Tension** = scalar that drives evolution.
"""

from __future__ import annotations

from typing import Any, Callable

from boggers.core.config import TSOSConfig
from boggers.graph.store import GraphStore
from boggers.wave.cycle import WaveEngine, WaveResult


class ConvergenceWave:
    """Thin facade around `WaveEngine` with Convergence presets."""

    def __init__(self, store: GraphStore, config: TSOSConfig | None = None) -> None:
        self.engine = WaveEngine(store, config=config)

    def run(
        self,
        *,
        on_step: Callable[[dict[str, Any]], None] | None = None,
    ) -> WaveResult:
        """Run a single Convergence Wave (aggressive merge + boosted tension)."""
        return self.engine.run_wave(convergence_mode=True, on_step=on_step)
