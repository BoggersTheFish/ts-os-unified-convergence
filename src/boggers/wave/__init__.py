"""
**Wave Cycle** — the exact 11-step autonomous TS-OS loop:

1. ELECT STRONGEST
2. PROPAGATE (topo + cosine)
3. RELAX
4. NORMALISE
5. PRUNE EDGES
6. MERGE SIMILAR
7. SPLIT OVERACTIVATED
8. DETECT CONTRADICTIONS
9. RESOLVE CONTRADICTIONS
10. TENSION DETECT & BREAK/EVOLVE (spawn via LLM)
11. INCREMENTAL SAVE

**Tension** = scalar that drives evolution.
"""

from .cycle import WaveEngine
from .convergence import ConvergenceWave

__all__ = ["WaveEngine", "ConvergenceWave"]
