"""Wave engine tests."""

from __future__ import annotations

import pytest

from boggers.graph.seed import seed_historical_nodes
from boggers.wave.convergence import ConvergenceWave
from boggers.wave.cycle import WaveEngine, WaveResult


def test_wave_engine_runs_on_empty(tmp_store) -> None:
    eng = WaveEngine(tmp_store)
    r = eng.run_wave()
    assert isinstance(r, WaveResult)
    assert r.tension >= 0.0


def test_wave_engine_with_seeded_nodes(tmp_store) -> None:
    seed_historical_nodes(tmp_store)
    eng = WaveEngine(tmp_store)
    r = eng.run_wave(convergence_mode=False)
    assert r.elected_id is not None


def test_convergence_mode_runs(tmp_store) -> None:
    seed_historical_nodes(tmp_store)
    eng = WaveEngine(tmp_store)
    r = eng.run_wave(convergence_mode=True)
    assert len(r.steps) == 11


def test_convergence_wave_facade(tmp_store) -> None:
    seed_historical_nodes(tmp_store)
    cw = ConvergenceWave(tmp_store)
    r = cw.run()
    assert r.tension >= 0.0


def test_on_step_callback(tmp_store) -> None:
    seed_historical_nodes(tmp_store)
    seen: list[dict] = []
    eng = WaveEngine(tmp_store)

    def _cb(step: dict) -> None:
        seen.append(step)

    eng.run_wave(on_step=_cb)
    assert len(seen) == 11


@pytest.mark.parametrize("mode", [True, False])
def test_convergence_flag_variants(tmp_store, mode: bool) -> None:
    seed_historical_nodes(tmp_store)
    eng = WaveEngine(tmp_store)
    r = eng.run_wave(convergence_mode=mode)
    assert isinstance(r, WaveResult)
