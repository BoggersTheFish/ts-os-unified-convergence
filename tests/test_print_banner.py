"""Completion banner script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_print_script_outputs_banner() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "print_convergence.py"
    out = subprocess.check_output([sys.executable, str(script)], text=True)
    assert "Convergence complete" in out
    assert "24" in out
    assert "Wave 17" in out
