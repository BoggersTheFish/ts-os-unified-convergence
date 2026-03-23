"""Manifesto file presence."""

from __future__ import annotations

from pathlib import Path


def test_manifesto_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    m = root / "MANIFESTO.md"
    assert m.exists()
    text = m.read_text(encoding="utf-8")
    assert "Wave 17" in text
    assert "mermaid" in text.lower()
