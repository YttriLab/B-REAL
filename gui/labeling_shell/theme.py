"""Dark QSS theme, applied once at the QApplication level.

Standard widgets (buttons, inputs, tabs, tables, menus, ...) pick up the theme
automatically. A handful of custom-painted widgets (VideoView, UncertaintyTimeline)
intentionally ignore it — their colors are semantic (black letterbox, green->red
uncertainty heatmap), not decorative.
"""

from __future__ import annotations

from pathlib import Path

_QSS_PATH = Path(__file__).with_suffix(".qss")


def apply_theme(app) -> None:
    app.setStyleSheet(_QSS_PATH.read_text(encoding="utf-8"))


def mark_accent(button) -> None:
    """Tag a QPushButton as a primary call-to-action (accent-colored)."""
    button.setProperty("accent", "true")


def mark_ghost(button) -> None:
    """Tag a QPushButton as a low-emphasis action (flat, muted border)."""
    button.setProperty("ghost", "true")


def mark_compact(button) -> None:
    """Tag a QPushButton as tightly-padded (for cramped spots like table cells)."""
    button.setProperty("compact", "true")
