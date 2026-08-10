"""qt-material dark theme, applied once at the QApplication level.

Swapped in for the earlier hand-rolled QSS (theme.qss, now removed) — qt-material
gives a more polished/animated Material Design look (raised buttons, colored
underlines on focused inputs, etc.) for one function call, at the cost of not
controlling every rule by hand.

Standard widgets pick up the theme automatically. A handful of custom-painted
widgets (VideoView, UncertaintyTimeline) intentionally ignore it — their colors
are semantic (black letterbox, green->red uncertainty heatmap), not decorative.

qt-material doesn't have a built-in notion of "this button is the primary call
to action" (every QPushButton gets the same outlined-accent look), so the
mark_accent/mark_compact QSS below is layered on top of its stylesheet using the
QTMATERIAL_* theme colors it exports via os.environ after apply_stylesheet runs.
"""

from __future__ import annotations

import os

import qt_material

THEME_NAME = "dark_blue.xml"


def apply_theme(app) -> None:
    qt_material.apply_stylesheet(app, theme=THEME_NAME)
    app.setStyleSheet(app.styleSheet() + _extra_qss())


def _extra_qss() -> str:
    primary = os.environ.get("QTMATERIAL_PRIMARYCOLOR", "#448aff")
    primary_light = os.environ.get("QTMATERIAL_PRIMARYLIGHTCOLOR", primary)
    primary_text = os.environ.get("QTMATERIAL_PRIMARYTEXTCOLOR", "#ffffff")
    secondary_dark = os.environ.get("QTMATERIAL_SECONDARYDARKCOLOR", "#31363b")
    return f"""
QPushButton[accent="true"] {{
    background-color: {primary};
    color: {primary_text};
    border: 2px solid {primary};
    font-weight: 600;
}}
QPushButton[accent="true"]:hover {{
    background-color: {primary_light};
    border-color: {primary_light};
}}
QPushButton[accent="true"]:disabled {{
    background-color: {secondary_dark};
    border-color: {secondary_dark};
    color: rgba(255, 255, 255, 0.3);
}}
QPushButton[ghost="true"] {{
    background: transparent;
    border: 1px solid {secondary_dark};
}}
QPushButton[compact="true"] {{
    height: 24px;
    padding: 0px 12px;
    min-width: 64px;
}}
/* qt-material's QListView::item rules collapse an embedded QLineEdit's
 * sizeHint to 0 height (reproduced in isolation: identical widget renders
 * fine standalone or in a plain QHBoxLayout, only collapses once placed via
 * QListWidget.setItemWidget). BehaviorsPanel rows use exactly that pattern
 * for the rename field, so without this override the name field is
 * invisible. Scoped to QListWidget descendants only. */
QListWidget QLineEdit {{
    height: 28px;
    min-height: 20px;
}}
/* Default header padding (0 24px each side) is sized for spacious Material
 * pages, not a dense desktop table — it was squeezing the "Behavior" column
 * header down to a clipped sliver once the actions column widened. */
QHeaderView::section {{
    padding: 0 8px;
}}
"""


def mark_accent(button) -> None:
    """Tag a QPushButton as a primary call-to-action (filled accent)."""
    button.setProperty("accent", "true")


def mark_ghost(button) -> None:
    """Tag a QPushButton as a low-emphasis action (flat, muted border)."""
    button.setProperty("ghost", "true")


def mark_compact(button) -> None:
    """Tag a QPushButton as tightly-padded (for cramped spots like table cells)."""
    button.setProperty("compact", "true")
