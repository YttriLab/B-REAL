"""One color per behavior, shared by everything that colors by behavior.

Behaviors take palette colors in their list (key) order, so a behavior keeps
its color across videos. A behavior or model class named "other" is the
neutral background gray. Model classes are matched to behaviors by name
(normalized); a class with no matching behavior takes the next free color.
"""

from __future__ import annotations

from PySide6 import QtGui

from .bouts import behavior_name_key

PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
           "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#86bcb6"]
OTHER = "#5f6368"


def behavior_colors(behavior_names, class_names=()) -> dict[str, QtGui.QColor]:
    """{name: color} for the behaviors and any model classes, keyed by the
    names as given."""
    out, by_key, k = {}, {}, 0
    for name in list(behavior_names) + list(class_names):
        key = behavior_name_key(name)
        if key in by_key:
            out[name] = by_key[key]
            continue
        if key == "other":
            color = QtGui.QColor(OTHER)
        else:
            color = QtGui.QColor(PALETTE[k % len(PALETTE)])
            k += 1
        by_key[key] = out[name] = color
    return out
