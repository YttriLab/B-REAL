"""Behavior + bout data model — the pure-logic core of the labeling UX.

Python successor to the state/logic in `core/behaviors.js` and the bout parts of
`core/videos.js` (handleKeyAction, insertBoutSorted, findOverlappingBouts,
merge/reassign, name normalization). No UI and no persistence here: the Qt panels
render this model, and project I/O serializes it.

Data model (matching the JS contract so CSV/.rlt/localStorage interoperate):
  Behavior: {id, name, key}         -- up to 10 behaviors, keys "1".."0"
  Bout:     {behavior_id, behavior_name, start_frame, end_frame}, kept sorted
            by start_frame; created by pressing a behavior's key twice.

One label per frame: a new bout replaces whatever it overlaps (trimming or
splitting older bouts), which is how a stretch gets relabeled, and matches
training's one-hot rows. Bout changes are undoable (`undo`).
"""

from __future__ import annotations

import bisect
import copy
import re
import uuid
from dataclasses import dataclass
from typing import Optional

# Number-row keys, in the order behaviors are auto-assigned (10th is "0").
KEY_ORDER = list("1234567890")


def normalize_behavior_name(s) -> str:
    """Trim ends and collapse internal whitespace runs (mirrors JS)."""
    return re.sub(r"\s+", " ", str("" if s is None else s).strip())


def behavior_name_key(s) -> str:
    """Canonical match key: normalized + case-insensitive."""
    return normalize_behavior_name(s).lower()


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Behavior:
    id: str
    name: str
    key: str = ""


@dataclass
class Bout:
    behavior_id: str
    behavior_name: str
    start_frame: int
    end_frame: int


@dataclass
class ToggleResult:
    """Outcome of pressing a behavior key once (see LabelModel.toggle)."""
    action: str  # "started" | "ended" | "zero_length" | "no_behavior"
    behavior: Optional[Behavior] = None
    bout: Optional[Bout] = None
    start_frame: Optional[int] = None
    replaced: int = 0  # frames of other behaviors the new bout relabeled


class LabelModel:
    MAX_UNDO = 100

    def __init__(self):
        self.behaviors: list[Behavior] = []
        self.bouts: list[Bout] = []
        self._active_starts: dict[str, int] = {}  # behavior_id -> start frame
        self._undo: list[tuple] = []  # (bouts, behaviors or None) before each change

    # ----------------------------------------------------------------- undo
    def snapshot(self, with_behaviors: bool = False) -> None:
        """Record the bouts (and, for a merge, the behaviors) before a change."""
        self._undo.append((copy.deepcopy(self.bouts),
                           copy.deepcopy(self.behaviors) if with_behaviors else None))
        del self._undo[:-self.MAX_UNDO]

    def can_undo(self) -> bool:
        return bool(self._undo)

    def undo(self) -> bool:
        if not self._undo:
            return False
        bouts, behaviors = self._undo.pop()
        self.bouts = bouts
        if behaviors is not None:
            self.behaviors = behaviors
        return True

    # ---------------------------------------------------------- behaviors
    def find_behavior_by_name(self, name) -> Optional[Behavior]:
        key = behavior_name_key(name)
        return next((b for b in self.behaviors if behavior_name_key(b.name) == key), None)

    def behavior_by_id(self, behavior_id) -> Optional[Behavior]:
        return next((b for b in self.behaviors if b.id == behavior_id), None)

    def behavior_by_key(self, key) -> Optional[Behavior]:
        return next((b for b in self.behaviors if b.key == key), None)

    def add_behavior(self, name, key: Optional[str] = None) -> Behavior:
        """Add a behavior (or return the existing one with the same name)."""
        norm = normalize_behavior_name(name)
        existing = self.find_behavior_by_name(norm)
        if existing:
            return existing
        if key is None:
            key = KEY_ORDER[len(self.behaviors)] if len(self.behaviors) < len(KEY_ORDER) else ""
        b = Behavior(id=_new_id(), name=norm, key=key)
        self.behaviors.append(b)
        return b

    def rename_behavior(self, behavior_id, new_name) -> bool:
        """Rename, updating bouts. Returns False on empty name or name clash."""
        b = self.behavior_by_id(behavior_id)
        if b is None:
            return False
        norm = normalize_behavior_name(new_name)
        if not norm:
            return False
        clash = next((o for o in self.behaviors
                      if o is not b and behavior_name_key(o.name) == behavior_name_key(norm)), None)
        if clash:
            return False
        if norm != b.name:
            b.name = norm
            for bout in self.bouts:
                if bout.behavior_id == b.id:
                    bout.behavior_name = norm
        return True

    def remove_behavior(self, behavior_id) -> None:
        self.behaviors = [b for b in self.behaviors if b.id != behavior_id]
        self._active_starts.pop(behavior_id, None)

    # --------------------------------------------------------------- bouts
    def insert_bout_sorted(self, bout: Bout) -> None:
        starts = [b.start_frame for b in self.bouts]
        self.bouts.insert(bisect.bisect_right(starts, bout.start_frame), bout)

    def place_bout(self, bout: Bout) -> int:
        """Insert `bout`, replacing whatever labels it overlaps: other bouts are
        trimmed, or split around it, and a same-behavior bout it overlaps or
        touches is merged into it. Returns how many frames of *other*
        behaviors it relabeled. Not snapshotted; callers do that."""
        s, e = bout.start_frame, bout.end_frame
        kept, replaced = [], 0
        for b in self.bouts:
            if b.end_frame < s or b.start_frame > e:
                kept.append(b)
                continue
            if b.behavior_id != bout.behavior_id:
                replaced += min(b.end_frame, e) - max(b.start_frame, s) + 1
            if b.start_frame < s:
                kept.append(Bout(b.behavior_id, b.behavior_name, b.start_frame, s - 1))
            if b.end_frame > e:
                kept.append(Bout(b.behavior_id, b.behavior_name, e + 1, b.end_frame))
        merged = []
        for b in kept:
            if b.behavior_id == bout.behavior_id and b.start_frame <= e + 1 and b.end_frame >= s - 1:
                s, e = min(s, b.start_frame), max(e, b.end_frame)
            else:
                merged.append(b)
        self.bouts = sorted(merged, key=lambda b: b.start_frame)
        self.insert_bout_sorted(Bout(bout.behavior_id, bout.behavior_name, s, e))
        return replaced

    def toggle(self, key, frame: int) -> ToggleResult:
        """Press a behavior's key once: first press starts a bout, second ends it.

        (Mirrors handleKeyAction, but stores min/max so a backward-scrubbed bout
        is never inverted.)
        """
        b = self.behavior_by_key(key)
        if b is None:
            return ToggleResult("no_behavior")
        if b.id not in self._active_starts:
            self._active_starts[b.id] = frame
            return ToggleResult("started", behavior=b, start_frame=frame)
        start = self._active_starts.pop(b.id)
        if start == frame:
            return ToggleResult("zero_length", behavior=b)
        bout = Bout(b.id, b.name, min(start, frame), max(start, frame))
        self.snapshot()
        replaced = self.place_bout(bout)
        return ToggleResult("ended", behavior=b, bout=bout, replaced=replaced)

    def active_behavior_ids(self) -> set:
        return set(self._active_starts)

    def is_recording(self, behavior_id) -> bool:
        return behavior_id in self._active_starts

    def cancel_active(self, behavior_id=None) -> None:
        if behavior_id is None:
            self._active_starts.clear()
        else:
            self._active_starts.pop(behavior_id, None)

    def find_overlapping_bouts(self) -> Optional[tuple]:
        """First pair of indices (i, j) whose bouts share any frame, else None."""
        n = len(self.bouts)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.bouts[i], self.bouts[j]
                if a.start_frame < b.end_frame and b.start_frame < a.end_frame:
                    return (i, j)
        return None

    def bout_at_frame(self, frame: int) -> Optional[Bout]:
        return next((b for b in self.bouts if b.start_frame <= frame <= b.end_frame), None)

    def remove_bout(self, index: int) -> None:
        self.snapshot()
        del self.bouts[index]

    def set_bout_frames(self, index: int, start: int, end: int) -> int:
        """Move a bout's ends (it then replaces what it newly overlaps)."""
        self.snapshot()
        bout = self.bouts.pop(index)
        return self.place_bout(Bout(bout.behavior_id, bout.behavior_name,
                                    min(start, end), max(start, end)))

    def reassign_bout(self, index: int, behavior_id) -> None:
        b = self.behavior_by_id(behavior_id)
        if b is not None:
            self.snapshot()
            self.bouts[index].behavior_id = b.id
            self.bouts[index].behavior_name = b.name

    # ---------------------------------------------------- merge / dedupe
    def _reassign(self, from_ids: set, target: Behavior) -> None:
        for bout in self.bouts:
            if bout.behavior_id in from_ids:
                bout.behavior_id = target.id
                bout.behavior_name = target.name

    def merge_behaviors(self, ids, target_id) -> None:
        target = self.behavior_by_id(target_id)
        if target is None:
            return
        from_ids = {i for i in ids if i != target_id and self.behavior_by_id(i)}
        if not from_ids:
            return
        self.snapshot(with_behaviors=True)
        self._reassign(from_ids, target)
        self.behaviors = [b for b in self.behaviors if b.id not in from_ids]
        for i in from_ids:
            self._active_starts.pop(i, None)

    def merge_duplicate_behaviors(self) -> int:
        """Collapse behaviors sharing a canonical name; remap bouts. Returns count removed."""
        survivors: dict[str, Behavior] = {}
        id_map: dict[str, str] = {}
        kept: list[Behavior] = []
        for b in self.behaviors:
            key = behavior_name_key(b.name)
            survivor = survivors.get(key)
            if survivor:
                id_map[b.id] = survivor.id
            else:
                b.name = normalize_behavior_name(b.name)
                survivors[key] = b
                kept.append(b)
        if not id_map:
            self.behaviors = kept
            return 0
        self.behaviors = kept
        name_by_id = {b.id: b.name for b in kept}
        for bout in self.bouts:
            sid = id_map.get(bout.behavior_id)
            if sid:
                bout.behavior_id = sid
                bout.behavior_name = name_by_id.get(sid, bout.behavior_name)
            elif bout.behavior_id in name_by_id:
                bout.behavior_name = name_by_id[bout.behavior_id]
        return len(id_map)

    # ------------------------------------------------------- serialization
    def to_dict(self) -> dict:
        return {
            "behaviors": [vars(b) for b in self.behaviors],
            "bouts": [vars(b) for b in self.bouts],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LabelModel":
        m = cls()
        m.behaviors = [Behavior(**b) for b in (data.get("behaviors") or [])]
        m.bouts = [Bout(**b) for b in (data.get("bouts") or [])]
        return m
