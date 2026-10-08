"""Active-learning state + label-space mapping (pure logic).

Python successor to core/active-learning.js + core/inference.js logic, notably
`mapLabelToModelClass`. The key correctness point (the binarized false-positive
fix): a human behavior label lives in the *labeling* space, while the model
predicts in its own (possibly binarized/unioned) *class* space. To decide whether
a prediction is "wrong" for a labeled frame we must map the human label into the
model's space; when that mapping is unknowable we return None and refuse to flag
a mismatch (better silent than a false alarm).

Inference results contract (from backend/services/inference_service.py):
    {"class_names": [...], "frames": [{"index", "predicted_class", "uncertainty",
     "predictions"}]}
`predicted_class` may be a class name or an index; `predictions` is the per-class
probability vector.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def map_label_to_model_class(human_label, class_names, *, binarize=None,
                             label_union=None) -> Optional[str]:
    """Map a human behavior name into the model's class space.

    Returns the model class name, or None if the mapping is unknowable (so callers
    must not treat the frame as misclassified).
    """
    if not human_label or not class_names:
        return None

    if label_union:  # {union_class_name: [member behavior names]}
        for union_name, members in label_union.items():
            if human_label == union_name or human_label in (members or []):
                return union_name if union_name in class_names else None

    if binarize:
        # Binary model: classes are ['other', <positive>] (positive may be a union).
        other = next((c for c in class_names if c.lower() == "other"), None)
        if human_label == binarize:
            return binarize if binarize in class_names else None
        return other  # any non-positive label is the negative class

    if human_label in class_names:
        return human_label
    return None


class AlState:
    """Per-video inference results, indexed by native frame."""

    def __init__(self):
        self.class_names: list[str] = []
        self.by_index: dict[int, dict] = {}
        self.binarize: Optional[str] = None
        self.label_union: Optional[dict] = None

    def load_results(self, results: dict, *, binarize=None, label_union=None) -> None:
        self.class_names = list(results.get("class_names") or [])
        self.by_index = {int(f["index"]): f for f in (results.get("frames") or [])}
        self.binarize = binarize
        self.label_union = label_union
        self._arrays = None
        self._runs = None

    def arrays(self):
        """Per-frame arrays over 0..max index, built once: (uncertainty float32,
        NaN where no prediction; predicted class index int32, -1 where none).
        What the timeline draws from — a long video is ~half a million frames."""
        if getattr(self, "_arrays", None) is None:
            n = (max(self.by_index) + 1) if self.by_index else 0
            unc = np.full(n, np.nan, np.float32)
            cls = np.full(n, -1, np.int32)
            index_of = {c: i for i, c in enumerate(self.class_names)}
            for i, f in self.by_index.items():
                u = f.get("uncertainty")
                if u is not None:
                    unc[i] = u
                pc = f.get("predicted_class")
                cls[i] = pc if isinstance(pc, int) else index_of.get(pc, -1)
            self._arrays = (unc, cls)
        return self._arrays

    def predicted_runs(self) -> list[tuple[int, int, int]]:
        """Contiguous runs of one predicted class: [(start, end, class_index)]."""
        if getattr(self, "_runs", None) is None:
            _unc, cls = self.arrays()
            runs = []
            if len(cls):
                edges = np.flatnonzero(np.diff(cls)) + 1
                starts = np.concatenate(([0], edges))
                ends = np.concatenate((edges - 1, [len(cls) - 1]))
                runs = [(int(a), int(b), int(cls[a])) for a, b in zip(starts, ends) if cls[a] >= 0]
            self._runs = runs
        return self._runs

    def frame_info(self, idx: int) -> Optional[dict]:
        return self.by_index.get(idx)

    def predicted_name(self, idx: int) -> Optional[str]:
        fi = self.frame_info(idx)
        if not fi:
            return None
        pc = fi.get("predicted_class")
        if isinstance(pc, int) and 0 <= pc < len(self.class_names):
            return self.class_names[pc]
        return pc

    def confidence(self, idx: int) -> Optional[float]:
        fi = self.frame_info(idx)
        preds = fi.get("predictions") if fi else None
        if isinstance(preds, (list, tuple)) and preds:
            return float(max(preds))
        return None

    def uncertainty(self, idx: int) -> Optional[float]:
        fi = self.frame_info(idx)
        return fi.get("uncertainty") if fi else None

    def is_misclassified(self, idx: int, human_label) -> bool:
        fi = self.frame_info(idx)
        if not fi or not human_label:
            return False
        mapped = map_label_to_model_class(
            human_label, self.class_names,
            binarize=self.binarize, label_union=self.label_union)
        if mapped is None:  # unknowable mapping -> never flag (the fix)
            return False
        return self.predicted_name(idx) != mapped

    def uncertain_frames(self, threshold: float) -> list[int]:
        return sorted(i for i, fi in self.by_index.items()
                      if (fi.get("uncertainty") or 0.0) >= threshold)

    def next_uncertain(self, after: int, threshold: float) -> Optional[int]:
        for i in sorted(self.by_index):
            if i > after and (self.by_index[i].get("uncertainty") or 0.0) >= threshold:
                return i
        return None

    def misclassified_in_bouts(self, bouts) -> set:
        """misclassified_frames for a list of bouts, visiting only labeled frames
        (an unlabeled frame can't disagree with a label) — ~1000x fewer than all
        predicted frames on a long video. With overlapping bouts the earlier one
        in the list wins, as bout_at_frame does."""
        out, seen = set(), set()
        for b in bouts:
            mapped = map_label_to_model_class(
                b.behavior_name, self.class_names,
                binarize=self.binarize, label_union=self.label_union)
            for i in range(b.start_frame, b.end_frame + 1):
                if i in seen:
                    continue
                seen.add(i)
                if mapped is not None and i in self.by_index and self.predicted_name(i) != mapped:
                    out.add(i)
        return out

    def misclassified_frames(self, label_at) -> set:
        """Set of frames whose prediction disagrees with the mapped human label.

        `label_at(idx)` returns the human behavior name at that frame (or None).
        """
        out = set()
        for i in self.by_index:
            if self.is_misclassified(i, label_at(i)):
                out.add(i)
        return out


def hud_info(al_state: Optional[AlState], frame: int, human_label) -> dict:
    """Readout for the on-video HUD: predicted class, confidence, label, mismatch."""
    if al_state is None:
        return {"predicted": None, "confidence": None, "uncertainty": None,
                "label": human_label, "mismatch": False}
    return {
        "predicted": al_state.predicted_name(frame),
        "confidence": al_state.confidence(frame),
        "uncertainty": al_state.uncertainty(frame),
        "label": human_label,
        "mismatch": al_state.is_misclassified(frame, human_label),
    }
