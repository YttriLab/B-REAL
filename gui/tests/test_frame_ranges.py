"""Phase 3: bouts -> preprocess/inference frame ranges (pure)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from labeling_shell.bouts import Bout, LabelModel  # noqa: E402
from labeling_shell.frame_ranges import compute_labeled_frame_ranges  # noqa: E402


def _bouts(*spans):
    m = LabelModel()
    b = m.add_behavior("x")
    for s, e in spans:
        m.insert_bout_sorted(Bout(b.id, b.name, s, e))
    return m.bouts


def test_lead_in_and_merge_adjacent():
    # groom[10,20], rear[22,25], fpg=3 -> (8,20),(20,25) -> merged [8,25]
    assert compute_labeled_frame_ranges(_bouts((10, 20), (22, 25)), 3) == [[8, 25]]


def test_gap_kept_separate():
    assert compute_labeled_frame_ranges(_bouts((10, 20), (30, 35)), 3) == [[8, 20], [28, 35]]


def test_lead_in_clamped_to_zero():
    assert compute_labeled_frame_ranges(_bouts((1, 5)), 3) == [[0, 5]]


def test_fpg_one_no_lead():
    assert compute_labeled_frame_ranges(_bouts((10, 12), (14, 16)), 1) == [[10, 12], [14, 16]]


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for fn in tests:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
        except AssertionError as e:
            failures.append(fn.__name__)
            print(f"  [FAIL] {fn.__name__}: {e}")
    print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILURE(S)")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(_main())
