"""Phase 2: behavior/bout model logic.

    .venv-gui/Scripts/python.exe gui/tests/test_bouts.py
    pytest gui/tests/test_bouts.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from labeling_shell.bouts import (  # noqa: E402
    Bout, LabelModel, behavior_name_key, normalize_behavior_name,
)


def test_add_behavior_assigns_keys_and_dedupes():
    m = LabelModel()
    a = m.add_behavior("groom")
    b = m.add_behavior("rear")
    assert a.key == "1" and b.key == "2"
    # same (normalized, case-insensitive) name returns the existing behavior
    again = m.add_behavior("  GROOM ")
    assert again is a
    assert len(m.behaviors) == 2


def test_name_normalization():
    assert normalize_behavior_name("  face   groom ") == "face groom"
    assert behavior_name_key("Face  Groom") == "face groom"


def test_toggle_start_then_end_creates_bout():
    m = LabelModel()
    m.add_behavior("groom")  # key "1"
    assert m.toggle("1", 10).action == "started"
    assert m.is_recording(m.behaviors[0].id)
    res = m.toggle("1", 25)
    assert res.action == "ended"
    assert (res.bout.start_frame, res.bout.end_frame) == (10, 25)
    assert not m.is_recording(m.behaviors[0].id)
    assert len(m.bouts) == 1


def test_toggle_zero_length_and_unknown_key():
    m = LabelModel()
    m.add_behavior("groom")
    m.toggle("1", 5)
    assert m.toggle("1", 5).action == "zero_length"
    assert not m.bouts
    assert m.toggle("9", 5).action == "no_behavior"


def test_toggle_backward_is_not_inverted():
    m = LabelModel()
    m.add_behavior("groom")
    m.toggle("1", 40)
    res = m.toggle("1", 10)  # ended earlier than started
    assert (res.bout.start_frame, res.bout.end_frame) == (10, 40)


def test_insert_bout_sorted():
    m = LabelModel()
    b = m.add_behavior("x")
    for sf in (30, 10, 50, 20):
        m.insert_bout_sorted(Bout(b.id, b.name, sf, sf + 5))
    assert [x.start_frame for x in m.bouts] == [10, 20, 30, 50]


def test_find_overlapping_bouts():
    m = LabelModel()
    b = m.add_behavior("x")
    m.insert_bout_sorted(Bout(b.id, b.name, 0, 10))
    m.insert_bout_sorted(Bout(b.id, b.name, 20, 30))
    assert m.find_overlapping_bouts() is None
    m.insert_bout_sorted(Bout(b.id, b.name, 5, 25))  # overlaps both
    assert m.find_overlapping_bouts() is not None


def test_rename_updates_bouts_and_rejects_clash():
    m = LabelModel()
    a = m.add_behavior("groom")
    m.add_behavior("rear")
    m.insert_bout_sorted(Bout(a.id, a.name, 0, 5))
    assert m.rename_behavior(a.id, "grooming") is True
    assert m.bouts[0].behavior_name == "grooming"
    assert m.rename_behavior(a.id, "rear") is False  # clashes with existing
    assert m.rename_behavior(a.id, "   ") is False   # empty


def test_merge_behaviors_reassigns_and_removes():
    m = LabelModel()
    a = m.add_behavior("groom")
    b = m.add_behavior("grooming")
    m.insert_bout_sorted(Bout(b.id, b.name, 0, 5))
    m.merge_behaviors([a.id, b.id], a.id)
    assert len(m.behaviors) == 1 and m.behaviors[0] is a
    assert all(bt.behavior_id == a.id and bt.behavior_name == "groom" for bt in m.bouts)


def test_merge_duplicate_behaviors():
    m = LabelModel()
    a = m.add_behavior("groom")
    # force a duplicate with a different case/spacing (bypass add_behavior dedupe)
    from labeling_shell.bouts import Behavior, _new_id
    dup = Behavior(id=_new_id(), name="GROOM", key="2")
    m.behaviors.append(dup)
    m.insert_bout_sorted(Bout(dup.id, dup.name, 0, 5))
    removed = m.merge_duplicate_behaviors()
    assert removed == 1
    assert len(m.behaviors) == 1
    assert m.bouts[0].behavior_id == a.id and m.bouts[0].behavior_name == "groom"


def test_bout_at_frame_and_serialization():
    m = LabelModel()
    b = m.add_behavior("x")
    m.insert_bout_sorted(Bout(b.id, b.name, 10, 20))
    assert m.bout_at_frame(15) is not None
    assert m.bout_at_frame(25) is None
    round_trip = LabelModel.from_dict(m.to_dict())
    assert round_trip.to_dict() == m.to_dict()


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
