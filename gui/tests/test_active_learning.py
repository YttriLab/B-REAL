"""Phase 4: active-learning logic (pure)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from labeling_shell.active_learning import (  # noqa: E402
    AlState, hud_info, map_label_to_model_class,
)

RESULTS = {
    "class_names": ["other", "groom"],
    "frames": [
        {"index": 10, "predicted_class": "groom", "uncertainty": 0.9, "predictions": [0.1, 0.9]},
        {"index": 11, "predicted_class": 1, "uncertainty": 0.2, "predictions": [0.3, 0.7]},
        {"index": 12, "predicted_class": "other", "uncertainty": 0.6, "predictions": [0.8, 0.2]},
    ],
}


def test_map_exact_and_unknowable():
    assert map_label_to_model_class("groom", ["other", "groom"]) == "groom"
    # not in classes, no binarize/union -> unknowable
    assert map_label_to_model_class("weird", ["other", "groom"]) is None
    assert map_label_to_model_class(None, ["a", "b"]) is None


def test_map_binarize():
    cn = ["other", "groom"]
    assert map_label_to_model_class("groom", cn, binarize="groom") == "groom"
    assert map_label_to_model_class("rear", cn, binarize="groom") == "other"


def test_map_union():
    cn = ["other", "grooming"]
    union = {"grooming": ["facewash", "paw"]}
    assert map_label_to_model_class("facewash", cn, label_union=union) == "grooming"
    assert map_label_to_model_class("grooming", cn, label_union=union) == "grooming"


def test_alstate_accessors():
    al = AlState()
    al.load_results(RESULTS)
    assert al.class_names == ["other", "groom"]
    assert al.predicted_name(11) == "groom"       # index 1 -> name
    assert al.predicted_name(10) == "groom"
    assert abs(al.confidence(11) - 0.7) < 1e-9
    assert al.uncertainty(10) == 0.9


def test_misclassification_and_unknowable():
    al = AlState()
    al.load_results(RESULTS)
    assert al.is_misclassified(10, "groom") is False   # pred groom == label groom
    assert al.is_misclassified(12, "groom") is True    # pred other != label groom
    assert al.is_misclassified(10, "weird") is False   # unknowable -> not flagged


def test_binarized_misclassification():
    al = AlState()
    al.load_results(RESULTS, binarize="groom")
    assert al.is_misclassified(12, "rear") is False    # rear->other, pred other
    assert al.is_misclassified(10, "rear") is True     # rear->other, pred groom


def test_uncertain_navigation_and_sets():
    al = AlState()
    al.load_results(RESULTS)
    assert al.uncertain_frames(0.5) == [10, 12]
    assert al.next_uncertain(10, 0.5) == 12
    assert al.next_uncertain(12, 0.5) is None
    mis = al.misclassified_frames(lambda i: "groom" if i == 12 else None)
    assert mis == {12}


def test_hud_info():
    al = AlState()
    al.load_results(RESULTS)
    info = hud_info(al, 12, "groom")
    assert info["predicted"] == "other" and info["mismatch"] is True
    assert hud_info(None, 12, "groom")["predicted"] is None


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
