import itertools

import numpy as np
import pytest

from model import _chronological_split_indices, train_test_split


SPLITS = ("train", "val", "test")
PERCENTAGES = {"train": 0.7, "val": 0.15, "test": 0.15}


def test_default_chronological_order_matches_historical_boundaries():
    result = _chronological_split_indices(100, 0.7, 0.15, 0.15)

    assert result["train"].tolist() == list(range(0, 70))
    assert result["val"].tolist() == list(range(70, 85))
    assert result["test"].tolist() == list(range(85, 100))


@pytest.mark.parametrize("order", itertools.permutations(SPLITS))
def test_all_chronological_permutations_are_contiguous_and_exact_size(order):
    result = _chronological_split_indices(100, 0.7, 0.15, 0.15, order)

    cursor = 0
    for split_name in order:
        expected_size = round(100 * PERCENTAGES[split_name])
        assert result[split_name].tolist() == list(range(cursor, cursor + expected_size))
        cursor += expected_size
    assert cursor == 100


def test_chronological_order_handles_integer_rounding_without_dropping_samples():
    result = _chronological_split_indices(11, 0.7, 0.15, 0.15, ("val", "train", "test"))
    combined = np.concatenate([result[name] for name in ("val", "train", "test")])

    assert combined.tolist() == list(range(11))
    assert {name: len(indices) for name, indices in result.items()} == {
        "val": 1,
        "train": 8,
        "test": 2,
    }


@pytest.mark.parametrize(
    "bad_order",
    [
        ("train", "val"),
        ("train", "train", "test"),
        ("train", "val", "holdout"),
    ],
)
def test_invalid_chronological_order_is_rejected(bad_order):
    with pytest.raises(ValueError, match="split_order"):
        _chronological_split_indices(100, 0.7, 0.15, 0.15, bad_order)


def test_train_test_split_uses_configured_chronological_order():
    paths = [[[f"frame-{i}"] for i in range(20)]]
    labels = [[i % 3 for i in range(20)]]
    boundary_dists = [[float(i) for i in range(20)]]

    train, val, test, _ = train_test_split(
        paths,
        labels,
        boundary_dists,
        train_percent=0.7,
        val_percent=0.15,
        test_percent=0.15,
        shuffle=False,
        split_order=("test", "train", "val"),
    )

    assert [row[0] for row in test[0]] == [f"frame-{i}" for i in range(3)]
    assert [row[0] for row in train[0]] == [f"frame-{i}" for i in range(3, 17)]
    assert [row[0] for row in val[0]] == [f"frame-{i}" for i in range(17, 20)]


def test_split_order_is_ignored_for_randomized_split():
    paths = [[[f"frame-{i}"] for i in range(30)]]
    labels = [[0] * 15 + [1] * 15]
    boundary_dists = [[float(i) for i in range(30)]]

    # The invalid value deliberately proves it is not consulted when shuffle=True.
    train_test_split(
        paths,
        labels,
        boundary_dists,
        shuffle=True,
        split_order=("ignored",),
    )
