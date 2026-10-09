"""The learning-rate finder's suggestion (backend/services/lr_valley.py)."""

import numpy as np

from backend.services.lr_valley import valley_lr


def test_valley_picks_the_descent_not_a_wobble_at_tiny_lrs():
    # Shaped like rig2's curve: noisy start, flat, a real descent 1e-4 -> 2e-3,
    # then divergence. Lightning's linear-lr slope picks the wobble near 1e-6.
    lrs = np.logspace(-6, -1, 100)
    lg = np.log10(lrs)
    losses = 0.68 - 0.1 * np.clip((lg + 4) / 1.3, 0, 1) + np.where(lg > -2.7, 0.15 * (lg + 2.7), 0)
    losses[:3] = [0.40, 0.55, 0.69]
    lr = valley_lr(lrs, losses)
    assert 1e-4 < lr < 2e-3


def test_valley_has_no_answer_when_the_loss_never_falls():
    assert valley_lr([1e-5, 1e-4, 1e-3], [0.5, 0.6, 0.7]) is None
    assert valley_lr([1e-5], [0.5]) is None
