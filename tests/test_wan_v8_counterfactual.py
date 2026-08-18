import numpy as np
import pytest

from worldarena_baseline.wan_v8_counterfactual import (
    build_joint14_counterfactual,
    negative_for_step,
)


def _actions() -> np.ndarray:
    values = np.zeros((5, 14), dtype=np.float64)
    values[:, 0] = np.arange(5)
    values[:, 7] = 10 + np.arange(5)
    values[:, 6] = np.linspace(0.1, 0.5, 5)
    values[:, 13] = np.linspace(0.9, 0.5, 5)
    return values


@pytest.mark.parametrize("family,direction", [("reverse", 0), ("shift", 1), ("shift", -1), ("swap", 0)])
def test_complete_joint_negative_preserves_anchor_and_changes_both_action_streams(family, direction):
    correct = _actions()
    wrong = build_joint14_counterfactual(correct, family, shift_direction=direction)
    np.testing.assert_array_equal(wrong[0], correct[0])
    assert not np.array_equal(wrong[:, :7], correct[:, :7])
    assert not np.array_equal(wrong[:, 7:], correct[:, 7:])
    assert np.isfinite(wrong).all()
    assert np.all((wrong[:, [6, 13]] >= 0) & (wrong[:, [6, 13]] <= 1))


def test_shift_uses_hold_padding_not_cyclic_wraparound():
    correct = _actions()
    plus = build_joint14_counterfactual(correct, "shift", shift_direction=1)
    minus = build_joint14_counterfactual(correct, "shift", shift_direction=-1)
    np.testing.assert_array_equal(plus[1], correct[0])
    np.testing.assert_array_equal(minus[-1], correct[-1])


def test_negative_cycle_is_balanced_and_alternates_shift_direction():
    assert [negative_for_step(step) for step in range(1, 7)] == [
        ("reverse", 0), ("shift", 1), ("swap", 0),
        ("reverse", 0), ("shift", -1), ("swap", 0),
    ]
