"""Physical complete-action counterfactuals for the direct v8 action band."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .action_causality import build_joint_counterfactual


CounterfactualFamily = Literal["reverse", "shift", "swap"]


def _require_joint14(actions: np.ndarray) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 14 or len(values) < 2:
        raise ValueError("complete action must have shape (T>=2,14)")
    if not np.isfinite(values).all():
        raise ValueError("complete action must be finite")
    if np.any(values[:, [6, 13]] < 0) or np.any(values[:, [6, 13]] > 1):
        raise ValueError("complete action grippers must be in [0,1]")
    return values


def build_joint14_counterfactual(
    actions: np.ndarray, family: CounterfactualFamily, *, shift_direction: int = 0
) -> np.ndarray:
    """Return a full joint14 action edit that preserves both native anchors.

    Raster/support/SE(3) are intentionally rendered from this result later; no
    condition tensor is time-reversed or arm-swapped in place.
    """
    correct = _require_joint14(actions)
    if family == "reverse":
        result = build_joint_counterfactual(correct, mode="reverse")
    elif family == "swap":
        result = build_joint_counterfactual(correct, mode="swap")
    elif family == "shift":
        if shift_direction not in (-1, 1):
            raise ValueError("shift direction must be +1 or -1")
        result = correct.copy()
        if shift_direction == 1:
            result[1:] = correct[:-1]
        else:
            result[1:-1] = correct[2:]
            result[-1] = correct[-1]
    else:
        raise ValueError(f"unsupported complete-action counterfactual: {family}")
    result[:, [6, 13]] = np.clip(result[:, [6, 13]], 0.0, 1.0)
    if not np.array_equal(result[0], correct[0]):
        raise RuntimeError("complete counterfactual changed frame-zero anchor")
    return _require_joint14(result)


def negative_for_step(step: int) -> tuple[CounterfactualFamily, int]:
    if type(step) is not int or step <= 0:
        raise ValueError("optimizer step must be positive")
    cycle: tuple[tuple[CounterfactualFamily, int], ...] = (
        ("reverse", 0), ("shift", 1), ("swap", 0),
        ("reverse", 0), ("shift", -1), ("swap", 0),
    )
    return cycle[(step - 1) % len(cycle)]
