from __future__ import annotations

import numpy as np


def resample_actions(actions: np.ndarray, num_frames: int = 81) -> np.ndarray:
    """Linearly resample a joint trajectory while preserving both endpoints."""
    values = np.asarray(actions)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("actions must have shape (T, action_dim) with T >= 1")
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    if values.shape[0] == 1:
        return np.repeat(values, num_frames, axis=0)

    source = np.arange(values.shape[0], dtype=np.float64)
    target = np.linspace(0.0, float(values.shape[0] - 1), num_frames)
    sampled = np.empty((num_frames, values.shape[1]), dtype=np.float64)
    for dimension in range(values.shape[1]):
        sampled[:, dimension] = np.interp(target, source, values[:, dimension])
    return sampled
