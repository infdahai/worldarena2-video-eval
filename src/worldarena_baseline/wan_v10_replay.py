"""Deterministic single-topology replay for Wan v10."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

from .wan_balanced_sampling import build_balanced_history


V10_TARGET = {
    "single_dominant": 0.40,
    "bimanual_heavy": 0.35,
    "mixed": 0.20,
    "quiet": 0.05,
}


def _seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def build_v10_replay(
    sample_roles: Mapping[str, str],
    *,
    steps: int,
    world_size: int,
    seed: int,
    text_dropout_rate: float = 0.1,
) -> tuple[dict[str, object], ...]:
    if steps <= 0 or world_size <= 0:
        raise ValueError("v10 replay steps/world_size must be positive")
    if type(seed) is not int or seed < 0:
        raise ValueError("v10 replay seed must be a non-negative integer")
    if not 0 <= text_dropout_rate < 1:
        raise ValueError("v10 text dropout rate must be in [0,1)")
    roles = dict(sample_roles)
    history = build_balanced_history(
        roles,
        target=V10_TARGET,
        seed=seed,
        samples_seen=steps * world_size,
    )
    rows: list[dict[str, object]] = []
    for offset, sample in enumerate(history):
        optimizer_step = offset // world_size + 1
        rank = offset % world_size
        dropout_draw = _seed(seed, "text-dropout", optimizer_step, rank, sample) / (1 << 63)
        rows.append(
            {
                "contract": "wan-v10-replay-row/1",
                "optimizer_step": optimizer_step,
                "rank": rank,
                "sample": sample,
                "sample_role": roles[sample],
                "noise_seed": _seed(seed, "noise", optimizer_step, rank, sample),
                "timestep_seed": _seed(seed, "timestep", optimizer_step, rank, sample),
                "negative_family": "reverse" if optimizer_step % 2 else "swap",
                "action_dropout": False,
                "text_dropout": dropout_draw < text_dropout_rate,
            }
        )
    return tuple(rows)


def replay_sha256(rows: tuple[dict[str, object], ...]) -> str:
    if not rows:
        raise ValueError("v10 replay cannot be empty")
    payload = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()

