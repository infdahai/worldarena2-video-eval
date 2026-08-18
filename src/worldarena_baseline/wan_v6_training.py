"""Immutable training and replay contracts for v6 trajectory correction."""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn


V6_CHECKPOINT_STEPS = (10, 25, 50, 100)
WARMUP_STEPS = 5
LEARNING_RATE = 1e-4
_SHA256 = re.compile(r"[0-9a-f]{64}")


def v6_training_contract() -> dict[str, Any]:
    return {
        "contract": "wan-action-lite-v6-training/1",
        "world_size": 7,
        "rank_mapping": list(range(7)),
        "frames": 81,
        "height": 480,
        "width": 640,
        "micro_batch": 1,
        "dataset_rows": 1000,
        "max_steps": 100,
        "warmup_steps": WARMUP_STEPS,
        "learning_rate": LEARNING_RATE,
        "rank": 8,
        "adapter_dim": 256,
        "injection_points": [8, 16, 24],
        "checkpoint_steps": list(V6_CHECKPOINT_STEPS),
        "discovery_indices": list(range(900, 908)),
        "position_gradient_ratio": 0.4,
        "velocity_gradient_ratio": 0.2,
        "huber_delta": 0.02,
        "parent_frozen": True,
        "backbone_frozen": True,
        "probe_frozen": True,
        "trainable_family": "gripper_trajectory_correction_only",
    }


def build_v6_replay_manifest(
    *, dataset_size: int, dataset_manifest_sha256: str, seed: int
) -> dict[str, Any]:
    if dataset_size != 1000:
        raise ValueError("v6 replay requires exact clean-1000 dataset")
    if not isinstance(dataset_manifest_sha256, str) or _SHA256.fullmatch(
        dataset_manifest_sha256
    ) is None:
        raise ValueError("dataset manifest hash must be lowercase SHA-256")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = random.Random(seed)
    records: list[list[dict[str, int]]] = []
    for step in range(1, V6_CHECKPOINT_STEPS[-1] + 1):
        rank_records = []
        for rank in range(7):
            rank_records.append(
                {
                    "rank": rank,
                    "sample_index": ((step - 1) * 7 + rank) % dataset_size,
                    "timestep": rng.randint(1, 999),
                    "noise_seed": rng.randrange(0, 2**63),
                }
            )
        records.append(rank_records)
    return {
        "contract": "wan-action-lite-v6-replay/1",
        "world_size": 7,
        "rank_mapping": list(range(7)),
        "max_steps": V6_CHECKPOINT_STEPS[-1],
        "dataset_size": dataset_size,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "seed": seed,
        "records": records,
    }


def build_clean_parent_replay_manifest(
    *, dataset_manifest_sha256: str, seed: int
) -> dict[str, Any]:
    """Describe the exact deterministic 10-step provisional warmstart draws."""
    if not isinstance(dataset_manifest_sha256, str) or _SHA256.fullmatch(
        dataset_manifest_sha256
    ) is None:
        raise ValueError("dataset manifest hash must be lowercase SHA-256")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    records: list[list[dict[str, Any]]] = []
    for step in range(1, 11):
        rank_records = []
        for rank in range(7):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + step * 7 + rank)
            rank_records.append(
                {
                    "rank": rank,
                    "sample_index": ((step - 1) * 7 + rank) % 1000,
                    "timestep": float(torch.rand((), generator=generator).item()),
                    "noise_seed": int(
                        torch.randint(0, 2**31, (), generator=generator).item()
                    ),
                    "action_dropout": bool(
                        torch.rand((), generator=generator).item() < 0.1
                    ),
                    "text_dropout": bool(
                        torch.rand((), generator=generator).item() < 0.1
                    ),
                }
            )
        records.append(rank_records)
    return {
        "contract": "wan-action-lite-v6-clean-parent-replay/1",
        "world_size": 7,
        "rank_mapping": list(range(7)),
        "steps": 10,
        "dataset_size": 1000,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "seed": seed,
        "records": records,
    }


def load_v6_replay_manifest(
    path: Path | str,
    *,
    expected_dataset_size: int,
    expected_dataset_manifest_sha256: str,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        expected = build_v6_replay_manifest(
            dataset_size=expected_dataset_size,
            dataset_manifest_sha256=expected_dataset_manifest_sha256,
            seed=payload["seed"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("v6 replay differs from its deterministic contract") from exc
    if payload != expected:
        raise ValueError("v6 replay differs from its deterministic contract")
    return payload


def v6_optimizer_parameter_group(module: nn.Module) -> dict[str, Any]:
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not parameters or len(parameters) != len({id(parameter) for parameter in parameters}):
        raise ValueError("v6 correction trainable parameters are empty or duplicated")
    return {"name": "correction", "params": parameters, "lr": LEARNING_RATE}


def set_v6_learning_rate(optimizer, step: int) -> float:
    if type(step) is not int or not 1 <= step <= V6_CHECKPOINT_STEPS[-1]:
        raise ValueError("v6 optimizer step must be in [1,100]")
    if len(optimizer.param_groups) != 1 or optimizer.param_groups[0].get("name") != "correction":
        raise ValueError("v6 optimizer must contain exactly the correction group")
    rate = LEARNING_RATE * min(step / WARMUP_STEPS, 1.0)
    optimizer.param_groups[0]["lr"] = rate
    return rate


def v6_checkpoint_due(step: int) -> bool:
    return step in V6_CHECKPOINT_STEPS


def v6_discovery_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    """Apply the immutable step25 mechanism gate from the v6 design."""
    reasons: list[str] = []
    limits = {
        "position_error_improvement": 0.15,
        "velocity_error_improvement": 0.15,
        "routing_retention": 0.90,
        "fm_regression": 0.05,
    }
    try:
        if float(metrics["position_error_improvement"]) < limits[
            "position_error_improvement"
        ]:
            reasons.append("position_improvement_below_15_percent")
        if float(metrics["velocity_error_improvement"]) < limits[
            "velocity_error_improvement"
        ]:
            reasons.append("velocity_improvement_below_15_percent")
        if float(metrics["routing_retention"]) < limits["routing_retention"]:
            reasons.append("routing_retention_below_90_percent")
        if float(metrics["fm_regression"]) > limits["fm_regression"]:
            reasons.append("flow_matching_regression_above_5_percent")
    except (KeyError, TypeError, ValueError):
        reasons.append("missing_or_invalid_metric")
    if metrics.get("correct_better_than_shift_reverse_swap") is not True:
        reasons.append("counterfactual_ordering_failed")
    return {"pass": not reasons, "reasons": reasons, "limits": limits}


def anti_exploitation_gate(
    *, latent_probe_improvement: float, rgb_proxy_improvement: float
) -> dict[str, Any]:
    """Require independent RGB movement in the same direction as latent probe gain."""
    reasons: list[str] = []
    try:
        latent = float(latent_probe_improvement)
        rgb = float(rgb_proxy_improvement)
    except (TypeError, ValueError):
        return {"pass": False, "reasons": ["invalid_improvement"]}
    if not math.isfinite(latent) or not (latent > 0):
        reasons.append("latent_probe_did_not_improve")
    if not math.isfinite(rgb) or not (rgb > 0):
        reasons.append("rgb_proxy_did_not_improve")
    return {"pass": not reasons, "reasons": reasons}
