"""Strict checkpoint contract for v7.1 geometry LoRA counterfactual training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import torch
from torch import Tensor, nn

from .wan_v71_training import V71_LR, v71_optimizer_group
from .wan_v7_model import STAGE_A_BLOCKS, v71_trainable_parameter_names


V71_CF_CHECKPOINT_CONTRACT: Final = "wan-action-v71-se3-geometry-lora-cf-checkpoint/1"
V71_CF_STEPS: Final = (10, 25, 50)


def _model_state(model: nn.Module) -> dict[str, Tensor]:
    names = v71_trainable_parameter_names(model, rank=16)
    state = model.state_dict()
    return {name: state[name].detach().cpu().clone() for name in sorted(names)}


def _expected_names() -> set[str]:
    suffixes = {
        "channel_gate",
        "q_lora.down",
        "q_lora.up",
        "k_lora.down",
        "k_lora.up",
        "v_lora.down",
        "v_lora.up",
        "o_lora.down",
        "o_lora.up",
    }
    return {
        f"geometry_wrappers.{block}.{suffix}"
        for block in STAGE_A_BLOCKS
        for suffix in suffixes
    }


def _lineage(
    *,
    parent_sha256: str,
    source_hashes: Mapping[str, str],
    cache_sha256: str,
    replay_sha256: str,
    preflight_sha256: str,
) -> dict[str, Any]:
    return {
        "parent_sha256": parent_sha256,
        "source_hashes": dict(source_hashes),
        "cache_sha256": cache_sha256,
        "replay_sha256": replay_sha256,
        "preflight_sha256": preflight_sha256,
    }


def _config(*, lambda_cf: float, tau: float, init_seed: int) -> dict[str, Any]:
    if not (0 < lambda_cf < float("inf")) or not (0 < tau < float("inf")):
        raise ValueError("v7.1-CF calibration must be finite and positive")
    if type(init_seed) is not int or init_seed < 0:
        raise ValueError("v7.1-CF initialization seed is invalid")
    return {
        "blocks": list(STAGE_A_BLOCKS),
        "rank": 16,
        "lr": V71_LR,
        "loss": "weighted-fm+geometry-counterfactual-softplus",
        "lambda_cf": float(lambda_cf),
        "tau": float(tau),
        "init_seed": init_seed,
        "counterfactual_scope": "geometry-only",
        "negative_schedule": "reverse,shift+1,swap,reverse,shift-1,swap",
    }


def build_v71_cf_checkpoint(
    *,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lambda_cf: float,
    tau: float,
    init_seed: int,
    parent_sha256: str,
    source_hashes: Mapping[str, str],
    cache_sha256: str,
    replay_sha256: str,
    preflight_sha256: str,
) -> dict[str, Any]:
    if step not in V71_CF_STEPS:
        raise ValueError("v7.1-CF checkpoint step is not approved")
    payload = {
        "contract": V71_CF_CHECKPOINT_CONTRACT,
        "step": step,
        "config": _config(lambda_cf=lambda_cf, tau=tau, init_seed=init_seed),
        **_lineage(
            parent_sha256=parent_sha256,
            source_hashes=source_hashes,
            cache_sha256=cache_sha256,
            replay_sha256=replay_sha256,
            preflight_sha256=preflight_sha256,
        ),
        "model": _model_state(model),
        "optimizer": optimizer.state_dict(),
    }
    validate_v71_cf_checkpoint(
        payload,
        expected={
            **_lineage(
                parent_sha256=parent_sha256,
                source_hashes=source_hashes,
                cache_sha256=cache_sha256,
                replay_sha256=replay_sha256,
                preflight_sha256=preflight_sha256,
            ),
            "lambda_cf": lambda_cf,
            "tau": tau,
            "init_seed": init_seed,
        },
    )
    return payload


def validate_v71_cf_checkpoint(
    payload: Mapping[str, Any], *, expected: Mapping[str, Any]
) -> None:
    if payload.get("contract") != V71_CF_CHECKPOINT_CONTRACT:
        raise ValueError("v7.1-CF checkpoint contract differs")
    if payload.get("step") not in V71_CF_STEPS or payload.get("config") != _config(
        lambda_cf=float(expected["lambda_cf"]),
        tau=float(expected["tau"]),
        init_seed=int(expected["init_seed"]),
    ):
        raise ValueError("v7.1-CF checkpoint config differs")
    for name in (
        "parent_sha256",
        "source_hashes",
        "cache_sha256",
        "replay_sha256",
        "preflight_sha256",
    ):
        if payload.get(name) != expected.get(name):
            raise ValueError("v7.1-CF checkpoint lineage differs")
    state = payload.get("model")
    if not isinstance(state, Mapping) or set(state) != _expected_names():
        raise ValueError("v7.1-CF checkpoint parameter names differ")
    if any(
        not isinstance(value, Tensor) or not torch.isfinite(value).all()
        for value in state.values()
    ):
        raise ValueError("v7.1-CF checkpoint parameter tensors are invalid")
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise ValueError("v7.1-CF checkpoint optimizer is missing")
    groups = optimizer.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("v7.1-CF checkpoint optimizer groups differ")
    group = groups[0]
    if (
        not isinstance(group, Mapping)
        or group.get("lr") != V71_LR
        or group.get("name") != "geometry_qkvo_lora_and_channel_gate"
        or not isinstance(group.get("params"), list)
        or len(group["params"]) != 27
    ):
        raise ValueError("v7.1-CF checkpoint optimizer group differs")
__all__ = [
    "V71_CF_CHECKPOINT_CONTRACT",
    "V71_CF_STEPS",
    "build_v71_cf_checkpoint",
    "validate_v71_cf_checkpoint",
    "v71_optimizer_group",
]
