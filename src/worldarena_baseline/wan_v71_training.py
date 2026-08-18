"""Strict optimizer and checkpoint contract for v7.1 geometry LoRA."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import torch
from torch import Tensor, nn

from .wan_v7_model import STAGE_A_BLOCKS, v71_trainable_parameter_names


V71_CHECKPOINT_CONTRACT: Final = "wan-action-v71-se3-geometry-lora-checkpoint/1"
V71_LR: Final = 1e-4
V71_STEPS: Final = (10, 25, 50, 100)
V71_CONFIG: Final = {
    "blocks": list(STAGE_A_BLOCKS),
    "rank": 16,
    "lr": V71_LR,
    "loss": "weighted-fm-only",
}


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


def v71_optimizer_group(model: nn.Module) -> dict[str, Any]:
    names = v71_trainable_parameter_names(model, rank=16)
    named = dict(model.named_parameters())
    return {
        "params": [named[name] for name in sorted(names)],
        "lr": V71_LR,
        "name": "geometry_qkvo_lora_and_channel_gate",
    }


def _model_state(model: nn.Module) -> dict[str, Tensor]:
    names = v71_trainable_parameter_names(model, rank=16)
    state = model.state_dict()
    return {name: state[name].detach().cpu().clone() for name in sorted(names)}


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


def build_v71_checkpoint(
    *,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    parent_sha256: str,
    source_hashes: Mapping[str, str],
    cache_sha256: str,
    replay_sha256: str,
    preflight_sha256: str,
) -> dict[str, Any]:
    if step not in V71_STEPS:
        raise ValueError("v7.1 checkpoint step is not approved")
    payload = {
        "contract": V71_CHECKPOINT_CONTRACT,
        "step": step,
        "config": dict(V71_CONFIG),
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
    validate_v71_checkpoint(payload, expected=_lineage(
        parent_sha256=parent_sha256,
        source_hashes=source_hashes,
        cache_sha256=cache_sha256,
        replay_sha256=replay_sha256,
        preflight_sha256=preflight_sha256,
    ))
    return payload


def validate_v71_checkpoint(
    payload: Mapping[str, Any], *, expected: Mapping[str, Any]
) -> None:
    if payload.get("contract") != V71_CHECKPOINT_CONTRACT:
        raise ValueError("v7.1 checkpoint contract differs")
    if payload.get("step") not in V71_STEPS or payload.get("config") != V71_CONFIG:
        raise ValueError("v7.1 checkpoint config differs")
    for name in (
        "parent_sha256",
        "source_hashes",
        "cache_sha256",
        "replay_sha256",
        "preflight_sha256",
    ):
        if payload.get(name) != expected.get(name):
            raise ValueError("v7.1 checkpoint lineage differs")
    state = payload.get("model")
    if not isinstance(state, Mapping) or set(state) != _expected_names():
        raise ValueError("v7.1 checkpoint parameter names differ")
    if any(not isinstance(value, Tensor) or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("v7.1 checkpoint parameter tensors are invalid")
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise ValueError("v7.1 checkpoint optimizer is missing")
    groups = optimizer.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("v7.1 checkpoint optimizer groups differ")
    group = groups[0]
    if (
        not isinstance(group, Mapping)
        or group.get("lr") != V71_LR
        or group.get("name") != "geometry_qkvo_lora_and_channel_gate"
        or not isinstance(group.get("params"), list)
        or len(group["params"]) != 27
    ):
        raise ValueError("v7.1 checkpoint optimizer group differs")
