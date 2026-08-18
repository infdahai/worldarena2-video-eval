"""Optimizer, schedule, and checkpoint contracts for Wan v9."""

from __future__ import annotations

from collections.abc import Mapping
import math
import re
from typing import Any

import torch
from torch import nn

from .wan_v9_model import v9_trainable_parameter_names


NATIVE_LR = 1e-6
ACTION_CROSS_LR = 1e-4
GATE_LR = 5e-5
WARMUP_STEPS = 25
MAX_STEPS = 500
CLIP_NORM = 1.0
CHECKPOINT_STEPS = (10, 25, 50, 100, 150, 200, 250, 300, 400, 500)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def classify_v9_parameter(name: str) -> str:
    if name.endswith("channel_gate"):
        return "channel_gates"
    if ".base." in name and any(f".base.{projection}." in name for projection in ("q", "k", "v", "o")):
        return "native_qkvo"
    if name.startswith("action_tokenizer.") or ".cross." in name:
        return "action_cross"
    raise ValueError(f"unclassified v9 trainable parameter: {name}")


def _named_trainables(model: nn.Module) -> dict[str, nn.Parameter]:
    expected = v9_trainable_parameter_names(model)
    actual = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(actual) != expected or not actual:
        raise ValueError("v9 trainable whitelist drift")
    return actual


def build_v9_optimizer(model: nn.Module) -> torch.optim.AdamW:
    named = _named_trainables(model)
    families: dict[str, list[tuple[str, nn.Parameter]]] = {
        "native_qkvo": [], "action_cross": [], "channel_gates": [],
    }
    for name, parameter in named.items():
        families[classify_v9_parameter(name)].append((name, parameter))
    if any(not values for values in families.values()):
        raise ValueError("v9 optimizer families are incomplete")
    flattened = [parameter for values in families.values() for _, parameter in values]
    if len({id(parameter) for parameter in flattened}) != len(flattened):
        raise ValueError("v9 optimizer parameters are duplicated")
    settings = {
        "native_qkvo": (NATIVE_LR, 0.01),
        "action_cross": (ACTION_CROSS_LR, 0.01),
        "channel_gates": (GATE_LR, 0.0),
    }
    groups = []
    for family in ("native_qkvo", "action_cross", "channel_gates"):
        values = sorted(families[family], key=lambda item: item[0])
        lr, weight_decay = settings[family]
        groups.append({
            "name": family,
            "params": [parameter for _, parameter in values],
            "param_names": [name for name, _ in values],
            "lr": lr,
            "weight_decay": weight_decay,
        })
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def v9_lr_multiplier(step: int) -> float:
    if type(step) is not int or not 1 <= step <= MAX_STEPS:
        raise ValueError("v9 scheduler step must be in [1,500]")
    if step <= WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    return 0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_v9_learning_rates(optimizer: torch.optim.Optimizer, step: int) -> tuple[float, float, float]:
    names = [group.get("name") for group in optimizer.param_groups]
    if names != ["native_qkvo", "action_cross", "channel_gates"]:
        raise ValueError("v9 optimizer group order differs")
    scale = v9_lr_multiplier(step)
    values = (NATIVE_LR * scale, ACTION_CROSS_LR * scale, GATE_LR * scale)
    for group, value in zip(optimizer.param_groups, values, strict=True):
        group["lr"] = value
    return values


def _validate_lineage(lineage: Mapping[str, str]) -> None:
    required = {"parent_sha256", "source_closure_sha256"}
    if not required <= set(lineage):
        raise ValueError("v9 checkpoint lineage lacks parent/source closure")
    for key, value in lineage.items():
        if not isinstance(key, str) or not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError("v9 checkpoint lineage values must be lowercase SHA-256")


def _validate_optimizer_payload(payload: Mapping[str, Any], *, step: int) -> None:
    groups = payload.get("param_groups")
    state = payload.get("state")
    if not isinstance(groups, list) or len(groups) != 3 or not isinstance(state, Mapping):
        raise ValueError("v9 optimizer state is incomplete")
    expected_names = ["native_qkvo", "action_cross", "channel_gates"]
    expected_lrs = tuple(base * v9_lr_multiplier(step) for base in (NATIVE_LR, ACTION_CROSS_LR, GATE_LR))
    referenced: list[int] = []
    for group, name, lr in zip(groups, expected_names, expected_lrs, strict=True):
        if group.get("name") != name or not math.isclose(float(group.get("lr", -1)), lr, rel_tol=0, abs_tol=1e-15):
            raise ValueError("v9 optimizer name/LR differs from scheduler contract")
        names = group.get("param_names")
        parameters = group.get("params")
        if not isinstance(names, list) or not names or not isinstance(parameters, list) or len(names) != len(parameters):
            raise ValueError("v9 optimizer parameter inventory is incomplete")
        referenced.extend(parameters)
    if len(referenced) != len(set(referenced)) or set(referenced) != set(state):
        raise ValueError("v9 optimizer state does not cover every trainable parameter")
    for parameter_id in referenced:
        item = state[parameter_id]
        if not isinstance(item, Mapping) or not {"step", "exp_avg", "exp_avg_sq"} <= set(item):
            raise ValueError("v9 AdamW state is incomplete")


def build_v9_checkpoint(
    *,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lineage: Mapping[str, str],
    calibration: Mapping[str, float],
    gates: Mapping[str, Any],
    gradient_history: Mapping[str, float],
    negative_cycle_index: int,
) -> dict[str, Any]:
    if step not in CHECKPOINT_STEPS:
        raise ValueError("v9 checkpoint step is not approved")
    phase = "mechanism" if step <= 250 else "trajectory"
    if phase == "trajectory" and gates.get("step250", {}).get("pass") is not True:
        raise ValueError("v9 Phase T requires a passing step250 receipt")
    _validate_lineage(lineage)
    lambda_cf = float(calibration.get("lambda_cf", float("nan")))
    if not math.isfinite(lambda_cf) or lambda_cf <= 0:
        raise ValueError("v9 lambda_cf calibration is invalid")
    if negative_cycle_index != step % 10:
        raise ValueError("v9 negative cycle index differs from completed step")
    if not gradient_history or any(not math.isfinite(float(value)) or float(value) < 0 for value in gradient_history.values()):
        raise ValueError("v9 gradient history is invalid")
    names = v9_trainable_parameter_names(model)
    state = model.state_dict()
    trainable_state = {name: state[name].detach().cpu() for name in names}
    optimizer_state = optimizer.state_dict()
    _validate_optimizer_payload(optimizer_state, step=step)
    return {
        "contract": "wan-v9-phase-locked-checkpoint/1",
        "step": step,
        "phase": phase,
        "lineage": dict(lineage),
        "calibration": dict(calibration),
        "gates": dict(gates),
        "gradient_history": dict(gradient_history),
        "negative_cycle_index": negative_cycle_index,
        "lr_multiplier": v9_lr_multiplier(step),
        "clip_norm": CLIP_NORM,
        "trainable_names": sorted(names),
        "model": trainable_state,
        "optimizer": optimizer_state,
    }


def validate_v9_checkpoint(
    payload: Mapping[str, Any],
    *,
    expected_lineage: Mapping[str, str],
    expected_phase: str,
) -> None:
    if payload.get("contract") != "wan-v9-phase-locked-checkpoint/1":
        raise ValueError("v9 checkpoint contract differs")
    _validate_lineage(expected_lineage)
    if payload.get("lineage") != dict(expected_lineage):
        raise ValueError("v9 checkpoint lineage differs")
    step = payload.get("step")
    if type(step) is not int or step not in CHECKPOINT_STEPS:
        raise ValueError("v9 checkpoint step differs")
    if payload.get("phase") != expected_phase:
        raise ValueError("v9 checkpoint phase differs")
    if expected_phase == "trajectory" and payload.get("gates", {}).get("step250", {}).get("pass") is not True:
        raise ValueError("v9 Phase T checkpoint lacks passing step250 receipt")
    if payload.get("negative_cycle_index") != step % 10:
        raise ValueError("v9 checkpoint negative cycle index differs")
    names = payload.get("trainable_names")
    model = payload.get("model")
    if not isinstance(names, list) or not names or not isinstance(model, Mapping) or set(model) != set(names):
        raise ValueError("v9 checkpoint model inventory is incomplete")
    _validate_optimizer_payload(payload.get("optimizer", {}), step=step)

