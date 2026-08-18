"""Optimizer, scheduler, phase and checkpoint contracts for Wan v8."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .wan_v8_model import V8_BLOCKS, v8_trainable_parameter_names


NATIVE_LR = 1e-6
GATE_LR = 5e-5
WARMUP_STEPS = 25
MAX_STEPS = 250
CHECKPOINT_STEPS = (10, 25, 50, 100, 150, 200, 250)


def _named_trainables(model: nn.Module) -> dict[str, nn.Parameter]:
    expected = v8_trainable_parameter_names(model)
    named = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(named) != expected:
        raise ValueError("v8 trainable whitelist drift")
    return named


def build_v8_optimizer(model: nn.Module) -> torch.optim.AdamW:
    named = _named_trainables(model)
    gates = [(name, value) for name, value in named.items() if name.endswith("channel_gate")]
    native = [(name, value) for name, value in named.items() if not name.endswith("channel_gate")]
    if len(gates) != len(V8_BLOCKS) or not native:
        raise ValueError("v8 optimizer families are incomplete")
    if len({id(value) for _, value in gates + native}) != len(gates) + len(native):
        raise ValueError("v8 optimizer parameters are duplicated")
    return torch.optim.AdamW(
        [
            {"name": "native_qkvo", "params": [v for _, v in native], "param_names": [n for n, _ in native], "lr": NATIVE_LR, "weight_decay": 0.01},
            {"name": "geometry_gates", "params": [v for _, v in gates], "param_names": [n for n, _ in gates], "lr": GATE_LR, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def v8_lr_multiplier(step: int) -> float:
    if type(step) is not int or not 1 <= step <= MAX_STEPS:
        raise ValueError("v8 step must be in [1,250]")
    if step <= WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    return 0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_v8_learning_rates(optimizer: torch.optim.Optimizer, step: int) -> tuple[float, float]:
    if [group.get("name") for group in optimizer.param_groups] != ["native_qkvo", "geometry_gates"]:
        raise ValueError("v8 optimizer group contract mismatch")
    scale = v8_lr_multiplier(step)
    values = (NATIVE_LR * scale, GATE_LR * scale)
    for group, value in zip(optimizer.param_groups, values, strict=True):
        group["lr"] = value
    return values


def _decision(metrics: Mapping[str, Any], *, step: int) -> dict[str, Any]:
    threshold = 11 if step == 100 else 14
    reasons: list[str] = []
    for family in ("reverse", "shift", "swap"):
        item = metrics.get(family)
        if not isinstance(item, Mapping) or int(item.get("wins", -1)) < threshold:
            reasons.append(f"{family}_wins_below_{threshold}")
        if not isinstance(item, Mapping) or float(item.get("mean_margin", float("-inf"))) <= 0:
            reasons.append(f"{family}_margin_not_positive")
    scalar_names = ("routing_retention", "fm_regression", "position_regression", "velocity_regression") if step == 100 else ("routing_retention", "fm_regression", "position_improvement", "velocity_improvement")
    for name in scalar_names:
        try: value=float(metrics[name])
        except (KeyError,TypeError,ValueError): value=float("nan")
        if not math.isfinite(value): reasons.append(f"{name}_not_finite")
    if float(metrics.get("routing_retention", float("-inf"))) < 0.90:
        reasons.append("routing_retention_below_90_percent")
    if float(metrics.get("fm_regression", float("inf"))) >= 0.02 if step == 100 else float(metrics.get("fm_regression", float("inf"))) > 0.02:
        reasons.append("fm_regression_above_limit")
    if step == 100:
        if float(metrics.get("position_regression", float("inf"))) > 0.02:
            reasons.append("position_regression_above_2_percent")
        if float(metrics.get("velocity_regression", float("inf"))) > 0.02:
            reasons.append("velocity_regression_above_2_percent")
    else:
        if float(metrics.get("position_improvement", float("-inf"))) <= 0.05:
            reasons.append("position_improvement_not_above_5_percent")
        if float(metrics.get("velocity_improvement", float("-inf"))) <= 0.05:
            reasons.append("velocity_improvement_not_above_5_percent")
    return {"step": step, "pass": not reasons, "reasons": reasons, "win_threshold": threshold}


def step100_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return _decision(metrics, step=100)


def step250_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return _decision(metrics, step=250)


def build_v8_checkpoint(
    *, step: int, model: nn.Module, optimizer: torch.optim.Optimizer,
    lineage: Mapping[str, str], calibration: Mapping[str, float], gates: Mapping[str, Any],
) -> dict[str, Any]:
    if step not in CHECKPOINT_STEPS:
        raise ValueError("v8 checkpoint step is not approved")
    phase = "mechanism" if step <= 100 else "trajectory"
    if phase == "trajectory" and not bool(gates.get("step100", {}).get("pass")):
        raise ValueError("Phase T requires passing step100")
    state = {name: value.detach().cpu() for name, value in model.state_dict().items() if name in v8_trainable_parameter_names(model)}
    if not state:
        raise ValueError("v8 checkpoint trainable state is empty")
    return {
        "contract": "wan-v8-direct-action-band-checkpoint/1", "step": step, "phase": phase,
        "lineage": dict(lineage), "calibration": dict(calibration), "gates": dict(gates),
        "trainable_names": sorted(state), "model": state, "optimizer": optimizer.state_dict(),
        "lr_multiplier": v8_lr_multiplier(step),
    }


def validate_v8_checkpoint(payload: Mapping[str, Any], *, lineage: Mapping[str, str], expected_phase: str) -> None:
    if payload.get("contract") != "wan-v8-direct-action-band-checkpoint/1":
        raise ValueError("v8 checkpoint contract mismatch")
    if payload.get("lineage") != dict(lineage):
        raise ValueError("v8 checkpoint lineage mismatch")
    step = payload.get("step")
    if type(step) is not int or step not in CHECKPOINT_STEPS:
        raise ValueError("v8 checkpoint step mismatch")
    if payload.get("phase") != expected_phase:
        raise ValueError("v8 checkpoint phase mismatch")
    if expected_phase == "trajectory" and not bool(payload.get("gates", {}).get("step100", {}).get("pass")):
        raise ValueError("Phase T requires passing step100")
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, Mapping) or len(optimizer.get("param_groups", [])) != 2:
        raise ValueError("v8 optimizer state is incomplete")
