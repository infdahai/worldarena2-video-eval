"""Optimizer, schedule, and checkpoint lineage for Wan v10."""

from __future__ import annotations

from collections.abc import Mapping
import copy
import math
import re
from typing import Any

import torch
from torch import nn

from .wan_v10_model import v10_trainable_parameter_names


BASE_LRS = {
    "native_qkvo": 1e-6,
    "relation_encoders": 1e-4,
    "relation_gates": 5e-5,
    "hidden_eef_heads": 1e-4,
}
WEIGHT_DECAYS = {
    "native_qkvo": 0.01,
    "relation_encoders": 0.01,
    "relation_gates": 0.0,
    "hidden_eef_heads": 0.01,
}
GROUP_ORDER = tuple(BASE_LRS)
WARMUP_STEPS = 50
MAX_STEPS = 500
CHECKPOINT_STEPS = (50, 250, 500)
CLIP_NORM = 1.0
LINEAGE_KEYS = {
    "source_closure_sha256",
    "config_sha256",
    "parent_sha256",
    "data_sha256",
    "cache_sha256",
    "replay_sha256",
    "audit_sha256",
    "probe_sha256",
    "calibration_sha256",
}
STRATA = {"single_dominant", "bimanual_heavy", "mixed", "quiet"}
_SHA = re.compile(r"[0-9a-f]{64}")


def classify_v10_parameter(name: str) -> str:
    if name.endswith("relation_gate"):
        return "relation_gates"
    if "hidden_eef_head" in name:
        return "hidden_eef_heads"
    if name.startswith("relation_encoder.") or ".relation_q." in name or ".relation_k." in name:
        return "relation_encoders"
    if ".base." in name and any(f".base.{projection}." in name for projection in ("q", "k", "v", "o")):
        return "native_qkvo"
    raise ValueError(f"unclassified v10 trainable parameter: {name}")


def _named_trainables(model: nn.Module) -> dict[str, nn.Parameter]:
    expected = v10_trainable_parameter_names(model)
    actual = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(actual) != expected or not actual:
        raise ValueError("v10 trainable whitelist drift")
    if len({id(parameter) for parameter in actual.values()}) != len(actual):
        raise ValueError("v10 trainable parameter alias detected")
    return actual


def build_v10_optimizer(model: nn.Module) -> torch.optim.AdamW:
    named = _named_trainables(model)
    families: dict[str, list[tuple[str, nn.Parameter]]] = {name: [] for name in GROUP_ORDER}
    for name, parameter in named.items():
        families[classify_v10_parameter(name)].append((name, parameter))
    if any(not family for family in families.values()):
        raise ValueError("v10 optimizer family is empty")
    groups = []
    for family in GROUP_ORDER:
        values = sorted(families[family], key=lambda item: item[0])
        groups.append(
            {
                "name": family,
                "params": [parameter for _, parameter in values],
                "param_names": [name for name, _ in values],
                "lr": BASE_LRS[family],
                "weight_decay": WEIGHT_DECAYS[family],
            }
        )
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def v10_lr_multiplier(step: int) -> float:
    if type(step) is not int or not 1 <= step <= MAX_STEPS:
        raise ValueError("v10 scheduler step must be in [1,500]")
    if step <= WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    return 0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * progress))


def set_v10_learning_rates(optimizer: torch.optim.Optimizer, step: int) -> tuple[float, ...]:
    if [group.get("name") for group in optimizer.param_groups] != list(GROUP_ORDER):
        raise ValueError("v10 optimizer group order differs")
    multiplier = v10_lr_multiplier(step)
    values = tuple(BASE_LRS[name] * multiplier for name in GROUP_ORDER)
    for group, value in zip(optimizer.param_groups, values, strict=True):
        group["lr"] = value
    return values


def _validate_lineage(lineage: Mapping[str, str]) -> None:
    if set(lineage) != LINEAGE_KEYS:
        raise ValueError("v10 checkpoint lineage keys differ")
    if any(not isinstance(value, str) or _SHA.fullmatch(value) is None for value in lineage.values()):
        raise ValueError("v10 checkpoint lineage must contain lowercase SHA-256 values")


def _validate_calibration(calibration: Mapping[str, Any]) -> None:
    if calibration.get("contract") != "wan-v10-loss-calibration/1" or calibration.get("frozen") is not True:
        raise ValueError("v10 checkpoint calibration contract differs")
    lambdas = calibration.get("lambdas")
    if not isinstance(lambdas, Mapping) or set(lambdas) != {"cf", "phase", "hidden", "position", "velocity"}:
        raise ValueError("v10 checkpoint calibration lambdas differ")
    if any(not math.isfinite(float(value)) or not 1e-4 <= float(value) <= 100 for value in lambdas.values()):
        raise ValueError("v10 checkpoint calibration lambda is invalid")


def _validate_optimizer(payload: Mapping[str, Any], *, step: int) -> None:
    groups, state = payload.get("param_groups"), payload.get("state")
    if not isinstance(groups, list) or len(groups) != 4 or not isinstance(state, Mapping) or not state:
        raise ValueError("v10 optimizer state is incomplete")
    expected_lrs = tuple(BASE_LRS[name] * v10_lr_multiplier(step) for name in GROUP_ORDER)
    referenced: list[int] = []
    for group, family, lr in zip(groups, GROUP_ORDER, expected_lrs, strict=True):
        if group.get("name") != family or not math.isclose(float(group.get("lr", -1)), lr, rel_tol=0, abs_tol=1e-15):
            raise ValueError("v10 optimizer group/LR differs")
        if float(group.get("weight_decay", -1)) != WEIGHT_DECAYS[family]:
            raise ValueError("v10 optimizer weight decay differs")
        names, parameters = group.get("param_names"), group.get("params")
        if not isinstance(names, list) or not names or not isinstance(parameters, list) or len(names) != len(parameters):
            raise ValueError("v10 optimizer parameter inventory is incomplete")
        referenced.extend(parameters)
    if len(referenced) != len(set(referenced)) or set(referenced) != set(state):
        raise ValueError("v10 optimizer state does not cover every parameter")
    for identifier in referenced:
        item = state[identifier]
        if not isinstance(item, Mapping) or not {"step", "exp_avg", "exp_avg_sq"} <= set(item):
            raise ValueError("v10 optimizer AdamW state is incomplete")


def build_v10_checkpoint(
    *,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lineage: Mapping[str, str],
    calibration: Mapping[str, Any],
    initialization_seed: int,
    topology: Mapping[str, Any],
    gates: Mapping[str, Any],
    samples_seen: int,
    realized_strata: Mapping[str, int],
) -> dict[str, Any]:
    if step not in CHECKPOINT_STEPS:
        raise ValueError("v10 checkpoint step is not approved")
    _validate_lineage(lineage)
    _validate_calibration(calibration)
    if type(initialization_seed) is not int or initialization_seed < 0:
        raise ValueError("v10 initialization seed is invalid")
    if topology != {"world_size": 1, "physical_gpus": [6]}:
        raise ValueError("v10 initial topology must be physical GPU6 only")
    if type(samples_seen) is not int or samples_seen <= 0 or set(realized_strata) != STRATA:
        raise ValueError("v10 samples/strata are invalid")
    if any(type(value) is not int or value < 0 for value in realized_strata.values()) or sum(realized_strata.values()) != samples_seen:
        raise ValueError("v10 realized strata do not sum to samples seen")
    if step == 500 and gates.get("step250", {}).get("pass") is not True:
        raise ValueError("v10 step500 requires a passing gated step250")
    names = v10_trainable_parameter_names(model)
    named = dict(model.named_parameters())
    model_state = {name: named[name].detach().cpu() for name in names}
    optimizer_state = optimizer.state_dict()
    _validate_optimizer(optimizer_state, step=step)
    return {
        "contract": "wan-v10-relational-checkpoint/1",
        "step": step,
        "completed_phase": "mechanism" if step <= 250 else "trajectory",
        "resume_phase": "mechanism" if step <= 250 else "complete",
        "gated": False,
        "lineage": dict(lineage),
        "calibration": dict(calibration),
        "initialization_seed": initialization_seed,
        "topology": dict(topology),
        "gates": copy.deepcopy(dict(gates)),
        "samples_seen": samples_seen,
        "realized_strata": dict(realized_strata),
        "lr_multiplier": v10_lr_multiplier(step),
        "clip_norm": CLIP_NORM,
        "trainable_names": sorted(names),
        "model": model_state,
        "optimizer": optimizer_state,
    }


def promote_v10_checkpoint(payload: Mapping[str, Any], gate_receipt: Mapping[str, Any]) -> dict[str, Any]:
    step = payload.get("step")
    if step not in CHECKPOINT_STEPS or gate_receipt.get("step") != step or gate_receipt.get("pass") is not True:
        raise ValueError("v10 checkpoint promotion requires its passing gate")
    result = copy.deepcopy(dict(payload))
    result["gated"] = True
    result["gates"][f"step{step}"] = copy.deepcopy(dict(gate_receipt))
    if step == 250:
        result["resume_phase"] = "trajectory"
    return result


def validate_v10_checkpoint(
    payload: Mapping[str, Any],
    *,
    expected_lineage: Mapping[str, str],
    expected_phase: str,
    require_gated: bool,
) -> None:
    if payload.get("contract") != "wan-v10-relational-checkpoint/1":
        raise ValueError("v10 checkpoint contract differs")
    _validate_lineage(expected_lineage)
    if payload.get("lineage") != dict(expected_lineage):
        raise ValueError("v10 checkpoint lineage differs")
    step = payload.get("step")
    if type(step) is not int or step not in CHECKPOINT_STEPS:
        raise ValueError("v10 checkpoint step differs")
    _validate_calibration(payload.get("calibration", {}))
    if payload.get("resume_phase") != expected_phase:
        if expected_phase == "trajectory" and step == 250:
            raise ValueError("v10 trajectory phase requires gated step250")
        raise ValueError("v10 checkpoint resume phase differs")
    if require_gated and payload.get("gated") is not True:
        raise ValueError("v10 continuation requires a gated checkpoint")
    if expected_phase == "trajectory" and payload.get("gates", {}).get("step250", {}).get("pass") is not True:
        raise ValueError("v10 trajectory phase requires gated step250")
    names, model = payload.get("trainable_names"), payload.get("model")
    if not isinstance(names, list) or not names or not isinstance(model, Mapping) or set(model) != set(names):
        raise ValueError("v10 checkpoint model inventory is incomplete")
    topology = payload.get("topology")
    if topology != {"world_size": 1, "physical_gpus": [6]}:
        raise ValueError("v10 checkpoint topology differs")
    strata = payload.get("realized_strata")
    if not isinstance(strata, Mapping) or set(strata) != STRATA or sum(strata.values()) != payload.get("samples_seen"):
        raise ValueError("v10 checkpoint exposure is incomplete")
    _validate_optimizer(payload.get("optimizer", {}), step=step)

