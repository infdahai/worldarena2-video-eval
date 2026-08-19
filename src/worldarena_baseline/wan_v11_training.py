from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import re

import torch
from torch import Tensor, nn


CHECKPOINT_STEPS = (100, 500, 2060)
V11_BASE_LRS = {
    "tokenizer": 1e-4,
    "updater": 5e-5,
    "read": 5e-5,
    "write": 5e-5,
    "gate": 1e-3,
    "eef": 1e-4,
}
_LINEAGE_KEYS = {
    "parent_sha256",
    "source_closure_sha256",
    "replay_sha256",
    "data_manifest_sha256",
    "audit_manifest_sha256",
    "cache_sha256",
    "calibration_sha256",
}


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _family(name: str) -> str:
    if name.startswith("tokenizer."):
        return "tokenizer"
    if name.endswith(".gate"):
        return "gate"
    if ".eef_head." in name:
        return "eef"
    if ".update." in name:
        return "updater"
    if ".read_" in name:
        return "read"
    if ".write_" in name:
        return "write"
    raise ValueError(f"v11 trainable parameter has no optimizer family: {name}")


def build_v11_optimizer(
    model: nn.Module,
    trainable_names: set[str],
) -> torch.optim.AdamW:
    actual = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if set(actual) != trainable_names or not actual:
        raise ValueError("v11 optimizer names differ from the exact trainable whitelist")
    grouped: dict[str, list[nn.Parameter]] = {name: [] for name in V11_BASE_LRS}
    for name in sorted(actual):
        grouped[_family(name)].append(actual[name])
    if any(not parameters for parameters in grouped.values()):
        empty = [name for name, parameters in grouped.items() if not parameters]
        raise ValueError(f"v11 optimizer family is empty: {empty}")
    param_groups = []
    for name, base_lr in V11_BASE_LRS.items():
        param_groups.append(
            {
                "params": grouped[name],
                "name": name,
                "lr": base_lr,
                "initial_lr": base_lr,
                "weight_decay": 0.0 if name == "gate" else 0.01,
            }
        )
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)


def v11_lr_factor(completed_step: int) -> float:
    if type(completed_step) is not int or not 0 <= completed_step <= 2060:
        raise ValueError("v11 LR step must be in [0, 2060]")
    if completed_step <= 50:
        return completed_step / 50
    progress = (completed_step - 50) / (2060 - 50)
    return 0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_v11_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=v11_lr_factor)


def _validate_lineage(lineage: Mapping[str, str]) -> dict[str, str]:
    if set(lineage) != _LINEAGE_KEYS:
        raise ValueError("v11 lineage keys differ from the exact contract")
    result = dict(lineage)
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in result.values()):
        raise ValueError("v11 lineage values must be lowercase SHA256")
    return result


def _validate_optimizer_state(
    state: Mapping[str, object],
    *,
    completed_step: int,
) -> None:
    groups = state.get("param_groups")
    values = state.get("state")
    if not isinstance(groups, list) or not isinstance(values, Mapping):
        raise ValueError("v11 AdamW state is incomplete")
    by_name = {group.get("name"): group for group in groups if isinstance(group, Mapping)}
    if set(by_name) != set(V11_BASE_LRS):
        raise ValueError("v11 AdamW parameter groups differ")
    all_parameter_ids: list[object] = []
    factor = v11_lr_factor(completed_step)
    for name, base_lr in V11_BASE_LRS.items():
        group = by_name[name]
        parameters = group.get("params")
        if not isinstance(parameters, list) or not parameters:
            raise ValueError("v11 AdamW parameter group is empty")
        all_parameter_ids.extend(parameters)
        if float(group.get("initial_lr", -1)) != base_lr:
            raise ValueError("v11 AdamW base LR differs")
        if not math.isclose(
            float(group.get("lr", -1)),
            base_lr * factor,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError("v11 AdamW scheduled LR differs")
        expected_decay = 0.0 if name == "gate" else 0.01
        if float(group.get("weight_decay", -1)) != expected_decay:
            raise ValueError("v11 AdamW weight decay differs")
    if len(set(all_parameter_ids)) != len(all_parameter_ids) or set(values) != set(
        all_parameter_ids
    ):
        raise ValueError("v11 AdamW state does not cover every parameter exactly once")
    for parameter_id in all_parameter_ids:
        item = values[parameter_id]
        if not isinstance(item, Mapping) or not {"step", "exp_avg", "exp_avg_sq"}.issubset(item):
            raise ValueError("v11 AdamW state is missing step/exp_avg/exp_avg_sq")
        if not isinstance(item["exp_avg"], Tensor) or not isinstance(item["exp_avg_sq"], Tensor):
            raise ValueError("v11 AdamW moments must be tensors")
        if item["exp_avg"].shape != item["exp_avg_sq"].shape:
            raise ValueError("v11 AdamW moment shapes differ")
        step = int(item["step"].item() if isinstance(item["step"], Tensor) else item["step"])
        if step != completed_step:
            raise ValueError("v11 AdamW step differs from checkpoint step")


def build_v11_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    trainable_names: set[str],
    completed_step: int,
    lineage: Mapping[str, str],
    calibration: Mapping[str, object],
) -> dict[str, object]:
    if completed_step not in CHECKPOINT_STEPS:
        raise ValueError("v11 checkpoints are only allowed at 100, 500, and 2060")
    lineage_value = _validate_lineage(lineage)
    if calibration.get("contract") != "wan-v11-loss-calibration/1" or calibration.get(
        "frozen"
    ) is not True:
        raise ValueError("v11 calibration is not frozen under the expected contract")
    if canonical_json_sha256(calibration) != lineage_value["calibration_sha256"]:
        raise ValueError("v11 calibration content differs from lineage")
    named = dict(model.named_parameters())
    if trainable_names != {name for name, value in named.items() if value.requires_grad}:
        raise ValueError("v11 checkpoint trainable names differ")
    optimizer_state = optimizer.state_dict()
    _validate_optimizer_state(optimizer_state, completed_step=completed_step)
    scheduler_state = scheduler.state_dict()
    if int(scheduler_state.get("last_epoch", -1)) != completed_step:
        raise ValueError("v11 scheduler step differs from checkpoint step")
    return {
        "contract": "wan-v11-checkpoint/1",
        "completed_step": completed_step,
        "samples_seen": completed_step,
        "topology": {"cuda_visible_devices": "6", "world_size": 1},
        "lineage": lineage_value,
        "calibration": dict(calibration),
        "trainable_names": sorted(trainable_names),
        "trainable_state": {
            name: named[name].detach().cpu().clone() for name in sorted(trainable_names)
        },
        "optimizer": optimizer_state,
        "scheduler": scheduler_state,
        "global_clip_norm": 1.0,
    }


def validate_v11_checkpoint(
    payload: Mapping[str, object],
    *,
    model: nn.Module,
    trainable_names: set[str],
    expected_step: int,
    expected_lineage: Mapping[str, str],
) -> None:
    if payload.get("contract") != "wan-v11-checkpoint/1":
        raise ValueError("v11 checkpoint contract differs")
    if payload.get("completed_step") != expected_step or payload.get("samples_seen") != expected_step:
        raise ValueError("v11 checkpoint step/samples_seen differs")
    if payload.get("topology") != {"cuda_visible_devices": "6", "world_size": 1}:
        raise ValueError("v11 checkpoint topology differs")
    if payload.get("lineage") != _validate_lineage(expected_lineage):
        raise ValueError("v11 checkpoint lineage differs")
    calibration = payload.get("calibration")
    if not isinstance(calibration, Mapping) or canonical_json_sha256(calibration) != expected_lineage[
        "calibration_sha256"
    ]:
        raise ValueError("v11 checkpoint calibration lineage differs")
    if payload.get("trainable_names") != sorted(trainable_names):
        raise ValueError("v11 checkpoint trainable names differ")
    named = dict(model.named_parameters())
    state = payload.get("trainable_state")
    if not isinstance(state, Mapping) or set(state) != trainable_names:
        raise ValueError("v11 checkpoint trainable state differs")
    for name, value in state.items():
        if not isinstance(value, Tensor) or value.shape != named[name].shape:
            raise ValueError("v11 checkpoint parameter shape differs")
    optimizer_state = payload.get("optimizer")
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("v11 checkpoint AdamW state is absent")
    _validate_optimizer_state(optimizer_state, completed_step=expected_step)
    scheduler = payload.get("scheduler")
    if not isinstance(scheduler, Mapping) or scheduler.get("last_epoch") != expected_step:
        raise ValueError("v11 checkpoint scheduler state differs")
    if payload.get("global_clip_norm") != 1.0:
        raise ValueError("v11 checkpoint clip norm differs")
