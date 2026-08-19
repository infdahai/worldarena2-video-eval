from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from worldarena_baseline.wan_v11_training import (
    V11_BASE_LRS,
    build_v11_checkpoint,
    build_v11_optimizer,
    build_v11_scheduler,
    canonical_json_sha256,
    validate_v11_checkpoint,
    v11_lr_factor,
)


_CALIBRATION = {"contract": "wan-v11-loss-calibration/1", "frozen": True}


class _Arm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.read_q = nn.Linear(2, 2)
        self.update = nn.Linear(2, 2)
        self.write_o = nn.Linear(2, 2)
        self.gate = nn.Parameter(torch.zeros(()))
        self.eef_head = nn.Linear(2, 2)


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = nn.Linear(2, 2)
        self.controller_stages = nn.ModuleDict(
            {"6": nn.ModuleDict({"left": _Arm(), "right": _Arm()})}
        )


def _lineage() -> dict[str, str]:
    lineage = {
        key: str(index) * 64
        for index, key in enumerate(
            (
                "parent_sha256",
                "source_closure_sha256",
                "replay_sha256",
                "data_manifest_sha256",
                "audit_manifest_sha256",
                "cache_sha256",
                "calibration_sha256",
            ),
            start=1,
        )
    }
    lineage["calibration_sha256"] = canonical_json_sha256(_CALIBRATION)
    return lineage


def _trained_state():
    model = _Tiny()
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    optimizer = build_v11_optimizer(model, names)
    scheduler = build_v11_scheduler(optimizer)
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        sum(parameter.square().sum() for parameter in model.parameters()).backward()
        optimizer.step()
        scheduler.step()
    return model, optimizer, scheduler, names


def test_optimizer_has_exact_six_nonempty_families() -> None:
    model = _Tiny()
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    optimizer = build_v11_optimizer(model, names)

    groups = {group["name"]: group for group in optimizer.param_groups}
    assert set(groups) == {"tokenizer", "updater", "read", "write", "gate", "eef"}
    assert all(group["params"] for group in groups.values())
    assert {name: group["initial_lr"] for name, group in groups.items()} == V11_BASE_LRS
    assert groups["gate"]["weight_decay"] == 0
    assert all(groups[name]["weight_decay"] == 0.01 for name in groups if name != "gate")


def test_lr_contract_is_warmup_then_cosine_to_twenty_percent() -> None:
    assert v11_lr_factor(0) == 0.0
    assert v11_lr_factor(25) == 0.5
    assert v11_lr_factor(50) == 1.0
    assert v11_lr_factor(2060) == pytest.approx(0.2)
    assert v11_lr_factor(1000) > v11_lr_factor(2060)


def test_checkpoint_roundtrip_binds_lineage_topology_and_optimizer() -> None:
    model, optimizer, scheduler, names = _trained_state()

    payload = build_v11_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        trainable_names=names,
        completed_step=100,
        lineage=_lineage(),
        calibration=_CALIBRATION,
    )

    validate_v11_checkpoint(
        payload,
        model=model,
        trainable_names=names,
        expected_step=100,
        expected_lineage=_lineage(),
    )
    assert payload["topology"] == {"cuda_visible_devices": "6", "world_size": 1}
    assert payload["samples_seen"] == 100


@pytest.mark.parametrize("field", ["parent_sha256", "replay_sha256", "calibration_sha256"])
def test_checkpoint_rejects_foreign_lineage(field: str) -> None:
    model, optimizer, scheduler, names = _trained_state()
    payload = build_v11_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        trainable_names=names,
        completed_step=100,
        lineage=_lineage(),
        calibration=_CALIBRATION,
    )
    expected = _lineage()
    expected[field] = "f" * 64

    with pytest.raises(ValueError, match="lineage"):
        validate_v11_checkpoint(
            payload,
            model=model,
            trainable_names=names,
            expected_step=100,
            expected_lineage=expected,
        )


def test_checkpoint_rejects_partial_adamw_state() -> None:
    model, optimizer, scheduler, names = _trained_state()
    payload = build_v11_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        trainable_names=names,
        completed_step=100,
        lineage=_lineage(),
        calibration=_CALIBRATION,
    )
    damaged = copy.deepcopy(payload)
    first = next(iter(damaged["optimizer"]["state"].values()))
    del first["exp_avg_sq"]

    with pytest.raises(ValueError, match="AdamW"):
        validate_v11_checkpoint(
            damaged,
            model=model,
            trainable_names=names,
            expected_step=100,
            expected_lineage=_lineage(),
        )
