from __future__ import annotations

import copy
import hashlib

import pytest


torch = pytest.importorskip("torch")
from torch import nn

import worldarena_baseline.wan_v10_training as training  # noqa: E402


class _ContractModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.native = nn.Parameter(torch.ones(2))
        self.relation = nn.Parameter(torch.ones(3))
        self.gate = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.hidden = nn.Parameter(torch.ones(4))


def _patch_names(monkeypatch, model):
    names = {
        "relation_wrappers.6.base.q.weight": model.native,
        "relation_encoder.network.0.weight": model.relation,
        "relation_wrappers.6.relation_gate": model.gate,
        "relation_wrappers.11.hidden_eef_head.weight": model.hidden,
    }
    monkeypatch.setattr(training, "v10_trainable_parameter_names", lambda _model: set(names))
    monkeypatch.setattr(
        model,
        "named_parameters",
        lambda recurse=True: iter(names.items()),
    )
    monkeypatch.setattr(model, "state_dict", lambda: names)
    return names


def _populate_adam(optimizer, model):
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _lineage():
    return {
        key: hashlib.sha256(key.encode("utf-8")).hexdigest()
        for key in (
            "source_closure_sha256", "config_sha256", "parent_sha256", "data_sha256",
            "cache_sha256", "replay_sha256", "audit_sha256", "probe_sha256",
            "calibration_sha256",
        )
    }


def _calibration():
    return {
        "contract": "wan-v10-loss-calibration/1",
        "curriculum_contract": "wan-v10-loss-curriculum/1",
        "target_ratios": {
            "cf": 0.25, "phase": 0.45, "hidden": 0.2,
            "position": 0.3, "velocity": 0.2,
        },
        "lambdas": {"cf": 1.0, "phase": 1.0, "hidden": 0.5, "position": 0.5, "velocity": 0.5},
        "frozen": True,
    }


def test_optimizer_has_four_exact_disjoint_parameter_groups(monkeypatch) -> None:
    model = _ContractModel()
    _patch_names(monkeypatch, model)
    optimizer = training.build_v10_optimizer(model)
    assert [group["name"] for group in optimizer.param_groups] == [
        "native_qkvo", "relation_encoders", "relation_gates", "hidden_eef_heads"
    ]
    assert [group["lr"] for group in optimizer.param_groups] == [1e-6, 1e-4, 5e-5, 1e-4]
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.01, 0.01, 0.0, 0.01]


def test_scheduler_warmup_and_cosine_end_are_exact(monkeypatch) -> None:
    model = _ContractModel()
    _patch_names(monkeypatch, model)
    optimizer = training.build_v10_optimizer(model)
    assert training.v10_lr_multiplier(1) == pytest.approx(0.02)
    assert training.v10_lr_multiplier(50) == 1.0
    assert training.v10_lr_multiplier(500) == pytest.approx(0.2)
    assert training.set_v10_learning_rates(optimizer, 50) == (1e-6, 1e-4, 5e-5, 1e-4)


def test_checkpoint_requires_complete_optimizer_and_gated_phase_parent(monkeypatch) -> None:
    model = _ContractModel()
    _patch_names(monkeypatch, model)
    optimizer = training.build_v10_optimizer(model)
    _populate_adam(optimizer, model)
    training.set_v10_learning_rates(optimizer, 300)
    payload = training.build_v10_checkpoint(
        step=300,
        model=model,
        optimizer=optimizer,
        lineage=_lineage(),
        calibration=_calibration(),
        initialization_seed=19,
        topology={"world_size": 1, "physical_gpus": [6]},
        gates={"step150": {"step": 150, "pass": True}},
        samples_seen=300,
        realized_strata={"single_dominant": 120, "bimanual_heavy": 105, "mixed": 60, "quiet": 15},
    )
    with pytest.raises(ValueError, match="gated step300"):
        training.validate_v10_checkpoint(
            payload,
            expected_lineage=_lineage(),
            expected_phase="trajectory",
            require_gated=True,
        )
    gated = training.promote_v10_checkpoint(
        payload,
        {"step": 300, "pass": True, "reasons": []},
    )
    training.validate_v10_checkpoint(
        gated,
        expected_lineage=_lineage(),
        expected_phase="trajectory",
        require_gated=True,
    )
    broken = copy.deepcopy(gated)
    broken["optimizer"]["state"] = {}
    with pytest.raises(ValueError, match="optimizer"):
        training.validate_v10_checkpoint(
            broken,
            expected_lineage=_lineage(),
            expected_phase="trajectory",
            require_gated=True,
        )
