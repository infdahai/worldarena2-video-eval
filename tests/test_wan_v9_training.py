from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from worldarena_baseline.wan_v9_model import v9_trainable_parameter_names  # noqa: E402
from worldarena_baseline.wan_v9_training import (  # noqa: E402
    ACTION_CROSS_LR,
    GATE_LR,
    NATIVE_LR,
    build_v9_checkpoint,
    build_v9_optimizer,
    set_v9_learning_rates,
    validate_v9_checkpoint,
    v9_lr_multiplier,
)


class _Fixture(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.native = torch.nn.Parameter(torch.ones(2))
        self.cross = torch.nn.Parameter(torch.ones(3))
        self.gate = torch.nn.Parameter(torch.zeros(4, dtype=torch.float32))


def _patch_whitelist(monkeypatch, model: _Fixture) -> None:
    names = {"native", "cross", "gate"}
    monkeypatch.setattr("worldarena_baseline.wan_v9_training.v9_trainable_parameter_names", lambda actual: names if actual is model else set())
    monkeypatch.setattr("worldarena_baseline.wan_v9_training.classify_v9_parameter", lambda name: {"native": "native_qkvo", "cross": "action_cross", "gate": "channel_gates"}[name])


def test_optimizer_has_three_exact_groups_and_cosine_schedule(monkeypatch) -> None:
    model = _Fixture()
    _patch_whitelist(monkeypatch, model)
    optimizer = build_v9_optimizer(model)
    assert [(group["name"], group["lr"], group["weight_decay"]) for group in optimizer.param_groups] == [
        ("native_qkvo", NATIVE_LR, 0.01),
        ("action_cross", ACTION_CROSS_LR, 0.01),
        ("channel_gates", GATE_LR, 0.0),
    ]
    assert v9_lr_multiplier(1) == pytest.approx(1 / 25)
    assert v9_lr_multiplier(25) == 1.0
    assert v9_lr_multiplier(500) == pytest.approx(0.2)
    assert set_v9_learning_rates(optimizer, 500) == pytest.approx(
        (NATIVE_LR * 0.2, ACTION_CROSS_LR * 0.2, GATE_LR * 0.2)
    )


def test_checkpoint_rejects_phase_t_without_step250_and_lineage_drift(monkeypatch) -> None:
    model = _Fixture()
    _patch_whitelist(monkeypatch, model)
    optimizer = build_v9_optimizer(model)
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    set_v9_learning_rates(optimizer, 250)
    lineage = {"parent_sha256": "a" * 64, "source_closure_sha256": "b" * 64}
    with pytest.raises(ValueError, match="step250"):
        build_v9_checkpoint(
            step=300, model=model, optimizer=optimizer, lineage=lineage,
            calibration={"lambda_cf": 1.0}, gates={"step250": {"pass": False}},
            gradient_history={"ok": 1.0}, negative_cycle_index=0,
        )
    payload = build_v9_checkpoint(
        step=250, model=model, optimizer=optimizer, lineage=lineage,
        calibration={"lambda_cf": 1.0}, gates={"step250": {"pass": True}},
        gradient_history={"ok": 1.0}, negative_cycle_index=0,
    )
    validate_v9_checkpoint(payload, expected_lineage=lineage, expected_phase="mechanism")
    with pytest.raises(ValueError, match="lineage"):
        validate_v9_checkpoint(payload, expected_lineage={**lineage, "parent_sha256": "c" * 64}, expected_phase="mechanism")
