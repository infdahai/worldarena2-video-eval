from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from worldarena_baseline.wan_v10_objective import (  # noqa: E402
    calibrate_v10_lambdas,
    counterfactual_ranking_loss,
    hidden_eef_loss,
    phase_ranking_loss,
    robot_region_energy,
    trajectory_losses,
    v10_loss_schedule,
)


def test_robot_energy_uses_only_correct_support_and_active_destination() -> None:
    target = torch.zeros(1, 2, 21, 2, 2)
    prediction = torch.zeros_like(target)
    prediction[:, :, 1, 0, 0] = 2
    prediction[:, :, 0] = 1000
    support = torch.zeros(1, 2, 21, 1, 1)
    support[:, 0, 1] = 1
    weight = torch.ones(1, 1, 21, 2, 2)
    valid = torch.ones_like(weight)
    active = torch.zeros(1, 2, 20, dtype=torch.bool)
    active[:, 0, 0] = True
    energy, eligible = robot_region_energy(
        prediction,
        target,
        correct_support=support,
        loss_weight=weight,
        valid_mask=valid,
        motion_active=active,
    )
    assert energy[0, 0].item() == pytest.approx(1.0)
    assert eligible.sum().item() == 1


def test_counterfactual_ranking_prefers_lower_correct_energy() -> None:
    correct = torch.tensor([[1.0, 4.0]], requires_grad=True)
    wrong = torch.tensor([[2.0, 1.0]], requires_grad=True)
    eligible = torch.tensor([[True, False]])
    result = counterfactual_ranking_loss(correct, wrong, eligible, tau=0.1)
    assert result["margin"].item() == pytest.approx(1.0)
    assert result["eligible_fraction"].item() == pytest.approx(0.5)
    result["loss"].backward()
    assert correct.grad is not None and wrong.grad is not None


def test_phase_ranking_uses_real_neighbors_without_boundary_padding() -> None:
    target = torch.zeros(1, 2, 5, 2)
    target[:, :, :, 0] = torch.arange(5)
    prediction = target.clone().requires_grad_(True)
    valid = torch.ones(1, 2, 5, dtype=torch.bool)
    discriminative = torch.ones_like(valid)
    result = phase_ranking_loss(
        prediction,
        target,
        valid=valid,
        motion_discriminative=discriminative,
        tau=0.1,
    )
    assert result["plus_valid_count"].item() == 8
    assert result["minus_valid_count"].item() == 8
    assert result["plus_margin"].item() > 0
    assert result["minus_margin"].item() > 0
    result["loss"].backward()
    assert prediction.grad is not None


def test_hidden_eef_loss_masks_invalid_arm_time_and_sigma() -> None:
    predictions = {
        11: torch.zeros(1, 3, 2, 1, 2, requires_grad=True),
        17: torch.zeros(1, 3, 2, 1, 2, requires_grad=True),
    }
    target = torch.zeros(1, 3, 2, 1, 2)
    target[:, 1, 0, 0, 0] = 1
    valid = torch.zeros(1, 3, 2, dtype=torch.bool)
    valid[:, 1, 0] = True
    result = hidden_eef_loss(
        predictions,
        target,
        valid=valid,
        sigma_weight=torch.ones(1),
    )
    assert result["valid_count"].item() == 1
    result["loss"].backward()
    assert predictions[11].grad is not None
    assert torch.count_nonzero(predictions[11].grad[:, 0]).item() == 0


def test_trajectory_loss_is_forbidden_in_mechanism_phase() -> None:
    predicted = torch.zeros(1, 2, 21, 2)
    kwargs = dict(
        target_position=torch.zeros_like(predicted),
        position_valid=torch.ones(1, 2, 21, dtype=torch.bool),
        target_velocity=torch.zeros_like(predicted),
        velocity_valid=torch.ones(1, 2, 21, dtype=torch.bool),
        sigma_weight=torch.ones(1),
    )
    with pytest.raises(ValueError, match="trajectory phase"):
        trajectory_losses(predicted, phase="mechanism", **kwargs)
    result = trajectory_losses(predicted, phase="trajectory", **kwargs)
    assert result["position_loss"].item() == 0
    assert result["velocity_loss"].item() == 0


def test_lambda_calibration_targets_exact_native_qkvo_ratios() -> None:
    result = calibrate_v10_lambdas(
        fm_gradients=[torch.tensor([3.0, 4.0])],
        objective_gradients={
            "cf": [torch.tensor([0.0, 2.0])],
            "phase": [torch.tensor([0.0, 5.0])],
            "hidden": [torch.tensor([0.0, 1.0])],
            "position": [torch.tensor([0.0, 2.5])],
            "velocity": [torch.tensor([0.0, 10.0])],
        },
    )
    assert result["lambdas"] == pytest.approx(
        {"cf": 0.625, "phase": 0.45, "hidden": 1.0, "position": 0.6, "velocity": 0.1}
    )
    assert result["target_ratios"] == {
        "cf": 0.25, "phase": 0.45, "hidden": 0.2, "position": 0.3, "velocity": 0.2,
    }
    with pytest.raises(ValueError, match="positive finite"):
        calibrate_v10_lambdas(
            fm_gradients=[torch.ones(1)],
            objective_gradients={name: [torch.zeros(1)] for name in result["lambdas"]},
        )


def test_loss_curriculum_stages_objectives_and_hidden_backbone_gradient() -> None:
    assert v10_loss_schedule(1) == {
        "stage": "semantics", "cf": 1.0, "hidden": 1.0,
        "hidden_backbone": 0.0, "phase": 0.0, "position": 0.0, "velocity": 0.0,
    }
    assert v10_loss_schedule(25)["hidden_backbone"] == 0.0
    assert v10_loss_schedule(50)["hidden_backbone"] == pytest.approx(0.5)
    assert v10_loss_schedule(150)["hidden_backbone"] == 1.0
    timing = v10_loss_schedule(175)
    assert timing["stage"] == "timing"
    assert timing["cf"] == pytest.approx(0.6)
    assert timing["phase"] == pytest.approx(0.5)
    assert timing["position"] == timing["velocity"] == 0.0
    assert v10_loss_schedule(200)["phase"] == 1.0
    trajectory = v10_loss_schedule(325)
    assert trajectory["stage"] == "trajectory"
    assert trajectory["cf"] == pytest.approx(0.4)
    assert trajectory["phase"] == 1.0
    assert trajectory["hidden"] == pytest.approx(0.5)
    assert trajectory["position"] == trajectory["velocity"] == pytest.approx(0.5)
    final = v10_loss_schedule(500)
    assert final["hidden"] == final["hidden_backbone"] == 0.0
    assert final["position"] == final["velocity"] == 1.0
