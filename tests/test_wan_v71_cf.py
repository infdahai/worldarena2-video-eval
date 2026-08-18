from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from worldarena_baseline.wan_v71_cf import (  # noqa: E402
    calibrate_cf_lambda,
    geometry_only_counterfactual,
    negative_for_step,
    ranking_gradient_coefficients,
    smooth_pairwise_ranking,
    support_weighted_fm_energy,
)


def _condition() -> dict[str, torch.Tensor]:
    raster = torch.arange(10 * 81 * 2 * 2, dtype=torch.float32).reshape(
        1, 10, 81, 2, 2
    )
    support = torch.ones(1, 2, 21, 1, 1)
    transform = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, 2, 21, 1, 1)
    transform[:, 0, :, 0, 3] = torch.arange(21)
    transform[:, 1, :, 1, 3] = torch.arange(21) + 100
    present = torch.ones(1, 2, 21, dtype=torch.bool)
    return {
        "action_raster": raster,
        "condition_support": support,
        "action_present": torch.ones(1),
        "se3_arm_transform": transform,
        "se3_arm_present": present,
    }


@pytest.mark.parametrize("variant", ["reverse", "shift", "swap"])
def test_counterfactual_changes_only_geometry_branch(variant: str) -> None:
    correct = _condition()
    wrong = geometry_only_counterfactual(
        correct, variant, shift_direction=-1 if variant == "shift" else 1
    )

    for name in ("action_raster", "condition_support", "action_present"):
        assert wrong[name] is correct[name]
    assert not torch.equal(wrong["se3_arm_transform"], correct["se3_arm_transform"])
    assert torch.equal(correct["se3_arm_transform"][:, :, 0], wrong["se3_arm_transform"][:, :, 0])


def test_shift_uses_both_directions_without_moving_anchor() -> None:
    correct = _condition()
    plus = geometry_only_counterfactual(correct, "shift", shift_direction=1)
    minus = geometry_only_counterfactual(correct, "shift", shift_direction=-1)

    original = correct["se3_arm_transform"]
    assert torch.equal(plus["se3_arm_transform"][:, :, 0], original[:, :, 0])
    assert torch.equal(plus["se3_arm_transform"][:, :, 1], original[:, :, 0])
    assert torch.equal(minus["se3_arm_transform"][:, :, 0], original[:, :, 0])
    assert torch.equal(minus["se3_arm_transform"][:, :, 1], original[:, :, 2])


def test_negative_schedule_cycles_and_alternates_shift_direction() -> None:
    assert [negative_for_step(step) for step in range(1, 8)] == [
        ("reverse", 0),
        ("shift", 1),
        ("swap", 0),
        ("reverse", 0),
        ("shift", -1),
        ("swap", 0),
        ("reverse", 0),
    ]


def test_support_energy_ignores_background_and_uses_existing_weight() -> None:
    target = torch.zeros(1, 2, 2, 2, 2)
    prediction = torch.zeros_like(target)
    prediction[:, :, :, 0, 0] = 2
    prediction[:, :, :, 1, 1] = 100
    support = torch.zeros(1, 2, 2, 2, 2)
    support[:, 0, :, 0, 0] = 1
    loss_weight = torch.ones(1, 1, 2, 2, 2)
    loss_weight[:, :, :, 0, 0] = 3
    valid = torch.ones_like(loss_weight)

    energy = support_weighted_fm_energy(
        prediction,
        target,
        condition_support=support,
        loss_weight=loss_weight,
        valid_mask=valid,
    )

    assert energy.shape == (1,)
    assert torch.allclose(energy, torch.tensor([4.0]))


def test_smooth_pairwise_ranking_and_manual_two_forward_coefficients_match() -> None:
    correct = torch.tensor([0.4], requires_grad=True)
    wrong = torch.tensor([0.7], requires_grad=True)
    tau = 0.2
    direct = smooth_pairwise_ranking(correct, wrong, tau=tau)
    direct.backward()
    expected_correct = correct.grad.detach().clone()
    expected_wrong = wrong.grad.detach().clone()

    correct_coefficient, wrong_coefficient = ranking_gradient_coefficients(
        correct.detach(), wrong.detach(), tau=tau
    )
    assert torch.allclose(correct_coefficient, expected_correct)
    assert torch.allclose(wrong_coefficient, expected_wrong)
    assert torch.allclose(wrong.detach() - correct.detach(), torch.tensor([0.3]))


def test_lambda_calibration_matches_gate_gradient_norms_one_to_one() -> None:
    fm = (torch.tensor([3.0, 4.0]), torch.tensor([0.0]))
    cf = (torch.tensor([0.0, 2.0]), torch.tensor([0.0]))
    value = calibrate_cf_lambda(fm, cf)
    assert value == pytest.approx(2.5)

    with pytest.raises(ValueError, match="non-zero"):
        calibrate_cf_lambda(fm, (torch.zeros(2),))
