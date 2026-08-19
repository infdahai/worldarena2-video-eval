from __future__ import annotations

import pytest
import torch

from worldarena_baseline.wan_v11_objective import (
    binding_loss,
    calibrate_v11_lambdas,
    correct_binding_half,
    per_arm_fm_energy,
    visual_read_eef_loss,
    wrong_binding_half,
)


def test_per_arm_energy_keeps_arms_separate_and_uses_fixed_masks() -> None:
    prediction = torch.zeros(1, 1, 2, 2, 2)
    target = torch.zeros_like(prediction)
    prediction[0, 0, 0, 0, 0] = 2
    prediction[0, 0, 1, 1, 1] = 4
    masks = torch.zeros(1, 2, 2, 2, 2)
    masks[0, 0, 0, 0, 0] = 1
    masks[0, 1, 1, 1, 1] = 1
    valid = torch.ones(1, 2, dtype=torch.bool)

    energy = per_arm_fm_energy(prediction, target, masks, valid)

    assert energy.tolist() == [[4.0, 16.0]]


def test_per_arm_energy_fails_closed_on_empty_eligible_mask() -> None:
    with pytest.raises(ValueError, match="empty mask"):
        per_arm_fm_energy(
            torch.zeros(1, 2, 2, 2, 2),
            torch.zeros(1, 2, 2, 2, 2),
            torch.zeros(1, 2, 2, 2, 2),
            torch.ones(1, 2, dtype=torch.bool),
        )


def test_binding_loss_excludes_invalid_arm_events() -> None:
    correct = torch.tensor([[1.0, 100.0]])
    wrong = torch.tensor([[2.0, 0.0]])
    eligible = torch.tensor([[True, False]])

    result = binding_loss(correct, wrong, eligible, tau=0.5)

    expected = 0.5 * torch.nn.functional.softplus(torch.tensor(-2.0))
    assert torch.allclose(result["loss"], expected)
    assert result["eligible_count"].item() == 1
    assert result["mean_margin"].item() == 1.0


def test_visual_eef_loss_uses_only_visual_read_logits_and_valid_labels() -> None:
    logits = {
        6: (torch.zeros(1, 21, 2, 3), torch.zeros(1, 21, 2, 3)),
        16: (torch.zeros(1, 21, 2, 3), torch.zeros(1, 21, 2, 3)),
        24: (torch.zeros(1, 21, 2, 3), torch.zeros(1, 21, 2, 3)),
    }
    target = torch.zeros(1, 21, 2, 2, 3)
    valid = torch.zeros(1, 21, 2, dtype=torch.bool)
    valid[0, 4, 1] = True

    result = visual_read_eef_loss(logits, target, valid)

    assert torch.allclose(result["loss"], torch.log(torch.tensor(2.0)))
    assert result["eligible_count"].item() == 1


def test_visual_eef_loss_rejects_no_labels() -> None:
    logits = {6: (torch.zeros(1, 21, 2, 3), torch.zeros(1, 21, 2, 3))}
    with pytest.raises(ValueError, match="no eligible"):
        visual_read_eef_loss(
            logits,
            torch.zeros(1, 21, 2, 2, 3),
            torch.zeros(1, 21, 2, dtype=torch.bool),
        )


def test_sequential_binding_gradients_match_simultaneous_oracle() -> None:
    simultaneous_parameter = torch.tensor(0.3, requires_grad=True)
    correct = (simultaneous_parameter - 1).square().reshape(1, 1)
    wrong = (simultaneous_parameter + 2).square().reshape(1, 1)
    binding_loss(correct, wrong, torch.ones(1, 1, dtype=torch.bool), tau=0.7)[
        "loss"
    ].backward()
    expected = simultaneous_parameter.grad.detach().clone()

    sequential_parameter = torch.tensor(0.3, requires_grad=True)
    correct_ref = (sequential_parameter.detach() - 1).square().reshape(1, 1)
    wrong_grad = (sequential_parameter + 2).square().reshape(1, 1)
    wrong_binding_half(
        correct_ref,
        wrong_grad,
        torch.ones(1, 1, dtype=torch.bool),
        tau=0.7,
    ).backward()
    wrong_ref = wrong_grad.detach()
    correct_grad = (sequential_parameter - 1).square().reshape(1, 1)
    correct_binding_half(
        correct_grad,
        wrong_ref,
        torch.ones(1, 1, dtype=torch.bool),
        tau=0.7,
    ).backward()

    assert torch.allclose(sequential_parameter.grad, expected, atol=1e-6, rtol=1e-6)


def test_calibration_emits_frozen_contract_and_rejects_zero_gradient() -> None:
    result = calibrate_v11_lambdas(
        fm_gradients=[torch.tensor([3.0, 4.0])],
        binding_gradients=[torch.tensor([1.0])],
        eef_gradients=[torch.tensor([2.0])],
    )

    assert result["contract"] == "wan-v11-loss-calibration/1"
    assert result["lambdas"] == {"binding": 2.5, "eef": 0.5}
    assert result["frozen"] is True
    with pytest.raises(ValueError, match="positive finite"):
        calibrate_v11_lambdas(
            fm_gradients=[torch.tensor([1.0])],
            binding_gradients=[torch.tensor([0.0])],
            eef_gradients=[torch.tensor([1.0])],
        )
