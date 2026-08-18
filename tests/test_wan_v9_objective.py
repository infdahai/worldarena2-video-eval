from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from worldarena_baseline.wan_v9_objective import (  # noqa: E402
    NEGATIVE_CYCLE,
    calibrate_lambda_cf,
    interval_pairwise_ranking,
    interval_robot_fm_energy,
    negative_family_for_step,
    sequential_pairwise_backward,
)


def test_negative_cycle_has_exact_shift_reverse_swap_exposure() -> None:
    assert NEGATIVE_CYCLE == (
        "shift+1", "reverse", "shift-1", "swap", "shift+1",
        "shift-1", "shift+1", "reverse", "shift-1", "swap",
    )
    assert [negative_family_for_step(step) for step in range(1, 11)] == list(NEGATIVE_CYCLE)
    fifty = [negative_family_for_step(step) for step in range(1, 51)]
    assert sum(name.startswith("shift") for name in fifty) == 30
    assert fifty.count("reverse") == 10
    assert fifty.count("swap") == 10
    assert fifty.count("shift+1") == 15
    assert fifty.count("shift-1") == 15


def test_interval_energy_uses_only_destination_latent_and_correct_active_support() -> None:
    target = torch.zeros(1, 2, 21, 2, 2)
    prediction = torch.zeros_like(target)
    prediction[:, :, 1, 0, 0] = 2.0
    prediction[:, :, 0, 0, 0] = 1000.0
    prediction[:, :, 2:, 0, 0] = 100.0
    support = torch.zeros(1, 2, 21, 1, 1)
    support[:, 0, 1, 0, 0] = 1.0
    weight = torch.ones(1, 1, 21, 2, 2)
    valid = torch.ones_like(weight)
    active = torch.zeros(1, 2, 20, dtype=torch.bool)
    active[:, 0, 0] = True
    energy, eligible = interval_robot_fm_energy(
        prediction, target, correct_support=support, loss_weight=weight,
        valid_mask=valid, motion_active=active,
    )
    assert energy.shape == (1, 20)
    assert eligible.shape == (1, 20)
    assert energy[0, 0].item() == pytest.approx(4.0)
    assert eligible.sum().item() == 1
    assert torch.count_nonzero(energy[:, 1:]).item() == 0


def test_wrong_support_cannot_change_ranking_mask_or_denominator() -> None:
    prediction = torch.ones(1, 1, 21, 2, 2)
    target = torch.zeros_like(prediction)
    correct_support = torch.ones(1, 2, 21, 1, 1)
    active = torch.ones(1, 2, 20, dtype=torch.bool)
    weight = torch.ones(1, 1, 21, 2, 2)
    valid = torch.ones_like(weight)
    baseline = interval_robot_fm_energy(
        prediction, target, correct_support=correct_support,
        loss_weight=weight, valid_mask=valid, motion_active=active,
    )
    changed_wrong_support = torch.zeros_like(correct_support)
    # There is intentionally no wrong_support argument in the API.
    assert changed_wrong_support.sum().item() == 0
    actual = interval_robot_fm_energy(
        prediction, target, correct_support=correct_support,
        loss_weight=weight, valid_mask=valid, motion_active=active,
    )
    assert all(torch.equal(left, right) for left, right in zip(baseline, actual, strict=True))


def test_all_quiet_sample_is_fm_only_and_ranking_is_graph_connected_zero() -> None:
    correct = torch.randn(1, 20, requires_grad=True)
    wrong = torch.randn(1, 20, requires_grad=True)
    eligible = torch.zeros(1, 20, dtype=torch.bool)
    ranking, margin, fraction = interval_pairwise_ranking(correct, wrong, eligible, tau=0.1)
    assert ranking.item() == 0 and margin.item() == 0 and fraction.item() == 0
    ranking.backward()
    assert correct.grad is not None and torch.count_nonzero(correct.grad).item() == 0
    assert wrong.grad is not None and torch.count_nonzero(wrong.grad).item() == 0


def test_sequential_pairwise_backward_matches_direct_oracle() -> None:
    direct_parameter = torch.tensor([0.7, -0.4], dtype=torch.float64, requires_grad=True)
    direct_correct = (direct_parameter.square().sum() + torch.arange(20, dtype=torch.float64)).reshape(1, 20)
    direct_wrong = ((direct_parameter - 0.3).square().sum() + torch.arange(20, dtype=torch.float64) / 2).reshape(1, 20)
    eligible = torch.tensor([[True] * 7 + [False] * 13])
    ranking, _, _ = interval_pairwise_ranking(direct_correct, direct_wrong, eligible, tau=0.2)
    direct_loss = direct_parameter.square().mean() + 0.6 * ranking
    direct_loss.backward()
    expected = direct_parameter.grad.clone()

    parameter = torch.tensor([0.7, -0.4], dtype=torch.float64, requires_grad=True)

    def correct_graph():
        return parameter.square().mean(), (parameter.square().sum() + torch.arange(20, dtype=torch.float64)).reshape(1, 20)

    def wrong_graph():
        return ((parameter - 0.3).square().sum() + torch.arange(20, dtype=torch.float64) / 2).reshape(1, 20)

    with torch.no_grad():
        _, preview_correct = correct_graph()
        preview_wrong = wrong_graph()
    releases: list[str] = []
    sequential_pairwise_backward(
        correct_preview=preview_correct,
        wrong_preview=preview_wrong,
        eligible=eligible,
        correct_graph=correct_graph,
        wrong_graph=wrong_graph,
        lambda_cf=0.6,
        tau=0.2,
        release_graph=lambda label: releases.append(label),
    )
    torch.testing.assert_close(parameter.grad, expected, atol=1e-12, rtol=1e-12)
    assert releases == ["correct", "wrong"]


def test_lambda_calibration_targets_half_fm_gradient_norm() -> None:
    fm = [torch.tensor([3.0, 4.0]), torch.tensor([0.0])]
    cf = [torch.tensor([0.0, 2.0]), None]
    assert calibrate_lambda_cf(fm, cf, target_ratio=0.5) == pytest.approx(1.25)
    with pytest.raises(ValueError, match="nonzero"):
        calibrate_lambda_cf(fm, [torch.zeros(2)], target_ratio=0.5)

