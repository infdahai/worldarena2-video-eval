from __future__ import annotations

import pytest
import torch

from worldarena_baseline.wan_v11_audit import evaluate_v11_gate, summarize_paired_arm


def test_paired_summary_reports_each_arm_without_cancellation() -> None:
    correct = torch.tensor([[1.0, 4.0], [3.0, 2.0]])
    wrong = torch.tensor([[2.0, 1.0], [2.0, 3.0]])
    eligible = torch.tensor([[True, True], [True, False]])

    summary = summarize_paired_arm(correct, wrong, eligible)

    assert summary["left"]["eligible"] == 2
    assert summary["left"]["wins"] == 1
    assert summary["right"]["eligible"] == 1
    assert summary["right"]["wins"] == 0


def _health() -> dict[str, object]:
    return {
        "gradient_families": {
            name: True for name in ("tokenizer", "updater", "read", "write", "gate", "eef")
        },
        "frozen_gradients_absent": True,
        "fm_finite": True,
        "max_allocated_gib": 21.9,
        "max_reserved_gib": 21.9,
        "left_slot_rms": 0.1,
        "right_slot_rms": 0.1,
        "controller_to_parent_rms": 0.5,
        "phase_plus_wins": 18,
        "phase_minus_wins": 18,
        "no_state_leak": True,
    }


def _direction() -> dict[str, object]:
    return {
        "left_win_rate": 0.55,
        "right_win_rate": 0.55,
        "left_mean_margin": 0.1,
        "right_mean_margin": 0.0,
        "left_locality_ratio": 1.21,
        "right_locality_ratio": 1.21,
        "phase_plus_wins": 18,
        "phase_minus_wins": 18,
        "routing_retention": 0.9,
        "fm_regression": 0.02,
        "controller_improvement": 0.02,
    }


def _hard() -> dict[str, object]:
    return {
        "left_win_rate": 0.6,
        "right_win_rate": 0.6,
        "left_mean_margin": 0.01,
        "right_mean_margin": 0.01,
        "overall_swap_wins": 14,
        "left_locality_ratio": 1.5,
        "right_locality_ratio": 1.5,
        "bimanual_win_rate": 0.5,
        "bimanual_mean_margin": 0.0,
        "phase_plus_wins": 18,
        "phase_minus_wins": 18,
        "routing_retention": 0.9,
        "fm_regression": 0.02,
        "controller_energy_improvement": 0.02,
        "controller_paired_win_rate": 0.6,
        "visual_eef_improvement": 0.05,
        "visual_eef_paired_win_rate": 0.6,
        "background_within_noise": True,
        "photometric_within_noise": True,
        "broken_rate": 0.0,
    }


def test_step100_health_gate_passes_boundary_and_one_failure_blocks() -> None:
    assert evaluate_v11_gate(100, _health())["decision"] == "continue"
    failed = _health()
    failed["max_reserved_gib"] = 22.0
    result = evaluate_v11_gate(100, failed)
    assert result["decision"] == "stop"
    assert "max_reserved_gib" in result["failures"]


def test_step500_direction_gate_and_conjunctive_early_stop() -> None:
    assert evaluate_v11_gate(500, _direction())["decision"] == "continue"
    marginal = _direction()
    marginal["left_win_rate"] = 0.5
    assert evaluate_v11_gate(500, marginal)["decision"] == "continue-marginal"
    stop = _direction()
    stop.update(
        left_mean_margin=0.0,
        right_mean_margin=0.0,
        left_locality_ratio=1.05,
        right_locality_ratio=1.05,
        controller_improvement=0.01,
    )
    assert evaluate_v11_gate(500, stop)["decision"] == "stop"


def test_step2060_requires_every_hard_gate() -> None:
    assert evaluate_v11_gate(2060, _hard())["decision"] == "promote"
    for key in (
        "left_win_rate",
        "right_locality_ratio",
        "routing_retention",
        "visual_eef_improvement",
        "background_within_noise",
        "broken_rate",
    ):
        failed = _hard()
        if isinstance(failed[key], bool):
            failed[key] = False
        elif key == "broken_rate":
            failed[key] = 0.01
        else:
            failed[key] = float(failed[key]) - 0.001
        result = evaluate_v11_gate(2060, failed)
        assert result["decision"] == "reject"
        assert key in result["failures"]


def test_gate_rejects_unknown_step() -> None:
    with pytest.raises(ValueError, match="100, 500, or 2060"):
        evaluate_v11_gate(25, {})
