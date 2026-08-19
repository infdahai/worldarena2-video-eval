from __future__ import annotations

from collections.abc import Mapping
from statistics import median

import torch
from torch import Tensor


def summarize_paired_arm(
    correct: Tensor,
    wrong: Tensor,
    eligible: Tensor,
) -> dict[str, dict[str, float | int]]:
    if correct.shape != wrong.shape or correct.ndim != 2 or correct.shape[1] != 2:
        raise ValueError("v11 paired audit energies must share shape (N, 2)")
    if eligible.shape != correct.shape or eligible.dtype != torch.bool:
        raise ValueError("v11 paired audit eligibility differs")
    if not torch.isfinite(correct).all() or not torch.isfinite(wrong).all():
        raise ValueError("v11 paired audit energies must be finite")
    result: dict[str, dict[str, float | int]] = {}
    for arm_index, arm_name in enumerate(("left", "right")):
        mask = eligible[:, arm_index]
        values = (wrong[:, arm_index] - correct[:, arm_index])[mask].float().tolist()
        wins = sum(value > 0 for value in values)
        result[arm_name] = {
            "eligible": len(values),
            "invalid": int(correct.shape[0] - len(values)),
            "wins": wins,
            "win_rate": wins / len(values) if values else 0.0,
            "mean_margin": sum(values) / len(values) if values else 0.0,
            "median_margin": median(values) if values else 0.0,
        }
    return result


def _require(metrics: Mapping[str, object], names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in metrics]
    if missing:
        raise ValueError(f"v11 audit metrics are missing: {missing}")


def evaluate_v11_gate(step: int, metrics: Mapping[str, object]) -> dict[str, object]:
    if step not in (100, 500, 2060):
        raise ValueError("v11 gate step must be 100, 500, or 2060")
    if step == 100:
        names = (
            "gradient_families",
            "frozen_gradients_absent",
            "fm_finite",
            "max_allocated_gib",
            "max_reserved_gib",
            "left_slot_rms",
            "right_slot_rms",
            "controller_to_parent_rms",
            "phase_plus_wins",
            "phase_minus_wins",
            "no_state_leak",
        )
        _require(metrics, names)
        gradients = metrics["gradient_families"]
        expected_families = {"tokenizer", "updater", "read", "write", "gate", "eef"}
        failures: list[str] = []
        if not isinstance(gradients, Mapping) or set(gradients) != expected_families or not all(
            value is True for value in gradients.values()
        ):
            failures.append("gradient_families")
        checks = {
            "frozen_gradients_absent": metrics["frozen_gradients_absent"] is True,
            "fm_finite": metrics["fm_finite"] is True,
            "max_allocated_gib": float(metrics["max_allocated_gib"]) < 22,
            "max_reserved_gib": float(metrics["max_reserved_gib"]) < 22,
            "left_slot_rms": float(metrics["left_slot_rms"]) > 0,
            "right_slot_rms": float(metrics["right_slot_rms"]) > 0,
            "controller_to_parent_rms": float(metrics["controller_to_parent_rms"]) <= 1,
            "phase_plus_wins": int(metrics["phase_plus_wins"]) >= 16,
            "phase_minus_wins": int(metrics["phase_minus_wins"]) >= 16,
            "no_state_leak": metrics["no_state_leak"] is True,
        }
        failures.extend(name for name, passed in checks.items() if not passed)
        return {"decision": "continue" if not failures else "stop", "failures": failures}

    if step == 500:
        names = (
            "left_win_rate",
            "right_win_rate",
            "left_mean_margin",
            "right_mean_margin",
            "left_locality_ratio",
            "right_locality_ratio",
            "phase_plus_wins",
            "phase_minus_wins",
            "routing_retention",
            "fm_regression",
            "controller_improvement",
        )
        _require(metrics, names)
        direction = {
            "left_win_rate": float(metrics["left_win_rate"]) > 0.5,
            "right_win_rate": float(metrics["right_win_rate"]) > 0.5,
            "positive_margin": max(
                float(metrics["left_mean_margin"]), float(metrics["right_mean_margin"])
            )
            > 0,
            "no_severe_drift": min(
                float(metrics["left_mean_margin"]), float(metrics["right_mean_margin"])
            )
            >= -0.05,
            "left_locality_ratio": float(metrics["left_locality_ratio"]) > 1.2,
            "right_locality_ratio": float(metrics["right_locality_ratio"]) > 1.2,
            "phase_plus_wins": int(metrics["phase_plus_wins"]) >= 18,
            "phase_minus_wins": int(metrics["phase_minus_wins"]) >= 18,
            "routing_retention": float(metrics["routing_retention"]) >= 0.9,
            "fm_regression": float(metrics["fm_regression"]) <= 0.02,
        }
        if all(direction.values()):
            return {"decision": "continue", "failures": []}
        early_stop = (
            float(metrics["left_mean_margin"]) <= 0
            and float(metrics["right_mean_margin"]) <= 0
            and float(metrics["left_locality_ratio"]) <= 1.05
            and float(metrics["right_locality_ratio"]) <= 1.05
            and float(metrics["controller_improvement"]) <= 0.01
        )
        failures = [name for name, passed in direction.items() if not passed]
        return {
            "decision": "stop" if early_stop else "continue-marginal",
            "failures": failures,
        }

    names = (
        "left_win_rate",
        "right_win_rate",
        "left_mean_margin",
        "right_mean_margin",
        "overall_swap_wins",
        "left_locality_ratio",
        "right_locality_ratio",
        "bimanual_win_rate",
        "bimanual_mean_margin",
        "phase_plus_wins",
        "phase_minus_wins",
        "routing_retention",
        "fm_regression",
        "controller_energy_improvement",
        "controller_paired_win_rate",
        "visual_eef_improvement",
        "visual_eef_paired_win_rate",
        "background_within_noise",
        "photometric_within_noise",
        "broken_rate",
    )
    _require(metrics, names)
    hard = {
        "left_win_rate": float(metrics["left_win_rate"]) >= 0.6,
        "right_win_rate": float(metrics["right_win_rate"]) >= 0.6,
        "left_mean_margin": float(metrics["left_mean_margin"]) > 0,
        "right_mean_margin": float(metrics["right_mean_margin"]) > 0,
        "overall_swap_wins": int(metrics["overall_swap_wins"]) >= 14,
        "left_locality_ratio": float(metrics["left_locality_ratio"]) >= 1.5,
        "right_locality_ratio": float(metrics["right_locality_ratio"]) >= 1.5,
        "bimanual_win_rate": float(metrics["bimanual_win_rate"]) >= 0.5,
        "bimanual_mean_margin": float(metrics["bimanual_mean_margin"]) >= 0,
        "phase_plus_wins": int(metrics["phase_plus_wins"]) >= 18,
        "phase_minus_wins": int(metrics["phase_minus_wins"]) >= 18,
        "routing_retention": float(metrics["routing_retention"]) >= 0.9,
        "fm_regression": float(metrics["fm_regression"]) <= 0.02,
        "controller_energy_improvement": float(metrics["controller_energy_improvement"]) >= 0.02,
        "controller_paired_win_rate": float(metrics["controller_paired_win_rate"]) >= 0.6,
        "visual_eef_improvement": float(metrics["visual_eef_improvement"]) >= 0.05,
        "visual_eef_paired_win_rate": float(metrics["visual_eef_paired_win_rate"]) >= 0.6,
        "background_within_noise": metrics["background_within_noise"] is True,
        "photometric_within_noise": metrics["photometric_within_noise"] is True,
        "broken_rate": float(metrics["broken_rate"]) == 0,
    }
    failures = [name for name, passed in hard.items() if not passed]
    return {"decision": "promote" if not failures else "reject", "failures": failures}
