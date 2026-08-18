"""Direct action-separation and trajectory objectives for Wan v10."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .wan_gripper_trajectory_loss import gripper_trajectory_terms
from .wan_v9_objective import interval_robot_fm_energy


CALIBRATION_TARGETS = {
    "cf": 1.0,
    "phase": 1.0,
    "hidden": 0.5,
    "position": 0.5,
    "velocity": 0.5,
}


def robot_region_energy(
    prediction: Tensor,
    target: Tensor,
    *,
    correct_support: Tensor,
    loss_weight: Tensor,
    valid_mask: Tensor,
    motion_active: Tensor,
) -> tuple[Tensor, Tensor]:
    """Use only the fixed correct-action support to score every condition."""

    return interval_robot_fm_energy(
        prediction,
        target,
        correct_support=correct_support,
        loss_weight=loss_weight,
        valid_mask=valid_mask,
        motion_active=motion_active,
    )


def counterfactual_ranking_loss(
    correct_energy: Tensor,
    wrong_energy: Tensor,
    eligible: Tensor,
    *,
    tau: float = 0.1,
) -> dict[str, Tensor]:
    if correct_energy.shape != wrong_energy.shape or correct_energy.ndim != 2:
        raise ValueError("v10 counterfactual energies must share shape (B,T)")
    if eligible.shape != correct_energy.shape or eligible.dtype != torch.bool:
        raise ValueError("v10 counterfactual eligibility shape differs")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("v10 counterfactual temperature must be positive finite")
    if not torch.isfinite(correct_energy).all() or not torch.isfinite(wrong_energy).all():
        raise ValueError("v10 counterfactual energies must be finite")
    mask = eligible.float()
    count = mask.sum()
    graph_zero = (correct_energy.sum() + wrong_energy.sum()) * 0
    if not bool(eligible.any()):
        return {
            "loss": graph_zero,
            "margin": graph_zero.detach(),
            "eligible_fraction": mask.mean(),
        }
    difference = correct_energy.float() - wrong_energy.float()
    loss = (F.softplus(difference / tau) * mask).sum() / count
    margin = ((wrong_energy.float() - correct_energy.float()) * mask).sum() / count
    return {"loss": loss, "margin": margin, "eligible_fraction": mask.mean()}


def _masked_phase_pair(
    current: Tensor,
    neighbor: Tensor,
    eligible: Tensor,
    *,
    tau: float,
) -> tuple[Tensor, Tensor, Tensor]:
    mask = eligible.float()
    count = mask.sum()
    zero = (current.sum() + neighbor.sum()) * 0
    if not bool(eligible.any()):
        return zero, zero.detach(), count
    loss = (F.softplus((current - neighbor) / tau) * mask).sum() / count
    margin = ((neighbor - current) * mask).sum() / count
    return loss, margin, count


def phase_ranking_loss(
    predicted_position: Tensor,
    target_position: Tensor,
    *,
    valid: Tensor,
    motion_discriminative: Tensor,
    tau: float = 0.1,
) -> dict[str, Tensor]:
    """Prefer the destination target over real adjacent targets, without padding."""

    if predicted_position.shape != target_position.shape or predicted_position.ndim != 4:
        raise ValueError("v10 phase positions must share shape (B,2,T,2)")
    if predicted_position.shape[1] != 2 or predicted_position.shape[-1] != 2:
        raise ValueError("v10 phase positions must contain two arms and xy")
    if valid.shape != predicted_position.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("v10 phase validity shape differs")
    if motion_discriminative.shape != valid.shape or motion_discriminative.dtype != torch.bool:
        raise ValueError("v10 phase discriminative mask shape differs")
    if predicted_position.shape[2] < 2:
        raise ValueError("v10 phase ranking needs at least two latent times")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("v10 phase temperature must be positive finite")
    if not torch.isfinite(predicted_position).all() or not torch.isfinite(target_position).all():
        raise ValueError("v10 phase positions must be finite")

    current_distance = (predicted_position.float() - target_position.float()).square().sum(dim=-1)
    plus_distance = (
        predicted_position[:, :, :-1].float() - target_position[:, :, 1:].float()
    ).square().sum(dim=-1)
    minus_distance = (
        predicted_position[:, :, 1:].float() - target_position[:, :, :-1].float()
    ).square().sum(dim=-1)
    plus_valid = (
        valid[:, :, :-1]
        & valid[:, :, 1:]
        & motion_discriminative[:, :, :-1]
    )
    minus_valid = (
        valid[:, :, 1:]
        & valid[:, :, :-1]
        & motion_discriminative[:, :, 1:]
    )
    plus_loss, plus_margin, plus_count = _masked_phase_pair(
        current_distance[:, :, :-1], plus_distance, plus_valid, tau=tau
    )
    minus_loss, minus_margin, minus_count = _masked_phase_pair(
        current_distance[:, :, 1:], minus_distance, minus_valid, tau=tau
    )
    active_terms = int(bool(plus_valid.any())) + int(bool(minus_valid.any()))
    loss = (plus_loss + minus_loss) / max(active_terms, 1)
    return {
        "loss": loss,
        "plus_loss": plus_loss,
        "minus_loss": minus_loss,
        "plus_margin": plus_margin,
        "minus_margin": minus_margin,
        "plus_valid_count": plus_count,
        "minus_valid_count": minus_count,
    }


def hidden_eef_loss(
    predictions: Mapping[int, Tensor],
    target_heatmap: Tensor,
    *,
    valid: Tensor,
    sigma_weight: Tensor,
) -> dict[str, Tensor]:
    if set(predictions) != {11, 17}:
        raise ValueError("v10 hidden EEF predictions must come from blocks 11 and 17")
    if target_heatmap.ndim != 5 or target_heatmap.shape[2] != 2:
        raise ValueError("target heatmap must have shape (B,T,2,H,W)")
    if valid.shape != target_heatmap.shape[:3] or valid.dtype != torch.bool:
        raise ValueError("hidden EEF validity shape differs")
    if sigma_weight.shape != (target_heatmap.shape[0],):
        raise ValueError("hidden EEF sigma weight must have shape (B,)")
    if not torch.isfinite(target_heatmap).all() or not torch.isfinite(sigma_weight).all():
        raise ValueError("hidden EEF targets and weights must be finite")
    if torch.any((target_heatmap < 0) | (target_heatmap > 1)) or torch.any(sigma_weight < 0):
        raise ValueError("hidden EEF targets/weights are outside their range")
    weight = valid.float() * sigma_weight.float()[:, None, None]
    denominator = weight.sum() * target_heatmap.shape[-2] * target_heatmap.shape[-1]
    losses: list[Tensor] = []
    for point in (11, 17):
        prediction = predictions[point]
        if prediction.shape != target_heatmap.shape or not torch.isfinite(prediction).all():
            raise ValueError("hidden EEF prediction shape or finiteness differs")
        per_pixel = F.binary_cross_entropy_with_logits(
            prediction.float(), target_heatmap.float(), reduction="none"
        )
        losses.append(
            (per_pixel * weight[..., None, None]).sum() / denominator.clamp_min(1)
        )
    graph_zero = sum(value.sum() for value in predictions.values()) * 0
    combined = torch.stack(losses).mean() if bool(weight.sum() > 0) else graph_zero
    return {"loss": combined, "valid_count": valid.sum(), "weighted_count": weight.sum()}


def trajectory_losses(
    predicted_position: Tensor,
    *,
    phase: str,
    target_position: Tensor,
    position_valid: Tensor,
    target_velocity: Tensor,
    velocity_valid: Tensor,
    sigma_weight: Tensor,
    huber_delta: float = 0.02,
) -> dict[str, Tensor]:
    if phase != "trajectory":
        raise ValueError("v10 position/velocity losses require trajectory phase")
    return gripper_trajectory_terms(
        predicted_position,
        target_position=target_position,
        position_valid=position_valid,
        target_velocity=target_velocity,
        velocity_valid=velocity_valid,
        sigma_weight=sigma_weight,
        huber_delta=huber_delta,
    )


def _gradient_norm(gradients: Sequence[Tensor | None], *, label: str) -> float:
    values = [value.detach().float().square().sum() for value in gradients if value is not None]
    if not values:
        raise ValueError(f"v10 {label} gradient norm must be positive finite")
    norm = float(torch.stack(values).sum().sqrt().cpu())
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError(f"v10 {label} gradient norm must be positive finite")
    return norm


def calibrate_v10_lambdas(
    *,
    fm_gradients: Sequence[Tensor | None],
    objective_gradients: Mapping[str, Sequence[Tensor | None]],
) -> dict[str, object]:
    if set(objective_gradients) != set(CALIBRATION_TARGETS):
        raise ValueError("v10 calibration objective families differ")
    fm_norm = _gradient_norm(fm_gradients, label="fm")
    objective_norms = {
        name: _gradient_norm(objective_gradients[name], label=name)
        for name in CALIBRATION_TARGETS
    }
    lambdas = {
        name: CALIBRATION_TARGETS[name] * fm_norm / objective_norms[name]
        for name in CALIBRATION_TARGETS
    }
    if any(not 1e-4 <= value <= 100 for value in lambdas.values()):
        raise ValueError("v10 calibrated lambda is outside [1e-4,100]")
    return {
        "contract": "wan-v10-loss-calibration/1",
        "fm_gradient_norm": fm_norm,
        "objective_gradient_norms": objective_norms,
        "target_ratios": dict(CALIBRATION_TARGETS),
        "lambdas": lambdas,
        "frozen": True,
    }

