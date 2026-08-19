from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch
from torch import Tensor
from torch.nn import functional as F


def per_arm_fm_energy(
    prediction: Tensor,
    target: Tensor,
    arm_masks: Tensor,
    valid: Tensor,
) -> Tensor:
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("prediction and target must share shape (B, C, T, H, W)")
    batch, _channels, time, height, width = prediction.shape
    if tuple(arm_masks.shape) != (batch, 2, time, height, width):
        raise ValueError("arm masks must have shape (B, 2, T, H, W)")
    if tuple(valid.shape) != (batch, 2) or valid.dtype != torch.bool:
        raise ValueError("arm energy validity must have shape (B, 2) and bool dtype")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("prediction and target must be finite")
    if not torch.isfinite(arm_masks).all() or torch.any(arm_masks < 0):
        raise ValueError("arm masks must be finite and nonnegative")

    token_error = (prediction.float() - target.float()).square().mean(dim=1)
    masks = arm_masks.float()
    denominator = masks.sum(dim=(-1, -2, -3))
    if torch.any(valid & (denominator <= 0)):
        raise ValueError("eligible arm energy has an empty mask")
    numerator = (token_error[:, None] * masks).sum(dim=(-1, -2, -3))
    energy = numerator / denominator.clamp_min(1)
    return torch.where(valid, energy, torch.zeros_like(energy))


def _binding_term(
    correct_energy: Tensor,
    wrong_energy: Tensor,
    eligible: Tensor,
    *,
    tau: float,
) -> Tensor:
    if correct_energy.shape != wrong_energy.shape or correct_energy.ndim != 2:
        raise ValueError("binding energies must share shape (B, 2)")
    if eligible.shape != correct_energy.shape or eligible.dtype != torch.bool:
        raise ValueError("binding eligibility differs from energy shape")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("binding temperature must be positive finite")
    if not torch.isfinite(correct_energy).all() or not torch.isfinite(wrong_energy).all():
        raise ValueError("binding energies must be finite")
    mask = eligible.float()
    count = mask.sum()
    if not bool(eligible.any()):
        return (correct_energy.sum() + wrong_energy.sum()) * 0
    # The 1/2 coefficient makes the two detached sequential halves exactly
    # reproduce this simultaneous objective's full parameter gradient.
    return 0.5 * (
        F.softplus((correct_energy.float() - wrong_energy.float()) / tau) * mask
    ).sum() / count


def binding_loss(
    correct_energy: Tensor,
    wrong_energy: Tensor,
    eligible: Tensor,
    *,
    tau: float = 0.1,
) -> dict[str, Tensor]:
    loss = _binding_term(correct_energy, wrong_energy, eligible, tau=tau)
    mask = eligible.float()
    count = mask.sum()
    zero = (correct_energy.sum() + wrong_energy.sum()).detach() * 0
    if bool(eligible.any()):
        margins = wrong_energy.float() - correct_energy.float()
        mean_margin = (margins * mask).sum() / count
        median_margin = margins[eligible].median()
        win_rate = ((margins > 0) & eligible).float().sum() / count
    else:
        mean_margin = median_margin = win_rate = zero
    return {
        "loss": loss,
        "eligible_count": count,
        "mean_margin": mean_margin,
        "median_margin": median_margin,
        "win_rate": win_rate,
    }


def wrong_binding_half(
    correct_reference: Tensor,
    wrong_energy: Tensor,
    eligible: Tensor,
    *,
    tau: float = 0.1,
) -> Tensor:
    if correct_reference.requires_grad:
        raise ValueError("wrong half requires a detached correct reference")
    return _binding_term(correct_reference, wrong_energy, eligible, tau=tau)


def correct_binding_half(
    correct_energy: Tensor,
    wrong_reference: Tensor,
    eligible: Tensor,
    *,
    tau: float = 0.1,
) -> Tensor:
    if wrong_reference.requires_grad:
        raise ValueError("correct half requires a detached wrong reference")
    return _binding_term(correct_energy, wrong_reference, eligible, tau=tau)


def visual_read_eef_loss(
    predictions: Mapping[int, tuple[Tensor, Tensor]],
    targets: Tensor,
    valid: Tensor,
) -> dict[str, Tensor]:
    if targets.ndim != 5 or targets.shape[2] != 2:
        raise ValueError("EEF targets must have shape (B, T, 2, H, W)")
    if valid.shape != targets.shape[:3] or valid.dtype != torch.bool:
        raise ValueError("EEF validity must have shape (B, T, 2) and bool dtype")
    if not bool(valid.any()):
        raise ValueError("visual EEF loss has no eligible labels")
    if tuple(predictions) != (6, 16, 24):
        raise ValueError("visual EEF predictions must come from stages 6, 16, and 24")
    if not torch.isfinite(targets).all() or torch.any((targets < 0) | (targets > 1)):
        raise ValueError("EEF targets must be finite values in [0, 1]")

    mask = valid[..., None, None].float()
    denominator = valid.sum() * targets.shape[-2] * targets.shape[-1]
    losses: list[Tensor] = []
    for point in (6, 16, 24):
        left, right = predictions[point]
        logits = torch.stack((left, right), dim=2)
        if logits.shape != targets.shape or not torch.isfinite(logits).all():
            raise ValueError(f"stage {point} EEF logits differ from target shape/finiteness")
        pixel = F.binary_cross_entropy_with_logits(
            logits.float(), targets.float(), reduction="none"
        )
        losses.append((pixel * mask).sum() / denominator)
    return {
        "loss": torch.stack(losses).mean(),
        "eligible_count": valid.sum(),
    }


def _gradient_norm(gradients: Sequence[Tensor | None], *, label: str) -> float:
    values = [value.detach().float().square().sum() for value in gradients if value is not None]
    if not values:
        raise ValueError(f"v11 {label} gradient norm must be positive finite")
    norm = float(torch.stack(values).sum().sqrt().cpu())
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError(f"v11 {label} gradient norm must be positive finite")
    return norm


def calibrate_v11_lambdas(
    *,
    fm_gradients: Sequence[Tensor | None],
    binding_gradients: Sequence[Tensor | None],
    eef_gradients: Sequence[Tensor | None],
) -> dict[str, object]:
    fm_norm = _gradient_norm(fm_gradients, label="fm")
    binding_norm = _gradient_norm(binding_gradients, label="binding")
    eef_norm = _gradient_norm(eef_gradients, label="eef")
    lambdas = {
        "binding": 0.5 * fm_norm / binding_norm,
        "eef": 0.2 * fm_norm / eef_norm,
    }
    if any(not 1e-4 <= value <= 1000 for value in lambdas.values()):
        raise ValueError("v11 calibrated lambda is outside [1e-4, 1000]")
    return {
        "contract": "wan-v11-loss-calibration/1",
        "fm_gradient_norm": fm_norm,
        "objective_gradient_norms": {
            "binding": binding_norm,
            "eef": eef_norm,
        },
        "target_ratios": {"binding": 0.5, "eef": 0.2},
        "lambdas": lambdas,
        "frozen": True,
    }
