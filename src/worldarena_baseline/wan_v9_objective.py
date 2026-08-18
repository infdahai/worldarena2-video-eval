"""Interval-local counterfactual objective and sequential VJP for Wan v9."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math

import torch
from torch import Tensor
from torch.nn import functional as F


NEGATIVE_CYCLE = (
    "shift+1",
    "reverse",
    "shift-1",
    "swap",
    "shift+1",
    "shift-1",
    "shift+1",
    "reverse",
    "shift-1",
    "swap",
)


def negative_family_for_step(step: int) -> str:
    if type(step) is not int or step <= 0:
        raise ValueError("v9 optimizer step must be a positive integer")
    return NEGATIVE_CYCLE[(step - 1) % len(NEGATIVE_CYCLE)]


def interval_robot_fm_energy(
    prediction: Tensor,
    target: Tensor,
    *,
    correct_support: Tensor,
    loss_weight: Tensor,
    valid_mask: Tensor,
    motion_active: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return one FM energy for each correct-action transition.

    Transition ``i`` is evaluated only at destination latent ``i+1``.  The
    support, valid mask, activity, and denominator are all from the correct
    action and therefore cannot be changed by a wrong condition.
    """
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("prediction and target must share shape (B,C,21,H,W)")
    batch, channels, times, height, width = prediction.shape
    if times != 21:
        raise ValueError("v9 interval energy requires exactly 21 latent times")
    if correct_support.ndim != 5 or correct_support.shape[:3] != (batch, 2, 21):
        raise ValueError("correct_support must have shape (B,2,21,Hs,Ws)")
    if loss_weight.shape != (batch, 1, 21, height, width):
        raise ValueError("loss_weight must match v9 latent geometry")
    if valid_mask.shape != loss_weight.shape:
        raise ValueError("valid_mask must match loss_weight")
    if motion_active.shape != (batch, 2, 20) or motion_active.dtype != torch.bool:
        raise ValueError("motion_active must have boolean shape (B,2,20)")
    if not all(torch.isfinite(value).all() for value in (prediction, target, correct_support, loss_weight, valid_mask)):
        raise ValueError("v9 interval objective inputs must be finite")
    active_support = (
        correct_support[:, :, 1:].float().clamp(0, 1)
        * motion_active[..., None, None].float()
    ).amax(dim=1, keepdim=True)
    active_support = F.interpolate(active_support, size=(20, height, width), mode="nearest")
    weights = (
        active_support
        * loss_weight[:, :, 1:].float()
        * valid_mask[:, :, 1:].float().clamp(0, 1)
    )
    denominator = weights.sum(dim=(-2, -1)).squeeze(1) * channels
    active_interval = motion_active.any(dim=1)
    eligible = active_interval & (denominator > 0)
    squared = (prediction[:, :, 1:].float() - target[:, :, 1:].float()).square()
    numerator = (squared * weights).sum(dim=(1, 3, 4))
    energy = torch.where(eligible, numerator / denominator.clamp_min(1e-12), torch.zeros_like(numerator))
    return energy, eligible


def interval_pairwise_ranking(
    correct_energy: Tensor,
    wrong_energy: Tensor,
    eligible: Tensor,
    *,
    tau: float = 0.1,
) -> tuple[Tensor, Tensor, Tensor]:
    if correct_energy.shape != wrong_energy.shape or correct_energy.ndim != 2:
        raise ValueError("interval energies must share shape (B,20)")
    if eligible.shape != correct_energy.shape or eligible.dtype != torch.bool:
        raise ValueError("ranking eligibility must be boolean and match energies")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("ranking temperature must be finite and positive")
    if not torch.isfinite(correct_energy).all() or not torch.isfinite(wrong_energy).all():
        raise ValueError("ranking energies must be finite")
    mask = eligible.float()
    count = mask.sum()
    graph_zero = (correct_energy.sum() + wrong_energy.sum()) * 0.0
    if not bool(eligible.any()):
        return graph_zero, graph_zero.detach(), mask.mean()
    difference = correct_energy.float() - wrong_energy.float()
    ranking = (F.softplus(difference / tau) * mask).sum() / count
    margin = ((wrong_energy.float() - correct_energy.float()) * mask).sum() / count
    return ranking, margin, mask.mean()


def _ranking_coefficients(
    correct_energy: Tensor,
    wrong_energy: Tensor,
    eligible: Tensor,
    *,
    tau: float,
) -> tuple[Tensor, Tensor]:
    if correct_energy.shape != wrong_energy.shape or eligible.shape != correct_energy.shape:
        raise ValueError("preview interval shapes differ")
    mask = eligible.float()
    count = mask.sum().clamp_min(1.0)
    coefficient = torch.sigmoid((correct_energy.float() - wrong_energy.float()) / tau) * mask / (tau * count)
    return coefficient.detach(), -coefficient.detach()


def sequential_pairwise_backward(
    *,
    correct_preview: Tensor,
    wrong_preview: Tensor,
    eligible: Tensor,
    correct_graph: Callable[[], tuple[Tensor, Tensor]],
    wrong_graph: Callable[[], Tensor],
    lambda_cf: float,
    tau: float,
    release_graph: Callable[[str], None],
) -> dict[str, Tensor]:
    """Accumulate exact pairwise gradients without retaining two model graphs."""
    if not math.isfinite(lambda_cf) or lambda_cf <= 0:
        raise ValueError("lambda_cf must be finite and positive")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    if correct_preview.requires_grad or wrong_preview.requires_grad:
        raise ValueError("preview energies must be detached")
    correct_coefficient, wrong_coefficient = _ranking_coefficients(
        correct_preview, wrong_preview, eligible, tau=tau
    )
    fm_loss: Tensor | None = None
    try:
        fm_loss, correct_energy = correct_graph()
        if fm_loss.ndim != 0 or correct_energy.shape != correct_preview.shape:
            raise ValueError("correct graph outputs violate the v9 objective contract")
        correct_objective = fm_loss + float(lambda_cf) * (correct_energy * correct_coefficient).sum()
        correct_objective.backward()
    finally:
        release_graph("correct")
    try:
        wrong_energy = wrong_graph()
        if wrong_energy.shape != wrong_preview.shape:
            raise ValueError("wrong graph energy violates the v9 objective contract")
        wrong_objective = float(lambda_cf) * (wrong_energy * wrong_coefficient).sum()
        wrong_objective.backward()
    finally:
        release_graph("wrong")
    ranking, margin, eligible_fraction = interval_pairwise_ranking(
        correct_preview, wrong_preview, eligible, tau=tau
    )
    assert fm_loss is not None
    return {
        "fm_loss": fm_loss.detach(),
        "ranking_loss": ranking.detach(),
        "margin": margin.detach(),
        "eligible_fraction": eligible_fraction.detach(),
    }


def _gradient_norm(gradients: Sequence[Tensor | None]) -> Tensor:
    squares = [gradient.detach().float().square().sum() for gradient in gradients if gradient is not None]
    if not squares:
        return torch.tensor(0.0)
    return torch.stack(squares).sum().sqrt()


def calibrate_lambda_cf(
    fm_gradients: Sequence[Tensor | None],
    cf_gradients: Sequence[Tensor | None],
    *,
    target_ratio: float = 0.5,
) -> float:
    if not math.isfinite(target_ratio) or target_ratio <= 0:
        raise ValueError("target gradient ratio must be finite and positive")
    fm_norm = _gradient_norm(fm_gradients)
    cf_norm = _gradient_norm(cf_gradients)
    if not torch.isfinite(fm_norm) or not torch.isfinite(cf_norm):
        raise ValueError("v9 calibration gradient norms must be finite")
    if not bool((fm_norm > 0) & (cf_norm > 0)):
        raise ValueError("v9 calibration gradient norms must be nonzero")
    return float((float(target_ratio) * fm_norm / cf_norm).cpu())

