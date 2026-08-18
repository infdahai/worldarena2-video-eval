"""Counterfactual objective utilities for v7.1 SE(3) geometry LoRA."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


V71_CF_TAU = 0.1


def detached_float(value: Tensor) -> float:
    """Convert a scalar tensor to Python without autograd conversion warnings."""

    if value.numel() != 1:
        raise ValueError("metric tensor must contain exactly one value")
    return float(value.detach().float().cpu().item())


def _reverse_after_anchor(value: Tensor) -> Tensor:
    if value.shape[2] < 2:
        raise ValueError("counterfactual time axis must contain at least two steps")
    return torch.cat((value[:, :, :1], value[:, :, 1:].flip(2)), dim=2)


def _shift_after_anchor(value: Tensor, *, direction: int) -> Tensor:
    if direction not in (-1, 1):
        raise ValueError("shift direction must be +1 or -1")
    if value.shape[2] < 2:
        raise ValueError("counterfactual time axis must contain at least two steps")
    anchor = value[:, :, :1]
    if direction == 1:
        shifted = torch.cat((anchor, value[:, :, :-1]), dim=2)
    else:
        shifted = torch.cat((anchor, value[:, :, 2:], value[:, :, -1:]), dim=2)
    return shifted


def geometry_only_counterfactual(
    condition: Mapping[str, Tensor],
    variant: str,
    *,
    shift_direction: int = 1,
) -> dict[str, Tensor]:
    """Perturb only SE(3) inputs while preserving the frozen raster parent.

    ``action_raster``, ``condition_support`` and ``action_present`` are kept as
    the exact same tensor objects.  This prevents the already-trained frozen
    adapter from contributing to the correct-vs-wrong energy difference.
    """

    required = {
        "action_raster",
        "condition_support",
        "action_present",
        "se3_arm_transform",
        "se3_arm_present",
    }
    if set(condition) != required:
        raise ValueError("v7.1-CF condition keys differ from the frozen contract")
    transform = condition["se3_arm_transform"]
    present = condition["se3_arm_present"]
    if transform.ndim != 5 or transform.shape[1] != 2 or transform.shape[-2:] != (4, 4):
        raise ValueError("SE(3) transform must have shape (B, 2, T, 4, 4)")
    if present.shape != transform.shape[:3]:
        raise ValueError("SE(3) presence must have shape (B, 2, T)")
    result = dict(condition)
    if variant == "reverse":
        result["se3_arm_transform"] = _reverse_after_anchor(transform)
        result["se3_arm_present"] = _reverse_after_anchor(present)
    elif variant == "shift":
        result["se3_arm_transform"] = _shift_after_anchor(
            transform, direction=shift_direction
        )
        result["se3_arm_present"] = _shift_after_anchor(
            present, direction=shift_direction
        )
    elif variant == "swap":
        left_anchor = transform[:, 0, :1]
        right_anchor = transform[:, 1, :1]
        left_relative = torch.linalg.solve(left_anchor, transform[:, 0])
        right_relative = torch.linalg.solve(right_anchor, transform[:, 1])
        result["se3_arm_transform"] = torch.stack(
            (
                left_anchor @ right_relative,
                right_anchor @ left_relative,
            ),
            dim=1,
        ).squeeze(2)
        swapped_presence = present[:, [1, 0]].clone()
        swapped_presence[:, :, 0] = present[:, :, 0]
        result["se3_arm_present"] = swapped_presence
    else:
        raise ValueError(f"unsupported geometry counterfactual: {variant}")
    return result


def negative_for_step(step: int) -> tuple[str, int]:
    """Return the deterministic reverse/shift/swap schedule for a 1-based step."""

    if type(step) is not int or step <= 0:
        raise ValueError("counterfactual step must be a positive integer")
    phase = (step - 1) % 3
    if phase == 0:
        return "reverse", 0
    if phase == 2:
        return "swap", 0
    shift_index = (step - 2) // 3
    return "shift", 1 if shift_index % 2 == 0 else -1


def support_weighted_fm_energy(
    prediction: Tensor,
    target: Tensor,
    *,
    condition_support: Tensor,
    loss_weight: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """Return per-sample unreduced FM energy on existing action support only."""

    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("prediction and target must share shape (B, C, T, H, W)")
    expected = (prediction.shape[0], 1, *prediction.shape[2:])
    if loss_weight.shape != expected or valid_mask.shape != expected:
        raise ValueError("loss_weight and valid_mask must match latent geometry")
    if (
        condition_support.ndim != 5
        or condition_support.shape[:3]
        != (prediction.shape[0], 2, prediction.shape[2])
    ):
        raise ValueError("condition_support must have shape (B, 2, T, Hs, Ws)")
    if not all(
        torch.isfinite(value).all()
        for value in (prediction, target, condition_support, loss_weight, valid_mask)
    ):
        raise ValueError("counterfactual FM inputs must be finite")
    support = condition_support.float().amax(dim=1, keepdim=True).clamp(0, 1)
    support = F.interpolate(support, size=prediction.shape[2:], mode="nearest")
    weight = (
        support
        * loss_weight.to(device=prediction.device, dtype=torch.float32)
        * valid_mask.to(device=prediction.device, dtype=torch.float32).clamp(0, 1)
    )
    denominator = weight.flatten(1).sum(dim=1) * prediction.shape[1]
    if not bool((denominator > 0).all()):
        raise ValueError("counterfactual FM requires non-empty valid action support")
    squared = (prediction.float() - target.float()).square()
    numerator = (squared * weight).flatten(1).sum(dim=1)
    return numerator / denominator


def smooth_pairwise_ranking(
    correct_energy: Tensor, wrong_energy: Tensor, *, tau: float = V71_CF_TAU
) -> Tensor:
    if correct_energy.shape != wrong_energy.shape or correct_energy.ndim != 1:
        raise ValueError("ranking energies must share shape (B,)")
    if not tau > 0:
        raise ValueError("ranking temperature must be positive")
    return F.softplus((correct_energy.float() - wrong_energy.float()) / tau).mean()


def ranking_gradient_coefficients(
    correct_energy: Tensor, wrong_energy: Tensor, *, tau: float = V71_CF_TAU
) -> tuple[Tensor, Tensor]:
    """Coefficients for exact two-forward accumulation of pairwise gradients."""

    if correct_energy.shape != wrong_energy.shape or correct_energy.ndim != 1:
        raise ValueError("ranking energies must share shape (B,)")
    if not tau > 0:
        raise ValueError("ranking temperature must be positive")
    coefficient = torch.sigmoid(
        (correct_energy.float() - wrong_energy.float()) / tau
    ) / (tau * correct_energy.numel())
    return coefficient, -coefficient


def _gradient_norm(gradients: Sequence[Tensor | None]) -> Tensor:
    values = [value.detach().float().square().sum() for value in gradients if value is not None]
    if not values:
        return torch.tensor(0.0)
    return torch.stack(values).sum().sqrt()


def calibrate_cf_lambda(
    fm_gate_gradients: Sequence[Tensor | None],
    cf_gate_gradients: Sequence[Tensor | None],
) -> float:
    """Match counterfactual gate-gradient norm to FM gate-gradient norm 1:1."""

    fm_norm = _gradient_norm(fm_gate_gradients)
    cf_norm = _gradient_norm(cf_gate_gradients)
    if not torch.isfinite(fm_norm) or not torch.isfinite(cf_norm):
        raise ValueError("gate gradient norms must be finite")
    if not bool((fm_norm > 0) & (cf_norm > 0)):
        raise ValueError("gate gradient norms must be non-zero")
    return float((fm_norm / cf_norm).cpu())
