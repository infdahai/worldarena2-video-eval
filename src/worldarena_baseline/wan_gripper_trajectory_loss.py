"""Differentiable gripper trajectory objective and calibration for v6."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .wan_gripper_probe import soft_argmax_2d


SIGMA_CONTRACT = "wan-action-lite-v6-sigma-reliability/1"
LAMBDA_CONTRACT = "wan-action-lite-v6-lambda-calibration/1"


def predicted_clean_latent(
    noisy: torch.Tensor,
    prediction: torch.Tensor,
    *,
    sigma: torch.Tensor,
) -> torch.Tensor:
    if noisy.shape != prediction.shape or noisy.ndim != 5:
        raise ValueError("noisy and prediction must share shape (B,C,T,H,W)")
    if sigma.ndim != 1 or sigma.shape[0] != noisy.shape[0]:
        raise ValueError("sigma must have shape (B,)")
    if not torch.isfinite(sigma).all() or not torch.all((sigma >= 0) & (sigma <= 1)):
        raise ValueError("sigma must be finite and in [0,1]")
    return noisy - sigma.to(device=noisy.device, dtype=noisy.dtype).view(
        noisy.shape[0], 1, 1, 1, 1
    ) * prediction


def freeze_probe_for_trajectory_loss(probe: nn.Module) -> nn.Module:
    return probe.requires_grad_(False).eval()


def _masked_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    sigma_weight: torch.Tensor,
    *,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape or prediction.shape[:-1] != valid.shape:
        raise ValueError("trajectory values and validity mask have incompatible shapes")
    if tuple(prediction.shape[-1:]) != (2,):
        raise ValueError("trajectory values must end in x/y coordinates")
    if sigma_weight.ndim != 1 or sigma_weight.shape[0] != prediction.shape[0]:
        raise ValueError("sigma_weight must have shape (B,)")
    if delta <= 0:
        raise ValueError("Huber delta must be positive")
    point_loss = F.huber_loss(
        prediction.float(), target.float(), reduction="none", delta=delta
    ).sum(dim=-1)
    weight = valid.float() * sigma_weight.float()[:, None, None]
    denominator = weight.flatten(1).sum(dim=1)
    eligible = denominator > 0
    per_sample = (point_loss * weight).flatten(1).sum(dim=1) / denominator.clamp_min(1)
    zero = prediction.float().sum() * 0
    loss = per_sample[eligible].mean() if bool(eligible.any()) else zero
    return loss, valid.sum()


def gripper_trajectory_terms(
    predicted_position: torch.Tensor,
    *,
    target_position: torch.Tensor,
    position_valid: torch.Tensor,
    target_velocity: torch.Tensor,
    velocity_valid: torch.Tensor,
    sigma_weight: torch.Tensor,
    huber_delta: float = 0.02,
) -> dict[str, torch.Tensor]:
    if predicted_position.ndim != 4 or tuple(predicted_position.shape[1:]) != (
        2,
        21,
        2,
    ):
        raise ValueError("predicted position must have shape (B,2,21,2)")
    predicted_velocity = torch.zeros_like(predicted_position)
    predicted_velocity[:, :, 1:] = (
        predicted_position[:, :, 1:] - predicted_position[:, :, :-1]
    )
    position_loss, position_count = _masked_huber(
        predicted_position,
        target_position,
        position_valid,
        sigma_weight,
        delta=huber_delta,
    )
    velocity_loss, velocity_count = _masked_huber(
        predicted_velocity,
        target_velocity,
        velocity_valid,
        sigma_weight,
        delta=huber_delta,
    )
    return {
        "position_loss": position_loss,
        "velocity_loss": velocity_loss,
        "position_valid_count": position_count,
        "velocity_valid_count": velocity_count,
        "predicted_velocity": predicted_velocity,
    }


def probe_trajectory_terms(
    predicted_clean: torch.Tensor,
    probe: nn.Module,
    **targets,
) -> dict[str, torch.Tensor]:
    logits = probe(predicted_clean)
    positions = soft_argmax_2d(logits)
    return {
        **gripper_trajectory_terms(positions, **targets),
        "predicted_position": positions,
    }


def calibrate_sigma_reliability(
    buckets: Iterable[Mapping[str, Any]],
    *,
    median_limit_px: float = 1.5,
    p90_limit_px: float = 4.0,
) -> dict[str, Any]:
    values = [dict(bucket) for bucket in buckets]
    if not values:
        raise ValueError("sigma calibration buckets cannot be empty")
    weights: list[float] = []
    previous_upper = 0.0
    normalized: list[dict[str, float]] = []
    for index, bucket in enumerate(values):
        try:
            lower = float(bucket["lower"])
            upper = float(bucket["upper"])
            median = float(bucket["median_error_px"])
            p90 = float(bucket["p90_error_px"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("sigma calibration bucket is incomplete") from exc
        if not all(math.isfinite(value) for value in (lower, upper, median, p90)):
            raise ValueError("sigma calibration values must be finite")
        if lower != previous_upper or not lower < upper <= 1:
            raise ValueError("sigma calibration buckets must exactly partition [0,1]")
        previous_upper = upper
        normalized.append(
            {
                "lower": lower,
                "upper": upper,
                "median_error_px": median,
                "p90_error_px": p90,
            }
        )
        weights.append(float(median <= median_limit_px and p90 <= p90_limit_px))
    if previous_upper != 1.0 or not any(weights):
        raise ValueError("sigma calibration has no reliable complete partition")
    return {
        "contract": SIGMA_CONTRACT,
        "median_limit_px": median_limit_px,
        "p90_limit_px": p90_limit_px,
        "buckets": normalized,
        "weights": weights,
        "frozen": True,
    }


def sigma_weights(sigma: torch.Tensor, calibration: Mapping[str, Any]) -> torch.Tensor:
    if calibration.get("contract") != SIGMA_CONTRACT or calibration.get("frozen") is not True:
        raise ValueError("invalid or mutable sigma calibration")
    result = torch.zeros_like(sigma, dtype=torch.float32)
    for bucket, weight in zip(calibration.get("buckets", []), calibration.get("weights", [])):
        lower, upper = float(bucket["lower"]), float(bucket["upper"])
        selected = (sigma >= lower) & (sigma < upper if upper < 1 else sigma <= upper)
        result[selected] = float(weight)
    return result


def _gradient_norm(loss: torch.Tensor, parameters: tuple[nn.Parameter, ...]) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=False
    )
    return torch.sqrt(
        sum(gradient.float().square().sum() for gradient in gradients)
    )


def b_only_gradient_norm_from_grads(correction) -> torch.Tensor:
    """Read a norm after a regular backward, which is safe for FSDP models."""
    parameters = tuple(correction.zero_projection_parameters())
    if not parameters or any(parameter.grad is None for parameter in parameters):
        raise ValueError("all B parameters must have gradients")
    return torch.sqrt(
        sum(parameter.grad.float().square().sum() for parameter in parameters)
    )


def calibrate_b_only_lambdas_from_norms(
    *,
    fm_norm: torch.Tensor,
    position_norm: torch.Tensor,
    velocity_norm: torch.Tensor,
    position_ratio: float = 0.4,
    velocity_ratio: float = 0.2,
) -> dict[str, Any]:
    """Build immutable trajectory lambdas from independently measured B norms."""
    if position_ratio <= 0 or velocity_ratio <= 0:
        raise ValueError("gradient target ratios must be positive")
    norms = (fm_norm, position_norm, velocity_norm)
    if any(value.numel() != 1 or not torch.isfinite(value) or value <= 0 for value in norms):
        raise ValueError("B-only calibration gradients must be positive finite scalars")
    lambda_position = position_ratio * fm_norm / position_norm
    lambda_velocity = velocity_ratio * fm_norm / velocity_norm
    if any(
        not math.isfinite(float(value.detach()))
        or not 1e-6 <= float(value.detach()) <= 1e6
        for value in (lambda_position, lambda_velocity)
    ):
        raise ValueError("calibrated lambda is outside the safe contract")
    return {
        "contract": LAMBDA_CONTRACT,
        "parameter_family": "B",
        "fm_gradient_norm": float(fm_norm.detach()),
        "position_gradient_norm": float(position_norm.detach()),
        "velocity_gradient_norm": float(velocity_norm.detach()),
        "lambda_position": float(lambda_position.detach()),
        "lambda_velocity": float(lambda_velocity.detach()),
        "scaled_position_to_fm": float(
            (lambda_position * position_norm / fm_norm).detach()
        ),
        "scaled_velocity_to_fm": float(
            (lambda_velocity * velocity_norm / fm_norm).detach()
        ),
        "frozen": True,
    }


def calibrate_b_only_lambdas(
    correction,
    *,
    fm_loss: torch.Tensor,
    position_loss: torch.Tensor,
    velocity_loss: torch.Tensor,
    position_ratio: float = 0.4,
    velocity_ratio: float = 0.2,
) -> dict[str, Any]:
    parameters = tuple(correction.zero_projection_parameters())
    if not parameters:
        raise ValueError("correction exposes no B parameters")
    fm_norm = _gradient_norm(fm_loss, parameters)
    position_norm = _gradient_norm(position_loss, parameters)
    velocity_norm = _gradient_norm(velocity_loss, parameters)
    return calibrate_b_only_lambdas_from_norms(
        fm_norm=fm_norm,
        position_norm=position_norm,
        velocity_norm=velocity_norm,
        position_ratio=position_ratio,
        velocity_ratio=velocity_ratio,
    )
