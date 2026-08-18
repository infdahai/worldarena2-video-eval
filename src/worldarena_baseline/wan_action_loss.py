from __future__ import annotations

from enum import IntEnum

import torch
import torch.nn.functional as F

from .action_condition import causal_time_groups


class TemporalRole(IntEnum):
    """Per-latent-token arm scheduling contract derived from commanded action."""

    QUIET = 0
    LEFT_ONLY = 1
    RIGHT_ONLY = 2
    BOTH = 3
    AMBIGUOUS = 4


def _causal_group_max(values: torch.Tensor) -> torch.Tensor:
    if values.ndim < 2 or values.shape[1] != 81:
        raise ValueError("causal grouping requires an 81-frame tensor")
    return torch.stack(
        [values[:, list(group)].amax(dim=1) for group in causal_time_groups()],
        dim=1,
    )


def _arm_motion_score(raster: torch.Tensor, *, offset: int) -> torch.Tensor:
    flow = torch.linalg.vector_norm(
        raster[:, [offset + 2, offset + 3]].float(), dim=1
    ).amax(dim=(-2, -1))
    eef = raster[:, offset + 1].float()
    eef_change = torch.zeros_like(flow)
    eef_change[:, 1:] = (eef[:, 1:] - eef[:, :-1]).abs().amax(dim=(-2, -1))
    return _causal_group_max(flow + eef_change)


def _temporal_window_max(values: torch.Tensor, *, window: int) -> torch.Tensor:
    if window <= 0 or window % 2 == 0:
        raise ValueError("temporal_window must be a positive odd integer")
    if window == 1:
        return values
    return F.max_pool1d(
        values[:, None], kernel_size=window, stride=1, padding=window // 2
    )[:, 0]


def _arm_region(raster: torch.Tensor, *, offset: int) -> torch.Tensor:
    spatial = raster[:, [offset, offset + 1]].float().amax(dim=1).clamp(0, 1)
    grouped = _causal_group_max(spatial)
    batch, frames, height, width = grouped.shape
    pooled = F.max_pool2d(
        grouped.reshape(batch * frames, 1, height, width), kernel_size=2, stride=2
    )
    return pooled.reshape(batch, 1, frames, height // 2, width // 2)


def temporal_arm_role_masks(
    raster: torch.Tensor,
    *,
    active_threshold: float = 0.2,
    quiet_threshold: float = 0.05,
    temporal_window: int = 3,
) -> dict[str, torch.Tensor]:
    """Derive a five-state role schedule from commanded per-arm motion.

    Only unambiguous single-arm windows produce auxiliary loss masks. The arm
    support maps themselves remain non-exclusive; overlap is excluded only
    from the inactive-arm penalty to avoid contradictory supervision.
    """
    if raster.ndim != 5 or tuple(raster.shape[1:]) != (10, 81, 60, 80):
        raise ValueError("v3 action raster must have shape (B, 10, 81, 60, 80)")
    if not 0 <= quiet_threshold < active_threshold:
        raise ValueError("role thresholds must satisfy 0 <= quiet < active")
    if temporal_window <= 0 or temporal_window % 2 == 0:
        raise ValueError("temporal_window must be a positive odd integer")
    if not torch.isfinite(raster).all():
        raise ValueError("action raster must be finite")

    left_score = _temporal_window_max(
        _arm_motion_score(raster, offset=0), window=temporal_window
    )
    right_score = _temporal_window_max(
        _arm_motion_score(raster, offset=5), window=temporal_window
    )
    left_high = left_score >= active_threshold
    right_high = right_score >= active_threshold
    left_low = left_score <= quiet_threshold
    right_low = right_score <= quiet_threshold
    left_only = left_high & right_low
    right_only = right_high & left_low
    both = left_high & right_high
    quiet = left_low & right_low

    role = torch.full_like(left_score, int(TemporalRole.AMBIGUOUS), dtype=torch.long)
    role[left_only] = int(TemporalRole.LEFT_ONLY)
    role[right_only] = int(TemporalRole.RIGHT_ONLY)
    role[both] = int(TemporalRole.BOTH)
    role[quiet] = int(TemporalRole.QUIET)
    # latent0 is the conditioned first frame and is never an auxiliary target.
    role[:, 0] = int(TemporalRole.QUIET)
    left_only[:, 0] = False
    right_only[:, 0] = False

    left_region = _arm_region(raster, offset=0)
    right_region = _arm_region(raster, offset=5)
    overlap = torch.minimum(left_region, right_region)
    left_present = left_region > 0
    right_present = right_region > 0
    left_active = left_only[:, None, :, None, None].to(raster.dtype)
    right_active = right_only[:, None, :, None, None].to(raster.dtype)
    return {
        "role": role,
        "left_active": left_active,
        "right_active": right_active,
        "active_region": left_active * left_region + right_active * right_region,
        "quiet_region": (
            left_active * right_region * (~left_present).to(raster.dtype)
            + right_active * left_region * (~right_present).to(raster.dtype)
        ),
        "support_overlap": overlap,
        "left_score": left_score,
        "right_score": right_score,
    }


def _masked_mse_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    expanded = mask.float().expand_as(prediction)
    denominator = expanded.flatten(1).sum(dim=1)
    numerator = ((prediction.float() - target.float()).square() * expanded).flatten(1).sum(dim=1)
    eligible = denominator > 0
    return numerator / denominator.clamp_min(1), eligible


def temporal_identity_objective(
    correct_prediction: torch.Tensor,
    swap_prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    raster: torch.Tensor,
    valid_mask: torch.Tensor,
    margin: float = 0.1,
    active_threshold: float = 0.2,
    quiet_threshold: float = 0.05,
    temporal_window: int = 3,
) -> dict[str, torch.Tensor]:
    """Compute swap ranking and inactive-arm FM on unambiguous role tokens."""
    if correct_prediction.shape != target.shape or swap_prediction.shape != target.shape:
        raise ValueError("correct, swap, and target predictions must have the same shape")
    if correct_prediction.ndim != 5 or tuple(correct_prediction.shape[2:]) != (21, 30, 40):
        raise ValueError("temporal identity objective requires v3 latent geometry")
    if valid_mask.shape != (correct_prediction.shape[0], 1, 21, 30, 40):
        raise ValueError("valid_mask must have shape (B, 1, 21, 30, 40)")
    if margin < 0:
        raise ValueError("identity margin must be non-negative")
    roles = temporal_arm_role_masks(
        raster,
        active_threshold=active_threshold,
        quiet_threshold=quiet_threshold,
        temporal_window=temporal_window,
    )
    valid = valid_mask.to(device=correct_prediction.device, dtype=torch.float32).clamp(0, 1)
    active_mask = roles["active_region"].to(correct_prediction.device) * valid
    quiet_mask = roles["quiet_region"].to(correct_prediction.device) * valid
    correct_active, eligible = _masked_mse_per_sample(
        correct_prediction, target, active_mask
    )
    swap_active, swap_eligible = _masked_mse_per_sample(
        swap_prediction, target, active_mask
    )
    if not torch.equal(eligible, swap_eligible):
        raise RuntimeError("correct and swap eligibility mismatch")
    quiet_error, quiet_eligible = _masked_mse_per_sample(
        correct_prediction, target, quiet_mask
    )

    zero = correct_prediction.float().sum() * 0
    eligible_float = eligible.float()
    eligible_count = eligible_float.sum()
    ranking = (
        (F.relu(margin + correct_active - swap_active) * eligible_float).sum()
        / eligible_count.clamp_min(1)
        if bool(eligible.any())
        else zero
    )
    quiet_float = quiet_eligible.float()
    quiet_count = quiet_float.sum()
    quiet_loss = (
        (quiet_error * quiet_float).sum() / quiet_count.clamp_min(1)
        if bool(quiet_eligible.any())
        else zero
    )
    correct_mean = (
        (correct_active * eligible_float).sum() / eligible_count.clamp_min(1)
        if bool(eligible.any())
        else zero
    )
    swap_mean = (
        (swap_active * eligible_float).sum() / eligible_count.clamp_min(1)
        if bool(eligible.any())
        else zero.detach()
    )
    return {
        "ranking_loss": ranking,
        "quiet_loss": quiet_loss,
        "correct_active_error": correct_mean,
        "swap_active_error": swap_mean,
        "inactive_arm_error_ratio": quiet_loss.detach() / correct_mean.detach().clamp_min(1e-6),
        "eligible_fraction": eligible_float.mean(),
        "left_only_fraction": (roles["role"] == int(TemporalRole.LEFT_ONLY)).float().mean(),
        "right_only_fraction": (roles["role"] == int(TemporalRole.RIGHT_ONLY)).float().mean(),
        "both_fraction": (roles["role"] == int(TemporalRole.BOTH)).float().mean(),
        "quiet_fraction": (roles["role"] == int(TemporalRole.QUIET)).float().mean(),
        "ambiguous_fraction": (
            roles["role"] == int(TemporalRole.AMBIGUOUS)
        ).float().mean(),
        "support_overlap_fraction": roles["support_overlap"].float().mean(),
    }


def swap_left_right_action_condition(
    condition: dict[str, torch.Tensor | None],
) -> dict[str, torch.Tensor | None]:
    """Swap all arm-specific condition fields while preserving shared inputs."""
    raster = condition["action_raster"]
    support = condition["condition_support"]
    if raster is None or raster.ndim != 5 or raster.shape[1] != 10:
        raise ValueError("action_raster must have ten v3 channels")
    if support is None or support.ndim != 5 or support.shape[1] != 2:
        raise ValueError("condition_support must have two arm channels")
    return {
        **condition,
        "action_raster": torch.cat((raster[:, 5:], raster[:, :5]), dim=1),
        "left_pose": condition.get("right_pose"),
        "right_pose": condition.get("left_pose"),
        "condition_support": support[:, [1, 0]],
    }


def flow_matching_sample(
    latent: torch.Tensor,
    timestep: torch.Tensor,
    *,
    noise: torch.Tensor | None = None,
    max_timestep: float = 1000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if latent.ndim != 5:
        raise ValueError("latent must have shape (B, C, T, H, W)")
    if timestep.ndim != 1 or timestep.shape[0] != latent.shape[0]:
        raise ValueError("timestep must have shape (B,)")
    if max_timestep <= 0:
        raise ValueError("max_timestep must be positive")
    if noise is None:
        noise = torch.randn_like(latent)
    if noise.shape != latent.shape:
        raise ValueError("noise and latent must have the same shape")
    sigma = (
        timestep.to(device=latent.device, dtype=latent.dtype)
        .div(max_timestep)
        .clamp(0, 1)
        .view(latent.shape[0], 1, 1, 1, 1)
    )
    noisy = (1 - sigma) * latent + sigma * noise
    target = noise - latent
    return noisy, target


def ti2v_flow_matching_sample(
    latent: torch.Tensor,
    timestep: torch.Tensor,
    *,
    noise: torch.Tensor | None = None,
    max_timestep: float = 1000.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if latent.ndim != 5 or latent.shape[2] < 2:
        raise ValueError("TI2V latent must have shape (B, C, T>=2, H, W)")
    if latent.shape[3] % 2 or latent.shape[4] % 2:
        raise ValueError("TI2V latent height and width must be divisible by 2")
    noisy, target = flow_matching_sample(
        latent,
        timestep,
        noise=noise,
        max_timestep=max_timestep,
    )
    valid_mask = latent.new_ones(
        latent.shape[0],
        1,
        latent.shape[2],
        latent.shape[3],
        latent.shape[4],
    )
    valid_mask[:, :, 0] = 0
    noisy = valid_mask * noisy + (1 - valid_mask) * latent
    token_mask = valid_mask[:, 0, :, ::2, ::2].flatten(1)
    token_timestep = token_mask * timestep.to(
        device=latent.device,
        dtype=latent.dtype,
    ).unsqueeze(1)
    return noisy, target, token_timestep, valid_mask


def action_focus_mask(
    raster: torch.Tensor,
    *,
    latent_size: tuple[int, int, int],
) -> torch.Tensor:
    if raster.ndim != 5 or raster.shape[1] != 8:
        raise ValueError("action raster must have shape (B, 8, T, H, W)")
    if any(size <= 0 for size in latent_size):
        raise ValueError("latent_size values must be positive")
    activity = raster.abs().amax(dim=1, keepdim=True).clamp(0, 1)
    return F.adaptive_max_pool3d(activity, output_size=latent_size)


def action_weight_map(
    raster: torch.Tensor,
    *,
    latent_size: tuple[int, int, int],
    trajectory_weight: float = 1.0,
    gripper_weight: float = 2.0,
) -> torch.Tensor:
    if raster.ndim != 5 or raster.shape[1] != 8:
        raise ValueError("action raster must have shape (B, 8, T, H, W)")
    if any(size <= 0 for size in latent_size):
        raise ValueError("latent_size values must be positive")
    if trajectory_weight < 0 or gripper_weight < 0:
        raise ValueError("action weights must be non-negative")
    trajectory = raster[:, :6].abs().amax(dim=1, keepdim=True).clamp(0, 1)
    gripper = raster[:, [0, 1, 6, 7]].abs().amax(dim=1, keepdim=True).clamp(0, 1)
    trajectory = F.adaptive_max_pool3d(trajectory, output_size=latent_size)
    gripper = F.adaptive_max_pool3d(gripper, output_size=latent_size)
    weights = 1.0 + trajectory_weight * trajectory + gripper_weight * gripper
    denominator = weights.mean(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
    return weights / denominator


def weighted_flow_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    focus_mask: torch.Tensor | None = None,
    focus_weight: float = 2.0,
    valid_mask: torch.Tensor | None = None,
    loss_weight: torch.Tensor | None = None,
    weight_map: torch.Tensor | None = None,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if prediction.ndim != 5:
        raise ValueError("prediction must have shape (B, C, T, H, W)")
    if not torch.isfinite(prediction).all():
        raise ValueError("prediction must be finite")
    if not torch.isfinite(target).all():
        raise ValueError("target must be finite")
    if focus_weight < 0:
        raise ValueError("focus_weight must be non-negative")
    squared_error = (prediction.float() - target.float()).square()
    if loss_weight is not None and weight_map is not None:
        raise ValueError("loss_weight and weight_map are mutually exclusive")
    supplied_weight = loss_weight if loss_weight is not None else weight_map
    if focus_mask is not None and supplied_weight is not None:
        raise ValueError("focus_mask and weight_map are mutually exclusive")
    if focus_mask is None and valid_mask is None and supplied_weight is None:
        return squared_error.mean()
    expected_mask_shape = (prediction.shape[0], 1, *prediction.shape[2:])
    if loss_weight is not None:
        if prediction.shape[2:] != (21, 30, 40):
            raise ValueError(
                "prediction and target must use v3 latent geometry "
                "(B, C, 21, 30, 40) when loss_weight is supplied"
            )
        if loss_weight.shape != expected_mask_shape:
            raise ValueError(f"loss_weight must have shape {expected_mask_shape}")
    if supplied_weight is not None:
        if weight_map is not None and weight_map.shape != expected_mask_shape:
            raise ValueError(f"weight_map must have shape {expected_mask_shape}")
        weights = supplied_weight.to(device=prediction.device, dtype=torch.float32)
        if not torch.isfinite(weights).all():
            raise ValueError(
                "loss_weight must be finite" if loss_weight is not None else "weight_map must be finite"
            )
        if not (weights > 0).all():
            raise ValueError(
                "loss_weight must be strictly positive"
                if loss_weight is not None
                else "weight_map must be strictly positive"
            )
    else:
        weights = torch.ones(
            expected_mask_shape,
            device=prediction.device,
            dtype=torch.float32,
        )
    if focus_mask is not None:
        if focus_mask.shape != expected_mask_shape:
            raise ValueError(f"focus_mask must have shape {expected_mask_shape}")
        weights = weights + focus_weight * focus_mask.float().clamp(0, 1)
    if valid_mask is not None:
        if valid_mask.shape != expected_mask_shape:
            raise ValueError(f"valid_mask must have shape {expected_mask_shape}")
        valid = valid_mask.to(device=prediction.device, dtype=torch.float32)
        if not torch.isfinite(valid).all():
            raise ValueError("valid_mask must be finite")
        weights = weights * valid.clamp(0, 1)
    expanded_weights = weights.expand_as(squared_error)
    if supplied_weight is not None:
        denominator = expanded_weights.flatten(1).sum(dim=1)
        if (denominator <= 0).any():
            raise ValueError("no valid latent elements for a sample")
        numerator = (squared_error * expanded_weights).flatten(1).sum(dim=1)
        return (numerator / denominator).mean()
    denominator = expanded_weights.sum()
    return (squared_error * expanded_weights).sum() / denominator.clamp_min(1)
