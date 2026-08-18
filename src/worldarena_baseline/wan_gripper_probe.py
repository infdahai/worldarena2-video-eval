"""Frozen frame-local latent probe for v6 gripper trajectory supervision."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .action_condition import causal_time_groups


PROBE_CONTRACT = "wan-action-lite-v6-gripper-probe/1"
SPLIT_CONTRACT = "wan-action-lite-v6-probe-split/1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_LATENT_SHAPE = (48, 21, 30, 40)
_RASTER_SHAPE = (10, 81, 60, 80)


class GripperTrajectoryProbe(nn.Module):
    """Decode two gripper heatmaps independently for every latent frame."""

    def __init__(self, *, hidden_dim: int = 64) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(48, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2 or 1, kernel_size=2, stride=2),
            nn.SiLU(),
        )
        self.head = nn.Conv2d(hidden_dim // 2 or 1, 2, kernel_size=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 5 or tuple(latent.shape[1:]) != _LATENT_SHAPE:
            raise ValueError("latent must have shape (B,48,21,30,40)")
        if not torch.isfinite(latent).all():
            raise ValueError("latent must be finite")
        batch = latent.shape[0]
        frames = latent.permute(0, 2, 1, 3, 4).reshape(batch * 21, 48, 30, 40)
        logits = self.head(self.frame_encoder(frames))
        return logits.reshape(batch, 21, 2, 60, 80).permute(0, 2, 1, 3, 4)


def soft_argmax_2d(logits: torch.Tensor) -> torch.Tensor:
    """Return normalized x/y coordinates for heatmaps shaped (..., H, W)."""
    if logits.ndim < 2 or tuple(logits.shape[-2:]) != (60, 80):
        raise ValueError("heatmap logits must end in shape (60,80)")
    if not torch.isfinite(logits).all():
        raise ValueError("heatmap logits must be finite")
    probabilities = torch.softmax(logits.float().flatten(-2), dim=-1).reshape(
        *logits.shape[:-2], 60, 80
    )
    x = torch.linspace(0, 1, 80, device=logits.device, dtype=probabilities.dtype)
    y = torch.linspace(0, 1, 60, device=logits.device, dtype=probabilities.dtype)
    return torch.stack(
        (
            (probabilities * x.reshape(1, 80)).sum(dim=(-2, -1)),
            (probabilities * y.reshape(60, 1)).sum(dim=(-2, -1)),
        ),
        dim=-1,
    )


def probe_training_loss(
    *,
    logits: torch.Tensor,
    target_heatmap: torch.Tensor,
    target_position: torch.Tensor,
    position_valid: torch.Tensor,
) -> torch.Tensor:
    """Train heatmap identity and localization with a raster-pixel objective."""
    if logits.ndim != 5 or tuple(logits.shape[-2:]) != (60, 80):
        raise ValueError("logits must have shape (B,2,T,60,80)")
    if target_heatmap.shape != logits.shape:
        raise ValueError("target_heatmap must match logits")
    if tuple(target_position.shape) != (*logits.shape[:-2], 2):
        raise ValueError("target_position must have shape (B,2,T,2)")
    if position_valid.shape != logits.shape[:-2] or position_valid.dtype != torch.bool:
        raise ValueError("position_valid must be boolean with shape (B,2,T)")
    if not all(
        torch.isfinite(value).all()
        for value in (logits, target_heatmap, target_position)
    ):
        raise ValueError("probe supervision must be finite")
    target = target_heatmap.float()
    target = target / target.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    valid = position_valid.to(logits.dtype)
    cross_entropy = -(
        target.flatten(-2) * torch.log_softmax(logits.float().flatten(-2), dim=-1)
    ).sum(dim=-1)
    heatmap_loss = (cross_entropy * valid).sum() / valid.sum().clamp_min(1)
    predicted_position = soft_argmax_2d(logits)
    raster_scale = logits.new_tensor((79.0, 59.0))
    pixel_delta = (predicted_position - target_position.float()) * raster_scale
    position_loss = (
        F.huber_loss(pixel_delta, torch.zeros_like(pixel_delta), reduction="none", delta=1.0)
        .sum(dim=-1)
        .mul(valid)
        .sum()
        / valid.sum().clamp_min(1)
    )
    return heatmap_loss + position_loss


def _heatmap_position(heatmap: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mass = heatmap.float().sum(dim=(-2, -1))
    x = torch.linspace(0, 1, 80, device=heatmap.device, dtype=torch.float32)
    y = torch.linspace(0, 1, 60, device=heatmap.device, dtype=torch.float32)
    denominator = mass.clamp_min(1e-12)
    position = torch.stack(
        (
            (heatmap.float() * x.reshape(1, 80)).sum(dim=(-2, -1)) / denominator,
            (heatmap.float() * y.reshape(60, 1)).sum(dim=(-2, -1)) / denominator,
        ),
        dim=-1,
    )
    return position, mass > 0


def build_gripper_targets(
    raster: torch.Tensor,
    *,
    observability: torch.Tensor | None,
    min_observed_fraction: float = 0.75,
) -> dict[str, torch.Tensor]:
    """Causally pack explicit observable EEF heatmaps into 21 latent labels."""
    if raster.ndim != 5 or tuple(raster.shape[1:]) != _RASTER_SHAPE:
        raise ValueError("raster must have shape (B,10,81,60,80)")
    if observability is None:
        raise ValueError("explicit real-image observability is required")
    if tuple(observability.shape) != (raster.shape[0], 2, 81):
        raise ValueError("observability must have shape (B,2,81)")
    if observability.dtype != torch.bool:
        raise ValueError("observability must be boolean")
    if not 0 < min_observed_fraction <= 1:
        raise ValueError("min_observed_fraction must be in (0,1]")
    if not torch.isfinite(raster).all():
        raise ValueError("raster must be finite")

    eef = raster[:, [1, 6]].float()
    heatmaps: list[torch.Tensor] = []
    validity: list[torch.Tensor] = []
    for group in causal_time_groups():
        indices = list(group)
        observed = observability[:, :, indices]
        weights = observed.to(eef.dtype)[..., None, None]
        denominator = weights.sum(dim=2).clamp_min(1)
        heatmap = (eef[:, :, indices] * weights).sum(dim=2) / denominator
        observed_fraction = observed.float().mean(dim=2)
        heatmaps.append(heatmap)
        validity.append(observed_fraction >= min_observed_fraction)
    packed_heatmap = torch.stack(heatmaps, dim=2)
    valid = torch.stack(validity, dim=2)
    position, nonempty = _heatmap_position(packed_heatmap)
    valid &= nonempty
    valid[:, :, 0] = False
    velocity = torch.zeros_like(position)
    velocity[:, :, 1:] = position[:, :, 1:] - position[:, :, :-1]
    velocity_valid = torch.zeros_like(valid)
    velocity_valid[:, :, 1:] = valid[:, :, 1:] & valid[:, :, :-1]
    return {
        "heatmap": packed_heatmap,
        "position": position,
        "position_valid": valid,
        "velocity": velocity,
        "velocity_valid": velocity_valid,
    }


def match_projected_grippers_to_detections(
    *,
    projected_xy: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    maximum_distance_px: float = 64.0,
    minimum_score: float = 0.3,
) -> torch.Tensor:
    """Return RGB-backed visibility using proximity to kinematic arm identity."""
    if tuple(projected_xy.shape) != (2, 2):
        raise ValueError("projected_xy must have shape (2,2)")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N,4)")
    if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
        raise ValueError("scores must have shape (N,)")
    if maximum_distance_px <= 0 or not 0 <= minimum_score <= 1:
        raise ValueError("observability thresholds are invalid")
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
        raise ValueError("detection inputs must be finite")
    finite_coordinates = torch.isfinite(projected_xy)
    if not torch.all(finite_coordinates.all(dim=1) | (~finite_coordinates).all(dim=1)):
        raise ValueError("each projected arm must contain two finite values or two NaNs")
    selected = scores >= minimum_score
    candidates = boxes[selected]
    visible = torch.zeros(2, dtype=torch.bool, device=projected_xy.device)
    valid_arm_indices = torch.where(finite_coordinates.all(dim=1))[0]
    if candidates.numel() == 0 or valid_arm_indices.numel() == 0:
        return visible
    candidates = candidates.to(projected_xy)
    points = projected_xy[valid_arm_indices].float()
    dx = torch.maximum(
        candidates[None, :, 0] - points[:, None, 0],
        points[:, None, 0] - candidates[None, :, 2],
    ).clamp_min(0)
    dy = torch.maximum(
        candidates[None, :, 1] - points[:, None, 1],
        points[:, None, 1] - candidates[None, :, 3],
    ).clamp_min(0)
    distances = torch.sqrt(dx.square() + dy.square())
    pairs = sorted(
        (
            (float(distances[arm_index, detection]), int(arm), detection)
            for arm_index, arm in enumerate(valid_arm_indices.tolist())
            for detection in range(candidates.shape[0])
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    assigned_arms: set[int] = set()
    assigned_detections: set[int] = set()
    for distance, arm, detection in pairs:
        if distance > maximum_distance_px:
            break
        if arm in assigned_arms or detection in assigned_detections:
            continue
        visible[arm] = True
        assigned_arms.add(arm)
        assigned_detections.add(detection)
    return visible


def projected_gripper_visibility_from_detections(
    *,
    projected_xy: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    maximum_distance_px: float = 64.0,
    minimum_score: float = 0.3,
) -> torch.Tensor:
    """Mark each FK-projected EEF visible when any RGB robot box supports it.

    Unlike identity matching, visibility is not one-to-one: one contiguous
    detected robot-arm region can contain both EEFs during crossing, contact,
    or occlusion.  Arm identity stays bound to the supplied FK projection.
    """
    if tuple(projected_xy.shape) != (2, 2):
        raise ValueError("projected_xy must have shape (2,2)")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N,4)")
    if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
        raise ValueError("scores must have shape (N,)")
    if maximum_distance_px <= 0 or not 0 <= minimum_score <= 1:
        raise ValueError("observability thresholds are invalid")
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
        raise ValueError("detection inputs must be finite")
    finite_coordinates = torch.isfinite(projected_xy)
    if not torch.all(finite_coordinates.all(dim=1) | (~finite_coordinates).all(dim=1)):
        raise ValueError("each projected arm must contain two finite values or two NaNs")
    visible = torch.zeros(2, dtype=torch.bool, device=projected_xy.device)
    candidates = boxes[scores >= minimum_score]
    arm_indices = torch.where(finite_coordinates.all(dim=1))[0]
    if candidates.numel() == 0 or arm_indices.numel() == 0:
        return visible
    points = projected_xy[arm_indices].float()
    candidates = candidates.to(points)
    dx = torch.maximum(
        candidates[None, :, 0] - points[:, None, 0],
        points[:, None, 0] - candidates[None, :, 2],
    ).clamp_min(0)
    dy = torch.maximum(
        candidates[None, :, 1] - points[:, None, 1],
        points[:, None, 1] - candidates[None, :, 3],
    ).clamp_min(0)
    visible[arm_indices] = torch.sqrt(dx.square() + dy.square()).le(
        maximum_distance_px
    ).any(dim=1)
    return visible


def masks_to_xyxy_boxes(masks: torch.Tensor) -> torch.Tensor:
    """Convert non-empty boolean masks into half-open pixel xyxy boxes."""
    if masks.ndim != 3 or masks.dtype != torch.bool:
        raise ValueError("masks must be boolean with shape (N,H,W)")
    boxes: list[torch.Tensor] = []
    for mask in masks:
        coordinates = torch.nonzero(mask, as_tuple=False)
        if coordinates.numel() == 0:
            continue
        lower = coordinates.amin(dim=0)
        upper = coordinates.amax(dim=0) + 1
        boxes.append(
            torch.stack((lower[1], lower[0], upper[1], upper[0])).to(torch.float32)
        )
    if not boxes:
        return torch.empty((0, 4), dtype=torch.float32, device=masks.device)
    return torch.stack(boxes)


def validate_rgb_observability_receipts(
    receipts: Iterable[Mapping[str, Any]],
    *,
    expected_split_sha256: str,
    expected_samples: int,
) -> dict[str, Any]:
    values = [dict(receipt) for receipt in receipts]
    expected_hash = _require_sha(
        expected_split_sha256, label="expected observability split SHA-256"
    )
    if len(values) != 7 or expected_samples <= 0:
        raise ValueError("observability requires seven workers and positive samples")
    for rank, receipt in enumerate(values):
        if (
            receipt.get("contract") != "wan-action-lite-v6-rgb-observability/2"
            or receipt.get("backend") != "sam3-video-predictor-robot-arm"
            or receipt.get("rank") != rank
            or receipt.get("world_size") != 7
            or receipt.get("split_sha256") != expected_hash
            or type(receipt.get("samples")) is not int
            or receipt["samples"] <= 0
        ):
            raise ValueError(f"observability worker receipt mismatch at rank {rank}")
    if sum(receipt["samples"] for receipt in values) != expected_samples:
        raise ValueError("observability worker sample total mismatch")
    return {
        "passed": True,
        "world_size": 7,
        "rank_mapping": list(range(7)),
        "samples": expected_samples,
        "split_sha256": expected_hash,
        "backend": "sam3-video-predictor-robot-arm",
    }


def _hash_rank(sample: str, *, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{sample}".encode()).digest()


def build_gripper_probe_split(
    rows: Iterable[Mapping[str, Any]],
    *,
    heldout_fraction: float = 0.1,
    seed: int = 20260818,
) -> dict[str, Any]:
    """Build a deterministic task-stratified split without changing row bytes."""
    if not 0 < heldout_fraction < 1:
        raise ValueError("heldout_fraction must be in (0,1)")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for raw in rows:
        row = dict(raw)
        sample, task = row.get("sample"), row.get("task")
        if not isinstance(sample, str) or not sample or sample in seen:
            raise ValueError("probe split samples must be unique non-empty strings")
        if not isinstance(task, str) or not task:
            raise ValueError("probe split rows require a task")
        seen.add(sample)
        grouped[task].append(row)
    if not grouped:
        raise ValueError("probe split cannot be empty")
    train: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    for task, task_rows in sorted(grouped.items()):
        ordered = sorted(
            task_rows,
            key=lambda row: (_hash_rank(str(row["sample"]), seed=seed), str(row["sample"])),
        )
        count = max(1, math.floor(len(ordered) * heldout_fraction))
        if count >= len(ordered):
            raise ValueError(f"task {task} has too few rows for a disjoint split")
        heldout.extend(ordered[:count])
        train.extend(ordered[count:])
    train.sort(key=lambda row: str(row["sample"]))
    heldout.sort(key=lambda row: str(row["sample"]))
    return {
        "contract": SPLIT_CONTRACT,
        "seed": seed,
        "heldout_fraction": heldout_fraction,
        "train_rows": train,
        "heldout_rows": heldout,
    }


def probe_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    limits = {
        "median_error_px": 1.5,
        "p90_error_px": 4.0,
        "left_median_error_px": 1.5,
        "right_median_error_px": 1.5,
        "bimanual_crossing_median_error_px": 3.0,
    }
    reasons: list[str] = []
    for name, limit in limits.items():
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"probe metric {name} is missing or invalid") from exc
        if not math.isfinite(value) or value > limit:
            reasons.append(f"{name}>{limit}")
    try:
        correct = float(metrics["correct_arm_error_px"])
        swapped = float(metrics["swapped_arm_error_px"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("probe arm-mapping metrics are missing or invalid") from exc
    if not math.isfinite(correct) or not math.isfinite(swapped) or correct >= swapped:
        reasons.append("correct_arm_not_better_than_swapped")
    if metrics.get("future_independence_exact") is not True:
        reasons.append("future_independence_failed")
    return {"pass": not reasons, "reasons": reasons, "limits": limits}


def _require_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def validate_probe_checkpoint(
    payload: Mapping[str, Any],
    *,
    expected_split_sha256: str,
    expected_source_manifest_sha256: str,
) -> None:
    if payload.get("contract") != PROBE_CONTRACT:
        raise ValueError("probe checkpoint contract mismatch")
    if payload.get("input_channels") != 48 or payload.get("output_arms") != 2:
        raise ValueError("probe checkpoint geometry mismatch")
    if _require_sha(payload.get("split_sha256"), label="split SHA-256") != _require_sha(
        expected_split_sha256, label="expected split SHA-256"
    ):
        raise ValueError("probe split SHA-256 mismatch")
    if _require_sha(
        payload.get("source_manifest_sha256"), label="source manifest SHA-256"
    ) != _require_sha(
        expected_source_manifest_sha256,
        label="expected source manifest SHA-256",
    ):
        raise ValueError("probe source manifest SHA-256 mismatch")
    _require_sha(payload.get("observability_sha256"), label="observability SHA-256")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("probe state_dict is empty")
    if payload.get("frozen") is not True:
        raise ValueError("probe checkpoint is not frozen")
    if "optimizer" in payload:
        raise ValueError("frozen probe checkpoint must not contain optimizer state")
    gate = probe_gate(payload.get("heldout_metrics", {}))
    if not gate["pass"]:
        raise ValueError(f"probe heldout gate failed: {gate['reasons']}")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
