from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch
from torch import Tensor, nn


CounterfactualKind = Literal["wrong-left", "wrong-right", "active-arm-null"]


@dataclass(frozen=True)
class ArmActionContent:
    """Per-arm action content before conversion into v11 controller slots."""

    anchor: Tensor
    translation: Tensor
    rotation: Tensor
    image_motion: Tensor
    gripper: Tensor
    arm_present: Tensor
    motion_active: Tensor
    support: Tensor


@dataclass(frozen=True)
class BimanualCondition:
    left: ArmActionContent
    right: ArmActionContent
    destination_time: Tensor


@dataclass(frozen=True)
class BimanualSlots:
    left: Tensor
    right: Tensor
    destination_time: Tensor


class _ArmSlotTokenizer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("slot width must be positive")
        self.translation = self._mlp(4, width)
        self.rotation = self._mlp(4, width)
        self.image_motion = self._mlp(6, width)
        self.gripper = self._mlp(3, width)
        self.token_type = nn.Parameter(torch.empty(4, width))
        nn.init.normal_(self.token_type, mean=0.0, std=0.02)

    @staticmethod
    def _mlp(input_width: int, output_width: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_width, output_width),
            nn.SiLU(),
            nn.Linear(output_width, output_width),
        )

    def forward(self, arm: ArmActionContent) -> Tensor:
        translation = torch.cat(
            (arm.translation, torch.linalg.vector_norm(arm.translation, dim=-1, keepdim=True)),
            dim=-1,
        )
        rotation = torch.cat(
            (arm.rotation, torch.linalg.vector_norm(arm.rotation, dim=-1, keepdim=True)),
            dim=-1,
        )
        interval_slots = torch.stack(
            (
                self.translation(translation),
                self.rotation(rotation),
                self.image_motion(arm.image_motion),
                self.gripper(arm.gripper),
            ),
            dim=2,
        )
        interval_slots = interval_slots + self.token_type[None, None]
        interval_slots = interval_slots * arm.arm_present[:, 1:, None, None].to(
            interval_slots.dtype
        )
        latent_zero = interval_slots.new_zeros(
            interval_slots.shape[0], 1, interval_slots.shape[2], interval_slots.shape[3]
        )
        return torch.cat((latent_zero, interval_slots), dim=1)


class BimanualSlotTokenizer(nn.Module):
    """Independent left/right tokenizers; weights are intentionally not shared."""

    def __init__(self, width: int = 384) -> None:
        super().__init__()
        self.left = _ArmSlotTokenizer(width)
        self.right = _ArmSlotTokenizer(width)


def _require_shape(name: str, value: Tensor, shape: tuple[int, ...]) -> None:
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")


def _validate_arm(name: str, arm: ArmActionContent, batch: int) -> None:
    _require_shape(f"{name}.anchor", arm.anchor, (batch, 6))
    _require_shape(f"{name}.translation", arm.translation, (batch, 20, 3))
    _require_shape(f"{name}.rotation", arm.rotation, (batch, 20, 3))
    _require_shape(f"{name}.image_motion", arm.image_motion, (batch, 20, 6))
    _require_shape(f"{name}.gripper", arm.gripper, (batch, 20, 3))
    _require_shape(f"{name}.arm_present", arm.arm_present, (batch, 21))
    _require_shape(f"{name}.motion_active", arm.motion_active, (batch, 20))
    if arm.support.ndim != 4 or tuple(arm.support.shape[:2]) != (batch, 21):
        raise ValueError(f"{name}.support must have shape (B, 21, H, W)")
    if arm.arm_present.dtype != torch.bool or arm.motion_active.dtype != torch.bool:
        raise ValueError(f"{name} presence and motion masks must be bool")
    for field_name in (
        "anchor",
        "translation",
        "rotation",
        "image_motion",
        "gripper",
        "support",
    ):
        if not torch.isfinite(getattr(arm, field_name)).all():
            raise ValueError(f"{name}.{field_name} must be finite")


def _validate_condition(condition: BimanualCondition) -> int:
    if condition.destination_time.ndim != 2 or condition.destination_time.shape[1] != 21:
        raise ValueError("destination time must have shape (B, 21)")
    batch = condition.destination_time.shape[0]
    expected = torch.arange(
        21,
        device=condition.destination_time.device,
        dtype=condition.destination_time.dtype,
    ).expand(batch, -1)
    if not torch.equal(condition.destination_time, expected):
        raise ValueError("destination time must be exactly 0..20 for every sample")
    _validate_arm("left", condition.left, batch)
    _validate_arm("right", condition.right, batch)
    return batch


def pack_initial_slots(
    tokenizer: BimanualSlotTokenizer,
    condition: BimanualCondition,
) -> BimanualSlots:
    _validate_condition(condition)
    return BimanualSlots(
        left=tokenizer.left(condition.left),
        right=tokenizer.right(condition.right),
        destination_time=condition.destination_time,
    )


def _motion_from(destination: ArmActionContent, source: ArmActionContent) -> ArmActionContent:
    return replace(
        destination,
        translation=source.translation.clone(),
        rotation=source.rotation.clone(),
        image_motion=source.image_motion.clone(),
        gripper=source.gripper.clone(),
    )


def _null_active_arm(arm: ArmActionContent, active: Tensor) -> ArmActionContent:
    mask = active[:, None, None]
    gripper = arm.gripper.clone()
    gripper_delta = torch.where(mask, torch.zeros_like(gripper[..., 2:]), gripper[..., 2:])
    gripper = torch.cat((gripper[..., :2], gripper_delta), dim=-1)
    return replace(
        arm,
        translation=torch.where(mask, torch.zeros_like(arm.translation), arm.translation),
        rotation=torch.where(mask, torch.zeros_like(arm.rotation), arm.rotation),
        image_motion=torch.where(mask, torch.zeros_like(arm.image_motion), arm.image_motion),
        gripper=gripper,
    )


def build_counterfactual(
    condition: BimanualCondition,
    kind: CounterfactualKind,
) -> BimanualCondition:
    _validate_condition(condition)
    if kind == "wrong-left":
        return replace(condition, left=_motion_from(condition.left, condition.right))
    if kind == "wrong-right":
        return replace(condition, right=_motion_from(condition.right, condition.left))
    if kind != "active-arm-null":
        raise ValueError(f"unsupported counterfactual kind: {kind}")

    left_active = condition.left.motion_active.any(dim=1)
    right_active = condition.right.motion_active.any(dim=1)
    if not torch.all(torch.logical_xor(left_active, right_active)):
        raise ValueError("active-arm-null requires exactly one active arm per sample")
    return replace(
        condition,
        left=_null_active_arm(condition.left, left_active),
        right=_null_active_arm(condition.right, right_active),
    )
