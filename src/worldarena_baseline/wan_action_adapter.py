from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint

from .action_condition import causal_time_groups


_V3_FRAMES = 81
_V3_RASTER_HEIGHT = 60
_V3_RASTER_WIDTH = 80
_V3_TOKEN_GRID = (21, 15, 20)
_V3_TOKEN_COUNT = 21 * 15 * 20
_INJECTION_POINTS = (0, 8, 16, 24)


def wrap_action_adapter_ddp(
    module: nn.Module,
    *,
    use_pose: bool,
    device_ids: list[int] | None,
) -> DistributedDataParallel:
    """Wrap the adapter for null-action-safe distributed training.

    A strict null action carries ``pose=None``.  Pose projections therefore
    intentionally do not participate in those iterations and DDP must discover
    the unused family before the following reduction begins.
    """

    return DistributedDataParallel(
        module,
        device_ids=device_ids,
        broadcast_buffers=False,
        find_unused_parameters=use_pose,
    )


class CheckpointedWanBlock(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, *args, **kwargs):
        if not self.training or not torch.is_grad_enabled():
            return self.block(*args, **kwargs)
        return checkpoint(self.block, *args, use_reentrant=False, **kwargs)


def enable_wan_block_checkpointing(backbone: nn.Module) -> None:
    if not hasattr(backbone, "blocks"):
        raise ValueError("backbone must expose a blocks sequence")
    backbone.blocks = nn.ModuleList(
        [
            block if isinstance(block, CheckpointedWanBlock) else CheckpointedWanBlock(block)
            for block in backbone.blocks
        ]
    )


def _causal_group_mean(features: torch.Tensor) -> torch.Tensor:
    """Mean frame features with Wan's explicit 0, [1:5], ..., [77:81] layout."""
    if features.ndim < 3 or features.shape[2] != _V3_FRAMES:
        raise ValueError("features must have 81 frames in dimension 2")
    return torch.stack(
        [features[:, :, list(group)].mean(dim=2) for group in causal_time_groups()],
        dim=2,
    )


class PerArmRasterEncoder(nn.Module):
    """One shared, frame-wise 2D encoder for each five-channel arm raster."""

    def __init__(self, *, adapter_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        if adapter_dim <= 0 or hidden_dim <= 0:
            raise ValueError("adapter_dim and hidden_dim must be positive")
        # `hidden_dim` remains part of the public constructor so checkpoint and
        # launcher configuration stays stable; the v3 path intentionally has no
        # high-resolution temporal hidden tensor.
        del hidden_dim
        self.encoder = nn.Conv2d(5, adapter_dim, kernel_size=4, stride=4)

    @staticmethod
    def _validate(raster: torch.Tensor) -> None:
        if raster.ndim != 5 or tuple(raster.shape[1:]) != (
            5,
            _V3_FRAMES,
            _V3_RASTER_HEIGHT,
            _V3_RASTER_WIDTH,
        ):
            raise ValueError("arm raster must have shape (B, 5, 81, 60, 80)")
        if not torch.isfinite(raster).all():
            raise ValueError("arm raster must be finite")

    def forward_grid(self, raster: torch.Tensor) -> torch.Tensor:
        self._validate(raster)
        batch_size = raster.shape[0]
        framewise = raster.permute(0, 2, 1, 3, 4).reshape(
            batch_size * _V3_FRAMES, 5, _V3_RASTER_HEIGHT, _V3_RASTER_WIDTH
        )
        encoded = self.encoder(framewise)
        encoded = encoded.reshape(batch_size, _V3_FRAMES, -1, 15, 20).permute(
            0, 2, 1, 3, 4
        )
        return _causal_group_mean(encoded)

    def forward(self, raster: torch.Tensor) -> torch.Tensor:
        return self.forward_grid(raster).permute(0, 2, 3, 4, 1).reshape(
            raster.shape[0], _V3_TOKEN_COUNT, -1
        )


# Backward-compatible import name for callers that explicitly construct an encoder.
ActionRasterEncoder = PerArmRasterEncoder


class PoseTemporalEncoder(nn.Module):
    def __init__(self, *, adapter_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(11, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, adapter_dim),
        )

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        if pose.ndim != 3 or tuple(pose.shape[1:]) != (_V3_FRAMES, 11):
            raise ValueError("pose must have shape (B, 81, 11)")
        if not torch.isfinite(pose).all():
            raise ValueError("pose must be finite")
        encoded = self.mlp(pose).transpose(1, 2)
        return _causal_group_mean(encoded).transpose(1, 2)


class WanActionAdapter(nn.Module):
    def __init__(
        self,
        *,
        adapter_dim: int = 256,
        wan_dim: int = 3072,
        hidden_dim: int = 128,
        injection_points: tuple[int, ...] = _INJECTION_POINTS,
        max_timestep: float = 1000.0,
        use_pose: bool = False,
        raster_support_gating: bool = False,
    ) -> None:
        super().__init__()
        if adapter_dim <= 0 or wan_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if not injection_points or len(set(injection_points)) != len(injection_points):
            raise ValueError("injection_points must be non-empty and unique")
        if any(point < 0 for point in injection_points):
            raise ValueError("injection_points must be non-negative")
        if max_timestep <= 0:
            raise ValueError("max_timestep must be positive")
        self.injection_points = injection_points
        self.max_timestep = max_timestep
        self.use_pose = bool(use_pose)
        self.raster_support_gating = bool(raster_support_gating)
        self.arm_encoder = PerArmRasterEncoder(adapter_dim=adapter_dim, hidden_dim=hidden_dim)
        self.action_present_embeddings = nn.Embedding(2, adapter_dim)
        self.left_raster_projections = self._projections(adapter_dim, wan_dim)
        self.right_raster_projections = self._projections(adapter_dim, wan_dim)
        if self.use_pose:
            self.left_pose_encoder = PoseTemporalEncoder(adapter_dim=adapter_dim)
            self.right_pose_encoder = PoseTemporalEncoder(adapter_dim=adapter_dim)
            self.left_pose_projections = self._projections(adapter_dim, wan_dim)
            self.right_pose_projections = self._projections(adapter_dim, wan_dim)
        else:
            self.left_pose_encoder = None
            self.right_pose_encoder = None
            self.left_pose_projections = nn.ModuleDict()
            self.right_pose_projections = nn.ModuleDict()
        gate_dim = max(16, min(128, adapter_dim))
        self.timestep_gates = nn.Sequential(
            nn.Linear(1, gate_dim),
            nn.SiLU(),
            nn.Linear(gate_dim, len(injection_points)),
            nn.Sigmoid(),
        )
        self.last_residual_norms: dict[int, torch.Tensor] = {}
        self.last_residual_component_norms: dict[int, dict[str, torch.Tensor]] = {}

    def _projections(self, adapter_dim: int, wan_dim: int) -> nn.ModuleDict:
        projections = nn.ModuleDict(
            {str(point): nn.Linear(adapter_dim, wan_dim) for point in self.injection_points}
        )
        for projection in projections.values():
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        return projections

    @staticmethod
    def _split_arm_rasters(raster: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if raster.ndim != 5 or tuple(raster.shape[1:]) != (
            10,
            _V3_FRAMES,
            _V3_RASTER_HEIGHT,
            _V3_RASTER_WIDTH,
        ):
            raise ValueError("action raster must have shape (B, 10, 81, 60, 80)")
        if not torch.isfinite(raster).all():
            raise ValueError("action raster must be finite")
        return raster[:, :5], raster[:, 5:]

    @staticmethod
    def _action_present(
        action_present: torch.Tensor | None, *, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if action_present is None:
            return torch.ones(batch_size, dtype=torch.long, device=device)
        if action_present.ndim == 2 and action_present.shape == (batch_size, 1):
            action_present = action_present[:, 0]
        if action_present.ndim != 1 or action_present.shape[0] != batch_size:
            raise ValueError("action_present must have shape (B,) or (B, 1)")
        if not torch.isfinite(action_present).all() or not torch.all(
            (action_present == 0) | (action_present == 1)
        ):
            raise ValueError("action_present must contain only zero or one")
        return action_present.to(device=device, dtype=torch.long)

    @staticmethod
    def _validate_support(support: torch.Tensor, *, batch_size: int) -> None:
        if tuple(support.shape) != (batch_size, 2, *_V3_TOKEN_GRID):
            raise ValueError("condition_support must have shape (B, 2, 21, 15, 20)")
        if not torch.isfinite(support).all():
            raise ValueError("condition_support must be finite")

    def encode_raster_arms(
        self, raster: torch.Tensor, *, action_present: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left_raster, right_raster = self._split_arm_rasters(raster)
        present = self._action_present(
            action_present, batch_size=raster.shape[0], device=raster.device
        )
        embedding = self.action_present_embeddings(present)[:, None, :]
        return self.arm_encoder(left_raster) + embedding, self.arm_encoder(right_raster) + embedding

    def _encode_pose_arm(
        self,
        pose: torch.Tensor,
        support: torch.Tensor,
        encoder: PoseTemporalEncoder,
    ) -> torch.Tensor:
        pose_tokens = encoder(pose)
        return (
            pose_tokens[:, :, None, None, :]
            .expand(-1, -1, 15, 20, -1)
            .mul(support[:, :, :, :, None])
            .reshape(pose.shape[0], _V3_TOKEN_COUNT, -1)
        )

    def encode_pose_arms(
        self,
        left_pose: torch.Tensor | None,
        right_pose: torch.Tensor | None,
        condition_support: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(batch_size, _V3_TOKEN_COUNT, self.action_present_embeddings.embedding_dim, device=device, dtype=dtype)
        if left_pose is None and right_pose is None:
            return zeros, zeros.clone()
        if not self.use_pose or self.left_pose_encoder is None or self.right_pose_encoder is None:
            raise ValueError("pose branch is disabled")
        if left_pose is None or right_pose is None:
            raise ValueError("left_pose and right_pose must be supplied together")
        if condition_support is None:
            raise ValueError("condition_support is required with pose")
        self._validate_support(condition_support, batch_size=batch_size)
        if (
            left_pose.device != device
            or right_pose.device != device
            or condition_support.device != device
        ):
            raise ValueError("pose and support tensors must be on the raster device")
        support = condition_support.to(dtype=dtype)
        return (
            self._encode_pose_arm(left_pose.to(dtype=dtype), support[:, 0], self.left_pose_encoder),
            self._encode_pose_arm(right_pose.to(dtype=dtype), support[:, 1], self.right_pose_encoder),
        )

    def encode_arms(
        self,
        raster: torch.Tensor,
        *,
        left_pose: torch.Tensor | None = None,
        right_pose: torch.Tensor | None = None,
        condition_support: torch.Tensor | None = None,
        action_present: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return combined arm features for diagnostic consumers only."""
        left, right = self.encode_raster_arms(raster, action_present=action_present)
        left_pose_tokens, right_pose_tokens = self.encode_pose_arms(
            left_pose, right_pose, condition_support,
            batch_size=raster.shape[0], device=raster.device, dtype=left.dtype,
        )
        return left + left_pose_tokens, right + right_pose_tokens

    def _gates(self, timestep: torch.Tensor, *, batch_size: int, seq_len: int, dtype: torch.dtype) -> torch.Tensor:
        if timestep.ndim == 1 and timestep.shape[0] == batch_size:
            gate_input = timestep[:, None, None]
        elif timestep.ndim == 2 and tuple(timestep.shape) == (batch_size, seq_len):
            gate_input = timestep[:, :, None]
        else:
            raise ValueError("timestep must have shape (B,) or (B, seq_len)")
        return self.timestep_gates(gate_input.to(dtype=dtype) / self.max_timestep)

    def residual_components(
        self,
        raster: torch.Tensor,
        timestep: torch.Tensor,
        *,
        seq_len: int,
        left_pose: torch.Tensor | None = None,
        right_pose: torch.Tensor | None = None,
        condition_support: torch.Tensor | None = None,
        action_present: torch.Tensor | None = None,
    ) -> dict[int, dict[str, torch.Tensor]]:
        if seq_len < _V3_TOKEN_COUNT:
            raise ValueError(f"seq_len must include {_V3_TOKEN_COUNT} video tokens")
        if self.raster_support_gating and condition_support is None:
            raise ValueError("condition_support is required when raster support gating is enabled")
        if condition_support is not None:
            self._validate_support(condition_support, batch_size=raster.shape[0])
        left_raster, right_raster = self.encode_raster_arms(raster, action_present=action_present)
        left_pose_tokens, right_pose_tokens = self.encode_pose_arms(
            left_pose, right_pose, condition_support,
            batch_size=raster.shape[0], device=raster.device, dtype=left_raster.dtype,
        )
        padding = seq_len - _V3_TOKEN_COUNT
        left_raster_support = right_raster_support = None
        if self.raster_support_gating:
            assert condition_support is not None
            support = condition_support.to(device=raster.device, dtype=left_raster.dtype)
            left_raster_support = support[:, 0].reshape(raster.shape[0], _V3_TOKEN_COUNT, 1)
            right_raster_support = support[:, 1].reshape(raster.shape[0], _V3_TOKEN_COUNT, 1)
        if padding:
            pad = left_raster.new_zeros(raster.shape[0], padding, left_raster.shape[-1])
            left_raster, right_raster = torch.cat((left_raster, pad), dim=1), torch.cat((right_raster, pad.clone()), dim=1)
            left_pose_tokens, right_pose_tokens = torch.cat((left_pose_tokens, pad.clone()), dim=1), torch.cat((right_pose_tokens, pad.clone()), dim=1)
            if left_raster_support is not None and right_raster_support is not None:
                support_pad = left_raster.new_zeros(raster.shape[0], padding, 1)
                left_raster_support = torch.cat((left_raster_support, support_pad), dim=1)
                right_raster_support = torch.cat((right_raster_support, support_pad.clone()), dim=1)
        gates = self._gates(timestep, batch_size=raster.shape[0], seq_len=seq_len, dtype=left_raster.dtype)
        components = {}
        for index, point in enumerate(self.injection_points):
            gate = gates[..., index].unsqueeze(-1)
            zero_pose = left_raster.new_zeros(
                raster.shape[0], seq_len, self.left_raster_projections[str(point)].out_features
            )
            point_components = {
                "left_raster": self.left_raster_projections[str(point)](left_raster) * gate,
                "right_raster": self.right_raster_projections[str(point)](right_raster) * gate,
                "left_pose": (
                    self.left_pose_projections[str(point)](left_pose_tokens) * gate
                    if self.use_pose
                    else zero_pose
                ),
                "right_pose": (
                    self.right_pose_projections[str(point)](right_pose_tokens) * gate
                    if self.use_pose
                    else zero_pose.clone()
                ),
            }
            if left_raster_support is not None and right_raster_support is not None:
                point_components["left_raster"] = (
                    point_components["left_raster"] * left_raster_support
                )
                point_components["right_raster"] = (
                    point_components["right_raster"] * right_raster_support
                )
            if padding:
                for name, value in point_components.items():
                    point_components[name] = torch.cat(
                        (
                            value[:, :_V3_TOKEN_COUNT],
                            value.new_zeros(
                                raster.shape[0], padding, value.shape[-1]
                            ),
                        ),
                        dim=1,
                    )
            if any(not value.isfinite().all() for value in point_components.values()):
                raise ValueError("action residual component is not finite")
            components[point] = point_components
        self.last_residual_component_norms = {
            point: {
                name: value.norm(dim=(1, 2))
                for name, value in point_components.items()
            }
            for point, point_components in components.items()
        }
        return components

    def residuals(
        self,
        raster: torch.Tensor,
        timestep: torch.Tensor,
        *,
        seq_len: int,
        left_pose: torch.Tensor | None = None,
        right_pose: torch.Tensor | None = None,
        condition_support: torch.Tensor | None = None,
        action_present: torch.Tensor | None = None,
    ) -> dict[int, torch.Tensor]:
        components = self.residual_components(
            raster,
            timestep,
            seq_len=seq_len,
            left_pose=left_pose,
            right_pose=right_pose,
            condition_support=condition_support,
            action_present=action_present,
        )
        residuals = {
            point: sum(point_components.values())
            for point, point_components in components.items()
        }
        if any(not residual.isfinite().all() for residual in residuals.values()):
            raise ValueError("action residual is not finite")
        self.last_residual_norms = {
            point: residual.norm(dim=(1, 2)) for point, residual in residuals.items()
        }
        return residuals

    def forward(self, raster: torch.Tensor, timestep: torch.Tensor, **kwargs) -> dict[int, torch.Tensor]:
        return self.residuals(raster, timestep, **kwargs)


def load_raster_adapter_checkpoint(
    pose_adapter: WanActionAdapter, raster_state: Mapping[str, torch.Tensor]
) -> None:
    """Strictly map a raster-only adapter checkpoint into its pose-enabled fork."""
    if not pose_adapter.use_pose:
        raise ValueError("target adapter must enable pose")
    target = pose_adapter.state_dict()
    pose_prefixes = (
        "left_pose_encoder.",
        "right_pose_encoder.",
        "left_pose_projections.",
        "right_pose_projections.",
    )
    pose_family_keys = {
        key for key in target if key.startswith(pose_prefixes)
    }
    expected_raster_keys = set(target).difference(pose_family_keys)
    if set(raster_state) != expected_raster_keys:
        raise ValueError("raster checkpoint does not strictly match pose adapter")
    mapped = dict(target)
    for name, value in raster_state.items():
        if target[name].shape != value.shape:
            raise ValueError(f"checkpoint tensor shape differs for {name}")
        mapped[name] = value
    pose_adapter.load_state_dict(mapped, strict=True)


def load_adapter_initialization_checkpoint(
    adapter: WanActionAdapter,
    state: Mapping[str, torch.Tensor],
    *,
    source_use_pose: bool,
    allow_raster_to_pose: bool,
) -> None:
    """Load an exact adapter state or an explicitly authorized raster fork."""

    if bool(source_use_pose) == adapter.use_pose:
        adapter.load_state_dict(state, strict=True)
        return
    if allow_raster_to_pose and adapter.use_pose and not source_use_pose:
        load_raster_adapter_checkpoint(adapter, state)
        return
    raise ValueError("adapter initialization use_pose mismatch")


class ActionConditionedWan(nn.Module):
    def __init__(self, backbone: nn.Module, adapter: WanActionAdapter) -> None:
        super().__init__()
        if not hasattr(backbone, "blocks"):
            raise ValueError("backbone must expose a blocks sequence")
        adapter_module = getattr(adapter, "module", adapter)
        if max(adapter_module.injection_points) >= len(backbone.blocks):
            raise ValueError("adapter injection point exceeds backbone block count")
        self.backbone = backbone
        self.adapter = adapter

    @staticmethod
    def _pre_hook(residual: torch.Tensor):
        def inject(_module: nn.Module, args: tuple) -> tuple:
            if not args:
                raise RuntimeError("Wan block pre-hook received no token tensor")
            tokens = args[0]
            if tokens.shape != residual.shape:
                raise RuntimeError(
                    f"action residual shape {tuple(residual.shape)} does not match Wan tokens {tuple(tokens.shape)}"
                )
            return (tokens + residual.to(device=tokens.device, dtype=tokens.dtype), *args[1:])

        return inject

    def forward(
        self,
        x,
        t: torch.Tensor,
        context,
        seq_len: int,
        y=None,
        *,
        action_raster: torch.Tensor,
        left_pose: torch.Tensor | None = None,
        right_pose: torch.Tensor | None = None,
        condition_support: torch.Tensor | None = None,
        action_present: torch.Tensor | None = None,
        action_scale: float = 1.0,
    ):
        if not torch.isfinite(torch.tensor(action_scale)):
            raise ValueError("action_scale must be finite")
        residuals = self.adapter(
            action_raster, t, seq_len=seq_len, left_pose=left_pose,
            right_pose=right_pose, condition_support=condition_support,
            action_present=action_present,
        )
        handles = [
            self.backbone.blocks[point].register_forward_pre_hook(
                self._pre_hook(residual * action_scale)
            )
            for point, residual in residuals.items()
        ]
        try:
            return self.backbone(x, t, context, seq_len, y=y)
        finally:
            for handle in handles:
                handle.remove()
