"""Geometry-only Q/K/V/O LoRA branch for Wan v7.1."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .wan_se3_attention import (
    ArmGroupedSE3Geometry,
    AttentionCallable,
    RopeApplyCallable,
)


class GeometryLoRADelta(nn.Module):
    """A bias-free FP32 low-rank delta used only by the geometry path."""

    def __init__(self, width: int, rank: int) -> None:
        super().__init__()
        if width <= 0 or rank <= 0 or rank > width:
            raise ValueError("geometry LoRA width/rank is invalid")
        self.width = int(width)
        self.rank = int(rank)
        self.down = nn.Parameter(torch.empty(rank, width, dtype=torch.float32))
        self.up = nn.Parameter(torch.zeros(width, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))

    def forward(self, value: Tensor) -> Tensor:
        if value.shape[-1] != self.width:
            raise ValueError("geometry LoRA input width differs")
        result = F.linear(F.linear(value.float(), self.down), self.up)
        return result.to(dtype=value.dtype)


class SE3GeometryLoRAAttention(nn.Module):
    """Frozen native Wan attention plus a learnable SE(3) LoRA branch.

    Native Q/K/V/O weights and native RoPE execution remain untouched.  The
    geometry branch forks the frozen raw projections before Wan 3D RoPE, adds
    its own rank-limited Q/K/V deltas, applies fixed arm-grouped SE(3), then
    uses frozen O plus a geometry-only O delta.  A zero-initialized output
    channel gate makes installation bitwise equal to the frozen native path.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        attention_fn: AttentionCallable,
        rope_apply_fn: RopeApplyCallable,
        num_heads: int = 24,
        head_dim: int = 128,
        rank: int = 16,
    ) -> None:
        super().__init__()
        for name in ("q", "k", "v", "o"):
            if not hasattr(base, name):
                raise TypeError(f"base attention must expose {name}")
        if not (hasattr(base, "q_norm") and hasattr(base, "k_norm")) and not (
            hasattr(base, "norm_q") and hasattr(base, "norm_k")
        ):
            raise TypeError("base attention must expose q_norm/k_norm or norm_q/norm_k")
        width = int(num_heads) * int(head_dim)
        self.base = base.requires_grad_(False)
        self.geometry = ArmGroupedSE3Geometry(
            attention_fn=attention_fn,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        self.geometry.requires_grad_(False)
        self.rope_apply_fn = rope_apply_fn
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.rank = int(rank)
        self.q_lora = GeometryLoRADelta(width, rank)
        self.k_lora = GeometryLoRADelta(width, rank)
        self.v_lora = GeometryLoRADelta(width, rank)
        self.o_lora = GeometryLoRADelta(width, rank)
        self.channel_gate = nn.Parameter(torch.zeros(width, dtype=torch.float32))

    def forward(
        self,
        x: Tensor,
        seq_lens: Tensor,
        grid_sizes: Tensor,
        freqs: object,
        *,
        arm_transform: Tensor,
        arm_inverse: Tensor | None = None,
        arm_present: Tensor,
        checkpoint_replay_release: Callable[[], None] | None = None,
    ) -> Tensor:
        try:
            return self._forward_with_condition(
                x,
                seq_lens,
                grid_sizes,
                freqs,
                arm_transform=arm_transform,
                arm_inverse=arm_inverse,
                arm_present=arm_present,
            )
        finally:
            if checkpoint_replay_release is not None:
                checkpoint_replay_release()

    def _forward_with_condition(
        self,
        x: Tensor,
        seq_lens: Tensor,
        grid_sizes: Tensor,
        freqs: object,
        *,
        arm_transform: Tensor,
        arm_inverse: Tensor | None,
        arm_present: Tensor,
    ) -> Tensor:
        width = self.num_heads * self.head_dim
        if x.ndim != 3 or x.shape[-1] != width:
            raise ValueError(f"expected hidden width {width}")

        raw_q = self.base.q(x)
        raw_k = self.base.k(x)
        raw_v = self.base.v(x)
        original_q = self._q_normalizer()(raw_q).reshape(
            *x.shape[:2], self.num_heads, self.head_dim
        )
        original_k = self._k_normalizer()(raw_k).reshape(
            *x.shape[:2], self.num_heads, self.head_dim
        )
        original_v = raw_v.reshape(*x.shape[:2], self.num_heads, self.head_dim)

        native_q = self.rope_apply_fn(original_q.clone(), grid_sizes, freqs)
        native_k = self.rope_apply_fn(original_k.clone(), grid_sizes, freqs)
        native_heads = self.geometry.attention_fn(native_q, native_k, original_v, seq_lens)
        if native_heads.shape != original_q.shape:
            raise ValueError("attention_fn must return the same shape as q")
        original_output = self.base.o(native_heads.flatten(2))

        geometry_q = self._q_normalizer()(raw_q + self.q_lora(x)).reshape(
            *x.shape[:2], self.num_heads, self.head_dim
        )
        geometry_k = self._k_normalizer()(raw_k + self.k_lora(x)).reshape(
            *x.shape[:2], self.num_heads, self.head_dim
        )
        geometry_v = (raw_v + self.v_lora(x)).reshape(
            *x.shape[:2], self.num_heads, self.head_dim
        )
        geometry_heads = self.geometry(
            geometry_q,
            geometry_k,
            geometry_v,
            grid_sizes=grid_sizes,
            arm_transform=arm_transform,
            arm_inverse=arm_inverse,
            arm_present=arm_present,
            seq_lens=seq_lens,
        )
        geometry_flat = geometry_heads.flatten(2)
        if getattr(self.base.o, "bias", None) is None:
            frozen_o = self.base.o(geometry_flat)
        else:
            frozen_o = F.linear(geometry_flat, self.base.o.weight, bias=None)
        geometry_output = frozen_o + self.o_lora(geometry_flat)
        gated = geometry_output.float() * self.channel_gate.view(1, 1, -1)
        return original_output + gated.to(dtype=original_output.dtype)

    def _q_normalizer(self) -> Callable[[Tensor], Tensor]:
        return self.base.q_norm if hasattr(self.base, "q_norm") else self.base.norm_q

    def _k_normalizer(self) -> Callable[[Tensor], Tensor]:
        return self.base.k_norm if hasattr(self.base, "k_norm") else self.base.norm_k

    def enable_geometry_training(self) -> None:
        """Restore only the v7.1 LoRA families and channel gate after freezing Wan."""

        self.base.requires_grad_(False)
        self.geometry.requires_grad_(False)
        for module in (self.q_lora, self.k_lora, self.v_lora, self.o_lora):
            module.requires_grad_(True)
        self.channel_gate.requires_grad_(True)
