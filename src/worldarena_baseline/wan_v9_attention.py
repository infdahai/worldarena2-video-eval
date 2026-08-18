"""Strict local action cross-attention for Wan v9."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PhaseLockedActionCrossAttention(nn.Module):
    """Attend each visual time only to its preceding four-token arm banks."""

    def __init__(
        self,
        *,
        visual_width: int = 3072,
        action_width: int = 256,
        inner_width: int = 768,
        num_heads: int = 8,
        expected_latent_times: int = 21,
    ) -> None:
        super().__init__()
        if min(visual_width, action_width, inner_width, num_heads, expected_latent_times) <= 0:
            raise ValueError("v9 attention dimensions must be positive")
        if num_heads % 2 or inner_width % num_heads:
            raise ValueError("v9 attention requires two equal arm head groups")
        if expected_latent_times < 2:
            raise ValueError("v9 attention needs at least two latent times")
        self.visual_width = int(visual_width)
        self.action_width = int(action_width)
        self.inner_width = int(inner_width)
        self.num_heads = int(num_heads)
        self.head_dim = inner_width // num_heads
        self.expected_latent_times = int(expected_latent_times)
        self.q = nn.Linear(visual_width, inner_width)
        self.k = nn.Linear(action_width, inner_width)
        self.v = nn.Linear(action_width, inner_width)
        self.o = nn.Linear(inner_width, visual_width)
        self.channel_gate = nn.Parameter(torch.zeros(visual_width, dtype=torch.float32))

    def _apply(self, fn, recurse: bool = True):  # type: ignore[override]
        result = super()._apply(fn, recurse=recurse)
        # Gate math and optimizer state remain FP32 even when the module's
        # projections are converted to BF16 for production.
        self.channel_gate.data = self.channel_gate.data.float()
        if self.channel_gate.grad is not None:
            self.channel_gate.grad.data = self.channel_gate.grad.data.float()
        return result

    def _validate(
        self,
        visual: Tensor,
        grid_sizes: Tensor,
        action_tokens: Tensor,
        arm_present: Tensor,
        seq_lens: Tensor | None,
    ) -> tuple[list[int], list[int]]:
        if visual.ndim != 3 or visual.shape[-1] != self.visual_width:
            raise ValueError("visual must have shape (batch,sequence,visual_width)")
        batch = visual.shape[0]
        if grid_sizes.shape != (batch, 3):
            raise ValueError("grid_sizes must have shape (batch,3)")
        if action_tokens.shape != (batch, self.expected_latent_times - 1, 2, 4, self.action_width):
            raise ValueError("action tokens violate the phase-locked four-token contract")
        if arm_present.shape != (batch, self.expected_latent_times - 1, 2) or arm_present.dtype != torch.bool:
            raise ValueError("arm_present violates the phase-locked transition contract")
        if not torch.isfinite(visual).all() or not torch.isfinite(action_tokens).all():
            raise ValueError("v9 attention inputs contain non-finite values")
        grids = grid_sizes.detach().cpu().to(dtype=torch.int64).tolist()
        times: list[int] = []
        lengths: list[int] = []
        for time, height, width in grids:
            if time != self.expected_latent_times:
                raise ValueError("latent time differs from the v9 contract")
            if height <= 0 or width <= 0:
                raise ValueError("spatial grid dimensions must be positive")
            times.append(time)
            lengths.append(time * height * width)
        if any(length > visual.shape[1] for length in lengths):
            raise ValueError("visual sequence is shorter than grid size")
        if seq_lens is not None:
            if seq_lens.shape != (batch,):
                raise ValueError("seq_lens must have shape (batch,)")
            actual = seq_lens.detach().cpu().to(dtype=torch.int64).tolist()
            if actual != lengths:
                raise ValueError("seq_lens differs from grid token counts")
        return times, lengths

    def forward(
        self,
        visual: Tensor,
        *,
        grid_sizes: Tensor,
        action_tokens: Tensor,
        arm_present: Tensor,
        seq_lens: Tensor | None = None,
    ) -> Tensor:
        times, lengths = self._validate(visual, grid_sizes, action_tokens, arm_present, seq_lens)
        compute_dtype = self.q.weight.dtype
        q = self.q(visual.to(dtype=compute_dtype)).reshape(
            visual.shape[0], visual.shape[1], self.num_heads, self.head_dim
        )
        k = self.k(action_tokens.to(dtype=self.k.weight.dtype)).reshape(
            action_tokens.shape[0], self.expected_latent_times - 1, 2, 4,
            self.num_heads, self.head_dim,
        )
        v = self.v(action_tokens.to(dtype=self.v.weight.dtype)).reshape_as(k)
        attended = torch.zeros_like(q)
        group_width = self.num_heads // 2
        for batch_index, (time, length) in enumerate(zip(times, lengths, strict=True)):
            spatial = length // time
            local_q = q[batch_index, :length].reshape(time, spatial, self.num_heads, self.head_dim)
            local_output = attended[batch_index, :length].reshape(time, spatial, self.num_heads, self.head_dim)
            for arm, head_start in ((0, 0), (1, group_width)):
                head_stop = head_start + group_width
                queries = local_q[1:, :, head_start:head_stop]
                keys = k[batch_index, : time - 1, arm, :, head_start:head_stop].permute(0, 2, 1, 3)
                values = v[batch_index, : time - 1, arm, :, head_start:head_stop].permute(0, 2, 1, 3)
                scores = torch.einsum("tshd,thmd->tshm", queries, keys) / math.sqrt(self.head_dim)
                probabilities = torch.softmax(scores.float(), dim=-1).to(dtype=values.dtype)
                result = torch.einsum("tshm,thmd->tshd", probabilities, values)
                presence = arm_present[batch_index, : time - 1, arm].reshape(time - 1, 1, 1, 1)
                local_output[1:, :, head_start:head_stop] = result * presence.to(dtype=result.dtype)
        projected = self.o(attended.flatten(2))
        temporal_mask = torch.zeros(
            visual.shape[:2], dtype=torch.bool, device=visual.device
        )
        for batch_index, (time, length) in enumerate(zip(times, lengths, strict=True)):
            spatial = length // time
            temporal_mask[batch_index, spatial:length] = True
        gated = projected.float() * self.channel_gate.reshape(1, 1, -1)
        gated = gated * temporal_mask[..., None]
        return gated.to(dtype=visual.dtype)

