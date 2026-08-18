"""Arm-grouped SE(3) attention utilities for the bounded Wan v7 probe.

The module deliberately contains no Wan runtime imports.  The caller supplies
the production RoPE and attention callables so this branch cannot accidentally
substitute a second attention implementation.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


AttentionCallable = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
RopeApplyCallable = Callable[[Tensor, Tensor, object], Tensor]


def apply_group_action(value: Tensor, matrix: Tensor, *, transpose: bool = False) -> Tensor:
    """Apply ``I_(head_dim / 4) kron matrix`` without materializing it.

    The last value dimension is interpreted as consecutive 4-vectors.  Vectors
    use the column convention, so ``matrix`` computes ``A @ vector`` and the
    transpose flag computes ``A.T @ vector``.  Leading dimensions may broadcast
    between ``value`` and ``matrix``.
    """

    if value.ndim < 1:
        raise ValueError("value must have a head dimension")
    if value.shape[-1] % 4:
        raise ValueError("head_dim must be divisible by four")
    if matrix.ndim < 2 or matrix.shape[-2:] != (4, 4):
        raise ValueError("matrix must end with shape (4, 4)")
    if not torch.isfinite(matrix).all():
        raise ValueError("matrix must be finite")

    # FP32 keeps SE(3) contractions stable under BF16 training.  Preserve FP64
    # when explicitly requested by an analytic contract test.
    work_dtype = torch.promote_types(value.dtype, matrix.dtype)
    if work_dtype in (torch.float16, torch.bfloat16):
        work_dtype = torch.float32
    blocks = value.to(dtype=work_dtype).reshape(*value.shape[:-1], value.shape[-1] // 4, 4)
    action = matrix.to(dtype=work_dtype).transpose(-1, -2) if transpose else matrix.to(dtype=work_dtype)
    try:
        output = torch.einsum("...rc,...nc->...nr", action, blocks)
    except RuntimeError as exc:
        raise ValueError("matrix leading dimensions are not broadcastable to value") from exc
    return output.reshape_as(value).to(dtype=value.dtype)


def invert_arm_transforms(arm_transform: Tensor) -> Tensor:
    """Build the inverse action map once for a cached SE(3) condition.

    Callers that install multiple v7 wrappers should invoke this once per batch
    condition and pass the result as ``arm_inverse`` to every selected block.
    """

    if arm_transform.ndim != 5 or arm_transform.shape[1] != 2 or arm_transform.shape[-2:] != (4, 4):
        raise ValueError("arm_transform must have shape (B, 2, T, 4, 4)")
    if not torch.isfinite(arm_transform).all():
        raise ValueError("arm_transform must be finite")
    try:
        inverse = torch.linalg.inv(arm_transform.float())
    except RuntimeError as exc:
        raise ValueError("arm_transform must be invertible") from exc
    if not torch.isfinite(inverse).all():
        raise ValueError("arm_transform must be invertible and finite")
    return inverse


class ArmGroupedSE3Geometry(nn.Module):
    """A parallel, fixed-ownership SE(3) attention branch.

    Heads ``[0, left_head_count)`` always receive the left-arm condition and
    the remaining heads receive the right-arm condition.  The returned heads
    are presence-masked *before* Wan's frozen output projection.
    """

    def __init__(
        self,
        *,
        attention_fn: AttentionCallable,
        num_heads: int = 24,
        head_dim: int = 128,
        left_head_count: int | None = None,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if head_dim <= 0 or head_dim % 4:
            raise ValueError("head_dim must be positive and divisible by four")
        left = num_heads // 2 if left_head_count is None else left_head_count
        if not 0 < left < num_heads:
            raise ValueError("left_head_count must split the head groups")
        self.attention_fn = attention_fn
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.left_head_count = int(left)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        grid_sizes: Tensor,
        arm_transform: Tensor,
        arm_inverse: Tensor | None = None,
        arm_present: Tensor,
        seq_lens: Tensor,
    ) -> Tensor:
        """Return presence-masked geometry heads of shape ``(B,L,H,D)``."""

        self._validate_inputs(q, k, v, grid_sizes, arm_transform, arm_inverse, arm_present, seq_lens)
        # The caller supplies this shared tensor when one condition is consumed
        # by multiple selected blocks.  The fallback keeps the pure operator
        # independently usable, while v7 integration must precompute it once.
        if arm_inverse is None:
            arm_inverse = invert_arm_transforms(arm_transform)
        token_transform, token_inverse, token_present, token_valid = self._expand_condition(
            q.shape[1], grid_sizes, arm_transform, arm_inverse, arm_present, seq_lens
        )

        left_transform = token_transform[:, :, 0].unsqueeze(2)
        right_transform = token_transform[:, :, 1].unsqueeze(2)
        left_inverse = token_inverse[:, :, 0].unsqueeze(2)
        right_inverse = token_inverse[:, :, 1].unsqueeze(2)

        split = self.left_head_count
        q_geometry = torch.cat(
            (
                apply_group_action(q[:, :, :split], left_transform, transpose=True),
                apply_group_action(q[:, :, split:], right_transform, transpose=True),
            ),
            dim=2,
        )
        k_geometry = torch.cat(
            (
                apply_group_action(k[:, :, :split], left_inverse),
                apply_group_action(k[:, :, split:], right_inverse),
            ),
            dim=2,
        )
        v_geometry = torch.cat(
            (
                apply_group_action(v[:, :, :split], left_inverse),
                apply_group_action(v[:, :, split:], right_inverse),
            ),
            dim=2,
        )

        geometry_heads = self.attention_fn(q_geometry, k_geometry, v_geometry, seq_lens)
        if geometry_heads.shape != q.shape:
            raise ValueError("attention_fn must return the same shape as q")
        if not torch.isfinite(geometry_heads).all():
            raise ValueError("attention_fn returned non-finite geometry heads")

        output = torch.cat(
            (
                apply_group_action(geometry_heads[:, :, :split], left_transform),
                apply_group_action(geometry_heads[:, :, split:], right_transform),
            ),
            dim=2,
        )
        head_present = torch.cat(
            (
                token_present[:, :, 0:1].expand(-1, -1, split),
                token_present[:, :, 1:2].expand(-1, -1, self.num_heads - split),
            ),
            dim=2,
        )
        keep = (head_present & token_valid.unsqueeze(-1)).unsqueeze(-1)
        return torch.where(keep, output, torch.zeros((), dtype=output.dtype, device=output.device))

    def _validate_inputs(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        grid_sizes: Tensor,
        arm_transform: Tensor,
        arm_inverse: Tensor | None,
        arm_present: Tensor,
        seq_lens: Tensor,
    ) -> None:
        if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
            raise ValueError("q, k, and v must share shape (B, L, H, D)")
        batch, _, heads, width = q.shape
        if heads != self.num_heads or width != self.head_dim:
            raise ValueError(
                f"expected q/k/v head shape (*, *, {self.num_heads}, {self.head_dim})"
            )
        if grid_sizes.shape != (batch, 3):
            raise ValueError("grid_sizes must have shape (B, 3)")
        if arm_transform.ndim != 5 or arm_transform.shape[:2] != (batch, 2) or arm_transform.shape[-2:] != (4, 4):
            raise ValueError("arm_transform must have shape (B, 2, T, 4, 4)")
        if arm_present.shape != arm_transform.shape[:3]:
            raise ValueError("arm_present must have shape (B, 2, T)")
        if arm_inverse is not None and arm_inverse.shape != arm_transform.shape:
            raise ValueError("arm_inverse must have the same shape as arm_transform")
        if arm_inverse is not None and arm_inverse.device != arm_transform.device:
            raise ValueError("arm_inverse must be on the same device as arm_transform")
        if arm_transform.dtype != torch.float32:
            raise ValueError("arm_transform must remain float32 from the v7 condition cache")
        if arm_inverse is not None and arm_inverse.dtype != torch.float32:
            raise ValueError("arm_inverse must remain float32 from the v7 condition cache")
        if seq_lens.shape != (batch,):
            raise ValueError("seq_lens must have shape (B,)")
        if not torch.isfinite(arm_transform).all():
            raise ValueError("arm_transform must be finite")
        if arm_inverse is not None and not torch.isfinite(arm_inverse).all():
            raise ValueError("arm_inverse must be finite")
        if arm_inverse is not None:
            # The cached inverse is intentionally FP32.  Validate it against
            # the analytic rigid inverse in FP64, per cached matrix.  A batch
            # member with a large translation must not relax another member's
            # mismatch tolerance.
            transform64 = arm_transform.double()
            inverse64 = arm_inverse.double()
            rotation = transform64[..., :3, :3]
            translation = transform64[..., :3, 3:4]
            expected_inverse = torch.eye(
                4, dtype=torch.float64, device=arm_transform.device
            ).expand_as(transform64).clone()
            expected_inverse[..., :3, :3] = rotation.transpose(-1, -2)
            expected_inverse[..., :3, 3:4] = -(rotation.transpose(-1, -2) @ translation)
            local_magnitude = torch.stack(
                (
                    transform64.abs().amax(dim=(-2, -1)),
                    inverse64.abs().amax(dim=(-2, -1)),
                    expected_inverse.abs().amax(dim=(-2, -1)),
                )
            ).amax(dim=0)
            tolerance = torch.maximum(
                torch.full_like(local_magnitude, 1e-4),
                2.0 * torch.finfo(torch.float32).eps * local_magnitude,
            )
            residual = (inverse64 - expected_inverse).abs().amax(dim=(-2, -1))
            if torch.any(residual > tolerance):
                raise ValueError("arm_inverse does not match arm_transform")
        if not torch.isfinite(q).all() or not torch.isfinite(k).all() or not torch.isfinite(v).all():
            raise ValueError("q, k, and v must be finite")
        if not torch.all(grid_sizes > 0):
            raise ValueError("grid_sizes must be positive")
        if not torch.all(grid_sizes[:, 0] == arm_transform.shape[2]):
            raise ValueError("arm_transform time dimension must match grid_sizes[:, 0]")
        if not torch.all((seq_lens >= 0) & (seq_lens <= q.shape[1])):
            raise ValueError("seq_lens must be within the token dimension")

    @staticmethod
    def _expand_condition(
        token_count: int,
        grid_sizes: Tensor,
        arm_transform: Tensor,
        arm_inverse: Tensor,
        arm_present: Tensor,
        seq_lens: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Expand time-major latent-frame data over visual spatial tokens only."""

        batch = arm_transform.shape[0]
        device = arm_transform.device
        identity = torch.eye(4, dtype=arm_transform.dtype, device=device)
        transforms = identity.reshape(1, 1, 1, 4, 4).repeat(batch, token_count, 2, 1, 1)
        inverses = identity.reshape(1, 1, 1, 4, 4).repeat(batch, token_count, 2, 1, 1)
        present = torch.zeros(batch, token_count, 2, dtype=torch.bool, device=device)
        valid = torch.zeros(batch, token_count, dtype=torch.bool, device=device)
        for index in range(batch):
            time, height, width = (int(value) for value in grid_sizes[index].tolist())
            visual_tokens = time * height * width
            legal_tokens = min(token_count, visual_tokens, int(seq_lens[index].item()))
            if legal_tokens == 0:
                continue
            frame_index = torch.arange(legal_tokens, device=device) // (height * width)
            transforms[index, :legal_tokens] = arm_transform[index, :, frame_index].permute(1, 0, 2, 3)
            inverses[index, :legal_tokens] = arm_inverse[index, :, frame_index].permute(1, 0, 2, 3)
            present[index, :legal_tokens] = arm_present[index, :, frame_index].transpose(0, 1).bool()
            valid[index, :legal_tokens] = True
        return transforms, inverses, present, valid


class SE3AugmentedSelfAttention(nn.Module):
    """Refactor one frozen Wan self-attention module with a zero-gated branch."""

    def __init__(
        self,
        base: nn.Module,
        *,
        attention_fn: AttentionCallable,
        rope_apply_fn: RopeApplyCallable,
        num_heads: int = 24,
        head_dim: int = 128,
    ) -> None:
        super().__init__()
        for name in ("q", "k", "v", "o"):
            if not hasattr(base, name):
                raise TypeError(f"base attention must expose {name}")
        if not (hasattr(base, "q_norm") and hasattr(base, "k_norm")) and not (
            hasattr(base, "norm_q") and hasattr(base, "norm_k")
        ):
            raise TypeError("base attention must expose q_norm/k_norm or norm_q/norm_k")
        self.base = base
        self.base.requires_grad_(False)
        self.geometry = ArmGroupedSE3Geometry(
            attention_fn=attention_fn,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        self.rope_apply_fn = rope_apply_fn
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.gate = nn.Parameter(torch.zeros(num_heads, head_dim, dtype=torch.float32))

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
        """Run attention and release a checkpoint lease after consumption.

        Non-reentrant activation checkpointing may terminate a replay early
        with an internal exception after the tensors needed for backward have
        been saved.  A module forward hook is not guaranteed to run on that
        path, so the lease lives in this ``finally`` instead.
        """

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
        if x.ndim != 3:
            raise ValueError("x must have shape (B, L, width)")
        expected_width = self.num_heads * self.head_dim
        if x.shape[-1] != expected_width:
            raise ValueError(f"expected hidden width {expected_width}")

        # This is the only projection / QK-normalization evaluation.  The
        # resulting tensors are forked before the original RoPE boundary.
        q = self.base.q(x).reshape(*x.shape[:2], self.num_heads, self.head_dim)
        k = self.base.k(x).reshape(*x.shape[:2], self.num_heads, self.head_dim)
        v = self.base.v(x).reshape(*x.shape[:2], self.num_heads, self.head_dim)
        q = self._q_normalizer()(q)
        k = self._k_normalizer()(k)
        pre_rope_q, pre_rope_k, pre_rope_v = q, k, v

        # RoPE implementations are allowed to mutate their Q/K arguments;
        # geometry must always consume independent normalized pre-RoPE tensors.
        # Official Wan applies RoPE to Q and K independently and requires the
        # latent grid as its second argument.
        original_q = self.rope_apply_fn(pre_rope_q.clone(), grid_sizes, freqs)
        original_k = self.rope_apply_fn(pre_rope_k.clone(), grid_sizes, freqs)
        original_heads = self.geometry.attention_fn(original_q, original_k, pre_rope_v, seq_lens)
        if original_heads.shape != pre_rope_q.shape:
            raise ValueError("attention_fn must return the same shape as q")
        original_output = self.base.o(original_heads.flatten(2))

        geometry_heads = self.geometry(
            pre_rope_q,
            pre_rope_k,
            pre_rope_v,
            grid_sizes=grid_sizes,
            arm_transform=arm_transform,
            arm_inverse=arm_inverse,
            arm_present=arm_present,
            seq_lens=seq_lens,
        )
        gated_heads = geometry_heads.float() * self.gate.unsqueeze(0).unsqueeze(0)
        # Reuse Wan's frozen output weight but not its bias.  The original
        # path already contributes that bias once; adding it again would make
        # a zero gate differ from the unwrapped Wan attention.
        geometry_output = torch.nn.functional.linear(
            gated_heads.flatten(2).to(dtype=original_output.dtype),
            self.base.o.weight,
            bias=None,
        )
        return original_output + geometry_output

    def _q_normalizer(self) -> Callable[[Tensor], Tensor]:
        return self.base.q_norm if hasattr(self.base, "q_norm") else self.base.norm_q

    def _k_normalizer(self) -> Callable[[Tensor], Tensor]:
        return self.base.k_norm if hasattr(self.base, "k_norm") else self.base.norm_k
