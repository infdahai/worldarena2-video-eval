"""Factorized action relations inside Wan native fused self-attention."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import inspect
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


AttentionCallable = Callable[..., Tensor]
RopeApplyCallable = Callable[[Tensor, Tensor, object], Tensor]


@dataclass(frozen=True)
class RelationCondition:
    """Per-arm action state aligned to destination visual latent slots."""

    anchored_se3: Tensor
    velocity: Tensor
    uv: Tensor
    gripper: Tensor
    arm_present: Tensor
    motion_active: Tensor
    support: Tensor
    destination_time: Tensor


class ActionRelationEncoder(nn.Module):
    """Shared compact state encoder used by every selected Wan block."""

    def __init__(self, *, state_width: int = 20, encoded_width: int = 128) -> None:
        super().__init__()
        if state_width != 20:
            raise ValueError("v10 relation state width must be 20")
        if encoded_width <= 0:
            raise ValueError("encoded_width must be positive")
        self.state_width = int(state_width)
        self.encoded_width = int(encoded_width)
        self.register_buffer("input_mean", torch.zeros(state_width, dtype=torch.float32))
        self.register_buffer("input_scale", torch.ones(state_width, dtype=torch.float32))
        self.network = nn.Sequential(
            nn.Linear(state_width, encoded_width),
            nn.SiLU(),
            nn.Linear(encoded_width, encoded_width),
        )

    def set_normalization(self, mean: Tensor, scale: Tensor) -> None:
        if mean.shape != (self.state_width,) or scale.shape != (self.state_width,):
            raise ValueError("v10 relation normalization shape differs")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
            raise ValueError("v10 relation normalization must be finite")
        if torch.any(scale <= 0):
            raise ValueError("v10 relation normalization scale must be positive")
        self.input_mean.copy_(mean.float())
        self.input_scale.copy_(scale.float())

    def forward(self, state: Tensor, present: Tensor) -> Tensor:
        if state.ndim != 4 or state.shape[-2:] != (2, self.state_width):
            raise ValueError("relation state must have shape (B,L,2,20)")
        if present.shape != state.shape[:-1] or present.dtype != torch.bool:
            raise ValueError("relation presence must have shape (B,L,2)")
        if not torch.isfinite(state).all():
            raise ValueError("relation state must be finite")
        work = state.float()
        work = (work - self.input_mean) / self.input_scale
        encoded = self.network(work.to(dtype=self.network[0].weight.dtype))
        return encoded * present[..., None].to(dtype=encoded.dtype)


def assert_augmented_attention_supported(attention_fn: AttentionCallable) -> None:
    """Reject callables that cannot preserve Wan's native softmax scale."""

    try:
        signature = inspect.signature(attention_fn)
    except (TypeError, ValueError) as exc:
        raise TypeError("v10 attention callable must expose a Python signature") from exc
    parameters = signature.parameters
    has_keywords = any(
        item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()
    )
    required = {"q", "k", "v", "k_lens", "window_size", "softmax_scale"}
    if not has_keywords and not required <= set(parameters):
        raise TypeError("v10 attention callable lacks explicit softmax_scale support")


class FactorizedRelationalSelfAttention(nn.Module):
    """Wan native self-attention with rank-factorized arm relation scores."""

    def __init__(
        self,
        base: nn.Module,
        *,
        state_encoder: ActionRelationEncoder,
        attention_fn: AttentionCallable,
        rope_apply_fn: RopeApplyCallable,
        num_heads: int = 24,
        head_dim: int = 128,
        relation_rank: int = 16,
        expected_latent_times: int = 21,
        hidden_eef_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if not all(hasattr(base, name) for name in ("q", "k", "v", "o")):
            raise TypeError("base attention must expose q/k/v/o")
        if num_heads <= 0 or num_heads % 3:
            raise ValueError("v10 attention heads must split into three equal banks")
        if head_dim <= 0 or relation_rank <= 0 or expected_latent_times <= 1:
            raise ValueError("v10 attention dimensions must be positive")
        assert_augmented_attention_supported(attention_fn)
        self.base = base
        self.state_encoder = state_encoder
        self.attention_fn = attention_fn
        self.rope_apply_fn = rope_apply_fn
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.relation_rank = int(relation_rank)
        self.expected_latent_times = int(expected_latent_times)
        self.hidden_eef_head = hidden_eef_head
        self.arm_head_count = num_heads // 3
        relation_width = self.arm_head_count * relation_rank
        self.relation_q = nn.Linear(state_encoder.encoded_width, relation_width)
        self.relation_k = nn.Linear(state_encoder.encoded_width, relation_width)
        self.relation_gate = nn.Parameter(
            torch.zeros(2 * self.arm_head_count, dtype=torch.float32)
        )
        self.last_augmented_head_dim = head_dim + relation_rank
        self.register_buffer("last_left_relation", torch.zeros(self.arm_head_count), persistent=False)
        self.register_buffer("last_right_relation", torch.zeros(self.arm_head_count), persistent=False)
        self.register_buffer("last_global_relation", torch.zeros(self.arm_head_count), persistent=False)
        self.last_hidden_eef_prediction: Tensor | None = None
        self.bound_condition: RelationCondition | None = None
        self._condition_token: object | None = None
        self._checkpoint_condition: RelationCondition | None = None
        self._checkpoint_token: object | None = None
        self.condition_use_count = 0

    def _apply(self, fn, recurse: bool = True):  # type: ignore[override]
        result = super()._apply(fn, recurse=recurse)
        self.relation_gate.data = self.relation_gate.data.float()
        if self.relation_gate.grad is not None:
            self.relation_gate.grad.data = self.relation_gate.grad.data.float()
        return result

    def _q_norm(self) -> nn.Module:
        if hasattr(self.base, "norm_q"):
            return self.base.norm_q
        if hasattr(self.base, "q_norm"):
            return self.base.q_norm
        raise TypeError("base attention lacks Q normalization")

    def _k_norm(self) -> nn.Module:
        if hasattr(self.base, "norm_k"):
            return self.base.norm_k
        if hasattr(self.base, "k_norm"):
            return self.base.k_norm
        raise TypeError("base attention lacks K normalization")

    def _validate_and_pack_state(
        self,
        visual: Tensor,
        seq_lens: Tensor,
        grid_sizes: Tensor,
        condition: RelationCondition,
    ) -> tuple[Tensor, Tensor]:
        if visual.ndim != 3 or visual.shape[-1] != self.num_heads * self.head_dim:
            raise ValueError("visual must have shape (B,L,num_heads*head_dim)")
        batch = visual.shape[0]
        if grid_sizes.shape != (batch, 3) or seq_lens.shape != (batch,):
            raise ValueError("grid_sizes/seq_lens batch shape differs")
        time = self.expected_latent_times
        common = (batch, time, 2)
        expected = {
            "anchored_se3": common + (6,),
            "velocity": common + (6,),
            "uv": common + (2,),
            "gripper": common + (2,),
            "arm_present": common,
            "motion_active": common,
            "destination_time": (batch, time),
        }
        for name, shape in expected.items():
            value = getattr(condition, name)
            if value.shape != shape:
                raise ValueError(f"{name} violates v10 latent time contract")
        if condition.arm_present.dtype != torch.bool or condition.motion_active.dtype != torch.bool:
            raise ValueError("relation presence/activity must be boolean")
        if condition.support.ndim != 5 or condition.support.shape[:3] != common:
            raise ValueError("support must have shape (B,T,2,H,W)")
        numeric = (
            condition.anchored_se3,
            condition.velocity,
            condition.uv,
            condition.gripper,
            condition.support,
            condition.destination_time,
        )
        if any(not torch.isfinite(value).all() for value in numeric):
            raise ValueError("v10 relation condition must be finite")
        slots = torch.arange(time, device=condition.destination_time.device)
        if not torch.equal(condition.destination_time, slots.reshape(1, time).expand(batch, -1)):
            raise ValueError("destination time slots must remain fixed")

        height, width = condition.support.shape[-2:]
        grids = grid_sizes.detach().cpu().to(dtype=torch.int64).tolist()
        lengths: list[int] = []
        for latent_time, grid_height, grid_width in grids:
            if latent_time != time:
                raise ValueError("grid latent time differs from v10 contract")
            if (grid_height, grid_width) != (height, width):
                raise ValueError("support grid differs from visual grid")
            lengths.append(latent_time * grid_height * grid_width)
        if lengths != seq_lens.detach().cpu().to(dtype=torch.int64).tolist():
            raise ValueError("seq_lens differs from visual grid")
        if any(length > visual.shape[1] for length in lengths):
            raise ValueError("visual sequence is shorter than its grid")

        packed = visual.new_zeros((batch, visual.shape[1], 2, 20), dtype=torch.float32)
        packed_present = torch.zeros(
            (batch, visual.shape[1], 2), dtype=torch.bool, device=visual.device
        )
        for index, length in enumerate(lengths):
            spatial = height * width
            def expand(value: Tensor) -> Tensor:
                return value[index].to(device=visual.device).unsqueeze(1).expand(time, spatial, *value.shape[2:])

            support = condition.support[index].to(device=visual.device).permute(0, 2, 3, 1).reshape(time, spatial, 2, 1)
            destination = condition.destination_time[index].to(device=visual.device, dtype=torch.float32)
            destination = (destination / (time - 1) * 2 - 1).reshape(time, 1, 1, 1).expand(time, spatial, 2, 1)
            state = torch.cat(
                (
                    expand(condition.anchored_se3),
                    expand(condition.velocity),
                    expand(condition.uv),
                    expand(condition.gripper),
                    expand(condition.arm_present).unsqueeze(-1).float(),
                    expand(condition.motion_active).unsqueeze(-1).float(),
                    support,
                    destination,
                ),
                dim=-1,
            ).reshape(length, 2, 20)
            present = expand(condition.arm_present).reshape(length, 2)
            packed[index, :length] = state
            packed_present[index, :length] = present
        return packed, packed_present

    def _relation_factors(self, state: Tensor, present: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self.state_encoder(state, present)
        batch, length = encoded.shape[:2]
        q_arms = self.relation_q(encoded).reshape(
            batch, length, 2, self.arm_head_count, self.relation_rank
        )
        k_arms = self.relation_k(encoded).reshape_as(q_arms)
        zeros = torch.zeros_like(q_arms[:, :, 0])
        q_relation = torch.cat((q_arms[:, :, 0], q_arms[:, :, 1], zeros), dim=2)
        k_relation = torch.cat((k_arms[:, :, 0], k_arms[:, :, 1], zeros), dim=2)
        gate = torch.cat(
            (
                self.relation_gate,
                torch.zeros(self.arm_head_count, device=self.relation_gate.device),
            )
        ).reshape(1, 1, self.num_heads, 1)
        q_relation = q_relation * gate.to(device=q_relation.device, dtype=q_relation.dtype)
        with torch.no_grad():
            qk = q_relation.detach().abs().sum(dim=(0, 1, 3)) + k_relation.detach().abs().sum(dim=(0, 1, 3))
            self.last_left_relation.copy_(qk[: self.arm_head_count].float())
            self.last_right_relation.copy_(qk[self.arm_head_count : 2 * self.arm_head_count].float())
            self.last_global_relation.copy_(qk[2 * self.arm_head_count :].float())
        return q_relation, k_relation

    def forward(
        self,
        visual: Tensor,
        seq_lens: Tensor,
        grid_sizes: Tensor,
        freqs: object,
        *,
        relation_condition: RelationCondition,
        checkpoint_replay_release: Callable[[], None] | None = None,
    ) -> Tensor:
        try:
            state, present = self._validate_and_pack_state(
                visual, seq_lens, grid_sizes, relation_condition
            )
            work = visual.to(dtype=self.base.q.weight.dtype)
            batch, length = work.shape[:2]
            q = self._q_norm()(self.base.q(work)).reshape(
                batch, length, self.num_heads, self.head_dim
            )
            k = self._k_norm()(self.base.k(work)).reshape_as(q)
            v = self.base.v(work).reshape_as(q)
            q = self.rope_apply_fn(q, grid_sizes, freqs)
            k = self.rope_apply_fn(k, grid_sizes, freqs)
            q_relation, k_relation = self._relation_factors(state, present)
            q_aug = torch.cat((q, q_relation.to(dtype=q.dtype)), dim=-1)
            k_aug = torch.cat((k, k_relation.to(dtype=k.dtype)), dim=-1)
            v_aug = F.pad(v, (0, self.relation_rank))
            attended = self.attention_fn(
                q=q_aug,
                k=k_aug,
                v=v_aug,
                k_lens=seq_lens,
                window_size=getattr(self.base, "window_size", (-1, -1)),
                softmax_scale=1.0 / math.sqrt(self.head_dim),
            )
            if attended.shape != v_aug.shape or not torch.isfinite(attended).all():
                raise ValueError("fused attention returned invalid augmented heads")
            native_heads = attended[..., : self.head_dim]
            output = self.base.o(native_heads.flatten(2).to(dtype=self.base.o.weight.dtype))
            if self.hidden_eef_head is not None:
                logits = self.hidden_eef_head(output)
                height, width = relation_condition.support.shape[-2:]
                if output.shape[1] != self.expected_latent_times * height * width:
                    raise ValueError("hidden EEF head requires an unpadded production token grid")
                self.last_hidden_eef_prediction = logits.reshape(
                    output.shape[0], self.expected_latent_times, height, width, 2
                ).permute(0, 1, 4, 2, 3)
            else:
                self.last_hidden_eef_prediction = None
            return output
        finally:
            if checkpoint_replay_release is not None:
                checkpoint_replay_release()
