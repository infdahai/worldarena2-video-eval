from __future__ import annotations

import math

import pytest


torch = pytest.importorskip("torch")
from torch import nn

from worldarena_baseline.wan_v10_attention import (  # noqa: E402
    ActionRelationEncoder,
    FactorizedRelationalSelfAttention,
    RelationCondition,
)


class _BaseAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(24, 24, bias=False)
        self.k = nn.Linear(24, 24, bias=False)
        self.v = nn.Linear(24, 24, bias=False)
        self.o = nn.Linear(24, 24, bias=False)
        self.norm_q = nn.Identity()
        self.norm_k = nn.Identity()
        self.window_size = (-1, -1)
        with torch.no_grad():
            for projection in (self.q, self.k, self.v, self.o):
                projection.weight.copy_(torch.eye(24))


def _rope(value, grid_sizes, freqs):
    del grid_sizes, freqs
    return value


def _fused_attention(*, q, k, v, k_lens, window_size, softmax_scale):
    del k_lens, window_size
    scores = torch.einsum("blhd,bshd->bhls", q, k) * softmax_scale
    probability = torch.softmax(scores.float(), dim=-1).to(v.dtype)
    return torch.einsum("bhls,bshd->blhd", probability, v)


def _condition(*, right_present: bool = True) -> RelationCondition:
    batch, time, arms, height, width = 1, 3, 2, 1, 2
    anchored = torch.zeros(batch, time, arms, 6)
    anchored[:, :, 0, 0] = torch.tensor([0.0, 0.4, 0.8])
    anchored[:, :, 1, 1] = torch.tensor([0.0, -0.3, -0.6])
    velocity = torch.zeros_like(anchored)
    velocity[:, 1:] = anchored[:, 1:] - anchored[:, :-1]
    uv = torch.zeros(batch, time, arms, 2)
    uv[..., 0] = 0.25
    uv[..., 1] = 0.75
    gripper = torch.zeros(batch, time, arms, 2)
    present = torch.ones(batch, time, arms, dtype=torch.bool)
    present[:, :, 1] = right_present
    motion = torch.ones(batch, time, arms, dtype=torch.bool)
    support = torch.ones(batch, time, arms, height, width)
    destination_time = torch.arange(time).reshape(1, time)
    return RelationCondition(
        anchored_se3=anchored,
        velocity=velocity,
        uv=uv,
        gripper=gripper,
        arm_present=present,
        motion_active=motion,
        support=support,
        destination_time=destination_time,
    )


def _wrapper() -> FactorizedRelationalSelfAttention:
    torch.manual_seed(11)
    return FactorizedRelationalSelfAttention(
        _BaseAttention(),
        state_encoder=ActionRelationEncoder(state_width=20, encoded_width=12),
        attention_fn=_fused_attention,
        rope_apply_fn=_rope,
        num_heads=6,
        head_dim=4,
        relation_rank=2,
        expected_latent_times=3,
    ).double()


def _inputs():
    visual = torch.arange(1, 145, dtype=torch.float64).reshape(1, 6, 24) / 100
    seq_lens = torch.tensor([6], dtype=torch.long)
    grid_sizes = torch.tensor([[3, 1, 2]], dtype=torch.long)
    return visual, seq_lens, grid_sizes


def _native_reference(wrapper, visual, seq_lens, grid_sizes):
    base = wrapper.base
    q = base.norm_q(base.q(visual)).reshape(1, 6, 6, 4)
    k = base.norm_k(base.k(visual)).reshape_as(q)
    v = base.v(visual).reshape_as(q)
    heads = _fused_attention(
        q=q,
        k=k,
        v=v,
        k_lens=seq_lens,
        window_size=(-1, -1),
        softmax_scale=1 / math.sqrt(4),
    )
    return base.o(heads.flatten(2))


def test_zero_relation_gate_matches_native_attention() -> None:
    wrapper = _wrapper()
    visual, seq_lens, grid_sizes = _inputs()
    actual = wrapper(
        visual,
        seq_lens,
        grid_sizes,
        None,
        relation_condition=_condition(),
    )
    expected = _native_reference(wrapper, visual, seq_lens, grid_sizes)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    assert wrapper.last_augmented_head_dim == 6


def test_arm_head_banks_are_disjoint_and_global_relation_is_zero() -> None:
    wrapper = _wrapper()
    visual, seq_lens, grid_sizes = _inputs()
    with torch.no_grad():
        wrapper.relation_gate.fill_(0.5)
    baseline = wrapper(
        visual, seq_lens, grid_sizes, None, relation_condition=_condition()
    )
    changed = _condition()
    changed.anchored_se3[:, :, 1, 2] = 1000
    actual = wrapper(
        visual, seq_lens, grid_sizes, None, relation_condition=changed
    )
    assert torch.equal(actual[..., :8], baseline[..., :8])
    assert not torch.equal(actual[..., 8:16], baseline[..., 8:16])
    assert torch.equal(actual[..., 16:], baseline[..., 16:])
    assert torch.count_nonzero(wrapper.last_global_relation).item() == 0


def test_absent_arm_relation_is_exact_zero_and_support_may_overlap() -> None:
    wrapper = _wrapper()
    visual, seq_lens, grid_sizes = _inputs()
    condition = _condition(right_present=False)
    condition.support[:, :, 0].fill_(1)
    condition.support[:, :, 1].fill_(1)
    wrapper(
        visual, seq_lens, grid_sizes, None, relation_condition=condition
    )
    assert torch.count_nonzero(wrapper.last_right_relation).item() == 0
    assert torch.count_nonzero(wrapper.last_left_relation).item() > 0


def test_nonzero_gate_backpropagates_to_native_and_relation_families() -> None:
    wrapper = _wrapper()
    visual, seq_lens, grid_sizes = _inputs()
    visual.requires_grad_(True)
    with torch.no_grad():
        wrapper.relation_gate.fill_(0.2)
    wrapper(
        visual, seq_lens, grid_sizes, None, relation_condition=_condition()
    ).square().mean().backward()
    assert wrapper.relation_gate.grad is not None
    assert torch.count_nonzero(wrapper.relation_gate.grad).item() > 0
    for projection in (wrapper.base.q, wrapper.base.k, wrapper.base.v, wrapper.base.o):
        assert projection.weight.grad is not None
        assert torch.count_nonzero(projection.weight.grad).item() > 0
    for projection in (wrapper.relation_q, wrapper.relation_k):
        assert projection.weight.grad is not None
        assert torch.count_nonzero(projection.weight.grad).item() > 0


def test_input_contract_rejects_wrong_time_and_nonfinite_state() -> None:
    wrapper = _wrapper()
    visual, seq_lens, grid_sizes = _inputs()
    with pytest.raises(ValueError, match="latent time"):
        wrapper(
            visual[:, :4],
            torch.tensor([4]),
            torch.tensor([[2, 1, 2]]),
            None,
            relation_condition=_condition(),
        )
    bad = _condition()
    bad.uv[0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        wrapper(
            visual, seq_lens, grid_sizes, None, relation_condition=bad
        )


def test_relation_gate_stays_fp32_when_module_is_converted() -> None:
    wrapper = _wrapper().to(dtype=torch.bfloat16)
    assert wrapper.relation_gate.dtype == torch.float32

