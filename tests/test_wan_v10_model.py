from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
from torch import nn

from worldarena_baseline.wan_v10_attention import RelationCondition  # noqa: E402
from worldarena_baseline.wan_v10_model import (  # noqa: E402
    V10_BLOCKS,
    ParentPlusRelationalWan,
    install_v10_relational_band,
    relation_gates_enabled,
    v10_initialized_state_sha256,
    v10_trainable_parameter_names,
)


WIDTH = 24


def _fused(*, q, k, v, k_lens, window_size, softmax_scale):
    del k_lens, window_size
    scores = torch.einsum("blhd,bshd->bhls", q, k) * softmax_scale
    probabilities = torch.softmax(scores.float(), dim=-1).to(v.dtype)
    return torch.einsum("bhls,bshd->blhd", probabilities, v)


def _rope(value, grid_sizes, freqs):
    del grid_sizes, freqs
    return value


class _Attention(nn.Module):
    _v10_test_fixture = True

    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(WIDTH, WIDTH)
        self.k = nn.Linear(WIDTH, WIDTH)
        self.v = nn.Linear(WIDTH, WIDTH)
        self.o = nn.Linear(WIDTH, WIDTH)
        self.norm_q = nn.Identity()
        self.norm_k = nn.Identity()
        self.window_size = (-1, -1)

    def forward(self, x, seq_lens, grid_sizes, freqs):
        del seq_lens, grid_sizes, freqs
        return self.o(self.q(x) + self.k(x) + self.v(x))


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        return self.self_attn(x, seq_lens, grid_sizes, freqs)


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(30))

    def forward(self, x, t, context, seq_len, y=None):
        del t, context, y
        seq_lens = torch.tensor([seq_len], device=x.device)
        grid_sizes = torch.tensor([[3, 1, 2]], device=x.device)
        for block in self.blocks:
            x = block(x, seq_lens, grid_sizes, None)
        return x


class _Parent(nn.Module):
    injection_points = (0, 8, 16, 24)
    raster_support_gating = True

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, raster, timestep, *, seq_len, condition_support, action_present):
        del raster, timestep, condition_support, action_present
        return {point: torch.zeros(1, seq_len, WIDTH) for point in self.injection_points}


def _condition() -> RelationCondition:
    return RelationCondition(
        anchored_se3=torch.zeros(1, 3, 2, 6),
        velocity=torch.zeros(1, 3, 2, 6),
        uv=torch.zeros(1, 3, 2, 2),
        gripper=torch.zeros(1, 3, 2, 2),
        arm_present=torch.ones(1, 3, 2, dtype=torch.bool),
        motion_active=torch.ones(1, 3, 2, dtype=torch.bool),
        support=torch.ones(1, 3, 2, 1, 2),
        destination_time=torch.arange(3).reshape(1, 3),
    )


def _install(seed: int = 19):
    torch.manual_seed(5)
    backbone = _Backbone()
    wrappers = install_v10_relational_band(
        backbone,
        V10_BLOCKS,
        attention_fn=_fused,
        rope_apply_fn=_rope,
        initialization_seed=seed,
        num_heads=6,
        head_dim=4,
        relation_rank=2,
        expected_latent_times=3,
        encoded_width=12,
    )
    model = ParentPlusRelationalWan(backbone, _Parent(), wrappers)
    return model, wrappers


def _inputs():
    return {
        "x": torch.randn(1, 6, WIDTH),
        "t": torch.zeros(1),
        "context": None,
        "seq_len": 6,
        "action_raster": torch.zeros(1, 10, 81, 60, 80),
        "condition_support": torch.ones(1, 2, 3, 1, 2),
        "action_present": torch.ones(1),
        "relation_condition": _condition(),
    }


def test_installs_exact_blocks_and_is_deterministic() -> None:
    first, wrappers = _install()
    second, _ = _install()
    assert V10_BLOCKS == tuple(range(6, 18))
    assert tuple(wrappers) == V10_BLOCKS
    assert v10_initialized_state_sha256(first) == v10_initialized_state_sha256(second)


def test_trainable_whitelist_contains_exact_four_families() -> None:
    model, _ = _install()
    names = v10_trainable_parameter_names(model)
    assert names
    assert any(name.startswith("relation_encoder.") for name in names)
    assert sum(name.endswith("relation_gate") for name in names) == 12
    assert sum("hidden_eef_head" in name for name in names) == 4
    assert all(not parameter.requires_grad for parameter in model.parent_adapter.parameters())


def test_zero_gates_match_native_and_nonzero_gates_change_output() -> None:
    model, wrappers = _install()
    values = _inputs()
    with relation_gates_enabled(model, enabled=False):
        zero = model(**values)
    with torch.no_grad():
        for wrapper in wrappers.values():
            wrapper.relation_gate.fill_(0.3)
    enabled = model(**values)
    assert not torch.equal(enabled, zero)
    with relation_gates_enabled(model, enabled=False):
        ablated = model(**values)
    torch.testing.assert_close(ablated, zero)


def test_forward_returns_two_hidden_eef_predictions_and_clears_binding() -> None:
    model, wrappers = _install()
    output = model(**_inputs())
    assert torch.isfinite(output).all()
    predictions = model.hidden_eef_predictions()
    assert set(predictions) == {11, 17}
    assert all(value.shape == (1, 3, 2, 1, 2) for value in predictions.values())
    assert all(wrapper.bound_condition is None for wrapper in wrappers.values())


def test_installation_rolls_back_on_late_invalid_block() -> None:
    backbone = _Backbone()
    originals = {index: backbone.blocks[index].self_attn for index in V10_BLOCKS}
    del backbone.blocks[17].self_attn.q
    with pytest.raises((TypeError, ValueError), match="q/k/v/o"):
        install_v10_relational_band(
            backbone,
            V10_BLOCKS,
            attention_fn=_fused,
            rope_apply_fn=_rope,
            initialization_seed=19,
            num_heads=6,
            head_dim=4,
            relation_rank=2,
            expected_latent_times=3,
            encoded_width=12,
        )
    assert {index: backbone.blocks[index].self_attn for index in V10_BLOCKS} == originals

