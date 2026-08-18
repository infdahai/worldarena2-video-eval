from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from worldarena_baseline.wan_v71_attention import SE3GeometryLoRAAttention  # noqa: E402


def _attention(q, k, v, seq_lens):
    del seq_lens
    return v + 0.01 * q + 0.01 * k


def _rope(value, grid_sizes, freqs):
    del grid_sizes, freqs
    return value + 100


class _Base(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(8, 8, bias=False)
        self.k = nn.Linear(8, 8, bias=False)
        self.v = nn.Linear(8, 8, bias=False)
        self.o = nn.Linear(8, 8, bias=True)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        with torch.no_grad():
            for layer in (self.q, self.k, self.v, self.o):
                layer.weight.copy_(torch.eye(8))
            self.o.bias.fill_(2.0)

    def forward(self, x, seq_lens, grid_sizes, freqs):
        q = _rope(self.q(x).reshape(1, 3, 2, 4), grid_sizes, freqs)
        k = _rope(self.k(x).reshape(1, 3, 2, 4), grid_sizes, freqs)
        v = self.v(x).reshape(1, 3, 2, 4)
        return self.o(_attention(q, k, v, seq_lens).flatten(2))


def _kwargs(present: bool = True):
    return {
        "seq_lens": torch.tensor([3]),
        "grid_sizes": torch.tensor([[1, 1, 3]]),
        "freqs": "rope",
        "arm_transform": torch.eye(4, dtype=torch.float32)
        .reshape(1, 1, 1, 4, 4)
        .repeat(1, 2, 1, 1, 1),
        "arm_present": torch.full((1, 2, 1), present, dtype=torch.bool),
    }


def test_v71_zero_channel_gate_is_exact_frozen_wan_output() -> None:
    base = _Base()
    wrapper = SE3GeometryLoRAAttention(
        base,
        attention_fn=_attention,
        rope_apply_fn=_rope,
        num_heads=2,
        head_dim=4,
        rank=2,
    )
    x = torch.randn(1, 3, 8)

    assert torch.equal(wrapper(x, **_kwargs()), base(x, **{k: v for k, v in _kwargs().items() if k not in {"arm_transform", "arm_present"}}))
    assert torch.count_nonzero(wrapper.channel_gate) == 0


def test_v71_lora_is_geometry_only_and_all_four_families_receive_gradients() -> None:
    base = _Base()
    wrapper = SE3GeometryLoRAAttention(
        base,
        attention_fn=_attention,
        rope_apply_fn=_rope,
        num_heads=2,
        head_dim=4,
        rank=2,
    )
    wrapper.channel_gate.data.fill_(0.1)
    for delta in (wrapper.q_lora, wrapper.k_lora, wrapper.v_lora, wrapper.o_lora):
        delta.up.data.fill_(0.1)
    output = wrapper(torch.randn(1, 3, 8), **_kwargs())
    output.sum().backward()

    assert all(parameter.grad is None for parameter in base.parameters())
    assert wrapper.channel_gate.grad is not None
    for delta in (wrapper.q_lora, wrapper.k_lora, wrapper.v_lora, wrapper.o_lora):
        assert delta.down.grad is not None and torch.count_nonzero(delta.down.grad)
        assert delta.up.grad is not None and torch.count_nonzero(delta.up.grad)


def test_v71_absent_action_has_exactly_zero_geometry_residual() -> None:
    base = _Base()
    wrapper = SE3GeometryLoRAAttention(
        base,
        attention_fn=_attention,
        rope_apply_fn=_rope,
        num_heads=2,
        head_dim=4,
        rank=2,
    )
    wrapper.channel_gate.data.fill_(1.0)
    for delta in (wrapper.q_lora, wrapper.k_lora, wrapper.v_lora, wrapper.o_lora):
        delta.up.data.fill_(1.0)
    x = torch.randn(1, 3, 8)
    expected = base(x, **{k: v for k, v in _kwargs(False).items() if k not in {"arm_transform", "arm_present"}})
    assert torch.equal(wrapper(x, **_kwargs(False)), expected)


def test_v71_parameter_contract_is_rank16_qkvo_plus_one_channel_gate() -> None:
    wrapper = SE3GeometryLoRAAttention(
        _Base(),
        attention_fn=_attention,
        rope_apply_fn=_rope,
        num_heads=2,
        head_dim=4,
        rank=2,
    )
    names = {name for name, value in wrapper.named_parameters() if value.requires_grad}
    assert names == {
        "channel_gate",
        "q_lora.down",
        "q_lora.up",
        "k_lora.down",
        "k_lora.up",
        "v_lora.down",
        "v_lora.up",
        "o_lora.down",
        "o_lora.up",
    }
