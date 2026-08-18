from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from worldarena_baseline.wan_v9_model import (  # noqa: E402
    V9_BLOCKS,
    ParentPlusPhaseLockedActionWan,
    install_v9_attention,
    v9_trainable_parameter_names,
)
from worldarena_baseline.wan_v9_tokens import V9ActionTokenizer  # noqa: E402


WIDTH = 8


def _statistics() -> dict[str, object]:
    result: dict[str, object] = {
        "contract": "wan-v9-transition-normalization/1",
        "receipt_sha256": "a" * 64,
    }
    for name, width in (("translation", 4), ("rotation", 4), ("image", 6), ("gripper", 3)):
        result[name] = {"mean": [0.0] * width, "std": [1.0] * width}
    return result


class _Attention(nn.Module):
    _v9_test_fixture = True
    _v9_test_inner_width = 8

    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(WIDTH, WIDTH)
        self.k = nn.Linear(WIDTH, WIDTH)
        self.v = nn.Linear(WIDTH, WIDTH)
        self.o = nn.Linear(WIDTH, WIDTH)

    def forward(self, x, seq_lens, grid_sizes, freqs):
        del seq_lens, grid_sizes, freqs
        # Keep every native projection in the graph for whitelist/gradient tests.
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
        grid_sizes = torch.tensor([[21, 1, 1]], device=x.device)
        freqs = None
        for block in self.blocks:
            x = block(x, seq_lens, grid_sizes, freqs)
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


def _features() -> dict[str, torch.Tensor]:
    return {
        "translation": torch.randn(1, 2, 20, 4),
        "rotation": torch.randn(1, 2, 20, 4),
        "image": torch.randn(1, 2, 20, 6),
        "gripper": torch.randn(1, 2, 20, 3),
    }


def _inputs() -> dict[str, object]:
    return {
        "x": torch.randn(1, 21, WIDTH),
        "t": torch.zeros(1),
        "context": None,
        "seq_len": 21,
        "action_raster": torch.zeros(1, 10, 81, 60, 80),
        "condition_support": torch.ones(1, 2, 21, 1, 1),
        "action_present": torch.ones(1),
        "transition_features": _features(),
        "transition_arm_present": torch.ones(1, 2, 20, dtype=torch.bool),
    }


def _model() -> ParentPlusPhaseLockedActionWan:
    backbone = _Backbone()
    tokenizer = V9ActionTokenizer(_statistics(), action_width=8, hidden_width=8)
    wrappers = install_v9_attention(backbone, V9_BLOCKS, action_width=8)
    return ParentPlusPhaseLockedActionWan(backbone, _Parent(), tokenizer, wrappers)


def test_installs_exact_blocks_and_zero_init_equals_clean_parent() -> None:
    torch.manual_seed(3)
    backbone = _Backbone()
    values = _inputs()
    expected = backbone(values["x"], values["t"], values["context"], values["seq_len"])
    tokenizer = V9ActionTokenizer(_statistics(), action_width=8, hidden_width=8)
    wrappers = install_v9_attention(backbone, V9_BLOCKS, action_width=8)
    model = ParentPlusPhaseLockedActionWan(backbone, _Parent(), tokenizer, wrappers)
    actual = model(**values)
    assert V9_BLOCKS == (8, 9, 10, 11, 12, 13)
    assert tuple(wrappers) == V9_BLOCKS
    assert torch.equal(actual, expected)


def test_trainable_whitelist_contains_only_native_cross_tokenizer_and_gates() -> None:
    model = _model()
    names = v9_trainable_parameter_names(model)
    assert names
    assert any("action_tokenizer.translation_mlp" in name for name in names)
    assert sum(name.endswith("channel_gate") for name in names) == 6
    assert all("geometry" not in name and "se3" not in name.lower() for name in names)
    assert all(not parameter.requires_grad for parameter in model.parent_adapter.parameters())


def test_condition_is_cleared_and_null_action_zeros_cross_path() -> None:
    model = _model()
    values = _inputs()
    for wrapper in model.action_wrappers.values():
        wrapper.cross.channel_gate.data.fill_(1.0)
    values["action_present"] = torch.zeros(1)
    output = model(**values)
    assert torch.isfinite(output).all()
    assert all(wrapper.bound_condition is None for wrapper in model.action_wrappers.values())
    assert all(wrapper._checkpoint_condition is None for wrapper in model.action_wrappers.values())


def test_installation_rejects_wrong_blocks_and_rolls_back() -> None:
    with pytest.raises(ValueError, match="8-13"):
        install_v9_attention(_Backbone(), (8, 16, 24), action_width=8)
    backbone = _Backbone()
    originals = {index: backbone.blocks[index].self_attn for index in V9_BLOCKS}
    del backbone.blocks[11].self_attn.q
    with pytest.raises((TypeError, ValueError), match="q/k/v/o"):
        install_v9_attention(backbone, V9_BLOCKS, action_width=8)
    assert {index: backbone.blocks[index].self_attn for index in V9_BLOCKS} == originals
