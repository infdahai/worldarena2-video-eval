from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from worldarena_baseline.wan_v7_model import (  # noqa: E402
    ParentPlusSE3Wan,
    install_v71_attention,
    v71_trainable_parameter_names,
)


WIDTH = 24 * 128
BLOCKS = (8, 16, 24)


def _attention(q, k, v, seq_lens):
    del q, k, seq_lens
    return v


def _rope(value, grid_sizes, freqs):
    del grid_sizes, freqs
    return value


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = WIDTH
        self.num_heads = 24
        self.head_dim = 128
        self.q = nn.Identity()
        self.k = nn.Identity()
        self.v = nn.Identity()
        self.o = nn.Identity()
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        del seq_lens, grid_sizes, freqs
        return x


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
        seq_lens = torch.tensor([seq_len])
        grid = torch.tensor([[21, 1, 1]])
        for block in self.blocks:
            x = block(x, seq_lens, grid, "rope")
        return x


class _Parent(nn.Module):
    injection_points = (0, 8, 16, 24)
    raster_support_gating = True

    def forward(self, raster, timestep, *, seq_len, condition_support, action_present):
        del raster, timestep, condition_support, action_present
        return {point: torch.zeros(1, seq_len, WIDTH) for point in self.injection_points}


def _inputs():
    return {
        "x": torch.randn(1, 21, WIDTH),
        "t": torch.zeros(1),
        "context": None,
        "seq_len": 21,
        "action_raster": torch.zeros(1, 10, 81, 60, 80),
        "condition_support": torch.ones(1, 2, 21, 15, 20),
        "action_present": torch.ones(1),
        "se3_arm_transform": torch.eye(4, dtype=torch.float32)
        .reshape(1, 1, 1, 4, 4)
        .repeat(1, 2, 21, 1, 1),
        "se3_arm_present": torch.ones(1, 2, 21, dtype=torch.bool),
    }


def test_v71_installs_only_three_geometry_lora_branches_and_freezes_wan() -> None:
    backbone = _Backbone()
    values = _inputs()
    expected = backbone(**{k: v for k, v in values.items() if k in {"x", "t", "context", "seq_len"}})
    wrappers = install_v71_attention(backbone, BLOCKS, _rope, _attention, rank=16)
    model = ParentPlusSE3Wan(backbone, _Parent(), wrappers)

    assert torch.equal(model(**values), expected)
    names = v71_trainable_parameter_names(model, rank=16)
    assert len(names) == 27
    assert all("geometry_wrappers" in name for name in names)
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 1_188_864
