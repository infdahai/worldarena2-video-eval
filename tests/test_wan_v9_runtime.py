from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

from worldarena_baseline.wan_v9_model import (  # noqa: E402
    V9_BLOCKS,
    ParentPlusPhaseLockedActionWan,
    install_v9_attention,
    v9_trainable_parameter_names,
)
from worldarena_baseline.wan_v9_tokens import V9ActionTokenizer  # noqa: E402
from worldarena_baseline.wan_v9_training import (  # noqa: E402
    build_v9_optimizer,
    set_v9_learning_rates,
)


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
    _v9_test_inner_width = WIDTH

    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(WIDTH, WIDTH)
        self.k = nn.Linear(WIDTH, WIDTH)
        self.v = nn.Linear(WIDTH, WIDTH)
        self.o = nn.Linear(WIDTH, WIDTH)

    def forward(self, value, seq_lens, grid_sizes, freqs):
        del seq_lens, grid_sizes, freqs
        return self.o(self.q(value) + self.k(value) + self.v(value))


class _InnerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()

    def forward(self, value, seq_lens, grid_sizes, freqs):
        return self.self_attn(value, seq_lens, grid_sizes, freqs)


class _CheckpointedBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _InnerBlock()

    def forward(self, value, seq_lens, grid_sizes, freqs):
        return checkpoint(
            lambda tensor: self.block(tensor, seq_lens, grid_sizes, freqs),
            value,
            use_reentrant=False,
        )


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_CheckpointedBlock() for _ in range(30))

    def forward(self, value, timestep, context, seq_len, y=None):
        del timestep, context, y
        seq_lens = torch.tensor([seq_len], device=value.device)
        grid_sizes = torch.tensor([[21, 1, 1]], device=value.device)
        for block in self.blocks:
            value = block(value, seq_lens, grid_sizes, None)
        return value


class _Parent(nn.Module):
    injection_points = (0, 8, 16, 24)
    raster_support_gating = True

    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Parameter(torch.ones(()))

    def forward(self, raster, timestep, *, seq_len, condition_support, action_present):
        del raster, timestep, condition_support, action_present
        return {point: self.frozen.detach() * torch.zeros(1, seq_len, WIDTH) for point in self.injection_points}


def _model() -> ParentPlusPhaseLockedActionWan:
    backbone = _Backbone()
    wrappers = install_v9_attention(backbone, V9_BLOCKS, action_width=WIDTH)
    tokenizer = V9ActionTokenizer(_statistics(), action_width=WIDTH, hidden_width=WIDTH)
    return ParentPlusPhaseLockedActionWan(backbone, _Parent(), tokenizer, wrappers)


def _inputs() -> dict[str, object]:
    return {
        "x": torch.randn(1, 21, WIDTH, requires_grad=True),
        "t": torch.zeros(1),
        "context": None,
        "seq_len": 21,
        "action_raster": torch.zeros(1, 10, 81, 60, 80),
        "condition_support": torch.ones(1, 2, 21, 1, 1),
        "action_present": torch.ones(1),
        "transition_features": {
            "translation": torch.randn(1, 2, 20, 4),
            "rotation": torch.randn(1, 2, 20, 4),
            "image": torch.randn(1, 2, 20, 6),
            "gripper": torch.randn(1, 2, 20, 3),
        },
        "transition_arm_present": torch.ones(1, 2, 20, dtype=torch.bool),
    }


def test_checkpoint_lease_survives_rejected_second_forward_then_releases() -> None:
    model = _model()
    for wrapper in model.action_wrappers.values():
        wrapper.cross.channel_gate.data.fill_(0.1)
    first = model(**_inputs())
    assert all(wrapper._checkpoint_condition is not None for wrapper in model.action_wrappers.values())
    with pytest.raises(RuntimeError, match="already has a bound condition"):
        model(**_inputs())
    first.square().mean().backward()
    assert all(wrapper.bound_condition is None for wrapper in model.action_wrappers.values())
    assert all(wrapper._checkpoint_condition is None for wrapper in model.action_wrappers.values())


def test_three_optimizer_steps_cover_every_trainable_family_and_keep_parent_frozen() -> None:
    torch.manual_seed(19)
    model = _model()
    for wrapper in model.action_wrappers.values():
        wrapper.cross.channel_gate.data.fill_(0.1)
    optimizer = build_v9_optimizer(model)
    trainable = v9_trainable_parameter_names(model)
    seen = {name: False for name in trainable}
    for step in range(1, 4):
        optimizer.zero_grad(set_to_none=True)
        set_v9_learning_rates(optimizer, step)
        output = model(**_inputs())
        output.square().mean().backward()
        for name, parameter in model.named_parameters():
            if name in seen and parameter.grad is not None:
                seen[name] |= bool(torch.count_nonzero(parameter.grad).item())
        optimizer.step()
        model.release_completed_backward_conditions()
    assert all(seen.values()), sorted(name for name, value in seen.items() if not value)
    assert all(parameter.grad is None for parameter in model.parent_adapter.parameters())
    assert all(wrapper._checkpoint_condition is None for wrapper in model.action_wrappers.values())
