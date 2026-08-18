from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from worldarena_baseline.wan_v7_model import (  # noqa: E402
    ParentPlusSE3Wan,
    install_v7_attention,
    v7_trainable_parameter_names,
)
from worldarena_baseline.wan_action_adapter import enable_wan_block_checkpointing  # noqa: E402


WIDTH = 24 * 128
BLOCKS = (8, 16, 24)


def _attention(q, k, v, seq_lens):
    del q, k, seq_lens
    return v


def _rope(value, grid_sizes, freqs):
    del grid_sizes, freqs
    return value


class _WanAttention(nn.Module):
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
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, x, seq_lens, grid_sizes, freqs):
        del seq_lens, grid_sizes, freqs
        return x * self.scale


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _WanAttention()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        return self.self_attn(x, seq_lens, grid_sizes, freqs)


class _Backbone(nn.Module):
    def __init__(self, *, raises: bool = False, returns_list: bool = False) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(30))
        self.raises = raises
        self.returns_list = returns_list
        self.block_calls = {index: 0 for index in range(30)}
        for index, block in enumerate(self.blocks):
            block.register_forward_hook(self._count(index))

    def _count(self, index: int):
        def count(_module, _inputs, _output):
            self.block_calls[index] += 1

        return count

    def forward(self, x, t, context, seq_len, y=None):
        del t, context, y
        seq_lens = torch.tensor([seq_len], device=x.device)
        grid_sizes = torch.tensor([[21, 1, 1]], device=x.device)
        freqs = torch.ones(1, device=x.device)
        for block in self.blocks:
            x = block(x, seq_lens, grid_sizes, freqs)
        if self.raises:
            raise RuntimeError("backbone failure")
        return [x] if self.returns_list else x


class _Parent(nn.Module):
    injection_points = (0, 8, 16, 24)
    raster_support_gating = True

    def __init__(self, *, residual_value: float = 0.0) -> None:
        super().__init__()
        self.calls = 0
        self.weight = nn.Parameter(torch.ones(1))
        self.residual_value = residual_value

    def forward(self, raster, timestep, *, seq_len, condition_support, action_present):
        del raster, timestep, condition_support, action_present
        self.calls += 1
        result = {
            point: torch.zeros(1, seq_len, WIDTH)
            for point in self.injection_points
        }
        result[0].fill_(self.residual_value)
        return result


def _inputs():
    return {
        "x": torch.randn(1, 21, WIDTH),
        "t": torch.zeros(1),
        "context": None,
        "seq_len": 21,
        "action_raster": torch.zeros(1, 10, 81, 60, 80),
        "condition_support": torch.ones(1, 2, 21, 15, 20),
        "action_present": torch.ones(1),
        "se3_arm_transform": torch.eye(
            4, dtype=torch.float32
        ).reshape(1, 1, 1, 4, 4).repeat(1, 2, 21, 1, 1),
        "se3_arm_present": torch.ones(1, 2, 21, dtype=torch.bool),
    }


def _model(backbone: nn.Module | None = None, *, activation_checkpoint: bool = False):
    backbone = backbone or _Backbone()
    parent = _Parent()
    wrappers = install_v7_attention(backbone, BLOCKS, _rope, _attention)
    if activation_checkpoint:
        enable_wan_block_checkpointing(backbone)
    return ParentPlusSE3Wan(backbone, parent, wrappers), parent


def test_zero_gate_is_exact_clean_parent_and_only_three_gates_trainable() -> None:
    backbone = _Backbone()
    values = _inputs()
    parent_output = backbone(values["x"], values["t"], values["context"], values["seq_len"])
    model, parent = _model(backbone)

    actual = model(**values)

    assert torch.equal(parent_output, actual)
    assert model.selected_blocks == (8, 16, 24)
    assert tuple(model.geometry_wrappers) == ("8", "16", "24")
    assert v7_trainable_parameter_names(model) == {
        "geometry_wrappers.8.gate",
        "geometry_wrappers.16.gate",
        "geometry_wrappers.24.gate",
    }
    assert sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    ) == 9216
    assert parent.calls == 1
    assert all(wrapper.bound_condition is None for wrapper in model.geometry_wrappers.values())


def test_official_wan_tensor_list_output_is_preserved() -> None:
    backbone = _Backbone(returns_list=True)
    model, _ = _model(backbone)
    actual = model(**_inputs())
    assert isinstance(actual, list) and len(actual) == 1
    assert isinstance(actual[0], torch.Tensor)


def test_conditions_are_cleared_after_success_and_exception() -> None:
    values = _inputs()
    model, _parent = _model()
    model(**values)
    assert all(wrapper.bound_condition is None for wrapper in model.geometry_wrappers.values())

    failing, _parent = _model(_Backbone(raises=True))
    with pytest.raises(RuntimeError, match="backbone failure"):
        failing(**values)
    assert all(wrapper.bound_condition is None for wrapper in failing.geometry_wrappers.values())


def test_noncheckpointed_forward_does_not_leave_a_private_condition_lease() -> None:
    """A private condition lease is only legal for a checkpoint replay.

    This regression distinguishes normal sequential batches from the pending
    graph case below.  Retaining a non-checkpoint lease would make the second
    production batch fail before it reaches Wan attention.
    """

    model, _parent = _model()
    values = _inputs()
    model(**values)
    assert all(
        wrapper._checkpoint_condition is None
        for wrapper in model.geometry_wrappers.values()
    )
    model(**values)


def test_bound_condition_survives_activation_checkpoint_recomputation() -> None:
    model, _parent = _model(activation_checkpoint=True)
    values = _inputs()
    values["x"].requires_grad_()
    model(**values).sum().backward()
    assert all(wrapper.condition_use_count == 2 for wrapper in model.geometry_wrappers.values())
    assert all(wrapper.bound_condition is None for wrapper in model.geometry_wrappers.values())
    assert all(
        wrapper._checkpoint_condition is None
        for wrapper in model.geometry_wrappers.values()
    )


def test_existing_parent_raster_residual_hook_fires_once() -> None:
    backbone = _Backbone()
    parent = _Parent(residual_value=0.25)
    wrappers = install_v7_attention(backbone, BLOCKS, _rope, _attention)
    model = ParentPlusSE3Wan(backbone, parent, wrappers)
    values = _inputs()

    actual = model(**values)

    torch.testing.assert_close(actual, values["x"] + 0.25)
    assert parent.calls == 1
    assert backbone.block_calls[0] == 1


def test_null_action_zeros_both_head_groups() -> None:
    model, _parent = _model()
    values = _inputs()
    values["action_present"] = torch.zeros(1)
    for wrapper in model.geometry_wrappers.values():
        wrapper.gate.data.fill_(1.0)
    output = model(**values)
    expected = _Backbone()(values["x"], values["t"], values["context"], values["seq_len"])
    assert torch.equal(output, expected)


def test_install_rejects_non_stage_a_or_invalid_backbone_shapes() -> None:
    with pytest.raises(ValueError, match="Stage A"):
        install_v7_attention(_Backbone(), (4, 8, 16), _rope, _attention)
    with pytest.raises(ValueError, match="unique"):
        install_v7_attention(_Backbone(), (8, 8, 16), _rope, _attention)
    wrong_shape = _Backbone()
    wrong_shape.blocks[8].self_attn.hidden_size = 1024
    with pytest.raises(ValueError, match="3072/24"):
        install_v7_attention(wrong_shape, BLOCKS, _rope, _attention)
    backbone = _Backbone()
    install_v7_attention(backbone, BLOCKS, _rope, _attention)
    with pytest.raises(ValueError, match="already"):
        install_v7_attention(backbone, BLOCKS, _rope, _attention)


def test_install_is_atomic_when_a_late_wrapper_construction_fails() -> None:
    backbone = _Backbone()
    originals = {index: backbone.blocks[index].self_attn for index in BLOCKS}
    del originals[16].q_norm
    del originals[16].k_norm

    with pytest.raises(TypeError, match="q_norm/k_norm"):
        install_v7_attention(backbone, BLOCKS, _rope, _attention)

    assert {
        index: backbone.blocks[index].self_attn for index in BLOCKS
    } == originals
    assert all(
        parameter.requires_grad
        for attention in originals.values()
        for parameter in attention.parameters()
    )


def test_failed_condition_bind_preserves_an_older_pending_checkpoint_binding() -> None:
    model, _parent = _model()
    values = _inputs()
    sentinel = object()
    model.geometry_wrappers["16"]._checkpoint_condition = sentinel

    with pytest.raises(RuntimeError, match="already has a bound condition"):
        model(**values)

    assert model.geometry_wrappers["16"]._checkpoint_condition is sentinel
    assert model.geometry_wrappers["8"].bound_condition is None
    assert model.geometry_wrappers["8"]._checkpoint_condition is None
    model.geometry_wrappers["16"]._checkpoint_condition = None
    model(**values)


def test_rejected_second_forward_preserves_first_checkpoint_until_backward() -> None:
    model, _parent = _model(activation_checkpoint=True)
    first = _inputs()
    first["x"].requires_grad_()
    first_output = model(**first)

    with pytest.raises(RuntimeError, match="already has a bound condition"):
        model(**_inputs())

    assert all(
        wrapper._checkpoint_condition is not None
        for wrapper in model.geometry_wrappers.values()
    )
    first_output.sum().backward()
    assert all(wrapper.condition_use_count == 2 for wrapper in model.geometry_wrappers.values())
    assert all(
        wrapper._checkpoint_condition is None
        for wrapper in model.geometry_wrappers.values()
    )
