from __future__ import annotations

import torch
from torch import nn

from worldarena_baseline.wan_v11_controller import BimanualControllerStage
from worldarena_baseline.wan_v11_model import (
    ParentPlusBimanualControllerWan,
    v11_trainable_parameter_names,
)
from worldarena_baseline.wan_v11_state import (
    ArmActionContent,
    BimanualCondition,
    BimanualSlotTokenizer,
)


class _Block(nn.Module):
    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index
        self.weight = nn.Parameter(torch.tensor(0.001 * (index + 1)))

    def forward(self, x, **_kwargs):
        return x + self.weight


class _Head(nn.Module):
    def forward(self, x, _e):
        return x


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_type = "t2v"
        self.patch_embedding = nn.Conv3d(2, 8, kernel_size=1, bias=False)
        self.freq_dim = 4
        self.dim = 8
        self.time_embedding = nn.Linear(4, 8, bias=False)
        self.time_projection = nn.Linear(8, 48, bias=False)
        self.text_embedding = nn.Linear(4, 8, bias=False)
        self.text_len = 2
        self.freqs = torch.ones(2, 2)
        self.blocks = nn.ModuleList([_Block(index) for index in range(27)])
        self.head = _Head()

    def unpatchify(self, x, _grid_sizes):
        return [x]

    def forward(self, x, t, context, seq_len, y=None):
        from worldarena_baseline.wan_v11_model import run_wan_backbone_loop

        return run_wan_backbone_loop(self, x, t, context, seq_len, y=y).video


class _Parent(nn.Module):
    injection_points = (0, 8, 16, 24)
    raster_support_gating = True

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.marker = nn.Parameter(torch.tensor(0.0))

    def forward(self, action_raster, _t, *, seq_len, **_kwargs):
        batch = action_raster.shape[0]
        return {
            point: action_raster.new_zeros(batch, seq_len, self.width)
            for point in self.injection_points
        }


def _arm(offset: float) -> ArmActionContent:
    return ArmActionContent(
        anchor=torch.full((1, 6), offset),
        translation=torch.full((1, 20, 3), offset + 0.1),
        rotation=torch.full((1, 20, 3), offset + 0.2),
        image_motion=torch.full((1, 20, 6), offset + 0.3),
        gripper=torch.full((1, 20, 3), offset + 0.4),
        arm_present=torch.ones(1, 21, dtype=torch.bool),
        motion_active=torch.ones(1, 20, dtype=torch.bool),
        support=torch.cat(
            (
                torch.zeros(1, 1, 2, 3),
                torch.ones(1, 20, 2, 3),
            ),
            dim=1,
        ),
    )


def _inputs():
    condition = BimanualCondition(
        left=_arm(0.0),
        right=_arm(1.0),
        destination_time=torch.arange(21).reshape(1, 21),
    )
    return {
        "x": [torch.randn(2, 21, 2, 3)],
        "t": torch.tensor([500.0]),
        "context": [torch.randn(2, 4)],
        "seq_len": 21 * 2 * 3,
        "action_raster": torch.randn(1, 21, 10, 2, 3),
        "condition_support": torch.ones(1, 21, 2, 3),
        "action_present": torch.ones(1),
        "bimanual_condition": condition,
    }


def _model() -> ParentPlusBimanualControllerWan:
    backbone = _Backbone()
    parent = _Parent(width=8)
    tokenizer = BimanualSlotTokenizer(width=8)
    stages = {
        point: BimanualControllerStage(
            visual_width=8,
            slot_width=8,
            heads=2,
            support_shape=(2, 3),
        )
        for point in (6, 16, 24)
    }
    return ParentPlusBimanualControllerWan(backbone, parent, tokenizer, stages)


def test_zero_writeback_matches_frozen_parent_exactly() -> None:
    torch.manual_seed(4)
    model = _model()
    inputs = _inputs()

    expected = model.backbone(
        inputs["x"], inputs["t"], inputs["context"], inputs["seq_len"]
    )
    actual = model(**inputs)

    assert len(actual.video) == len(expected)
    assert torch.equal(actual.video[0], expected[0])
    assert tuple(actual.eef_logits) == (6, 16, 24)


def test_stages_run_after_exact_blocks_with_functional_slot_state() -> None:
    model = _model()
    inputs = _inputs()

    result = model(**inputs)

    assert tuple(result.telemetry) == (6, 16, 24)
    assert result.final_slots.left.shape == (1, 21, 4, 8)
    assert result.final_slots.right.shape == (1, 21, 4, 8)
    for module in model.modules():
        assert not hasattr(module, "bound_condition")
        assert not hasattr(module, "_checkpoint_condition")


def test_only_tokenizer_and_controller_are_trainable() -> None:
    model = _model()

    names = v11_trainable_parameter_names(model)

    assert names
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(not parameter.requires_grad for parameter in model.parent_adapter.parameters())
    assert any(name.startswith("tokenizer.") for name in names)
    assert any(name.startswith("controller_stages.6.left.") for name in names)
    assert any(name.startswith("controller_stages.24.right.") for name in names)


def test_action_present_zero_removes_both_controller_streams() -> None:
    model = _model()
    inputs = _inputs()
    inputs["action_present"].zero_()

    result = model(**inputs)

    assert torch.count_nonzero(result.final_slots.left).item() == 0
    assert torch.count_nonzero(result.final_slots.right).item() == 0
    for logits in result.eef_logits.values():
        assert torch.count_nonzero(logits[0]).item() == 0
        assert torch.count_nonzero(logits[1]).item() == 0


def test_bfloat16_backbone_uses_outer_autocast_for_time_embedding() -> None:
    backbone = _Backbone().to(torch.bfloat16)
    inputs = _inputs()
    x = [item.to(torch.bfloat16) for item in inputs["x"]]
    context = [item.to(torch.bfloat16) for item in inputs["context"]]

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = backbone(x, inputs["t"], context, inputs["seq_len"])

    assert output[0].dtype == torch.float32
