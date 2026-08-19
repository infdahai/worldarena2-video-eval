from __future__ import annotations

import torch

from worldarena_baseline.wan_v11_controller import BimanualControllerStage
from worldarena_baseline.wan_v11_sparse import select_tube_tokens
from worldarena_baseline.wan_v11_state import BimanualSlots


def _fixture(*, visual_values: float = 1.0):
    batch, time, height, width, visual_width, slot_width = 1, 21, 2, 3, 24, 8
    visual = torch.arange(
        batch * time * height * width * visual_width,
        dtype=torch.float32,
    ).reshape(batch, time * height * width, visual_width)
    visual = visual / visual.numel() + visual_values
    left = torch.randn(batch, time, 4, slot_width)
    right = torch.randn(batch, time, 4, slot_width)
    left[:, 0].zero_()
    right[:, 0].zero_()
    slots = BimanualSlots(left, right, torch.arange(time).expand(batch, -1))
    support = torch.zeros(batch, time, 2, height, width)
    support[:, 1:, 0, 0, 1] = 1
    support[:, 1:, 0, 0, 2] = 0.75
    support[:, 1:, 1, 1, 1] = 1
    support[:, 1:, 1, 1, 2] = 0.75
    selection = select_tube_tokens(support, max_tokens=4)
    present = torch.ones(batch, time, 2, dtype=torch.bool)
    present[:, 0] = False
    stage = BimanualControllerStage(
        visual_width=visual_width,
        slot_width=slot_width,
        heads=2,
        support_shape=(height, width),
    )
    return stage, visual, slots, selection, support, present


def _enable_writes(stage: BimanualControllerStage) -> None:
    with torch.no_grad():
        for arm in (stage.left, stage.right):
            arm.write_o.weight.fill_(0.05)


def test_left_and_right_controller_parameters_are_disjoint() -> None:
    stage, *_ = _fixture()

    left = {id(parameter) for parameter in stage.left.parameters()}
    right = {id(parameter) for parameter in stage.right.parameters()}

    assert left
    assert right
    assert left.isdisjoint(right)


def test_forced_zero_visual_values_make_visual_read_exact_zero() -> None:
    stage, visual, slots, selection, support, present = _fixture()

    result = stage(
        visual,
        slots,
        selection,
        support,
        present,
        force_zero_visual_values=True,
    )

    assert torch.count_nonzero(result.visual_reads[0]).item() == 0
    assert torch.count_nonzero(result.visual_reads[1]).item() == 0
    assert torch.count_nonzero(result.eef_logits[0]).item() == 0
    assert torch.count_nonzero(result.eef_logits[1]).item() == 0


def test_absent_arm_and_latent_zero_remain_exact_zero() -> None:
    stage, visual, slots, selection, support, present = _fixture()
    _enable_writes(stage)
    present[:, :, 0] = False

    result = stage(visual, slots, selection, support, present)

    assert torch.count_nonzero(result.slots.left).item() == 0
    assert torch.count_nonzero(result.visual_reads[0]).item() == 0
    assert torch.count_nonzero(result.direct_writes[0]).item() == 0
    assert torch.count_nonzero(result.slots.right[:, 0]).item() == 0
    assert torch.count_nonzero(result.direct_writes[1][:, : 6]).item() == 0


def test_overlap_adds_both_arm_writes_and_stays_inside_support() -> None:
    stage, visual, slots, selection, support, present = _fixture()
    support[:, 1:, 1] = support[:, 1:, 0]
    selection = select_tube_tokens(support, max_tokens=4)
    _enable_writes(stage)

    result = stage(visual, slots, selection, support, present)

    delta = result.direct_writes[0] + result.direct_writes[1]
    expected_visual = visual + result.direct_writes[0]
    expected_visual = expected_visual + result.direct_writes[1]
    assert torch.equal(result.visual, expected_visual)
    flat_union = (support.amax(dim=2) > 0).reshape(1, 21 * 6, 1)
    assert torch.count_nonzero(delta * (~flat_union)).item() == 0
    assert torch.count_nonzero(delta).item() > 0


def test_controller_gradients_reach_both_separate_streams_and_gate_stays_fp32() -> None:
    stage, visual, slots, selection, support, present = _fixture()
    _enable_writes(stage)
    visual.requires_grad_(True)
    left = slots.left.detach().clone().requires_grad_(True)
    right = slots.right.detach().clone().requires_grad_(True)
    slots = BimanualSlots(left, right, slots.destination_time)

    result = stage(visual, slots, selection, support, present)
    loss = (
        result.visual.square().mean()
        + result.slots.left.square().mean()
        + result.slots.right.square().mean()
        + result.eef_logits[0].square().mean()
        + result.eef_logits[1].square().mean()
    )
    loss.backward()

    for arm in (stage.left, stage.right):
        for family in (arm.read_q, arm.read_v, arm.update, arm.write_q, arm.write_o, arm.eef_head):
            assert any(
                parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
                for parameter in family.parameters()
            )
        assert arm.gate.grad is not None
    assert torch.count_nonzero(left.grad).item() > 0
    assert torch.count_nonzero(right.grad).item() > 0

    stage.to(dtype=torch.bfloat16)
    assert stage.left.gate.dtype == torch.float32
    assert stage.right.gate.dtype == torch.float32
