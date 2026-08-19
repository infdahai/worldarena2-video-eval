from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from worldarena_baseline.wan_v11_state import (
    ArmActionContent,
    BimanualCondition,
    BimanualSlotTokenizer,
    build_counterfactual,
    pack_initial_slots,
)


def _arm(*, offset: float, active: bool = True) -> ArmActionContent:
    batch = 2
    anchor = torch.zeros(batch, 6)
    anchor[:, 0] = offset
    translation = torch.full((batch, 20, 3), offset + 0.1)
    rotation = torch.full((batch, 20, 3), offset + 0.2)
    image_motion = torch.full((batch, 20, 6), offset + 0.3)
    gripper = torch.full((batch, 20, 3), offset + 0.4)
    present = torch.ones(batch, 21, dtype=torch.bool)
    motion_active = torch.full((batch, 20), active, dtype=torch.bool)
    support = torch.zeros(batch, 21, 3, 4)
    support[..., int(offset) % 3, int(offset) % 4] = 1
    return ArmActionContent(
        anchor=anchor,
        translation=translation,
        rotation=rotation,
        image_motion=image_motion,
        gripper=gripper,
        arm_present=present,
        motion_active=motion_active,
        support=support,
    )


def _condition(*, right_active: bool = True) -> BimanualCondition:
    return BimanualCondition(
        left=_arm(offset=0.0),
        right=_arm(offset=1.0, active=right_active),
        destination_time=torch.arange(21).expand(2, -1).clone(),
    )


def test_pack_initial_slots_has_exact_time_contract() -> None:
    condition = _condition()
    tokenizer = BimanualSlotTokenizer(width=384)

    slots = pack_initial_slots(tokenizer, condition)

    assert slots.left.shape == (2, 21, 4, 384)
    assert slots.right.shape == (2, 21, 4, 384)
    assert torch.count_nonzero(slots.left[:, 0]).item() == 0
    assert torch.count_nonzero(slots.right[:, 0]).item() == 0
    assert torch.equal(slots.destination_time, condition.destination_time)


def test_pack_initial_slots_zeroes_an_absent_arm_time() -> None:
    condition = _condition()
    condition.left.arm_present[:, 7] = False
    tokenizer = BimanualSlotTokenizer(width=384)

    slots = pack_initial_slots(tokenizer, condition)

    assert torch.count_nonzero(slots.left[:, 7]).item() == 0
    assert torch.count_nonzero(slots.right[:, 7]).item() > 0


def test_left_and_right_tokenizers_share_no_parameters() -> None:
    tokenizer = BimanualSlotTokenizer(width=384)

    left = {id(parameter) for parameter in tokenizer.left.parameters()}
    right = {id(parameter) for parameter in tokenizer.right.parameters()}

    assert left
    assert right
    assert left.isdisjoint(right)


@pytest.mark.parametrize(
    ("kind", "changed", "unchanged"),
    (("wrong-left", "left", "right"), ("wrong-right", "right", "left")),
)
def test_wrong_arm_changes_only_motion_content(
    kind: str,
    changed: str,
    unchanged: str,
) -> None:
    condition = _condition()

    wrong = build_counterfactual(condition, kind)

    destination = getattr(condition, changed)
    source = getattr(condition, unchanged)
    changed_arm = getattr(wrong, changed)
    unchanged_arm = getattr(wrong, unchanged)
    for name in ("anchor", "arm_present", "motion_active", "support"):
        assert torch.equal(getattr(changed_arm, name), getattr(destination, name))
    for name in ("translation", "rotation", "image_motion", "gripper"):
        assert torch.equal(getattr(changed_arm, name), getattr(source, name))
    for field in fields(ArmActionContent):
        assert torch.equal(getattr(unchanged_arm, field.name), getattr(source, field.name))
    assert torch.equal(wrong.destination_time, condition.destination_time)


def test_active_arm_null_retains_identity_and_zeroes_motion() -> None:
    condition = _condition(right_active=False)
    condition.right.motion_active.zero_()
    condition.right.translation.zero_()
    condition.right.rotation.zero_()
    condition.right.image_motion.zero_()
    condition.right.gripper[..., 2].zero_()

    wrong = build_counterfactual(condition, "active-arm-null")

    assert torch.equal(wrong.left.anchor, condition.left.anchor)
    assert torch.equal(wrong.left.arm_present, condition.left.arm_present)
    assert torch.equal(wrong.left.support, condition.left.support)
    assert torch.count_nonzero(wrong.left.translation).item() == 0
    assert torch.count_nonzero(wrong.left.rotation).item() == 0
    assert torch.count_nonzero(wrong.left.image_motion).item() == 0
    assert torch.equal(wrong.left.gripper[..., :2], condition.left.gripper[..., :2])
    assert torch.count_nonzero(wrong.left.gripper[..., 2]).item() == 0
    for field in fields(ArmActionContent):
        assert torch.equal(
            getattr(wrong.right, field.name),
            getattr(condition.right, field.name),
        )


def test_active_arm_null_rejects_two_active_arms() -> None:
    with pytest.raises(ValueError, match="exactly one active arm"):
        build_counterfactual(_condition(), "active-arm-null")


def test_condition_rejects_noncanonical_destination_slots() -> None:
    condition = _condition()
    condition.destination_time[:, 8] = 9

    with pytest.raises(ValueError, match="destination time"):
        pack_initial_slots(BimanualSlotTokenizer(width=384), condition)
