from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn

from .wan_v11_sparse import SparseTubeSelection, scatter_arm_residual
from .wan_v11_state import BimanualSlots


@dataclass(frozen=True)
class ArmStageOutput:
    slots: Tensor
    visual_read: Tensor
    direct_write: Tensor
    eef_logits: Tensor
    telemetry: dict[str, Tensor]


@dataclass(frozen=True)
class ControllerOutput:
    visual: Tensor
    slots: BimanualSlots
    visual_reads: tuple[Tensor, Tensor]
    direct_writes: tuple[Tensor, Tensor]
    eef_logits: tuple[Tensor, Tensor]
    telemetry: dict[str, dict[str, Tensor]]


def _arm_selection(selection: SparseTubeSelection, arm: int) -> SparseTubeSelection:
    if selection.indices.ndim != 4 or selection.indices.shape[2] != 2:
        raise ValueError("bimanual selection must have shape (B, 21, 2, K)")
    return SparseTubeSelection(
        indices=selection.indices[:, :, arm],
        valid=selection.valid[:, :, arm],
        weights=selection.weights[:, :, arm],
        spatial_shape=selection.spatial_shape,
    )


def _gather_arm_visual(visual: Tensor, selection: SparseTubeSelection) -> Tensor:
    if visual.ndim != 3 or selection.indices.ndim != 3:
        raise ValueError("visual or arm selection has an invalid rank")
    batch, sequence, channels = visual.shape
    if selection.indices.shape[:2] != (batch, 21):
        raise ValueError("arm selection differs from visual batch/time")
    height, width = selection.spatial_shape
    spatial = height * width
    if sequence != 21 * spatial:
        raise ValueError("visual sequence differs from arm support grid")
    offset = torch.arange(21, device=visual.device).view(1, 21, 1) * spatial
    index = selection.indices.to(visual.device) + offset
    expanded = visual[:, None].expand(batch, 21, sequence, channels)
    gathered = torch.gather(expanded, 2, index[..., None].expand(*index.shape, channels))
    return gathered * selection.valid.to(visual.device)[..., None].to(visual.dtype)


def _masked_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    valid: Tensor,
    *,
    heads: int,
) -> Tensor:
    batch_time, query_count, width = query.shape
    key_count = key.shape[1]
    if width % heads != 0:
        raise ValueError("attention width must be divisible by heads")
    head_width = width // heads
    q = query.reshape(batch_time, query_count, heads, head_width).transpose(1, 2)
    k = key.reshape(batch_time, key_count, heads, head_width).transpose(1, 2)
    v = value.reshape(batch_time, key_count, heads, head_width).transpose(1, 2)
    logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) / sqrt(head_width)
    key_valid = valid[:, None, None, :]
    logits = logits.masked_fill(~key_valid, float("-inf"))
    weights = torch.softmax(logits, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).to(v.dtype)
    output = torch.matmul(weights, v)
    return output.transpose(1, 2).reshape(batch_time, query_count, width)


class ArmControllerStage(nn.Module):
    def __init__(
        self,
        *,
        visual_width: int,
        slot_width: int,
        heads: int,
        support_shape: tuple[int, int],
    ) -> None:
        super().__init__()
        if slot_width % heads != 0:
            raise ValueError("slot width must be divisible by heads")
        self.visual_width = visual_width
        self.slot_width = slot_width
        self.heads = heads
        self.support_shape = support_shape

        self.read_q = nn.Linear(slot_width, slot_width, bias=False)
        self.read_k = nn.Linear(visual_width, slot_width, bias=False)
        self.read_v = nn.Linear(visual_width, slot_width, bias=False)
        self.read_o = nn.Linear(slot_width, slot_width, bias=False)
        self.update = nn.Sequential(
            nn.Linear(2 * slot_width, 2 * slot_width),
            nn.SiLU(),
            nn.Linear(2 * slot_width, slot_width),
        )
        self.write_q = nn.Linear(visual_width, slot_width, bias=False)
        self.write_k = nn.Linear(slot_width, slot_width, bias=False)
        self.write_v = nn.Linear(slot_width, slot_width, bias=False)
        self.write_o = nn.Linear(slot_width, visual_width, bias=False)
        nn.init.zeros_(self.write_o.weight)
        self.gate = nn.Parameter(torch.zeros((), dtype=torch.float32))
        heatmap_size = support_shape[0] * support_shape[1]
        self.eef_head = nn.Sequential(
            nn.Linear(slot_width, slot_width, bias=False),
            nn.SiLU(),
            nn.Linear(slot_width, heatmap_size, bias=False),
        )

    def _apply(self, fn):  # type: ignore[no-untyped-def]
        super()._apply(fn)
        if self.gate.dtype != torch.float32:
            self.gate.data = self.gate.data.float()
        return self

    def forward(
        self,
        visual: Tensor,
        slots: Tensor,
        selection: SparseTubeSelection,
        support: Tensor,
        present: Tensor,
        *,
        force_zero_visual_values: bool = False,
        force_zero_controller: bool = False,
    ) -> ArmStageOutput:
        if slots.ndim != 4 or slots.shape[1:3] != (21, 4):
            raise ValueError("arm slots must have shape (B, 21, 4, W)")
        batch, time, slot_count, width = slots.shape
        if width != self.slot_width or tuple(present.shape) != (batch, time):
            raise ValueError("slot width or presence shape differs from controller")
        if tuple(support.shape) != (batch, time, *self.support_shape):
            raise ValueError("arm support shape differs from controller")

        local = _gather_arm_visual(visual, selection)
        count = local.shape[2]
        flat_slots = slots.reshape(batch * time, slot_count, width)
        flat_local = local.reshape(batch * time, count, self.visual_width)
        flat_valid = selection.valid.to(visual.device).reshape(batch * time, count)

        read_value = self.read_v(flat_local)
        if force_zero_visual_values:
            read_value = torch.zeros_like(read_value)
        visual_read = self.read_o(
            _masked_attention(
                self.read_q(flat_slots),
                self.read_k(flat_local),
                read_value,
                flat_valid,
                heads=self.heads,
            )
        ).reshape(batch, time, slot_count, width)

        time_present = present.clone()
        time_present[:, 0] = False
        state_mask = time_present[:, :, None, None].to(slots.dtype)
        visual_read = visual_read * state_mask
        update = self.update(torch.cat((slots, visual_read), dim=-1))
        updated = (slots + update) * state_mask

        flat_updated = updated.reshape(batch * time, slot_count, width)
        local_delta = self.write_o(
            _masked_attention(
                self.write_q(flat_local),
                self.write_k(flat_updated),
                self.write_v(flat_updated),
                torch.ones(
                    batch * time,
                    slot_count,
                    dtype=torch.bool,
                    device=visual.device,
                ),
                heads=self.heads,
            )
        ).reshape(batch, time, count, self.visual_width)
        local_mask = (
            selection.valid.to(visual.device)[..., None].to(local_delta.dtype)
            * time_present[:, :, None, None].to(local_delta.dtype)
        )
        local_delta = local_delta * local_mask
        direct = scatter_arm_residual(
            local_delta,
            selection,
            support,
            sequence_length=visual.shape[1],
        )
        if force_zero_controller:
            direct = torch.zeros_like(direct)
        else:
            # The output projection and the FP32 parameter both initialize at zero.
            # Using a zero-centered residual scale avoids a dead product of two zeros.
            direct = direct * (1.0 + self.gate.float()).to(direct.dtype)

        height, width_px = self.support_shape
        eef_logits = self.eef_head(visual_read.mean(dim=2)).reshape(
            batch, time, height, width_px
        )
        eef_logits = eef_logits * time_present[:, :, None, None].to(eef_logits.dtype)
        telemetry = {
            "visual_read_rms": visual_read.float().square().mean().sqrt().detach(),
            "direct_write_rms": direct.float().square().mean().sqrt().detach(),
            "gate": self.gate.float().detach(),
        }
        return ArmStageOutput(updated, visual_read, direct, eef_logits, telemetry)


class BimanualControllerStage(nn.Module):
    def __init__(
        self,
        *,
        visual_width: int = 3072,
        slot_width: int = 384,
        heads: int = 8,
        support_shape: tuple[int, int] = (15, 20),
    ) -> None:
        super().__init__()
        kwargs = {
            "visual_width": visual_width,
            "slot_width": slot_width,
            "heads": heads,
            "support_shape": support_shape,
        }
        self.left = ArmControllerStage(**kwargs)
        self.right = ArmControllerStage(**kwargs)

    def forward(
        self,
        visual: Tensor,
        slots: BimanualSlots,
        selection: SparseTubeSelection,
        support: Tensor,
        present: Tensor,
        *,
        force_zero_visual_values: bool = False,
        force_zero_controller: bool = False,
    ) -> ControllerOutput:
        if support.ndim != 5 or support.shape[2] != 2:
            raise ValueError("support must have shape (B, 21, 2, H, W)")
        if present.ndim != 3 or present.shape[2] != 2:
            raise ValueError("presence must have shape (B, 21, 2)")
        left = self.left(
            visual,
            slots.left,
            _arm_selection(selection, 0),
            support[:, :, 0],
            present[:, :, 0],
            force_zero_visual_values=force_zero_visual_values,
            force_zero_controller=force_zero_controller,
        )
        right = self.right(
            visual,
            slots.right,
            _arm_selection(selection, 1),
            support[:, :, 1],
            present[:, :, 1],
            force_zero_visual_values=force_zero_visual_values,
            force_zero_controller=force_zero_controller,
        )
        return ControllerOutput(
            visual=visual + left.direct_write + right.direct_write,
            slots=BimanualSlots(left.slots, right.slots, slots.destination_time),
            visual_reads=(left.visual_read, right.visual_read),
            direct_writes=(left.direct_write, right.direct_write),
            eef_logits=(left.eef_logits, right.eef_logits),
            telemetry={"left": left.telemetry, "right": right.telemetry},
        )
