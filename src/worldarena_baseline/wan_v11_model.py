from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor, nn

from .wan_v11_controller import BimanualControllerStage
from .wan_v11_sparse import select_tube_tokens
from .wan_v11_state import (
    ArmActionContent,
    BimanualCondition,
    BimanualSlots,
    BimanualSlotTokenizer,
    pack_initial_slots,
)


CONTROLLER_POINTS = (6, 16, 24)


@dataclass(frozen=True)
class WanLoopOutput:
    video: list[Tensor]
    final_visual: Tensor
    grid_sizes: Tensor


@dataclass(frozen=True)
class V11ModelOutput:
    video: list[Tensor]
    final_slots: BimanualSlots
    eef_logits: dict[int, tuple[Tensor, Tensor]]
    telemetry: dict[int, dict[str, dict[str, Tensor]]]


def _sinusoidal_embedding_1d(dim: int, position: Tensor) -> Tensor:
    if dim % 2:
        raise ValueError("Wan sinusoidal embedding width must be even")
    half = dim // 2
    position = position.to(torch.float64)
    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half, device=position.device).to(position).div(half)),
    )
    return torch.cat((torch.cos(sinusoid), torch.sin(sinusoid)), dim=1)


def validate_upstream_wan_forward_source(source: str) -> None:
    markers = (
        "x = [self.patch_embedding",
        "e = self.time_embedding",
        "context = self.text_embedding",
        "for block in self.blocks:",
        "x = block(x, **kwargs)",
        "x = self.head(x, e)",
        "x = self.unpatchify(x, grid_sizes)",
    )
    offsets = [source.find(marker) for marker in markers]
    if any(offset < 0 for offset in offsets) or offsets != sorted(offsets):
        raise RuntimeError("upstream Wan forward structure differs from the v11 pinned loop")


def run_wan_backbone_loop(
    backbone: nn.Module,
    x: list[Tensor],
    t: Tensor,
    context: list[Tensor],
    seq_len: int,
    y: list[Tensor] | None = None,
    *,
    before_block: Callable[[int, Tensor], Tensor] | None = None,
    after_block: Callable[[int, Tensor, Tensor], Tensor] | None = None,
) -> WanLoopOutput:
    if getattr(backbone, "model_type", None) == "i2v" and y is None:
        raise ValueError("i2v Wan requires conditional video input")
    device = backbone.patch_embedding.weight.device
    if backbone.freqs.device != device:
        backbone.freqs = backbone.freqs.to(device)
    if y is not None:
        x = [torch.cat((item, condition), dim=0) for item, condition in zip(x, y)]

    visual_list = [backbone.patch_embedding(item.unsqueeze(0)) for item in x]
    grid_sizes = torch.stack(
        [torch.tensor(item.shape[2:], dtype=torch.long) for item in visual_list]
    )
    tokens = [item.flatten(2).transpose(1, 2) for item in visual_list]
    seq_lens = torch.tensor([item.size(1) for item in tokens], dtype=torch.long)
    if int(seq_lens.max()) > seq_len:
        raise ValueError("Wan token sequence exceeds seq_len")
    visual = torch.cat(
        [
            torch.cat(
                (item, item.new_zeros(1, seq_len - item.size(1), item.size(2))),
                dim=1,
            )
            for item in tokens
        ]
    )

    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    time_context = (
        torch.amp.autocast("cuda", dtype=torch.float32)
        if device.type == "cuda"
        else nullcontext()
    )
    with time_context:
        batch = t.size(0)
        flat_t = t.flatten()
        e = backbone.time_embedding(
            _sinusoidal_embedding_1d(backbone.freq_dim, flat_t)
            .unflatten(0, (batch, seq_len))
            .float()
        )
        e0 = backbone.time_projection(e).unflatten(2, (6, backbone.dim))
        if device.type == "cuda" and (
            e.dtype != torch.float32 or e0.dtype != torch.float32
        ):
            raise RuntimeError("Wan time embeddings must remain FP32")

    context_lens = None
    encoded_context = backbone.text_embedding(
        torch.stack(
            [
                torch.cat(
                    (
                        item,
                        item.new_zeros(backbone.text_len - item.size(0), item.size(1)),
                    )
                )
                for item in context
            ]
        )
    )
    kwargs: dict[str, Any] = {
        "e": e0,
        "seq_lens": seq_lens,
        "grid_sizes": grid_sizes,
        "freqs": backbone.freqs,
        "context": encoded_context,
        "context_lens": context_lens,
    }
    for index, block in enumerate(backbone.blocks):
        if before_block is not None:
            visual = before_block(index, visual)
        block_input = visual
        visual = block(visual, **kwargs)
        if after_block is not None:
            visual = after_block(index, visual, block_input)

    headed = backbone.head(visual, e)
    video = [item.float() for item in backbone.unpatchify(headed, grid_sizes)]
    return WanLoopOutput(video=video, final_visual=visual, grid_sizes=grid_sizes)


def _mask_arm(arm: ArmActionContent, sample_present: Tensor) -> ArmActionContent:
    present = arm.arm_present & sample_present[:, None]
    active = arm.motion_active & sample_present[:, None]
    return replace(arm, arm_present=present, motion_active=active)


class ParentPlusBimanualControllerWan(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        parent_adapter: nn.Module,
        tokenizer: BimanualSlotTokenizer,
        controller_stages: Mapping[int, BimanualControllerStage],
    ) -> None:
        super().__init__()
        if tuple(controller_stages) != CONTROLLER_POINTS:
            raise ValueError("v11 controller stages must be exactly 6, 16, and 24")
        parent = getattr(parent_adapter, "module", parent_adapter)
        points = tuple(getattr(parent, "injection_points", ()))
        if not points or not bool(getattr(parent, "raster_support_gating", False)):
            raise ValueError("v11 requires the frozen support-gated clean parent")
        if not hasattr(backbone, "blocks") or len(backbone.blocks) <= CONTROLLER_POINTS[-1]:
            raise ValueError("Wan backbone does not contain all v11 controller points")
        self.backbone = backbone
        self.parent_adapter = parent_adapter
        self.tokenizer = tokenizer
        self.controller_stages = nn.ModuleDict(
            {str(point): controller_stages[point] for point in CONTROLLER_POINTS}
        )
        self.parent_injection_points = points
        self.backbone.requires_grad_(False)
        self.parent_adapter.requires_grad_(False)
        self.tokenizer.requires_grad_(True)
        self.controller_stages.requires_grad_(True)

    def forward(
        self,
        x: list[Tensor],
        t: Tensor,
        context: list[Tensor],
        seq_len: int,
        y: list[Tensor] | None = None,
        *,
        action_raster: Tensor,
        condition_support: Tensor,
        action_present: Tensor | None,
        bimanual_condition: BimanualCondition,
        action_scale: float = 1.0,
        force_zero_visual_values: bool = False,
        force_zero_controller: bool = False,
    ) -> V11ModelOutput:
        scale = float(action_scale)
        if not torch.isfinite(torch.tensor(scale)):
            raise ValueError("action_scale must be finite")
        batch = bimanual_condition.destination_time.shape[0]
        sample_present = torch.ones(
            batch,
            dtype=torch.bool,
            device=bimanual_condition.destination_time.device,
        )
        if action_present is not None:
            flat = action_present.reshape(-1)
            if flat.shape != (batch,) or not torch.all((flat == 0) | (flat == 1)):
                raise ValueError("action_present must contain one zero/one value per sample")
            sample_present = flat.bool().to(sample_present.device)
        effective = replace(
            bimanual_condition,
            left=_mask_arm(bimanual_condition.left, sample_present),
            right=_mask_arm(bimanual_condition.right, sample_present),
        )
        slots = pack_initial_slots(self.tokenizer, effective)
        support = torch.stack((effective.left.support, effective.right.support), dim=2)
        selection = select_tube_tokens(support, max_tokens=48)
        present = torch.stack(
            (effective.left.arm_present, effective.right.arm_present), dim=2
        )

        with torch.no_grad():
            parent_residuals = self.parent_adapter(
                action_raster,
                t,
                seq_len=seq_len,
                condition_support=condition_support,
                action_present=action_present,
            )
        if not isinstance(parent_residuals, Mapping) or set(parent_residuals) != set(
            self.parent_injection_points
        ):
            raise RuntimeError("frozen parent residual keys differ from its injection points")

        eef_logits: dict[int, tuple[Tensor, Tensor]] = {}
        telemetry: dict[int, dict[str, dict[str, Tensor]]] = {}

        def before_block(index: int, visual: Tensor) -> Tensor:
            if index not in parent_residuals:
                return visual
            residual = parent_residuals[index]
            if not isinstance(residual, Tensor) or residual.shape != visual.shape:
                raise RuntimeError("frozen parent residual shape differs from Wan tokens")
            return visual + residual.to(device=visual.device, dtype=visual.dtype) * scale

        def after_block(index: int, visual: Tensor, _block_input: Tensor) -> Tensor:
            nonlocal slots
            key = str(index)
            if key not in self.controller_stages:
                return visual
            stage_output = self.controller_stages[key](
                visual,
                slots,
                selection,
                support,
                present,
                force_zero_visual_values=force_zero_visual_values,
                force_zero_controller=force_zero_controller,
            )
            slots = stage_output.slots
            eef_logits[index] = stage_output.eef_logits
            telemetry[index] = stage_output.telemetry
            return stage_output.visual

        result = run_wan_backbone_loop(
            self.backbone,
            x,
            t,
            context,
            seq_len,
            y=y,
            before_block=before_block,
            after_block=after_block,
        )
        if tuple(eef_logits) != CONTROLLER_POINTS or tuple(telemetry) != CONTROLLER_POINTS:
            raise RuntimeError("not all v11 controller stages executed")
        return V11ModelOutput(result.video, slots, eef_logits, telemetry)


def v11_trainable_parameter_names(model: nn.Module) -> set[str]:
    if not isinstance(model, ParentPlusBimanualControllerWan):
        raise TypeError("v11 whitelist requires ParentPlusBimanualControllerWan")
    allowed_ids = {id(parameter) for parameter in model.tokenizer.parameters()}
    allowed_ids.update(id(parameter) for parameter in model.controller_stages.parameters())
    actual = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if not actual or {id(parameter) for parameter in actual.values()} != allowed_ids:
        raise ValueError("v11 trainable parameters differ from tokenizer/controller contract")
    if len(actual) != len(allowed_ids):
        raise ValueError("v11 trainable parameter aliases are duplicated")
    gates = [parameter for name, parameter in actual.items() if name.endswith(".gate")]
    if len(gates) != 6 or any(parameter.dtype != torch.float32 for parameter in gates):
        raise ValueError("v11 requires six FP32 controller gates")
    return set(actual)
