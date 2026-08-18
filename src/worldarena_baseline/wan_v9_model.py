"""Wan blocks 8-13 integration for v9 phase-locked action attention."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .wan_v9_attention import PhaseLockedActionCrossAttention
from .wan_v9_tokens import V9ActionTokenizer


V9_BLOCKS = (8, 9, 10, 11, 12, 13)
WAN_WIDTH = 3072


@dataclass(frozen=True)
class _BoundV9Condition:
    action_tokens: Tensor
    arm_present: Tensor


def _unwrap(module: nn.Module) -> nn.Module:
    child = getattr(module, "module", None)
    return child if isinstance(child, nn.Module) else module


def _self_attention_owner(block: nn.Module) -> nn.Module:
    if hasattr(block, "self_attn"):
        return block
    child = getattr(block, "block", None)
    if isinstance(child, nn.Module) and hasattr(child, "self_attn"):
        return child
    raise ValueError("Wan block must expose self_attn")


def _projection_width(base: nn.Module) -> int:
    if not all(hasattr(base, name) for name in ("q", "k", "v", "o")):
        raise TypeError("Wan attention must expose q/k/v/o")
    weight = getattr(getattr(base, "q"), "weight", None)
    if not isinstance(weight, Tensor) or weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
        raise ValueError("Wan q projection must expose one square weight")
    width = int(weight.shape[0])
    if not bool(getattr(base, "_v9_test_fixture", False)) and width != WAN_WIDTH:
        raise ValueError("v9 production Wan attention width must be 3072")
    return width


class NativePlusPhaseLockedAttention(nn.Module):
    """Original Wan self-attention plus an independent local action path."""

    def __init__(self, base: nn.Module, cross: PhaseLockedActionCrossAttention) -> None:
        super().__init__()
        self.base = base
        self.cross = cross
        self.bound_condition: _BoundV9Condition | None = None
        self._condition_token: object | None = None
        self._checkpoint_condition: _BoundV9Condition | None = None
        self._checkpoint_token: object | None = None
        self.condition_use_count = 0

    def forward(
        self,
        x: Tensor,
        seq_lens: Tensor,
        grid_sizes: Tensor,
        freqs: object,
        *,
        action_tokens: Tensor,
        arm_present: Tensor,
        checkpoint_replay_release=None,
    ) -> Tensor:
        try:
            native = self.base(x, seq_lens, grid_sizes, freqs)
            if not isinstance(native, Tensor) or native.shape != x.shape:
                raise ValueError("native Wan self-attention output shape differs")
            cross = self.cross(
                x,
                grid_sizes=grid_sizes,
                action_tokens=action_tokens,
                arm_present=arm_present,
                seq_lens=seq_lens,
            )
            return native + cross
        finally:
            if checkpoint_replay_release is not None:
                checkpoint_replay_release()


def _install_condition_hook(wrapper: NativePlusPhaseLockedAttention) -> None:
    def inject(module: nn.Module, args: tuple[object, ...], kwargs: dict[str, object]):
        if "action_tokens" in kwargs or "arm_present" in kwargs:
            raise RuntimeError("v9 action condition must be model-bound")
        condition = module.bound_condition  # type: ignore[attr-defined]
        replay_release = None
        if condition is None:
            condition = module._checkpoint_condition  # type: ignore[attr-defined]
            if condition is not None:
                token = module._checkpoint_token  # type: ignore[attr-defined]

                def replay_release() -> None:
                    if module._checkpoint_token is token:  # type: ignore[attr-defined]
                        module._checkpoint_condition = None  # type: ignore[attr-defined]
                        module._checkpoint_token = None  # type: ignore[attr-defined]
        if condition is None:
            raise RuntimeError("v9 action attention executed without a bound condition")
        module.condition_use_count += 1  # type: ignore[attr-defined]
        return args, {
            **kwargs,
            "action_tokens": condition.action_tokens,
            "arm_present": condition.arm_present,
            "checkpoint_replay_release": replay_release,
        }

    wrapper.register_forward_pre_hook(inject, with_kwargs=True)


def install_v9_attention(
    backbone: nn.Module,
    block_indices: Sequence[int],
    *,
    action_width: int = 256,
) -> dict[int, NativePlusPhaseLockedAttention]:
    indices = tuple(block_indices)
    if indices != V9_BLOCKS:
        raise ValueError("v9 permits exactly Wan blocks 8-13")
    if not hasattr(backbone, "blocks") or len(backbone.blocks) <= V9_BLOCKS[-1]:
        raise ValueError("Wan backbone lacks v9 blocks 8-13")
    originals: list[tuple[int, nn.Module, nn.Module, int, int]] = []
    for index in indices:
        owner = _self_attention_owner(backbone.blocks[index])
        base = owner.self_attn
        if isinstance(base, NativePlusPhaseLockedAttention):
            raise ValueError("v9 attention is already installed")
        width = _projection_width(base)
        test_fixture = bool(getattr(base, "_v9_test_fixture", False))
        if not test_fixture and action_width != 256:
            raise ValueError("v9 production action width must be 256")
        inner = int(getattr(base, "_v9_test_inner_width", 768 if not test_fixture else width))
        originals.append((index, owner, base, width, inner))
    wrappers: dict[int, NativePlusPhaseLockedAttention] = {}
    try:
        for index, owner, base, width, inner in originals:
            wrapper = NativePlusPhaseLockedAttention(
                base,
                PhaseLockedActionCrossAttention(
                    visual_width=width,
                    action_width=action_width,
                    inner_width=inner,
                    num_heads=8,
                    expected_latent_times=21,
                ),
            )
            _install_condition_hook(wrapper)
            owner.self_attn = wrapper
            wrappers[index] = wrapper
    except Exception:
        for _index, owner, base, _width, _inner in originals:
            owner.self_attn = base
        raise
    return wrappers


class ParentPlusPhaseLockedActionWan(nn.Module):
    """Frozen raster parent plus v9 native and phase-locked action attention."""

    def __init__(
        self,
        backbone: nn.Module,
        parent_adapter: nn.Module,
        action_tokenizer: V9ActionTokenizer,
        wrappers: Mapping[int, NativePlusPhaseLockedAttention],
    ) -> None:
        super().__init__()
        if tuple(wrappers) != V9_BLOCKS:
            raise ValueError("v9 wrappers must be exactly blocks 8-13")
        parent = _unwrap(parent_adapter)
        points = tuple(getattr(parent, "injection_points", ()))
        if not points or not bool(getattr(parent, "raster_support_gating", False)):
            raise ValueError("v9 requires the frozen support-gated raster parent")
        for index, wrapper in wrappers.items():
            if _self_attention_owner(backbone.blocks[index]).self_attn is not wrapper:
                raise ValueError("v9 wrapper is not installed at its declared block")
        self.action_tokenizer = action_tokenizer
        self.action_wrappers = nn.ModuleDict({str(index): wrapper for index, wrapper in wrappers.items()})
        self.backbone = backbone
        self.parent_adapter = parent_adapter
        self.selected_blocks = V9_BLOCKS
        self.parent_injection_points = points
        self._enable_exact_trainables()

    def _enable_exact_trainables(self) -> None:
        self.backbone.requires_grad_(False)
        self.parent_adapter.requires_grad_(False)
        self.action_tokenizer.requires_grad_(True)
        for wrapper in self.action_wrappers.values():
            wrapper.requires_grad_(False)
            for projection_name in ("q", "k", "v", "o"):
                getattr(wrapper.base, projection_name).requires_grad_(True)
            wrapper.cross.requires_grad_(True)

    @staticmethod
    def _pre_hook(residual: Tensor):
        def inject(_module: nn.Module, args: tuple[object, ...]) -> tuple[object, ...]:
            if not args or not isinstance(args[0], Tensor) or args[0].shape != residual.shape:
                raise RuntimeError("frozen parent residual shape differs from Wan tokens")
            return (args[0] + residual.to(device=args[0].device, dtype=args[0].dtype), *args[1:])

        return inject

    @staticmethod
    def _action_mask(action_present: Tensor | None, *, batch: int, device: torch.device) -> Tensor:
        if action_present is None:
            return torch.ones(batch, dtype=torch.bool, device=device)
        if action_present.ndim == 2 and action_present.shape == (batch, 1):
            action_present = action_present[:, 0]
        if action_present.shape != (batch,) or not torch.isfinite(action_present).all():
            raise ValueError("action_present must have finite shape (batch,)")
        if not torch.all((action_present == 0) | (action_present == 1)):
            raise ValueError("action_present must contain only zero or one")
        return action_present.to(device=device, dtype=torch.bool)

    def _uses_activation_checkpoint(self, point: int) -> bool:
        block = self.backbone.blocks[point]
        return isinstance(getattr(block, "block", None), nn.Module) and _self_attention_owner(block).self_attn is self.action_wrappers[str(point)]

    def _bind_condition(self, condition: _BoundV9Condition) -> object:
        for wrapper in self.action_wrappers.values():
            if wrapper.bound_condition is not None or wrapper._checkpoint_condition is not None:
                raise RuntimeError("v9 action wrapper already has a bound condition")
        token = object()
        for point, wrapper in self.action_wrappers.items():
            wrapper.condition_use_count = 0
            wrapper.bound_condition = condition
            wrapper._condition_token = token
            if self._uses_activation_checkpoint(int(point)):
                wrapper._checkpoint_condition = condition
                wrapper._checkpoint_token = token
        return token

    def _clear_condition(self, token: object, *, retain_for_checkpoint: bool) -> None:
        for wrapper in self.action_wrappers.values():
            if wrapper._condition_token is token:
                wrapper.bound_condition = None
                wrapper._condition_token = None
            if not retain_for_checkpoint and wrapper._checkpoint_token is token:
                wrapper._checkpoint_condition = None
                wrapper._checkpoint_token = None

    def release_completed_backward_conditions(self) -> None:
        for wrapper in self.action_wrappers.values():
            if wrapper.bound_condition is not None:
                raise RuntimeError("cannot release v9 condition during active forward")
            wrapper._checkpoint_condition = None
            wrapper._checkpoint_token = None

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        context: object,
        seq_len: int,
        y: object | None = None,
        *,
        action_raster: Tensor,
        condition_support: Tensor,
        action_present: Tensor | None,
        transition_features: Mapping[str, Tensor],
        transition_arm_present: Tensor,
        action_scale: float = 1.0,
    ) -> Tensor | list[Tensor]:
        if not torch.isfinite(torch.as_tensor(action_scale)):
            raise ValueError("action_scale must be finite")
        if transition_arm_present.ndim != 3 or tuple(transition_arm_present.shape[1:]) != (2, 20):
            raise ValueError("transition_arm_present must have shape (batch,2,20)")
        action_mask = self._action_mask(
            action_present, batch=transition_arm_present.shape[0], device=transition_arm_present.device
        )
        effective_present = transition_arm_present.bool() & action_mask[:, None, None]
        tokens, present = self.action_tokenizer(transition_features, effective_present)
        condition = _BoundV9Condition(tokens, present)
        with torch.no_grad():
            parent = self.parent_adapter(
                action_raster, t, seq_len=seq_len,
                condition_support=condition_support, action_present=action_present,
            )
        if not isinstance(parent, Mapping) or set(parent) != set(self.parent_injection_points):
            raise RuntimeError("frozen parent residual keys differ")
        handles = []
        binding_token: object | None = None
        result_tensors: tuple[Tensor, ...] = ()
        try:
            binding_token = self._bind_condition(condition)
            for point in self.parent_injection_points:
                residual = parent[point]
                if not isinstance(residual, Tensor):
                    raise TypeError("frozen parent residual must be a tensor")
                handles.append(self.backbone.blocks[point].register_forward_pre_hook(self._pre_hook(residual * float(action_scale))))
            result = self.backbone(x, t, context, seq_len, y=y)
            if isinstance(result, Tensor):
                result_tensors = (result,)
            elif isinstance(result, list) and result and all(isinstance(item, Tensor) for item in result):
                result_tensors = tuple(result)
            else:
                raise TypeError("Wan backbone must return a tensor or non-empty tensor list")
            return result
        finally:
            for handle in handles:
                handle.remove()
            if binding_token is not None:
                self._clear_condition(
                    binding_token,
                    retain_for_checkpoint=bool(result_tensors and torch.is_grad_enabled() and any(item.requires_grad for item in result_tensors)),
                )


def v9_trainable_parameter_names(model: nn.Module) -> set[str]:
    if not isinstance(model, ParentPlusPhaseLockedActionWan):
        raise TypeError("v9 whitelist requires ParentPlusPhaseLockedActionWan")
    allowed_ids = {id(parameter) for parameter in model.action_tokenizer.parameters()}
    for wrapper in model.action_wrappers.values():
        allowed_ids.update(id(parameter) for parameter in wrapper.cross.parameters())
        for projection_name in ("q", "k", "v", "o"):
            allowed_ids.update(id(parameter) for parameter in getattr(wrapper.base, projection_name).parameters())
    actual = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not actual or {id(parameter) for parameter in actual.values()} != allowed_ids:
        raise ValueError("v9 trainable parameters differ from the exact native/cross/tokenizer/gate contract")
    if len(actual) != len(allowed_ids):
        raise ValueError("v9 trainable parameter aliases are duplicated")
    gate_names = {name for name in actual if name.endswith("channel_gate")}
    if len(gate_names) != len(V9_BLOCKS) or any(actual[name].dtype != torch.float32 for name in gate_names):
        raise ValueError("v9 requires six FP32 channel gates")
    return set(actual)

