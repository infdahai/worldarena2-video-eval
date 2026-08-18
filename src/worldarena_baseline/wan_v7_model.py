"""Frozen-parent Wan integration for the bounded v7 SE(3) mechanism probe.

This module deliberately patches only the three approved self-attention
instances after all parent/Wan weights have loaded.  It does not fork Wan's
source or introduce a second attention implementation: the production RoPE
and attention callables are supplied by the runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .wan_se3_attention import (
    AttentionCallable,
    RopeApplyCallable,
    SE3AugmentedSelfAttention,
    invert_arm_transforms,
)


STAGE_A_BLOCKS = (8, 16, 24)
WAN_WIDTH = 3072
WAN_HEADS = 24
WAN_HEAD_DIM = 128
SE3_LATENT_FRAMES = 21


@dataclass(frozen=True)
class _BoundSE3Condition:
    arm_transform: Tensor
    arm_inverse: Tensor
    arm_present: Tensor


def _unwrap(module: nn.Module) -> nn.Module:
    """Return a DDP-style wrapped module without making DDP a dependency."""

    child = getattr(module, "module", None)
    return child if isinstance(child, nn.Module) else module


def _infer_attention_dimensions(base: nn.Module) -> tuple[int, int, int]:
    """Verify the native module exposes the exact Stage-A Wan geometry."""

    q = getattr(base, "q", None)
    q_weight = getattr(q, "weight", None)
    width = None
    if isinstance(q_weight, Tensor) and q_weight.ndim == 2:
        if q_weight.shape[0] != q_weight.shape[1]:
            raise ValueError("Wan q projection must be square")
        width = int(q_weight.shape[0])
    if width is None:
        for name in ("hidden_size", "dim", "embed_dim", "inner_dim"):
            value = getattr(base, name, None)
            if isinstance(value, int):
                width = value
                break
    heads = None
    for name in ("num_heads", "heads", "n_heads"):
        value = getattr(base, name, None)
        if isinstance(value, int):
            heads = value
            break
    head_dim = None
    for name in ("head_dim", "dim_head"):
        value = getattr(base, name, None)
        if isinstance(value, int):
            head_dim = value
            break
    if width is None or heads is None:
        raise ValueError("cannot verify Wan attention width and head count")
    if head_dim is None:
        if width % heads:
            raise ValueError("Wan attention width must divide evenly across heads")
        head_dim = width // heads
    if (width, heads, head_dim) != (WAN_WIDTH, WAN_HEADS, WAN_HEAD_DIM):
        raise ValueError(
            "v7 Stage A requires Wan width/head count 3072/24 with head dimension 128"
        )
    return width, heads, head_dim


def _validate_installation_targets(
    backbone: nn.Module,
    block_indices: Sequence[int],
) -> tuple[int, ...]:
    if not hasattr(backbone, "blocks"):
        raise ValueError("backbone must expose a blocks sequence")
    indices = tuple(block_indices)
    if not indices:
        raise ValueError("block_indices must be non-empty")
    if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
        raise ValueError("block indices must be integers")
    if len(set(indices)) != len(indices):
        raise ValueError("block indices must be unique")
    if indices != STAGE_A_BLOCKS:
        raise ValueError("v7 Stage A permits only blocks (8, 16, 24)")
    blocks = backbone.blocks
    for index in indices:
        if index < 0 or index >= len(blocks):
            raise ValueError("selected attention block is missing from backbone")
        attention = _self_attention_owner(blocks[index]).self_attn
        if isinstance(attention, SE3AugmentedSelfAttention):
            raise ValueError("selected Wan attention is already v7 wrapped")
        _infer_attention_dimensions(attention)
    return indices


def _self_attention_owner(block: nn.Module) -> nn.Module:
    """Find a Wan attention module before or after block checkpoint wrapping."""

    if hasattr(block, "self_attn"):
        return block
    inner = getattr(block, "block", None)
    if isinstance(inner, nn.Module) and hasattr(inner, "self_attn"):
        return inner
    raise ValueError("selected Wan block must expose self_attn")


def _install_condition_hook(wrapper: SE3AugmentedSelfAttention) -> None:
    """Install a persistent native-call adapter on a v7 self-attention module.

    Wan blocks invoke ``self_attn(x, seq_lens, grid_sizes, freqs)`` and cannot
    receive the v7 condition in their public call signature.  The adapter
    injects a condition bound by :class:`ParentPlusSE3Wan` immediately before
    native attention executes.  It remains installed across activation
    checkpoint recomputation; only the bound per-forward tensors are cleared.
    """

    wrapper.bound_condition = None  # type: ignore[attr-defined]
    wrapper._condition_token = None  # type: ignore[attr-defined]
    wrapper._checkpoint_condition = None  # type: ignore[attr-defined]
    wrapper._checkpoint_token = None  # type: ignore[attr-defined]
    wrapper.condition_use_count = 0  # type: ignore[attr-defined]

    def inject_condition(
        module: nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        if any(
            name in kwargs
            for name in ("arm_transform", "arm_inverse", "arm_present")
        ):
            raise RuntimeError("v7 SE(3) attention condition must be model-bound")
        condition = getattr(module, "bound_condition", None)
        replay_release = None
        if condition is None:
            # Checkpointed Wan blocks replay after the public forward returns.
            # This private autograd-lifetime copy is cleared by the outer
            # module immediately after this checkpoint replay returns; it is
            # never used for a new model forward because _bind_condition
            # rejects an outstanding copy.
            condition = getattr(module, "_checkpoint_condition", None)
            if condition is not None:
                token = getattr(module, "_checkpoint_token", None)

                def replay_release() -> None:
                    if getattr(module, "_checkpoint_token", None) is token:
                        module._checkpoint_condition = None  # type: ignore[attr-defined]
                        module._checkpoint_token = None  # type: ignore[attr-defined]
        if condition is None:
            raise RuntimeError("v7 SE(3) attention executed without a bound condition")
        module.condition_use_count += 1  # type: ignore[attr-defined]
        return args, {
            **kwargs,
            "arm_transform": condition.arm_transform,
            "arm_inverse": condition.arm_inverse,
            "arm_present": condition.arm_present,
            "checkpoint_replay_release": replay_release,
        }

    # Persistent by design: checkpoint recomputation can happen after
    # ParentPlusSE3Wan.forward returns.  The public binding is reset in the
    # model's finally block.  Do not use an outer full-backward hook for the
    # private copy: in the production Wan checkpoint topology it can run
    # before a selected block is recomputed.  The wrapper's forward-level
    # finally releases it after replay instead.
    wrapper.register_forward_pre_hook(inject_condition, with_kwargs=True)


def install_v7_attention(
    backbone: nn.Module,
    block_indices: Sequence[int],
    rope_apply_fn: RopeApplyCallable,
    attention_fn: AttentionCallable,
) -> dict[int, SE3AugmentedSelfAttention]:
    """Replace exactly the three approved Wan self-attention modules in-place.

    Call only after loading the immutable clean parent and Wan state dicts.
    The original native attention remains reachable as ``wrapper.base`` and is
    frozen before replacement.
    """

    indices = _validate_installation_targets(backbone, block_indices)
    originals: dict[int, tuple[nn.Module, nn.Module]] = {}
    for index in indices:
        owner = _self_attention_owner(backbone.blocks[index])
        originals[index] = owner, owner.self_attn
    trainability = [
        (parameter, parameter.requires_grad)
        for _owner, base in originals.values()
        for parameter in base.parameters()
    ]
    wrappers: dict[int, SE3AugmentedSelfAttention] = {}
    try:
        for index in indices:
            owner, base = originals[index]
            base.requires_grad_(False)
            wrapper = SE3AugmentedSelfAttention(
                base,
                attention_fn=attention_fn,
                rope_apply_fn=rope_apply_fn,
                num_heads=WAN_HEADS,
                head_dim=WAN_HEAD_DIM,
            )
            _install_condition_hook(wrapper)
            owner.self_attn = wrapper
            wrappers[index] = wrapper
        return wrappers
    except Exception:
        for owner, base in originals.values():
            owner.self_attn = base
        for parameter, requires_grad in trainability:
            parameter.requires_grad_(requires_grad)
        raise


class ParentPlusSE3Wan(nn.Module):
    """Run an immutable support-gated parent plus three SE(3) channel gates."""

    def __init__(
        self,
        backbone: nn.Module,
        parent_adapter: nn.Module,
        geometry_wrappers: Mapping[int, SE3AugmentedSelfAttention],
    ) -> None:
        super().__init__()
        if not hasattr(backbone, "blocks"):
            raise ValueError("backbone must expose blocks")
        if tuple(geometry_wrappers) != STAGE_A_BLOCKS:
            raise ValueError("geometry wrappers must be exactly blocks (8, 16, 24)")
        if any(
            not isinstance(wrapper, SE3AugmentedSelfAttention)
            for wrapper in geometry_wrappers.values()
        ):
            raise TypeError("geometry wrappers must be SE3AugmentedSelfAttention")
        for point, wrapper in geometry_wrappers.items():
            if (
                point >= len(backbone.blocks)
                or _self_attention_owner(backbone.blocks[point]).self_attn is not wrapper
            ):
                raise ValueError("geometry wrapper must be installed on its selected Wan block")
            if not hasattr(wrapper, "bound_condition"):
                raise ValueError("geometry wrapper is missing its v7 condition hook")

        parent_module = _unwrap(parent_adapter)
        parent_points = tuple(getattr(parent_module, "injection_points", ()))
        if not parent_points:
            raise ValueError("clean parent adapter must expose injection_points")
        if max(parent_points) >= len(backbone.blocks):
            raise ValueError("parent injection point exceeds backbone blocks")
        if not bool(getattr(parent_module, "raster_support_gating", False)):
            raise ValueError("v7 requires the frozen support-gated clean parent")
        # Register these first so canonical parameter names are
        # ``geometry_wrappers.<block>.gate`` rather than a backbone alias.
        self.geometry_wrappers = nn.ModuleDict(
            {str(point): wrapper for point, wrapper in geometry_wrappers.items()}
        )
        self.backbone = backbone
        self.parent_adapter = parent_adapter
        self.selected_blocks = STAGE_A_BLOCKS
        self.parent_injection_points = parent_points

        self.backbone.requires_grad_(False)
        self.parent_adapter.requires_grad_(False)
        for wrapper in self.geometry_wrappers.values():
            wrapper.base.requires_grad_(False)
            wrapper.geometry.requires_grad_(False)
            wrapper.gate.requires_grad_(True)

    @staticmethod
    def _pre_hook(residual: Tensor):
        def inject(_module: nn.Module, args: tuple[object, ...]) -> tuple[object, ...]:
            if not args or not isinstance(args[0], Tensor) or args[0].shape != residual.shape:
                raise RuntimeError("frozen parent residual does not match Wan tokens")
            tokens = args[0]
            return (
                tokens + residual.to(device=tokens.device, dtype=tokens.dtype),
                *args[1:],
            )

        return inject

    @staticmethod
    def _action_mask(
        action_present: Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        if action_present is None:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        if action_present.ndim == 2 and action_present.shape == (batch_size, 1):
            action_present = action_present[:, 0]
        if action_present.ndim != 1 or action_present.shape[0] != batch_size:
            raise ValueError("action_present must have shape (B,) or (B, 1)")
        if not torch.isfinite(action_present).all() or not torch.all(
            (action_present == 0) | (action_present == 1)
        ):
            raise ValueError("action_present must contain only zero or one")
        return action_present.to(device=device, dtype=torch.bool)

    @staticmethod
    def _condition(
        arm_transform: Tensor,
        arm_present: Tensor,
        action_present: Tensor | None,
    ) -> _BoundSE3Condition:
        if arm_transform.ndim != 5 or tuple(arm_transform.shape[1:]) != (
            2,
            SE3_LATENT_FRAMES,
            4,
            4,
        ):
            raise ValueError("se3_arm_transform must have shape (B, 2, 21, 4, 4)")
        if arm_transform.dtype != torch.float32:
            raise ValueError("se3_arm_transform must remain float32")
        if arm_present.shape != arm_transform.shape[:3]:
            raise ValueError("se3_arm_present must have shape (B, 2, 21)")
        if arm_present.device != arm_transform.device:
            raise ValueError("SE(3) condition tensors must share a device")
        if arm_present.dtype != torch.bool:
            raise ValueError("se3_arm_present must be bool")
        if not torch.isfinite(arm_transform).all():
            raise ValueError("se3_arm_transform must be finite")
        action_mask = ParentPlusSE3Wan._action_mask(
            action_present, batch_size=arm_transform.shape[0], device=arm_transform.device
        )
        # Null action is semantically distinct from a hold action: it removes
        # both fixed head groups, while hold retains valid arm presence.
        effective_present = arm_present & action_mask[:, None, None]
        return _BoundSE3Condition(
            arm_transform=arm_transform,
            arm_inverse=invert_arm_transforms(arm_transform),
            arm_present=effective_present,
        )

    def _uses_activation_checkpoint(self, point: int) -> bool:
        """Recognize the repository's checkpoint wrapper without importing it.

        ``enable_wan_block_checkpointing`` wraps a Wan block in a module with
        a ``block`` attribute that owns the selected self-attention.  Stage A
        must retain the private condition only for that topology; retaining it
        for a normal forward would incorrectly reject the next batch.
        """

        block = self.backbone.blocks[point]
        return (
            isinstance(getattr(block, "block", None), nn.Module)
            and _self_attention_owner(block).self_attn is self.geometry_wrappers[str(point)]
        )

    def _bind_condition(self, condition: _BoundSE3Condition) -> object:
        # Validate all wrappers before mutating any.  In particular, an
        # attempted second forward must not erase a legitimate first forward's
        # condition while that first graph is waiting for checkpoint replay.
        for wrapper in self.geometry_wrappers.values():
            if (
                wrapper.bound_condition is not None  # type: ignore[attr-defined]
                or wrapper._checkpoint_condition is not None  # type: ignore[attr-defined]
            ):
                raise RuntimeError("v7 geometry wrapper already has a bound condition")
        token = object()
        for point, wrapper in self.geometry_wrappers.items():
            wrapper.condition_use_count = 0  # type: ignore[attr-defined]
            wrapper.bound_condition = condition  # type: ignore[attr-defined]
            wrapper._condition_token = token  # type: ignore[attr-defined]
            if self._uses_activation_checkpoint(int(point)):
                wrapper._checkpoint_condition = condition  # type: ignore[attr-defined]
                wrapper._checkpoint_token = token  # type: ignore[attr-defined]
            else:
                wrapper._checkpoint_condition = None  # type: ignore[attr-defined]
                wrapper._checkpoint_token = None  # type: ignore[attr-defined]
        return token

    def _clear_condition(self, token: object, *, retain_for_checkpoint: bool) -> None:
        for wrapper in self.geometry_wrappers.values():
            if wrapper._condition_token is token:  # type: ignore[attr-defined]
                wrapper.bound_condition = None  # type: ignore[attr-defined]
                wrapper._condition_token = None  # type: ignore[attr-defined]
            if not retain_for_checkpoint and wrapper._checkpoint_token is token:  # type: ignore[attr-defined]
                wrapper._checkpoint_condition = None  # type: ignore[attr-defined]
                wrapper._checkpoint_token = None  # type: ignore[attr-defined]

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
        action_present: Tensor | None = None,
        action_scale: float = 1.0,
        se3_arm_transform: Tensor,
        se3_arm_present: Tensor,
    ) -> Tensor:
        if not torch.isfinite(torch.as_tensor(action_scale)):
            raise ValueError("action_scale must be finite")
        condition = self._condition(se3_arm_transform, se3_arm_present, action_present)
        with torch.no_grad():
            parent = self.parent_adapter(
                action_raster,
                t,
                seq_len=seq_len,
                condition_support=condition_support,
                action_present=action_present,
            )
        if not isinstance(parent, Mapping) or set(parent) != set(self.parent_injection_points):
            raise RuntimeError("frozen parent residual keys do not match parent injection points")

        handles = []
        result: Tensor | list[Tensor] | None = None
        result_tensors: tuple[Tensor, ...] = ()
        binding_token: object | None = None
        try:
            binding_token = self._bind_condition(condition)
            for point in self.parent_injection_points:
                residual = parent[point]
                if not isinstance(residual, Tensor):
                    raise TypeError("frozen parent residual must be a tensor")
                handles.append(
                    self.backbone.blocks[point].register_forward_pre_hook(
                        self._pre_hook(residual * float(action_scale))
                    )
                )
            result = self.backbone(x, t, context, seq_len, y=y)
            if isinstance(result, Tensor):
                result_tensors = (result,)
            elif (
                isinstance(result, list)
                and result
                and all(isinstance(value, Tensor) for value in result)
            ):
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
                    retain_for_checkpoint=bool(
                        result_tensors
                        and any(value.requires_grad for value in result_tensors)
                        and torch.is_grad_enabled()
                    ),
                )


def v7_trainable_parameter_names(model: nn.Module) -> set[str]:
    """Return the fail-closed Stage-A gate-only trainable whitelist."""

    expected = {f"geometry_wrappers.{point}.gate" for point in STAGE_A_BLOCKS}
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if actual != expected:
        raise ValueError("v7 trainable parameters must be exactly the three geometry gates")
    return actual
