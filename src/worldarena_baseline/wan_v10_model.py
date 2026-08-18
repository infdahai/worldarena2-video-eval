"""Install and bind v10 relational attention to the frozen clean parent."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import random

import numpy as np
import torch
from torch import Tensor, nn

from .wan_v10_attention import (
    ActionRelationEncoder,
    AttentionCallable,
    FactorizedRelationalSelfAttention,
    RelationCondition,
    RopeApplyCallable,
)


V10_BLOCKS = tuple(range(6, 18))
HIDDEN_EEF_BLOCKS = (11, 17)
WAN_WIDTH = 3072


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
    if not bool(getattr(base, "_v10_test_fixture", False)) and width != WAN_WIDTH:
        raise ValueError("v10 production Wan attention width must be 3072")
    return width


def _set_initialization_seed(seed: int) -> None:
    if type(seed) is not int or seed < 0:
        raise ValueError("v10 initialization seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _install_condition_hook(wrapper: FactorizedRelationalSelfAttention) -> None:
    def inject(module: nn.Module, args: tuple[object, ...], kwargs: dict[str, object]):
        if "relation_condition" in kwargs:
            raise RuntimeError("v10 relation condition must be model-bound")
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
            raise RuntimeError("v10 relational attention executed without a bound condition")
        module.condition_use_count += 1  # type: ignore[attr-defined]
        return args, {
            **kwargs,
            "relation_condition": condition,
            "checkpoint_replay_release": replay_release,
        }

    wrapper.register_forward_pre_hook(inject, with_kwargs=True)


def install_v10_relational_band(
    backbone: nn.Module,
    block_indices: Sequence[int],
    *,
    attention_fn: AttentionCallable,
    rope_apply_fn: RopeApplyCallable,
    initialization_seed: int,
    num_heads: int = 24,
    head_dim: int = 128,
    relation_rank: int = 16,
    expected_latent_times: int = 21,
    encoded_width: int = 128,
) -> dict[int, FactorizedRelationalSelfAttention]:
    indices = tuple(block_indices)
    if indices != V10_BLOCKS:
        raise ValueError("v10 permits exactly Wan blocks 6-17")
    if not hasattr(backbone, "blocks") or len(backbone.blocks) <= V10_BLOCKS[-1]:
        raise ValueError("Wan backbone lacks v10 blocks 6-17")
    if relation_rank != (2 if num_heads != 24 else 16):
        raise ValueError("v10 production relation rank must be 16")
    originals: list[tuple[int, nn.Module, nn.Module]] = []
    trainability: list[tuple[nn.Parameter, bool]] = []
    for index in indices:
        owner = _self_attention_owner(backbone.blocks[index])
        base = owner.self_attn
        width = _projection_width(base)
        if width != num_heads * head_dim:
            raise ValueError("Wan attention width differs from v10 head contract")
        originals.append((index, owner, base))
        trainability.extend((parameter, parameter.requires_grad) for parameter in base.parameters())

    _set_initialization_seed(initialization_seed)
    state_encoder = ActionRelationEncoder(state_width=20, encoded_width=encoded_width)
    wrappers: dict[int, FactorizedRelationalSelfAttention] = {}
    try:
        for index, owner, base in originals:
            hidden_head = nn.Linear(num_heads * head_dim, 2) if index in HIDDEN_EEF_BLOCKS else None
            wrapper = FactorizedRelationalSelfAttention(
                base,
                state_encoder=state_encoder,
                attention_fn=attention_fn,
                rope_apply_fn=rope_apply_fn,
                num_heads=num_heads,
                head_dim=head_dim,
                relation_rank=relation_rank,
                expected_latent_times=expected_latent_times,
                hidden_eef_head=hidden_head,
            )
            _install_condition_hook(wrapper)
            owner.self_attn = wrapper
            wrappers[index] = wrapper
    except Exception:
        for _index, owner, base in originals:
            owner.self_attn = base
        for parameter, requires_grad in trainability:
            parameter.requires_grad_(requires_grad)
        raise
    return wrappers


class ParentPlusRelationalWan(nn.Module):
    """Frozen support-gated raster parent plus v10 native relational band."""

    def __init__(
        self,
        backbone: nn.Module,
        parent_adapter: nn.Module,
        wrappers: Mapping[int, FactorizedRelationalSelfAttention],
    ) -> None:
        super().__init__()
        if tuple(wrappers) != V10_BLOCKS:
            raise ValueError("v10 wrappers must be exactly blocks 6-17")
        parent = _unwrap(parent_adapter)
        points = tuple(getattr(parent, "injection_points", ()))
        if not points or not bool(getattr(parent, "raster_support_gating", False)):
            raise ValueError("v10 requires the frozen support-gated raster parent")
        first_encoder = next(iter(wrappers.values())).state_encoder
        if any(wrapper.state_encoder is not first_encoder for wrapper in wrappers.values()):
            raise ValueError("v10 blocks must share one relation state encoder")
        for index, wrapper in wrappers.items():
            if _self_attention_owner(backbone.blocks[index]).self_attn is not wrapper:
                raise ValueError("v10 wrapper is not installed at its declared block")
        self.relation_encoder = first_encoder
        self.relation_wrappers = nn.ModuleDict({str(index): wrapper for index, wrapper in wrappers.items()})
        self.backbone = backbone
        self.parent_adapter = parent_adapter
        self.selected_blocks = V10_BLOCKS
        self.parent_injection_points = points
        self._enable_exact_trainables()

    def _enable_exact_trainables(self) -> None:
        self.backbone.requires_grad_(False)
        self.parent_adapter.requires_grad_(False)
        self.relation_encoder.requires_grad_(True)
        for wrapper in self.relation_wrappers.values():
            for projection_name in ("q", "k", "v", "o"):
                getattr(wrapper.base, projection_name).requires_grad_(True)
            wrapper.relation_q.requires_grad_(True)
            wrapper.relation_k.requires_grad_(True)
            wrapper.relation_gate.requires_grad_(True)
            if wrapper.hidden_eef_head is not None:
                wrapper.hidden_eef_head.requires_grad_(True)

    @staticmethod
    def _pre_hook(residual: Tensor):
        def inject(_module: nn.Module, args: tuple[object, ...]) -> tuple[object, ...]:
            if not args or not isinstance(args[0], Tensor) or args[0].shape != residual.shape:
                raise RuntimeError("frozen parent residual shape differs from Wan tokens")
            return (args[0] + residual.to(device=args[0].device, dtype=args[0].dtype), *args[1:])

        return inject

    def _uses_activation_checkpoint(self, point: int) -> bool:
        block = self.backbone.blocks[point]
        return isinstance(getattr(block, "block", None), nn.Module)

    def _bind_condition(self, condition: RelationCondition) -> object:
        for wrapper in self.relation_wrappers.values():
            if wrapper.bound_condition is not None or wrapper._checkpoint_condition is not None:
                raise RuntimeError("v10 relation wrapper already has a bound condition")
        token = object()
        for point, wrapper in self.relation_wrappers.items():
            wrapper.condition_use_count = 0
            wrapper.bound_condition = condition
            wrapper._condition_token = token
            if self._uses_activation_checkpoint(int(point)):
                wrapper._checkpoint_condition = condition
                wrapper._checkpoint_token = token
        return token

    def _clear_condition(self, token: object, *, retain_for_checkpoint: bool) -> None:
        for wrapper in self.relation_wrappers.values():
            if wrapper._condition_token is token:
                wrapper.bound_condition = None
                wrapper._condition_token = None
            if not retain_for_checkpoint and wrapper._checkpoint_token is token:
                wrapper._checkpoint_condition = None
                wrapper._checkpoint_token = None

    def release_completed_backward_conditions(self) -> None:
        for wrapper in self.relation_wrappers.values():
            if wrapper.bound_condition is not None:
                raise RuntimeError("cannot release v10 condition during active forward")
            wrapper._checkpoint_condition = None
            wrapper._checkpoint_token = None

    def hidden_eef_predictions(self) -> dict[int, Tensor]:
        result: dict[int, Tensor] = {}
        for point in HIDDEN_EEF_BLOCKS:
            value = self.relation_wrappers[str(point)].last_hidden_eef_prediction
            if value is None:
                raise RuntimeError("v10 hidden EEF prediction is unavailable")
            result[point] = value
        return result

    def set_hidden_eef_backbone_scale(self, scale: float) -> None:
        value = float(scale)
        if not 0.0 <= value <= 1.0:
            raise ValueError("hidden EEF backbone scale must be in [0,1]")
        for point in HIDDEN_EEF_BLOCKS:
            self.relation_wrappers[str(point)].hidden_eef_backbone_scale = value

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
        relation_condition: RelationCondition,
        action_scale: float = 1.0,
    ) -> Tensor | list[Tensor]:
        if not torch.isfinite(torch.as_tensor(action_scale)):
            raise ValueError("action_scale must be finite")
        batch = relation_condition.arm_present.shape[0]
        present = torch.ones(batch, dtype=torch.bool, device=relation_condition.arm_present.device)
        if action_present is not None:
            flat = action_present.reshape(-1)
            if flat.shape != (batch,) or not torch.all((flat == 0) | (flat == 1)):
                raise ValueError("action_present must contain one zero/one value per sample")
            present = flat.bool().to(device=present.device)
        effective_condition = replace(
            relation_condition,
            arm_present=relation_condition.arm_present & present[:, None, None],
            motion_active=relation_condition.motion_active & present[:, None, None],
        )
        with torch.no_grad():
            parent = self.parent_adapter(
                action_raster,
                t,
                seq_len=seq_len,
                condition_support=condition_support,
                action_present=action_present,
            )
        if not isinstance(parent, Mapping) or set(parent) != set(self.parent_injection_points):
            raise RuntimeError("frozen parent residual keys differ")
        handles = []
        token: object | None = None
        result_tensors: tuple[Tensor, ...] = ()
        try:
            token = self._bind_condition(effective_condition)
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
            elif isinstance(result, list) and result and all(isinstance(item, Tensor) for item in result):
                result_tensors = tuple(result)
            else:
                raise TypeError("Wan backbone must return a tensor or non-empty tensor list")
            self.hidden_eef_predictions()
            return result
        finally:
            for handle in handles:
                handle.remove()
            if token is not None:
                self._clear_condition(
                    token,
                    retain_for_checkpoint=bool(
                        result_tensors
                        and torch.is_grad_enabled()
                        and any(item.requires_grad for item in result_tensors)
                    ),
                )


def v10_trainable_parameter_names(model: nn.Module) -> set[str]:
    if not isinstance(model, ParentPlusRelationalWan):
        raise TypeError("v10 whitelist requires ParentPlusRelationalWan")
    allowed_ids = {id(parameter) for parameter in model.relation_encoder.parameters()}
    for wrapper in model.relation_wrappers.values():
        for projection_name in ("q", "k", "v", "o"):
            allowed_ids.update(id(parameter) for parameter in getattr(wrapper.base, projection_name).parameters())
        allowed_ids.update(id(parameter) for parameter in wrapper.relation_q.parameters())
        allowed_ids.update(id(parameter) for parameter in wrapper.relation_k.parameters())
        allowed_ids.add(id(wrapper.relation_gate))
        if wrapper.hidden_eef_head is not None:
            allowed_ids.update(id(parameter) for parameter in wrapper.hidden_eef_head.parameters())
    actual = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not actual or {id(parameter) for parameter in actual.values()} != allowed_ids:
        raise ValueError("v10 trainable parameters differ from the exact four-family contract")
    if len(actual) != len(allowed_ids):
        raise ValueError("v10 trainable parameter aliases are duplicated")
    gates = [parameter for name, parameter in actual.items() if name.endswith("relation_gate")]
    if len(gates) != len(V10_BLOCKS) or any(parameter.dtype != torch.float32 for parameter in gates):
        raise ValueError("v10 requires twelve FP32 relation gates")
    return set(actual)


def v10_initialized_state_sha256(model: nn.Module) -> str:
    names = v10_trainable_parameter_names(model)
    named = dict(model.named_parameters())
    digest = hashlib.sha256()
    for name in sorted(names):
        value = named[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def relation_gates_enabled(model: ParentPlusRelationalWan, *, enabled: bool):
    if enabled:
        yield
        return
    saved = {
        key: wrapper.relation_gate.detach().clone()
        for key, wrapper in model.relation_wrappers.items()
    }
    try:
        with torch.no_grad():
            for wrapper in model.relation_wrappers.values():
                wrapper.relation_gate.zero_()
        yield
    finally:
        with torch.no_grad():
            for key, wrapper in model.relation_wrappers.items():
                wrapper.relation_gate.copy_(saved[key])
