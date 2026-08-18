"""Installation and fail-closed trainable whitelist for Wan v8 direct band."""
from __future__ import annotations
from collections.abc import Sequence
from torch import nn
from .wan_se3_attention import AttentionCallable, RopeApplyCallable
from .wan_v8_attention import DirectActionBandAttention
from .wan_v7_model import ParentPlusSE3Wan, _install_condition_hook, _self_attention_owner, _unwrap

V8_BLOCKS = (8, 9, 10, 11, 12, 13)

def _owner(block: nn.Module) -> nn.Module:
    if hasattr(block, "self_attn"): return block
    inner=getattr(block,"block",None)
    if isinstance(inner,nn.Module) and hasattr(inner,"self_attn"): return inner
    raise ValueError("Wan block lacks self_attn")

def install_v8_action_band(backbone: nn.Module, block_indices: Sequence[int], rope_apply_fn: RopeApplyCallable, attention_fn: AttentionCallable) -> dict[int,DirectActionBandAttention]:
    indices=tuple(block_indices)
    if indices!=V8_BLOCKS: raise ValueError("v8 permits exactly blocks 8-13")
    if not hasattr(backbone,"blocks"): raise ValueError("backbone lacks blocks")
    originals=[]
    for i in indices:
        if i>=len(backbone.blocks): raise ValueError("v8 selected block missing")
        owner=_owner(backbone.blocks[i]); base=owner.self_attn
        if isinstance(base,DirectActionBandAttention): raise ValueError("v8 action band already installed")
        for n in ("q","k","v","o"): 
            if not hasattr(base,n): raise ValueError("Wan attention lacks native Q/K/V/O")
        originals.append((i,owner,base))
    backbone.requires_grad_(False); wrapped={}
    try:
        for i,owner,base in originals:
            wrapper=DirectActionBandAttention(base,attention_fn=attention_fn,rope_apply_fn=rope_apply_fn)
            _install_condition_hook(wrapper)
            wrapper.enable_direct_training(); owner.self_attn=wrapper; wrapped[i]=wrapper
    except Exception:
        for _,owner,base in originals: owner.self_attn=base
        raise
    return wrapped

class ParentPlusDirectActionWan(ParentPlusSE3Wan):
    """Frozen raster parent plus six shared native/geometry attention blocks."""
    def __init__(self, backbone:nn.Module, parent_adapter:nn.Module, wrappers:dict[int,DirectActionBandAttention])->None:
        nn.Module.__init__(self)
        if tuple(wrappers)!=V8_BLOCKS: raise ValueError("v8 wrappers must be blocks 8-13")
        parent=_unwrap(parent_adapter); points=tuple(getattr(parent,"injection_points",()))
        if not points or not bool(getattr(parent,"raster_support_gating",False)): raise ValueError("v8 requires support-gated frozen parent")
        for point,wrapper in wrappers.items():
            if _self_attention_owner(backbone.blocks[point]).self_attn is not wrapper: raise ValueError("v8 wrapper is not installed")
        self.geometry_wrappers=nn.ModuleDict({str(i):w for i,w in wrappers.items()}); self.backbone=backbone; self.parent_adapter=parent_adapter
        self.selected_blocks=V8_BLOCKS; self.parent_injection_points=points
        self.backbone.requires_grad_(False); self.parent_adapter.requires_grad_(False)
        for wrapper in self.geometry_wrappers.values(): wrapper.enable_direct_training()

def v8_trainable_parameter_names(model: nn.Module) -> set[str]:
    suffixes=("base.q.weight","base.q.bias","base.k.weight","base.k.bias","base.v.weight","base.v.bias","base.o.weight","base.o.bias","channel_gate")
    actual={n for n,p in model.named_parameters() if p.requires_grad}
    allowed={n for n in actual if any(n.endswith(s) for s in suffixes) and any(f".{i}." in n or f"blocks.{i}." in n for i in V8_BLOCKS)}
    if actual!=allowed or not actual: raise ValueError("v8 trainable parameters must be only Q/K/V/O and gates in blocks 8-13")
    return actual
