"""Shared native-QKVO direct SE(3) attention branch used by v8."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .wan_se3_attention import ArmGroupedSE3Geometry, AttentionCallable, RopeApplyCallable


class DirectActionBandAttention(nn.Module):
    """Native Wan attention plus SE(3) attention sharing the same Q/K/V/O.

    Only the native four projection modules and ``channel_gate`` are trainable.
    With the gate at zero, this is exactly the original attention computation.
    """
    def __init__(self, base: nn.Module, *, attention_fn: AttentionCallable, rope_apply_fn: RopeApplyCallable, num_heads: int = 24, head_dim: int = 128) -> None:
        super().__init__()
        if not all(hasattr(base, name) for name in ("q", "k", "v", "o")):
            raise TypeError("base attention must expose q/k/v/o")
        self.base=base; self.geometry=ArmGroupedSE3Geometry(attention_fn=attention_fn,num_heads=num_heads,head_dim=head_dim)
        self.geometry.requires_grad_(False); self.rope_apply_fn=rope_apply_fn; self.num_heads=num_heads; self.head_dim=head_dim
        self.channel_gate=nn.Parameter(torch.zeros(num_heads*head_dim,dtype=torch.float32))

    def _qn(self): return self.base.q_norm if hasattr(self.base,"q_norm") else self.base.norm_q
    def _kn(self): return self.base.k_norm if hasattr(self.base,"k_norm") else self.base.norm_k

    def forward(self,x:Tensor,seq_lens:Tensor,grid_sizes:Tensor,freqs:object,*,arm_transform:Tensor,arm_present:Tensor,arm_inverse:Tensor|None=None,checkpoint_replay_release:Callable[[],None]|None=None)->Tensor:
        try:
            width=self.num_heads*self.head_dim
            if x.ndim!=3 or x.shape[-1]!=width: raise ValueError("Wan token width differs")
            q_raw,k_raw,v_raw=self.base.q(x),self.base.k(x),self.base.v(x)
            q=self._qn()(q_raw).reshape(*x.shape[:2],self.num_heads,self.head_dim); k=self._kn()(k_raw).reshape_as(q); v=v_raw.reshape_as(q)
            native=self.base.o(self.geometry.attention_fn(self.rope_apply_fn(q.clone(),grid_sizes,freqs),self.rope_apply_fn(k.clone(),grid_sizes,freqs),v,seq_lens).flatten(2))
            geom=self.geometry(q,k,v,grid_sizes=grid_sizes,arm_transform=arm_transform,arm_inverse=arm_inverse,arm_present=arm_present,seq_lens=seq_lens).flatten(2)
            geo_o=F.linear(geom,self.base.o.weight,None)
            return native+(geo_o.float()*self.channel_gate.view(1,1,-1)).to(native.dtype)
        finally:
            if checkpoint_replay_release is not None: checkpoint_replay_release()

    def enable_direct_training(self)->None:
        self.geometry.requires_grad_(False)
        for module in (self.base.q,self.base.k,self.base.v,self.base.o): module.requires_grad_(True)
        self.channel_gate.requires_grad_(True)
