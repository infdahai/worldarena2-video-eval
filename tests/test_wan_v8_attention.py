import pytest

torch = pytest.importorskip("torch")
from torch import nn

from worldarena_baseline.wan_v8_attention import DirectActionBandAttention


def _attn(q,k,v,seq):
    del seq
    return v + .01*q + .01*k

def _rope(x,grid,freq):
    del grid,freq
    return x+3

class Base(nn.Module):
    def __init__(self):
        super().__init__(); self.q=nn.Linear(8,8,bias=False); self.k=nn.Linear(8,8,bias=False); self.v=nn.Linear(8,8,bias=False); self.o=nn.Linear(8,8,bias=True); self.q_norm=nn.Identity(); self.k_norm=nn.Identity()
    def forward(self,x,seq,grid,freq): return self.o(_attn(_rope(self.q(x).reshape(1,3,2,4),grid,freq),_rope(self.k(x).reshape(1,3,2,4),grid,freq),self.v(x).reshape(1,3,2,4),seq).flatten(2))

def _kw(): return dict(seq_lens=torch.tensor([3]),grid_sizes=torch.tensor([[1,1,3]]),freqs=None,arm_transform=torch.eye(4).reshape(1,1,1,4,4).repeat(1,2,1,1,1),arm_present=torch.ones(1,2,1,dtype=torch.bool))

def test_zero_gate_matches_native_and_direct_projection_contract():
    base=Base(); wrapper=DirectActionBandAttention(base,attention_fn=_attn,rope_apply_fn=_rope,num_heads=2,head_dim=4); x=torch.randn(1,3,8)
    assert torch.equal(wrapper(x,**_kw()),base(x,_kw()["seq_lens"],_kw()["grid_sizes"],None))
    wrapper.channel_gate.data.fill_(.1); wrapper(x,**_kw()).sum().backward()
    assert all(getattr(base,n).weight.grad is not None for n in ("q","k","v","o"))
    assert wrapper.channel_gate.grad is not None
