import pytest
torch=pytest.importorskip("torch")
from torch import nn
from worldarena_baseline.wan_v8_model import V8_BLOCKS,install_v8_action_band

class A(nn.Module):
 def __init__(self): super().__init__(); self.q=nn.Linear(8,8);self.k=nn.Linear(8,8);self.v=nn.Linear(8,8);self.o=nn.Linear(8,8);self.q_norm=nn.Identity();self.k_norm=nn.Identity()
class B(nn.Module):
 def __init__(self): super().__init__();self.self_attn=A()
class M(nn.Module):
 def __init__(self): super().__init__();self.blocks=nn.ModuleList(B() for _ in range(30))
def test_v8_installs_exact_six_blocks():
 m=M(); w=install_v8_action_band(m,V8_BLOCKS,lambda x,g,f:x,lambda q,k,v,s:v)
 assert tuple(w)==V8_BLOCKS
 with pytest.raises(ValueError): install_v8_action_band(m,(8,16,24),lambda x,g,f:x,lambda q,k,v,s:v)
