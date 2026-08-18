import pytest
torch = pytest.importorskip("torch")
from torch import nn
from worldarena_baseline.wan_v8_training import build_v8_optimizer, set_v8_learning_rates, step100_gate

class Tiny(nn.Module):
    def __init__(self):
        super().__init__(); self.geometry_wrappers=nn.ModuleDict()

def test_step100_gate_requires_all_three_negatives():
    m={name:{"wins":11,"mean_margin":.1} for name in ("reverse","shift","swap")}
    m.update(routing_retention=.9,fm_regression=.01,position_regression=.01,velocity_regression=.01)
    assert step100_gate(m)["pass"]
    m["swap"]["wins"]=10
    assert not step100_gate(m)["pass"]
