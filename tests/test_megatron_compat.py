from __future__ import annotations

import importlib
import sys
from dataclasses import is_dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def compat_modules(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(ROOT / "compat"))
    for module_name in tuple(sys.modules):
        if module_name == "megatron" or module_name.startswith("megatron."):
            monkeypatch.delitem(sys.modules, module_name)

    core = importlib.import_module("megatron.core")
    parallel_state = importlib.import_module("megatron.core.parallel_state")
    return core, parallel_state


class FakeDistributed:
    def __init__(self, *, initialized: bool, world_size: int = 1, rank: int = 0):
        self._initialized = initialized
        self._world_size = world_size
        self._rank = rank
        self.default_group = object()
        self.group = SimpleNamespace(WORLD=self.default_group)

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return self._initialized

    def get_world_size(self) -> int:
        return self._world_size

    def get_rank(self) -> int:
        return self._rank


def test_model_parallel_config_constructs_single_rank_defaults(compat_modules) -> None:
    core, _ = compat_modules

    config = core.ModelParallelConfig()

    assert is_dataclass(config)
    assert config.tensor_model_parallel_size == 1
    assert config.pipeline_model_parallel_size == 1
    assert config.context_parallel_size == 1
    assert config.expert_model_parallel_size == 1


def test_model_parallel_config_rejects_parallel_execution(compat_modules) -> None:
    core, _ = compat_modules

    with pytest.raises(NotImplementedError, match="single-rank inference"):
        core.ModelParallelConfig(tensor_model_parallel_size=2)


def test_parallel_state_exposes_single_rank_api_and_default_group(
    compat_modules, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, parallel_state = compat_modules
    distributed = FakeDistributed(initialized=True)
    monkeypatch.setattr(
        parallel_state, "_load_torch_distributed", lambda: distributed
    )

    assert parallel_state.is_initialized() is True

    dimensions = (
        "data_parallel",
        "context_parallel",
        "tensor_model_parallel",
        "pipeline_model_parallel",
        "expert_model_parallel",
    )
    for dimension in dimensions:
        assert getattr(parallel_state, f"get_{dimension}_rank")() == 0
        assert getattr(parallel_state, f"get_{dimension}_world_size")() == 1
        assert (
            getattr(parallel_state, f"get_{dimension}_group")()
            is distributed.default_group
        )

    assert parallel_state.get_model_parallel_group() is distributed.default_group

    parallel_state.cp_size_t = 7
    parallel_state.destroy_model_parallel()
    assert parallel_state.cp_size_t == 1


def test_parallel_state_is_not_initialized_without_default_group(
    compat_modules, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, parallel_state = compat_modules
    distributed = FakeDistributed(initialized=False)
    monkeypatch.setattr(
        parallel_state, "_load_torch_distributed", lambda: distributed
    )

    assert parallel_state.is_initialized() is False
    assert parallel_state.get_data_parallel_rank() == 0
    assert parallel_state.get_data_parallel_world_size() == 1
    with pytest.raises(RuntimeError, match="torch.distributed default process group"):
        parallel_state.get_data_parallel_group()


def test_parallel_state_rejects_multi_rank_execution(
    compat_modules, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, parallel_state = compat_modules
    distributed = FakeDistributed(initialized=True, world_size=2)
    monkeypatch.setattr(
        parallel_state, "_load_torch_distributed", lambda: distributed
    )

    assert parallel_state.is_initialized() is False
    with pytest.raises(NotImplementedError, match="single-rank inference"):
        parallel_state.get_tensor_model_parallel_world_size()
    with pytest.raises(NotImplementedError, match="single-rank inference"):
        parallel_state.get_model_parallel_group()
