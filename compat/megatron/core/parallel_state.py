"""Inference-only, single-rank subset of Megatron parallel state.

OSCAR imports these accessors even when model parallelism is disabled.  Rank and
world-size queries expose the single-rank identity.  Group queries return
PyTorch's initialized default process group; this module never creates or owns
process groups.
"""

from __future__ import annotations

from typing import Any


cp_size_t = 1

_UNSUPPORTED_MESSAGE = (
    "The local megatron compatibility layer supports single-rank inference only; "
    "install the full megatron-core package for model parallelism or training."
)
_GROUP_REQUIRED_MESSAGE = (
    "An initialized torch.distributed default process group is required when "
    "requesting a Megatron compatibility group."
)


def _load_torch_distributed() -> Any | None:
    try:
        import torch.distributed as distributed
    except (ImportError, ModuleNotFoundError):
        return None
    return distributed


def _validated_distributed() -> Any | None:
    distributed = _load_torch_distributed()
    if (
        distributed is None
        or not distributed.is_available()
        or not distributed.is_initialized()
    ):
        return None
    if distributed.get_world_size() != 1 or distributed.get_rank() != 0:
        raise NotImplementedError(_UNSUPPORTED_MESSAGE)
    return distributed


def _single_rank() -> int:
    _validated_distributed()
    return 0


def _single_world_size() -> int:
    _validated_distributed()
    return 1


def _default_group() -> Any:
    distributed = _validated_distributed()
    if distributed is None:
        raise RuntimeError(_GROUP_REQUIRED_MESSAGE)
    return distributed.group.WORLD


def is_initialized() -> bool:
    try:
        return _validated_distributed() is not None
    except NotImplementedError:
        return False


def get_data_parallel_rank(*_args: Any, **_kwargs: Any) -> int:
    return _single_rank()


def get_data_parallel_world_size(*_args: Any, **_kwargs: Any) -> int:
    return _single_world_size()


def get_data_parallel_group(*_args: Any, **_kwargs: Any) -> Any:
    return _default_group()


def get_context_parallel_rank(*_args: Any, **_kwargs: Any) -> int:
    return _single_rank()


def get_context_parallel_world_size(*_args: Any, **_kwargs: Any) -> int:
    return _single_world_size()


def get_context_parallel_group(*_args: Any, **_kwargs: Any) -> Any:
    return _default_group()


def get_tensor_model_parallel_rank(*_args: Any, **_kwargs: Any) -> int:
    return _single_rank()


def get_tensor_model_parallel_world_size(*_args: Any, **_kwargs: Any) -> int:
    return _single_world_size()


def get_tensor_model_parallel_group(*_args: Any, **_kwargs: Any) -> Any:
    return _default_group()


def get_pipeline_model_parallel_rank(*_args: Any, **_kwargs: Any) -> int:
    return _single_rank()


def get_pipeline_model_parallel_world_size(*_args: Any, **_kwargs: Any) -> int:
    return _single_world_size()


def get_pipeline_model_parallel_group(*_args: Any, **_kwargs: Any) -> Any:
    return _default_group()


def get_expert_model_parallel_rank(*_args: Any, **_kwargs: Any) -> int:
    return _single_rank()


def get_expert_model_parallel_world_size(*_args: Any, **_kwargs: Any) -> int:
    return _single_world_size()


def get_expert_model_parallel_group(*_args: Any, **_kwargs: Any) -> Any:
    return _default_group()


def get_model_parallel_group(*_args: Any, **_kwargs: Any) -> Any:
    return _default_group()


def destroy_model_parallel() -> None:
    """Reset shim-owned state without destroying PyTorch's default group."""

    global cp_size_t
    _validated_distributed()
    cp_size_t = 1
