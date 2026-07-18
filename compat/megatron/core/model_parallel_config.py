"""Single-rank configuration accepted by OSCAR's config factory."""

from __future__ import annotations

from dataclasses import dataclass


_UNSUPPORTED_MESSAGE = (
    "The local megatron compatibility layer supports single-rank inference only; "
    "install the full megatron-core package for model parallelism or training."
)


@dataclass
class ModelParallelConfig:
    """Minimal Megatron model-parallel config for OSCAR inference construction."""

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int | None = None
    virtual_pipeline_model_parallel_size: int | None = None
    sequence_parallel: bool = False

    def __post_init__(self) -> None:
        sizes = (
            self.tensor_model_parallel_size,
            self.pipeline_model_parallel_size,
            self.context_parallel_size,
            self.expert_model_parallel_size,
        )
        optional_sizes = (
            self.expert_tensor_parallel_size,
            self.virtual_pipeline_model_parallel_size,
        )
        if (
            any(size != 1 for size in sizes)
            or any(size not in (None, 1) for size in optional_sizes)
            or self.sequence_parallel
        ):
            raise NotImplementedError(_UNSUPPORTED_MESSAGE)
