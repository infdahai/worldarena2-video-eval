"""Small subset of :mod:`megatron.core` required by OSCAR inference."""

from . import parallel_state
from .model_parallel_config import ModelParallelConfig

__all__ = ["ModelParallelConfig", "parallel_state"]
