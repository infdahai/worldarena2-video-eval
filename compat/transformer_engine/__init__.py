"""Small inference-only Transformer Engine compatibility layer for OSCAR.

OSCAR's selected 2B config uses the pure-PyTorch ``minimal_a2a`` attention
backend.  It still imports Transformer Engine for RMSNorm and RoPE helpers,
so this module provides those two operations without requiring a local nvcc
toolchain.  It is intentionally not a training or FP8 implementation.
"""

from . import pytorch

__version__ = "2.11.0+compat"
__all__ = ["pytorch"]
