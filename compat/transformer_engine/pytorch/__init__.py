from __future__ import annotations

import torch
from torch import nn

from . import attention


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        *,
        device=None,
        dtype=None,
        **_: object,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(
            value.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(value.dtype) * self.weight


__all__ = ["RMSNorm", "attention"]
