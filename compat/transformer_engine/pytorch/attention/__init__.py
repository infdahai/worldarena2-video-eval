from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .rope import apply_rotary_pos_emb


class DotProductAttention(nn.Module):
    """Inference-compatible fallback for the TE attention API."""

    def __init__(
        self,
        num_attention_heads: int,
        kv_channels: int,
        *,
        attention_dropout: float = 0.0,
        qkv_format: str = "bshd",
        attn_mask_type: str = "no_mask",
        **_: object,
    ) -> None:
        super().__init__()
        self.dropout = attention_dropout
        self.qkv_format = qkv_format
        self.attn_mask_type = attn_mask_type

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        if self.qkv_format == "sbhd":
            query, key, value = (item.transpose(0, 1) for item in (query, key, value))
        query, key, value = (item.transpose(1, 2) for item in (query, key, value))
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.attn_mask_type == "causal",
        ).transpose(1, 2)
        if self.qkv_format == "sbhd":
            output = output.transpose(0, 1)
        return output.contiguous()


__all__ = ["DotProductAttention", "apply_rotary_pos_emb"]
