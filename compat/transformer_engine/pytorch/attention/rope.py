from __future__ import annotations

import torch


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(
    value: torch.Tensor,
    frequencies: torch.Tensor,
    *,
    tensor_format: str = "sbhd",
    **_: object,
) -> torch.Tensor:
    """Apply non-interleaved RoPE for TE's ``bshd``/``sbhd`` layouts."""

    if tensor_format not in {"bshd", "sbhd"}:
        raise ValueError(f"unsupported tensor format: {tensor_format}")
    angles = frequencies
    while angles.ndim > 2 and angles.shape[1] == 1:
        angles = angles.squeeze(1)
    if angles.ndim == 1:
        angles = angles.unsqueeze(0)
    angles = angles[..., : value.shape[-1]]
    if tensor_format == "bshd":
        angles = angles.unsqueeze(0).unsqueeze(2)
    else:
        angles = angles.unsqueeze(1).unsqueeze(2)
    cosine = angles.cos().to(dtype=value.dtype, device=value.device)
    sine = angles.sin().to(dtype=value.dtype, device=value.device)
    return value * cosine + _rotate_half(value) * sine
