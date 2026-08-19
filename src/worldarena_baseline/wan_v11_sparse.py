from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SparseTubeSelection:
    indices: Tensor
    valid: Tensor
    weights: Tensor
    spatial_shape: tuple[int, int]


def select_tube_tokens(support: Tensor, *, max_tokens: int = 48) -> SparseTubeSelection:
    if support.ndim != 5 or support.shape[1] != 21 or support.shape[2] != 2:
        raise ValueError("support must have shape (B, 21, 2, H, W)")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not torch.isfinite(support).all() or torch.any(support < 0):
        raise ValueError("support must be finite and nonnegative")

    batch, time, arms, height, width = support.shape
    spatial = height * width
    flat = support.reshape(batch, time, arms, spatial)
    order = torch.argsort(flat, dim=-1, descending=True, stable=True)
    take = min(max_tokens, spatial)
    chosen = order[..., :take]
    weights = torch.gather(flat, -1, chosen)
    valid = weights > 0
    chosen = torch.where(valid, chosen, torch.zeros_like(chosen))

    if take < max_tokens:
        padding_shape = (*chosen.shape[:-1], max_tokens - take)
        chosen = torch.cat((chosen, chosen.new_zeros(padding_shape)), dim=-1)
        valid = torch.cat((valid, valid.new_zeros(padding_shape)), dim=-1)
        weights = torch.cat((weights, weights.new_zeros(padding_shape)), dim=-1)
    return SparseTubeSelection(
        indices=chosen,
        valid=valid,
        weights=weights,
        spatial_shape=(height, width),
    )


def _validate_grid_sizes(
    grid_sizes: Tensor,
    *,
    batch: int,
    height: int,
    width: int,
) -> None:
    if tuple(grid_sizes.shape) != (batch, 3):
        raise ValueError("grid sizes must have shape (B, 3)")
    expected = torch.tensor(
        [21, height, width],
        device=grid_sizes.device,
        dtype=grid_sizes.dtype,
    ).expand(batch, -1)
    if not torch.equal(grid_sizes, expected):
        raise ValueError("grid sizes must exactly match (21, H, W)")


def gather_visual_tokens(
    visual: Tensor,
    selection: SparseTubeSelection,
    *,
    grid_sizes: Tensor,
) -> Tensor:
    if visual.ndim != 3:
        raise ValueError("visual must have shape (B, L, C)")
    if selection.indices.ndim != 4 or selection.indices.shape[1:3] != (21, 2):
        raise ValueError("selection must have shape (B, 21, 2, K)")
    if selection.valid.shape != selection.indices.shape:
        raise ValueError("selection valid mask shape differs from indices")
    batch, sequence, channels = visual.shape
    if selection.indices.shape[0] != batch:
        raise ValueError("selection batch differs from visual batch")
    height, width = selection.spatial_shape
    spatial = height * width
    _validate_grid_sizes(
        grid_sizes,
        batch=batch,
        height=height,
        width=width,
    )
    if sequence != 21 * spatial:
        raise ValueError("visual sequence length differs from the selection grid")
    if torch.any(selection.indices < 0) or torch.any(selection.indices >= spatial):
        raise ValueError("selection contains an out-of-grid index")

    time_offset = torch.arange(21, device=visual.device).view(1, 21, 1, 1) * spatial
    global_index = selection.indices.to(visual.device) + time_offset
    expanded_visual = visual[:, None, None].expand(batch, 21, 2, sequence, channels)
    gathered = torch.gather(
        expanded_visual,
        3,
        global_index[..., None].expand(*global_index.shape, channels),
    )
    return gathered * selection.valid.to(visual.device)[..., None].to(visual.dtype)


def scatter_arm_residual(
    local: Tensor,
    selection: SparseTubeSelection,
    support: Tensor,
    *,
    sequence_length: int,
) -> Tensor:
    if local.ndim != 4:
        raise ValueError("local residual must have shape (B, 21, K, C)")
    batch, time, count, channels = local.shape
    if time != 21:
        raise ValueError("local residual must contain 21 latent times")
    if tuple(selection.indices.shape) != (batch, time, count):
        raise ValueError("selection shape differs from local residual")
    if selection.valid.shape != selection.indices.shape:
        raise ValueError("selection valid mask shape differs from indices")
    if support.ndim != 4 or tuple(support.shape[:2]) != (batch, time):
        raise ValueError("support must have shape (B, 21, H, W)")
    height, width = selection.spatial_shape
    if tuple(support.shape[-2:]) != (height, width):
        raise ValueError("support grid differs from selection grid")
    spatial = height * width
    if sequence_length != time * spatial:
        raise ValueError("sequence length differs from support grid")
    if torch.any(selection.indices < 0) or torch.any(selection.indices >= spatial):
        raise ValueError("selection contains an out-of-grid index")

    time_offset = torch.arange(time, device=local.device).view(1, time, 1) * spatial
    index = (selection.indices.to(local.device) + time_offset).reshape(batch, -1)
    valid = selection.valid.to(local.device).reshape(batch, -1, 1).to(local.dtype)
    source = local.reshape(batch, -1, channels) * valid
    output = local.new_zeros((batch, sequence_length, channels))
    output.scatter_add_(1, index[..., None].expand_as(source), source)
    flat_support = support.reshape(batch, sequence_length, 1).to(local.dtype)
    return output * flat_support
