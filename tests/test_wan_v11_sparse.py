from __future__ import annotations

import pytest
import torch

from worldarena_baseline.wan_v11_sparse import (
    SparseTubeSelection,
    gather_visual_tokens,
    scatter_arm_residual,
    select_tube_tokens,
)


def test_select_tube_tokens_is_stable_and_pads() -> None:
    support = torch.zeros(1, 21, 2, 3, 4)
    support[0, 3, 0, 1, 2] = 1
    support[0, 4, 1, 0, 3] = 0.5
    support[0, 4, 1, 0, 1] = 0.5

    selected = select_tube_tokens(support, max_tokens=48)

    assert selected.indices.shape == (1, 21, 2, 48)
    assert selected.valid.shape == selected.indices.shape
    assert selected.weights.shape == selected.indices.shape
    assert selected.valid[0, 3, 0].sum().item() == 1
    assert selected.indices[0, 3, 0, 0].item() == 6
    assert selected.indices[0, 4, 1, :2].tolist() == [1, 3]
    assert torch.count_nonzero(selected.indices[~selected.valid]).item() == 0


def test_select_tube_tokens_is_deterministic() -> None:
    support = torch.rand(2, 21, 2, 5, 6)

    first = select_tube_tokens(support, max_tokens=7)
    second = select_tube_tokens(support.clone(), max_tokens=7)

    assert torch.equal(first.indices, second.indices)
    assert torch.equal(first.valid, second.valid)
    assert torch.equal(first.weights, second.weights)


def test_gather_visual_tokens_preserves_arm_and_time() -> None:
    height, width = 2, 3
    sequence = 21 * height * width
    visual = torch.arange(sequence, dtype=torch.float32).reshape(1, sequence, 1)
    support = torch.zeros(1, 21, 2, height, width)
    support[0, 5, 0, 0, 2] = 1
    support[0, 5, 1, 1, 1] = 1
    selected = select_tube_tokens(support, max_tokens=2)

    gathered = gather_visual_tokens(visual, selected, grid_sizes=torch.tensor([[21, 2, 3]]))

    assert gathered.shape == (1, 21, 2, 2, 1)
    assert gathered[0, 5, 0, 0, 0].item() == 5 * 6 + 2
    assert gathered[0, 5, 1, 0, 0].item() == 5 * 6 + 4
    assert torch.count_nonzero(gathered[~selected.valid]).item() == 0


def test_gather_rejects_wrong_grid_or_sequence() -> None:
    support = torch.zeros(1, 21, 2, 2, 3)
    selected = select_tube_tokens(support, max_tokens=2)

    with pytest.raises(ValueError, match="grid"):
        gather_visual_tokens(
            torch.zeros(1, 21 * 2 * 3, 4),
            selected,
            grid_sizes=torch.tensor([[20, 2, 3]]),
        )
    with pytest.raises(ValueError, match="sequence"):
        gather_visual_tokens(
            torch.zeros(1, 21 * 2 * 3 - 1, 4),
            selected,
            grid_sizes=torch.tensor([[21, 2, 3]]),
        )


def test_scatter_masks_outside_support_and_padded_tokens() -> None:
    support = torch.zeros(1, 21, 2, 3)
    support[0, 8, 1, 2] = 1
    full = torch.zeros(1, 21, 1, 3, 2)
    full[0, 8, 0, 0] = torch.tensor([2.0, 3.0])
    selection = SparseTubeSelection(
        indices=torch.zeros(1, 21, 1, 3, dtype=torch.long),
        valid=torch.zeros(1, 21, 1, 3, dtype=torch.bool),
        weights=torch.zeros(1, 21, 1, 3),
        spatial_shape=(2, 3),
    )
    selection.indices[0, 8, 0, 0] = 5
    selection.valid[0, 8, 0, 0] = True

    output = scatter_arm_residual(
        full[:, :, 0],
        SparseTubeSelection(
            indices=selection.indices[:, :, 0],
            valid=selection.valid[:, :, 0],
            weights=selection.weights[:, :, 0],
            spatial_shape=selection.spatial_shape,
        ),
        support,
        sequence_length=21 * 6,
    )

    assert output[0, 8 * 6 + 5].tolist() == [2.0, 3.0]
    assert torch.count_nonzero(output).item() == 2


def test_scatter_adds_duplicate_writes_without_cross_time_leakage() -> None:
    support = torch.zeros(1, 21, 2, 2)
    support[0, 7, 0, 1] = 1
    indices = torch.zeros(1, 21, 2, dtype=torch.long)
    valid = torch.zeros(1, 21, 2, dtype=torch.bool)
    weights = torch.zeros(1, 21, 2)
    indices[0, 7] = 1
    valid[0, 7] = True
    weights[0, 7] = 1
    local = torch.zeros(1, 21, 2, 1)
    local[0, 7, :, 0] = torch.tensor([2.0, 5.0])

    output = scatter_arm_residual(
        local,
        SparseTubeSelection(indices, valid, weights, spatial_shape=(2, 2)),
        support,
        sequence_length=21 * 4,
    )

    assert output[0, 7 * 4 + 1, 0].item() == 7.0
    assert torch.count_nonzero(output[:,: 7 * 4]).item() == 0
    assert torch.count_nonzero(output[:, 8 * 4 :]).item() == 0
