from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from worldarena_baseline.wan_se3_attention import (
    ArmGroupedSE3Geometry,
    SE3AugmentedSelfAttention,
    apply_group_action,
)


def _rotation_z(angle: float, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    matrix = torch.eye(4, dtype=dtype)
    matrix[0, 0] = math.cos(angle)
    matrix[0, 1] = -math.sin(angle)
    matrix[1, 0] = math.sin(angle)
    matrix[1, 1] = math.cos(angle)
    matrix[0, 3] = 0.25
    return matrix


def _transforms(
    *,
    batch: int = 1,
    frames: int = 21,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    result = torch.eye(4, dtype=dtype).reshape(1, 1, 1, 4, 4).repeat(batch, 2, frames, 1, 1)
    result[:, 0] = _rotation_z(0.25, dtype=dtype)
    result[:, 1] = _rotation_z(-0.4, dtype=dtype)
    return result


def _materialized_reference(value: torch.Tensor, matrix: torch.Tensor, *, transpose: bool = False) -> torch.Tensor:
    action = matrix.transpose(-1, -2) if transpose else matrix
    copies = value.shape[-1] // 4
    kron = torch.kron(torch.eye(copies, dtype=value.dtype), action.reshape(-1, 4, 4)[0])
    if action.numel() != 16:
        raise AssertionError("this small reference deliberately supports one action")
    return torch.einsum("ij,...j->...i", kron, value)


def _mean_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    del q, k, seq_lens
    return v + 0.25


def _qkv(
    *,
    batch: int = 1,
    tokens: int = 21 * 15 * 20,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.arange(batch * tokens * 24 * 128, dtype=dtype).reshape(batch, tokens, 24, 128)
    return values + 1, values + 2, values + 3


def _geometry() -> ArmGroupedSE3Geometry:
    return ArmGroupedSE3Geometry(attention_fn=_mean_attention)


def test_broadcast_transform_matches_materialized_kron() -> None:
    value = torch.randn(2, 5, 4, 8, dtype=torch.float64)
    matrix = _rotation_z(0.37).reshape(1, 1, 1, 4, 4)
    actual = apply_group_action(value, matrix)
    expected = _materialized_reference(value, matrix)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_transpose_orientation_matches_materialized_kron() -> None:
    value = torch.randn(2, 5, 4, 8, dtype=torch.float64)
    matrix = _rotation_z(0.37).reshape(1, 1, 1, 4, 4)
    actual = apply_group_action(value, matrix, transpose=True)
    expected = _materialized_reference(value, matrix, transpose=True)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_action_and_inverse_restore_the_original_value() -> None:
    value = torch.randn(2, 5, 4, 8, dtype=torch.float64)
    matrix = _rotation_z(0.37).reshape(1, 1, 1, 4, 4)
    restored = apply_group_action(apply_group_action(value, matrix), torch.linalg.inv(matrix))
    torch.testing.assert_close(restored, value, atol=1e-10, rtol=1e-10)


def test_apply_group_action_rejects_wrong_head_width_and_non_finite_matrix() -> None:
    with pytest.raises(ValueError, match="divisible by four"):
        apply_group_action(torch.zeros(1, 3, 7), torch.eye(4))
    invalid = torch.eye(4)
    invalid[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        apply_group_action(torch.zeros(1, 3, 8), invalid)


def test_absent_arm_is_exact_zero_before_output_projection() -> None:
    q, k, v = _qkv(tokens=21 * 15 * 20)
    result = _geometry()(
        q,
        k,
        v,
        grid_sizes=torch.tensor([[21, 15, 20]]),
        arm_transform=_transforms(),
        arm_present=torch.tensor([[[False] * 21, [True] * 21]]),
        seq_lens=torch.tensor([21 * 15 * 20]),
    )
    assert torch.count_nonzero(result[:, :, :12]) == 0
    assert torch.count_nonzero(result[:, :, 12:]) > 0


def test_time_major_21x15x20_mapping_and_text_padding() -> None:
    visual = 21 * 15 * 20
    q = torch.zeros(1, visual + 4, 24, 128)
    k = torch.zeros_like(q)
    v = torch.ones_like(q)
    transforms = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, 2, 21, 1, 1)
    transforms[0, 0, 1, 0, 3] = 7.0
    result = _geometry()(
        q,
        k,
        v,
        grid_sizes=torch.tensor([[21, 15, 20]]),
        arm_transform=transforms,
        arm_present=torch.ones(1, 2, 21, dtype=torch.bool),
        seq_lens=torch.tensor([visual]),
    )
    first_frame_token = 0
    second_frame_token = 15 * 20
    assert torch.equal(result[0, first_frame_token, :12], torch.ones_like(result[0, first_frame_token, :12]) * 1.25)
    assert not torch.equal(result[0, first_frame_token, :12], result[0, second_frame_token, :12])
    assert torch.count_nonzero(result[0, visual:]) == 0


def test_swapping_arm_assignment_changes_only_fixed_head_group_inputs() -> None:
    q, k, v = _qkv(tokens=21 * 15 * 20)
    common = dict(
        grid_sizes=torch.tensor([[21, 15, 20]]),
        arm_present=torch.ones(1, 2, 21, dtype=torch.bool),
        seq_lens=torch.tensor([21 * 15 * 20]),
    )
    normal = _geometry()(q, k, v, arm_transform=_transforms(), **common)
    swapped = _geometry()(q, k, v, arm_transform=_transforms()[:, [1, 0]], **common)
    assert not torch.equal(normal[:, :, :12], swapped[:, :, :12])
    assert not torch.equal(normal[:, :, 12:], swapped[:, :, 12:])


class _FakeWanAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(8, 8, bias=False)
        self.k = nn.Linear(8, 8, bias=False)
        self.v = nn.Linear(8, 8, bias=False)
        self.o = nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            for layer in (self.q, self.k, self.v, self.o):
                layer.weight.copy_(torch.eye(8))

    def q_norm(self, value: torch.Tensor) -> torch.Tensor:
        return value + 10

    def k_norm(self, value: torch.Tensor) -> torch.Tensor:
        return value + 20

    def forward(self, x: torch.Tensor, seq_lens: torch.Tensor, grid_sizes: torch.Tensor, freqs: object) -> torch.Tensor:
        q = self.q_norm(self.q(x).reshape(x.shape[0], x.shape[1], 2, 4))
        k = self.k_norm(self.k(x).reshape(x.shape[0], x.shape[1], 2, 4))
        v = self.v(x).reshape(x.shape[0], x.shape[1], 2, 4)
        q, k = _fake_rope(q, k, freqs)
        return self.o(_mean_attention(q, k, v, seq_lens).flatten(2))


def _fake_rope(q: torch.Tensor, k: torch.Tensor, freqs: object) -> tuple[torch.Tensor, torch.Tensor]:
    assert freqs == "rope"
    return q + 100, k + 200


def _in_place_fake_rope(q: torch.Tensor, k: torch.Tensor, freqs: object) -> tuple[torch.Tensor, torch.Tensor]:
    assert freqs == "rope"
    q.add_(100)
    k.add_(200)
    return q, k


def test_wrapper_forks_normalized_qkv_before_rope_and_zero_gate_is_bitwise_equal() -> None:
    base = _FakeWanAttention()
    observed: dict[str, torch.Tensor] = {}

    def geometry_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        observed["q"] = q.detach().clone()
        observed["k"] = k.detach().clone()
        observed["v"] = v.detach().clone()
        return _mean_attention(q, k, v, seq_lens)

    wrapper = SE3AugmentedSelfAttention(
        base,
        attention_fn=geometry_attention,
        rope_apply_fn=_fake_rope,
        num_heads=2,
        head_dim=4,
    )
    x = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    kwargs = dict(
        seq_lens=torch.tensor([3]),
        grid_sizes=torch.tensor([[1, 1, 3]]),
        freqs="rope",
        arm_transform=torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, 2, 1, 1, 1),
        arm_present=torch.ones(1, 2, 1, dtype=torch.bool),
    )
    expected = base(x, kwargs["seq_lens"], kwargs["grid_sizes"], kwargs["freqs"])
    actual = wrapper(x, **kwargs)
    assert torch.equal(actual, expected)
    torch.testing.assert_close(observed["q"], (x + 10).reshape(1, 3, 2, 4))
    torch.testing.assert_close(observed["k"], (x + 20).reshape(1, 3, 2, 4))
    torch.testing.assert_close(observed["v"], x.reshape(1, 3, 2, 4))


def test_wrapper_only_gate_receives_gradients() -> None:
    base = _FakeWanAttention()
    wrapper = SE3AugmentedSelfAttention(
        base,
        attention_fn=_mean_attention,
        rope_apply_fn=_fake_rope,
        num_heads=2,
        head_dim=4,
    )
    x = torch.randn(1, 3, 8)
    output = wrapper(
        x,
        seq_lens=torch.tensor([3]),
        grid_sizes=torch.tensor([[1, 1, 3]]),
        freqs="rope",
        arm_transform=torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, 2, 1, 1, 1),
        arm_present=torch.ones(1, 2, 1, dtype=torch.bool),
    )
    output.sum().backward()
    assert wrapper.gate.grad is not None
    assert torch.count_nonzero(wrapper.gate.grad) > 0
    assert all(parameter.grad is None for parameter in base.parameters())


def test_wrapper_keeps_geometry_qkv_pre_rope_when_rope_mutates_in_place() -> None:
    base = _FakeWanAttention()
    observed: dict[str, torch.Tensor] = {}

    def geometry_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        observed["q"] = q.detach().clone()
        observed["k"] = k.detach().clone()
        observed["v"] = v.detach().clone()
        return _mean_attention(q, k, v, seq_lens)

    wrapper = SE3AugmentedSelfAttention(
        base,
        attention_fn=geometry_attention,
        rope_apply_fn=_in_place_fake_rope,
        num_heads=2,
        head_dim=4,
    )
    x = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    wrapper(
        x,
        seq_lens=torch.tensor([3]),
        grid_sizes=torch.tensor([[1, 1, 3]]),
        freqs="rope",
        arm_transform=torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, 2, 1, 1, 1),
        arm_present=torch.ones(1, 2, 1, dtype=torch.bool),
    )
    torch.testing.assert_close(observed["q"], (x + 10).reshape(1, 3, 2, 4))
    torch.testing.assert_close(observed["k"], (x + 20).reshape(1, 3, 2, 4))
    torch.testing.assert_close(observed["v"], x.reshape(1, 3, 2, 4))


def test_geometry_uses_cached_inverse_without_recomputing_per_block(monkeypatch: pytest.MonkeyPatch) -> None:
    q, k, v = _qkv(tokens=21 * 15 * 20)
    transform = _transforms()
    cached_inverse = torch.linalg.inv(transform)

    def unexpected_inverse(_: torch.Tensor) -> torch.Tensor:
        raise AssertionError("geometry must use the supplied per-condition inverse")

    monkeypatch.setattr(torch.linalg, "inv", unexpected_inverse)
    result = _geometry()(
        q,
        k,
        v,
        grid_sizes=torch.tensor([[21, 15, 20]]),
        arm_transform=transform,
        arm_inverse=cached_inverse,
        arm_present=torch.ones(1, 2, 21, dtype=torch.bool),
        seq_lens=torch.tensor([21 * 15 * 20]),
    )
    assert torch.count_nonzero(result) > 0


def test_geometry_rejects_cached_inverse_from_a_different_condition() -> None:
    q, k, v = _qkv(tokens=21 * 15 * 20)
    transform = _transforms()
    other_condition = transform.clone()
    other_condition[0, 0, 0, 0, 3] += 1.0
    mismatched_inverse = torch.linalg.inv(other_condition)
    with pytest.raises(ValueError, match="inverse.*match"):
        _geometry()(
            q,
            k,
            v,
            grid_sizes=torch.tensor([[21, 15, 20]]),
            arm_transform=transform,
            arm_inverse=mismatched_inverse,
            arm_present=torch.ones(1, 2, 21, dtype=torch.bool),
            seq_lens=torch.tensor([21 * 15 * 20]),
        )


def test_geometry_accepts_valid_cached_inverse_with_large_translation() -> None:
    q, k, v = _qkv(tokens=1)
    transform = _transforms(frames=1)
    transform[:, :, :, 0, 3] = 1_000_000.0
    transform[:, :, :, 1, 3] = -1_000_000.0
    cached_inverse = torch.linalg.inv(transform)
    result = _geometry()(
        q,
        k,
        v,
        grid_sizes=torch.tensor([[1, 1, 1]]),
        arm_transform=transform,
        arm_inverse=cached_inverse,
        arm_present=torch.ones(1, 2, 1, dtype=torch.bool),
        seq_lens=torch.tensor([1]),
    )
    assert torch.count_nonzero(result) > 0
