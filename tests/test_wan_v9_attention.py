from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from worldarena_baseline.wan_v9_attention import (  # noqa: E402
    PhaseLockedActionCrossAttention,
)


def _module() -> PhaseLockedActionCrossAttention:
    module = PhaseLockedActionCrossAttention(
        visual_width=8,
        action_width=8,
        inner_width=8,
        num_heads=8,
        expected_latent_times=3,
    ).double()
    with torch.no_grad():
        for projection in (module.q, module.k, module.v, module.o):
            projection.weight.copy_(torch.eye(8, dtype=torch.float64))
            if projection.bias is not None:
                projection.bias.zero_()
        module.channel_gate.fill_(1.0)
    return module


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    visual = torch.arange(1, 25, dtype=torch.float64).reshape(1, 3, 8) / 10
    actions = torch.arange(1, 1 + 1 * 2 * 2 * 4 * 8, dtype=torch.float64).reshape(1, 2, 2, 4, 8) / 100
    present = torch.ones(1, 2, 2, dtype=torch.bool)
    grid = torch.tensor([[3, 1, 1]], dtype=torch.long)
    return visual, actions, present, grid


def test_latent_zero_is_exact_zero_even_with_output_bias() -> None:
    module = _module()
    with torch.no_grad():
        module.o.bias.fill_(7.0)
    visual, actions, present, grid = _inputs()
    output = module(visual, grid_sizes=grid, action_tokens=actions, arm_present=present)
    assert torch.count_nonzero(output[:, 0]).item() == 0
    assert torch.count_nonzero(output[:, 1:]).item() > 0


def test_destination_time_reads_only_preceding_transition() -> None:
    module = _module()
    visual, actions, present, grid = _inputs()
    baseline = module(visual, grid_sizes=grid, action_tokens=actions, arm_present=present)
    changed = actions.clone()
    changed[:, 0] += 10_000
    actual = module(visual, grid_sizes=grid, action_tokens=changed, arm_present=present)
    assert not torch.equal(actual[:, 1], baseline[:, 1])
    assert torch.equal(actual[:, 2], baseline[:, 2])


def test_head_groups_cannot_read_the_other_arm_bank() -> None:
    module = _module()
    visual, actions, present, grid = _inputs()
    baseline = module(visual, grid_sizes=grid, action_tokens=actions, arm_present=present)
    changed = actions.clone()
    changed[:, :, 1] += 10_000
    actual = module(visual, grid_sizes=grid, action_tokens=changed, arm_present=present)
    assert torch.equal(actual[..., :4], baseline[..., :4])
    assert not torch.equal(actual[..., 4:], baseline[..., 4:])


def test_absent_arm_output_is_exact_zero_for_its_head_group() -> None:
    module = _module()
    visual, actions, present, grid = _inputs()
    present[:, :, 1] = False
    output = module(visual, grid_sizes=grid, action_tokens=actions, arm_present=present)
    assert torch.count_nonzero(output[..., 4:]).item() == 0
    assert torch.count_nonzero(output[:, 1:, :4]).item() > 0


def test_nonzero_gate_produces_query_key_value_output_gradients() -> None:
    torch.manual_seed(7)
    module = PhaseLockedActionCrossAttention(
        visual_width=16, action_width=12, inner_width=16, num_heads=8,
        expected_latent_times=3,
    ).double()
    with torch.no_grad():
        module.channel_gate.fill_(0.5)
    visual = torch.randn(2, 6, 16, dtype=torch.float64, requires_grad=True)
    actions = torch.randn(2, 2, 2, 4, 12, dtype=torch.float64, requires_grad=True)
    present = torch.ones(2, 2, 2, dtype=torch.bool)
    grid = torch.tensor([[3, 1, 2], [3, 1, 2]], dtype=torch.long)
    module(visual, grid_sizes=grid, action_tokens=actions, arm_present=present).square().mean().backward()
    for name in ("q", "k", "v", "o"):
        gradient = getattr(module, name).weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient).item() > 0


def test_zero_gate_is_exact_zero_and_grid_drift_fails_closed() -> None:
    module = _module()
    visual, actions, present, grid = _inputs()
    with torch.no_grad():
        module.channel_gate.zero_()
    assert torch.count_nonzero(module(visual, grid_sizes=grid, action_tokens=actions, arm_present=present)).item() == 0
    with pytest.raises(ValueError, match="latent time"):
        module(visual[:, :2], grid_sizes=torch.tensor([[2, 1, 1]]), action_tokens=actions, arm_present=present)

