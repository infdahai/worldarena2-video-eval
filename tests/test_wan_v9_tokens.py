from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from worldarena_baseline.wan_v9_tokens import V9ActionTokenizer  # noqa: E402


def _statistics() -> dict[str, object]:
    return {
        "contract": "wan-v9-transition-normalization/1",
        "receipt_sha256": "a" * 64,
        "translation": {"mean": [0.0] * 4, "std": [2.0] * 4},
        "rotation": {"mean": [0.0] * 4, "std": [3.0] * 4},
        "image": {"mean": [0.0] * 6, "std": [4.0] * 6},
        "gripper": {"mean": [0.0] * 3, "std": [5.0] * 3},
    }


def _features(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "translation": torch.randn(batch, 2, 20, 4),
        "rotation": torch.randn(batch, 2, 20, 4),
        "image": torch.randn(batch, 2, 20, 6),
        "gripper": torch.randn(batch, 2, 20, 3),
    }


def test_exactly_four_tokens_per_arm_interval() -> None:
    tokenizer = V9ActionTokenizer(_statistics(), action_width=256, hidden_width=32)
    present = torch.ones(2, 2, 20, dtype=torch.bool)
    tokens, actual_present = tokenizer(_features(), present)
    assert tokens.shape == (2, 20, 2, 4, 256)
    assert actual_present.shape == (2, 20, 2)
    assert torch.equal(actual_present, present.permute(0, 2, 1))


def test_absent_arm_tokens_are_bitwise_zero_after_type_embedding() -> None:
    tokenizer = V9ActionTokenizer(_statistics(), action_width=16, hidden_width=8)
    present = torch.ones(1, 2, 20, dtype=torch.bool)
    present[:, 1, 3:8] = False
    tokens, _ = tokenizer(_features(batch=1), present)
    assert torch.count_nonzero(tokens[:, 3:8, 1]).item() == 0
    assert torch.count_nonzero(tokens[:, 3:8, 0]).item() > 0


def test_tokenizer_has_no_phase_arm_or_fifth_token_parameters() -> None:
    tokenizer = V9ActionTokenizer(_statistics(), action_width=16, hidden_width=8)
    names = set(dict(tokenizer.named_parameters()))
    assert tokenizer.type_embedding.shape == (4, 16)
    assert not any("phase" in name or "arm_embedding" in name for name in names)
    assert {name.split(".")[0] for name in names if name.endswith("0.weight")} == {
        "translation_mlp", "rotation_mlp", "image_mlp", "gripper_mlp",
    }


def test_all_four_modalities_and_type_embeddings_receive_gradients() -> None:
    tokenizer = V9ActionTokenizer(_statistics(), action_width=16, hidden_width=8)
    present = torch.ones(2, 2, 20, dtype=torch.bool)
    tokens, _ = tokenizer(_features(), present)
    tokens.square().mean().backward()
    for name, parameter in tokenizer.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad).item() > 0, name


def test_invalid_statistics_and_shapes_fail_closed() -> None:
    statistics = _statistics()
    statistics["image"] = {"mean": [0.0] * 5, "std": [1.0] * 5}
    with pytest.raises(ValueError, match="image statistics"):
        V9ActionTokenizer(statistics)
    tokenizer = V9ActionTokenizer(_statistics(), action_width=16, hidden_width=8)
    bad = _features(batch=1)
    bad["translation"] = bad["translation"][:, :, :19]
    with pytest.raises(ValueError, match="translation"):
        tokenizer(bad, torch.ones(1, 2, 20, dtype=torch.bool))
