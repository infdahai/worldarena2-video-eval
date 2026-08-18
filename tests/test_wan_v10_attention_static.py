from __future__ import annotations

import ast
from pathlib import Path


SOURCE = Path("src/worldarena_baseline/wan_v10_attention.py")
PROBE = Path("scripts/probe_wan_v10_augmented_attention.py")


def test_v10_attention_never_builds_explicit_pairwise_bias() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    assert "attn_bias" not in text
    assert "pairwise_bias" not in text
    assert "scaled_dot_product_attention" not in text
    assert 'einsum("blhd,bshd' not in text


def test_v10_attention_calls_supplied_fused_kernel_with_native_scale() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    attention_calls = [
        node for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "attention_fn"
    ]
    assert len(attention_calls) == 1
    keywords = {keyword.arg for keyword in attention_calls[0].keywords}
    assert {"q", "k", "v", "k_lens", "window_size", "softmax_scale"} <= keywords


def test_feasibility_probe_requires_real_flash_and_production_shape() -> None:
    text = PROBE.read_text(encoding="utf-8")
    assert "25200" in text
    assert "FLASH_ATTN_2_AVAILABLE" in text
    assert "FLASH_ATTN_3_AVAILABLE" in text
    assert "softmax_scale=1.0 / math.sqrt(128)" in text
    assert "quadratic_fallback" in text
    assert "/data/di/worldarena2_track1_20260815" in text
