from __future__ import annotations

import importlib.util
from pathlib import Path


def test_v10_model_module_exists() -> None:
    assert importlib.util.find_spec("worldarena_baseline.wan_v10_model") is not None


def test_v10_model_declares_exact_block_band_and_hidden_heads() -> None:
    text = Path("src/worldarena_baseline/wan_v10_model.py").read_text(encoding="utf-8")
    assert "V10_BLOCKS = tuple(range(6, 18))" in text
    assert "HIDDEN_EEF_BLOCKS = (11, 17)" in text
    assert "relation_gates_enabled" in text
