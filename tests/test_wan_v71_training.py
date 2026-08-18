from __future__ import annotations

import copy

import pytest


torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from worldarena_baseline.wan_v71_training import (  # noqa: E402
    V71_LR,
    build_v71_checkpoint,
    validate_v71_checkpoint,
    v71_optimizer_group,
)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.geometry_wrappers = nn.ModuleDict()
        for block in (8, 16, 24):
            wrapper = nn.Module()
            wrapper.channel_gate = nn.Parameter(torch.zeros(3072))
            for family in ("q_lora", "k_lora", "v_lora", "o_lora"):
                delta = nn.Module()
                delta.down = nn.Parameter(torch.zeros(16, 3072))
                delta.up = nn.Parameter(torch.zeros(3072, 16))
                setattr(wrapper, family, delta)
            self.geometry_wrappers[str(block)] = wrapper


def _expected() -> dict[str, object]:
    return {
        "parent_sha256": "a" * 64,
        "source_hashes": {
            "source_manifest_sha256": "b" * 64,
            "source_code_sha256": "c" * 64,
        },
        "cache_sha256": "d" * 64,
        "replay_sha256": "e" * 64,
        "preflight_sha256": "f" * 64,
    }


def test_v71_checkpoint_roundtrip_binds_exact_geometry_lora_state() -> None:
    model = _Model()
    optimizer = torch.optim.AdamW([v71_optimizer_group(model)], weight_decay=0.0)
    model.geometry_wrappers["8"].channel_gate.grad = torch.ones(3072)
    optimizer.step()
    payload = build_v71_checkpoint(
        step=10, model=model, optimizer=optimizer, **_expected()
    )

    validate_v71_checkpoint(payload, expected=_expected())
    assert payload["config"] == {
        "blocks": [8, 16, 24],
        "rank": 16,
        "lr": V71_LR,
        "loss": "weighted-fm-only",
    }
    assert len(payload["model"]) == 27


def test_v71_checkpoint_rejects_lineage_or_parameter_tampering() -> None:
    model = _Model()
    optimizer = torch.optim.AdamW([v71_optimizer_group(model)], weight_decay=0.0)
    model.geometry_wrappers["8"].channel_gate.grad = torch.ones(3072)
    optimizer.step()
    payload = build_v71_checkpoint(
        step=25, model=model, optimizer=optimizer, **_expected()
    )

    altered = copy.deepcopy(payload)
    altered["source_hashes"]["source_code_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="lineage"):
        validate_v71_checkpoint(altered, expected=_expected())

    altered = copy.deepcopy(payload)
    altered["model"].pop(next(iter(altered["model"])))
    with pytest.raises(ValueError, match="parameter names"):
        validate_v71_checkpoint(altered, expected=_expected())
