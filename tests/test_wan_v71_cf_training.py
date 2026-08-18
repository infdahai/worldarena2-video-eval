from __future__ import annotations

import copy

import pytest


torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from worldarena_baseline.wan_v71_cf_training import (  # noqa: E402
    V71_CF_STEPS,
    build_v71_cf_checkpoint,
    validate_v71_cf_checkpoint,
)
from worldarena_baseline.wan_v71_training import v71_optimizer_group  # noqa: E402


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


def _lineage() -> dict[str, object]:
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


def test_cf_checkpoint_binds_dynamic_calibration_and_fresh_lineage() -> None:
    model = _Model()
    optimizer = torch.optim.AdamW([v71_optimizer_group(model)], weight_decay=0.0)
    model.geometry_wrappers["8"].channel_gate.grad = torch.ones(3072)
    optimizer.step()
    payload = build_v71_cf_checkpoint(
        step=10,
        model=model,
        optimizer=optimizer,
        lambda_cf=1.75,
        tau=0.1,
        init_seed=20260818,
        **_lineage(),
    )

    validate_v71_cf_checkpoint(
        payload,
        expected={**_lineage(), "lambda_cf": 1.75, "tau": 0.1, "init_seed": 20260818},
    )
    assert payload["step"] in V71_CF_STEPS
    assert payload["config"]["loss"] == "weighted-fm+geometry-counterfactual-softplus"
    assert payload["config"]["negative_schedule"] == "reverse,shift+1,swap,reverse,shift-1,swap"


def test_cf_checkpoint_rejects_lambda_or_parent_tampering() -> None:
    model = _Model()
    optimizer = torch.optim.AdamW([v71_optimizer_group(model)], weight_decay=0.0)
    model.geometry_wrappers["8"].channel_gate.grad = torch.ones(3072)
    optimizer.step()
    payload = build_v71_cf_checkpoint(
        step=25,
        model=model,
        optimizer=optimizer,
        lambda_cf=1.75,
        tau=0.1,
        init_seed=20260818,
        **_lineage(),
    )

    altered = copy.deepcopy(payload)
    altered["config"]["lambda_cf"] = 1.0
    with pytest.raises(ValueError, match="config"):
        validate_v71_cf_checkpoint(
            altered,
            expected={**_lineage(), "lambda_cf": 1.75, "tau": 0.1, "init_seed": 20260818},
        )

    altered = copy.deepcopy(payload)
    altered["parent_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="lineage"):
        validate_v71_cf_checkpoint(
            altered,
            expected={**_lineage(), "lambda_cf": 1.75, "tau": 0.1, "init_seed": 20260818},
        )
