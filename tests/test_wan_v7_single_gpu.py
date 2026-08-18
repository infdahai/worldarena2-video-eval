from __future__ import annotations

import copy

import pytest


torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from worldarena_baseline.wan_v6_training import build_v6_replay_manifest  # noqa: E402
from worldarena_baseline.wan_v7_training import (  # noqa: E402
    build_v7_checkpoint,
    build_v7_replay_from_v6,
    build_v7_single_gpu_checkpoint,
    build_v7_single_gpu_replay,
    validate_v7_single_gpu_checkpoint,
    v7_optimizer_group,
)


_CLEAN1000_SHA = "fc54f851099ea213431efcb64a2fcbfd5c01335e18404f685768f00ece89b289"
_PARENT_SHA = "105fb760fd371885ba362d26ef2352c260755e47cd036f46711181edc3b30ca2"


class _Gate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(24, 128, dtype=torch.float32))


class _StageAModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.geometry_wrappers = nn.ModuleDict(
            {str(block): _Gate() for block in (8, 16, 24)}
        )


def _v6_replay() -> dict:
    return build_v6_replay_manifest(
        dataset_size=1000,
        dataset_manifest_sha256=_CLEAN1000_SHA,
        seed=20260818,
    )


def _optimizer_payload(model: nn.Module, rate: float = 2e-5) -> dict:
    optimizer = torch.optim.AdamW([v7_optimizer_group(model, rate)], weight_decay=0.0)
    sum(parameter.sum() for parameter in model.parameters()).backward()
    optimizer.step()
    return optimizer.state_dict()


def _expected(replay: dict) -> dict:
    return {
        "parent_sha256": _PARENT_SHA,
        "source_hashes": {
            "source_manifest_sha256": "b" * 64,
            "source_code_sha256": "c" * 64,
        },
        "cache_sha256": "e" * 64,
        "replay_sha256": replay["replay_sha256"],
        "calibrated_lr": 2e-5,
        "v6_replay": _v6_replay(),
        "replay": replay,
    }


def test_single_gpu_replay_uses_only_physical_rank_six() -> None:
    v6_replay = _v6_replay()

    replay = build_v7_single_gpu_replay(v6_replay)

    assert replay["contract"] == "wan-action-v7-se3-single-gpu-replay/1"
    assert replay["world_size"] == 1
    assert replay["rank_mapping"] == [6]
    assert len(replay["records"]) == 50
    assert all(len(step) == 1 for step in replay["records"])
    assert [step[0] for step in replay["records"]] == [
        step[6] for step in v6_replay["records"][:50]
    ]


def test_single_gpu_checkpoint_rejects_seven_rank_replay() -> None:
    model = _StageAModel()
    replay = build_v7_replay_from_v6(_v6_replay())
    with pytest.raises(ValueError, match="single-gpu replay"):
        build_v7_single_gpu_checkpoint(
            step=10,
            model=model,
            optimizer=_optimizer_payload(model),
            replay=replay,
            v6_replay=_v6_replay(),
            parent_sha256=_PARENT_SHA,
            source_hashes=_expected(build_v7_single_gpu_replay(_v6_replay()))["source_hashes"],
            cache_sha256="e" * 64,
            preflight_receipt={"calibrated_lr": 2e-5},
        )


def test_single_gpu_checkpoint_rejects_seven_rank_checkpoint_contract() -> None:
    model = _StageAModel()
    replay = build_v7_single_gpu_replay(_v6_replay())
    expected = _expected(replay)
    checkpoint = build_v7_single_gpu_checkpoint(
        step=10,
        model=model,
        optimizer=_optimizer_payload(model),
        replay=replay,
        v6_replay=expected["v6_replay"],
        parent_sha256=expected["parent_sha256"],
        source_hashes=expected["source_hashes"],
        cache_sha256=expected["cache_sha256"],
        preflight_receipt={"calibrated_lr": expected["calibrated_lr"]},
    )
    validate_v7_single_gpu_checkpoint(checkpoint, expected=expected)

    foreign = copy.deepcopy(checkpoint)
    foreign["contract"] = "wan-action-v7-se3-checkpoint/1"
    with pytest.raises(ValueError, match="single-gpu checkpoint contract"):
        validate_v7_single_gpu_checkpoint(foreign, expected=expected)
