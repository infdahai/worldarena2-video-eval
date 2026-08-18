from __future__ import annotations

import copy
import hashlib
import json

import pytest


torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from worldarena_baseline.wan_v6_training import build_v6_replay_manifest  # noqa: E402
from worldarena_baseline.wan_v7_training import (  # noqa: E402
    build_v7_checkpoint,
    build_v7_replay_from_v6,
    validate_v7_checkpoint,
    v7_discovery_gate,
    v7_optimizer_group,
    v7_training_contract,
)


_SHA = "fc54f851099ea213431efcb64a2fcbfd5c01335e18404f685768f00ece89b289"
_FROZEN_PARENT_SHA = "105fb760fd371885ba362d26ef2352c260755e47cd036f46711181edc3b30ca2"


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
        self.backbone = nn.Linear(1, 1, bias=False)
        self.backbone.requires_grad_(False)


def _v6_payload() -> dict:
    return build_v6_replay_manifest(
        dataset_size=1000, dataset_manifest_sha256=_SHA, seed=20260818
    )


def _replay() -> dict:
    return build_v7_replay_from_v6(
        _v6_payload(), clean1000_manifest_sha256=_SHA
    )


def _optimizer_payload(model: nn.Module, lr: float = 2e-5) -> dict:
    optimizer = torch.optim.AdamW([v7_optimizer_group(model, lr)], weight_decay=0.0)
    sum(parameter.sum() for parameter in model.parameters() if parameter.requires_grad).backward()
    optimizer.step()
    return optimizer.state_dict()


def _source_hashes() -> dict[str, str]:
    return {"source_manifest_sha256": "b" * 64, "source_code_sha256": "c" * 64}


def _expected(replay: dict, *, lr: float = 2e-5) -> dict:
    return {
        "parent_sha256": _FROZEN_PARENT_SHA,
        "source_hashes": _source_hashes(),
        "cache_sha256": "e" * 64,
        "replay_sha256": replay["replay_sha256"],
        "calibrated_lr": lr,
        "v6_replay": _v6_payload(),
        "replay": replay,
    }


def _checkpoint(step: int = 25) -> tuple[dict, dict]:
    model = _StageAModel()
    replay = _replay()
    expected = _expected(replay)
    checkpoint = build_v7_checkpoint(
        step=step,
        model=model,
        optimizer=_optimizer_payload(model),
        replay=replay,
        v6_replay=expected["v6_replay"],
        parent_sha256=expected["parent_sha256"],
        source_hashes=expected["source_hashes"],
        cache_sha256=expected["cache_sha256"],
        preflight_receipt={"calibrated_lr": expected["calibrated_lr"]},
        completed_step=step,
    )
    return checkpoint, expected


def test_v7_contract_is_the_approved_bounded_gate_only_probe() -> None:
    assert v7_training_contract() == {
        "contract": "wan-action-v7-se3-mechanism/1",
        "world_size": 7,
        "rank_mapping": [0, 1, 2, 3, 4, 5, 6],
        "dataset_rows": 1000,
        "max_steps": 50,
        "checkpoint_steps": [10, 25, 50],
        "injection_points": [8, 16, 24],
        "head_groups": {"left": [0, 12], "right": [12, 24]},
        "trainable_parameters": 9216,
        "loss": "weighted_flow_matching_only",
    }


def test_v7_replay_is_exact_first_50_steps_of_v6() -> None:
    source = _v6_payload()
    inferred = build_v7_replay_from_v6(source)
    actual = build_v7_replay_from_v6(source, clean1000_manifest_sha256=_SHA)
    assert inferred == actual
    assert actual["records"] == source["records"][:50]
    assert actual["max_steps"] == 50
    assert actual["dataset_manifest_sha256"] == _SHA
    assert actual["world_size"] == 7
    assert actual["rank_mapping"] == list(range(7))


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset_manifest_sha256", "b" * 64),
        ("world_size", 8),
        ("rank_mapping", list(range(8))),
    ],
)
def test_v7_replay_rejects_v6_lineage_drift(field: str, value: object) -> None:
    source = _v6_payload()
    source[field] = value
    with pytest.raises(ValueError, match="v6 replay"):
        build_v7_replay_from_v6(source, clean1000_manifest_sha256=_SHA)


def test_v7_replay_rejects_changed_first_50_record() -> None:
    source = _v6_payload()
    source["records"][0][0]["noise_seed"] += 1
    with pytest.raises(ValueError, match="first 50"):
        build_v7_replay_from_v6(source, clean1000_manifest_sha256=_SHA)


def test_v7_replay_rejects_foreign_but_internally_consistent_manifest() -> None:
    foreign = build_v6_replay_manifest(
        dataset_size=1000, dataset_manifest_sha256="f" * 64, seed=20260818
    )
    with pytest.raises(ValueError, match="trusted clean-1000"):
        build_v7_replay_from_v6(foreign)
    with pytest.raises(ValueError, match="trusted clean-1000"):
        build_v7_replay_from_v6(
            foreign, clean1000_manifest_sha256="f" * 64
        )


def test_v7_optimizer_contains_exactly_the_three_float32_gates() -> None:
    model = _StageAModel()
    group = v7_optimizer_group(model, calibrated_lr=2e-5)
    assert group["name"] == "se3_channel_gates"
    assert group["lr"] == pytest.approx(2e-5)
    assert len(group["params"]) == 3
    assert {tuple(parameter.shape) for parameter in group["params"]} == {(24, 128)}

    model.backbone.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="gate-only"):
        v7_optimizer_group(model, calibrated_lr=2e-5)


def test_checkpoint_rejects_any_non_gate_trainable_state() -> None:
    payload, expected = _checkpoint()
    payload["model"]["backbone.blocks.8.self_attn.q.weight"] = torch.ones(1)
    with pytest.raises(ValueError, match="gate-only"):
        validate_v7_checkpoint(payload, expected=expected)


def test_checkpoint_binds_lineage_gates_optimizer_and_resume_step() -> None:
    payload, expected = _checkpoint(step=25)
    validate_v7_checkpoint(payload, expected=expected)
    assert payload["scheduler"] == {"completed_step": 25, "calibrated_lr": 2e-5}
    assert payload["preflight"]["calibrated_lr"] == pytest.approx(2e-5)
    assert set(payload["model"]) == {
        "geometry_wrappers.8.gate",
        "geometry_wrappers.16.gate",
        "geometry_wrappers.24.gate",
    }

    wrong_step = copy.deepcopy(payload)
    wrong_step["scheduler"]["completed_step"] = 10
    with pytest.raises(ValueError, match="resume"):
        validate_v7_checkpoint(wrong_step, expected=expected)
    wrong_dtype = copy.deepcopy(payload)
    wrong_dtype["model"]["geometry_wrappers.8.gate"] = torch.zeros(24, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="float32"):
        validate_v7_checkpoint(wrong_dtype, expected=expected)
    wrong_optimizer = copy.deepcopy(payload)
    wrong_optimizer["optimizer"] = {"state": {}, "param_groups": []}
    with pytest.raises(ValueError, match="optimizer"):
        validate_v7_checkpoint(wrong_optimizer, expected=expected)


def test_checkpoint_rejects_mismatched_replay_parent_source_cache_or_lr() -> None:
    payload, expected = _checkpoint()
    for field, value in (
        ("parent_sha256", "f" * 64),
        ("cache_sha256", "f" * 64),
        ("replay_sha256", "f" * 64),
    ):
        broken = copy.deepcopy(payload)
        broken[field] = value
        with pytest.raises(ValueError):
            validate_v7_checkpoint(broken, expected=expected)
    source = copy.deepcopy(payload)
    source["source_hashes"]["source_code_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="source"):
        validate_v7_checkpoint(source, expected=expected)
    calibration = copy.deepcopy(payload)
    calibration["preflight"]["calibrated_lr"] = 1e-4
    with pytest.raises(ValueError, match="calibrated"):
        validate_v7_checkpoint(calibration, expected=expected)
    optimizer_lr = copy.deepcopy(payload)
    optimizer_lr["optimizer"]["param_groups"][0]["lr"] = 1e-3
    with pytest.raises(ValueError, match="calibrated"):
        validate_v7_checkpoint(optimizer_lr, expected=expected)


def test_checkpoint_rejects_foreign_but_internally_consistent_parent() -> None:
    model = _StageAModel()
    replay = _replay()
    with pytest.raises(ValueError, match="frozen parent"):
        build_v7_checkpoint(
            step=10,
            model=model,
            optimizer=_optimizer_payload(model),
            replay=replay,
            v6_replay=_v6_payload(),
            parent_sha256="f" * 64,
            source_hashes=_source_hashes(),
            cache_sha256="e" * 64,
            preflight_receipt={"calibrated_lr": 2e-5},
        )


def test_checkpoint_rejects_forged_but_self_consistent_v7_replay() -> None:
    model = _StageAModel()
    replay = _replay()
    expected = _expected(replay)
    forged = copy.deepcopy(replay)
    forged["dataset_manifest_sha256"] = "f" * 64
    forged["source_v6_replay_sha256"] = "f" * 64
    replay_without_digest = {key: value for key, value in forged.items() if key != "replay_sha256"}
    forged["replay_sha256"] = hashlib.sha256(
        json.dumps(replay_without_digest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="deterministic v6 prefix"):
        build_v7_checkpoint(
            step=10,
            model=model,
            optimizer=_optimizer_payload(model),
            replay=forged,
            v6_replay=expected["v6_replay"],
            parent_sha256=expected["parent_sha256"],
            source_hashes=expected["source_hashes"],
            cache_sha256=expected["cache_sha256"],
            preflight_receipt={"calibrated_lr": expected["calibrated_lr"]},
        )

    payload, _ = _checkpoint()
    forged_expected = dict(expected, replay=forged, replay_sha256=forged["replay_sha256"])
    payload["replay_sha256"] = forged["replay_sha256"]
    with pytest.raises(ValueError, match="deterministic v6 prefix"):
        validate_v7_checkpoint(payload, expected=forged_expected)


def test_discovery_gates_are_bounded_and_fail_closed() -> None:
    step10 = {
        "step": 10,
        "finite_fm": True,
        "finite_gates": True,
        "nonzero_gates": True,
        "residual_bounded": True,
        "correct_head_attribution": True,
        "original_parameters_unchanged": True,
    }
    assert v7_discovery_gate(step10)["pass"] is True

    step25 = {
        "step": 25,
        "counterfactual": {
            "reverse": {"paired_separation": True},
            "shift": {"paired_separation": False},
            "swap": {"paired_separation": False},
        },
        "position_improvement": 0.01,
        "velocity_improvement": 0.0,
    }
    assert v7_discovery_gate(step25)["pass"] is True

    step50 = {
        "step": 50,
        "aggregate_wins": 6,
        "position_improvement": 0.01,
        "velocity_improvement": 0.01,
        "routing_retention": 0.90,
        "fm_regression": 0.02,
        "absent_arm_output": False,
        "head_ownership_violation": False,
        "non_finite": False,
        "residual_domination": False,
    }
    assert v7_discovery_gate(step50)["pass"] is True
    assert v7_discovery_gate(dict(step50, aggregate_wins=5))["pass"] is False
    assert v7_discovery_gate(dict(step25, position_improvement=0.0))["pass"] is False


def test_replay_builder_cli_round_trips_exact_contract(tmp_path) -> None:
    manifest = tmp_path / "clean-1000.jsonl"
    manifest.write_text("{}\n" * 1000, encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    source = _v6_payload()
    replay = tmp_path / "v6-replay.json"
    replay.write_text(json.dumps(source), encoding="utf-8")
    output = tmp_path / "v7-replay.json"
    from scripts.build_wan_v7_replay import main

    with pytest.raises(ValueError, match="trusted clean-1000"):
        main(["--v6-replay", str(replay), "--clean-1000-manifest", str(manifest), "--output", str(output)])
    assert not output.exists()
