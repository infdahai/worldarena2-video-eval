"""Fail-closed clean parent and resumable checkpoint contracts for v6."""

from __future__ import annotations

import re
import math
from collections.abc import Mapping
from typing import Any


CLEAN_PARENT_CONTRACT = "wan-action-lite-v6-clean-gated-parent/1"
V6_CHECKPOINT_CONTRACT = "wan-action-lite-v6-checkpoint/1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def build_clean_gated_parent_payload(
    generic_checkpoint: Mapping[str, Any],
    *,
    base_parent_sha256: str,
    source_manifest_sha256: str,
    replay_sha256: str,
    data_leakage_receipt_sha256: str,
    leakage_validation: Mapping[str, Any],
    evaluation_manifest_sha256: Mapping[str, str],
    audit_summary: Mapping[str, Any],
    smoke_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze a generic support-gated step10 into the strict v6 parent contract."""
    config = generic_checkpoint.get("config")
    expected = {
        "structure_version": 3,
        "use_pose": False,
        "raster_support_gating": True,
        "adapter_only": True,
        "learning_rate": 2e-5,
        "warmup_steps": 10,
        "action_dropout": 0.1,
        "text_dropout": 0.1,
    }
    if generic_checkpoint.get("step") != 10 or not isinstance(config, Mapping):
        raise ValueError("clean parent source checkpoint is not step10")
    if any(config.get(name) != value for name, value in expected.items()):
        raise ValueError("clean parent source checkpoint recipe mismatch")
    adapter = generic_checkpoint.get("adapter")
    if not isinstance(adapter, Mapping) or not adapter:
        raise ValueError("clean parent source adapter is empty")
    for name, value in adapter.items():
        if not hasattr(value, "isfinite") or not bool(value.isfinite().all()):
            raise ValueError(f"clean parent source adapter tensor is invalid: {name}")
    if (
        leakage_validation.get("passed") is not True
        or leakage_validation.get("collision_count") != 0
        or leakage_validation.get("train_rows") != 1000
    ):
        raise ValueError("clean parent zero-leakage validation failed")
    if (
        smoke_receipt.get("contract")
        != "wan-action-lite-v3.2-production-smoke/1"
        or smoke_receipt.get("passed") is not True
        or smoke_receipt.get("completed_steps") != 3
        or [item.get("rank") for item in smoke_receipt.get("ranks", [])]
        != list(range(7))
    ):
        raise ValueError("clean parent production smoke failed")

    isolation = audit_summary.get("arm_isolation_score", {})
    routing = audit_summary.get("gradient_routing_ratio", {})
    margins = audit_summary.get("robot_margin_medians", {})
    wins = audit_summary.get("robot_margin_wins", {})
    names = ("swap", "reverse", "random-valid")
    try:
        audit_passed = (
            audit_summary.get("classification") == "causal_signal_present"
            and audit_summary.get("failure_reasons") == []
            and all(math.isfinite(float(isolation[arm])) and float(isolation[arm]) >= 1.5 for arm in ("left", "right"))
            and all(math.isfinite(float(routing[arm])) and float(routing[arm]) >= 1.5 for arm in ("left", "right"))
            and all(math.isfinite(float(margins[name])) and float(margins[name]) > 0 for name in names)
            and all(int(wins[name]) >= 7 for name in names)
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        audit_passed = False
    if not audit_passed:
        raise ValueError("clean parent causality audit failed")

    payload = {
        "contract": CLEAN_PARENT_CONTRACT,
        "step": 10,
        "config": dict(config),
        "base_parent_sha256": _sha(base_parent_sha256, label="base parent SHA-256"),
        "stage1": {
            "source_manifest_sha256": _sha(
                source_manifest_sha256, label="source manifest SHA-256"
            ),
            "replay_sha256": _sha(replay_sha256, label="parent replay SHA-256"),
            "world_size": 7,
            "rank_mapping": list(range(7)),
        },
        "data_leakage_receipt_sha256": _sha(
            data_leakage_receipt_sha256, label="leakage receipt SHA-256"
        ),
        "evaluation_manifest_sha256": {
            str(name): _sha(digest, label=f"evaluation {name} SHA-256")
            for name, digest in evaluation_manifest_sha256.items()
        },
        "adapter": dict(adapter),
        "parent_gate": {
            "passed": True,
            "zero_evaluation_leakage": True,
            "correct_better_than_swap_reverse_random": True,
            "routing_improved": True,
            "training_health_passed": True,
            "smoke_passed": True,
        },
        "audit_summary": dict(audit_summary),
        "leakage_validation": dict(leakage_validation),
    }
    validate_clean_gated_parent(
        payload,
        expected_base_parent_sha256=base_parent_sha256,
        expected_source_manifest_sha256=source_manifest_sha256,
    )
    return payload


def validate_clean_gated_parent(
    payload: Mapping[str, Any],
    *,
    expected_base_parent_sha256: str,
    expected_source_manifest_sha256: str,
) -> None:
    if payload.get("contract") != CLEAN_PARENT_CONTRACT:
        raise ValueError("clean gated parent contract mismatch")
    if payload.get("step") != 10:
        raise ValueError("clean gated parent must be step10")
    config = payload.get("config")
    if not isinstance(config, Mapping) or config.get("structure_version") != 3:
        raise ValueError("clean gated parent structure mismatch")
    if bool(config.get("use_pose")) or config.get("raster_support_gating") is not True:
        raise ValueError("clean gated parent must be raster-only with support gating")
    if _sha(payload.get("base_parent_sha256"), label="base parent SHA-256") != _sha(
        expected_base_parent_sha256, label="expected base parent SHA-256"
    ):
        raise ValueError("clean gated base parent SHA-256 mismatch")
    stage1 = payload.get("stage1")
    if not isinstance(stage1, Mapping):
        raise ValueError("clean gated parent lacks Stage-1 lineage")
    if _sha(stage1.get("source_manifest_sha256"), label="source manifest SHA-256") != _sha(
        expected_source_manifest_sha256, label="expected source manifest SHA-256"
    ):
        raise ValueError("clean gated parent source manifest SHA-256 mismatch")
    _sha(stage1.get("replay_sha256"), label="parent replay SHA-256")
    if stage1.get("world_size") != 7 or stage1.get("rank_mapping") != list(range(7)):
        raise ValueError("clean gated parent topology mismatch")
    _sha(payload.get("data_leakage_receipt_sha256"), label="leakage receipt SHA-256")
    evaluation = payload.get("evaluation_manifest_sha256")
    if not isinstance(evaluation, Mapping) or not evaluation:
        raise ValueError("clean gated parent lacks evaluation provenance")
    for name, digest in evaluation.items():
        if not isinstance(name, str) or not name:
            raise ValueError("clean gated evaluation name is invalid")
        _sha(digest, label=f"evaluation {name} SHA-256")
    adapter = payload.get("adapter")
    if not isinstance(adapter, Mapping) or not adapter:
        raise ValueError("clean gated parent adapter is empty")
    gate = payload.get("parent_gate")
    required = (
        "passed",
        "zero_evaluation_leakage",
        "correct_better_than_swap_reverse_random",
        "routing_improved",
        "training_health_passed",
        "smoke_passed",
    )
    if not isinstance(gate, Mapping) or any(gate.get(name) is not True for name in required):
        raise ValueError("clean gated parent gate is incomplete or failed")


def _validate_optimizer(optimizer: Any) -> None:
    if not isinstance(optimizer, Mapping):
        raise ValueError("v6 checkpoint optimizer is missing")
    state, groups = optimizer.get("state"), optimizer.get("param_groups")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("v6 checkpoint optimizer state is empty")
    if not isinstance(groups, list) or not groups:
        raise ValueError("v6 checkpoint optimizer groups are empty")
    referenced: set[Any] = set()
    for group in groups:
        if not isinstance(group, Mapping) or group.get("name") != "correction":
            raise ValueError("v6 checkpoint optimizer group mismatch")
        parameters = group.get("params")
        if not isinstance(parameters, list) or not parameters:
            raise ValueError("v6 checkpoint optimizer group has no parameters")
        referenced.update(parameters)
    if not referenced <= set(state):
        raise ValueError("v6 checkpoint optimizer parameter state is incomplete")
    for parameter in referenced:
        entry = state[parameter]
        if not isinstance(entry, Mapping) or not {"step", "exp_avg", "exp_avg_sq"} <= set(entry):
            raise ValueError("v6 checkpoint optimizer Adam state is incomplete")


def validate_v6_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    expected_parent_sha256: str,
    expected_probe_sha256: str,
    expected_config: Mapping[str, Any],
) -> None:
    if payload.get("contract") != V6_CHECKPOINT_CONTRACT:
        raise ValueError("v6 checkpoint contract mismatch")
    step = payload.get("step")
    if type(step) is not int or step not in (10, 25, 50, 100):
        raise ValueError("v6 checkpoint step is outside the approved schedule")
    if payload.get("config") != dict(expected_config):
        raise ValueError("v6 checkpoint config mismatch")
    if _sha(payload.get("parent_sha256"), label="parent SHA-256") != _sha(
        expected_parent_sha256, label="expected parent SHA-256"
    ):
        raise ValueError("v6 checkpoint parent SHA-256 mismatch")
    if _sha(payload.get("probe_sha256"), label="probe SHA-256") != _sha(
        expected_probe_sha256, label="expected probe SHA-256"
    ):
        raise ValueError("v6 checkpoint probe SHA-256 mismatch")
    for field in (
        "source_manifest_sha256",
        "replay_sha256",
        "sigma_calibration_sha256",
        "lambda_calibration_sha256",
    ):
        _sha(payload.get(field), label=field.replace("_", " "))
    if payload.get("world_size") != 7 or payload.get("rank_mapping") != list(range(7)):
        raise ValueError("v6 checkpoint topology mismatch")
    correction = payload.get("correction")
    if not isinstance(correction, Mapping) or not correction:
        raise ValueError("v6 checkpoint correction state is empty")
    _validate_optimizer(payload.get("optimizer"))
    scheduler = payload.get("scheduler")
    if not isinstance(scheduler, Mapping) or scheduler.get("completed_step") != step:
        raise ValueError("v6 checkpoint scheduler state mismatch")
    calibration = payload.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("v6 checkpoint calibration is missing")
    sigma, lambdas = calibration.get("sigma"), calibration.get("lambda")
    if (
        not isinstance(sigma, Mapping)
        or sigma.get("contract") != "wan-action-lite-v6-sigma-reliability/1"
        or sigma.get("frozen") is not True
        or not isinstance(lambdas, Mapping)
        or lambdas.get("contract") != "wan-action-lite-v6-lambda-calibration/1"
        or lambdas.get("frozen") is not True
        or lambdas.get("parameter_family") != "B"
    ):
        raise ValueError("v6 checkpoint calibration is mutable or invalid")
    telemetry = payload.get("telemetry")
    if not isinstance(telemetry, Mapping) or telemetry.get("B_grad_seen") is not True:
        raise ValueError("v6 checkpoint telemetry lacks B gradients")
