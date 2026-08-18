"""Immutable replay, optimizer, checkpoint, and audit contracts for Wan v7.

This module intentionally owns only the bounded Stage-A mechanism experiment.
It has no GPU, Wan-runtime, download, or filesystem-side-effect dependency, so
the trainer must validate these serializable contracts before it can allocate a
device or resume a run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from .wan_v6_training import build_v6_replay_manifest


V7_CONTRACT = "wan-action-v7-se3-mechanism/1"
V7_REPLAY_CONTRACT = "wan-action-v7-se3-replay/1"
V7_CHECKPOINT_CONTRACT = "wan-action-v7-se3-checkpoint/1"
V7_SINGLE_GPU_CONTRACT = "wan-action-v7-se3-single-gpu/1"
V7_SINGLE_GPU_REPLAY_CONTRACT = "wan-action-v7-se3-single-gpu-replay/1"
V7_SINGLE_GPU_CHECKPOINT_CONTRACT = "wan-action-v7-se3-single-gpu-checkpoint/1"
V7_WORLD_SIZE = 7
V7_RANK_MAPPING = tuple(range(V7_WORLD_SIZE))
V7_SINGLE_GPU_WORLD_SIZE = 1
V7_SINGLE_GPU_RANK_MAPPING = (6,)
V7_MAX_STEPS = 50
V7_CHECKPOINT_STEPS = (10, 25, 50)
V7_GATE_SHAPE = (24, 128)
V7_GATE_NAMES = tuple(
    f"geometry_wrappers.{block}.gate" for block in (8, 16, 24)
)
_SHA256 = re.compile(r"[0-9a-f]{64}")

# The clean-1000 hash is deliberately not supplied by a launcher, shell, or
# replay input.  Its tracked pin file is independently hash-bound here; the
# frozen parent is the user-approved clean-gated step-10 checkpoint.
_TRAINING_ROOT = Path(__file__).resolve().parents[2]
_TRUSTED_LINEAGE_PINS = _TRAINING_ROOT / "source_inputs/trusted-wan-v7-se3-lineage-pins.json"
_TRUSTED_LINEAGE_PINS_SHA256 = "50ba7efe44a6232962a55b90c9b3fca02aea839a1565e59cc7c9b396e420f6c8"
FROZEN_V7_PARENT_SHA256 = "105fb760fd371885ba362d26ef2352c260755e47cd036f46711181edc3b30ca2"


def v7_training_contract() -> dict[str, Any]:
    """Return the complete immutable Stage-A training boundary."""

    return {
        "contract": V7_CONTRACT,
        "world_size": V7_WORLD_SIZE,
        "rank_mapping": list(V7_RANK_MAPPING),
        "dataset_rows": 1000,
        "max_steps": V7_MAX_STEPS,
        "checkpoint_steps": list(V7_CHECKPOINT_STEPS),
        "injection_points": [8, 16, 24],
        "head_groups": {"left": [0, 12], "right": [12, 24]},
        "trainable_parameters": 9216,
        "loss": "weighted_flow_matching_only",
    }


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _canonical_sha256(payload: Mapping[str, Any], *, omit: str | None = None) -> str:
    canonical = {
        key: value for key, value in payload.items() if key != omit
    }
    try:
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("contract payload is not canonical JSON") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _trusted_clean1000_manifest_sha256() -> str:
    """Read the source-controlled clean-1000 pin without accepting overrides."""

    try:
        pin_bytes = _TRUSTED_LINEAGE_PINS.read_bytes()
    except OSError as exc:
        raise ValueError("trusted v7 lineage pins are unreadable") from exc
    if hashlib.sha256(pin_bytes).hexdigest() != _TRUSTED_LINEAGE_PINS_SHA256:
        raise ValueError("trusted v7 lineage pins differ from source-controlled hash")
    try:
        pins = json.loads(pin_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("trusted v7 lineage pins are invalid JSON") from exc
    try:
        digest = pins["artifacts"]["clean1000_manifest"]["sha256"]
    except (KeyError, TypeError) as exc:
        raise ValueError("trusted v7 clean-1000 pin is missing") from exc
    return _require_sha256(digest, label="trusted clean-1000 manifest SHA-256")


def _validate_v6_replay(
    payload: Mapping[str, Any], *, clean1000_manifest_sha256: str
) -> dict[str, Any]:
    """Reject anything other than the deterministic v6 clean-1000 schedule."""

    clean_hash = _require_sha256(
        clean1000_manifest_sha256, label="clean-1000 manifest SHA-256"
    )
    if not isinstance(payload, Mapping):
        raise ValueError("v6 replay must be a mapping")
    if payload.get("contract") != "wan-action-lite-v6-replay/1":
        raise ValueError("v6 replay contract mismatch")
    if payload.get("dataset_size") != 1000:
        raise ValueError("v6 replay dataset size is not clean-1000")
    if payload.get("dataset_manifest_sha256") != clean_hash:
        # This is a provenance boundary, not a recoverable replay formatting
        # error.  Keep the error anchored to the source-controlled trusted
        # clean-1000 contract so callers cannot mistake a foreign but
        # internally-consistent replay for a valid parent.
        raise ValueError("v6 replay does not use the trusted clean-1000 manifest")
    if payload.get("world_size") != V7_WORLD_SIZE:
        raise ValueError("v6 replay world size mismatch")
    if payload.get("rank_mapping") != list(V7_RANK_MAPPING):
        raise ValueError("v6 replay rank mapping mismatch")
    if payload.get("max_steps") != 100:
        raise ValueError("v6 replay max steps mismatch")
    seed = payload.get("seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("v6 replay seed is invalid")
    expected = build_v6_replay_manifest(
        dataset_size=1000, dataset_manifest_sha256=clean_hash, seed=seed
    )
    records = payload.get("records")
    if not isinstance(records, list) or len(records) < V7_MAX_STEPS:
        raise ValueError("v6 replay is missing the first 50 records")
    if records[:V7_MAX_STEPS] != expected["records"][:V7_MAX_STEPS]:
        raise ValueError("v6 replay first 50 records differ from deterministic source")
    if dict(payload) != expected:
        raise ValueError("v6 replay differs from immutable deterministic source")
    return expected


def build_v7_replay_from_v6(
    v6_replay: Mapping[str, Any], *, clean1000_manifest_sha256: str | None = None
) -> dict[str, Any]:
    """Derive the exact first 50 seven-rank draws from immutable v6 replay."""

    trusted_hash = _trusted_clean1000_manifest_sha256()
    if clean1000_manifest_sha256 is not None and clean1000_manifest_sha256 != trusted_hash:
        raise ValueError("v6 replay does not use the trusted clean-1000 manifest")
    source = _validate_v6_replay(
        v6_replay, clean1000_manifest_sha256=trusted_hash
    )
    payload: dict[str, Any] = {
        "contract": V7_REPLAY_CONTRACT,
        "source_v6_replay_sha256": _canonical_sha256(source),
        "world_size": V7_WORLD_SIZE,
        "rank_mapping": list(V7_RANK_MAPPING),
        "dataset_rows": 1000,
        "dataset_manifest_sha256": trusted_hash,
        "max_steps": V7_MAX_STEPS,
        "records": copy.deepcopy(source["records"][:V7_MAX_STEPS]),
    }
    payload["replay_sha256"] = _canonical_sha256(payload)
    return payload


def v7_single_gpu_training_contract() -> dict[str, Any]:
    """Return the isolated GPU6 mechanism-probe contract.

    This must remain separate from :func:`v7_training_contract`: a result
    produced with one physical device is evidence only and cannot resume the
    seven-rank Stage-A lineage.
    """

    return {
        "contract": V7_SINGLE_GPU_CONTRACT,
        "world_size": V7_SINGLE_GPU_WORLD_SIZE,
        "rank_mapping": list(V7_SINGLE_GPU_RANK_MAPPING),
        "dataset_rows": 1000,
        "max_steps": V7_MAX_STEPS,
        "checkpoint_steps": list(V7_CHECKPOINT_STEPS),
        "injection_points": [8, 16, 24],
        "head_groups": {"left": [0, 12], "right": [12, 24]},
        "trainable_parameters": 9216,
        "loss": "weighted_flow_matching_only",
    }


def build_v7_single_gpu_replay(v6_replay: Mapping[str, Any]) -> dict[str, Any]:
    """Select physical rank six from the pinned seven-rank v6 prefix.

    The input remains a full trusted v6 replay so the sample/noise/timestep
    schedule is identical to the seven-rank experiment.  Only each step's
    rank-six record enters this intentionally non-resumable single-GPU
    lineage.
    """

    trusted_hash = _trusted_clean1000_manifest_sha256()
    source = _validate_v6_replay(v6_replay, clean1000_manifest_sha256=trusted_hash)
    payload: dict[str, Any] = {
        "contract": V7_SINGLE_GPU_REPLAY_CONTRACT,
        "source_v6_replay_sha256": _canonical_sha256(source),
        "world_size": V7_SINGLE_GPU_WORLD_SIZE,
        "rank_mapping": list(V7_SINGLE_GPU_RANK_MAPPING),
        "dataset_rows": 1000,
        "dataset_manifest_sha256": trusted_hash,
        "max_steps": V7_MAX_STEPS,
        "records": [[copy.deepcopy(step[6])] for step in source["records"][:V7_MAX_STEPS]],
    }
    payload["replay_sha256"] = _canonical_sha256(payload)
    return payload


def _validate_v7_single_gpu_replay(
    payload: Mapping[str, Any], *, v6_replay: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed unless a replay is exactly the isolated rank-six prefix."""

    if not isinstance(payload, Mapping) or payload.get("contract") != V7_SINGLE_GPU_REPLAY_CONTRACT:
        raise ValueError("v7 single-gpu replay contract mismatch")
    required = {
        "contract",
        "source_v6_replay_sha256",
        "world_size",
        "rank_mapping",
        "dataset_rows",
        "dataset_manifest_sha256",
        "max_steps",
        "records",
        "replay_sha256",
    }
    if set(payload) != required:
        raise ValueError("v7 single-gpu replay schema mismatch")
    _require_sha256(payload.get("source_v6_replay_sha256"), label="source v6 replay SHA-256")
    _require_sha256(payload.get("dataset_manifest_sha256"), label="clean-1000 manifest SHA-256")
    replay_hash = _require_sha256(payload.get("replay_sha256"), label="v7 single-gpu replay SHA-256")
    if (
        payload.get("world_size") != V7_SINGLE_GPU_WORLD_SIZE
        or payload.get("rank_mapping") != list(V7_SINGLE_GPU_RANK_MAPPING)
    ):
        raise ValueError("v7 single-gpu replay topology mismatch")
    if payload.get("dataset_rows") != 1000 or payload.get("max_steps") != V7_MAX_STEPS:
        raise ValueError("v7 single-gpu replay data or step contract mismatch")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != V7_MAX_STEPS:
        raise ValueError("v7 single-gpu replay record count mismatch")
    if any(not isinstance(record, list) or len(record) != V7_SINGLE_GPU_WORLD_SIZE for record in records):
        raise ValueError("v7 single-gpu replay rank record shape mismatch")
    if replay_hash != _canonical_sha256(payload, omit="replay_sha256"):
        raise ValueError("v7 single-gpu replay SHA-256 mismatch")
    canonical = build_v7_single_gpu_replay(v6_replay)
    if dict(payload) != canonical:
        raise ValueError("v7 single-gpu replay differs from the trusted deterministic v6 rank-six prefix")
    return canonical


def _validate_v7_replay(
    payload: Mapping[str, Any], *, v6_replay: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("contract") != V7_REPLAY_CONTRACT:
        raise ValueError("v7 replay contract mismatch")
    required = {
        "contract",
        "source_v6_replay_sha256",
        "world_size",
        "rank_mapping",
        "dataset_rows",
        "dataset_manifest_sha256",
        "max_steps",
        "records",
        "replay_sha256",
    }
    if set(payload) != required:
        raise ValueError("v7 replay schema mismatch")
    _require_sha256(payload.get("source_v6_replay_sha256"), label="source v6 replay SHA-256")
    _require_sha256(payload.get("dataset_manifest_sha256"), label="clean-1000 manifest SHA-256")
    replay_hash = _require_sha256(payload.get("replay_sha256"), label="v7 replay SHA-256")
    if payload.get("world_size") != V7_WORLD_SIZE or payload.get("rank_mapping") != list(V7_RANK_MAPPING):
        raise ValueError("v7 replay topology mismatch")
    if payload.get("dataset_rows") != 1000 or payload.get("max_steps") != V7_MAX_STEPS:
        raise ValueError("v7 replay data or step contract mismatch")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != V7_MAX_STEPS:
        raise ValueError("v7 replay record count mismatch")
    if any(not isinstance(record, list) or len(record) != V7_WORLD_SIZE for record in records):
        raise ValueError("v7 replay rank record shape mismatch")
    if replay_hash != _canonical_sha256(payload, omit="replay_sha256"):
        raise ValueError("v7 replay SHA-256 mismatch")
    canonical = build_v7_replay_from_v6(v6_replay)
    if dict(payload) != canonical:
        raise ValueError("v7 replay differs from the trusted deterministic v6 prefix")
    return canonical


def _require_calibrated_lr(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("calibrated learning rate must be a finite positive float")
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("calibrated learning rate must be a finite positive float") from exc
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("calibrated learning rate must be a finite positive float")
    return rate


def _require_gate_tensor(value: Any, *, label: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{label} must be a tensor")
    if tuple(value.shape) != V7_GATE_SHAPE:
        raise ValueError(f"{label} must have shape {V7_GATE_SHAPE}")
    if value.dtype != torch.float32:
        raise ValueError(f"{label} must remain float32")
    if not torch.isfinite(value).all():
        raise ValueError(f"{label} must be finite")
    return value


def _gate_state_from_model(model: nn.Module | Mapping[str, Any]) -> dict[str, Tensor]:
    if isinstance(model, nn.Module):
        trainable = {
            name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        if set(trainable) != set(V7_GATE_NAMES):
            raise ValueError("v7 model trainable state must be gate-only")
        state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in trainable.items()
        }
    elif isinstance(model, Mapping):
        state = dict(model)
    else:
        raise TypeError("v7 model must be a module or gate-state mapping")
    if set(state) != set(V7_GATE_NAMES):
        raise ValueError("v7 model state must be gate-only")
    result: dict[str, Tensor] = {}
    for name in V7_GATE_NAMES:
        result[name] = _require_gate_tensor(state[name], label=name).detach().cpu().clone()
    return result


def v7_optimizer_group(model: nn.Module, calibrated_lr: float) -> dict[str, Any]:
    """Return the only legal Stage-A optimizer group: three channel gates."""

    rate = _require_calibrated_lr(calibrated_lr)
    state = _gate_state_from_model(model)
    named_parameters = dict(model.named_parameters())
    parameters = [named_parameters[name] for name in V7_GATE_NAMES]
    if len({id(parameter) for parameter in parameters}) != len(V7_GATE_NAMES):
        raise ValueError("v7 gate-only optimizer parameters are duplicated")
    if sum(parameter.numel() for parameter in parameters) != 9216:
        raise ValueError("v7 gate-only optimizer parameter count mismatch")
    # ``state`` validates the persisted tensors independently of the optimizer.
    del state
    return {"name": "se3_channel_gates", "params": parameters, "lr": rate}


def _optimizer_state(optimizer: Any) -> dict[str, Any]:
    raw = optimizer.state_dict() if isinstance(optimizer, torch.optim.Optimizer) else optimizer
    if not isinstance(raw, Mapping):
        raise ValueError("v7 checkpoint optimizer is missing")
    return copy.deepcopy(dict(raw))


def _validate_optimizer_state(optimizer: Any, *, calibrated_lr: float) -> None:
    if not isinstance(optimizer, Mapping):
        raise ValueError("v7 checkpoint optimizer is missing")
    state, groups = optimizer.get("state"), optimizer.get("param_groups")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("v7 checkpoint optimizer state is empty")
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("v7 checkpoint optimizer must have exactly one group")
    group = groups[0]
    if not isinstance(group, Mapping) or group.get("name") != "se3_channel_gates":
        raise ValueError("v7 checkpoint optimizer group must be se3_channel_gates")
    parameters = group.get("params")
    if not isinstance(parameters, list) or len(parameters) != 3 or len(set(parameters)) != 3:
        raise ValueError("v7 checkpoint optimizer must reference exactly three unique gates")
    if set(parameters) != set(state):
        raise ValueError("v7 checkpoint optimizer state must cover exactly the three gates")
    group_rate = _require_calibrated_lr(group.get("lr"))
    if group_rate != calibrated_lr:
        raise ValueError("v7 checkpoint optimizer calibrated learning rate mismatch")
    for parameter in parameters:
        entry = state[parameter]
        if not isinstance(entry, Mapping) or not {"step", "exp_avg", "exp_avg_sq"} <= set(entry):
            raise ValueError("v7 checkpoint optimizer Adam state is incomplete")
        for field in ("exp_avg", "exp_avg_sq"):
            _require_gate_tensor(entry[field], label=f"optimizer {field}")
        step = entry["step"]
        if isinstance(step, Tensor):
            if step.numel() != 1 or not torch.isfinite(step).all() or float(step.item()) <= 0:
                raise ValueError("v7 checkpoint optimizer step is invalid")
        elif type(step) is not int or step <= 0:
            raise ValueError("v7 checkpoint optimizer step is invalid")


def _source_hashes(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "source_manifest_sha256",
        "source_code_sha256",
    }:
        raise ValueError("source hashes schema mismatch")
    return {
        name: _require_sha256(value[name], label=name.replace("_", " "))
        for name in sorted(value)
    }


def build_v7_checkpoint(
    *,
    step: int,
    model: nn.Module | Mapping[str, Any],
    optimizer: Any,
    replay: Mapping[str, Any],
    v6_replay: Mapping[str, Any],
    parent_sha256: str,
    source_hashes: Mapping[str, Any],
    cache_sha256: str,
    preflight_receipt: Mapping[str, Any],
    completed_step: int | None = None,
) -> dict[str, Any]:
    """Build an independently resumable, gate-only Stage-A checkpoint.

    ``preflight_receipt`` must carry its calibrated gate learning rate under
    ``calibrated_lr``; the launcher is never an authority for that value.
    """

    if type(step) is not int or step not in V7_CHECKPOINT_STEPS:
        raise ValueError("v7 checkpoint step is outside the approved schedule")
    if completed_step is None:
        completed_step = step
    if type(completed_step) is not int or completed_step != step:
        raise ValueError("v7 checkpoint resume step must equal checkpoint step")
    if not isinstance(preflight_receipt, Mapping):
        raise ValueError("v7 checkpoint preflight receipt is missing")
    rate = _require_calibrated_lr(preflight_receipt.get("calibrated_lr"))
    replay_payload = _validate_v7_replay(replay, v6_replay=v6_replay)
    state = _gate_state_from_model(model)
    optimizer_payload = _optimizer_state(optimizer)
    _validate_optimizer_state(optimizer_payload, calibrated_lr=rate)
    parent = _require_sha256(parent_sha256, label="parent SHA-256")
    if parent != FROZEN_V7_PARENT_SHA256:
        raise ValueError("v7 checkpoint does not use the frozen parent SHA-256")
    payload = {
        "contract": V7_CHECKPOINT_CONTRACT,
        "step": step,
        "config": v7_training_contract(),
        "world_size": V7_WORLD_SIZE,
        "rank_mapping": list(V7_RANK_MAPPING),
        "parent_sha256": parent,
        "source_hashes": _source_hashes(source_hashes),
        "cache_sha256": _require_sha256(cache_sha256, label="cache SHA-256"),
        "replay_sha256": replay_payload["replay_sha256"],
        "model": state,
        "optimizer": optimizer_payload,
        "scheduler": {"completed_step": completed_step, "calibrated_lr": rate},
        "preflight": dict(preflight_receipt),
    }
    validate_v7_checkpoint(
        payload,
        expected={
            "parent_sha256": parent_sha256,
            "source_hashes": source_hashes,
            "cache_sha256": cache_sha256,
            "replay_sha256": replay_payload["replay_sha256"],
            "calibrated_lr": rate,
            "v6_replay": v6_replay,
            "replay": replay_payload,
        },
    )
    return payload


def validate_v7_checkpoint(payload: Mapping[str, Any], *, expected: Mapping[str, Any]) -> None:
    """Fail closed before Stage-A checkpoint state is loaded for a resume."""

    if not isinstance(payload, Mapping) or payload.get("contract") != V7_CHECKPOINT_CONTRACT:
        raise ValueError("v7 checkpoint contract mismatch")
    if payload.get("config") != v7_training_contract():
        raise ValueError("v7 checkpoint config mismatch")
    step = payload.get("step")
    if type(step) is not int or step not in V7_CHECKPOINT_STEPS:
        raise ValueError("v7 checkpoint step is outside the approved schedule")
    if payload.get("world_size") != V7_WORLD_SIZE or payload.get("rank_mapping") != list(V7_RANK_MAPPING):
        raise ValueError("v7 checkpoint topology mismatch")
    if not isinstance(expected, Mapping):
        raise ValueError("v7 checkpoint expected lineage is missing")
    parent = _require_sha256(payload.get("parent_sha256"), label="parent SHA-256")
    if parent != FROZEN_V7_PARENT_SHA256:
        raise ValueError("v7 checkpoint does not use the frozen parent SHA-256")
    expected_parent = _require_sha256(expected.get("parent_sha256"), label="expected parent SHA-256")
    if expected_parent != FROZEN_V7_PARENT_SHA256 or parent != expected_parent:
        raise ValueError("v7 checkpoint parent SHA-256 mismatch")
    sources = _source_hashes(payload.get("source_hashes"))
    expected_sources = _source_hashes(expected.get("source_hashes"))
    if sources != expected_sources:
        raise ValueError("v7 checkpoint source hashes mismatch")
    cache = _require_sha256(payload.get("cache_sha256"), label="cache SHA-256")
    expected_cache = _require_sha256(expected.get("cache_sha256"), label="expected cache SHA-256")
    if cache != expected_cache:
        raise ValueError("v7 checkpoint cache SHA-256 mismatch")
    replay = _require_sha256(payload.get("replay_sha256"), label="replay SHA-256")
    expected_replay = _require_sha256(expected.get("replay_sha256"), label="expected replay SHA-256")
    if replay != expected_replay:
        raise ValueError("v7 checkpoint replay SHA-256 mismatch")
    expected_replay_payload = expected.get("replay")
    expected_v6_replay = expected.get("v6_replay")
    if not isinstance(expected_replay_payload, Mapping) or not isinstance(expected_v6_replay, Mapping):
        raise ValueError("v7 checkpoint expected replay lineage is missing")
    canonical_replay = _validate_v7_replay(
        expected_replay_payload, v6_replay=expected_v6_replay
    )
    if canonical_replay["replay_sha256"] != replay:
        raise ValueError("v7 checkpoint replay differs from trusted deterministic v6 prefix")
    _gate_state_from_model(payload.get("model"))
    scheduler = payload.get("scheduler")
    if not isinstance(scheduler, Mapping) or scheduler.get("completed_step") != step:
        raise ValueError("v7 checkpoint resume step is inconsistent")
    scheduler_rate = _require_calibrated_lr(scheduler.get("calibrated_lr"))
    preflight = payload.get("preflight")
    if not isinstance(preflight, Mapping):
        raise ValueError("v7 checkpoint preflight receipt is missing")
    preflight_rate = _require_calibrated_lr(preflight.get("calibrated_lr"))
    expected_rate = _require_calibrated_lr(expected.get("calibrated_lr"))
    if scheduler_rate != preflight_rate or scheduler_rate != expected_rate:
        raise ValueError("v7 checkpoint calibrated learning rate mismatch")
    _validate_optimizer_state(payload.get("optimizer"), calibrated_lr=expected_rate)


def build_v7_single_gpu_checkpoint(
    *,
    step: int,
    model: nn.Module | Mapping[str, Any],
    optimizer: Any,
    replay: Mapping[str, Any],
    v6_replay: Mapping[str, Any],
    parent_sha256: str,
    source_hashes: Mapping[str, Any],
    cache_sha256: str,
    preflight_receipt: Mapping[str, Any],
    completed_step: int | None = None,
) -> dict[str, Any]:
    """Build a checkpoint usable only by the physical-GPU6 probe.

    The implementation intentionally does not call the seven-rank checkpoint
    builder: that would make an accidental topology conversion appear valid.
    """

    if type(step) is not int or step not in V7_CHECKPOINT_STEPS:
        raise ValueError("v7 single-gpu checkpoint step is outside the approved schedule")
    if completed_step is None:
        completed_step = step
    if type(completed_step) is not int or completed_step != step:
        raise ValueError("v7 single-gpu checkpoint resume step must equal checkpoint step")
    if not isinstance(preflight_receipt, Mapping):
        raise ValueError("v7 single-gpu checkpoint preflight receipt is missing")
    rate = _require_calibrated_lr(preflight_receipt.get("calibrated_lr"))
    replay_payload = _validate_v7_single_gpu_replay(replay, v6_replay=v6_replay)
    state = _gate_state_from_model(model)
    optimizer_payload = _optimizer_state(optimizer)
    _validate_optimizer_state(optimizer_payload, calibrated_lr=rate)
    parent = _require_sha256(parent_sha256, label="parent SHA-256")
    if parent != FROZEN_V7_PARENT_SHA256:
        raise ValueError("v7 single-gpu checkpoint does not use the frozen parent SHA-256")
    payload = {
        "contract": V7_SINGLE_GPU_CHECKPOINT_CONTRACT,
        "step": step,
        "config": v7_single_gpu_training_contract(),
        "world_size": V7_SINGLE_GPU_WORLD_SIZE,
        "rank_mapping": list(V7_SINGLE_GPU_RANK_MAPPING),
        "parent_sha256": parent,
        "source_hashes": _source_hashes(source_hashes),
        "cache_sha256": _require_sha256(cache_sha256, label="cache SHA-256"),
        "replay_sha256": replay_payload["replay_sha256"],
        "model": state,
        "optimizer": optimizer_payload,
        "scheduler": {"completed_step": completed_step, "calibrated_lr": rate},
        "preflight": dict(preflight_receipt),
    }
    validate_v7_single_gpu_checkpoint(
        payload,
        expected={
            "parent_sha256": parent_sha256,
            "source_hashes": source_hashes,
            "cache_sha256": cache_sha256,
            "replay_sha256": replay_payload["replay_sha256"],
            "calibrated_lr": rate,
            "v6_replay": v6_replay,
            "replay": replay_payload,
        },
    )
    return payload


def validate_v7_single_gpu_checkpoint(
    payload: Mapping[str, Any], *, expected: Mapping[str, Any]
) -> None:
    """Reject a seven-rank artifact before any single-GPU resume can load it."""

    if (
        not isinstance(payload, Mapping)
        or payload.get("contract") != V7_SINGLE_GPU_CHECKPOINT_CONTRACT
    ):
        raise ValueError("v7 single-gpu checkpoint contract mismatch")
    if payload.get("config") != v7_single_gpu_training_contract():
        raise ValueError("v7 single-gpu checkpoint config mismatch")
    step = payload.get("step")
    if type(step) is not int or step not in V7_CHECKPOINT_STEPS:
        raise ValueError("v7 single-gpu checkpoint step is outside the approved schedule")
    if (
        payload.get("world_size") != V7_SINGLE_GPU_WORLD_SIZE
        or payload.get("rank_mapping") != list(V7_SINGLE_GPU_RANK_MAPPING)
    ):
        raise ValueError("v7 single-gpu checkpoint topology mismatch")
    if not isinstance(expected, Mapping):
        raise ValueError("v7 single-gpu checkpoint expected lineage is missing")
    parent = _require_sha256(payload.get("parent_sha256"), label="parent SHA-256")
    expected_parent = _require_sha256(expected.get("parent_sha256"), label="expected parent SHA-256")
    if (
        parent != FROZEN_V7_PARENT_SHA256
        or expected_parent != FROZEN_V7_PARENT_SHA256
        or parent != expected_parent
    ):
        raise ValueError("v7 single-gpu checkpoint parent SHA-256 mismatch")
    sources = _source_hashes(payload.get("source_hashes"))
    expected_sources = _source_hashes(expected.get("source_hashes"))
    if sources != expected_sources:
        raise ValueError("v7 single-gpu checkpoint source hashes mismatch")
    cache = _require_sha256(payload.get("cache_sha256"), label="cache SHA-256")
    expected_cache = _require_sha256(expected.get("cache_sha256"), label="expected cache SHA-256")
    if cache != expected_cache:
        raise ValueError("v7 single-gpu checkpoint cache SHA-256 mismatch")
    replay = _require_sha256(payload.get("replay_sha256"), label="single-gpu replay SHA-256")
    expected_replay = _require_sha256(expected.get("replay_sha256"), label="expected single-gpu replay SHA-256")
    if replay != expected_replay:
        raise ValueError("v7 single-gpu checkpoint replay SHA-256 mismatch")
    expected_replay_payload = expected.get("replay")
    expected_v6_replay = expected.get("v6_replay")
    if not isinstance(expected_replay_payload, Mapping) or not isinstance(expected_v6_replay, Mapping):
        raise ValueError("v7 single-gpu checkpoint expected replay lineage is missing")
    canonical_replay = _validate_v7_single_gpu_replay(
        expected_replay_payload, v6_replay=expected_v6_replay
    )
    if canonical_replay["replay_sha256"] != replay:
        raise ValueError("v7 single-gpu checkpoint replay differs from trusted rank-six prefix")
    _gate_state_from_model(payload.get("model"))
    scheduler = payload.get("scheduler")
    if not isinstance(scheduler, Mapping) or scheduler.get("completed_step") != step:
        raise ValueError("v7 single-gpu checkpoint resume step is inconsistent")
    scheduler_rate = _require_calibrated_lr(scheduler.get("calibrated_lr"))
    preflight = payload.get("preflight")
    if not isinstance(preflight, Mapping):
        raise ValueError("v7 single-gpu checkpoint preflight receipt is missing")
    preflight_rate = _require_calibrated_lr(preflight.get("calibrated_lr"))
    expected_rate = _require_calibrated_lr(expected.get("calibrated_lr"))
    if scheduler_rate != preflight_rate or scheduler_rate != expected_rate:
        raise ValueError("v7 single-gpu checkpoint calibrated learning rate mismatch")
    _validate_optimizer_state(payload.get("optimizer"), calibrated_lr=expected_rate)


def _finite_metric(metrics: Mapping[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def v7_discovery_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the fixed health (10), mechanism (25), or promotion (50) gate."""

    if not isinstance(metrics, Mapping):
        return {"pass": False, "reasons": ["metrics_must_be_a_mapping"]}
    step = metrics.get("step")
    reasons: list[str] = []
    if step == 10:
        for name in (
            "finite_fm",
            "finite_gates",
            "nonzero_gates",
            "residual_bounded",
            "correct_head_attribution",
            "original_parameters_unchanged",
        ):
            if metrics.get(name) is not True:
                reasons.append(f"{name}_failed")
        return {"pass": not reasons, "step": step, "reasons": reasons}
    if step == 25:
        families = metrics.get("counterfactual")
        separated = False
        if isinstance(families, Mapping):
            separated = any(
                isinstance(value, Mapping) and value.get("paired_separation") is True
                for value in families.values()
            )
        if not separated:
            reasons.append("no_paired_counterfactual_separation")
        position = _finite_metric(metrics, "position_improvement")
        velocity = _finite_metric(metrics, "velocity_improvement")
        if not ((position is not None and position > 0) or (velocity is not None and velocity > 0)):
            reasons.append("no_positive_probe_direction")
        return {"pass": not reasons, "step": step, "reasons": reasons}
    if step == 50:
        aggregate_wins = metrics.get("aggregate_wins")
        if type(aggregate_wins) is not int or aggregate_wins < 6:
            reasons.append("aggregate_wins_below_6_of_8")
        position = _finite_metric(metrics, "position_improvement")
        velocity = _finite_metric(metrics, "velocity_improvement")
        if position is None or position <= 0:
            reasons.append("position_improvement_not_positive")
        if velocity is None or velocity <= 0:
            reasons.append("velocity_improvement_not_positive")
        routing = _finite_metric(metrics, "routing_retention")
        if routing is None or routing < 0.90:
            reasons.append("routing_retention_below_90_percent")
        fm = _finite_metric(metrics, "fm_regression")
        if fm is None or fm > 0.02:
            reasons.append("flow_matching_regression_above_2_percent")
        for name in (
            "absent_arm_output",
            "head_ownership_violation",
            "non_finite",
            "residual_domination",
        ):
            if metrics.get(name) is not False:
                reasons.append(f"{name}_detected")
        return {"pass": not reasons, "step": step, "reasons": reasons}
    return {"pass": False, "reasons": ["unsupported_gate_step"]}
