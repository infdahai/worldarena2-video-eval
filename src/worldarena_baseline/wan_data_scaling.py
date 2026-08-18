"""Fail-closed lineage contracts for the clean50 v2 data-scaling experiment."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Mapping


class DataScalingContractError(ValueError):
    pass


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise DataScalingContractError(f"{label} must be a lowercase SHA-256")
    return value


def _optimizer_is_reusable(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and isinstance(value.get("state"), Mapping)
        and bool(value["state"])
        and isinstance(value.get("param_groups"), list)
        and bool(value["param_groups"])
        and all(
            isinstance(group, Mapping)
            and isinstance(group.get("params"), list)
            and bool(group["params"])
            for group in value["param_groups"]
        )
    )


def validate_data_scaling_parent(
    payload: Mapping, *, expected_world_size: int
) -> bool:
    """Return pose mode after validating a formal Stage2-A step-400 parent."""
    config = payload.get("config")
    stage2 = payload.get("stage2")
    if payload.get("step") != 400 or not isinstance(stage2, Mapping) or stage2.get(
        "completed_step"
    ) != 400:
        raise DataScalingContractError("data scaling requires a formal step-400 parent")
    if not isinstance(config, Mapping) or config.get("structure_version") != 3:
        raise DataScalingContractError("data-scaling parent config is invalid")
    if config.get("use_pose") is not False:
        raise DataScalingContractError("first data-scaling experiment must be raster-only")
    exact = {
        "adapter_only": False,
        "lora_rank": 16,
        "lora_alpha": 16,
        "lora_block_start": 8,
        "lora_block_end": 30,
        "lora_targets": ["self_attn.q", "self_attn.v"],
        "action_dropout": 0.1,
        "text_dropout": 0.1,
        "focus_weight": 2.0,
        "trajectory_weight": 1.0,
        "gripper_weight": 2.0,
        "learning_rate": 1e-4,
        "lora_learning_rate": 2e-5,
        "warmup_steps": 50,
        "max_grad_norm": 1.0,
    }
    if any(config.get(key) != value for key, value in exact.items()):
        raise DataScalingContractError("data-scaling parent architecture/training contract differs")
    if stage2.get("world_size") != expected_world_size or stage2.get(
        "rank_mapping"
    ) != list(range(expected_world_size)):
        raise DataScalingContractError("data-scaling parent world size or rank mapping differs")
    _require_sha256(stage2.get("selected_stage1_parent_sha256"), "Stage2 parent lineage")
    if not isinstance(payload.get("adapter"), Mapping) or not payload["adapter"]:
        raise DataScalingContractError("data-scaling parent adapter state is missing")
    if not isinstance(payload.get("lora"), Mapping) or not payload["lora"]:
        raise DataScalingContractError("data-scaling parent LoRA state is missing")
    if not _optimizer_is_reusable(payload.get("adapter_optimizer")) or not _optimizer_is_reusable(
        payload.get("lora_optimizer")
    ):
        raise DataScalingContractError("data-scaling parent optimizer state is not reusable")
    return False


def validate_data_scaling_request(
    *, dataset_size: int, total_steps: int, save_every: int, smoke: bool = False
) -> None:
    if smoke:
        if dataset_size != 500 or total_steps != 3 or save_every != 3:
            raise DataScalingContractError(
                "data-scaling smoke requires v2-500, 3 steps, and save_every 3"
            )
        return
    if dataset_size == 500:
        if total_steps != 100 or save_every != 50:
            raise DataScalingContractError("v2-500 screen requires 100 steps and save_every 50")
        return
    if dataset_size == 1000:
        if total_steps not in {300, 450, 600} or save_every != 100:
            raise DataScalingContractError(
                "v2-1000 general requires 300/450/600 steps and save_every 100"
            )
        return
    raise DataScalingContractError("data scaling dataset_size must be v2-500 or v2-1000")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_selection_receipt(
    receipt_path: Path | str,
    *,
    manifest_path: Path | str,
    dataset_size: int,
) -> dict[str, str]:
    receipt_file = Path(receipt_path)
    manifest_file = Path(manifest_path)
    try:
        payload = json.loads(receipt_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataScalingContractError("data selection receipt is unreadable") from exc
    if payload.get("schema_version") != 1 or payload.get("contract") != (
        "wan-action-lite-v3.2-data-selection/1"
    ):
        raise DataScalingContractError("data selection receipt contract differs")
    manifests = payload.get("manifests")
    key = f"v2-{dataset_size}"
    selected = manifests.get(key) if isinstance(manifests, Mapping) else None
    if not isinstance(selected, Mapping):
        raise DataScalingContractError(f"selection receipt lacks {key}")
    if selected.get("episodes") != dataset_size or selected.get("tasks") != 36:
        raise DataScalingContractError("selection receipt episode/task coverage differs")
    manifest_sha256 = _sha256(manifest_file)
    if selected.get("sha256") != manifest_sha256:
        raise DataScalingContractError("selection receipt manifest hash differs")
    return {
        "manifest_sha256": manifest_sha256,
        "receipt_sha256": _sha256(receipt_file),
    }


def build_data_scaling_metadata(
    *,
    parent_checkpoint_sha256: str,
    source_manifest_sha256: str,
    selection_receipt_sha256: str,
    dataset_size: int,
    completed_local_step: int,
    world_size: int,
) -> dict:
    parent = _require_sha256(parent_checkpoint_sha256, "data-scaling parent")
    source = _require_sha256(source_manifest_sha256, "data-scaling source manifest")
    receipt = _require_sha256(selection_receipt_sha256, "data-scaling selection receipt")
    if dataset_size not in {500, 1000}:
        raise DataScalingContractError("data-scaling dataset size is invalid")
    if completed_local_step <= 0:
        raise DataScalingContractError("data-scaling local step must be positive")
    if world_size not in {7, 8}:
        raise DataScalingContractError("data-scaling world size must be seven or eight")
    return {
        "schema_version": 1,
        "contract": "wan-action-lite-v3.2-data-scaling-checkpoint/1",
        "parent_checkpoint_sha256": parent,
        "parent_completed_step": 400,
        "source_manifest_sha256": source,
        "selection_receipt_sha256": receipt,
        "dataset_size": dataset_size,
        "completed_local_step": completed_local_step,
        "total_path_updates": 400 + completed_local_step,
        "optimizer_state_reused": True,
        "world_size": world_size,
        "rank_mapping": list(range(world_size)),
    }


def validate_cached_manifest_identity(
    source_manifest: Path | str,
    cached_manifest: Path | str,
    *,
    expected_rows: int,
) -> None:
    def rows(path: Path | str) -> list[dict]:
        return [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    source = rows(source_manifest)
    cached = rows(cached_manifest)
    if len(source) != expected_rows or len(cached) != expected_rows:
        raise DataScalingContractError("source/cached manifest row count differs")
    source_samples = [row.get("sample") for row in source]
    cached_samples = [row.get("sample") for row in cached]
    if source_samples != cached_samples or any(not sample for sample in source_samples):
        raise DataScalingContractError("source/cached manifest sample order differs")
