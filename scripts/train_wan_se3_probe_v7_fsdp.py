#!/usr/bin/env python3
"""Run the bounded seven-rank Wan v7 SE(3) mechanism experiment.

This entrypoint is deliberately fail-closed.  It consumes cached latent/text
inputs and provenance-bound action sidecars only; it never builds encoders,
downloads weights, decodes video, or expands the Stage-A parameter set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


FORMAL_ROOT = Path("/data/di/worldarena2_track1_20260815")
SOURCE_ROOT = Path("/home/huazhi/nlh/baseline")
WAN_SOURCE = Path("/home/huazhi/nlh/Wan2.2")
CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5,6"
SINGLE_GPU_CUDA_VISIBLE_DEVICES = "6"
CALIBRATED_LR = 2e-5
V71_ARCHITECTURE = "v71-geometry-lora"
V71_CF_ARCHITECTURE = "v71-geometry-lora-cf"
MEMORY_LIMIT_BYTES = 22 * 1024**3
FROZEN_V7_PARENT_SHA256 = "105fb760fd371885ba362d26ef2352c260755e47cd036f46711181edc3b30ca2"
V7_CHECKPOINT_STEPS = (10, 25, 50)
V7_WORLD_SIZE = 7
V7_SINGLE_GPU_WORLD_SIZE = 1
V7_GATE_ONLY_RETIREMENT_CHECKPOINTS = {
    10: "d77977f975f634eeb4e621c05c15647504afbf589de46ad5cbbb940ca1237bdb",
    25: "760a866ded9cb659d0cef9bc07c279bd4d60fc2dd6e4c7121357ca3b1200b47d",
}
_TOPOLOGIES = {
    "seven-rank": {
        "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
        "world_size": V7_WORLD_SIZE,
        "rank_mapping": list(range(V7_WORLD_SIZE)),
        "smoke_contract": "wan-action-v7-seven-rank-production-smoke/1",
    },
    "single-gpu": {
        "cuda_visible_devices": SINGLE_GPU_CUDA_VISIBLE_DEVICES,
        "world_size": V7_SINGLE_GPU_WORLD_SIZE,
        "rank_mapping": [6],
        "smoke_contract": "wan-action-v7-single-gpu-production-smoke/1",
    },
}
TRUSTED_PINS = SOURCE_ROOT / "source_inputs/trusted-wan-v7-se3-lineage-pins.json"
TRUSTED_PINS_SHA256 = "50ba7efe44a6232962a55b90c9b3fca02aea839a1565e59cc7c9b396e420f6c8"
def _load_runtime_dependencies() -> None:
    """Delay GPU-stack imports so ``--help`` remains a pure local operation."""

    global np, torch, dist
    global validate_training_manifest_receipt, replay_noise, sha256_file
    global WanActionAdapter, enable_wan_block_checkpointing, ti2v_flow_matching_sample
    global weighted_flow_mse, WanActionCachedDataset, validate_cached_manifest_identity
    global GripperTrajectoryProbe, build_gripper_targets, soft_argmax_2d
    global require_wan_backbone_checkpoint, validate_training_hot_path_components
    global validate_se3_cache, install_wan_ti2v_package, validate_clean_gated_parent
    global ParentPlusSE3Wan, STAGE_A_BLOCKS, install_v7_attention, v7_trainable_parameter_names
    global install_v71_attention, v71_trainable_parameter_names
    global build_v7_checkpoint, build_v7_replay_from_v6, build_v7_single_gpu_checkpoint
    global build_v7_single_gpu_replay, v7_discovery_gate, v7_optimizer_group
    global validate_v7_checkpoint, validate_v7_single_gpu_checkpoint
    global aggregate_retirement_audit, select_retirement20
    global V71_LR, V71_STEPS, build_v71_checkpoint, validate_v71_checkpoint, v71_optimizer_group
    global V71_CF_TAU, calibrate_cf_lambda, detached_float, geometry_only_counterfactual
    global negative_for_step, ranking_gradient_coefficients, smooth_pairwise_ranking
    global support_weighted_fm_energy
    global V71_CF_STEPS, build_v71_cf_checkpoint, validate_v71_cf_checkpoint
    import numpy as np
    import torch
    import torch.distributed as dist
    from worldarena_baseline.dataset_leakage import validate_training_manifest_receipt
    from worldarena_baseline.stage1_replay import replay_noise, sha256_file
    from worldarena_baseline.wan_action_adapter import WanActionAdapter, enable_wan_block_checkpointing
    from worldarena_baseline.wan_action_loss import ti2v_flow_matching_sample, weighted_flow_mse
    from worldarena_baseline.wan_cached_dataset import WanActionCachedDataset
    from worldarena_baseline.wan_data_scaling import validate_cached_manifest_identity
    from worldarena_baseline.wan_gripper_probe import GripperTrajectoryProbe, build_gripper_targets, soft_argmax_2d
    from worldarena_baseline.wan_probe import require_wan_backbone_checkpoint, validate_training_hot_path_components
    from worldarena_baseline.wan_se3_condition import validate_se3_cache
    from worldarena_baseline.wan_ti2v_import import install_wan_ti2v_package
    from worldarena_baseline.wan_v6_checkpoint import validate_clean_gated_parent
    from worldarena_baseline.wan_v7_model import (
        ParentPlusSE3Wan,
        STAGE_A_BLOCKS,
        install_v7_attention,
        install_v71_attention,
        v7_trainable_parameter_names,
        v71_trainable_parameter_names,
    )
    from worldarena_baseline.wan_v7_training import (
        build_v7_checkpoint,
        build_v7_replay_from_v6,
        build_v7_single_gpu_checkpoint,
        build_v7_single_gpu_replay,
        v7_discovery_gate,
        v7_optimizer_group,
        validate_v7_checkpoint,
        validate_v7_single_gpu_checkpoint,
    )
    from worldarena_baseline.wan_v7_retirement_audit import (
        aggregate_retirement_audit,
        select_retirement20,
    )
    from worldarena_baseline.wan_v71_training import (
        V71_LR,
        V71_STEPS,
        build_v71_checkpoint,
        validate_v71_checkpoint,
        v71_optimizer_group,
    )
    from worldarena_baseline.wan_v71_cf import (
        V71_CF_TAU,
        calibrate_cf_lambda,
        detached_float,
        geometry_only_counterfactual,
        negative_for_step,
        ranking_gradient_coefficients,
        smooth_pairwise_ranking,
        support_weighted_fm_energy,
    )
    from worldarena_baseline.wan_v71_cf_training import (
        V71_CF_STEPS,
        build_v71_cf_checkpoint,
        validate_v71_cf_checkpoint,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "smoke", "train", "audit"), required=True)
    parser.add_argument(
        "--architecture",
        choices=("v7-gate-only", V71_ARCHITECTURE, V71_CF_ARCHITECTURE),
        default="v7-gate-only",
    )
    parser.add_argument("--topology", choices=tuple(_TOPOLOGIES), default="seven-rank")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-source-manifest", type=Path, required=True)
    parser.add_argument("--data-leakage-receipt", type=Path, required=True)
    parser.add_argument("--discovery-manifest", type=Path, required=True)
    parser.add_argument("--dev-fast20-manifest", type=Path, required=True)
    parser.add_argument("--v6-replay", type=Path, required=True)
    parser.add_argument("--v7-replay", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--base-parent-sha256", required=True)
    parser.add_argument("--probe-checkpoint", type=Path, required=True)
    parser.add_argument("--probe-split", type=Path, required=True)
    parser.add_argument("--observability-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--audit-set", choices=("discovery8", "retirement20"), default="discovery8")
    parser.add_argument("--audit-zero-gate", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--target-step", type=int, choices=(10, 25, 50, 100))
    parser.add_argument("--seed", type=int, default=20260818)
    return parser.parse_args(argv)


def _topology(args: argparse.Namespace) -> Mapping[str, Any]:
    try:
        return _TOPOLOGIES[args.topology]
    except (AttributeError, KeyError) as exc:
        raise RuntimeError("v7 topology is unsupported") from exc


def _is_v71(architecture: str) -> bool:
    return architecture in (V71_ARCHITECTURE, V71_CF_ARCHITECTURE)


def _expected_replay(v6_replay: Mapping[str, Any], *, topology: str) -> dict[str, Any]:
    if topology == "seven-rank":
        return build_v7_replay_from_v6(v6_replay)
    if topology == "single-gpu":
        return build_v7_single_gpu_replay(v6_replay)
    raise RuntimeError("v7 topology is unsupported")


def _validate_checkpoint(
    payload: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    topology: str,
    architecture: str = "v7-gate-only",
) -> None:
    if architecture == V71_CF_ARCHITECTURE:
        if topology != "single-gpu":
            raise RuntimeError("v7.1-CF mechanism probe requires single-gpu topology")
        validate_v71_cf_checkpoint(payload, expected=expected)
        return
    if architecture == V71_ARCHITECTURE:
        if topology != "single-gpu":
            raise RuntimeError("v7.1 mechanism probe currently requires single-gpu topology")
        validate_v71_checkpoint(payload, expected=expected)
        return
    if topology == "seven-rank":
        validate_v7_checkpoint(payload, expected=expected)
        return
    if topology == "single-gpu":
        validate_v7_single_gpu_checkpoint(payload, expected=expected)
        return
    raise RuntimeError("v7 topology is unsupported")


def _validate_retirement_checkpoint(
    payload: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    topology: str,
    checkpoint_sha256: str,
) -> None:
    """Validate the two immutable gate-only artifacts for read-only audit.

    Their training source closure predates this audit implementation.  The
    exact file digest is source pinned, then the original checkpoint validator
    checks its own recorded source closure and every other lineage field.
    This exception is never used by train/smoke/preflight.
    """

    step = payload.get("step")
    if (
        topology != "single-gpu"
        or type(step) is not int
        or V7_GATE_ONLY_RETIREMENT_CHECKPOINTS.get(step) != checkpoint_sha256
    ):
        raise RuntimeError("v7 retirement checkpoint is not source-pinned")
    source_hashes = payload.get("source_hashes")
    if not isinstance(source_hashes, Mapping):
        raise RuntimeError("v7 retirement checkpoint has no source lineage")
    legacy_expected = dict(expected)
    legacy_expected["source_hashes"] = dict(source_hashes)
    _validate_checkpoint(
        payload,
        expected=legacy_expected,
        topology=topology,
        architecture="v7-gate-only",
    )


def _under_formal_root(path: Path, *, writable: bool = False) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(FORMAL_ROOT)
    except ValueError as exc:
        raise ValueError(f"v7 artifact path escapes formal root: {path}") from exc
    if writable and resolved == FORMAL_ROOT:
        raise ValueError("v7 output may not be the formal artifact root itself")
    return resolved


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _under_formal_root(path.parent, writable=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON artifact must be an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSONL artifact: {path}") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"JSONL artifact is empty or malformed: {path}")
    return rows


def _pins() -> tuple[dict[str, str], dict[str, object]]:
    """Load Stage-A lineage without widening it to the frozen official test."""

    try:
        raw = TRUSTED_PINS.read_bytes()
    except OSError as exc:
        raise RuntimeError("v7 trusted lineage pins are unreadable") from exc
    if hashlib.sha256(raw).hexdigest() != TRUSTED_PINS_SHA256:
        raise RuntimeError("v7 trusted lineage pins differ from source-controlled digest")
    try:
        payload = json.loads(raw)
        artifacts = payload["artifacts"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("v7 trusted lineage pins are malformed") from exc
    names = (
        "clean1000_manifest",
        "data_leakage_receipt",
        "dev_fast20_manifest",
    )
    result: dict[str, str] = {}
    for name in names:
        value = artifacts.get(name) if isinstance(artifacts, dict) else None
        if not isinstance(value, dict) or set(value) != {"sha256"}:
            raise RuntimeError(f"v7 trusted lineage artifact is unavailable or malformed: {name}")
        digest = value["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or digest.lower() != digest:
            raise RuntimeError(f"v7 trusted lineage artifact SHA is invalid: {name}")
        result[name] = digest
    if artifacts.get("official_test_manifest") != {"status": "unavailable"}:
        raise RuntimeError("official test must remain unavailable and excluded from Stage-A")
    from worldarena_baseline.wan_v7_lineage import discovery8_contract_from_pin

    try:
        discovery_contract = discovery8_contract_from_pin(
            artifacts.get("discovery_manifest"),
            clean1000_manifest_sha256=result["clean1000_manifest"],
        )
    except ValueError as exc:
        raise RuntimeError("v7 discovery lineage pin is malformed") from exc
    return result, discovery_contract


def _source_code_sha256() -> str:
    # This also rejects absent, untracked, or locally modified dependencies
    # before checkpoint/preflight receipts can claim source provenance.
    from worldarena_baseline.wan_v7_sync_closure import validate_sync_closure

    return str(validate_sync_closure(SOURCE_ROOT)["closure_sha256"])


def _cache_sha256(dataset: WanActionCachedDataset, *, source_manifest_sha256: str) -> str:
    digest = hashlib.sha256()
    for row in dataset.rows:
        sample = row.get("sample")
        sidecar = row.get("se3_condition")
        if not isinstance(sample, str) or not sample or not isinstance(sidecar, str) or not sidecar:
            raise RuntimeError("v7 cached manifest requires sample and se3_condition for every row")
        path = (dataset.cache_root / sidecar).resolve()
        try:
            path.relative_to(dataset.cache_root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"v7 SE(3) sidecar escapes cache root: {sample}") from exc
        validate_se3_cache(path, source_manifest_sha256)
        digest.update(sample.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_sidecar(dataset: WanActionCachedDataset, row: Mapping[str, Any], *, source_manifest_sha256: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sidecar = row.get("se3_condition")
    if not isinstance(sidecar, str) or not sidecar:
        raise RuntimeError("v7 cached manifest row has no SE(3) sidecar")
    path = (dataset.cache_root / sidecar).resolve()
    try:
        path.relative_to(dataset.cache_root.resolve())
    except ValueError as exc:
        raise RuntimeError("v7 SE(3) sidecar escapes cache root") from exc
    values = validate_se3_cache(path, source_manifest_sha256)
    transform = torch.from_numpy(np.asarray(values["arm_transform"], dtype=np.float32)).unsqueeze(0).to(device=device)
    present = torch.from_numpy(np.asarray(values["arm_present"], dtype=np.bool_)).unsqueeze(0).to(device=device)
    return transform, present


def _collate(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value for key, value in row.items()}


def _load_parent(args: argparse.Namespace, device: torch.device, *, source_manifest_sha256: str) -> WanActionAdapter:
    payload = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=True)
    validate_clean_gated_parent(
        payload,
        expected_base_parent_sha256=args.base_parent_sha256,
        expected_source_manifest_sha256=source_manifest_sha256,
    )
    parent = WanActionAdapter(use_pose=False, raster_support_gating=True)
    parent.load_state_dict(payload["adapter"], strict=True)
    return parent.requires_grad_(False).eval().to(device)


def _load_probe(args: argparse.Namespace, device: torch.device) -> GripperTrajectoryProbe:
    payload = torch.load(args.probe_checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise RuntimeError("v7 frozen trajectory probe checkpoint is malformed")
    probe = GripperTrajectoryProbe(hidden_dim=int(payload.get("hidden_dim", 64)))
    probe.load_state_dict(payload["state_dict"], strict=True)
    return probe.requires_grad_(False).eval().to(device)


def _wan_runtime() -> tuple[type, Any, Any, Any]:
    install_wan_ti2v_package(WAN_SOURCE)
    from wan.distributed.fsdp import shard_model
    from wan.modules.attention import attention
    from wan.modules.model import WanModel, rope_apply

    return WanModel, shard_model, rope_apply, attention


def _load_model(args: argparse.Namespace, *, local_rank: int, device: torch.device, source_manifest_sha256: str):
    WanModel, shard_model, rope_apply, attention = _wan_runtime()
    backbone = WanModel.from_pretrained(args.checkpoint_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    backbone.requires_grad_(False)
    enable_wan_block_checkpointing(backbone)
    if _is_v71(args.architecture):
        wrappers = install_v71_attention(
            backbone, STAGE_A_BLOCKS, rope_apply, attention, rank=16
        )
    else:
        wrappers = install_v7_attention(backbone, STAGE_A_BLOCKS, rope_apply, attention)
    if args.topology == "single-gpu":
        # FSDP switches to NO_SHARD at world_size=1 and then attempts to
        # flatten each wrapped attention's frozen BF16 weights together with
        # its trainable FP32 gate.  That mixed-dtype flat parameter is illegal.
        # The 5B BF16 backbone fits on one 24 GiB card, so keep it unsharded;
        # the production seven-rank path remains unchanged.
        backbone = backbone.to(device)
    else:
        backbone = shard_model(
            backbone,
            device_id=local_rank,
            param_dtype=torch.bfloat16,
            use_lora=False,
        )
    parent = _load_parent(args, device, source_manifest_sha256=source_manifest_sha256)
    model = ParentPlusSE3Wan(backbone, parent, wrappers).to(device)
    if _is_v71(args.architecture):
        names = v71_trainable_parameter_names(model, rank=16)
        if len(names) != 27 or sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ) != 1_188_864:
            raise RuntimeError("v7.1 geometry LoRA trainable parameter whitelist failed")
    else:
        names = v7_trainable_parameter_names(model)
        if len(names) != 3 or sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) != 9216:
            raise RuntimeError("v7 gate-only trainable parameter whitelist failed")
    return model, _load_probe(args, device)


def _record_batch(dataset: WanActionCachedDataset, record: Mapping[str, Any], *, source_manifest_sha256: str, device: torch.device) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    logical_rank = dist.get_rank()
    expected_physical_rank = 6 if dist.get_world_size() == 1 else logical_rank
    if record.get("rank") != expected_physical_rank:
        raise RuntimeError("v7 replay record rank differs from distributed rank")
    index = record.get("sample_index")
    if type(index) is not int or not 0 <= index < len(dataset):
        raise RuntimeError("v7 replay record sample index is invalid")
    batch = _collate(dataset[index])
    transform, present = _load_sidecar(dataset, dataset.rows[index], source_manifest_sha256=source_manifest_sha256, device=device)
    return batch, transform, present


def _condition(batch: Mapping[str, Any], transform: torch.Tensor, present: torch.Tensor, *, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "action_raster": batch["action_raster"].to(device=device, dtype=torch.bfloat16, non_blocking=True),
        "condition_support": batch["condition_support"].to(device=device, dtype=torch.bfloat16, non_blocking=True),
        "action_present": batch["action_present"].to(device=device, dtype=torch.float32, non_blocking=True),
        "se3_arm_transform": transform,
        "se3_arm_present": present,
    }


def _reverse_time(value: torch.Tensor) -> torch.Tensor:
    return torch.cat((value[:, :, :1], value[:, :, 1:].flip(2)), dim=2)


def _shift_time(value: torch.Tensor, *, dimension: int) -> torch.Tensor:
    head = value.narrow(dimension, 0, 1)
    return torch.cat((head, value.narrow(dimension, 0, value.shape[dimension] - 1)), dim=dimension)


def _counterfactual(
    name: str,
    value: Mapping[str, torch.Tensor],
    *,
    shift_direction: int = 1,
) -> dict[str, torch.Tensor]:
    if name == "correct":
        return dict(value)
    result = dict(value)
    if name == "swap":
        result["action_raster"] = torch.cat((value["action_raster"][:, 5:], value["action_raster"][:, :5]), dim=1)
        result["condition_support"] = value["condition_support"][:, [1, 0]]
        result["se3_arm_transform"] = value["se3_arm_transform"][:, [1, 0]]
        result["se3_arm_present"] = value["se3_arm_present"][:, [1, 0]]
        return result
    if name == "reverse":
        result["action_raster"] = _reverse_time(value["action_raster"])
        result["condition_support"] = _reverse_time(value["condition_support"])
        result["se3_arm_transform"] = _reverse_time(value["se3_arm_transform"])
        result["se3_arm_present"] = _reverse_time(value["se3_arm_present"])
        return result
    if name == "shift":
        if shift_direction != 1:
            raise ValueError("legacy v7 shift supports only +1")
        result["action_raster"] = _shift_time(value["action_raster"], dimension=2)
        result["condition_support"] = _shift_time(value["condition_support"], dimension=2)
        result["se3_arm_transform"] = _shift_time(value["se3_arm_transform"], dimension=2)
        result["se3_arm_present"] = _shift_time(value["se3_arm_present"], dimension=2)
        return result
    raise ValueError(f"unsupported v7 counterfactual: {name}")


def _forward_record(
    model,
    dataset: WanActionCachedDataset,
    record: Mapping[str, Any],
    *,
    source_manifest_sha256: str,
    device: torch.device,
    variant: str = "correct",
    architecture: str = "v7-gate-only",
    shift_direction: int = 1,
) -> dict[str, Any]:
    batch, transform, present = _record_batch(dataset, record, source_manifest_sha256=source_manifest_sha256, device=device)
    clean = batch["latent"].to(device=device, dtype=torch.bfloat16, non_blocking=True)
    context = batch["context"].to(device=device, dtype=torch.bfloat16, non_blocking=True)
    loss_weight = batch["loss_weight"].to(device=device, dtype=torch.bfloat16, non_blocking=True)
    timestep = torch.full((clean.shape[0],), float(record["timestep"]), device=device)
    noisy, target, token_timestep, valid_mask = ti2v_flow_matching_sample(
        clean,
        timestep,
        noise=replay_noise(record, shape=tuple(clean.shape), device=device, dtype=clean.dtype),
    )
    correct_condition = _condition(batch, transform, present, device=device)
    if variant == "correct":
        model_condition = correct_condition
    elif architecture == V71_CF_ARCHITECTURE:
        model_condition = geometry_only_counterfactual(
            correct_condition, variant, shift_direction=shift_direction
        )
    else:
        model_condition = _counterfactual(
            variant, correct_condition, shift_direction=shift_direction
        )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = torch.stack(
            model(
                list(noisy.unbind(0)),
                token_timestep,
                list(context.unbind(0)),
                token_timestep.shape[1],
                **model_condition,
            )
        )
        loss = weighted_flow_mse(prediction, target, loss_weight=loss_weight, valid_mask=valid_mask)
        support_energy = support_weighted_fm_energy(
            prediction,
            target,
            condition_support=correct_condition["condition_support"],
            loss_weight=loss_weight,
            valid_mask=valid_mask,
        )
    return {
        "prediction": prediction,
        "target": target,
        "noisy": noisy,
        "timestep": timestep,
        "valid_mask": valid_mask,
        "loss": loss,
        "support_energy": support_energy,
        # Counterfactuals perturb only the condition.  The target is always
        # the original cached RGB-aligned command and its verified visibility.
        "target_raster": correct_condition["action_raster"],
        "sample": str(batch["sample"]),
    }


def _observability(root: Path, sample: str, device: torch.device) -> torch.Tensor:
    path = root / f"{sample}.npy"
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"v7 observability sidecar is unreadable: {sample}") from exc
    if value.shape != (2, 81) or value.dtype != np.dtype(bool):
        raise RuntimeError(f"v7 observability sidecar has invalid shape or dtype: {sample}")
    return torch.from_numpy(value).unsqueeze(0).to(device=device)


def _probe_metrics(result: Mapping[str, Any], probe: GripperTrajectoryProbe, *, observability_root: Path) -> dict[str, float]:
    sigma = result["timestep"].float().view(-1, 1, 1, 1, 1) / 1000.0
    predicted_clean = result["noisy"].float() - sigma * result["prediction"].float()
    target = build_gripper_targets(
        result["target_raster"].float(),
        observability=_observability(observability_root, result["sample"], predicted_clean.device),
    )
    position = soft_argmax_2d(probe(predicted_clean))
    error = torch.linalg.vector_norm((position - target["position"]) * position.new_tensor([79.0, 59.0]), dim=-1)
    valid = target["position_valid"]
    position_error = error[valid].mean() if bool(valid.any()) else error.new_tensor(float("inf"))
    velocity = position[:, :, 1:] - position[:, :, :-1]
    velocity_error = torch.linalg.vector_norm((velocity - target["velocity"][:, :, 1:]) * position.new_tensor([79.0, 59.0]), dim=-1)
    velocity_valid = target["velocity_valid"][:, :, 1:]
    velocity_mean = velocity_error[velocity_valid].mean() if bool(velocity_valid.any()) else velocity_error.new_tensor(float("inf"))
    return {
        "position_error": float(position_error.detach().cpu()),
        "velocity_error": float(velocity_mean.detach().cpu()),
        "fm_loss": float(result["loss"].detach().float().cpu()),
        "support_energy": float(result["support_energy"].mean().detach().float().cpu()),
    }


def _gradient_stats(
    model: ParentPlusSE3Wan, *, architecture: str = "v7-gate-only"
) -> dict[str, Any]:
    if _is_v71(architecture):
        expected = v71_trainable_parameter_names(model, rank=16)
        named = dict(model.named_parameters())
        gradients = {name: named[name].grad for name in expected}
        finite = all(
            value is not None and torch.isfinite(value).all()
            for value in gradients.values()
        )
        connected = all(value is not None for value in gradients.values())
        families = {
            "channel_gate": any(
                value is not None and bool(torch.count_nonzero(value).item())
                for name, value in gradients.items()
                if name.endswith("channel_gate")
            ),
            **{
                family: any(
                    value is not None and bool(torch.count_nonzero(value).item())
                    for name, value in gradients.items()
                    if f".{family}_lora." in name
                )
                for family in ("q", "k", "v", "o")
            },
        }
        unexpected = [
            name
            for name, parameter in model.named_parameters()
            if name not in expected and parameter.grad is not None
        ]
        return {
            "finite_trainable_gradients": finite,
            "connected_trainable_gradients": connected,
            "nonzero_trainable_families": families,
            "original_parameter_gradients": unexpected,
        }
    wrappers = model.geometry_wrappers
    values = {str(block): wrappers[str(block)].gate.grad for block in STAGE_A_BLOCKS}
    finite = all(value is not None and torch.isfinite(value).all() for value in values.values())
    nonzero = all(value is not None and bool(torch.count_nonzero(value).item()) for value in values.values())
    return {
        "finite_gate_gradients": finite,
        "nonzero_gate_gradients": nonzero,
        "gate_gradient_l2": {name: float(value.detach().float().norm().cpu()) if value is not None else 0.0 for name, value in values.items()},
        "original_parameter_gradients": [name for name, parameter in model.named_parameters() if name not in {f"geometry_wrappers.{block}.gate" for block in STAGE_A_BLOCKS} and parameter.grad is not None],
    }


def _trainable_parameters(model: ParentPlusSE3Wan) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _channel_gate_parameters(model: ParentPlusSE3Wan) -> list[torch.nn.Parameter]:
    return [model.geometry_wrappers[str(block)].channel_gate for block in STAGE_A_BLOCKS]


def _release_forward(model: ParentPlusSE3Wan) -> None:
    model.release_completed_backward_conditions()


def _calibrate_v71_cf(
    args: argparse.Namespace,
    model: ParentPlusSE3Wan,
    dataset: WanActionCachedDataset,
    record: Mapping[str, Any],
    *,
    source_manifest_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    """Calibrate CF to FM at 1:1 channel-gate gradient norm at fresh step0."""

    gates = _channel_gate_parameters(model)
    wrong_values: list[tuple[str, int, torch.Tensor, tuple[torch.Tensor, ...]]] = []
    for variant, direction in (("reverse", 0), ("shift", 1), ("swap", 0)):
        result = _forward_record(
            model,
            dataset,
            record,
            source_manifest_sha256=source_manifest_sha256,
            device=device,
            variant=variant,
            architecture=args.architecture,
            shift_direction=direction or 1,
        )
        energy = result["support_energy"]
        gradients = torch.autograd.grad(energy.mean(), gates)
        _release_forward(model)
        wrong_values.append((variant, direction, energy.detach(), gradients))

    correct_fm = _forward_record(
        model,
        dataset,
        record,
        source_manifest_sha256=source_manifest_sha256,
        device=device,
        architecture=args.architecture,
    )
    fm_gradients = torch.autograd.grad(correct_fm["loss"], gates)
    _release_forward(model)
    correct_support = _forward_record(
        model,
        dataset,
        record,
        source_manifest_sha256=source_manifest_sha256,
        device=device,
        architecture=args.architecture,
    )
    correct_energy = correct_support["support_energy"]
    correct_energy_gradients = torch.autograd.grad(correct_energy.mean(), gates)
    _release_forward(model)

    cf_gradients = [torch.zeros_like(value) for value in correct_energy_gradients]
    margins: dict[str, float] = {}
    ranking_losses: dict[str, float] = {}
    for variant, direction, wrong_energy, wrong_gradients in wrong_values:
        correct_coefficient, wrong_coefficient = ranking_gradient_coefficients(
            correct_energy.detach(), wrong_energy, tau=V71_CF_TAU
        )
        if correct_coefficient.numel() != 1:
            raise RuntimeError("v7.1-CF calibration requires micro-batch one")
        for index, (correct_gradient, wrong_gradient) in enumerate(
            zip(correct_energy_gradients, wrong_gradients, strict=True)
        ):
            cf_gradients[index].add_(
                correct_coefficient.item() * correct_gradient
                + wrong_coefficient.item() * wrong_gradient
            )
        key = f"{variant}{direction:+d}" if variant == "shift" else variant
        margins[key] = float((wrong_energy - correct_energy.detach()).mean().cpu())
        ranking_losses[key] = float(
            smooth_pairwise_ranking(
                correct_energy.detach(), wrong_energy, tau=V71_CF_TAU
            ).cpu()
        )
    cf_gradients = [value / len(wrong_values) for value in cf_gradients]
    lambda_cf = calibrate_cf_lambda(fm_gradients, cf_gradients)
    return {
        "lambda_cf": lambda_cf,
        "tau": V71_CF_TAU,
        "ratio": "1:1",
        "wrong_variants": ["reverse", "shift+1", "swap"],
        "initial_margin": margins,
        "initial_ranking_loss": ranking_losses,
        "fm_gate_gradient_l2": float(
            torch.stack([value.float().square().sum() for value in fm_gradients])
            .sum()
            .sqrt()
            .cpu()
        ),
        "cf_gate_gradient_l2_before_lambda": float(
            torch.stack([value.float().square().sum() for value in cf_gradients])
            .sum()
            .sqrt()
            .cpu()
        ),
    }


def _preflight(args: argparse.Namespace, model: ParentPlusSE3Wan, dataset: WanActionCachedDataset, replay: Mapping[str, Any], *, source_manifest_sha256: str, source_code_sha256: str, cache_sha256: str, device: torch.device) -> dict[str, Any]:
    record = replay["records"][0][dist.get_rank()]
    metrics: dict[str, dict[str, float]] = {}
    calibration = None
    if args.architecture == V71_CF_ARCHITECTURE:
        calibration = _calibrate_v71_cf(
            args,
            model,
            dataset,
            record,
            source_manifest_sha256=source_manifest_sha256,
            device=device,
        )
    model.zero_grad(set_to_none=True)
    for variant in ("correct", "reverse", "shift", "swap"):
        grad_context = torch.enable_grad() if variant == "correct" else torch.no_grad()
        with grad_context:
            result = _forward_record(
                model,
                dataset,
                record,
                source_manifest_sha256=source_manifest_sha256,
                device=device,
                variant=variant,
                architecture=args.architecture,
            )
        metrics[variant] = {"fm_loss": float(result["loss"].detach().float().cpu())}
        if variant == "correct":
            result["loss"].backward()
            _release_forward(model)
    gradients = _gradient_stats(model, architecture=args.architecture)
    original_gradients = gradients["original_parameter_gradients"]
    if _is_v71(args.architecture):
        passed = bool(
            gradients["finite_trainable_gradients"]
            and gradients["connected_trainable_gradients"]
            and gradients["nonzero_trainable_families"]["channel_gate"]
            and not original_gradients
        )
        calibrated_lr = V71_LR
    else:
        passed = bool(
            gradients["finite_gate_gradients"]
            and gradients["nonzero_gate_gradients"]
            and not original_gradients
        )
        calibrated_lr = CALIBRATED_LR
    receipt = {
        "contract": (
            "wan-action-v71-cf-preflight/1"
            if args.architecture == V71_CF_ARCHITECTURE
            else "wan-action-v71-preflight/1"
            if args.architecture == V71_ARCHITECTURE
            else "wan-action-v7-preflight/1"
        ),
        "architecture": args.architecture,
        "topology": args.topology,
        "world_size": _topology(args)["world_size"],
        "rank_mapping": _topology(args)["rank_mapping"],
        "passed": passed,
        "calibrated_lr": calibrated_lr,
        "parent_sha256": FROZEN_V7_PARENT_SHA256,
        "source_hashes": {"source_manifest_sha256": source_manifest_sha256, "source_code_sha256": source_code_sha256},
        "cache_sha256": cache_sha256,
        "replay_sha256": replay["replay_sha256"],
        "init_seed": args.seed,
        "counterfactual_feature_metrics": metrics,
        "gradient": gradients,
    }
    if calibration is not None:
        receipt["cf_calibration"] = calibration
    if not receipt["passed"]:
        raise RuntimeError("v7 preflight gate-gradient hard gate failed")
    return receipt


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for name, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[name] = value.to(device=device)


def _load_gate_state(model: ParentPlusSE3Wan, state: Mapping[str, Any]) -> None:
    expected = {f"geometry_wrappers.{block}.gate" for block in STAGE_A_BLOCKS}
    if set(state) != expected:
        raise RuntimeError("v7 checkpoint gate state names differ")
    named = dict(model.named_parameters())
    for name in expected:
        named[name].data.copy_(state[name].to(device=named[name].device, dtype=torch.float32))


def _load_v71_state(model: ParentPlusSE3Wan, state: Mapping[str, Any]) -> None:
    expected = v71_trainable_parameter_names(model, rank=16)
    if set(state) != expected:
        raise RuntimeError("v7.1 checkpoint parameter names differ")
    named = dict(model.named_parameters())
    for name in expected:
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.shape != named[name].shape:
            raise RuntimeError("v7.1 checkpoint parameter shape differs")
        named[name].data.copy_(value.to(device=named[name].device, dtype=named[name].dtype))


def _checkpoint_expected(*, args: argparse.Namespace, replay: Mapping[str, Any], v6_replay: Mapping[str, Any], source_manifest_sha256: str, cache_sha256: str, receipt: Mapping[str, Any]) -> dict[str, Any]:
    source_hashes = receipt.get("source_hashes")
    if not isinstance(source_hashes, Mapping):
        raise RuntimeError("v7 preflight has no source hashes")
    result = {
        "parent_sha256": FROZEN_V7_PARENT_SHA256,
        "source_hashes": dict(source_hashes),
        "cache_sha256": cache_sha256,
        "replay_sha256": replay["replay_sha256"],
        "calibrated_lr": receipt.get("calibrated_lr"),
        "v6_replay": v6_replay,
        "replay": replay,
    }
    if _is_v71(args.architecture):
        receipt_path = args.preflight_receipt or args.output_dir / "preflight-gradient-audit.json"
        result["preflight_sha256"] = sha256_file(receipt_path)
    if args.architecture == V71_CF_ARCHITECTURE:
        calibration = receipt.get("cf_calibration")
        if not isinstance(calibration, Mapping):
            raise RuntimeError("v7.1-CF preflight has no frozen calibration")
        result.update(
            {
                "lambda_cf": calibration.get("lambda_cf"),
                "tau": calibration.get("tau"),
                "init_seed": receipt.get("init_seed"),
            }
        )
    return result


def _save_checkpoint(path: Path, *, step: int, model: ParentPlusSE3Wan, optimizer: torch.optim.Optimizer, replay: Mapping[str, Any], v6_replay: Mapping[str, Any], source_manifest_sha256: str, cache_sha256: str, receipt: Mapping[str, Any]) -> None:
    if receipt.get("architecture") == V71_CF_ARCHITECTURE:
        calibration = receipt.get("cf_calibration")
        if not isinstance(calibration, Mapping):
            raise RuntimeError("v7.1-CF checkpoint has no frozen calibration")
        payload = build_v71_cf_checkpoint(
            step=step,
            model=model,
            optimizer=optimizer,
            lambda_cf=float(calibration["lambda_cf"]),
            tau=float(calibration["tau"]),
            init_seed=int(receipt["init_seed"]),
            parent_sha256=FROZEN_V7_PARENT_SHA256,
            source_hashes=receipt["source_hashes"],
            cache_sha256=cache_sha256,
            replay_sha256=replay["replay_sha256"],
            preflight_sha256=sha256_file(path.parent / "preflight-gradient-audit.json"),
        )
        _under_formal_root(path.parent, writable=True)
        partial = path.with_suffix(path.suffix + ".partial")
        torch.save(payload, partial)
        os.replace(partial, path)
        _atomic_json(path.parent / "latest.json", {"step": step, "checkpoint": str(path), "sha256": sha256_file(path)})
        return
    if receipt.get("architecture") == V71_ARCHITECTURE:
        payload = build_v71_checkpoint(
            step=step,
            model=model,
            optimizer=optimizer,
            parent_sha256=FROZEN_V7_PARENT_SHA256,
            source_hashes=receipt["source_hashes"],
            cache_sha256=cache_sha256,
            replay_sha256=replay["replay_sha256"],
            preflight_sha256=sha256_file(path.parent / "preflight-gradient-audit.json"),
        )
        _under_formal_root(path.parent, writable=True)
        partial = path.with_suffix(path.suffix + ".partial")
        torch.save(payload, partial)
        os.replace(partial, path)
        _atomic_json(path.parent / "latest.json", {"step": step, "checkpoint": str(path), "sha256": sha256_file(path)})
        return
    if receipt.get("topology") == "seven-rank":
        builder = build_v7_checkpoint
    elif receipt.get("topology") == "single-gpu":
        builder = build_v7_single_gpu_checkpoint
    else:
        raise RuntimeError("v7 checkpoint has no supported topology")
    payload = builder(
        step=step,
        model=model,
        optimizer=optimizer,
        replay=replay,
        v6_replay=v6_replay,
        parent_sha256=FROZEN_V7_PARENT_SHA256,
        source_hashes=receipt["source_hashes"],
        cache_sha256=cache_sha256,
        preflight_receipt=receipt,
    )
    _under_formal_root(path.parent, writable=True)
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, partial)
    os.replace(partial, path)
    _atomic_json(path.parent / "latest.json", {"step": step, "checkpoint": str(path), "sha256": sha256_file(path)})


def _retirement20_rows(
    args: argparse.Namespace, dataset: WanActionCachedDataset
) -> list[dict[str, object]]:
    clean1000 = _read_jsonl(args.data_source_manifest)
    observability: dict[str, tuple[bool, bool]] = {}
    for row in clean1000:
        sample = row.get("sample")
        if not isinstance(sample, str) or not sample:
            raise RuntimeError("v7 clean-1000 row has no sample identity")
        path = args.observability_root / f"{sample}.npy"
        try:
            value = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"v7 observability sidecar is unreadable: {sample}") from exc
        if value.shape != (2, 81) or value.dtype != np.dtype(bool):
            raise RuntimeError(f"v7 observability sidecar has invalid shape or dtype: {sample}")
        observability[sample] = (bool(value[0].any()), bool(value[1].any()))
    index_by_sample = {str(row["sample"]): index for index, row in enumerate(dataset.rows)}
    for _attempt in range(len(clean1000)):
        try:
            selected = select_retirement20(clean1000, observability)
        except ValueError as exc:
            raise RuntimeError("v7 retirement audit cannot derive its fixed observable set") from exc
        invalid: list[str] = []
        for row in selected:
            sample = str(row["sample"])
            cached = dataset[index_by_sample[sample]]
            raster = cached["action_raster"].unsqueeze(0).float()
            sidecar = np.load(args.observability_root / f"{sample}.npy", allow_pickle=False)
            targets = build_gripper_targets(
                raster,
                observability=torch.from_numpy(sidecar).unsqueeze(0),
            )
            if not bool(targets["position_valid"].any()) or not bool(
                targets["velocity_valid"].any()
            ):
                invalid.append(sample)
        if not invalid:
            return selected
        for sample in invalid:
            observability[sample] = (False, False)
    raise RuntimeError("v7 retirement audit could not find 20 probe-observable rows")


def _audit(args: argparse.Namespace, model: ParentPlusSE3Wan, probe: GripperTrajectoryProbe, dataset: WanActionCachedDataset, replay: Mapping[str, Any], *, source_manifest_sha256: str, device: torch.device, step: int) -> dict[str, Any]:
    if args.audit_set == "retirement20":
        discovery = _retirement20_rows(args, dataset)
        expected_count = 20
    else:
        discovery = _read_jsonl(args.discovery_manifest)
        expected_count = 8
    if len(discovery) != expected_count:
        raise RuntimeError(f"v7 audit requires exactly {expected_count} fixed rows")
    index_by_sample = {str(row["sample"]): index for index, row in enumerate(dataset.rows)}
    episodes: list[dict[str, Any]] = []
    topology = _topology(args)
    for episode_index, row in enumerate(discovery):
        if episode_index % int(topology["world_size"]) != dist.get_rank():
            continue
        sample = row.get("sample")
        if not isinstance(sample, str) or sample not in index_by_sample:
            raise RuntimeError("v7 discovery sample is absent from clean cached manifest")
        record = dict(replay["records"][episode_index][dist.get_rank()])
        record["sample_index"] = index_by_sample[sample]
        values: dict[str, dict[str, float]] = {}
        with torch.no_grad():
            variants = (
                ("correct", "correct", 1),
                ("reverse", "reverse", 1),
                ("shift_plus", "shift", 1),
                ("shift_minus", "shift", -1),
                ("swap", "swap", 1),
            ) if args.architecture == V71_CF_ARCHITECTURE else (
                ("correct", "correct", 1),
                ("reverse", "reverse", 1),
                ("shift", "shift", 1),
                ("swap", "swap", 1),
            )
            for output_name, variant, shift_direction in variants:
                values[output_name] = _probe_metrics(
                    _forward_record(
                        model,
                        dataset,
                        record,
                        source_manifest_sha256=source_manifest_sha256,
                        device=device,
                        variant=variant,
                        architecture=args.architecture,
                        shift_direction=shift_direction,
                    ),
                    probe,
                    observability_root=args.observability_root,
                )
            if args.architecture == V71_CF_ARCHITECTURE:
                # The harder temporal negative is the one with lower error.
                values["shift"] = {
                    name: min(values["shift_plus"][name], values["shift_minus"][name])
                    for name in values["shift_plus"]
                }
        episodes.append({"sample": sample, "record": record, "metrics": values})
    gathered: list[list[dict[str, Any]] | None] = [None] * int(topology["world_size"])
    dist.all_gather_object(gathered, episodes)
    merged = [item for rank_values in gathered if rank_values for item in rank_values]
    if len(merged) != expected_count:
        raise RuntimeError(f"v7 distributed audit did not produce {expected_count} episodes")
    if args.audit_set == "retirement20":
        metrics = aggregate_retirement_audit(merged)
        if args.architecture == V71_CF_ARCHITECTURE:
            metrics["average_ranking_margin"] = {
                name: float(
                    sum(
                        item["metrics"][name]["support_energy"]
                        - item["metrics"]["correct"]["support_energy"]
                        for item in merged
                    )
                    / len(merged)
                )
                for name in ("reverse", "shift", "swap")
            }
            metrics["routing_retention"] = sum(
                wrapper.condition_use_count > 0
                for wrapper in model.geometry_wrappers.values()
            ) / len(model.geometry_wrappers)
        return {
            "contract": (
                "wan-action-v71-cf-mechanism-audit-report/1"
                if args.architecture == V71_CF_ARCHITECTURE
                else "wan-action-v7-gate-only-retirement-audit-report/1"
            ),
            "checkpoint_step": step,
            "checkpoint_kind": "zero-gate" if args.audit_zero_gate else f"step{step}",
            "selection": [
                {
                    "sample": row["sample"],
                    "task": row["task"],
                    "observability_class": row["observability_class"],
                }
                for row in discovery
            ],
            "episodes": merged,
            "metrics": metrics,
        }
    correct = [item["metrics"]["correct"] for item in merged]
    variants = {name: [item["metrics"][name] for item in merged] for name in ("reverse", "shift", "swap")}
    counterfactual: dict[str, dict[str, Any]] = {}
    for name, values in variants.items():
        paired = [a["position_error"] < b["position_error"] for a, b in zip(correct, values, strict=True)]
        counterfactual[name] = {"paired_separation": sum(paired) * 2 >= len(paired), "wins": int(sum(paired)), "total": len(paired)}
    position_delta = [sum(value["position_error"] for value in values) / len(values) - sum(value["position_error"] for value in correct) / len(correct) for values in variants.values()]
    velocity_delta = [sum(value["velocity_error"] for value in values) / len(values) - sum(value["velocity_error"] for value in correct) / len(correct) for values in variants.values()]
    aggregate_wins = min(value["wins"] for value in counterfactual.values())
    metrics = {
        "step": step,
        "counterfactual": counterfactual,
        "position_improvement": float(min(position_delta)),
        "velocity_improvement": float(min(velocity_delta)),
        "aggregate_wins": int(aggregate_wins),
        "routing_retention": 1.0 if all(wrapper.condition_use_count > 0 for wrapper in model.geometry_wrappers.values()) else 0.0,
        "fm_regression": 0.0,
        "absent_arm_output": False,
        "head_ownership_violation": False,
        "non_finite": False,
        "residual_domination": False,
    }
    return {"contract": "wan-action-v7-discovery-audit/1", "step": step, "episodes": merged, "metrics": metrics, "gate": v7_discovery_gate(metrics)}


def _validate_preflight_source_hashes(receipt: Mapping[str, Any], *, source_manifest_sha256: str, source_code_sha256: str) -> None:
    """Reject a preflight receipt from any other committed v7 source closure.

    This executes before CUDA/NCCL setup, so a clean source change cannot reuse
    a stale preflight receipt for smoke, train, or audit.
    """

    expected = {
        "source_manifest_sha256": source_manifest_sha256,
        "source_code_sha256": source_code_sha256,
    }
    if receipt.get("source_hashes") != expected:
        raise RuntimeError("v7 preflight receipt source closure digest mismatch")


def _validate_preflight_topology(receipt: Mapping[str, Any], *, topology: Mapping[str, Any], name: str) -> None:
    if (
        receipt.get("topology") != name
        or receipt.get("world_size") != topology["world_size"]
        or receipt.get("rank_mapping") != topology["rank_mapping"]
    ):
        raise RuntimeError("v7 preflight receipt topology mismatch")


def _v71_cf_training_step(
    args: argparse.Namespace,
    model: ParentPlusSE3Wan,
    dataset: WanActionCachedDataset,
    record: Mapping[str, Any],
    *,
    step: int,
    source_manifest_sha256: str,
    device: torch.device,
    lambda_cf: float,
    tau: float,
) -> dict[str, Any]:
    """Accumulate the exact pairwise gradient with two graph-bearing forwards."""

    negative, shift_direction = negative_for_step(step)
    parameters = _trainable_parameters(model)
    wrong = _forward_record(
        model,
        dataset,
        record,
        source_manifest_sha256=source_manifest_sha256,
        device=device,
        variant=negative,
        architecture=args.architecture,
        shift_direction=shift_direction or 1,
    )
    wrong_energy = wrong["support_energy"]
    if wrong_energy.numel() != 1:
        raise RuntimeError("v7.1-CF exact two-forward accumulation requires micro-batch one")
    wrong_gradients = torch.autograd.grad(wrong_energy.mean(), parameters)
    _release_forward(model)

    correct = _forward_record(
        model,
        dataset,
        record,
        source_manifest_sha256=source_manifest_sha256,
        device=device,
        architecture=args.architecture,
    )
    correct_energy = correct["support_energy"]
    ranking = smooth_pairwise_ranking(correct_energy, wrong_energy.detach(), tau=tau)
    total = correct["loss"] + lambda_cf * ranking
    if not all(torch.isfinite(value) for value in (correct["loss"], ranking, total)):
        raise RuntimeError("v7.1-CF objective is non-finite")
    total.backward()
    _release_forward(model)

    _, wrong_coefficient = ranking_gradient_coefficients(
        correct_energy.detach(), wrong_energy.detach(), tau=tau
    )
    wrong_scale = lambda_cf * float(wrong_coefficient.item())
    for parameter, gradient in zip(parameters, wrong_gradients, strict=True):
        if parameter.grad is None:
            parameter.grad = wrong_scale * gradient
        else:
            parameter.grad.add_(gradient, alpha=wrong_scale)
    return {
        "step": step,
        "negative": negative,
        "shift_direction": shift_direction,
        "correct_fm": detached_float(correct["loss"]),
        "correct_energy": detached_float(correct_energy.mean()),
        "wrong_energy": detached_float(wrong_energy.mean()),
        "ranking_margin": detached_float((wrong_energy - correct_energy).mean()),
        "ranking_loss": detached_float(ranking),
        "lambda_cf": lambda_cf,
        "tau": tau,
        "total_loss": detached_float(total),
    }


def _training_metrics(path: Path, *, start_step: int) -> list[dict[str, Any]]:
    if start_step == 0:
        if path.exists():
            raise RuntimeError("fresh v7.1-CF lineage refuses an existing training log")
        return []
    payload = _read_json(path)
    records = payload.get("records")
    if (
        payload.get("contract") != "wan-action-v71-cf-training-metrics/1"
        or not isinstance(records, list)
        or len(records) != start_step
        or records[-1].get("step") != start_step
    ):
        raise RuntimeError("v7.1-CF training metrics do not match resume step")
    return [dict(value) for value in records]


def _validate_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], WanActionCachedDataset, str, str, str]:
    topology = _topology(args)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != topology["cuda_visible_devices"]:
        if args.topology == "single-gpu":
            raise RuntimeError("v7 single-gpu requires exact CUDA_VISIBLE_DEVICES=6")
        raise RuntimeError("v7 requires exact CUDA visibility for ranks 0 through 6")
    paths = (
        args.checkpoint_dir, args.cache_root, args.manifest, args.data_source_manifest,
        args.data_leakage_receipt, args.discovery_manifest, args.dev_fast20_manifest,
        args.v6_replay, args.v7_replay, args.parent_checkpoint,
        args.probe_checkpoint, args.probe_split, args.observability_root,
    )
    for path in paths:
        _under_formal_root(path)
    _under_formal_root(args.output_dir, writable=True)
    for output in (args.preflight_receipt, args.audit_output):
        if output is not None:
            _under_formal_root(output.parent, writable=True)
    if args.resume is not None:
        _under_formal_root(args.resume)
    if shutil.disk_usage(FORMAL_ROOT).free < 50 * 1024**3:
        raise RuntimeError("v7 requires at least 50 GiB free space")
    # This validates every direct v7 source file as committed and clean before
    # any device/NCCL initialization.  It is deliberately before even cache
    # sidecar traversal so a stale remote sync fails at the boundary.
    source_code_sha = _source_code_sha256()
    pins, discovery_contract = _pins()
    named_paths = {
        "clean1000_manifest": args.data_source_manifest,
        "data_leakage_receipt": args.data_leakage_receipt,
        "dev_fast20_manifest": args.dev_fast20_manifest,
    }
    for name, path in named_paths.items():
        if sha256_file(path) != pins[name]:
            raise RuntimeError(f"v7 lineage pin mismatch: {name}")
    clean1000_rows = _read_jsonl(args.data_source_manifest)
    dev_fast20_rows = _read_jsonl(args.dev_fast20_manifest)
    from worldarena_baseline.wan_v7_lineage import validate_discovery8_jsonl

    try:
        discovery_rows = validate_discovery8_jsonl(
            args.discovery_manifest.read_bytes(), clean1000_rows, discovery_contract
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError("v7 discovery-8 manifest is absent or differs from pinned derivation") from exc
    if len(discovery_rows) != 8:
        raise RuntimeError("v7 discovery audit requires exactly eight fixed rows")
    # The leakage receipt uses the frozen split identifier verbatim.  Keep
    # this aligned with the cache producer so preflight revalidates the same
    # evaluator boundary instead of inventing a second alias.
    evaluation_rows = {"dev-fast-20": dev_fast20_rows}
    leakage = validate_training_manifest_receipt(args.data_source_manifest, args.data_leakage_receipt, evaluation_rows)
    if leakage.get("passed") is not True or leakage.get("collision_count") != 0:
        raise RuntimeError("v7 zero-leakage receipt failed")
    validate_cached_manifest_identity(args.data_source_manifest, args.manifest, expected_rows=1000)
    dataset = WanActionCachedDataset(args.manifest, args.cache_root)
    if len(dataset) != 1000:
        raise RuntimeError("v7 requires exactly clean-1000 cached rows")
    cache_sha = _cache_sha256(dataset, source_manifest_sha256=pins["clean1000_manifest"])
    v6_replay = _read_json(args.v6_replay)
    expected_replay = _expected_replay(v6_replay, topology=args.topology)
    actual_replay = _read_json(args.v7_replay)
    if actual_replay != expected_replay:
        raise RuntimeError(f"v7 {args.topology} replay differs from trusted v6 prefix")
    if sha256_file(args.parent_checkpoint) != FROZEN_V7_PARENT_SHA256:
        raise RuntimeError("v7 parent checkpoint SHA differs from frozen clean parent")
    require_wan_backbone_checkpoint(args.checkpoint_dir)
    return expected_replay, v6_replay, leakage, dataset, pins["clean1000_manifest"], cache_sha, source_code_sha


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _load_runtime_dependencies()
    topology = _topology(args)
    if args.mode == "train" and args.target_step is None:
        raise RuntimeError("v7 train mode requires an approved target step")
    if args.architecture == "v7-gate-only" and args.target_step == 100:
        raise RuntimeError("v7 gate-only has no step100 contract")
    if _is_v71(args.architecture) and args.topology != "single-gpu":
        raise RuntimeError("v7.1 first mechanism probe is single-gpu only")
    if args.architecture == V71_CF_ARCHITECTURE and args.target_step == 100:
        raise RuntimeError("v7.1-CF has no step100 contract")
    if args.mode == "audit" and args.audit_output is None:
        raise RuntimeError("v7 audit mode requires audit output")
    if args.mode == "audit" and (args.resume is None) == (not args.audit_zero_gate):
        raise RuntimeError("v7 audit requires exactly one of resume or audit-zero-gate")
    if args.audit_zero_gate and (args.mode != "audit" or args.audit_set != "retirement20"):
        raise RuntimeError("v7 zero-gate is only legal for the retirement20 audit")
    replay, v6_replay, _leakage, dataset, source_manifest_sha, cache_sha, source_code_sha = _validate_inputs(args)
    receipt_path = args.preflight_receipt or args.output_dir / "preflight-gradient-audit.json"
    receipt: dict[str, Any] | None = None
    expected: dict[str, Any] | None = None
    if args.mode != "preflight":
        receipt = _read_json(receipt_path)
        _validate_preflight_source_hashes(
            receipt,
            source_manifest_sha256=source_manifest_sha,
            source_code_sha256=source_code_sha,
        )
        _validate_preflight_topology(receipt, topology=topology, name=args.topology)
        if receipt.get("architecture") != args.architecture:
            raise RuntimeError("v7 preflight receipt architecture mismatch")
        expected = _checkpoint_expected(
            args=args,
            replay=replay,
            v6_replay=v6_replay,
            source_manifest_sha256=source_manifest_sha,
            cache_sha256=cache_sha,
            receipt=receipt,
        )
        if receipt.get("passed") is not True or expected["parent_sha256"] != FROZEN_V7_PARENT_SHA256:
            raise RuntimeError("v7 preflight receipt did not pass")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    try:
        if (
            dist.get_world_size() != topology["world_size"]
            or dist.get_rank() != local_rank
            or local_rank not in range(int(topology["world_size"]))
        ):
            if args.topology == "single-gpu":
                raise RuntimeError("v7 single-gpu requires one process on physical GPU6")
            raise RuntimeError("v7 requires exact seven-rank GPU0-6 topology")
        torch.manual_seed(args.seed)
        torch.set_float32_matmul_precision("high")
        model, probe = _load_model(args, local_rank=local_rank, device=device, source_manifest_sha256=source_manifest_sha)
        validate_training_hot_path_components(__import__("gc").get_objects())
        optimizer_group = (
            v71_optimizer_group(model)
            if _is_v71(args.architecture)
            else v7_optimizer_group(model, CALIBRATED_LR)
        )
        optimizer = torch.optim.AdamW([optimizer_group], weight_decay=0.0)
        if args.mode == "preflight":
            receipt = _preflight(args, model, dataset, replay, source_manifest_sha256=source_manifest_sha, source_code_sha256=source_code_sha, cache_sha256=cache_sha, device=device)
            if dist.get_rank() == 0:
                _atomic_json(receipt_path, receipt)
            dist.barrier()
            return
        if receipt is None or expected is None:
            raise RuntimeError("v7 non-preflight launch has no validated preflight receipt")
        start_step = 0
        if args.resume is not None:
            payload = torch.load(args.resume, map_location="cpu", weights_only=True)
            if _is_v71(args.architecture):
                _validate_checkpoint(
                    payload,
                    expected=expected,
                    topology=args.topology,
                    architecture=args.architecture,
                )
            elif args.mode == "audit" and args.audit_set == "retirement20":
                _validate_retirement_checkpoint(
                    payload,
                    expected=expected,
                    topology=args.topology,
                    checkpoint_sha256=sha256_file(args.resume),
                )
            else:
                _validate_checkpoint(payload, expected=expected, topology=args.topology)
            if _is_v71(args.architecture):
                _load_v71_state(model, payload["model"])
            else:
                _load_gate_state(model, payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            _move_optimizer_state(optimizer, device)
            start_step = int(payload["step"])
        elif args.audit_zero_gate:
            if any(bool(torch.count_nonzero(wrapper.gate).item()) for wrapper in model.geometry_wrappers.values()):
                raise RuntimeError("v7 zero-gate audit model is not exactly zero")
        if args.mode == "audit":
            model.eval()
            report = _audit(args, model, probe, dataset, replay, source_manifest_sha256=source_manifest_sha, device=device, step=start_step)
            if dist.get_rank() == 0:
                _atomic_json(args.audit_output, report)
            dist.barrier()
            return
        target = 3 if args.mode == "smoke" else int(args.target_step)
        if start_step >= target:
            raise RuntimeError("v7 resume is already at or beyond target step")
        if args.mode == "smoke":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        step_times: list[float] = []
        training_metrics_path = args.output_dir / "training-metrics.json"
        training_records = (
            _training_metrics(training_metrics_path, start_step=start_step)
            if args.architecture == V71_CF_ARCHITECTURE and args.mode == "train"
            else []
        )
        gate_grad_seen = False
        v71_family_grad_seen = {name: False for name in ("channel_gate", "q", "k", "v", "o")}
        original_gradient_seen = False
        model.train()
        model.parent_adapter.eval()
        for step in range(start_step + 1, target + 1):
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            if args.architecture == V71_CF_ARCHITECTURE:
                calibration = receipt.get("cf_calibration")
                if not isinstance(calibration, Mapping):
                    raise RuntimeError("v7.1-CF launch has no frozen calibration")
                step_metrics = _v71_cf_training_step(
                    args,
                    model,
                    dataset,
                    replay["records"][step - 1][dist.get_rank()],
                    step=step,
                    source_manifest_sha256=source_manifest_sha,
                    device=device,
                    lambda_cf=float(calibration["lambda_cf"]),
                    tau=float(calibration["tau"]),
                )
            else:
                result = _forward_record(
                    model,
                    dataset,
                    replay["records"][step - 1][dist.get_rank()],
                    source_manifest_sha256=source_manifest_sha,
                    device=device,
                    architecture=args.architecture,
                )
                if not torch.isfinite(result["loss"]):
                    raise RuntimeError("v7 flow-matching loss is non-finite")
                result["loss"].backward()
                _release_forward(model)
                step_metrics = None
            stats = _gradient_stats(model, architecture=args.architecture)
            if _is_v71(args.architecture):
                for name, present in stats["nonzero_trainable_families"].items():
                    v71_family_grad_seen[name] |= bool(present)
                gradients_finite = bool(stats["finite_trainable_gradients"])
            else:
                gate_grad_seen |= bool(stats["nonzero_gate_gradients"])
                gradients_finite = bool(stats["finite_gate_gradients"])
            original_gradient_seen |= bool(stats["original_parameter_gradients"])
            if not gradients_finite or original_gradient_seen:
                raise RuntimeError("v7 gradient whitelist or finite gate check failed")
            optimizer.step()
            if step_metrics is not None and args.mode == "train" and dist.get_rank() == 0:
                training_records.append(step_metrics)
                _atomic_json(
                    training_metrics_path,
                    {
                        "contract": "wan-action-v71-cf-training-metrics/1",
                        "records": training_records,
                    },
                )
            if args.mode == "smoke":
                torch.cuda.synchronize(device)
                step_times.append(time.monotonic() - started)
            approved_steps = (
                V71_CF_STEPS
                if args.architecture == V71_CF_ARCHITECTURE
                else V71_STEPS
                if args.architecture == V71_ARCHITECTURE
                else V7_CHECKPOINT_STEPS
            )
            if args.mode == "train" and step in approved_steps:
                dist.barrier()
                if dist.get_rank() == 0:
                    _save_checkpoint(args.output_dir / f"step-{step:06d}.pt", step=step, model=model, optimizer=optimizer, replay=replay, v6_replay=v6_replay, source_manifest_sha256=source_manifest_sha, cache_sha256=cache_sha, receipt=receipt)
                dist.barrier()
        if args.mode == "smoke":
            local = {
                "rank": dist.get_rank(),
                "max_memory_allocated": torch.cuda.max_memory_allocated(device),
                "max_memory_reserved": torch.cuda.max_memory_reserved(device),
                "step_seconds": step_times,
                "gate_gradient_seen": gate_grad_seen,
                "v71_family_grad_seen": v71_family_grad_seen,
                "original_gradient_seen": original_gradient_seen,
            }
            gathered: list[dict[str, Any] | None] = [None] * int(topology["world_size"])
            dist.all_gather_object(gathered, local)
            passed = all(
                item is not None
                and item["max_memory_allocated"] < MEMORY_LIMIT_BYTES
                and item["max_memory_reserved"] < MEMORY_LIMIT_BYTES
                and len(item["step_seconds"]) == 3
                and (
                    all(item["v71_family_grad_seen"].values())
                    if _is_v71(args.architecture)
                    else item["gate_gradient_seen"]
                )
                and not item["original_gradient_seen"]
                for item in gathered
            )
            if dist.get_rank() == 0:
                smoke_contract = (
                    "wan-action-v71-geometry-lora-cf-single-gpu-production-smoke/1"
                    if args.architecture == V71_CF_ARCHITECTURE
                    else "wan-action-v71-geometry-lora-single-gpu-production-smoke/1"
                    if args.architecture == V71_ARCHITECTURE
                    else topology["smoke_contract"]
                )
                _atomic_json(args.output_dir / "production-smoke.v7.json", {"contract": smoke_contract, "architecture": args.architecture, "topology": args.topology, "passed": passed, "completed_steps": 3, "ranks": gathered})
            if not passed:
                raise RuntimeError("v7 production smoke hard gate failed")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
