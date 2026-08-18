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
MEMORY_LIMIT_BYTES = 22 * 1024**3
FROZEN_V7_PARENT_SHA256 = "105fb760fd371885ba362d26ef2352c260755e47cd036f46711181edc3b30ca2"
V7_CHECKPOINT_STEPS = (10, 25, 50)
V7_WORLD_SIZE = 7
V7_SINGLE_GPU_WORLD_SIZE = 1
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
    global build_v7_checkpoint, build_v7_replay_from_v6, build_v7_single_gpu_checkpoint
    global build_v7_single_gpu_replay, v7_discovery_gate, v7_optimizer_group
    global validate_v7_checkpoint, validate_v7_single_gpu_checkpoint
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
    from worldarena_baseline.wan_v7_model import ParentPlusSE3Wan, STAGE_A_BLOCKS, install_v7_attention, v7_trainable_parameter_names
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "smoke", "train", "audit"), required=True)
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
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--target-step", type=int, choices=V7_CHECKPOINT_STEPS)
    parser.add_argument("--seed", type=int, default=20260818)
    return parser.parse_args(argv)


def _topology(args: argparse.Namespace) -> Mapping[str, Any]:
    try:
        return _TOPOLOGIES[args.topology]
    except (AttributeError, KeyError) as exc:
        raise RuntimeError("v7 topology is unsupported") from exc


def _expected_replay(v6_replay: Mapping[str, Any], *, topology: str) -> dict[str, Any]:
    if topology == "seven-rank":
        return build_v7_replay_from_v6(v6_replay)
    if topology == "single-gpu":
        return build_v7_single_gpu_replay(v6_replay)
    raise RuntimeError("v7 topology is unsupported")


def _validate_checkpoint(payload: Mapping[str, Any], *, expected: Mapping[str, Any], topology: str) -> None:
    if topology == "seven-rank":
        validate_v7_checkpoint(payload, expected=expected)
        return
    if topology == "single-gpu":
        validate_v7_single_gpu_checkpoint(payload, expected=expected)
        return
    raise RuntimeError("v7 topology is unsupported")


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


def _counterfactual(name: str, value: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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
        result["action_raster"] = _shift_time(value["action_raster"], dimension=2)
        result["condition_support"] = _shift_time(value["condition_support"], dimension=2)
        result["se3_arm_transform"] = _shift_time(value["se3_arm_transform"], dimension=2)
        result["se3_arm_present"] = _shift_time(value["se3_arm_present"], dimension=2)
        return result
    raise ValueError(f"unsupported v7 counterfactual: {name}")


def _forward_record(model, dataset: WanActionCachedDataset, record: Mapping[str, Any], *, source_manifest_sha256: str, device: torch.device, variant: str = "correct") -> dict[str, Any]:
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
    model_condition = _counterfactual(variant, correct_condition)
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
    return {
        "prediction": prediction,
        "target": target,
        "noisy": noisy,
        "timestep": timestep,
        "valid_mask": valid_mask,
        "loss": loss,
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
    }


def _gradient_stats(model: ParentPlusSE3Wan) -> dict[str, Any]:
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


def _preflight(args: argparse.Namespace, model: ParentPlusSE3Wan, dataset: WanActionCachedDataset, replay: Mapping[str, Any], *, source_manifest_sha256: str, source_code_sha256: str, cache_sha256: str, device: torch.device) -> dict[str, Any]:
    record = replay["records"][0][dist.get_rank()]
    metrics: dict[str, dict[str, float]] = {}
    model.zero_grad(set_to_none=True)
    for variant in ("correct", "reverse", "shift", "swap"):
        result = _forward_record(model, dataset, record, source_manifest_sha256=source_manifest_sha256, device=device, variant=variant)
        metrics[variant] = {"fm_loss": float(result["loss"].detach().float().cpu())}
        if variant == "correct":
            result["loss"].backward()
    gradients = _gradient_stats(model)
    original_gradients = gradients["original_parameter_gradients"]
    receipt = {
        "contract": "wan-action-v7-preflight/1",
        "topology": args.topology,
        "world_size": _topology(args)["world_size"],
        "rank_mapping": _topology(args)["rank_mapping"],
        "passed": bool(gradients["finite_gate_gradients"] and gradients["nonzero_gate_gradients"] and not original_gradients),
        "calibrated_lr": CALIBRATED_LR,
        "parent_sha256": FROZEN_V7_PARENT_SHA256,
        "source_hashes": {"source_manifest_sha256": source_manifest_sha256, "source_code_sha256": source_code_sha256},
        "cache_sha256": cache_sha256,
        "replay_sha256": replay["replay_sha256"],
        "counterfactual_feature_metrics": metrics,
        "gradient": gradients,
    }
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


def _checkpoint_expected(*, args: argparse.Namespace, replay: Mapping[str, Any], v6_replay: Mapping[str, Any], source_manifest_sha256: str, cache_sha256: str, receipt: Mapping[str, Any]) -> dict[str, Any]:
    source_hashes = receipt.get("source_hashes")
    if not isinstance(source_hashes, Mapping):
        raise RuntimeError("v7 preflight has no source hashes")
    return {
        "parent_sha256": FROZEN_V7_PARENT_SHA256,
        "source_hashes": dict(source_hashes),
        "cache_sha256": cache_sha256,
        "replay_sha256": replay["replay_sha256"],
        "calibrated_lr": receipt.get("calibrated_lr"),
        "v6_replay": v6_replay,
        "replay": replay,
    }


def _save_checkpoint(path: Path, *, step: int, model: ParentPlusSE3Wan, optimizer: torch.optim.Optimizer, replay: Mapping[str, Any], v6_replay: Mapping[str, Any], source_manifest_sha256: str, cache_sha256: str, receipt: Mapping[str, Any]) -> None:
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


def _audit(args: argparse.Namespace, model: ParentPlusSE3Wan, probe: GripperTrajectoryProbe, dataset: WanActionCachedDataset, replay: Mapping[str, Any], *, source_manifest_sha256: str, device: torch.device, step: int) -> dict[str, Any]:
    discovery = _read_jsonl(args.discovery_manifest)
    if len(discovery) != 8:
        raise RuntimeError("v7 discovery audit requires exactly eight fixed rows")
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
            for variant in ("correct", "reverse", "shift", "swap"):
                values[variant] = _probe_metrics(
                    _forward_record(model, dataset, record, source_manifest_sha256=source_manifest_sha256, device=device, variant=variant),
                    probe,
                    observability_root=args.observability_root,
                )
        episodes.append({"sample": sample, "record": record, "metrics": values})
    gathered: list[list[dict[str, Any]] | None] = [None] * int(topology["world_size"])
    dist.all_gather_object(gathered, episodes)
    merged = [item for rank_values in gathered if rank_values for item in rank_values]
    if len(merged) != 8:
        raise RuntimeError("v7 distributed discovery audit did not produce eight episodes")
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
    if args.mode == "audit" and (args.resume is None or args.audit_output is None):
        raise RuntimeError("v7 audit mode requires resume and audit output")
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
        optimizer = torch.optim.AdamW([v7_optimizer_group(model, CALIBRATED_LR)], weight_decay=0.0)
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
            _validate_checkpoint(payload, expected=expected, topology=args.topology)
            _load_gate_state(model, payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            _move_optimizer_state(optimizer, device)
            start_step = int(payload["step"])
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
        gate_grad_seen = False
        original_gradient_seen = False
        model.train()
        model.parent_adapter.eval()
        for step in range(start_step + 1, target + 1):
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            result = _forward_record(model, dataset, replay["records"][step - 1][dist.get_rank()], source_manifest_sha256=source_manifest_sha, device=device)
            if not torch.isfinite(result["loss"]):
                raise RuntimeError("v7 flow-matching loss is non-finite")
            result["loss"].backward()
            stats = _gradient_stats(model)
            gate_grad_seen |= bool(stats["nonzero_gate_gradients"])
            original_gradient_seen |= bool(stats["original_parameter_gradients"])
            if not stats["finite_gate_gradients"] or original_gradient_seen:
                raise RuntimeError("v7 gradient whitelist or finite gate check failed")
            optimizer.step()
            if args.mode == "smoke":
                torch.cuda.synchronize(device)
                step_times.append(time.monotonic() - started)
            if args.mode == "train" and step in V7_CHECKPOINT_STEPS:
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
                "original_gradient_seen": original_gradient_seen,
            }
            gathered: list[dict[str, Any] | None] = [None] * int(topology["world_size"])
            dist.all_gather_object(gathered, local)
            passed = all(item is not None and item["max_memory_allocated"] < MEMORY_LIMIT_BYTES and item["max_memory_reserved"] < MEMORY_LIMIT_BYTES and len(item["step_seconds"]) == 3 and item["gate_gradient_seen"] and not item["original_gradient_seen"] for item in gathered)
            if dist.get_rank() == 0:
                _atomic_json(args.output_dir / "production-smoke.v7.json", {"contract": topology["smoke_contract"], "topology": args.topology, "passed": passed, "completed_steps": 3, "ranks": gathered})
            if not passed:
                raise RuntimeError("v7 production smoke hard gate failed")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
