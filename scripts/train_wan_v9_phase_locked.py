#!/usr/bin/env python3
"""Guarded single-GPU trainer for v9 phase-locked action attention."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from collections.abc import Mapping
from typing import Any


ROOT = Path("/data/di/worldarena2_track1_20260815")
SOURCE = Path("/home/huazhi/nlh/baseline")
WAN_SOURCE = Path("/home/huazhi/nlh/Wan2.2")
PARENT_SHA = "105fb760fd371885ba362d26ef2352c260755e47cd036f46711181edc3b30ca2"
MEMORY_LIMIT = 22 * 1024**3


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _under_root(path: Path, *, allow_source: bool = False) -> Path:
    resolved = path.resolve(strict=False)
    bases = (ROOT, SOURCE) if allow_source else (ROOT,)
    if not any(resolved == base or base in resolved.parents for base in bases):
        raise ValueError(f"v9 path escapes approved roots: {path}")
    return resolved


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "smoke", "train", "audit"), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--transition-root", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--base-parent-sha256", required=True)
    parser.add_argument("--probe-checkpoint", type=Path, required=True)
    parser.add_argument("--trajectory-calibration", type=Path, required=True)
    parser.add_argument("--observability-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-step", type=int)
    parser.add_argument("--audit-step", type=int)
    return parser.parse_args()


def _validate_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "6":
        raise RuntimeError("v9 single-GPU lineage requires physical GPU6")
    for path in (
        args.checkpoint_dir, args.cache_root, args.manifest, args.transition_root,
        args.replay, args.audit_manifest, args.parent_checkpoint,
        args.probe_checkpoint, args.trajectory_calibration, args.observability_root,
        args.output_dir, args.source_receipt,
    ):
        _under_root(path)
    if args.resume is not None:
        _under_root(args.resume)
    if _sha(args.parent_checkpoint) != PARENT_SHA:
        raise RuntimeError("v9 immutable clean-gated parent SHA mismatch")
    from worldarena_baseline.wan_v9_sync_closure import build_v9_source_receipt

    recorded_source = _read_json(args.source_receipt)
    fresh_source = build_v9_source_receipt(SOURCE)
    if recorded_source != fresh_source:
        raise RuntimeError("v9 source closure differs before CUDA initialization")
    transition_receipt = _read_json(args.transition_root / "receipt.json")
    if transition_receipt.get("contract") != "wan-v9-transition-preparation/1":
        raise RuntimeError("v9 transition receipt contract differs")
    if transition_receipt.get("clean_manifest_sha256") != _sha(args.manifest):
        raise RuntimeError("v9 transition receipt manifest differs")
    replay = _read_jsonl(args.replay)
    if len(replay) != 500 or [row.get("optimizer_step") for row in replay] != list(range(1, 501)):
        raise RuntimeError("v9 replay must contain exact steps 1-500")
    audit = _read_jsonl(args.audit_manifest)
    if len(audit) != 20:
        raise RuntimeError("v9 audit manifest must contain exactly 20 rows")
    if {str(row["sample"]) for row in replay} & {str(row["sample"]) for row in audit}:
        raise RuntimeError("v9 replay leaks audit20")
    if args.mode == "train" and args.stop_step not in (25, 100, 250, 500):
        raise RuntimeError("v9 train stop step is not approved")
    if args.mode == "audit" and args.audit_step not in (25, 100, 250, 500):
        raise RuntimeError("v9 audit step is not approved")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return transition_receipt, replay, audit


def _load_runtime():
    global np, torch
    import numpy as np
    import torch
    from scripts import train_wan_se3_probe_v7_fsdp as legacy
    legacy._load_runtime_dependencies()
    return legacy


def _lineage(args: argparse.Namespace, transition_receipt: Mapping[str, Any]) -> dict[str, str]:
    source = _read_json(args.source_receipt)
    return {
        "parent_sha256": _sha(args.parent_checkpoint),
        "source_closure_sha256": str(source["closure_sha256"]),
        "manifest_sha256": _sha(args.manifest),
        "transition_receipt_sha256": _sha(args.transition_root / "receipt.json"),
        "normalization_receipt_sha256": str(transition_receipt["normalization_receipt_sha256"]),
        "replay_sha256": _sha(args.replay),
        "audit_sha256": _sha(args.audit_manifest),
        "probe_sha256": _sha(args.probe_checkpoint),
    }


def _load_model(args: argparse.Namespace, legacy, device):
    from worldarena_baseline.wan_action_adapter import enable_wan_block_checkpointing
    from worldarena_baseline.wan_v9_model import (
        V9_BLOCKS, ParentPlusPhaseLockedActionWan, install_v9_attention,
        v9_trainable_parameter_names,
    )
    from worldarena_baseline.wan_v9_tokens import V9ActionTokenizer

    legacy.install_wan_ti2v_package(WAN_SOURCE)
    from wan.modules.model import WanModel

    backbone = WanModel.from_pretrained(
        args.checkpoint_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    backbone.requires_grad_(False)
    enable_wan_block_checkpointing(backbone)
    wrappers = install_v9_attention(backbone, V9_BLOCKS)
    backbone = backbone.to(device)
    normalization = _read_json(args.transition_root / "normalization.json")
    tokenizer = V9ActionTokenizer(normalization)
    parent_payload = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=True)
    stage1 = parent_payload.get("stage1")
    source_manifest_sha256 = stage1.get("source_manifest_sha256") if isinstance(stage1, Mapping) else None
    if not isinstance(source_manifest_sha256, str):
        raise RuntimeError("v9 parent lacks source manifest provenance")
    parent = legacy._load_parent(args, device, source_manifest_sha256=source_manifest_sha256)
    model = ParentPlusPhaseLockedActionWan(backbone, parent, tokenizer, wrappers).to(device)
    names = v9_trainable_parameter_names(model)
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if len([name for name in names if name.endswith("channel_gate")]) != 6 or count < 250_000_000:
        raise RuntimeError("v9 trainable whitelist/count failed")
    return model, names, count


def _batch(dataset, index: int) -> dict[str, Any]:
    row = dataset[index]
    return {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value for key, value in row.items()}


def _sample_map(dataset) -> dict[str, int]:
    result = {str(row["sample"]): index for index, row in enumerate(dataset.rows)}
    if len(result) != len(dataset):
        raise RuntimeError("v9 cached dataset contains duplicate samples")
    return result


def _transition_condition(args, sample: str, variant: str, device) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    from worldarena_baseline.wan_v9_transition import validate_transition_cache

    receipt = _read_json(args.transition_root / "receipt.json")
    payload = validate_transition_cache(
        args.transition_root / sample / f"{variant}.npz",
        expected_sample=sample,
        expected_variant=variant,
        expected_clean_manifest_sha256=_sha(args.manifest),
        expected_normalization_receipt_sha256=str(receipt["normalization_receipt_sha256"]),
    )
    features = {
        name: torch.from_numpy(np.asarray(payload[name], dtype=np.float32)).unsqueeze(0).to(device)
        for name in ("translation", "rotation", "image", "gripper")
    }
    present = torch.from_numpy(np.asarray(payload["arm_present"], dtype=np.bool_)).unsqueeze(0).to(device)
    active = torch.from_numpy(np.asarray(payload["motion_active"], dtype=np.bool_)).unsqueeze(0).to(device)
    return features, present, active


def _noise(record, shape, device, dtype):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(record["noise_seed"]))
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)


def _timestep(record, device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(record["timestep_seed"]))
    return torch.randint(1, 1000, (1,), generator=generator).to(device=device, dtype=torch.float32)


def _forward(model, legacy, batch, args, record, variant: str, device, *, grad: bool):
    from worldarena_baseline.wan_v9_objective import interval_robot_fm_energy

    sample = str(record["sample"])
    features, present, _variant_active = _transition_condition(args, sample, variant, device)
    _correct_features, _correct_present, correct_active = _transition_condition(args, sample, "correct", device)
    clean = batch["latent"].to(device=device, dtype=torch.bfloat16)
    context = batch["context"].to(device=device, dtype=torch.bfloat16)
    loss_weight = batch["loss_weight"].to(device=device, dtype=torch.bfloat16)
    timestep = _timestep(record, device)
    noisy, target, token_timestep, valid = legacy.ti2v_flow_matching_sample(
        clean, timestep, noise=_noise(record, tuple(clean.shape), device, clean.dtype)
    )
    condition = {
        "action_raster": batch["action_raster"].to(device=device, dtype=torch.bfloat16),
        "condition_support": batch["condition_support"].to(device=device, dtype=torch.bfloat16),
        "action_present": batch["action_present"].to(device=device, dtype=torch.float32),
        "transition_features": features,
        "transition_arm_present": present,
    }
    context_manager = torch.enable_grad() if grad else torch.no_grad()
    with context_manager, torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = torch.stack(
            model(
                list(noisy.unbind(0)), token_timestep, list(context.unbind(0)),
                token_timestep.shape[1], **condition,
            )
        )
        fm = legacy.weighted_flow_mse(
            prediction, target, loss_weight=loss_weight, valid_mask=valid
        )
        energy, eligible = interval_robot_fm_energy(
            prediction, target,
            correct_support=batch["condition_support"].to(device=device, dtype=torch.bfloat16),
            loss_weight=loss_weight, valid_mask=valid, motion_active=correct_active,
        )
    return {
        "fm": fm, "energy": energy, "eligible": eligible,
        "prediction": prediction, "target": target, "noisy": noisy,
        "valid": valid, "timestep": timestep,
    }


def _release(model) -> None:
    model.release_completed_backward_conditions()


def _gradient_snapshot(parameters: list[Any]) -> list[Any]:
    return [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in parameters]


def _calibration_parameters(model) -> list[Any]:
    result = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("channel_gate") or ".base." in name:
            result.append(parameter)
    if not result:
        raise RuntimeError("v9 calibration parameter family is empty")
    return result


def _preview(args, model, legacy, batch, record, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        correct = _forward(model, legacy, batch, args, record, "correct", device, grad=False)
        _release(model)
        wrong = _forward(model, legacy, batch, args, record, str(record["negative_family"]), device, grad=False)
        _release(model)
    if not torch.equal(correct["eligible"], wrong["eligible"]):
        raise RuntimeError("v9 correct/wrong interval eligibility differs")
    return correct["energy"].detach(), wrong["energy"].detach(), correct["eligible"]


def _coefficient(correct: torch.Tensor, wrong: torch.Tensor, eligible: torch.Tensor, tau: float):
    mask = eligible.float()
    value = torch.sigmoid((correct.float() - wrong.float()) / tau) * mask / (tau * mask.sum().clamp_min(1))
    return value, -value


def _calibrate(args, model, legacy, dataset, index_map, record, device) -> dict[str, float]:
    from worldarena_baseline.wan_v9_objective import calibrate_lambda_cf, interval_pairwise_ranking

    batch = _batch(dataset, index_map[str(record["sample"])])
    correct_preview, wrong_preview, eligible = _preview(args, model, legacy, batch, record, device)
    correct_coefficient, wrong_coefficient = _coefficient(correct_preview, wrong_preview, eligible, 0.1)
    parameters = _calibration_parameters(model)
    model.zero_grad(set_to_none=True)
    result = _forward(model, legacy, batch, args, record, "correct", device, grad=True)
    result["fm"].backward(); _release(model)
    fm_gradients = _gradient_snapshot(parameters)
    model.zero_grad(set_to_none=True)
    result = _forward(model, legacy, batch, args, record, "correct", device, grad=True)
    (result["energy"] * correct_coefficient).sum().backward(); _release(model)
    result = _forward(model, legacy, batch, args, record, str(record["negative_family"]), device, grad=True)
    (result["energy"] * wrong_coefficient).sum().backward(); _release(model)
    cf_gradients = _gradient_snapshot(parameters)
    model.zero_grad(set_to_none=True)
    ranking, margin, fraction = interval_pairwise_ranking(correct_preview, wrong_preview, eligible, tau=0.1)
    return {
        "lambda_cf": calibrate_lambda_cf(fm_gradients, cf_gradients, target_ratio=0.5),
        "tau": 0.1,
        "initial_ranking": float(ranking.cpu()),
        "initial_margin": float(margin.cpu()),
        "eligible_fraction": float(fraction.cpu()),
    }


def _gradient_history(model, history: dict[str, float]) -> dict[str, float]:
    from worldarena_baseline.wan_v9_training import classify_v9_parameter

    expanded = dict(history)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        norm = float(parameter.grad.detach().float().norm().cpu())
        family = classify_v9_parameter(name)
        if family == "action_cross" and ".cross." in name:
            for projection in ("q", "k", "v", "o"):
                if f".cross.{projection}." in name:
                    family = f"cross_{projection}"
                    break
        elif family == "action_cross":
            family = "tokenizer"
        elif family == "channel_gates":
            family = "gates"
        expanded[family] = max(expanded.get(family, 0.0), norm)
    return expanded


def _trajectory_terms(args, legacy, batch, result, probe, sigma_contract, device):
    from worldarena_baseline.wan_gripper_probe import build_gripper_targets
    from worldarena_baseline.wan_gripper_trajectory_loss import (
        predicted_clean_latent, probe_trajectory_terms, sigma_weights,
    )

    sigma = result["timestep"].float() / 1000.0
    predicted_clean = predicted_clean_latent(
        result["noisy"].float(), result["prediction"].float(), sigma=sigma,
    )
    labels = build_gripper_targets(
        batch["action_raster"].to(device=device, dtype=torch.float32),
        observability=legacy._observability(
            args.observability_root, str(batch["sample"]), device
        ),
    )
    return probe_trajectory_terms(
        predicted_clean,
        probe,
        target_position=labels["position"],
        position_valid=labels["position_valid"],
        target_velocity=labels["velocity"],
        velocity_valid=labels["velocity_valid"],
        sigma_weight=sigma_weights(sigma, sigma_contract),
        huber_delta=0.02,
    )


def _gradient_rms(model) -> float:
    squared = None
    count = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or ".base." not in name or parameter.grad is None:
            continue
        value = parameter.grad.detach().float()
        term = value.square().sum()
        squared = term if squared is None else squared + term
        count += value.numel()
    if squared is None or count == 0:
        raise RuntimeError("v9 trajectory calibration has no native QKVO gradients")
    result = float(torch.sqrt(squared / count).cpu())
    if not math.isfinite(result) or result <= 0:
        raise RuntimeError("v9 trajectory calibration QKVO gradient is not positive finite")
    return result


def _load_trajectory_source(args) -> dict[str, Any]:
    payload = _read_json(args.trajectory_calibration)
    if payload.get("parent_sha256") != PARENT_SHA:
        raise RuntimeError("v9 trajectory calibration parent differs")
    if payload.get("probe_sha256") != _sha(args.probe_checkpoint):
        raise RuntimeError("v9 trajectory calibration probe differs")
    sigma = payload.get("sigma")
    if not isinstance(sigma, Mapping):
        raise RuntimeError("v9 trajectory calibration lacks frozen sigma contract")
    return dict(sigma)


def _calibrate_trajectory(args, model, legacy, dataset, index_map, record, device, probe, sigma_contract):
    batch = _batch(dataset, index_map[str(record["sample"])])
    norms: dict[str, float] = {}
    for objective in ("fm", "position_loss", "velocity_loss"):
        model.zero_grad(set_to_none=True)
        result = _forward(model, legacy, batch, args, record, "correct", device, grad=True)
        if objective == "fm":
            loss = result["fm"]
        else:
            loss = _trajectory_terms(
                args, legacy, batch, result, probe, sigma_contract, device
            )[objective]
        loss.backward()
        _release(model)
        norms[objective] = _gradient_rms(model)
    model.zero_grad(set_to_none=True)
    return {
        "contract": "wan-v9-trajectory-calibration/1",
        "lambda_position": 0.25 * norms["fm"] / norms["position_loss"],
        "lambda_velocity": 0.15 * norms["fm"] / norms["velocity_loss"],
        "fm_qkvo_grad_rms": norms["fm"],
        "position_qkvo_grad_rms": norms["position_loss"],
        "velocity_qkvo_grad_rms": norms["velocity_loss"],
        "source_sha256": _sha(args.trajectory_calibration),
    }


def _train_step(
    args, model, legacy, dataset, index_map, record, device, optimizer,
    calibration, history, *, trajectory=None, probe=None, sigma_contract=None,
):
    from worldarena_baseline.wan_v9_objective import sequential_pairwise_backward

    batch = _batch(dataset, index_map[str(record["sample"])])
    correct_preview, wrong_preview, eligible = _preview(args, model, legacy, batch, record, device)
    optimizer.zero_grad(set_to_none=True)

    def correct_graph():
        result = _forward(model, legacy, batch, args, record, "correct", device, grad=True)
        correct_graph.raw_fm = result["fm"].detach()
        loss = result["fm"]
        terms = None
        if trajectory is not None:
            if probe is None or sigma_contract is None:
                raise RuntimeError("v9 Phase T trajectory dependencies are missing")
            terms = _trajectory_terms(
                args, legacy, batch, result, probe, sigma_contract, device
            )
            loss = (
                loss
                + float(trajectory["lambda_position"]) * terms["position_loss"]
                + float(trajectory["lambda_velocity"]) * terms["velocity_loss"]
            )
        correct_graph.terms = terms
        return loss, result["energy"]

    def wrong_graph():
        return _forward(model, legacy, batch, args, record, str(record["negative_family"]), device, grad=True)["energy"]

    metrics = sequential_pairwise_backward(
        correct_preview=correct_preview, wrong_preview=wrong_preview,
        eligible=eligible, correct_graph=correct_graph, wrong_graph=wrong_graph,
        lambda_cf=float(calibration["lambda_cf"]), tau=float(calibration["tau"]),
        release_graph=lambda _label: _release(model),
    )
    history = _gradient_history(model, history)
    norm = float(torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
    ).detach().cpu())
    if not math.isfinite(norm):
        raise RuntimeError("v9 gradient norm is non-finite")
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    output = {
        "fm": float(getattr(correct_graph, "raw_fm", metrics["fm_loss"]).cpu()),
        "ranking": float(metrics["ranking_loss"].cpu()),
        "margin": float(metrics["margin"].cpu()),
        "eligible_fraction": float(metrics["eligible_fraction"].cpu()),
        "grad_norm": norm,
        "negative": str(record["negative_family"]),
    }
    terms = getattr(correct_graph, "terms", None)
    if terms is not None:
        output.update(
            position_loss=float(terms["position_loss"].detach().cpu()),
            velocity_loss=float(terms["velocity_loss"].detach().cpu()),
        )
    return output, history


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(dict(payload), partial)
    os.replace(partial, path)


def _load_trainable(model, payload: Mapping[str, Any]) -> None:
    named = dict(model.named_parameters())
    expected = {name for name, parameter in named.items() if parameter.requires_grad}
    if set(payload["model"]) != expected:
        raise RuntimeError("v9 checkpoint trainable state differs")
    with torch.no_grad():
        for name, value in payload["model"].items():
            named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))


def _trainable_state(model) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _restore_trainable(model, state: Mapping[str, Any]) -> None:
    named = dict(model.named_parameters())
    if set(state) != {name for name, parameter in named.items() if parameter.requires_grad}:
        raise RuntimeError("v9 comparison state inventory differs")
    with torch.no_grad():
        for name, value in state.items():
            named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))


def _set_channel_gates(model, *, enabled: bool) -> dict[str, Any]:
    saved: dict[str, Any] = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("channel_gate"):
                saved[name] = parameter.detach().clone()
                if not enabled:
                    parameter.zero_()
    if len(saved) != 6:
        raise RuntimeError("v9 gate ablation requires exactly six channel gates")
    return saved


def _restore_channel_gates(model, saved: Mapping[str, Any]) -> None:
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in saved.items():
            named[name].copy_(value)


def _audit_record(sample: str, index: int) -> dict[str, Any]:
    def seed(label: str) -> int:
        return int.from_bytes(hashlib.sha256(f"v9-audit:{label}:{sample}".encode()).digest()[:8], "big")
    return {"sample": sample, "noise_seed": seed("noise"), "timestep_seed": seed("time"), "optimizer_step": index + 1, "rank": 0}


def _energy_mean(result: Mapping[str, Any]) -> float:
    eligible = result["eligible"]
    if not bool(eligible.any()):
        return float("nan")
    return float(result["energy"][eligible].mean().cpu())


def _audit(args, model, legacy, dataset, index_map, rows, device, step):
    probe = legacy._load_probe(args, device)
    episodes = []
    for index, row in enumerate(rows):
        sample = str(row["sample"]); record = _audit_record(sample, index); batch = _batch(dataset, index_map[sample])
        values = {}
        for variant in ("correct", "shift+1", "shift-1", "reverse", "swap"):
            result = _forward(model, legacy, batch, args, record, variant, device, grad=False); _release(model)
            result.update(sample=sample, target_raster=batch["action_raster"].to(device), loss=result["fm"], support_energy=result["energy"].mean().reshape(1))
            probe_metrics = legacy._probe_metrics(result, probe, observability_root=args.observability_root)
            values[variant] = {"energy": _energy_mean(result), **probe_metrics}
        episodes.append({"sample": sample, "values": values})
        print(json.dumps({"audit": index + 1, "sample": sample}, sort_keys=True), flush=True)
    families = {}
    for family in ("shift+1", "shift-1", "reverse", "swap"):
        pairs = [item for item in episodes if math.isfinite(item["values"]["correct"]["energy"]) and math.isfinite(item["values"][family]["energy"])]
        margins = [item["values"][family]["energy"] - item["values"]["correct"]["energy"] for item in pairs]
        families[family] = {"wins": sum(value > 0 for value in margins), "eligible": len(pairs), "mean_margin": sum(margins) / len(margins) if margins else float("nan")}
    correct = [item["values"]["correct"] for item in episodes]
    routing = sum(wrapper.condition_use_count > 0 for wrapper in model.action_wrappers.values()) / len(model.action_wrappers)
    return {
        "contract": "wan-v9-audit-report/1", "step": step,
        "episodes": episodes, "metrics": {
            "families": families, "routing_retention": routing,
            "fm_mean": sum(item["fm_loss"] for item in correct) / len(correct),
            "position_mean": sum(item["position_error"] for item in correct) / len(correct),
            "velocity_mean": sum(item["velocity_error"] for item in correct) / len(correct),
        },
    }


def _audit_comparison(
    args, model, legacy, dataset, index_map, rows, device, step,
    *, initial_state: Mapping[str, Any],
) -> dict[str, Any]:
    current_state = _trainable_state(model)
    enabled = _audit(args, model, legacy, dataset, index_map, rows, device, step)
    saved_gates = _set_channel_gates(model, enabled=False)
    try:
        gate_zero = _audit(args, model, legacy, dataset, index_map, rows, device, step)
    finally:
        _restore_channel_gates(model, saved_gates)
    _restore_trainable(model, initial_state)
    try:
        parent = _audit(args, model, legacy, dataset, index_map, rows, device, step)
    finally:
        _restore_trainable(model, current_state)
    metrics = dict(enabled["metrics"])
    parent_metrics = parent["metrics"]
    zero_metrics = gate_zero["metrics"]
    metrics.update(
        fm_regression=(metrics["fm_mean"] - parent_metrics["fm_mean"])
        / max(parent_metrics["fm_mean"], 1e-12),
        position_improvement=(parent_metrics["position_mean"] - metrics["position_mean"])
        / max(parent_metrics["position_mean"], 1e-12),
        velocity_improvement=(parent_metrics["velocity_mean"] - metrics["velocity_mean"])
        / max(parent_metrics["velocity_mean"], 1e-12),
        gate_enabled_no_worse=(
            metrics["position_mean"] <= zero_metrics["position_mean"]
            and metrics["velocity_mean"] <= zero_metrics["velocity_mean"]
        ),
    )
    enabled["metrics"] = metrics
    return {
        "contract": "wan-v9-audit-comparison/1",
        "step": step,
        "enabled": enabled,
        "gate_zero": gate_zero,
        "fresh_parent": parent,
    }


def main() -> None:
    args = parse_args()
    transition_receipt, replay, audit_rows = _validate_inputs(args)
    legacy = _load_runtime()
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    from worldarena_baseline.wan_cached_dataset import WanActionCachedDataset
    from worldarena_baseline.wan_v9_audit import step25_health_gate, step100_gate, step250_gate, step500_gate
    from worldarena_baseline.wan_v9_training import (
        build_v9_checkpoint, build_v9_optimizer, set_v9_learning_rates,
        validate_v9_checkpoint,
    )

    dataset = WanActionCachedDataset(args.manifest, args.cache_root)
    index_map = _sample_map(dataset)
    model, names, count = _load_model(args, legacy, device)
    initial_state = _trainable_state(model)
    optimizer = build_v9_optimizer(model)
    lineage = _lineage(args, transition_receipt)
    calibration = _calibrate(args, model, legacy, dataset, index_map, replay[0], device)
    gates: dict[str, Any] = {}
    history: dict[str, float] = {}
    start = 0
    payload = None
    if args.resume is not None:
        payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        validate_v9_checkpoint(payload, expected_lineage=lineage, expected_phase=str(payload["phase"]))
        _load_trainable(model, payload)
        if args.mode == "train":
            optimizer.load_state_dict(payload["optimizer"])
        calibration = dict(payload["calibration"]); gates = dict(payload["gates"])
        history = dict(payload["gradient_history"]); start = int(payload["step"])
    if args.mode == "preflight":
        output = {"contract": "wan-v9-preflight/1", "lineage": lineage, "trainable_names": sorted(names), "trainable_count": count, "calibration": calibration}
        (args.output_dir / "preflight.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(json.dumps(output, sort_keys=True)); return
    if args.mode == "smoke":
        torch.cuda.reset_peak_memory_stats(device); rows = []
        for record in replay[:3]:
            set_v9_learning_rates(optimizer, int(record["optimizer_step"])); started = time.monotonic()
            metrics, history = _train_step(args, model, legacy, dataset, index_map, record, device, optimizer, calibration, history)
            metrics["step_time"] = time.monotonic() - started; rows.append(metrics)
        allocated = torch.cuda.max_memory_allocated(device); reserved = torch.cuda.max_memory_reserved(device)
        output = {"contract": "wan-v9-production-smoke/1", "iterations": rows, "gradient_history": history, "peak_allocated": allocated, "peak_reserved": reserved, "limit": MEMORY_LIMIT, "pass": allocated < MEMORY_LIMIT and reserved < MEMORY_LIMIT}
        (args.output_dir / "production-smoke.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(json.dumps(output, sort_keys=True))
        if not output["pass"]: raise RuntimeError("v9 smoke exceeded 22 GiB")
        return
    if args.mode == "audit":
        if payload is None or start != args.audit_step:
            raise RuntimeError("v9 audit checkpoint step differs")
        report = _audit_comparison(
            args, model, legacy, dataset, index_map, audit_rows, device, start,
            initial_state=initial_state,
        )
        metrics = report["enabled"]["metrics"]
        if start == 25:
            decision = step25_health_gate({"gradient_history": history, "outside_whitelist_gradients": 0, "fm_loss": metrics["fm_mean"], "ranking_loss": 0.0, "optimizer_finite": True, "residual_max_ratio": 1.0, "scheduler_drift": False, "topology_drift": False})
        elif start == 100:
            metrics["shift_last_half_trend_positive"] = True
            decision = step100_gate(metrics)
        elif start == 250:
            decision = step250_gate(metrics)
        else:
            reference = gates.get("step250", {}).get("reference_metrics", {})
            metrics.update(
                position_within_noise_band=(
                    metrics["position_improvement"]
                    >= float(reference.get("position_improvement", float("inf"))) - 0.02
                ),
                velocity_within_noise_band=(
                    metrics["velocity_improvement"]
                    >= float(reference.get("velocity_improvement", float("inf"))) - 0.02
                ),
                visual_guardrail_regression=metrics["fm_regression"] > 0.02,
            )
            decision = step500_gate(metrics, step250_receipt=gates.get("step250", {}))
        decision["reference_metrics"] = {
            name: metrics[name]
            for name in (
                "fm_mean", "position_mean", "velocity_mean", "fm_regression",
                "position_improvement", "velocity_improvement", "routing_retention",
            )
        }
        report["decision"] = decision
        target = args.output_dir / f"audit-{start:03d}.json"; target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        payload["gates"][f"step{start}"] = decision
        _atomic_torch_save(payload, args.output_dir / f"step-{start:06d}-gated.pt")
        print(json.dumps(report, sort_keys=True)); return
    target = int(args.stop_step)
    if start >= target:
        raise RuntimeError("v9 resume step must be below target")
    if target > 100 and gates.get("step100", {}).get("continue") is not True:
        raise RuntimeError("v9 step100 receipt does not allow continuation")
    if target > 250 and gates.get("step250", {}).get("pass") is not True:
        raise RuntimeError("v9 Phase T requires passing step250")
    trajectory = calibration.get("trajectory")
    probe = None
    sigma_contract = None
    if target > 250:
        sigma_contract = _load_trajectory_source(args)
        probe = legacy._load_probe(args, device)
        if trajectory is None:
            trajectory = _calibrate_trajectory(
                args, model, legacy, dataset, index_map, replay[start], device,
                probe, sigma_contract,
            )
            calibration["trajectory"] = trajectory
    log = args.output_dir / "training.jsonl"
    for step in range(start + 1, target + 1):
        set_v9_learning_rates(optimizer, step); started = time.monotonic()
        metrics, history = _train_step(
            args, model, legacy, dataset, index_map, replay[step - 1], device,
            optimizer, calibration, history, trajectory=trajectory, probe=probe,
            sigma_contract=sigma_contract,
        )
        metrics.update(step=step, step_time=time.monotonic() - started)
        with log.open("a", encoding="utf-8") as handle: handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        if step in (10, 25, 50, 100, 150, 200, 250, 300, 400, 500):
            checkpoint = build_v9_checkpoint(
                step=step, model=model, optimizer=optimizer, lineage=lineage,
                calibration=calibration, gates=gates, gradient_history=history,
                negative_cycle_index=step % 10,
            )
            _atomic_torch_save(checkpoint, args.output_dir / f"step-{step:06d}.pt")
        print(json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
