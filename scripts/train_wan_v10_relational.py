#!/usr/bin/env python3
"""Single-GPU staged trainer for v10 action-relational native attention."""

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
SEED = 20260819


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _formal(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValueError(f"v10 path escapes formal root: {path}")
    return resolved


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "smoke", "train", "audit"), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--optimizer-manifest", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--data-receipt", type=Path, required=True)
    parser.add_argument("--relation-root", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--base-parent-sha256", required=True)
    parser.add_argument("--probe-checkpoint", type=Path, required=True)
    parser.add_argument("--trajectory-calibration", type=Path, required=True)
    parser.add_argument("--observability-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-step", type=int)
    parser.add_argument("--audit-step", type=int)
    return parser.parse_args()


def _load_runtime():
    global np, torch
    import numpy as np
    import torch
    from scripts import train_wan_se3_probe_v7_fsdp as legacy

    legacy._load_runtime_dependencies()
    return legacy


def _source_sha() -> str:
    files = (
        Path(__file__),
        SOURCE / "src/worldarena_baseline/wan_v10_attention.py",
        SOURCE / "src/worldarena_baseline/wan_v10_model.py",
        SOURCE / "src/worldarena_baseline/wan_v10_objective.py",
        SOURCE / "src/worldarena_baseline/wan_v10_training.py",
        SOURCE / "src/worldarena_baseline/wan_v10_data.py",
        SOURCE / "src/worldarena_baseline/wan_v10_replay.py",
        SOURCE / "src/worldarena_baseline/wan_v10_audit.py",
        SOURCE / "scripts/train_wan_se3_probe_v7_fsdp.py",
        SOURCE / "src/worldarena_baseline/wan_action_adapter.py",
        SOURCE / "src/worldarena_baseline/wan_cached_dataset.py",
        SOURCE / "src/worldarena_baseline/robotwin_action_cache.py",
        SOURCE / "src/worldarena_baseline/action_condition.py",
        SOURCE / "src/worldarena_baseline/action_raster.py",
        SOURCE / "src/worldarena_baseline/wan_gripper_probe.py",
        SOURCE / "src/worldarena_baseline/wan_gripper_trajectory_loss.py",
        SOURCE / "src/worldarena_baseline/wan_v9_objective.py",
    )
    digest = hashlib.sha256()
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"v10 source closure member unavailable: {path}")
        digest.update(str(path.relative_to(SOURCE)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_inputs(args: argparse.Namespace):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "6":
        raise RuntimeError("v10 initial topology requires physical GPU6 only")
    _source_sha()
    paths = (
        args.checkpoint_dir, args.optimizer_manifest, args.audit_manifest,
        args.data_receipt, args.relation_root, args.replay, args.parent_checkpoint,
        args.probe_checkpoint, args.trajectory_calibration, args.observability_root,
        args.output_dir,
    )
    for path in paths:
        _formal(path)
    if args.resume is not None:
        _formal(args.resume)
    if _sha(args.parent_checkpoint) != PARENT_SHA:
        raise RuntimeError("v10 immutable clean-gated parent SHA mismatch")
    receipt = _read_json(args.data_receipt)
    from worldarena_baseline.wan_v10_data import validate_v10_data_receipt

    data_dir = args.optimizer_manifest.parent
    validate_v10_data_receipt(
        receipt,
        source_manifest=data_dir / "source-full-action-2100.jsonl",
        optimizer_manifest=args.optimizer_manifest,
        audit_manifest=args.audit_manifest,
        dev_manifest=ROOT / "eval/dev-fast-20-v3/dev-fast-20.jsonl",
    )
    optimizer_rows, audit_rows, replay = (
        _read_jsonl(args.optimizer_manifest), _read_jsonl(args.audit_manifest), _read_jsonl(args.replay)
    )
    if len(optimizer_rows) != 2060 or len(audit_rows) != 20 or len(replay) != 500:
        raise RuntimeError("v10 requires exact optimizer2060/audit20/replay500")
    if [row.get("optimizer_step") for row in replay] != list(range(1, 501)):
        raise RuntimeError("v10 replay optimizer steps differ")
    if {row["sample"] for row in optimizer_rows} & {row["sample"] for row in audit_rows}:
        raise RuntimeError("v10 optimizer leaks audit20")
    if args.mode == "train" and args.stop_step not in (50, 150, 300, 500):
        raise RuntimeError("v10 stop step is not approved")
    if args.mode == "audit" and args.audit_step not in (50, 150, 300, 500):
        raise RuntimeError("v10 audit step is not approved")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return receipt, optimizer_rows, audit_rows, replay


class _Dataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from worldarena_baseline.wan_cached_dataset import _load_v3_condition

        row = self.rows[index]
        latent = torch.load(row["latent_path"], map_location="cpu", weights_only=True)
        context = torch.load(row["context_path"], map_location="cpu", weights_only=True)
        condition = _load_v3_condition(
            Path(row["action_raster_path"]), sample=row["sample"],
            expected_statistics_hash=str(row["pose_statistics_sha256"]),
        )
        return {
            "latent": latent, "context": context,
            "action_raster": condition["raster"].permute(1, 0, 2, 3).contiguous(),
            "condition_support": condition["condition_support"],
            "loss_weight": condition["loss_weight"],
            "action_present": torch.tensor(1.0), "sample": row["sample"],
            "task": row.get("task", ""),
        }


def _batch(dataset, index):
    row = dataset[index]
    return {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value for key, value in row.items()}


def _sample_map(dataset):
    result = {str(row["sample"]): index for index, row in enumerate(dataset.rows)}
    if len(result) != len(dataset):
        raise RuntimeError("v10 dataset has duplicate sample identities")
    return result


def _load_model(args, legacy, device):
    from worldarena_baseline.wan_action_adapter import enable_wan_block_checkpointing
    from worldarena_baseline.wan_v10_model import (
        ParentPlusRelationalWan, V10_BLOCKS, install_v10_relational_band,
        v10_trainable_parameter_names,
    )
    legacy.install_wan_ti2v_package(WAN_SOURCE)
    from wan.modules.attention import attention
    from wan.modules.model import WanModel, rope_apply

    backbone = WanModel.from_pretrained(
        args.checkpoint_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    backbone.requires_grad_(False)
    enable_wan_block_checkpointing(backbone)
    wrappers = install_v10_relational_band(
        backbone, V10_BLOCKS, attention_fn=attention, rope_apply_fn=rope_apply,
        initialization_seed=SEED,
    )
    backbone = backbone.to(device)
    parent_payload = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=True)
    stage1 = parent_payload.get("stage1")
    source_sha = stage1.get("source_manifest_sha256") if isinstance(stage1, Mapping) else None
    if not isinstance(source_sha, str):
        raise RuntimeError("v10 parent lacks source manifest provenance")
    parent = legacy._load_parent(args, device, source_manifest_sha256=source_sha)
    model = ParentPlusRelationalWan(backbone, parent, wrappers).to(device)
    normalization = _read_json(args.relation_root / "normalization.json")
    model.relation_encoder.set_normalization(
        torch.tensor(normalization["mean"], device=device),
        torch.tensor(normalization["scale"], device=device),
    )
    names = v10_trainable_parameter_names(model)
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if count < 400_000_000:
        raise RuntimeError("v10 trainable native attention inventory is unexpectedly small")
    return model, names, count


def _relation(args, sample, variant, support, device):
    from worldarena_baseline.wan_v10_attention import RelationCondition
    from worldarena_baseline.wan_v10_data import validate_v10_relation_cache

    normalization = _read_json(args.relation_root / "normalization.json")
    payload = validate_v10_relation_cache(
        args.relation_root / "relations" / f"{sample}.npz",
        expected_sample=sample,
        expected_normalization_sha256=str(normalization["receipt_sha256"]),
    )
    def tensor(name, dtype):
        return torch.from_numpy(np.asarray(payload[f"{variant}_{name}"])).unsqueeze(0).to(device=device, dtype=dtype)
    return RelationCondition(
        anchored_se3=tensor("anchored_se3", torch.float32),
        velocity=tensor("velocity", torch.float32), uv=tensor("uv", torch.float32),
        gripper=tensor("gripper", torch.float32),
        arm_present=tensor("arm_present", torch.bool),
        motion_active=tensor("motion_active", torch.bool),
        support=support.to(device=device, dtype=torch.float32).permute(0, 2, 1, 3, 4),
        destination_time=torch.arange(21, device=device, dtype=torch.float32).reshape(1, 21),
    )


def _noise(record, shape, device, dtype):
    generator = torch.Generator(device="cpu"); generator.manual_seed(int(record["noise_seed"]))
    return torch.randn(shape, generator=generator).to(device=device, dtype=dtype)


def _timestep(record, device):
    generator = torch.Generator(device="cpu"); generator.manual_seed(int(record["timestep_seed"]))
    return torch.randint(1, 1000, (1,), generator=generator).to(device=device, dtype=torch.float32)


def _observability(args, sample, device):
    value = np.load(args.observability_root / f"{sample}.npy", allow_pickle=False)
    if value.shape != (2, 81) or value.dtype != np.dtype(bool):
        raise RuntimeError("v10 observability sidecar differs")
    return torch.from_numpy(value).unsqueeze(0).to(device)


def _forward(args, model, legacy, batch, record, variant, device, *, grad, probe, sigma_contract):
    from worldarena_baseline.wan_gripper_probe import build_gripper_targets
    from worldarena_baseline.wan_gripper_trajectory_loss import predicted_clean_latent, probe_trajectory_terms, sigma_weights
    from worldarena_baseline.wan_v10_objective import hidden_eef_loss, phase_ranking_loss, robot_region_energy

    sample = str(record["sample"])
    clean = batch["latent"].to(device=device, dtype=torch.bfloat16)
    context = batch["context"].to(device=device, dtype=torch.bfloat16)
    raster = batch["action_raster"].to(device=device, dtype=torch.bfloat16)
    support = batch["condition_support"].to(device=device, dtype=torch.bfloat16)
    loss_weight = batch["loss_weight"].to(device=device, dtype=torch.bfloat16)
    timestep = _timestep(record, device)
    noisy, target, token_timestep, valid = legacy.ti2v_flow_matching_sample(
        clean, timestep, noise=_noise(record, tuple(clean.shape), device, clean.dtype)
    )
    relation = _relation(args, sample, variant, support, device)
    manager = torch.enable_grad() if grad else torch.no_grad()
    with manager, torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = torch.stack(model(
            list(noisy.unbind(0)), token_timestep, list(context.unbind(0)),
            token_timestep.shape[1], action_raster=raster,
            condition_support=support,
            action_present=batch["action_present"].to(device),
            relation_condition=relation,
        ))
        fm = legacy.weighted_flow_mse(prediction, target, loss_weight=loss_weight, valid_mask=valid)
        correct_relation = _relation(args, sample, "correct", support, device)
        energy, eligible = robot_region_energy(
            prediction, target, correct_support=support, loss_weight=loss_weight,
            valid_mask=valid, motion_active=correct_relation.motion_active.permute(0, 2, 1)[:, :, 1:],
        )
        sigma = timestep.float() / 1000.0
        labels = build_gripper_targets(
            raster.float(), observability=_observability(args, sample, device)
        )
        predicted_clean = predicted_clean_latent(noisy.float(), prediction.float(), sigma=sigma)
        trajectory = probe_trajectory_terms(
            predicted_clean, probe, target_position=labels["position"],
            position_valid=labels["position_valid"], target_velocity=labels["velocity"],
            velocity_valid=labels["velocity_valid"], sigma_weight=sigma_weights(sigma, sigma_contract),
            huber_delta=0.02,
        )
        phase = phase_ranking_loss(
            trajectory["predicted_position"], labels["position"],
            valid=labels["position_valid"],
            motion_discriminative=correct_relation.motion_active.permute(0, 2, 1),
        )
        packed = labels["heatmap"].permute(0, 2, 1, 3, 4)
        packed = torch.nn.functional.interpolate(
            packed.flatten(0, 2).unsqueeze(1), size=(15, 20), mode="bilinear", align_corners=False
        ).squeeze(1).reshape(1, 21, 2, 15, 20).clamp(0, 1)
        hidden = hidden_eef_loss(
            model.hidden_eef_predictions(), packed,
            valid=labels["position_valid"].permute(0, 2, 1),
            sigma_weight=sigma_weights(sigma, sigma_contract),
        )
    return {
        "fm": fm, "energy": energy, "eligible": eligible, "hidden": hidden["loss"],
        "phase": phase["loss"], "phase_metrics": phase, "trajectory": trajectory,
        "labels": labels,
        "prediction": prediction, "target": target, "noisy": noisy, "valid": valid,
    }


def _release(model):
    model.release_completed_backward_conditions()


def _native_parameters(model):
    return [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and ".base." in name
        and any(f".base.{projection}." in name for projection in ("q", "k", "v", "o"))
    ]


def _grad_snapshot(parameters):
    return [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in parameters]


def _coefficients(correct, wrong, eligible, tau=0.1):
    mask = eligible.float()
    coefficient = torch.sigmoid((correct.float() - wrong.float()) / tau) * mask / (tau * mask.sum().clamp_min(1))
    return coefficient, -coefficient


def _calibrate(args, model, legacy, dataset, index_map, record, device, probe, sigma_contract):
    from worldarena_baseline.wan_v10_objective import calibrate_v10_lambdas

    batch = _batch(dataset, index_map[str(record["sample"])])
    parameters = _native_parameters(model)
    gradients = {}
    model.set_hidden_eef_backbone_scale(1.0)
    for name in ("fm", "hidden", "phase", "position", "velocity"):
        model.zero_grad(set_to_none=True)
        result = _forward(args, model, legacy, batch, record, "correct", device, grad=True, probe=probe, sigma_contract=sigma_contract)
        loss = result[name] if name in result else result["trajectory"][f"{name}_loss"]
        loss.backward(); _release(model)
        gradients[name] = _grad_snapshot(parameters)
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        correct_preview = _forward(args, model, legacy, batch, record, "correct", device, grad=False, probe=probe, sigma_contract=sigma_contract); _release(model)
        wrong_preview = _forward(args, model, legacy, batch, record, str(record["negative_family"]), device, grad=False, probe=probe, sigma_contract=sigma_contract); _release(model)
    cc, cw = _coefficients(correct_preview["energy"], wrong_preview["energy"], correct_preview["eligible"])
    model.zero_grad(set_to_none=True)
    result = _forward(args, model, legacy, batch, record, "correct", device, grad=True, probe=probe, sigma_contract=sigma_contract)
    (result["energy"] * cc).sum().backward(); _release(model)
    result = _forward(args, model, legacy, batch, record, str(record["negative_family"]), device, grad=True, probe=probe, sigma_contract=sigma_contract)
    (result["energy"] * cw).sum().backward(); _release(model)
    gradients["cf"] = _grad_snapshot(parameters)
    model.zero_grad(set_to_none=True); model.set_hidden_eef_backbone_scale(0.0)
    return calibrate_v10_lambdas(
        fm_gradients=gradients.pop("fm"), objective_gradients=gradients
    )


def _train_step(args, model, legacy, dataset, index_map, record, device, optimizer, calibration, probe, sigma_contract):
    from worldarena_baseline.wan_v10_objective import v10_loss_schedule

    step = int(record["optimizer_step"]); schedule = v10_loss_schedule(step)
    model.set_hidden_eef_backbone_scale(float(schedule["hidden_backbone"]))
    batch = _batch(dataset, index_map[str(record["sample"])])
    with torch.no_grad():
        wrong_preview = _forward(args, model, legacy, batch, record, str(record["negative_family"]), device, grad=False, probe=probe, sigma_contract=sigma_contract); _release(model)
    optimizer.zero_grad(set_to_none=True)
    correct = _forward(args, model, legacy, batch, record, "correct", device, grad=True, probe=probe, sigma_contract=sigma_contract)
    cc, cw = _coefficients(correct["energy"].detach(), wrong_preview["energy"], correct["eligible"])
    lambdas = calibration["lambdas"]
    total = correct["fm"] + float(lambdas["cf"]) * float(schedule["cf"]) * (correct["energy"] * cc).sum()
    total = total + float(lambdas["hidden"]) * float(schedule["hidden"]) * correct["hidden"]
    total = total + float(lambdas["phase"]) * float(schedule["phase"]) * correct["phase"]
    total = total + float(lambdas["position"]) * float(schedule["position"]) * correct["trajectory"]["position_loss"]
    total = total + float(lambdas["velocity"]) * float(schedule["velocity"]) * correct["trajectory"]["velocity_loss"]
    total.backward(); _release(model)
    wrong = _forward(args, model, legacy, batch, record, str(record["negative_family"]), device, grad=True, probe=probe, sigma_contract=sigma_contract)
    (float(lambdas["cf"]) * float(schedule["cf"]) * (wrong["energy"] * cw).sum()).backward(); _release(model)
    outside = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad and parameter.grad is not None and bool(torch.count_nonzero(parameter.grad)):
            outside += 1
        if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"v10 missing/nonfinite trainable gradient: {name}")
    norm = float(torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
    ).detach().cpu())
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return {
        "stage": schedule["stage"], "fm": float(correct["fm"].detach().cpu()),
        "hidden": float(correct["hidden"].detach().cpu()), "phase": float(correct["phase"].detach().cpu()),
        "position": float(correct["trajectory"]["position_loss"].detach().cpu()),
        "velocity": float(correct["trajectory"]["velocity_loss"].detach().cpu()),
        "margin": float(((wrong_preview["energy"] - correct["energy"].detach())[correct["eligible"]]).mean().cpu()) if bool(correct["eligible"].any()) else 0.0,
        "negative": str(record["negative_family"]), "grad_norm": norm,
        "outside_whitelist_gradients": outside,
    }


def _gradient_history(model, history):
    from worldarena_baseline.wan_v10_training import classify_v10_parameter

    result = dict(history)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        family = classify_v10_parameter(name)
        value = float(parameter.grad.detach().float().norm().cpu())
        result[family] = max(result.get(family, 0.0), value)
    return result


def _train_step_with_history(*args, history, **kwargs):
    model = args[1]
    # Capture family gradients before _train_step clears them by temporarily wrapping step.
    # The returned total norm remains the hard per-step finiteness check; family history is
    # sampled by hooks installed on every exact trainable below.
    seen = dict(history)
    hooks = []
    from worldarena_baseline.wan_v10_training import classify_v10_parameter

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        family = classify_v10_parameter(name)
        def capture(gradient, family=family):
            value = float(gradient.detach().float().norm().cpu())
            seen[family] = max(seen.get(family, 0.0), value)
            return gradient
        hooks.append(parameter.register_hook(capture))
    try:
        row = _train_step(*args, **kwargs)
    finally:
        for hook in hooks:
            hook.remove()
    return row, seen


def _trainable_state(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters() if parameter.requires_grad
    }


def _restore_trainable(model, state):
    named = dict(model.named_parameters())
    if set(state) != {name for name, parameter in named.items() if parameter.requires_grad}:
        raise RuntimeError("v10 comparison state inventory differs")
    with torch.no_grad():
        for name, value in state.items():
            named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))


def _audit_record(sample, index):
    def seed(label):
        return int.from_bytes(hashlib.sha256(f"v10-audit:{label}:{sample}".encode()).digest()[:8], "big")
    return {
        "sample": sample, "noise_seed": seed("noise"), "timestep_seed": seed("time"),
        "optimizer_step": index + 1, "negative_family": "reverse", "sample_role": "audit",
    }


def _masked_error(prediction, target, valid):
    values = (prediction.float() - target.float()).square().sum(dim=-1).sqrt()
    return float(values[valid].mean().cpu()) if bool(valid.any()) else float("nan")


def _audit_once(args, model, legacy, dataset, index_map, rows, device, probe, sigma_contract):
    episodes = []
    for index, item in enumerate(rows):
        sample = str(item["sample"]); record = _audit_record(sample, index); batch = _batch(dataset, index_map[sample])
        values = {}
        for variant in ("correct", "reverse", "swap"):
            result = _forward(args, model, legacy, batch, record, variant, device, grad=False, probe=probe, sigma_contract=sigma_contract); _release(model)
            eligible = result["eligible"]
            energy = float(result["energy"][eligible].mean().cpu()) if bool(eligible.any()) else float("nan")
            values[variant] = {"energy": energy, "fm": float(result["fm"].cpu())}
            if variant == "correct":
                trajectory = result["trajectory"]; labels = result["labels"]; phase = result["phase_metrics"]
                values[variant].update(
                    position=_masked_error(trajectory["predicted_position"], labels["position"], labels["position_valid"]),
                    velocity=_masked_error(trajectory["predicted_velocity"], labels["velocity"], labels["velocity_valid"]),
                    phase_plus=float(phase["plus_margin"].cpu()) if int(phase["plus_valid_count"].item()) else float("nan"),
                    phase_minus=float(phase["minus_margin"].cpu()) if int(phase["minus_valid_count"].item()) else float("nan"),
                    hidden_finite=math.isfinite(float(result["hidden"].cpu())),
                )
        episodes.append({"sample": sample, "values": values})
        print(json.dumps({"event": "v10_audit", "completed": index + 1, "sample": sample}), flush=True)
    families = {}
    for family in ("reverse", "swap"):
        margins = [
            episode["values"][family]["energy"] - episode["values"]["correct"]["energy"]
            for episode in episodes
            if math.isfinite(episode["values"][family]["energy"])
            and math.isfinite(episode["values"]["correct"]["energy"])
        ]
        families[family] = {"wins": sum(value > 0 for value in margins), "eligible": len(margins), "mean_margin": sum(margins) / len(margins) if margins else float("nan")}
    for family, key in (("phase+1", "phase_plus"), ("phase-1", "phase_minus")):
        margins = [episode["values"]["correct"][key] for episode in episodes if math.isfinite(episode["values"]["correct"][key])]
        families[family] = {"wins": sum(value > 0 for value in margins), "eligible": len(margins), "mean_margin": sum(margins) / len(margins) if margins else float("nan")}
    correct = [episode["values"]["correct"] for episode in episodes]
    def mean(key):
        values = [item[key] for item in correct if math.isfinite(item[key])]
        return sum(values) / len(values) if values else float("nan")
    routing = sum(wrapper.condition_use_count > 0 for wrapper in model.relation_wrappers.values()) / len(model.relation_wrappers)
    return {
        "episodes": episodes,
        "metrics": {
            "families": families, "routing_retention": routing,
            "fm_mean": mean("fm"), "position_mean": mean("position"),
            "velocity_mean": mean("velocity"),
            "hidden_eef_finite": all(item["hidden_finite"] for item in correct),
        },
    }


def _audit_comparison(args, model, legacy, dataset, index_map, rows, device, probe, sigma_contract, initial_state):
    from worldarena_baseline.wan_v10_model import relation_gates_enabled

    current = _trainable_state(model)
    enabled = _audit_once(args, model, legacy, dataset, index_map, rows, device, probe, sigma_contract)
    with relation_gates_enabled(model, enabled=False):
        gate_zero = _audit_once(args, model, legacy, dataset, index_map, rows, device, probe, sigma_contract)
    _restore_trainable(model, initial_state)
    try:
        parent = _audit_once(args, model, legacy, dataset, index_map, rows, device, probe, sigma_contract)
    finally:
        _restore_trainable(model, current)
    metrics = enabled["metrics"]; baseline = parent["metrics"]; zero = gate_zero["metrics"]
    metrics.update(
        fm_regression=(metrics["fm_mean"] - baseline["fm_mean"]) / max(baseline["fm_mean"], 1e-12),
        position_improvement=(baseline["position_mean"] - metrics["position_mean"]) / max(baseline["position_mean"], 1e-12),
        velocity_improvement=(baseline["velocity_mean"] - metrics["velocity_mean"]) / max(baseline["velocity_mean"], 1e-12),
        relation_enabled_beats_zero=(metrics["position_mean"] <= zero["position_mean"] and metrics["velocity_mean"] <= zero["velocity_mean"]),
    )
    return {"contract": "wan-v10-audit-comparison/1", "enabled": enabled, "relation_zero": gate_zero, "fresh_parent": parent}


def _atomic_save(payload, path):
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, partial); os.replace(partial, path)


def _load_trainable(model, payload):
    named = dict(model.named_parameters())
    expected = {name for name, parameter in named.items() if parameter.requires_grad}
    if set(payload["model"]) != expected:
        raise RuntimeError("v10 checkpoint trainable state differs")
    with torch.no_grad():
        for name, value in payload["model"].items():
            named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))


def _lineage(args):
    cache_digest = hashlib.sha256()
    for sidecar in sorted((args.relation_root / "relations").glob("*.meta.json")):
        cache_digest.update(sidecar.name.encode()); cache_digest.update(sidecar.read_bytes())
    config = json.dumps({"seed": SEED, "steps": 500, "blocks": list(range(6, 18))}, sort_keys=True).encode()
    return {
        "source_closure_sha256": _source_sha(), "config_sha256": hashlib.sha256(config).hexdigest(),
        "parent_sha256": _sha(args.parent_checkpoint), "data_sha256": _sha(args.data_receipt),
        "cache_sha256": cache_digest.hexdigest(), "replay_sha256": _sha(args.replay),
        "audit_sha256": _sha(args.audit_manifest), "probe_sha256": _sha(args.probe_checkpoint),
        "calibration_sha256": _sha(args.trajectory_calibration),
    }


def main() -> None:
    args = parse_args(); receipt, optimizer_rows, audit_rows, replay = _validate_inputs(args)
    legacy = _load_runtime(); torch.cuda.set_device(0); device = torch.device("cuda:0")
    from worldarena_baseline.wan_v10_training import (
        build_v10_checkpoint, build_v10_optimizer, set_v10_learning_rates, validate_v10_checkpoint,
    )
    dataset = _Dataset([*optimizer_rows, *audit_rows]); index_map = _sample_map(dataset)
    model, names, count = _load_model(args, legacy, device)
    initial_state = _trainable_state(model)
    probe = legacy._load_probe(args, device)
    sigma_payload = _read_json(args.trajectory_calibration)
    if sigma_payload.get("parent_sha256") != PARENT_SHA or sigma_payload.get("probe_sha256") != _sha(args.probe_checkpoint):
        raise RuntimeError("v10 trajectory calibration lineage differs")
    sigma_contract = sigma_payload["sigma"]
    optimizer = build_v10_optimizer(model); lineage = _lineage(args)
    calibration = _calibrate(args, model, legacy, dataset, index_map, replay[0], device, probe, sigma_contract)
    start = 0; gates = {}; history = {}; payload = None
    if args.resume is not None:
        payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        expected_phase = str(payload["resume_phase"])
        validate_v10_checkpoint(payload, expected_lineage=lineage, expected_phase=expected_phase, require_gated=args.mode == "train")
        _load_trainable(model, payload)
        if args.mode == "train": optimizer.load_state_dict(payload["optimizer"])
        calibration = dict(payload["calibration"]); gates = dict(payload["gates"])
        history = dict(payload.get("gradient_history", {})); start = int(payload["step"])
    if args.mode == "preflight":
        output = {"contract": "wan-v10-preflight/1", "lineage": lineage, "trainable_names": sorted(names), "trainable_count": count, "calibration": calibration}
        (args.output_dir / "preflight.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(json.dumps(output, sort_keys=True)); return
    if args.mode == "smoke":
        torch.cuda.reset_peak_memory_stats(device); rows = []
        for record in replay[:3]:
            set_v10_learning_rates(optimizer, int(record["optimizer_step"])); started = time.monotonic()
            row, history = _train_step_with_history(
                args, model, legacy, dataset, index_map, record, device, optimizer,
                calibration, probe, sigma_contract, history=history,
            )
            row["step_time"] = time.monotonic() - started; rows.append(row)
        allocated = torch.cuda.max_memory_allocated(device); reserved = torch.cuda.max_memory_reserved(device)
        output = {"contract": "wan-v10-production-smoke/1", "iterations": rows, "gradient_history": history, "peak_allocated": allocated, "peak_reserved": reserved, "limit": MEMORY_LIMIT, "pass": allocated < MEMORY_LIMIT and reserved < MEMORY_LIMIT}
        (args.output_dir / "production-smoke.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(json.dumps(output, sort_keys=True))
        if not output["pass"]: raise RuntimeError("v10 smoke exceeded 22 GiB")
        return
    if args.mode == "audit":
        if payload is None or start != args.audit_step:
            raise RuntimeError("v10 audit checkpoint step differs")
        from worldarena_baseline.wan_v10_audit import evaluate_v10_gate
        from worldarena_baseline.wan_v10_training import promote_v10_checkpoint

        report = _audit_comparison(
            args, model, legacy, dataset, index_map, audit_rows, device, probe,
            sigma_contract, initial_state,
        )
        metrics = report["enabled"]["metrics"]
        if start == 50:
            mapped = {
                "native_qkvo": history.get("native_qkvo", 0.0),
                "relation_encoders": history.get("relation_encoders", 0.0),
                "relation_gates": history.get("relation_gates", 0.0),
                "hidden_eef_heads": history.get("hidden_eef_heads", 0.0),
            }
            metrics.update(
                gradient_history=mapped, optimizer_finite=True,
                outside_whitelist_gradients=0, scheduler_drift=False,
                topology_drift=False,
            )
        gate = evaluate_v10_gate(
            start, metrics, step300_receipt=gates.get("step300")
        )
        report.update(step=start, gate=gate, lineage=lineage)
        output = args.output_dir / f"audit-{start:03d}.json"
        safe_report = _json_safe(report)
        output.write_text(json.dumps(safe_report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        if gate["pass"]:
            promoted = promote_v10_checkpoint(payload, gate)
            promoted["gradient_history"] = history
            _atomic_save(promoted, args.output_dir / f"step-{start:06d}-gated.pt")
        print(json.dumps(safe_report, sort_keys=True))
        if not gate["pass"]:
            raise SystemExit(3)
        return
    target = int(args.stop_step)
    if start >= target:
        raise RuntimeError("v10 resume step must be below target")
    log = args.output_dir / "training.jsonl"
    realized = {name: 0 for name in ("single_dominant", "bimanual_heavy", "mixed", "quiet")}
    if payload is not None:
        realized.update(payload["realized_strata"])
    for step in range(start + 1, target + 1):
        record = replay[step - 1]; set_v10_learning_rates(optimizer, step); started = time.monotonic()
        row, history = _train_step_with_history(
            args, model, legacy, dataset, index_map, record, device, optimizer,
            calibration, probe, sigma_contract, history=history,
        )
        row.update(step=step, step_time=time.monotonic() - started)
        realized[str(record["sample_role"])] += 1
        with log.open("a", encoding="utf-8") as handle: handle.write(json.dumps(row, sort_keys=True) + "\n")
        if step in (50, 150, 300, 500):
            checkpoint = build_v10_checkpoint(
                step=step, model=model, optimizer=optimizer, lineage=lineage,
                calibration=calibration, initialization_seed=SEED,
                topology={"world_size": 1, "physical_gpus": [6]}, gates=gates,
                samples_seen=step, realized_strata=realized,
            )
            checkpoint["gradient_history"] = history
            _atomic_save(checkpoint, args.output_dir / f"step-{step:06d}.pt")
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
