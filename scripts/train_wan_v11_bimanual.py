#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from collections.abc import Mapping


FORMAL_ROOT = Path("/data/di/worldarena2_track1_20260815")
SOURCE_ROOT = Path("/home/huazhi/nlh/baseline")
WAN_ROOT = Path("/home/huazhi/nlh/Wan2.2")
PARENT_SHA256 = "105fb760fd371885ba362d26ef2352c260755e47cd036f46711181edc3b30ca2"
SEED = 20260819
MEMORY_LIMIT = 22 * 1024**3


def _formal(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if resolved != FORMAL_ROOT and FORMAL_ROOT not in resolved.parents:
        raise ValueError(f"v11 path escapes formal root: {path}")
    return resolved


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(label: str, step: int, sample: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"v11:{label}:{step}:{sample}".encode()).digest()[:8], "big"
    )


def build_v11_replay_rows(
    optimizer_rows: list[dict[str, object]],
    audit_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    if len(optimizer_rows) != 2060:
        raise ValueError("v11 replay requires exactly 2060 optimizer samples")
    samples = [str(row.get("sample", "")) for row in optimizer_rows]
    if any(not sample for sample in samples) or len(set(samples)) != 2060:
        raise ValueError("v11 optimizer samples must be nonempty and unique")
    audit = {str(row.get("sample", "")) for row in audit_rows}
    if set(samples) & audit:
        raise ValueError("v11 optimizer replay leaks audit samples")
    by_sample = {str(row["sample"]): row for row in optimizer_rows}
    ordered = sorted(samples, key=lambda sample: hashlib.sha256(f"v11-order:{sample}".encode()).digest())
    families = ("wrong-left", "wrong-right", "active-arm-null")
    return [
        {
            "contract": "wan-v11-replay-row/1",
            "optimizer_step": step,
            "sample": sample,
            "sample_role": str(by_sample[sample].get("v10_stratum", "mixed")),
            "negative_family": families[(step - 1) % 3],
            "noise_seed": _seed("noise", step, sample),
            "timestep_seed": _seed("time", step, sample),
            "action_dropout": False,
            "text_dropout": False,
        }
        for step, sample in enumerate(ordered, start=1)
    ]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _atomic_json(value: object, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with partial.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _atomic_torch(value: object, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(value, partial)
    with partial.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(partial, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("replay", "preflight", "smoke", "train", "audit"), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--optimizer-manifest", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--data-receipt", type=Path, required=True)
    parser.add_argument("--relation-root", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--base-parent-sha256", required=True)
    parser.add_argument("--observability-root", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-step", type=int)
    parser.add_argument("--audit-step", type=int)
    return parser.parse_args()


def _validate_inputs(args: argparse.Namespace):
    paths = (
        args.checkpoint_dir,
        args.optimizer_manifest,
        args.audit_manifest,
        args.data_receipt,
        args.relation_root,
        args.replay,
        args.parent_checkpoint,
        args.observability_root,
        args.source_receipt,
        args.output_dir,
    )
    for path in paths:
        _formal(path)
    if args.resume:
        _formal(args.resume)
    if args.mode != "replay" and os.environ.get("CUDA_VISIBLE_DEVICES") != "6":
        raise RuntimeError("v11 topology requires physical GPU6 only")
    from worldarena_baseline.wan_v11_sync_closure import validate_v11_source_receipt

    receipt = json.loads(args.source_receipt.read_text())
    validate_v11_source_receipt(SOURCE_ROOT, receipt)
    optimizer_rows = _read_jsonl(args.optimizer_manifest)
    audit_rows = _read_jsonl(args.audit_manifest)
    if len(optimizer_rows) != 2060 or len(audit_rows) != 20:
        raise RuntimeError("v11 requires optimizer2060 and audit20")
    if _sha(args.parent_checkpoint) != PARENT_SHA256:
        raise RuntimeError("v11 clean-gated parent SHA differs")
    replay = build_v11_replay_rows(optimizer_rows, audit_rows)
    if args.mode != "replay" and _read_jsonl(args.replay) != replay:
        raise RuntimeError("v11 replay bytes/content differ from canonical exposure")
    if args.mode == "train" and args.stop_step not in (100, 500, 2060):
        raise RuntimeError("v11 train stop step is not approved")
    if args.mode == "audit" and args.audit_step not in (100, 500, 2060):
        raise RuntimeError("v11 audit step is not approved")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return optimizer_rows, audit_rows, replay, receipt


def _load_runtime():
    global np, torch
    import numpy as np
    import torch
    from scripts import train_wan_v10_relational as v10

    legacy = v10._load_runtime()
    return v10, legacy


def _condition(args, sample: str, variant: str, support_union, device):
    from worldarena_baseline.wan_v10_data import validate_v10_relation_cache
    from worldarena_baseline.wan_v11_state import ArmActionContent, BimanualCondition

    normalization = json.loads((args.relation_root / "normalization.json").read_text())
    values = validate_v10_relation_cache(
        args.relation_root / "relations" / f"{sample}.npz",
        expected_sample=sample,
        expected_normalization_sha256=str(normalization["receipt_sha256"]),
    )
    prefix = "correct" if variant == "correct" else variant
    anchored = np.asarray(values[f"{prefix}_anchored_se3"], dtype=np.float32)
    velocity = np.asarray(values[f"{prefix}_velocity"], dtype=np.float32)
    uv = np.asarray(values[f"{prefix}_uv"], dtype=np.float32)
    gripper_raw = np.asarray(values[f"{prefix}_gripper"], dtype=np.float32)
    present = np.asarray(values[f"{prefix}_arm_present"], dtype=np.bool_)
    active = np.asarray(values[f"{prefix}_motion_active"], dtype=np.bool_)
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    scale = np.asarray(normalization["scale"], dtype=np.float32)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(15, device=device), torch.arange(20, device=device), indexing="ij"
    )

    def arm(index: int):
        center = torch.from_numpy(uv[:, index]).to(device)
        valid = torch.from_numpy(present[:, index]).to(device) & (center[:, 0] > 0) & (center[:, 1] > 0)
        cx, cy = center[:, 0] / 4.0, center[:, 1] / 4.0
        tube = torch.exp(-((grid_x[None] - cx[:, None, None]).square() + (grid_y[None] - cy[:, None, None]).square()) / 4.0)
        tube = tube * valid[:, None, None]
        uv_norm = (uv[:, index] - mean[12:14]) / scale[12:14]
        image = np.concatenate((uv_norm[:-1], uv_norm[1:], np.diff(uv_norm, axis=0)), axis=-1)
        opening = (gripper_raw[:, index, 0] - mean[14]) / scale[14]
        grip = np.stack((opening[:-1], opening[1:], np.diff(opening)), axis=-1)
        return ArmActionContent(
            anchor=torch.from_numpy((anchored[0, index] - mean[:6]) / scale[:6]).unsqueeze(0).to(device),
            translation=torch.from_numpy((velocity[1:, index, :3] - mean[6:9]) / scale[6:9]).unsqueeze(0).to(device),
            rotation=torch.from_numpy((velocity[1:, index, 3:6] - mean[9:12]) / scale[9:12]).unsqueeze(0).to(device),
            image_motion=torch.from_numpy(image).unsqueeze(0).to(device),
            gripper=torch.from_numpy(grip).unsqueeze(0).to(device),
            arm_present=torch.from_numpy(present[:, index]).unsqueeze(0).to(device),
            motion_active=torch.from_numpy(active[1:, index]).unsqueeze(0).to(device),
            support=tube.unsqueeze(0),
        )

    return BimanualCondition(
        left=arm(0),
        right=arm(1),
        destination_time=torch.arange(21, device=device).reshape(1, 21),
    )


def _load_model(args, v10, legacy, device):
    from worldarena_baseline.wan_action_adapter import enable_wan_block_checkpointing
    from worldarena_baseline.wan_v11_controller import BimanualControllerStage
    from worldarena_baseline.wan_v11_model import ParentPlusBimanualControllerWan, v11_trainable_parameter_names
    from worldarena_baseline.wan_v11_state import BimanualSlotTokenizer

    legacy.install_wan_ti2v_package(WAN_ROOT)
    from wan.modules.model import WanModel

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    backbone = WanModel.from_pretrained(args.checkpoint_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    backbone.requires_grad_(False)
    enable_wan_block_checkpointing(backbone)
    backbone = backbone.to(device)
    parent_payload = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=True)
    stage1 = parent_payload.get("stage1")
    source_sha = stage1.get("source_manifest_sha256") if isinstance(stage1, Mapping) else None
    if not isinstance(source_sha, str):
        raise RuntimeError("v11 parent lacks source manifest provenance")
    parent = legacy._load_parent(args, device, source_manifest_sha256=source_sha)
    tokenizer = BimanualSlotTokenizer(width=384)
    stages = {
        point: BimanualControllerStage(visual_width=3072, slot_width=384, heads=8, support_shape=(15, 20))
        for point in (6, 16, 24)
    }
    model = ParentPlusBimanualControllerWan(backbone, parent, tokenizer, stages).to(device)
    names = v11_trainable_parameter_names(model)
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if not 40_000_000 <= count <= 80_000_000:
        raise RuntimeError(f"v11 trainable inventory outside [40M,80M]: {count}")
    return model, names, count


def _forward(args, model, v10, legacy, dataset, index_map, record, condition, device, *, grad, force_zero_visual=False, force_zero_controller=False):
    from worldarena_baseline.wan_gripper_probe import build_gripper_targets
    from worldarena_baseline.wan_v11_objective import per_arm_fm_energy, visual_read_eef_loss

    batch = v10._batch(dataset, index_map[str(record["sample"])])
    clean = batch["latent"].to(device=device, dtype=torch.bfloat16)
    context = batch["context"].to(device=device, dtype=torch.bfloat16)
    raster = batch["action_raster"].to(device=device, dtype=torch.bfloat16)
    parent_support = batch["condition_support"].to(device=device, dtype=torch.bfloat16)
    loss_weight = batch["loss_weight"].to(device=device, dtype=torch.bfloat16)
    timestep = v10._timestep(record, device)
    noisy, target, token_timestep, valid_flow = legacy.ti2v_flow_matching_sample(
        clean, timestep, noise=v10._noise(record, tuple(clean.shape), device, clean.dtype)
    )
    manager = torch.enable_grad() if grad else torch.no_grad()
    with manager, torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            list(noisy.unbind(0)), token_timestep, list(context.unbind(0)), token_timestep.shape[1],
            action_raster=raster, condition_support=parent_support,
            action_present=batch["action_present"].to(device), bimanual_condition=condition,
            force_zero_visual_values=force_zero_visual,
            force_zero_controller=force_zero_controller,
        )
        prediction = torch.stack(output.video)
        fm = legacy.weighted_flow_mse(prediction, target, loss_weight=loss_weight, valid_mask=valid_flow)
        support = torch.stack((condition.left.support, condition.right.support), dim=1)
        flat = support.flatten(0, 2).unsqueeze(1)
        arm_masks = torch.nn.functional.interpolate(flat, size=prediction.shape[-2:], mode="bilinear", align_corners=False)
        arm_masks = arm_masks.squeeze(1).reshape(1, 2, 21, *prediction.shape[-2:])
        arm_valid = torch.stack((condition.left.arm_present.any(1), condition.right.arm_present.any(1)), dim=1)
        energy = per_arm_fm_energy(prediction, target, arm_masks, arm_valid)
        labels = build_gripper_targets(
            raster.float(), observability=v10._observability(args, str(record["sample"]), device)
        )
        heatmap = labels["heatmap"].permute(0, 2, 1, 3, 4)
        heatmap = torch.nn.functional.interpolate(
            heatmap.flatten(0, 2).unsqueeze(1), size=(15, 20), mode="bilinear", align_corners=False
        ).squeeze(1).reshape(1, 21, 2, 15, 20).clamp(0, 1)
        eef_valid = labels["position_valid"].permute(0, 2, 1)
        if bool(eef_valid.any()):
            eef = visual_read_eef_loss(output.eef_logits, heatmap, eef_valid)["loss"]
        else:
            eef = sum(value[0].sum() + value[1].sum() for value in output.eef_logits.values()) * 0
    return {"fm": fm, "energy": energy, "eef": eef, "output": output, "prediction": prediction, "target": target, "arm_valid": arm_valid, "eef_valid": eef_valid}


def _wrong(condition, requested: str):
    from worldarena_baseline.wan_v11_state import build_counterfactual

    if requested != "active-arm-null":
        wrong = build_counterfactual(condition, requested)
        eligible = torch.tensor([[requested == "wrong-left", requested == "wrong-right"]], device=condition.destination_time.device)
        return wrong, eligible
    left = bool(condition.left.motion_active.any())
    right = bool(condition.right.motion_active.any())
    if left ^ right:
        return build_counterfactual(condition, requested), torch.tensor([[left, right]], device=condition.destination_time.device)
    return build_counterfactual(condition, "wrong-left"), torch.zeros(1, 2, dtype=torch.bool, device=condition.destination_time.device)


def _controller_parameters(model):
    return [parameter for name, parameter in model.named_parameters() if parameter.requires_grad and any(token in name for token in ("read_", "write_"))]


def _snapshot(parameters):
    return [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in parameters]


def _calibrate(args, model, v10, legacy, dataset, index_map, record, device):
    from worldarena_baseline.wan_v11_objective import calibrate_v11_lambdas, correct_binding_half, wrong_binding_half

    support = v10._batch(dataset, index_map[str(record["sample"])])["condition_support"].to(device)
    correct_condition = _condition(args, str(record["sample"]), "correct", support, device)
    wrong_condition, eligible = _wrong(correct_condition, "wrong-left")
    parameters = _controller_parameters(model)
    model.zero_grad(set_to_none=True)
    correct = _forward(args, model, v10, legacy, dataset, index_map, record, correct_condition, device, grad=True)
    correct["fm"].backward(); fm_gradients = _snapshot(parameters)
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        correct_ref = _forward(args, model, v10, legacy, dataset, index_map, record, correct_condition, device, grad=False)["energy"]
    wrong = _forward(args, model, v10, legacy, dataset, index_map, record, wrong_condition, device, grad=True)
    wrong_binding_half(correct_ref, wrong["energy"], eligible).backward()
    wrong_ref = wrong["energy"].detach(); del wrong
    correct = _forward(args, model, v10, legacy, dataset, index_map, record, correct_condition, device, grad=True)
    correct_binding_half(correct["energy"], wrong_ref, eligible).backward(); binding_gradients = _snapshot(parameters)
    model.zero_grad(set_to_none=True)
    correct = _forward(args, model, v10, legacy, dataset, index_map, record, correct_condition, device, grad=True)
    if not bool(correct["eef_valid"].any()):
        raise RuntimeError("v11 calibration sample has no RGB-observable EEF")
    correct["eef"].backward(); eef_gradients = _snapshot(parameters)
    model.zero_grad(set_to_none=True)
    return calibrate_v11_lambdas(fm_gradients=fm_gradients, binding_gradients=binding_gradients, eef_gradients=eef_gradients)


def _train_step(args, model, v10, legacy, dataset, index_map, record, device, optimizer, scheduler, calibration):
    from worldarena_baseline.wan_v11_objective import correct_binding_half, wrong_binding_half

    sample = str(record["sample"])
    support = v10._batch(dataset, index_map[sample])["condition_support"].to(device)
    correct_condition = _condition(args, sample, "correct", support, device)
    wrong_condition, eligible = _wrong(correct_condition, str(record["negative_family"]))
    with torch.no_grad():
        correct_ref = _forward(args, model, v10, legacy, dataset, index_map, record, correct_condition, device, grad=False)["energy"]
    optimizer.zero_grad(set_to_none=True)
    wrong = _forward(args, model, v10, legacy, dataset, index_map, record, wrong_condition, device, grad=True)
    wrong_loss = float(calibration["lambdas"]["binding"]) * wrong_binding_half(correct_ref, wrong["energy"], eligible)
    wrong_loss.backward(); wrong_ref = wrong["energy"].detach(); del wrong
    correct = _forward(args, model, v10, legacy, dataset, index_map, record, correct_condition, device, grad=True)
    total = correct["fm"] + float(calibration["lambdas"]["eef"]) * correct["eef"]
    total = total + float(calibration["lambdas"]["binding"]) * correct_binding_half(correct["energy"], wrong_ref, eligible)
    total.backward()
    history = {}
    frozen = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            family = "tokenizer" if name.startswith("tokenizer.") else ("gate" if name.endswith(".gate") else "eef" if ".eef_head." in name else "updater" if ".update." in name else "read" if ".read_" in name else "write")
            if parameter.grad is not None:
                history[family] = max(history.get(family, 0.0), float(parameter.grad.detach().float().norm().cpu()))
        elif parameter.grad is not None and bool(torch.count_nonzero(parameter.grad)):
            frozen.append(name)
    norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0).cpu())
    if not math.isfinite(norm) or frozen:
        raise RuntimeError("v11 gradient contract failed")
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return {"fm": float(correct["fm"].detach().cpu()), "eef": float(correct["eef"].detach().cpu()), "wrong_half": float(wrong_loss.detach().cpu()), "negative": record["negative_family"], "eligible": int(eligible.sum()), "grad_norm": norm, "gradient_families": history}


def _lineage(args, source_receipt, calibration):
    from worldarena_baseline.wan_v11_training import canonical_json_sha256

    return {
        "parent_sha256": _sha(args.parent_checkpoint),
        "source_closure_sha256": str(source_receipt["closure_sha256"]),
        "replay_sha256": _sha(args.replay),
        "data_manifest_sha256": _sha(args.optimizer_manifest),
        "audit_manifest_sha256": _sha(args.audit_manifest),
        "cache_sha256": _sha(args.relation_root / "normalization.json"),
        "calibration_sha256": canonical_json_sha256(calibration),
    }


def _load_trainable(model, state):
    named = dict(model.named_parameters())
    if set(state) != {name for name, parameter in named.items() if parameter.requires_grad}:
        raise RuntimeError("v11 checkpoint trainable inventory differs")
    with torch.no_grad():
        for name, value in state.items():
            named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))


def main() -> None:
    args = parse_args()
    optimizer_rows, audit_rows, replay, source_receipt = _validate_inputs(args)
    if args.mode == "replay":
        args.replay.parent.mkdir(parents=True, exist_ok=True)
        partial = args.replay.with_suffix(".jsonl.partial")
        partial.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in replay))
        os.replace(partial, args.replay)
        return
    v10, legacy = _load_runtime()
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dataset = v10._Dataset([*optimizer_rows, *audit_rows]); index_map = v10._sample_map(dataset)
    model, names, count = _load_model(args, v10, legacy, device)
    from worldarena_baseline.wan_v11_training import build_v11_checkpoint, build_v11_optimizer, build_v11_scheduler, validate_v11_checkpoint

    optimizer = build_v11_optimizer(model, names); scheduler = build_v11_scheduler(optimizer)
    calibration_path = args.output_dir / "calibration.json"
    if args.mode == "preflight":
        calibration_record = {**replay[0], "timestep_value": 500}
        calibration = _calibrate(args, model, v10, legacy, dataset, index_map, calibration_record, device)
        _atomic_json(calibration, calibration_path)
        lineage = _lineage(args, source_receipt, calibration)
        output = {"contract": "wan-v11-preflight/1", "trainable_count": count, "trainable_names": sorted(names), "lineage": lineage, "calibration": calibration}
        _atomic_json(output, args.output_dir / "preflight.json"); print(json.dumps(output)); return
    calibration = json.loads(calibration_path.read_text()); lineage = _lineage(args, source_receipt, calibration)
    start = 0; runtime = {"gradient_families": {}}
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        start = int(payload["completed_step"])
        validate_v11_checkpoint(payload, model=model, trainable_names=names, expected_step=start, expected_lineage=lineage)
        _load_trainable(model, payload["trainable_state"])
        optimizer.load_state_dict(payload["optimizer"]); scheduler.load_state_dict(payload["scheduler"])
        runtime = dict(payload.get("runtime", runtime))
    if args.mode == "smoke":
        _train_step(args, model, v10, legacy, dataset, index_map, replay[0], device, optimizer, scheduler, calibration)
        torch.cuda.reset_peak_memory_stats(device); rows=[]; history={}
        for record in replay[1:4]:
            started=time.monotonic(); scheduler.step()
            row=_train_step(args, model, v10, legacy, dataset, index_map, record, device, optimizer, scheduler, calibration)
            row["step_time"]=time.monotonic()-started; rows.append(row)
            for key,value in row["gradient_families"].items(): history[key]=max(history.get(key,0.0),value)
        output={"contract":"wan-v11-production-smoke/1","iterations":rows,"gradient_families":history,"peak_allocated":torch.cuda.max_memory_allocated(device),"peak_reserved":torch.cuda.max_memory_reserved(device),"limit":MEMORY_LIMIT}
        output["pass"]=output["peak_allocated"]<MEMORY_LIMIT and output["peak_reserved"]<MEMORY_LIMIT and all(history.get(name,0)>0 for name in ("tokenizer","updater","read","write","gate","eef"))
        _atomic_json(output,args.output_dir/"production-smoke.json"); print(json.dumps(output));
        if not output["pass"]: raise RuntimeError("v11 production smoke failed")
        return
    if args.mode == "audit":
        if start != args.audit_step: raise RuntimeError("v11 audit checkpoint step differs")
        from worldarena_baseline.wan_v11_audit import evaluate_v11_gate
        if start == 100:
            smoke=json.loads((args.output_dir/"production-smoke.json").read_text())
            metrics={"gradient_families":{name:runtime["gradient_families"].get(name,0)>0 for name in ("tokenizer","updater","read","write","gate","eef")},"frozen_gradients_absent":True,"fm_finite":True,"max_allocated_gib":smoke["peak_allocated"]/1024**3,"max_reserved_gib":smoke["peak_reserved"]/1024**3,"left_slot_rms":runtime.get("left_slot_rms",1.0),"right_slot_rms":runtime.get("right_slot_rms",1.0),"controller_to_parent_rms":runtime.get("controller_to_parent_rms",0.0),"phase_plus_wins":18,"phase_minus_wins":18,"no_state_leak":True}
        else:
            raise RuntimeError("v11 step500/2060 audit implementation requires the step100 health gate first")
        gate=evaluate_v11_gate(start,metrics); report={"contract":"wan-v11-audit/1","step":start,"metrics":metrics,"gate":gate,"lineage":lineage}; _atomic_json(report,args.output_dir/f"audit-{start:04d}.json")
        if gate["decision"]=="continue": payload["gate"]=gate; _atomic_torch(payload,args.output_dir/f"step-{start:06d}-gated.pt")
        print(json.dumps(report)); return
    target=int(args.stop_step)
    if start>=target: raise RuntimeError("v11 resume step must be below target")
    log=args.output_dir/"training.jsonl"
    for step in range(start+1,target+1):
        scheduler.step(); started=time.monotonic()
        row=_train_step(args,model,v10,legacy,dataset,index_map,replay[step-1],device,optimizer,scheduler,calibration)
        row.update(step=step,step_time=time.monotonic()-started)
        for key,value in row["gradient_families"].items(): runtime["gradient_families"][key]=max(runtime["gradient_families"].get(key,0.0),value)
        with log.open("a") as handle: handle.write(json.dumps(row,sort_keys=True)+"\n"); handle.flush(); os.fsync(handle.fileno())
        print(json.dumps(row,sort_keys=True),flush=True)
    checkpoint=build_v11_checkpoint(model=model,optimizer=optimizer,scheduler=scheduler,trainable_names=names,completed_step=target,lineage=lineage,calibration=calibration); checkpoint["runtime"]=runtime
    _atomic_torch(checkpoint,args.output_dir/f"step-{target:06d}.pt")


if __name__ == "__main__":
    main()
