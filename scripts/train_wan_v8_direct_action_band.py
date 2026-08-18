#!/usr/bin/env python3
"""Guarded single-GPU Wan v8 direct action band trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

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
    permitted = (ROOT, SOURCE) if allow_source else (ROOT,)
    if not any(resolved == base or base in resolved.parents for base in permitted):
        raise ValueError(f"v8 path escapes approved root: {path}")
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "smoke", "phase-m", "audit100", "phase-t", "audit250"), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-receipt", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--negative-root", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--base-parent-sha256", required=True)
    parser.add_argument("--probe-checkpoint", type=Path, required=True)
    parser.add_argument("--probe-split", type=Path, required=True)
    parser.add_argument("--observability-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--target-step", type=int)
    return parser.parse_args()


def _load_runtime():
    global np, torch
    import numpy as np
    import torch
    from scripts import train_wan_se3_probe_v7_fsdp as legacy
    legacy._load_runtime_dependencies()
    return legacy


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _validate_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "6":
        raise RuntimeError("v8 single-GPU lineage requires physical GPU6 only")
    for path in (args.checkpoint_dir, args.cache_root, args.manifest, args.data_receipt, args.replay, args.audit_manifest, args.negative_root, args.parent_checkpoint, args.probe_checkpoint, args.probe_split, args.observability_root, args.output_dir):
        _under_root(path)
    if _sha(args.parent_checkpoint) != PARENT_SHA:
        raise RuntimeError("v8 immutable clean-gated parent SHA mismatch")
    receipt = _read_json(args.data_receipt)
    if receipt.get("contract") != "wan-v8-direct-action-band-data/1" or receipt.get("cached_manifest_sha256") != _sha(args.manifest):
        raise RuntimeError("v8 data receipt differs from cached manifest")
    replay = _read_jsonl(args.replay)
    if len(replay) != 250 or [row.get("optimizer_step") for row in replay] != list(range(1, 251)):
        raise RuntimeError("v8 replay must contain exact single-rank steps 1-250")
    audit = _read_jsonl(args.audit_manifest)
    if len(audit) != 20:
        raise RuntimeError("v8 audit manifest must contain exactly 20 rows")
    optimizer = {str(row["sample"]) for row in replay}; heldout = {str(row["sample"]) for row in audit}
    if optimizer & heldout or set(receipt.get("audit_samples", ())) != heldout:
        raise RuntimeError("v8 replay/audit isolation mismatch")
    if receipt.get("replay_rows") != 250 or receipt.get("steps") != 250 or receipt.get("world_size") != 1:
        raise RuntimeError("v8 replay dimensions differ from receipt")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return receipt, replay, audit


def _lineage(args: argparse.Namespace) -> dict[str, str]:
    return {
        "parent_sha256": _sha(args.parent_checkpoint), "manifest_sha256": _sha(args.manifest),
        "data_receipt_sha256": _sha(args.data_receipt), "replay_sha256": _sha(args.replay),
        "audit_sha256": _sha(args.audit_manifest), "probe_sha256": _sha(args.probe_checkpoint),
    }


def _load_model(args: argparse.Namespace, legacy, device):
    from worldarena_baseline.wan_action_adapter import enable_wan_block_checkpointing
    from worldarena_baseline.wan_v8_model import ParentPlusDirectActionWan, V8_BLOCKS, install_v8_action_band, v8_trainable_parameter_names
    legacy.install_wan_ti2v_package(WAN_SOURCE)
    from wan.modules.attention import attention
    from wan.modules.model import WanModel, rope_apply
    backbone = WanModel.from_pretrained(args.checkpoint_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    backbone.requires_grad_(False)
    enable_wan_block_checkpointing(backbone)
    wrappers = install_v8_action_band(backbone, V8_BLOCKS, rope_apply, attention)
    backbone = backbone.to(device)
    parent_payload = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=True)
    stage1 = parent_payload.get("stage1")
    source_sha = stage1.get("source_manifest_sha256") if isinstance(stage1, Mapping) else None
    if not isinstance(source_sha, str):
        raise RuntimeError("v8 parent lacks source manifest provenance")
    parent = legacy._load_parent(args, device, source_manifest_sha256=source_sha)
    model = ParentPlusDirectActionWan(backbone, parent, wrappers).to(device)
    names = v8_trainable_parameter_names(model)
    expected_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if len([name for name in names if name.endswith("channel_gate")]) != 6 or expected_count < 220_000_000:
        raise RuntimeError("v8 native QKVO trainable whitelist failed")
    return model, names, expected_count


def _sample_map(dataset) -> dict[str, int]:
    result = {str(row["sample"]): index for index, row in enumerate(dataset.rows)}
    if len(result) != len(dataset):
        raise RuntimeError("v8 cached dataset has duplicate samples")
    return result


def _correct_condition(args, batch, sample: str, device):
    from worldarena_baseline.wan_se3_condition import validate_se3_cache
    values = validate_se3_cache(args.cache_root / "wan_v8_se3_conditions" / f"{sample}.npz", _sha(args.manifest))
    return {
        "action_raster": batch["action_raster"].to(device=device, dtype=torch.bfloat16),
        "condition_support": batch["condition_support"].to(device=device, dtype=torch.bfloat16),
        "action_present": batch["action_present"].to(device=device, dtype=torch.float32),
        "se3_arm_transform": torch.from_numpy(np.asarray(values["arm_transform"], dtype=np.float32)).unsqueeze(0).to(device),
        "se3_arm_present": torch.from_numpy(np.asarray(values["arm_present"], dtype=np.bool_)).unsqueeze(0).to(device),
    }


def _wrong_condition(args, batch, sample: str, record: Mapping[str, Any], device):
    family = str(record["negative_family"])
    label = "shift_plus" if family == "shift" and int(record["shift_direction"]) == 1 else "shift_minus" if family == "shift" else family
    path = args.negative_root / f"{sample}.npz"
    with np.load(path, allow_pickle=False) as values:
        raster = np.asarray(values[f"{label}_raster"], dtype=np.float32)
        support = np.asarray(values[f"{label}_support"], dtype=np.float32)
        se3 = np.asarray(values[f"{label}_se3"], dtype=np.float32)
        present = np.asarray(values[f"{label}_arm_present"], dtype=np.bool_)
    return {
        "action_raster": torch.from_numpy(raster).permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=torch.bfloat16),
        "condition_support": torch.from_numpy(support).unsqueeze(0).to(device=device, dtype=torch.bfloat16),
        "action_present": batch["action_present"].to(device=device, dtype=torch.float32),
        "se3_arm_transform": torch.from_numpy(se3).unsqueeze(0).to(device),
        "se3_arm_present": torch.from_numpy(present).unsqueeze(0).to(device),
    }


def _noise(record, shape, device, dtype):
    generator = torch.Generator(device="cpu"); generator.manual_seed(int(record["noise_seed"]))
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)


def _timestep(record, device):
    generator = torch.Generator(device="cpu"); generator.manual_seed(int(record["timestep_seed"]))
    return torch.randint(1, 1000, (1,), generator=generator).to(device=device, dtype=torch.float32)


def _forward(model, legacy, batch, condition, record, device, *, grad: bool):
    clean=batch["latent"].to(device=device,dtype=torch.bfloat16); context=batch["context"].to(device=device,dtype=torch.bfloat16)
    loss_weight=batch["loss_weight"].to(device=device,dtype=torch.bfloat16); timestep=_timestep(record,device)
    noisy,target,token_timestep,valid=legacy.ti2v_flow_matching_sample(clean,timestep,noise=_noise(record,tuple(clean.shape),device,clean.dtype))
    context_manager=torch.enable_grad() if grad else torch.no_grad()
    with context_manager, torch.autocast("cuda",dtype=torch.bfloat16):
        prediction=torch.stack(model(list(noisy.unbind(0)),token_timestep,list(context.unbind(0)),token_timestep.shape[1],**condition))
        fm=legacy.weighted_flow_mse(prediction,target,loss_weight=loss_weight,valid_mask=valid)
        from worldarena_baseline.wan_v71_cf import support_weighted_fm_energy
        energy=support_weighted_fm_energy(prediction,target,condition_support=batch["condition_support"].to(device=device,dtype=torch.bfloat16),loss_weight=loss_weight,valid_mask=valid).mean()
    return {"fm":fm,"energy":energy,"prediction":prediction,"target":target,"noisy":noisy,"valid":valid,"timestep":timestep}


def _release(model):
    model.release_completed_backward_conditions()


def _coefficients(correct: float, wrong: float, tau: float=.1) -> tuple[float,float,float]:
    value=(correct-wrong)/tau
    sigmoid=1.0/(1.0+math.exp(-max(-60.0,min(60.0,value))))
    return sigmoid/tau,-sigmoid/tau,math.log1p(math.exp(max(-60.0,min(60.0,value))))


def _batch(dataset, index: int):
    row=dataset[index]
    return {key:value.unsqueeze(0) if isinstance(value,torch.Tensor) else value for key,value in row.items()}


def _gradient_norm(named: Mapping[str, Any]) -> float:
    total=0.0
    for name,parameter in named.items():
        if name.endswith("channel_gate"): continue
        if parameter.grad is None: raise RuntimeError(f"v8 disconnected QKVO gradient: {name}")
        if not torch.isfinite(parameter.grad).all(): raise RuntimeError(f"v8 nonfinite gradient: {name}")
        total += float(parameter.grad.detach().float().square().sum().cpu())
    return math.sqrt(total)


def _calibrate(args, model, legacy, dataset, index_map, record, device) -> dict[str,float]:
    sample=str(record["sample"]); batch=_batch(dataset,index_map[sample]); correct=_correct_condition(args,batch,sample,device); wrong=_wrong_condition(args,batch,sample,record,device)
    with torch.no_grad():
        ec=float(_forward(model,legacy,batch,correct,record,device,grad=False)["energy"].cpu()); _release(model)
        ew=float(_forward(model,legacy,batch,wrong,record,device,grad=False)["energy"].cpu()); _release(model)
    cc,cw,ranking=_coefficients(ec,ew)
    named={name:p for name,p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True); result=_forward(model,legacy,batch,correct,record,device,grad=True); result["fm"].backward(); _release(model); fm_norm=_gradient_norm(named)
    model.zero_grad(set_to_none=True); result=_forward(model,legacy,batch,correct,record,device,grad=True); (result["energy"]*cc).backward(); _release(model)
    result=_forward(model,legacy,batch,wrong,record,device,grad=True); (result["energy"]*cw).backward(); _release(model); cf_norm=_gradient_norm(named)
    model.zero_grad(set_to_none=True)
    if not fm_norm>0 or not cf_norm>0: raise RuntimeError("v8 calibration gradient norm is zero")
    return {"lambda_cf":0.5*fm_norm/cf_norm,"tau":.1,"fm_qkvo_grad_rms":fm_norm,"cf_qkvo_grad_rms":cf_norm,"initial_ranking":ranking,"initial_margin":ew-ec}


def _train_step(args,model,legacy,dataset,index_map,record,device,optimizer,lambda_cf):
    sample=str(record["sample"]); batch=_batch(dataset,index_map[sample]); correct=_correct_condition(args,batch,sample,device); wrong=_wrong_condition(args,batch,sample,record,device)
    with torch.no_grad():
        ec=float(_forward(model,legacy,batch,correct,record,device,grad=False)["energy"].cpu()); _release(model)
        ew=float(_forward(model,legacy,batch,wrong,record,device,grad=False)["energy"].cpu()); _release(model)
    cc,cw,ranking=_coefficients(ec,ew)
    optimizer.zero_grad(set_to_none=True)
    result=_forward(model,legacy,batch,correct,record,device,grad=True); (result["fm"]+lambda_cf*cc*result["energy"]).backward(); _release(model); fm=float(result["fm"].detach().cpu())
    result=_forward(model,legacy,batch,wrong,record,device,grad=True); (lambda_cf*cw*result["energy"]).backward(); _release(model)
    norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0).detach().cpu())
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return {"fm":fm,"ranking":ranking,"margin":ew-ec,"grad_norm":norm,"negative":str(record["negative_family"])}


def _atomic_torch_save(payload, path: Path):
    partial=path.with_suffix(path.suffix+".partial"); torch.save(payload,partial); os.replace(partial,path)


def _load_trainable(model,payload):
    named=dict(model.named_parameters())
    if set(payload["model"]) != {name for name,p in named.items() if p.requires_grad}: raise RuntimeError("v8 checkpoint model state mismatch")
    with torch.no_grad():
        for name,value in payload["model"].items(): named[name].copy_(value.to(device=named[name].device,dtype=named[name].dtype))


def main() -> None:
    args=parse_args(); receipt,replay,audit=_validate_inputs(args); legacy=_load_runtime()
    torch.cuda.set_device(0); device=torch.device("cuda:0")
    from worldarena_baseline.wan_cached_dataset import WanActionCachedDataset
    from worldarena_baseline.wan_v8_training import build_v8_checkpoint,build_v8_optimizer,set_v8_learning_rates,validate_v8_checkpoint
    dataset=WanActionCachedDataset(args.manifest,args.cache_root); index_map=_sample_map(dataset)
    model,names,count=_load_model(args,legacy,device); optimizer=build_v8_optimizer(model); lineage=_lineage(args); gates={}
    calibration=_calibrate(args,model,legacy,dataset,index_map,replay[0],device)
    start=0
    if args.resume:
        payload=torch.load(args.resume,map_location="cpu",weights_only=True); validate_v8_checkpoint(payload,lineage=lineage,expected_phase=str(payload["phase"])); _load_trainable(model,payload); optimizer.load_state_dict(payload["optimizer"]); calibration=dict(payload["calibration"]); gates=dict(payload["gates"]); start=int(payload["step"])
    if args.mode=="preflight":
        output={"contract":"wan-v8-preflight/1","trainable_count":count,"trainable_names":sorted(names),"calibration":calibration,"lineage":lineage,"cuda_visible_devices":os.environ["CUDA_VISIBLE_DEVICES"]}
        (args.output_dir/"preflight.json").write_text(json.dumps(output,indent=2,sort_keys=True)+"\n"); print(json.dumps(output,sort_keys=True)); return
    if args.mode=="smoke":
        torch.cuda.reset_peak_memory_stats(device); metrics=[]
        for record in replay[:3]:
            started=time.monotonic(); set_v8_learning_rates(optimizer,int(record["optimizer_step"])); row=_train_step(args,model,legacy,dataset,index_map,record,device,optimizer,float(calibration["lambda_cf"])); row["step_time"]=time.monotonic()-started; metrics.append(row)
        allocated=torch.cuda.max_memory_allocated(device); reserved=torch.cuda.max_memory_reserved(device)
        output={"contract":"wan-v8-production-smoke/1","iterations":metrics,"peak_allocated":allocated,"peak_reserved":reserved,"limit":MEMORY_LIMIT,"pass":allocated<MEMORY_LIMIT and reserved<MEMORY_LIMIT}
        (args.output_dir/"production-smoke.json").write_text(json.dumps(output,indent=2,sort_keys=True)+"\n"); print(json.dumps(output,sort_keys=True))
        if not output["pass"]: raise RuntimeError("v8 production smoke exceeded 22 GiB"); return
    if args.mode in ("audit100","audit250"):
        raise RuntimeError("v8 formal audit is only legal after the corresponding trained checkpoint")
    target=args.target_step
    if target not in ((25,50,100) if args.mode=="phase-m" else (150,200,250)): raise RuntimeError("v8 target step is invalid for phase")
    if args.mode=="phase-t" and not bool(gates.get("step100",{}).get("pass")): raise RuntimeError("Phase T requires passing step100")
    log=args.output_dir/"training.jsonl"
    for step in range(start+1,target+1):
        set_v8_learning_rates(optimizer,step); started=time.monotonic(); row=_train_step(args,model,legacy,dataset,index_map,replay[step-1],device,optimizer,float(calibration["lambda_cf"])); row.update(step=step,step_time=time.monotonic()-started)
        with log.open("a",encoding="utf-8") as handle: handle.write(json.dumps(row,sort_keys=True)+"\n")
        if step in (10,25,50,100,150,200,250):
            payload=build_v8_checkpoint(step=step,model=model,optimizer=optimizer,lineage=lineage,calibration=calibration,gates=gates); _atomic_torch_save(payload,args.output_dir/f"step-{step:06d}.pt")
        print(json.dumps(row,sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
