#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


FORMAL_ROOT = Path("/data/di/worldarena2_track1_20260815")


def _formal(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if resolved != FORMAL_ROOT and FORMAL_ROOT not in resolved.parents:
        raise ValueError(f"v11 audit path escapes formal root: {path}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(value: object, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with partial.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _soft_argmax(logits):
    import torch

    if logits.ndim != 5:
        raise ValueError("v11 EEF logits must have shape (B,2,T,H,W)")
    height, width = logits.shape[-2:]
    probabilities = torch.softmax(logits.float().flatten(-2), dim=-1).reshape_as(
        logits.float()
    )
    x = torch.linspace(0, 1, width, device=logits.device)
    y = torch.linspace(0, 1, height, device=logits.device)
    return torch.stack(
        (
            (probabilities * x.reshape(1, 1, 1, 1, width)).sum((-2, -1)),
            (probabilities * y.reshape(1, 1, 1, height, 1)).sum((-2, -1)),
        ),
        dim=-1,
    )


def _audit_record(sample: str, index: int) -> dict[str, object]:
    def seed(label: str) -> int:
        value = f"v11-health-audit:{label}:{index}:{sample}".encode()
        return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")

    return {
        "sample": sample,
        "optimizer_step": index + 1,
        "noise_seed": seed("noise"),
        "timestep_seed": seed("time"),
        "negative_family": "wrong-left",
        "sample_role": "audit",
    }


def _mean(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError("v11 health audit has no finite values")
    return sum(values) / len(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for value in vars(args).values():
        if isinstance(value, Path):
            _formal(value)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "6":
        raise RuntimeError("v11 health audit requires physical GPU6 only")

    import torch
    from scripts import train_wan_v11_bimanual as trainer
    from worldarena_baseline.wan_gripper_probe import build_gripper_targets
    from worldarena_baseline.wan_v10_objective import phase_ranking_loss
    from worldarena_baseline.wan_v11_training import validate_v11_checkpoint

    # Reuse the production input validator without pretending that this
    # independent observed audit is the launcher's placeholder audit mode.
    args.mode = "preflight"
    args.resume = None
    args.stop_step = None
    args.audit_step = None
    optimizer_rows, audit_rows, replay, source_receipt = trainer._validate_inputs(args)
    v10, legacy = trainer._load_runtime()
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dataset = v10._Dataset([*optimizer_rows, *audit_rows])
    index_map = v10._sample_map(dataset)
    model, names, _count = trainer._load_model(args, v10, legacy, device)
    calibration = json.loads((args.output_dir / "calibration.json").read_text())
    lineage = trainer._lineage(args, source_receipt, calibration)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    step = int(payload["completed_step"])
    validate_v11_checkpoint(
        payload,
        model=model,
        trainable_names=names,
        expected_step=step,
        expected_lineage=lineage,
    )
    trainer._load_trainable(model, payload["trainable_state"])
    model.eval()

    plus_margins: list[float] = []
    minus_margins: list[float] = []
    fm_values: list[float] = []
    left_slot_rms: list[float] = []
    right_slot_rms: list[float] = []
    direct_write_rms: list[float] = []
    duplicate_reference = None
    no_state_leak = True
    episode_rows: list[dict[str, object]] = []

    for index, item in enumerate(audit_rows):
        sample = str(item["sample"])
        record = _audit_record(sample, index)
        batch = v10._batch(dataset, index_map[sample])
        support = batch["condition_support"].to(device)
        condition = trainer._condition(args, sample, "correct", support, device)
        result = trainer._forward(
            args,
            model,
            v10,
            legacy,
            dataset,
            index_map,
            record,
            condition,
            device,
            grad=False,
        )
        logits = torch.stack(result["output"].eef_logits[24], dim=1)
        predicted = _soft_argmax(logits)
        labels = build_gripper_targets(
            batch["action_raster"].to(device).float(),
            observability=v10._observability(args, sample, device),
        )
        motion = torch.stack(
            (condition.left.motion_active, condition.right.motion_active), dim=1
        )
        motion = torch.cat(
            (
                torch.zeros(
                    motion.shape[0], 2, 1, dtype=torch.bool, device=motion.device
                ),
                motion,
            ),
            dim=2,
        )
        phase = phase_ranking_loss(
            predicted,
            labels["position"],
            valid=labels["position_valid"],
            motion_discriminative=motion,
        )
        plus_count = int(phase["plus_valid_count"].item())
        minus_count = int(phase["minus_valid_count"].item())
        plus = float(phase["plus_margin"].cpu()) if plus_count else None
        minus = float(phase["minus_margin"].cpu()) if minus_count else None
        if plus is not None:
            plus_margins.append(plus)
        if minus is not None:
            minus_margins.append(minus)
        fm_values.append(float(result["fm"].cpu()))
        left_slot_rms.append(
            float(result["output"].final_slots.left.float().square().mean().sqrt().cpu())
        )
        right_slot_rms.append(
            float(result["output"].final_slots.right.float().square().mean().sqrt().cpu())
        )
        direct_write_rms.append(
            max(
                float(arm["direct_write_rms"].cpu())
                for stage in result["output"].telemetry.values()
                for arm in stage.values()
            )
        )
        if index == 0:
            duplicate_reference = result["prediction"].detach().cpu()
            repeated = trainer._forward(
                args,
                model,
                v10,
                legacy,
                dataset,
                index_map,
                record,
                condition,
                device,
                grad=False,
            )["prediction"].detach().cpu()
            no_state_leak = torch.equal(duplicate_reference, repeated)
        episode_rows.append(
            {
                "sample": sample,
                "fm": fm_values[-1],
                "phase_plus_margin": plus,
                "phase_minus_margin": minus,
                "left_slot_rms": left_slot_rms[-1],
                "right_slot_rms": right_slot_rms[-1],
                "max_direct_write_rms": direct_write_rms[-1],
            }
        )
        print(json.dumps({"event": "v11_health_audit", "completed": index + 1, "sample": sample}), flush=True)

    smoke = json.loads((args.output_dir / "production-smoke.json").read_text())
    runtime = payload.get("runtime", {})
    families = runtime.get("gradient_families", {})
    required_families = ("tokenizer", "updater", "read", "write", "gate", "eef")
    metrics = {
        "gradient_families": {name: float(families.get(name, 0)) > 0 for name in required_families},
        "frozen_gradients_absent": all(
            parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad
        ),
        "fm_finite": all(math.isfinite(value) for value in fm_values),
        "fm_mean": _mean(fm_values),
        "max_allocated_gib": float(smoke["peak_allocated"]) / 1024**3,
        "max_reserved_gib": float(smoke["peak_reserved"]) / 1024**3,
        "left_slot_rms": _mean(left_slot_rms),
        "right_slot_rms": _mean(right_slot_rms),
        "controller_to_parent_rms": max(direct_write_rms),
        "phase_plus_wins": sum(value > 0 for value in plus_margins),
        "phase_plus_eligible": len(plus_margins),
        "phase_plus_mean_margin": _mean(plus_margins),
        "phase_minus_wins": sum(value > 0 for value in minus_margins),
        "phase_minus_eligible": len(minus_margins),
        "phase_minus_mean_margin": _mean(minus_margins),
        "no_state_leak": no_state_leak,
    }
    failures = []
    if not all(metrics["gradient_families"].values()):
        failures.append("gradient_families")
    for name in ("frozen_gradients_absent", "fm_finite", "no_state_leak"):
        if metrics[name] is not True:
            failures.append(name)
    if metrics["max_allocated_gib"] >= 22 or metrics["max_reserved_gib"] >= 22:
        failures.append("memory")
    if metrics["left_slot_rms"] <= 0 or metrics["right_slot_rms"] <= 0:
        failures.append("slot_collapse")
    if metrics["controller_to_parent_rms"] > 1:
        failures.append("controller_residual")
    if metrics["phase_plus_eligible"] != 20 or metrics["phase_plus_wins"] < 16:
        failures.append("phase_plus")
    if metrics["phase_minus_eligible"] != 20 or metrics["phase_minus_wins"] < 16:
        failures.append("phase_minus")
    report = {
        "contract": "wan-v11-observed-health-audit/1",
        "step": step,
        "decision": "continue" if not failures else "stop",
        "failures": failures,
        "metrics": metrics,
        "episodes": episode_rows,
        "provenance": {
            "checkpoint_sha256": _sha256(args.checkpoint),
            "training_source_closure_sha256": lineage["source_closure_sha256"],
            "audit_script_sha256": _sha256(Path(__file__)),
        },
    }
    _atomic_json(report, args.output)
    print(json.dumps(report, sort_keys=True))
    if failures:
        raise RuntimeError(f"v11 observed health audit failed: {failures}")


if __name__ == "__main__":
    main()
