#!/usr/bin/env python3
"""Build the v10 full-action inventory and leakage-free optimizer/audit split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from worldarena_baseline.robotwin_action_cache import validate_v3_condition_payload
from worldarena_baseline.robotwin_manifest import discover_training_episodes
from worldarena_baseline.wan_action_loss import TemporalRole, temporal_arm_role_masks
from worldarena_baseline.wan_balanced_sampling import classify_role_fractions
from worldarena_baseline.wan_gripper_probe import build_gripper_targets
from worldarena_baseline.wan_v10_data import build_v10_data_receipt, build_v10_split


FORMAL_ROOT = Path("/data/di/worldarena2_track1_20260815")


def _formal(path: Path) -> Path:
    result = path.resolve(strict=False)
    if result != FORMAL_ROOT and FORMAL_ROOT not in result.parents:
        raise ValueError(f"v10 path escapes formal root: {path}")
    return result


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with partial.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _write_jsonl(path: Path, rows: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> None:
    _write_bytes_atomic(
        path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode(),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_bytes_atomic(
        path,
        (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(),
    )


def _asset(sample: str, roots: list[Path], directory: str, suffix: str) -> Path | None:
    candidates = [root / directory / f"{sample}{suffix}" for root in roots]
    existing = [path.resolve() for path in candidates if path.is_file() and not path.is_symlink()]
    if not existing:
        return None
    digests = {_sha(path) for path in existing}
    if len(digests) != 1:
        raise ValueError(f"v10 duplicate cache assets differ for {sample}/{directory}")
    return sorted(existing)[0]


def _raster(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        statistics = str(np.asarray(archive["pose_statistics_sha256"]).item())
        validate_v3_condition_payload(archive, statistics)
        return np.asarray(archive["raster"], dtype=np.float32)


def _metadata(raster: np.ndarray, observability: Path | None) -> dict[str, Any]:
    tensor = torch.from_numpy(raster).permute(1, 0, 2, 3)[None]
    schedule = temporal_arm_role_masks(tensor)["role"][0]
    names = {
        int(TemporalRole.LEFT_ONLY): "left_only",
        int(TemporalRole.RIGHT_ONLY): "right_only",
        int(TemporalRole.BOTH): "both",
        int(TemporalRole.QUIET): "quiet",
        int(TemporalRole.AMBIGUOUS): "ambiguous",
    }
    fractions = {
        name: float((schedule == value).float().mean())
        for value, name in names.items()
    }
    stratum = classify_role_fractions(fractions)
    position_count = velocity_count = 0
    tags: set[str] = set()
    if stratum == "single_dominant":
        tags.add("single_arm")
    if stratum == "bimanual_heavy":
        tags.add("bimanual")
    if bool((schedule == int(TemporalRole.LEFT_ONLY)).any()) and bool(
        (schedule == int(TemporalRole.RIGHT_ONLY)).any()
    ):
        tags.add("sequential")
    if observability is not None:
        visible = np.load(observability, allow_pickle=False)
        if visible.shape != (2, 81) or visible.dtype != np.dtype(bool):
            raise ValueError(f"invalid RGB observability sidecar: {observability}")
        target = build_gripper_targets(
            tensor, observability=torch.from_numpy(visible)[None]
        )
        valid = target["position_valid"][0].permute(1, 0)
        position = target["position"][0].permute(1, 0, 2)
        position_count = int(valid.sum())
        velocity_count = int(target["velocity_valid"].sum())
        both = valid[:, 0] & valid[:, 1]
        if int(both.sum()) >= 2:
            scale = position.new_tensor((79.0, 59.0))
            delta = (position[:, 0] - position[:, 1]) * scale
            x = delta[both, 0]
            if bool(((x[:-1] * x[1:]) < 0).any()) or bool(
                (delta[both].square().sum(dim=-1).sqrt() <= 12).any()
            ):
                tags.add("crossing_or_overlap")
    return {
        "stratum": stratum,
        "role_fractions": fractions,
        "probe_observable": position_count >= 8 and velocity_count >= 4,
        "position_valid_count": position_count,
        "velocity_valid_count": velocity_count,
        "tags": sorted(tags),
    }


def inventory(args: argparse.Namespace) -> dict[str, Any]:
    rows = [episode.__dict__ for episode in discover_training_episodes(args.dataset_root)]
    rows = [
        {**row, "seen_prompts": list(row["seen_prompts"]), "unseen_prompts": list(row["unseen_prompts"])}
        for row in rows
    ]
    if len(rows) != 2100:
        raise ValueError(f"v10 active full-action source count differs from 2100: {len(rows)}")
    _write_jsonl(args.output_root / "source-full-action-2100.jsonl", rows)
    return {"phase": "inventory", "source_count": len(rows)}


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.output_root / "source-full-action-2100.jsonl"
    source = _read_jsonl(source_path)
    dev = _read_jsonl(args.dev_manifest)
    roots = [path.resolve() for path in args.cache_root]
    enriched: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(source, start=1):
        sample = str(row["sample"])
        action = _asset(sample, roots, "action_rasters_v3", ".npz")
        latent = _asset(sample, roots, "latents", ".pt")
        context = _asset(sample, roots, "contexts", ".pt")
        if None in (action, latent, context):
            raise FileNotFoundError(f"v10 base cache is incomplete for {sample}")
        observability = args.observability_root / f"{sample}.npy"
        observability = observability.resolve() if observability.is_file() else None
        item = {
            **row,
            "action_raster_path": str(action),
            "latent_path": str(latent),
            "context_path": str(context),
            "observability_path": str(observability) if observability else None,
        }
        with np.load(action, allow_pickle=False) as archive:
            item["pose_statistics_sha256"] = str(
                np.asarray(archive["pose_statistics_sha256"]).item()
            )
        enriched.append(item)
        metadata[sample] = _metadata(_raster(action), observability)
        if index % 100 == 0:
            print(json.dumps({"event": "v10_metadata", "completed": index, "total": len(source)}), flush=True)
    split = build_v10_split(enriched, dev, metadata, seed=20260819)
    optimizer_path = args.output_root / "optimizer-full-action-2060.jsonl"
    audit_path = args.output_root / "audit20.jsonl"
    metadata_path = args.output_root / "source-metadata.json"
    receipt_path = args.output_root / "data-receipt.json"
    _write_jsonl(optimizer_path, split.optimizer)
    _write_jsonl(audit_path, split.audit)
    _write_json(metadata_path, metadata)
    receipt = build_v10_data_receipt(
        source_manifest=source_path,
        optimizer_manifest=optimizer_path,
        audit_manifest=audit_path,
        dev_manifest=args.dev_manifest,
        split=split,
        official_test_status="unavailable",
    )
    receipt["metadata_sha256"] = _sha(metadata_path)
    receipt["cache_roots"] = [str(path) for path in roots]
    _write_json(receipt_path, receipt)
    return {"phase": "finalize", **receipt}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("inventory", "finalize"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, action="append", default=[])
    parser.add_argument("--observability-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for path in (
        args.dataset_root, args.dev_manifest, args.observability_root,
        args.output_root, *args.cache_root,
    ):
        _formal(path)
    if args.phase == "finalize" and not args.cache_root:
        parser.error("--phase finalize requires at least one --cache-root")
    return args


def main() -> None:
    args = parse_args()
    payload = inventory(args) if args.phase == "inventory" else finalize(args)
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
